from __future__ import annotations

import asyncio
import importlib.util
import math
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from areal import PPOTrainer, workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.hf_utils import load_hf_tokenizer

from examples.game_tutor_common import LLMConfig, OpenAIChatClient, normalize_messages, strip_answer


logger = logging.getLogger("GameTutorTeacherTraining")


@dataclass
class TeacherTrainingConfig(GRPOConfig):
    game: str = "werewolf"
    student_base_url: str = "http://127.0.0.1:30001/v1"
    student_api_key: str = "EMPTY"
    student_api_model: str = "Qwen/Qwen3-0.6B"
    student_temperature: float = 0.2
    student_top_p: float = 1.0
    student_max_tokens: int = 1024
    student_timeout: int = 120
    max_episode_turns: int = 40
    turn_discount: float = 1.0
    use_privileged_teacher_observation: bool = True
    env_kwargs: dict | None = None
    teacher_policy: str = ""
    consistency_coef: float = 0.15
    reasoning_coef: float = 0.10
    leakage_coef: float = 0.35
    task_reward_coef: float = 1.0
    invalid_action_penalty: float = 0.1
    action_quality_coef: float = 0.0
    tutor_quality_coef: float = 0.0
    direct_action_penalty: float = 0.0
    report_reward_components: bool = True


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _game_dir(game: str) -> Path:
    return _repo_root() / "examples" / f"{game}-tutor"


def _make_env(game: str, env_kwargs: dict[str, Any]):
    if game == "werewolf":
        module = _load_module("werewolf_teacher_training_env", _game_dir(game) / "werewolf_env.py")
        return module.WerewolfEnv(**env_kwargs)
    if game == "hanabi":
        module = _load_module("hanabi_teacher_training_env", _game_dir(game) / "hanabi_env.py")
        return module.HanabiEnv(**env_kwargs)
    raise ValueError(f"Unsupported environment game: {game}")


def _hanabi_metrics_module():
    return _load_module("hanabi_tutor_training_metrics", _game_dir("hanabi") / "hanabi_tutor_metrics.py")


def _werewolf_metrics_module():
    return _load_module(
        "werewolf_tutor_training_metrics",
        _game_dir("werewolf") / "werewolf_tutor_metrics.py",
    )


def _teacher_prompt(game: str, obs: str, guide: str, teacher_obs: str, policy: str) -> str:
    game_name = "Werewolf" if game == "werewolf" else "Hanabi"
    return (
        f"You are training as a {game_name} tutor. Give advice that helps the acting student choose, "
        "but do not output the action for them.\n"
        "Reward is higher when your advice is consistent with the student's eventual action, explicit "
        "about reasoning, avoids hidden-information leakage, and improves final game outcome.\n\n"
        f"Private teacher observation:\n{teacher_obs}\n\n"
        f"Student public observation:\n{obs}\n\n"
        f"Action guide:\n{guide}\n\n"
        f"{policy.strip()}\n\n"
        "Reply with concise tutoring thoughts. Prefer: public evidence, risks, alternatives, and one "
        "decision criterion. Do not reveal private card identities, unrevealed roles, or exact targets."
    ).strip()


def _student_prompt(game: str, obs: str, guide: str, advice: str) -> str:
    if game == "werewolf":
        instruction = (
            "Choose exactly one valid action for the current phase. Use <answer>...</answer>. "
            "Do not choose skip when a concrete vote, target, or discussion statement is useful."
        )
    else:
        instruction = "Choose exactly one legal Hanabi action. Use <answer>...</answer>."
    return (
        f"{obs}\n\nTeacher guidance:\n{advice or 'No teacher guidance.'}\n\n{guide}\n\n{instruction}"
    ).strip()


def _score_env(game: str, env: Any) -> float:
    stats = env.get_stats()
    if game == "hanabi":
        return float(stats.get("score", 0.0))
    return float(
        20.0 * stats.get("vill_wins", 0)
        - 20.0 * stats.get("were_wins", 0)
        + stats.get("villager_correct_votes", 0)
        - stats.get("villager_wrong_votes", 0)
        + stats.get("witch_correct_heals", 0)
        + stats.get("witch_correct_poisons", 0)
        + stats.get("hunter_correct_shots", 0)
    )


def _normalize_reward(game: str, score: float) -> float:
    if game == "hanabi":
        return max(-1.0, min(1.0, score / 25.0))
    return max(-1.0, min(1.0, score / 20.0))


def _hanabi_final_tutoring_reward(env: Any, components: list[tuple[float, ...]]) -> float:
    module = _hanabi_metrics_module()
    stats = env.get_stats()
    if components:
        avg_action_quality = sum(item[4] for item in components) / len(components)
        avg_tutor_quality = sum(item[5] for item in components) / len(components)
        avg_direct_action = sum(item[6] for item in components) / len(components)
        avg_leakage = sum(item[2] for item in components) / len(components)
        invalid_actions = sum(1 for item in components if item[3] > 0)
    else:
        avg_action_quality = 0.0
        avg_tutor_quality = 0.0
        avg_direct_action = 0.0
        avg_leakage = 0.0
        invalid_actions = 0
    score = module.hanabi_tutoring_score_from_stats(
        score=float(stats.get("score", 0.0)),
        target_score=float(getattr(env, "target_score", 25)),
        fuse_tokens=int(stats.get("fuse_tokens", 0)),
        max_fuse_tokens=int(getattr(env, "max_fuse_tokens", 3)),
        invalid_actions=invalid_actions,
        turns=max(1, int(stats.get("turns", len(components)))),
        avg_action_quality=avg_action_quality,
        avg_tutor_quality=avg_tutor_quality,
        avg_leakage=avg_leakage,
        avg_direct_action=avg_direct_action,
    )
    return max(-1.0, min(1.0, 2.0 * (score - 0.35)))


def _werewolf_final_tutoring_reward(env: Any, components: list[tuple[float, ...]]) -> float:
    module = _werewolf_metrics_module()
    if components:
        avg_action_quality = sum(item[4] for item in components) / len(components)
        avg_tutor_quality = sum(item[5] for item in components) / len(components)
        avg_direct_action = sum(item[6] for item in components) / len(components)
        avg_leakage = sum(item[2] for item in components) / len(components)
        invalid_actions = sum(1 for item in components if item[3] > 0)
    else:
        avg_action_quality = 0.0
        avg_tutor_quality = 0.0
        avg_direct_action = 0.0
        avg_leakage = 0.0
        invalid_actions = 0
    score = module.werewolf_tutoring_score_from_stats(
        stats=env.get_stats(),
        turns=max(1, len(components)),
        invalid_actions=invalid_actions,
        avg_action_quality=avg_action_quality,
        avg_tutor_quality=avg_tutor_quality,
        avg_leakage=avg_leakage,
        avg_direct_action=avg_direct_action,
    )
    return max(-1.0, min(1.0, score))


def _reasoning_score(text: str) -> float:
    lowered = text.lower()
    markers = [
        "because",
        "if ",
        "risk",
        "evidence",
        "public",
        "alternative",
        "uncertain",
        "consider",
        "therefore",
    ]
    marker_score = min(1.0, sum(1 for marker in markers if marker in lowered) / 4.0)
    length_score = min(1.0, len(text.split()) / 80.0)
    return 0.5 * marker_score + 0.5 * length_score


def _student_consistency_score(advice: str, student_action: str) -> float:
    action = strip_answer(student_action).lower()
    if not action:
        return 0.0
    advice_lower = advice.lower()
    if action in advice_lower:
        return 1.0
    action_tokens = {tok for tok in re.findall(r"[a-z0-9']+", action) if len(tok) > 1}
    advice_tokens = {tok for tok in re.findall(r"[a-z0-9']+", advice_lower) if len(tok) > 1}
    if not action_tokens:
        return 0.0
    return len(action_tokens & advice_tokens) / len(action_tokens)


def _leakage_penalty(game: str, advice: str) -> float:
    text = advice.lower()
    if game == "werewolf":
        patterns = [
            r"\bplayer\s*\d+\s+is\s+(?:a|an|the)?\s*(?:werewolf|villager|witch|foreseer|hunter)\b",
            r"\byou are\s+player\s*\d+.*\b(?:werewolf|villager|witch|foreseer|hunter)\b",
            r"\bthe\s+(?:werewolf|witch|foreseer|hunter)\s+is\s+player\s*\d+\b",
        ]
    else:
        patterns = [
            r"\byour\s+(?:card|hand).*(?:red|blue|green|yellow|white|r[1-5]|b[1-5]|g[1-5]|y[1-5]|w[1-5])\b",
            r"\bposition\s*\d+\s+is\s+(?:red|blue|green|yellow|white|[rbgyw][1-5])\b",
            r"\bplay\s+(?:red|blue|green|yellow|white|[rbgyw])[1-5]\b",
        ]
    return 1.0 if any(re.search(pattern, text) for pattern in patterns) else 0.0


def _response_to_tensor(resp, reward: float) -> dict[str, torch.Tensor]:
    seq = resp.input_tokens + resp.output_tokens
    logprobs = [0.0] * resp.input_len + resp.output_logprobs
    loss_mask = [0] * resp.input_len + [1] * resp.output_len
    versions = [-1] * resp.input_len + resp.output_versions
    return {
        "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool).unsqueeze(0),
        "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([float(reward)], dtype=torch.float32),
    }


class GameTeacherWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any,
        tokenizer: Any | str,
        game: str = "werewolf",
        student_base_url: str = "http://127.0.0.1:30001/v1",
        student_api_key: str = "EMPTY",
        student_api_model: str = "Qwen/Qwen3-0.6B",
        student_temperature: float = 0.2,
        student_top_p: float = 1.0,
        student_max_tokens: int = 1024,
        student_timeout: int = 120,
        max_episode_turns: int = 40,
        turn_discount: float = 1.0,
        use_privileged_teacher_observation: bool = True,
        env_kwargs: dict | None = None,
        teacher_policy: str = "",
        consistency_coef: float = 0.15,
        reasoning_coef: float = 0.10,
        leakage_coef: float = 0.35,
        task_reward_coef: float = 1.0,
        invalid_action_penalty: float = 0.1,
        action_quality_coef: float = 0.0,
        tutor_quality_coef: float = 0.0,
        direct_action_penalty: float = 0.0,
        report_reward_components: bool = True,
    ):
        self.tokenizer = load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.game = game
        self.student = OpenAIChatClient(
            LLMConfig(
                base_url=student_base_url,
                api_key=student_api_key,
                model=student_api_model,
                temperature=student_temperature,
                top_p=student_top_p,
                max_tokens=student_max_tokens,
                timeout=student_timeout,
            )
        )
        self.max_episode_turns = int(max_episode_turns)
        self.turn_discount = float(turn_discount)
        self.use_privileged_teacher_observation = bool(use_privileged_teacher_observation)
        self.env_kwargs = dict(env_kwargs or {})
        self.teacher_policy = teacher_policy
        self.consistency_coef = float(consistency_coef)
        self.reasoning_coef = float(reasoning_coef)
        self.leakage_coef = float(leakage_coef)
        self.task_reward_coef = float(task_reward_coef)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.action_quality_coef = float(action_quality_coef)
        self.tutor_quality_coef = float(tutor_quality_coef)
        self.direct_action_penalty = float(direct_action_penalty)
        self.report_reward_components = bool(report_reward_components)

    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        seed = int(data.get("seed", data.get("id", 0)) or 0)
        env = _make_env(self.game, self.env_kwargs)
        obs, guide, info = await env.sreset(seed=seed)
        done = False
        responses = []
        components = []

        for turn_idx in range(self.max_episode_turns):
            teacher_obs = info.get("teacher_observation", obs) if isinstance(info, dict) else obs
            if not self.use_privileged_teacher_observation:
                teacher_obs = obs
            prompt = _teacher_prompt(self.game, obs, guide, teacher_obs, self.teacher_policy)
            messages = normalize_messages(
                "You are a careful tutor. Help without leaking private information or directly choosing.",
                prompt,
            )
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)
            advice = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)

            student_prompt = _student_prompt(self.game, obs, guide, advice)
            acting_role = getattr(env, "agent_role", None)
            acting_phase = getattr(env, "phase", None)
            student_action = await self.student.complete(
                normalize_messages(
                    "You are the acting game player. Follow the required answer format exactly.",
                    student_prompt,
                )
            )
            obs, guide, step_reward, done, _, info = await env.step(("", [student_action]))
            parsed_action = strip_answer(student_action)
            consistency = _student_consistency_score(advice, student_action)
            reasoning = _reasoning_score(advice)
            leakage = _leakage_penalty(self.game, advice)
            invalid = 0.0 if parsed_action else 1.0
            if self.game == "hanabi":
                hanabi_metrics = _hanabi_metrics_module()
                action_quality = hanabi_metrics.hanabi_action_quality(
                    info.get("event", "") if isinstance(info, dict) else "",
                    float(step_reward if not isinstance(step_reward, list) else sum(step_reward)),
                    parsed_action,
                )
                tutor_quality = hanabi_metrics.hanabi_tutor_quality(advice)
                direct_action = hanabi_metrics.hanabi_direct_action_penalty(advice)
            elif self.game == "werewolf":
                werewolf_metrics = _werewolf_metrics_module()
                action_quality = werewolf_metrics.werewolf_action_quality(
                    parsed_action=parsed_action,
                    step_reward=step_reward,
                    role=acting_role,
                    phase=acting_phase,
                )
                tutor_quality = werewolf_metrics.werewolf_tutor_quality(advice)
                direct_action = werewolf_metrics.werewolf_direct_action_penalty(advice)
            else:
                action_quality = float(step_reward if not isinstance(step_reward, list) else sum(step_reward))
                tutor_quality = reasoning
                direct_action = 0.0
            shaped = (
                self.consistency_coef * consistency
                + self.reasoning_coef * reasoning
                + self.tutor_quality_coef * tutor_quality
                + self.action_quality_coef * action_quality
                - self.leakage_coef * leakage
                - self.invalid_action_penalty * invalid
                - self.direct_action_penalty * direct_action
            )
            responses.append((resp, shaped, turn_idx))
            components.append(
                (
                    consistency,
                    reasoning,
                    leakage,
                    invalid,
                    action_quality,
                    tutor_quality,
                    direct_action,
                    float(step_reward if not isinstance(step_reward, list) else sum(step_reward)),
                )
            )
            if done:
                break

        if self.game == "hanabi":
            final_task_reward = _hanabi_final_tutoring_reward(env, components)
        elif self.game == "werewolf":
            final_task_reward = _werewolf_final_tutoring_reward(env, components)
        else:
            final_task_reward = _normalize_reward(self.game, _score_env(self.game, env))
        tensors = []
        horizon = max(1, len(responses) - 1)
        for resp, shaped, turn_idx in responses:
            discount = self.turn_discount ** (horizon - turn_idx)
            reward = shaped + self.task_reward_coef * final_task_reward * discount
            tensors.append(_response_to_tensor(resp, reward))

        if not tensors:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "loss_mask": torch.empty((0, 0), dtype=torch.bool),
                "logprobs": torch.empty((0, 0), dtype=torch.float32),
                "versions": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.bool),
                "rewards": torch.empty((0,), dtype=torch.float32),
            }

        if self.report_reward_components and components:
            avg = lambda idx: sum(item[idx] for item in components) / len(components)
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                final_task_reward=final_task_reward,
                teacher_consistency=avg(0),
                teacher_reasoning=avg(1),
                teacher_leakage=avg(2),
                student_invalid_action=avg(3),
                action_quality=avg(4),
                hanabi_action_quality=avg(4),
                werewolf_action_quality=avg(4),
                teacher_quality=avg(5),
                teacher_direct_action=avg(6),
                env_step_reward=avg(7),
            )

        return concat_padded_tensors(tensors, pad_value=0.0)


def _load_training_dataset(config: Any, tokenizer: Any):
    path = str(config.train_dataset.path)
    if path.endswith((".json", ".jsonl")):
        from datasets import load_dataset

        return load_dataset("json", data_files=path, split="train")
    return get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=config.tokenizer_path,
    )


def run_game_teacher_training(args: list[str], game: str) -> None:
    config, _ = load_expr_config(args, TeacherTrainingConfig)
    config.game = game
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = _load_training_dataset(config, tokenizer)
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )
    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        game=game,
        student_base_url=config.student_base_url,
        student_api_key=config.student_api_key,
        student_api_model=config.student_api_model,
        student_temperature=config.student_temperature,
        student_top_p=config.student_top_p,
        student_max_tokens=config.student_max_tokens,
        student_timeout=config.student_timeout,
        max_episode_turns=config.max_episode_turns,
        turn_discount=config.turn_discount,
        use_privileged_teacher_observation=config.use_privileged_teacher_observation,
        env_kwargs=config.env_kwargs,
        teacher_policy=config.teacher_policy,
        consistency_coef=config.consistency_coef,
        reasoning_coef=config.reasoning_coef,
        leakage_coef=config.leakage_coef,
        task_reward_coef=config.task_reward_coef,
        invalid_action_penalty=config.invalid_action_penalty,
        action_quality_coef=config.action_quality_coef,
        tutor_quality_coef=config.tutor_quality_coef,
        direct_action_penalty=config.direct_action_penalty,
        report_reward_components=config.report_reward_components,
    )
    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="examples.game_tutor_teacher_training.GameTeacherWorkflow",
            workflow_kwargs=workflow_kwargs,
        )
