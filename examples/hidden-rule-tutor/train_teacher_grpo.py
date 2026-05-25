from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve()))

from areal import PPOTrainer, workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.hf_utils import load_hf_tokenizer

from game_tutor_teacher_training import _reasoning_score, _response_to_tensor
from hidden_rule_game.env import _rough_rule_match
from hidden_rule_game.rules import iter_rule_cycle, sample_examples

from evaluate_teacher import OpenAIStudent, _clip_text, _format_transcript, _strip_thinking


FORBIDDEN = [
    "final_rule",
    "the rule is",
    "a string is valid if",
    "you are correct",
    "you're correct",
    "correctly identified",
    "you identified",
    "your rule is",
]


@dataclass
class HiddenRuleTeacherGRPOConfig(GRPOConfig):
    student_base_url: str = "http://127.0.0.1:30001/v1"
    student_api_key: str = "EMPTY"
    student_api_model: str = "Qwen/Qwen3-0.6B"
    student_temperature: float = 0.2
    student_top_p: float = 1.0
    student_max_tokens: int = 1024
    student_timeout: int = 120
    rounds: int = 8
    examples_per_episode: int = 48
    success_threshold: float = 0.95
    turn_discount: float = 1.0
    consistency_coef: float = 0.15
    reasoning_coef: float = 0.10
    leakage_coef: float = 0.50
    task_reward_coef: float = 1.0
    teacher_policy: str = ""


def _labeled_examples_text(batch: list[dict[str, object]]) -> str:
    return "\n".join(f"{item['x']} -> {'valid' if item['y'] else 'invalid'}" for item in batch)


def _teacher_prompt(
    student_message: str,
    transcript: list[dict[str, str]],
    batch: list[dict[str, object]],
    policy: str,
) -> str:
    return (
        "You are a hidden-rule tutor. You may show the fresh labeled examples exactly, then scaffold "
        "the student's reasoning. Do not state the final rule, do not say whether the student's final "
        "hypothesis is correct, and do not give them a polished hypothesis to copy.\n\n"
        f"Dialogue so far:\n{_clip_text(_format_transcript(transcript)) or 'None'}\n\n"
        f"Student message:\n{_clip_text(_strip_thinking(student_message), 2000)}\n\n"
        f"Fresh labeled examples:\n{_labeled_examples_text(batch)}\n\n"
        f"{policy.strip()}\n\n"
        "Tutor reply. Include the fresh labeled examples, one useful contrast, one uncertainty or "
        "alternative hypothesis class, and one targeted next test request."
    ).strip()


def _student_message(student: OpenAIStudent, observation: str, transcript: list[dict[str, str]]) -> str:
    return student.act(observation, transcript)


def _consistency_score(reply: str, student_next_message: str) -> float:
    reply_tokens = {t for t in re.findall(r"[a-z0-9]+", reply.lower()) if len(t) > 2}
    student_tokens = {t for t in re.findall(r"[a-z0-9]+", student_next_message.lower()) if len(t) > 2}
    if not student_tokens:
        return 0.0
    teaching_terms = {"hypothesis", "alternative", "example", "valid", "invalid", "test", "length", "vowel", "letter"}
    return min(1.0, len((reply_tokens | teaching_terms) & student_tokens) / max(1, len(student_tokens)))


def _leakage_penalty(reply: str, batch: list[dict[str, object]]) -> float:
    lowered = reply.lower()
    if any(token in lowered for token in FORBIDDEN):
        return 1.0
    if not any(str(item["x"]) in reply for item in batch):
        return 0.5
    return 0.0


class HiddenRuleTeacherWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any,
        tokenizer: Any | str,
        student_base_url: str = "http://127.0.0.1:30001/v1",
        student_api_key: str = "EMPTY",
        student_api_model: str = "Qwen/Qwen3-0.6B",
        student_temperature: float = 0.2,
        student_top_p: float = 1.0,
        student_max_tokens: int = 1024,
        student_timeout: int = 120,
        rounds: int = 8,
        examples_per_episode: int = 48,
        success_threshold: float = 0.95,
        turn_discount: float = 1.0,
        consistency_coef: float = 0.15,
        reasoning_coef: float = 0.10,
        leakage_coef: float = 0.50,
        task_reward_coef: float = 1.0,
        teacher_policy: str = "",
    ):
        self.tokenizer = load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.student_cfg = {
            "base_url": student_base_url,
            "api_key": student_api_key,
            "model": student_api_model,
            "temperature": student_temperature,
            "top_p": student_top_p,
            "max_tokens": student_max_tokens,
            "timeout": student_timeout,
        }
        self.rounds = int(rounds)
        self.examples_per_episode = int(examples_per_episode)
        self.success_threshold = float(success_threshold)
        self.turn_discount = float(turn_discount)
        self.consistency_coef = float(consistency_coef)
        self.reasoning_coef = float(reasoning_coef)
        self.leakage_coef = float(leakage_coef)
        self.task_reward_coef = float(task_reward_coef)
        self.teacher_policy = teacher_policy

    async def arun_episode(self, engine: InferenceEngine, data: dict[str, Any]):
        seed = int(data.get("seed", data.get("id", 0)) or 0)
        import random

        rng = random.Random(seed)
        rule = next(iter_rule_cycle(rng))
        examples = sample_examples(rule, rng, self.examples_per_episode)
        heldout = examples[self.rounds * 4 :]
        transcript: list[dict[str, str]] = []
        student = OpenAIStudent(self.student_cfg)
        observation = "A new hidden string rule has been selected. Ask the tutor for evidence."
        responses = []
        components = []

        for turn_idx in range(self.rounds):
            student_msg = await asyncio_to_thread(_student_message, student, observation, transcript)
            batch = examples[turn_idx * 4 : (turn_idx + 1) * 4]
            prompt = _teacher_prompt(student_msg, transcript, batch, self.teacher_policy)
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a concise safe tutor."}, {"role": "user", "content": prompt}],
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
            reply = self.tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
            transcript.append({"student": student_msg, "tutor": reply})
            student.observe(student_msg, reply)
            observation = reply
            consistency = _consistency_score(reply, student_msg)
            reasoning = _reasoning_score(reply)
            leakage = _leakage_penalty(reply, batch)
            shaped = (
                self.consistency_coef * consistency
                + self.reasoning_coef * reasoning
                - self.leakage_coef * leakage
            )
            responses.append((resp, shaped, turn_idx))
            components.append((consistency, reasoning, leakage))
            if "final_rule:" in student_msg.lower():
                break

        guessed_rule = await asyncio_to_thread(student.guess_rule, transcript)
        predictions = [
            bool(student.predict_label(str(item["x"]), transcript, guessed_rule)) for item in heldout
        ]
        labels = [bool(item["y"]) for item in heldout]
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / max(1, len(labels))
        matched = _rough_rule_match(guessed_rule, rule.description)
        task_reward = accuracy + (0.25 if matched else 0.0)
        task_reward = max(-1.0, min(1.0, task_reward))

        tensors = []
        horizon = max(1, len(responses) - 1)
        for resp, shaped, turn_idx in responses:
            reward = shaped + self.task_reward_coef * task_reward * (self.turn_discount ** (horizon - turn_idx))
            tensors.append(_response_to_tensor(resp, reward))

        if components:
            avg = lambda idx: sum(item[idx] for item in components) / len(components)
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                hidden_rule_accuracy=accuracy,
                hidden_rule_task_reward=task_reward,
                hidden_rule_matched=float(matched),
                teacher_consistency=avg(0),
                teacher_reasoning=avg(1),
                teacher_leakage=avg(2),
            )

        return concat_padded_tensors(tensors, pad_value=0.0)


async def asyncio_to_thread(fn, *args):
    import asyncio

    return await asyncio.to_thread(fn, *args)


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, HiddenRuleTeacherGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    if str(config.train_dataset.path).endswith((".json", ".jsonl")):
        from datasets import load_dataset

        train_dataset = load_dataset("json", data_files=config.train_dataset.path, split="train")
    else:
        from areal.dataset import get_custom_dataset

        train_dataset = get_custom_dataset(
            split="train",
            dataset_config=config.train_dataset,
            tokenizer=tokenizer,
        )
    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        student_base_url=config.student_base_url,
        student_api_key=config.student_api_key,
        student_api_model=config.student_api_model,
        student_temperature=config.student_temperature,
        student_top_p=config.student_top_p,
        student_max_tokens=config.student_max_tokens,
        student_timeout=config.student_timeout,
        rounds=config.rounds,
        examples_per_episode=config.examples_per_episode,
        success_threshold=config.success_threshold,
        turn_discount=config.turn_discount,
        consistency_coef=config.consistency_coef,
        reasoning_coef=config.reasoning_coef,
        leakage_coef=config.leakage_coef,
        task_reward_coef=config.task_reward_coef,
        teacher_policy=config.teacher_policy,
    )
    with PPOTrainer(config, train_dataset=train_dataset) as trainer:
        trainer.train(
            workflow=HiddenRuleTeacherWorkflow,
            workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
