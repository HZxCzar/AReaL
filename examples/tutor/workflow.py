from __future__ import annotations

import asyncio
import json
import logging as py_logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:
    from areal import workflow_context
    from areal.api import ModelResponse, RolloutWorkflow
    from areal.utils import logging, stats_tracker
    from areal.utils.data import concat_padded_tensors
    from areal.utils.hf_utils import load_hf_tokenizer
except Exception:  # pragma: no cover - lightweight local test environments

    class _DummyWorkflowContext:
        @staticmethod
        def stat_scope():
            return "examples"

        @staticmethod
        def get():
            return type("WorkflowContext", (), {"task_id": None, "is_eval": False})()

    class _DummyTracker:
        @staticmethod
        def get(_scope):
            return _DummyTracker()

        def scalar(self, **_metrics):
            return None

    class _DummyLogging:
        @staticmethod
        def getLogger(name: str):
            return py_logging.getLogger(name)

    @dataclass
    class ModelRequest:  # type: ignore[no-redef]
        rid: str = field(default_factory=lambda: str(uuid.uuid4()))
        input_ids: list[int] = field(default_factory=list)
        gconfig: Any | None = None
        metadata: dict[str, Any] = field(default_factory=dict)
        tokenizer: Any | None = None

    @dataclass
    class ModelResponse:  # type: ignore[no-redef]
        input_tokens: list[int] = field(default_factory=list)
        output_tokens: list[int] = field(default_factory=list)
        output_logprobs: list[float] = field(default_factory=list)
        output_versions: list[int] = field(default_factory=list)
        stop_reason: str = "stop"
        tokenizer: Any | None = None

        @property
        def input_len(self) -> int:
            return len(self.input_tokens)

        @property
        def output_len(self) -> int:
            return len(self.output_tokens)

    class RolloutWorkflow:  # type: ignore[no-redef]
        pass

    def concat_padded_tensors(  # type: ignore[no-redef]
        tensor_dicts: list[dict[str, Any]], pad_value: float = 0.0
    ) -> dict[str, Any]:
        if not tensor_dicts:
            return {}
        if len(tensor_dicts) == 1:
            return dict(tensor_dicts[0])
        keys = set(tensor_dicts[0])
        for item in tensor_dicts[1:]:
            if set(item) != keys:
                raise ValueError("tensor dict keys must match")
        result: dict[str, Any] = {}
        for key in tensor_dicts[0]:
            values = [item[key] for item in tensor_dicts]
            if isinstance(values[0], torch.Tensor):
                max_shape = list(values[0].shape)
                for value in values[1:]:
                    max_shape = [
                        max(a, b) if idx > 0 else a
                        for idx, (a, b) in enumerate(zip(max_shape, value.shape))
                    ]
                padded = []
                for value in values:
                    pad = []
                    for current, target in reversed(
                        list(zip(value.shape[1:], max_shape[1:]))
                    ):
                        pad.extend([0, target - current])
                    pv = 0.0 if key == "attention_mask" else pad_value
                    padded.append(torch.nn.functional.pad(value, pad, value=pv))
                result[key] = torch.cat(padded, dim=0)
            else:
                result[key] = values[0]
        return result

    workflow_context = _DummyWorkflowContext()
    stats_tracker = _DummyTracker()
    logging = _DummyLogging()

    def load_hf_tokenizer(path):  # type: ignore[no-redef]
        return path


from examples.common.chat_budget import ChatContextBudget
from examples.common.openai_utils import (
    AsyncLLMCaller,
    AuxModelConfig,
    make_teacher_client,
)
from examples.common.parsing import parse_json_dict
from examples.tutor.core.callers import (
    ApiAuxiliaryCaller,
    AReaLEngineActorCaller,
    AReaLEngineAuxiliaryCaller,
    AReaLEngineChatCaller,
    ExternalActorCaller,
    TextCallResult,
)
from examples.tutor.core.generation_budget import (
    CONTEXT_BUDGET_TERMINATION_REASON,
    ContextBudgetLimitExceeded,
)
from examples.tutor.core.history import (
    trace_to_history_record,
    trace_to_json,
)
from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.parsers import parse_leak_check_result
from examples.tutor.core.rewards import EpisodeRewardComputer, artifact_to_trace
from examples.tutor.core.scoring import AnswerScorer, get_answer_scorer
from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.text import (
    strip_reasoning_for_context as _strip_reasoning_for_context,
)
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PublicHistoryState,
    StudentTurnState,
    TurnArtifact,
    TurnTrace,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import (
    ANSWER_JUDGE_USER_TEMPLATE,
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    LEAK_CHECK_USER_TEMPLATE,
    STUDENT_STATE_USER_TEMPLATE,
    TEACHER_STATE_USER_TEMPLATE,
    render_prompt,
)

logger = logging.getLogger("TutorWorkflow")

_REWARD_COMPONENT_ALIASES = {
    "success_credit": "success",
}


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


def _reward_component_key(name: str) -> str:
    return _REWARD_COMPONENT_ALIASES.get(name, name)


class TutorAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any | None = None,
        tokenizer: str | Any | None = None,
        answer_scorer: str = "aime",
        max_turns: int = 6,
        enable_thinking: bool = False,
        enable_leak_check: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        tool_call_parser: str = "qwen25",
        reasoning_parser: str = "qwen3",
        aux_mode: str = "api",
        aux_enable_thinking: bool = False,
        aux_base_url: str = "http://127.0.0.1:30000/v1",
        aux_model: str = "qwen-aux",
        aux_api_key: str = "EMPTY",
        aux_timeout: int = 120,
        aux_max_tokens: int = 1024,
        aux_temperature: float = 0.7,
        aux_top_p: float | None = None,
        max_concurrent_aux_calls: int = 8,
        aux_request_params: dict[str, Any] | None = None,
        success_reward: float = 1.0,
        leak_penalty: float = -1.0,
        assign_success_reward: bool = False,
        outcome_prior_turn_weight: float = 0.1,
        outcome_credit_gamma: float = 0.9,
        early_success_bonus: float = 0.3,
        enable_turn_penalty: bool = False,
        turn_penalty: float = -0.01,
        length_penalty_threshold_chars: int = 1200,
        length_penalty_per_100_chars: float = -0.005,
        length_penalty_min: float = -0.1,
        teacher_system_prompt: str = "",
        teacher_show_ground_truth: bool = False,
        student_system_prompt: str = "",
        leak_check_system_prompt: str = "",
        answer_judge_enabled: bool = False,
        answer_judge_max_tokens: int = 256,
        answer_judge_system_prompt: str = DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
        summary_system_prompt: str = "",
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 1,
        max_train_sample_tokens: int | None = None,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
        pairwise_reward_enabled: bool = False,
        pairwise_reference_lag_steps: int = 5,
        pairwise_reward_scale: float = 0.05,
        pairwise_compare_all_turns: bool = True,
        pairwise_judge_both_incorrect: bool = True,
    ):
        self.max_turns = max_turns
        self.answer_scorer_name = answer_scorer
        self.answer_scorer: AnswerScorer = get_answer_scorer(answer_scorer)
        self.enable_thinking = enable_thinking
        self.enable_leak_check = bool(enable_leak_check)
        self.gconfig = gconfig
        self.temperature = gconfig.temperature if gconfig is not None else temperature
        self.top_p = gconfig.top_p if gconfig is not None else top_p
        self.max_completion_tokens = (
            gconfig.max_new_tokens if gconfig is not None else max_completion_tokens
        )
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        if aux_mode not in {"api", "self"}:
            raise ValueError(f"aux_mode must be 'api' or 'self', got {aux_mode!r}")
        self.aux_mode = aux_mode
        self.aux_enable_thinking = bool(aux_enable_thinking)
        self.aux_base_url = aux_base_url
        self.aux_model = aux_model
        self.aux_api_key = aux_api_key
        self.aux_timeout = int(aux_timeout)
        self.aux_max_tokens = int(aux_max_tokens)
        self.aux_temperature = float(aux_temperature)
        self.aux_top_p = aux_top_p
        self.aux_request_params = dict(aux_request_params or {})
        self.max_concurrent_aux_calls = int(max_concurrent_aux_calls)
        self._self_aux_semaphore = asyncio.Semaphore(
            max(1, self.max_concurrent_aux_calls)
        )
        self.context_window_margin = int(context_window_margin)
        self.success_reward = float(success_reward)
        self.leak_penalty = float(leak_penalty)
        self.assign_success_reward = bool(assign_success_reward)
        self.outcome_prior_turn_weight = float(outcome_prior_turn_weight)
        self.outcome_credit_gamma = float(outcome_credit_gamma)
        self.early_success_bonus = float(early_success_bonus)
        self.enable_turn_penalty = bool(enable_turn_penalty)
        self.turn_penalty = float(turn_penalty)
        self.length_penalty_threshold_chars = int(length_penalty_threshold_chars)
        self.length_penalty_per_100_chars = float(length_penalty_per_100_chars)
        self.length_penalty_min = float(length_penalty_min)
        self.teacher_system_prompt = teacher_system_prompt.strip()
        self.teacher_show_ground_truth = bool(teacher_show_ground_truth)
        self.student_system_prompt = student_system_prompt.strip()
        self.leak_check_system_prompt = leak_check_system_prompt.strip()
        self.answer_judge_enabled = bool(answer_judge_enabled)
        self.answer_judge_max_tokens = max(1, int(answer_judge_max_tokens))
        self.answer_judge_system_prompt = answer_judge_system_prompt.strip()
        self._answer_judge_cache: dict[tuple[str, str, str], JudgeResult] = {}
        self.summary_system_prompt = summary_system_prompt.strip()
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
        self.max_train_sample_tokens = max_train_sample_tokens
        self.pairwise_reward_enabled = bool(pairwise_reward_enabled)
        self.pairwise_reference_lag_steps = max(0, int(pairwise_reference_lag_steps))
        self.pairwise_reward_scale = float(pairwise_reward_scale)
        self.pairwise_compare_all_turns = bool(pairwise_compare_all_turns)
        self.pairwise_judge_both_incorrect = bool(pairwise_judge_both_incorrect)
        self.last_history: list[dict[str, Any]] = []
        self.last_traces: list[TurnTrace] = []
        self.last_total_reward = 0.0
        self.tokenizer = (
            load_hf_tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.teacher_context_budget = ChatContextBudget(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            safety_margin=context_window_margin,
        )
        aux_config = AuxModelConfig(
            base_url=aux_base_url,
            model=aux_model,
            api_key=aux_api_key,
            timeout=aux_timeout,
            max_tokens=aux_max_tokens,
            temperature=aux_temperature,
            top_p=aux_top_p,
            max_concurrency=max_concurrent_aux_calls,
            request_params=self.aux_request_params,
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self.aux_caller = (
            ApiAuxiliaryCaller(AsyncLLMCaller(aux_config))
            if self.aux_mode == "api"
            else None
        )
        self.tokenizer_path = tokenizer_path
        self.model_context_length = model_context_length

    def get_lora_versions_for_episode(
        self,
        engine: Any,
        data: dict[str, Any],
        current_lora_version: int | None,
    ) -> set[int]:
        del engine, data
        if current_lora_version is None:
            return set()

        actor_version = int(current_lora_version)
        versions = {actor_version}
        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False

        if self.pairwise_reward_enabled and not is_eval:
            versions.add(max(0, actor_version - int(self.pairwise_reference_lag_steps)))
        return versions

    async def arun_episode(self, engine, data: dict[str, Any]):
        return await self._run_episode(data, engine=engine)

    async def run(self, data: dict[str, Any], **extra_kwargs):
        teacher_client = make_teacher_client(extra_kwargs)
        await self._run_episode(data, external_client=teacher_client)
        return self.last_total_reward

    async def _run_episode(
        self,
        data: dict[str, Any],
        engine: Any | None = None,
        direct_client: Any | None = None,
        external_client: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if direct_client is not None:
            external_client = direct_client
        if (engine is None) == (external_client is None):
            raise ValueError("Exactly one tutor generation source must be provided.")

        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        trajectory_id = uuid.uuid4().int & ((1 << 63) - 1)
        turn_artifacts: list[TurnArtifact] = []
        leak_count = 0
        termination_reason = "max_turns"
        episode_lora_version = None
        if engine is not None:
            try:
                context_lora_version = getattr(
                    workflow_context.get(), "lora_version", None
                )
            except Exception:
                context_lora_version = None
            if context_lora_version is not None:
                episode_lora_version = int(context_lora_version)
            elif hasattr(engine, "get_version"):
                episode_lora_version = int(engine.get_version())
        actor_chat_caller = (
            self._make_engine_chat_caller(
                engine,
                enable_thinking=self.enable_thinking,
            )
            if engine is not None
            else None
        )
        aux_chat_caller = (
            self._make_engine_chat_caller(
                engine,
                enable_thinking=self.aux_enable_thinking,
            )
            if engine is not None and self.aux_mode == "self"
            else None
        )
        actor_caller = self._make_actor_caller(
            chat_caller=actor_chat_caller,
            external_client=external_client,
        )
        aux_caller = self._make_auxiliary_caller(chat_caller=aux_chat_caller)
        answer_judge_caller = self._make_answer_judge_caller(
            chat_caller=aux_chat_caller
        )

        initial_student_answer, initial_student_error = await self._run_student(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(),
                previous_student_output="",
                latest_tutor_visible_output="(none, produce the first answer attempt)",
            ),
            aux_caller=aux_caller,
        )
        initial_student_answer = _strip_reasoning_for_context(initial_student_answer)
        initial_judge_result = await self._score_answer_async(
            task,
            ground_truth,
            initial_student_answer,
            answer_judge_caller=answer_judge_caller,
        )
        if initial_judge_result.correct:
            self.last_history = []
            self.last_traces = []
            self.last_total_reward = 0.0
            termination_reason = "pre_solved"
            episode_artifact = EpisodeArtifact(
                task=task,
                ground_truth=ground_truth,
                initial_student_answer=initial_student_answer,
                initial_student_error=initial_student_error,
                initial_judge_result=initial_judge_result,
                turns=[],
                termination_reason=termination_reason,
                pre_success=True,
                leak_count=0,
                latest_student_answer=initial_student_answer,
            )
            self._log_rollout_stats(
                total_reward=0.0,
                traces=[],
                termination_reason=episode_artifact.termination_reason,
                pre_success=episode_artifact.pre_success,
                leak_count=episode_artifact.leak_count,
            )
            self._maybe_dump_debug_trace(
                task=episode_artifact.task,
                ground_truth=episode_artifact.ground_truth,
                initial_student_answer=episode_artifact.initial_student_answer,
                latest_student_answer=episode_artifact.latest_student_answer,
                total_reward=0.0,
                traces=[],
                termination_reason=episode_artifact.termination_reason,
                pre_success=episode_artifact.pre_success,
                leak_count=episode_artifact.leak_count,
            )
            return None

        public_history = PublicHistoryState(
            summary=self._build_initial_public_summary(initial_student_answer),
            turn_count=0,
        )
        previous_tutor_visible_output = ""
        previous_student_output = initial_student_answer
        previous_feedback = TutorPrivateFeedback(
            kind="student_judged",
            student_output=initial_student_answer,
            judge_correct=False,
            judge_feedback=initial_judge_result.feedback,
        )

        for turn_idx in range(1, self.max_turns + 1):
            tutor_state = TutorTurnState(
                task=task,
                ground_truth=ground_truth,
                public_history=public_history,
                previous_tutor_visible_output=previous_tutor_visible_output,
                previous_feedback=previous_feedback,
                turn_idx=turn_idx,
                max_turns=self.max_turns,
            )
            try:
                response, tutor_raw_output = await self._generate_tutor_response(
                    tutor_state,
                    actor_caller=actor_caller,
                    lora_version=episode_lora_version,
                )
            except ContextBudgetLimitExceeded as exc:
                logger.info(
                    "Terminating tutor episode at turn %s due to context budget: %s",
                    turn_idx,
                    exc,
                )
                termination_reason = CONTEXT_BUDGET_TERMINATION_REASON
                break
            tutor_visible_output = _strip_reasoning_for_context(tutor_raw_output)
            public_before = public_history.summary

            leak_result = await self._run_optional_leak_check(
                task,
                ground_truth,
                tutor_visible_output,
                aux_caller=aux_caller,
            )
            if leak_result.leaked:
                leak_count += 1
                turn_artifacts.append(
                    TurnArtifact(
                        turn_idx=turn_idx,
                        tutor_state=tutor_state,
                        tutor_prompt=self._build_tutor_prompt(tutor_state),
                        tutor_response=response,
                        tutor_raw_output=tutor_raw_output,
                        tutor_visible_output=tutor_visible_output,
                        leak_result=leak_result,
                        public_history_before=public_before,
                        public_history_after=public_before,
                    )
                )
                previous_feedback = TutorPrivateFeedback(
                    kind="leak",
                    leak_feedback=leak_result.feedback,
                )
                previous_tutor_visible_output = tutor_visible_output
                termination_reason = (
                    "max_turns" if turn_idx == self.max_turns else "continue"
                )
                continue

            student_state = StudentTurnState(
                task=task,
                public_history=public_history,
                previous_student_output=previous_student_output,
                latest_tutor_visible_output=tutor_visible_output,
            )
            student_prompt = self._build_student_prompt_from_state(student_state)
            student_answer, student_error = await self._run_student(
                student_state,
                aux_caller=aux_caller,
            )
            student_answer = _strip_reasoning_for_context(student_answer)
            judge_result = await self._score_answer_async(
                task,
                ground_truth,
                student_answer,
                answer_judge_caller=answer_judge_caller,
            )

            if judge_result.correct:
                termination_reason = "success"
            else:
                termination_reason = (
                    "max_turns" if turn_idx == self.max_turns else "continue"
                )

            next_public_history = await self._run_public_summary_update(
                old_public_history=public_history,
                previous_student_answer=previous_student_output,
                tutor_visible_output=tutor_visible_output,
                current_student_answer=student_answer,
            )
            turn_artifacts.append(
                TurnArtifact(
                    turn_idx=turn_idx,
                    tutor_state=tutor_state,
                    tutor_prompt=self._build_tutor_prompt(tutor_state),
                    tutor_response=response,
                    tutor_raw_output=tutor_raw_output,
                    tutor_visible_output=tutor_visible_output,
                    leak_result=leak_result,
                    public_history_before=public_before,
                    public_history_after=next_public_history.summary,
                    student_state=student_state,
                    student_prompt=student_prompt,
                    student_output=student_answer,
                    student_error=student_error,
                    judge_result=judge_result,
                )
            )

            public_history = next_public_history
            previous_tutor_visible_output = tutor_visible_output
            previous_student_output = student_answer
            previous_feedback = TutorPrivateFeedback(
                kind="student_judged",
                student_output=student_answer,
                judge_correct=judge_result.correct,
                judge_feedback=judge_result.feedback,
            )
            if judge_result.correct:
                break

        episode_artifact = EpisodeArtifact(
            task=task,
            ground_truth=ground_truth,
            initial_student_answer=initial_student_answer,
            initial_student_error=initial_student_error,
            initial_judge_result=initial_judge_result,
            turns=turn_artifacts,
            termination_reason=termination_reason,
            pre_success=False,
            leak_count=leak_count,
            latest_student_answer=previous_student_output,
        )
        reward_computer = EpisodeRewardComputer(
            success_reward=self.success_reward,
            leak_penalty=self.leak_penalty,
            assign_success_reward=self.assign_success_reward,
            outcome_prior_turn_weight=self.outcome_prior_turn_weight,
            outcome_credit_gamma=self.outcome_credit_gamma,
            early_success_bonus=self.early_success_bonus,
            enable_turn_penalty=self.enable_turn_penalty,
            turn_penalty=self.turn_penalty,
            length_penalty_threshold_chars=self.length_penalty_threshold_chars,
            length_penalty_per_100_chars=self.length_penalty_per_100_chars,
            length_penalty_min=self.length_penalty_min,
        )
        pairwise_rewards: dict[int, float] = {}
        if self._should_run_pairwise_reward(engine, turn_artifacts):
            pairwise_results = await self._run_pairwise_evaluation(
                episode_artifact,
                episode_lora_version=episode_lora_version,
                chat_caller=actor_chat_caller,
                aux_caller=aux_caller,
                answer_judge_caller=answer_judge_caller,
            )
            pairwise_rewards = {
                result.turn_idx: result.reward
                for result in pairwise_results
                if result.reward
            }
        assignments = await reward_computer.compute(
            episode_artifact, pairwise_rewards=pairwise_rewards
        )
        traces = [
            artifact_to_trace(artifact, assignment)
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        history = [
            trace_to_history_record(
                trace,
                artifact.leak_result if artifact.leak_result.leaked else None,
                artifact.student_error,
            )
            for artifact, trace in zip(turn_artifacts, traces, strict=True)
        ]
        results = [
            response_to_tensordict(
                artifact.tutor_response,
                reward=assignment.reward,
                trajectory_id=trajectory_id,
                turn_idx=artifact.turn_idx,
            )
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        total_reward = float(sum(assignment.reward for assignment in assignments))
        self.last_history = history
        self.last_traces = traces
        self.last_total_reward = total_reward
        self._log_rollout_stats(
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            pre_success=episode_artifact.pre_success,
            leak_count=episode_artifact.leak_count,
        )
        self._maybe_dump_debug_trace(
            task=episode_artifact.task,
            ground_truth=episode_artifact.ground_truth,
            initial_student_answer=episode_artifact.initial_student_answer,
            latest_student_answer=episode_artifact.latest_student_answer,
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            pre_success=episode_artifact.pre_success,
            leak_count=episode_artifact.leak_count,
        )
        if not results:
            return None
        return concat_padded_tensors(results)

    def _make_engine_chat_caller(
        self,
        engine: Any,
        *,
        enable_thinking: bool,
    ) -> AReaLEngineChatCaller:
        return AReaLEngineChatCaller(
            engine=engine,
            tokenizer=self.tokenizer,
            enable_thinking=enable_thinking,
        )

    def _make_actor_caller(
        self,
        engine: Any | None = None,
        external_client: Any | None = None,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> AReaLEngineActorCaller | ExternalActorCaller:
        if chat_caller is None and engine is not None:
            chat_caller = self._make_engine_chat_caller(
                engine,
                enable_thinking=self.enable_thinking,
            )
        if chat_caller is not None:
            return AReaLEngineActorCaller(
                chat_caller=chat_caller,
                gconfig=self._generation_config(),
                max_completion_tokens=self.max_completion_tokens,
                max_train_sample_tokens=self.max_train_sample_tokens,
            )
        if external_client is None:
            raise ValueError("Exactly one tutor generation source must be provided.")
        return ExternalActorCaller(
            client=external_client,
            tokenizer=self.tokenizer,
            context_budget=self.teacher_context_budget,
            temperature=self.temperature,
            top_p=self.top_p,
            enable_thinking=self.enable_thinking,
            max_completion_tokens=self.max_completion_tokens,
            max_train_sample_tokens=self.max_train_sample_tokens,
        )

    def _make_auxiliary_caller(
        self,
        engine: Any | None = None,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller:
        if self.aux_mode == "api":
            if self.aux_caller is None:
                raise RuntimeError("API auxiliary caller is not initialized.")
            return self.aux_caller
        if chat_caller is None and engine is not None:
            chat_caller = self._make_engine_chat_caller(
                engine,
                enable_thinking=self.aux_enable_thinking,
            )
        if chat_caller is None:
            raise RuntimeError(
                "auxiliary_model.mode='self' requires an AReaL inference engine. "
                "Use arun_episode(engine, data) or switch auxiliary_model.mode to 'api'."
            )
        return AReaLEngineAuxiliaryCaller(
            chat_caller=chat_caller,
            base_gconfig=self.gconfig,
            max_completion_tokens=self.aux_max_tokens,
            temperature=self.aux_temperature,
            top_p=self.aux_top_p,
            max_concurrency=self.max_concurrent_aux_calls,
            context_length=self.teacher_context_budget.context_length,
            context_window_margin=self.context_window_margin,
            semaphore=self._self_aux_semaphore,
        )

    def _make_answer_judge_caller(
        self,
        *,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None:
        if not self.answer_judge_enabled:
            return None
        if self.aux_mode == "api":
            config = AuxModelConfig(
                base_url=self.aux_base_url,
                model=self.aux_model,
                api_key=self.aux_api_key,
                timeout=self.aux_timeout,
                max_tokens=self.answer_judge_max_tokens,
                temperature=0.0,
                top_p=self.aux_top_p,
                max_concurrency=self.max_concurrent_aux_calls,
                request_params=self.aux_request_params,
                tokenizer_path=self.tokenizer_path,
                context_length=self.model_context_length,
                context_window_margin=self.context_window_margin,
            )
            return ApiAuxiliaryCaller(AsyncLLMCaller(config))
        if chat_caller is None:
            raise RuntimeError(
                "answer_judge with auxiliary_model.mode='self' requires an "
                "AReaL inference engine."
            )
        return AReaLEngineAuxiliaryCaller(
            chat_caller=chat_caller,
            base_gconfig=self.gconfig,
            max_completion_tokens=self.answer_judge_max_tokens,
            temperature=0.0,
            top_p=self.aux_top_p,
            max_concurrency=self.max_concurrent_aux_calls,
            context_length=self.teacher_context_budget.context_length,
            context_window_margin=self.context_window_margin,
            semaphore=self._self_aux_semaphore,
        )

    async def _generate_tutor_response(
        self,
        tutor_state: TutorTurnState,
        *,
        engine: Any | None = None,
        external_client: Any | None = None,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller | None = None,
        lora_version: int | None = None,
        rid_prefix: str = "tutor",
    ) -> tuple[ModelResponse, str]:
        messages = self._build_tutor_messages(tutor_state)
        if actor_caller is None:
            actor_caller = self._make_actor_caller(engine, external_client)
        result = await actor_caller.generate(
            messages,
            lora_version=lora_version,
            rid_prefix=f"{rid_prefix}-{tutor_state.turn_idx}",
        )
        return result.response, result.raw_text

    async def _run_student(
        self,
        state: StudentTurnState,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> tuple[str, str | None]:
        prompt = self._build_student_prompt_from_state(state)
        result = await self._call_auxiliary_prompt(
            system_prompt=self.student_system_prompt,
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix=f"student-{state.public_history.turn_count}",
        )
        if result.error:
            return "", result.error
        return result.text, None

    async def _run_optional_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> LeakCheckResult:
        if not self.enable_leak_check:
            return LeakCheckResult(
                raw_output="",
                leaked=False,
                feedback="Leak check disabled.",
                parse_error=None,
                raw_result={"disabled": True},
            )
        return await self._run_leak_check(
            task,
            ground_truth,
            teacher_action,
            aux_caller=aux_caller,
        )

    async def _run_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> LeakCheckResult:
        prompt = self._build_leak_check_prompt(
            task,
            ground_truth,
            _strip_reasoning_for_context(teacher_action),
        )
        result = await self._call_auxiliary_prompt(
            system_prompt=self.leak_check_system_prompt,
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix="leak-check",
        )
        if result.error:
            return LeakCheckResult(
                raw_output="",
                leaked=True,
                feedback=f"Leak check failed: {result.error}",
                parse_error=result.error,
                raw_result={},
            )
        return parse_leak_check_result(result.text)

    async def _call_auxiliary_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        caller = aux_caller or self._make_auxiliary_caller(engine=None)
        return await caller.call_text(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            rid_prefix=rid_prefix,
        )

    async def _run_public_summary_update(
        self,
        *,
        old_public_history: PublicHistoryState,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> PublicHistoryState:
        entries = []
        existing_history = old_public_history.summary.strip()
        if existing_history:
            entries.append(existing_history)
        else:
            entries.append(self._build_initial_public_summary(previous_student_answer))

        tutor_round = old_public_history.turn_count + 1
        student_round = old_public_history.turn_count + 2
        entries.append(
            self._format_public_history_entry(
                "Tutor",
                tutor_round,
                tutor_visible_output,
            )
        )
        entries.append(
            self._format_public_history_entry(
                "Student",
                student_round,
                current_student_answer,
            )
        )
        return PublicHistoryState(
            summary="\n\n".join(entry for entry in entries if entry),
            turn_count=old_public_history.turn_count + 1,
        )

    def _build_tutor_prompt(self, state: TutorTurnState) -> str:
        feedback = state.previous_feedback
        return render_prompt(
            TEACHER_STATE_USER_TEMPLATE,
            task=state.task,
            ground_truth=state.ground_truth,
            show_ground_truth=self.teacher_show_ground_truth,
            public_history=state.public_history.summary
            or "No visible tutoring history yet.",
            previous_tutor_output=state.previous_tutor_visible_output or "(none yet)",
            feedback_kind=feedback.kind,
            student_output=feedback.student_output or "(empty)",
            judge_correct=feedback.judge_correct,
            judge_feedback=feedback.judge_feedback or "(empty)",
            leak_feedback=feedback.leak_feedback or "(empty)",
            current_round=state.turn_idx,
            max_turns=state.max_turns,
            remaining_rounds=max(state.max_turns - state.turn_idx + 1, 0),
        )

    def _build_student_prompt_from_state(self, state: StudentTurnState) -> str:
        return render_prompt(
            STUDENT_STATE_USER_TEMPLATE,
            task=state.task,
            public_history=state.public_history.summary
            or "No previous visible tutoring history.",
            previous_student_output=state.previous_student_output or "(empty)",
            teacher_feedback=state.latest_tutor_visible_output or "(none)",
        )

    def _build_leak_check_prompt(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> str:
        return render_prompt(
            LEAK_CHECK_USER_TEMPLATE,
            task=task,
            ground_truth=ground_truth,
            teacher_action=_strip_reasoning_for_context(teacher_action),
        )

    def _build_initial_public_summary(self, initial_student_answer: str) -> str:
        return self._format_public_history_entry("Student", 1, initial_student_answer)

    def _format_public_history_entry(
        self, speaker: str, round_idx: int, text: str
    ) -> str:
        visible_text = _strip_reasoning_for_context(text)
        return f"{speaker} round {round_idx}:\n{visible_text}"

    def _score_answer(
        self, task: str, ground_truth: str, student_answer: str
    ) -> JudgeResult:
        return self.answer_scorer(task, ground_truth, student_answer)

    async def _score_answer_async(
        self,
        task: str,
        ground_truth: str,
        student_answer: str,
        *,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> JudgeResult:
        exact_result = self._score_answer(task, ground_truth, student_answer)
        if exact_result.correct or not self.answer_judge_enabled:
            return exact_result

        extracted_answer = str(exact_result.raw_result.get("extracted_answer") or "")
        cache_key = (str(task), str(ground_truth), extracted_answer)
        cached = self._answer_judge_cache.get(cache_key)
        if cached is not None:
            return cached

        if answer_judge_caller is None:
            result = self._answer_judge_failed_result(
                exact_result,
                error="answer judge caller is unavailable",
            )
            self._answer_judge_cache[cache_key] = result
            return result

        prompt = self._build_answer_judge_prompt(task, ground_truth, extracted_answer)
        judge_call = await self._call_auxiliary_prompt(
            system_prompt=self.answer_judge_system_prompt,
            user_prompt=prompt,
            aux_caller=answer_judge_caller,
            rid_prefix="answer-judge",
        )
        if judge_call.error:
            result = self._answer_judge_failed_result(
                exact_result,
                error=judge_call.error,
            )
            self._answer_judge_cache[cache_key] = result
            return result

        result = self._parse_answer_judge_result(exact_result, judge_call)
        self._answer_judge_cache[cache_key] = result
        return result

    def _answer_judge_failed_result(
        self, exact_result: JudgeResult, *, error: str
    ) -> JudgeResult:
        raw_result = dict(exact_result.raw_result)
        raw_result["exact_match_correct"] = bool(exact_result.correct)
        raw_result["answer_judge"] = {
            "enabled": True,
            "used": False,
            "error": error,
        }
        return JudgeResult(
            raw_output=exact_result.raw_output,
            correct=exact_result.correct,
            feedback=exact_result.feedback,
            parse_error=exact_result.parse_error,
            raw_result=raw_result,
        )

    def _parse_answer_judge_result(
        self, exact_result: JudgeResult, judge_call: TextCallResult
    ) -> JudgeResult:
        parsed, parse_error = parse_json_dict(judge_call.text)
        if not isinstance(parsed, dict):
            return self._answer_judge_failed_result(
                exact_result,
                error=parse_error or "Expected JSON object from answer judge.",
            )

        correct = parsed.get("correct")
        if not isinstance(correct, bool):
            return self._answer_judge_failed_result(
                exact_result,
                error='Answer judge field "correct" must be a boolean.',
            )
        if parse_error:
            return self._answer_judge_failed_result(
                exact_result,
                error=parse_error,
            )

        raw_result = dict(exact_result.raw_result)
        raw_result["exact_match_correct"] = bool(exact_result.correct)
        raw_result["answer_judge"] = {
            "enabled": True,
            "used": True,
            "correct": correct,
            "raw_output": judge_call.raw_text or judge_call.text,
            "raw_result": parsed,
        }
        return JudgeResult(
            raw_output=judge_call.raw_text or judge_call.text,
            correct=correct,
            feedback=exact_result.feedback,
            parse_error=None,
            raw_result=raw_result,
        )

    def _build_answer_judge_prompt(
        self, task: str, ground_truth: str, extracted_answer: str
    ) -> str:
        return render_prompt(
            ANSWER_JUDGE_USER_TEMPLATE,
            task=task,
            ground_truth=ground_truth,
            extracted_answer=extracted_answer,
        )

    def _should_run_pairwise_reward(
        self, engine: Any | None, turn_artifacts: list[TurnArtifact]
    ) -> bool:
        if not self.pairwise_reward_enabled:
            return False
        if engine is None or not turn_artifacts:
            return False
        try:
            ctx = workflow_context.get()
        except Exception:
            return True
        return not bool(getattr(ctx, "is_eval", False))

    async def _run_pairwise_evaluation(
        self,
        episode_artifact: EpisodeArtifact,
        *,
        episode_lora_version: int | None,
        chat_caller: AReaLEngineChatCaller | None,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ):
        if episode_lora_version is None or chat_caller is None:
            return []
        reference_version = max(
            0, int(episode_lora_version) - self.pairwise_reference_lag_steps
        )

        async def generate_reference_tutor(
            tutor_state: TutorTurnState, reference_version: int
        ):
            return await self._generate_reference_tutor_response(
                tutor_state,
                chat_caller=chat_caller,
                reference_version=reference_version,
            )

        evaluator = PairwiseTutorEvaluator(
            reward_scale=self.pairwise_reward_scale,
            reward_caller=aux_caller,
            generate_reference_tutor=generate_reference_tutor,
            run_student=lambda state: self._run_student(state, aux_caller=aux_caller),
            run_leak_check=lambda task,
            ground_truth,
            teacher_action: self._run_optional_leak_check(
                task,
                ground_truth,
                teacher_action,
                aux_caller=aux_caller,
            ),
            score_answer=lambda task,
            ground_truth,
            student_output: self._score_answer_async(
                task,
                ground_truth,
                student_output,
                answer_judge_caller=answer_judge_caller,
            ),
            compare_all_turns=self.pairwise_compare_all_turns,
            judge_both_incorrect=self.pairwise_judge_both_incorrect,
        )
        results = await evaluator.evaluate(
            episode_artifact, reference_version=reference_version
        )
        return results

    async def _generate_reference_tutor_response(
        self,
        tutor_state: TutorTurnState,
        *,
        chat_caller: AReaLEngineChatCaller,
        reference_version: int,
    ) -> str:
        result = await chat_caller.generate(
            self._build_tutor_messages(tutor_state),
            gconfig=self._generation_config(),
            max_completion_tokens=self.max_completion_tokens,
            max_train_sample_tokens=self.max_train_sample_tokens,
            metadata={"lora_version": int(reference_version)},
            rid_prefix="reference-tutor",
        )
        return result.raw_text

    def _build_tutor_messages(
        self, tutor_state: TutorTurnState
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.teacher_system_prompt},
            {"role": "user", "content": self._build_tutor_prompt(tutor_state)},
        ]

    def _generation_config(self):
        if self.gconfig is not None and hasattr(self.gconfig, "new"):
            return self.gconfig.new(
                n_samples=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_completion_tokens,
            )
        return self.gconfig

    def _log_rollout_stats(
        self,
        *,
        total_reward: float,
        traces: list[TurnTrace],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        success_round = next(
            (
                trace.turn_idx
                for trace in traces
                if trace.judge_correct and not trace.leaked
            ),
            0,
        )
        metrics = {
            "reward": float(total_reward),
            "turns": len(traces),
            "leaks": int(leak_count),
            "pre_solved": float(pre_success),
            "solved": float(success_round > 0),
            "stop/max_turns": float(termination_reason == "max_turns"),
            "stop/context_limit": float(
                termination_reason == CONTEXT_BUDGET_TERMINATION_REASON
            ),
        }
        if success_round > 0:
            metrics["solve_turn"] = int(success_round)

        metrics.update(self._reward_component_metrics(traces))
        _safe_scalar(**metrics)

    def _enabled_reward_component_keys(self) -> list[str]:
        keys = []
        if self.success_reward or self.early_success_bonus:
            keys.append("success")
        if self.leak_penalty:
            keys.append("leak")
        if self.enable_turn_penalty and self.turn_penalty:
            keys.append("turn_penalty")
        if (
            self.length_penalty_threshold_chars > 0
            and self.length_penalty_per_100_chars
        ):
            keys.append("length_penalty")
        if self.pairwise_reward_enabled and self.pairwise_reward_scale:
            keys.append("pairwise")
        return keys

    def _reward_component_metrics(self, traces: list[TurnTrace]) -> dict[str, float]:
        component_totals = dict.fromkeys(self._enabled_reward_component_keys(), 0.0)
        for trace in traces:
            for raw_name, value in trace.reward_components.items():
                component_key = _reward_component_key(raw_name)
                component_totals[component_key] = component_totals.get(
                    component_key, 0.0
                ) + float(value)

        if not component_totals:
            return {}

        total_abs = sum(abs(value) for value in component_totals.values())
        metrics: dict[str, float] = {}
        for component_key, value in component_totals.items():
            metrics[f"reward_component/{component_key}"] = value
            metrics[f"reward_share/{component_key}"] = (
                abs(value) / total_abs if total_abs else 0.0
            )
        return metrics

    def _maybe_dump_debug_trace(
        self,
        *,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        latest_student_answer: str,
        total_reward: float,
        traces: list[TurnTrace],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        if not self.debug_trace_dir:
            return
        try:
            ctx = workflow_context.get()
            task_id = ctx.task_id
            if task_id is None:
                return
            if task_id % self.debug_trace_every_n_rollouts != 0:
                return
            out_dir = Path(self.debug_trace_dir) / ("eval" if ctx.is_eval else "train")
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / f"task_{task_id:08d}_{int(time.time() * 1000)}.json"
            payload = {
                "task_id": task_id,
                "is_eval": bool(ctx.is_eval),
                "termination_reason": termination_reason,
                "total_reward": float(total_reward),
                "num_turns": len(traces),
                "pre_success": bool(pre_success),
                "leak_count": int(leak_count),
                "task": task,
                "ground_truth": ground_truth,
                "initial_student_answer": initial_student_answer,
                "latest_student_answer": latest_student_answer,
                "turns": [trace_to_json(trace) for trace in traces],
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")
