from __future__ import annotations

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
    from areal.api import ModelRequest, ModelResponse, RolloutWorkflow
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
                    for current, target in reversed(list(zip(value.shape[1:], max_shape[1:]))):
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
from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig, make_teacher_client
from examples.tutor.core.aime import score_aime_answer
from examples.tutor.core.history import (
    trace_to_history_record,
    trace_to_json,
)
from examples.tutor.core.parsers import (
    parse_leak_check_result,
    parse_public_summary,
)
from examples.tutor.core.rewards import EpisodeRewardComputer, artifact_to_trace
from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.text import (
    compact_text as _compact_text,
    strip_reasoning_for_context as _strip_reasoning_for_context,
)
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PublicHistoryState,
    StudentTurnState,
    TutorPrivateFeedback,
    TutorTurnState,
    TurnArtifact,
    TurnTrace,
)
from examples.tutor.prompts import (
    LEAK_CHECK_USER_TEMPLATE,
    STUDENT_STATE_USER_TEMPLATE,
    SUMMARY_USER_TEMPLATE,
    TEACHER_STATE_USER_TEMPLATE,
    render_prompt,
)

logger = logging.getLogger("TutorWorkflow")


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


class TutorAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any | None = None,
        tokenizer: str | Any | None = None,
        max_turns: int = 6,
        enable_thinking: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        tool_call_parser: str = "qwen25",
        reasoning_parser: str = "qwen3",
        aux_base_url: str = "http://127.0.0.1:30000/v1",
        aux_model: str = "qwen-aux",
        aux_api_key: str = "EMPTY",
        aux_timeout: int = 120,
        aux_max_tokens: int = 1024,
        aux_temperature: float = 0.7,
        aux_top_p: float | None = None,
        max_concurrent_aux_calls: int = 8,
        api_params_config_path: str | None = None,
        api_params_key: str | None = None,
        success_reward: float = 1.0,
        leak_penalty: float = -1.0,
        outcome_prior_turn_weight: float = 0.1,
        outcome_credit_gamma: float = 0.9,
        early_success_bonus: float = 0.3,
        turn_penalty: float = -0.01,
        length_penalty_threshold_chars: int = 1200,
        length_penalty_per_100_chars: float = -0.005,
        length_penalty_min: float = -0.1,
        teacher_system_prompt: str = "",
        student_system_prompt: str = "",
        leak_check_system_prompt: str = "",
        summary_system_prompt: str = "",
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 1,
        max_episode_total_tokens: int | None = None,
        token_budget_penalty: float = -1.0,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
    ):
        self.max_turns = max_turns
        self.enable_thinking = enable_thinking
        self.gconfig = gconfig
        self.temperature = gconfig.temperature if gconfig is not None else temperature
        self.top_p = gconfig.top_p if gconfig is not None else top_p
        self.max_completion_tokens = (
            gconfig.max_new_tokens if gconfig is not None else max_completion_tokens
        )
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.success_reward = float(success_reward)
        self.leak_penalty = float(leak_penalty)
        self.outcome_prior_turn_weight = float(outcome_prior_turn_weight)
        self.outcome_credit_gamma = float(outcome_credit_gamma)
        self.early_success_bonus = float(early_success_bonus)
        self.turn_penalty = float(turn_penalty)
        self.length_penalty_threshold_chars = int(length_penalty_threshold_chars)
        self.length_penalty_per_100_chars = float(length_penalty_per_100_chars)
        self.length_penalty_min = float(length_penalty_min)
        self.teacher_system_prompt = teacher_system_prompt.strip()
        self.student_system_prompt = student_system_prompt.strip()
        self.leak_check_system_prompt = leak_check_system_prompt.strip()
        self.summary_system_prompt = summary_system_prompt.strip()
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
        self.max_episode_total_tokens = max_episode_total_tokens
        self.token_budget_penalty = token_budget_penalty
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
            api_params_config_path=api_params_config_path,
            api_params_key=api_params_key,
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self.aux_caller = AsyncLLMCaller(aux_config)

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
        turn_artifacts: list[TurnArtifact] = []
        leak_count = 0
        termination_reason = "max_turns"

        initial_student_answer, initial_student_error = await self._run_student(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(),
                previous_student_output="",
                latest_tutor_visible_output="(none, produce the first answer attempt)",
            )
        )
        initial_student_answer = _strip_reasoning_for_context(initial_student_answer)
        initial_judge_result = self._score_aime_answer(
            task, ground_truth, initial_student_answer
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
            response, tutor_raw_output = await self._generate_tutor_response(
                tutor_state,
                engine=engine,
                external_client=external_client,
            )
            tutor_visible_output = _strip_reasoning_for_context(tutor_raw_output)
            public_before = public_history.summary

            leak_result = await self._run_leak_check(
                task, ground_truth, tutor_visible_output
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
                termination_reason = "max_turns" if turn_idx == self.max_turns else "continue"
                continue

            student_state = StudentTurnState(
                task=task,
                public_history=public_history,
                previous_student_output=previous_student_output,
                latest_tutor_visible_output=tutor_visible_output,
            )
            student_prompt = self._build_student_prompt_from_state(student_state)
            student_answer, student_error = await self._run_student(student_state)
            student_answer = _strip_reasoning_for_context(student_answer)
            judge_result = self._score_aime_answer(task, ground_truth, student_answer)

            if judge_result.correct:
                termination_reason = "success"
            else:
                termination_reason = "max_turns" if turn_idx == self.max_turns else "continue"

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
            outcome_prior_turn_weight=self.outcome_prior_turn_weight,
            outcome_credit_gamma=self.outcome_credit_gamma,
            early_success_bonus=self.early_success_bonus,
            turn_penalty=self.turn_penalty,
            length_penalty_threshold_chars=self.length_penalty_threshold_chars,
            length_penalty_per_100_chars=self.length_penalty_per_100_chars,
            length_penalty_min=self.length_penalty_min,
        )
        assignments = await reward_computer.compute(episode_artifact)
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

    async def _generate_tutor_response(
        self,
        tutor_state: TutorTurnState,
        *,
        engine: Any | None,
        external_client: Any | None,
    ) -> tuple[ModelResponse, str]:
        messages = [
            {"role": "system", "content": self.teacher_system_prompt},
            {"role": "user", "content": self._build_tutor_prompt(tutor_state)},
        ]
        input_ids = self._apply_chat_template(messages)
        if engine is not None:
            req = ModelRequest(
                rid=f"tutor-{int(time.time() * 1000)}-{tutor_state.turn_idx}",
                input_ids=input_ids,
                gconfig=self._generation_config(),
                tokenizer=self.tokenizer,
            )
            response = await engine.agenerate(req)
            raw_output = self._decode_output(response)
            return response, raw_output

        safe_max_completion_tokens, _ = self.teacher_context_budget.clamp_max_completion_tokens(
            messages,
            int(self.max_completion_tokens),
        )
        response_obj = await external_client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=max(1, safe_max_completion_tokens),
        )
        raw_output = response_obj.choices[0].message.content or ""
        output_tokens = self._encode_text(raw_output)
        return (
            ModelResponse(
                input_tokens=list(input_ids),
                output_tokens=output_tokens,
                output_logprobs=[0.0] * len(output_tokens),
                output_versions=[0] * len(output_tokens),
                tokenizer=self.tokenizer,
            ),
            raw_output,
        )

    async def _run_student(
        self,
        state: StudentTurnState,
    ) -> tuple[str, str | None]:
        prompt = self._build_student_prompt_from_state(state)
        try:
            return _strip_reasoning_for_context(
                await self._call_student_prompt(prompt)
            ), None
        except Exception as exc:
            return "", str(exc)

    async def _call_student_prompt(self, prompt: str) -> str:
        answer = await self.aux_caller.call_text(
            [
                {"role": "system", "content": self.student_system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return _strip_reasoning_for_context(answer)

    async def _run_leak_check(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> LeakCheckResult:
        prompt = self._build_leak_check_prompt(
            task,
            ground_truth,
            _strip_reasoning_for_context(teacher_action),
        )
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.leak_check_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            return LeakCheckResult("", False, f"Leak check failed: {exc}", str(exc), {})
        return parse_leak_check_result(_strip_reasoning_for_context(raw_output))

    async def _run_public_summary_update(
        self,
        *,
        old_public_history: PublicHistoryState,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> PublicHistoryState:
        prompt = self._build_summary_prompt(
            old_public_history=old_public_history,
            previous_student_answer=_strip_reasoning_for_context(previous_student_answer),
            tutor_visible_output=_strip_reasoning_for_context(tutor_visible_output),
            current_student_answer=_strip_reasoning_for_context(current_student_answer),
        )
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            summary = parse_public_summary(raw_output)
        except Exception as exc:
            logger.warning("Public summary update failed: %s", exc)
            summary = self._fallback_public_summary(
                old_public_history.summary,
                previous_student_answer,
                tutor_visible_output,
                current_student_answer,
            )
        return PublicHistoryState(
            summary=_strip_reasoning_for_context(summary),
            turn_count=old_public_history.turn_count + 1,
        )

    def _build_tutor_prompt(self, state: TutorTurnState) -> str:
        feedback = state.previous_feedback
        return render_prompt(
            TEACHER_STATE_USER_TEMPLATE,
            task=state.task,
            ground_truth=state.ground_truth,
            public_history=state.public_history.summary or "No visible tutoring history yet.",
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
            public_history=state.public_history.summary or "No previous visible tutoring history.",
            previous_student_output=state.previous_student_output or "(empty)",
            teacher_feedback=state.latest_tutor_visible_output or "(none)",
        )

    def _build_summary_prompt(
        self,
        *,
        old_public_history: PublicHistoryState,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> str:
        return render_prompt(
            SUMMARY_USER_TEMPLATE,
            old_public_summary=old_public_history.summary or "(empty)",
            previous_student_answer=previous_student_answer or "(empty)",
            tutor_output=tutor_visible_output or "(empty)",
            current_student_answer=current_student_answer or "(empty)",
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
        return (
            "Initial student attempt: "
            f"{_compact_text(_strip_reasoning_for_context(initial_student_answer), 1200)}"
        )

    def _fallback_public_summary(
        self,
        old_summary: str,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
    ) -> str:
        update = (
            f"Previous student answer: {_compact_text(previous_student_answer)}\n"
            f"Tutor guidance shown: {_compact_text(tutor_visible_output)}\n"
            f"Student reply after guidance: {_compact_text(current_student_answer)}"
        )
        if not old_summary:
            return update
        return f"{old_summary}\n{update}"

    def _score_aime_answer(
        self, task: str, ground_truth: str, student_answer: str
    ) -> JudgeResult:
        return score_aime_answer(task, ground_truth, student_answer)

    def _generation_config(self):
        if self.gconfig is not None and hasattr(self.gconfig, "new"):
            return self.gconfig.new(
                n_samples=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_completion_tokens,
            )
        return self.gconfig

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return list(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    )
                )
            except TypeError:
                return list(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
        text = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        return self._encode_text(text)

    def _decode_output(self, response: ModelResponse) -> str:
        tokenizer = response.tokenizer or self.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(response.output_tokens, skip_special_tokens=False).replace(
                    "<|im_end|>", ""
                )
            except TypeError:
                return tokenizer.decode(response.output_tokens).replace("<|im_end|>", "")
        return "".join(chr(max(0, int(token))) for token in response.output_tokens)

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        return [ord(ch) for ch in text]

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
            (trace.turn_idx for trace in traces if trace.judge_correct and not trace.leaked),
            0,
        )
        _safe_scalar(
            reward=float(total_reward),
            num_turns=len(traces),
            samples_per_episode=len(traces),
            leak_count=int(leak_count),
            pre_success=float(pre_success),
            term_success=float(success_round > 0),
            success_round=int(success_round),
            termination_pre_solved=float(termination_reason == "pre_solved"),
            termination_success=float(termination_reason == "success"),
            termination_max_turns=float(termination_reason == "max_turns"),
        )

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

