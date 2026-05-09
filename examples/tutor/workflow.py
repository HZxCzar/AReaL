from __future__ import annotations

import json
import logging as py_logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
from examples.common.parsing import join_errors, parse_json_dict
from examples.tutor.prompts import (
    LEAK_CHECK_USER_TEMPLATE,
    PROGRESS_JUDGE_USER_TEMPLATE,
    STUDENT_STATE_USER_TEMPLATE,
    SUMMARY_USER_TEMPLATE,
    TEACHER_STATE_USER_TEMPLATE,
    TRANSFER_GENERATION_USER_TEMPLATE,
    TRANSFER_STUDENT_USER_TEMPLATE,
    render_prompt,
)

logger = logging.getLogger("TutorWorkflow")

ProgressLabel = Literal["improved", "same", "regressed", "unknown"]
ConfidenceLabel = Literal["high", "medium", "low"]
FeedbackKind = Literal["none", "student_judged", "leak"]


@dataclass(slots=True)
class JudgeResult:
    raw_output: str
    correct: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class LeakCheckResult:
    raw_output: str
    leaked: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class GeneratedProblemResult:
    raw_output: str
    task: str
    ground_truth: str
    similarity_notes: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class PublicHistoryState:
    summary: str = ""
    turn_count: int = 0


@dataclass(slots=True)
class TutorPrivateFeedback:
    kind: FeedbackKind = "none"
    student_output: str = ""
    judge_correct: bool = False
    judge_feedback: str = ""
    progress_label: ProgressLabel = "unknown"
    progress_feedback: str = ""
    leak_feedback: str = ""


@dataclass(slots=True)
class TutorTurnState:
    task: str
    ground_truth: str
    public_history: PublicHistoryState
    previous_tutor_visible_output: str
    previous_feedback: TutorPrivateFeedback
    turn_idx: int
    max_turns: int


@dataclass(slots=True)
class StudentTurnState:
    task: str
    public_history: PublicHistoryState
    previous_student_output: str
    latest_tutor_visible_output: str


@dataclass(slots=True)
class ProgressJudgment:
    raw_output: str
    label: ProgressLabel
    confidence: ConfidenceLabel
    feedback: str
    parse_error: str | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnTrace:
    turn_idx: int
    tutor_state: TutorTurnState
    tutor_raw_output: str
    tutor_visible_output: str
    leaked: bool
    student_output: str
    judge_correct: bool
    judge_feedback: str
    progress: ProgressJudgment
    reward: float
    reward_components: dict[str, float]
    public_history_before: str
    public_history_after: str


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
        success_reward: float | None = None,
        leak_penalty: float = -1.0,
        progress_improved_reward: float = 0.3,
        progress_same_reward: float = 0.0,
        progress_regressed_reward: float = -0.3,
        progress_unknown_reward: float = 0.0,
        term_success_reward: float | None = None,
        transfer_bonus_reward: float = 0.0,
        teacher_system_prompt: str = "",
        student_system_prompt: str = "",
        judge_system_prompt: str = "",
        leak_check_system_prompt: str = "",
        generator_system_prompt: str = "",
        summary_system_prompt: str = "",
        progress_judge_system_prompt: str = "",
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
        self.success_reward = (
            float(success_reward)
            if success_reward is not None
            else float(term_success_reward if term_success_reward is not None else 1.0)
        )
        self.leak_penalty = float(leak_penalty)
        self.progress_rewards = {
            "improved": float(progress_improved_reward),
            "same": float(progress_same_reward),
            "regressed": float(progress_regressed_reward),
            "unknown": float(progress_unknown_reward),
        }
        self.transfer_bonus_reward = transfer_bonus_reward
        self.teacher_system_prompt = teacher_system_prompt.strip()
        self.student_system_prompt = student_system_prompt.strip()
        self.judge_system_prompt = judge_system_prompt.strip()
        self.leak_check_system_prompt = leak_check_system_prompt.strip()
        self.generator_system_prompt = generator_system_prompt.strip()
        self.summary_system_prompt = summary_system_prompt.strip()
        self.progress_judge_system_prompt = progress_judge_system_prompt.strip()
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
        results: list[dict[str, torch.Tensor]] = []
        traces: list[TurnTrace] = []
        history: list[dict[str, Any]] = []
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
        judge_result = self._score_aime_answer(task, ground_truth, initial_student_answer)
        if judge_result.correct:
            self.last_history = []
            self.last_traces = []
            self.last_total_reward = 0.0
            termination_reason = "pre_solved"
            self._log_rollout_stats(
                total_reward=0.0,
                history=[],
                traces=[],
                termination_reason=termination_reason,
                pre_success=True,
                leak_count=0,
            )
            self._maybe_dump_debug_trace(
                task=task,
                ground_truth=ground_truth,
                initial_student_answer=initial_student_answer,
                latest_student_answer=initial_student_answer,
                total_reward=0.0,
                traces=[],
                termination_reason=termination_reason,
                pre_success=True,
                leak_count=0,
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
            judge_feedback=judge_result.feedback,
            progress_label="unknown",
            progress_feedback="Initial student attempt was incorrect.",
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
                reward = self.leak_penalty
                results.append(self._response_to_tensordict(response, reward=reward))
                progress = ProgressJudgment(
                    raw_output="",
                    label="unknown",
                    confidence="low",
                    feedback="Skipped because the tutor message leaked private answer information.",
                )
                trace = TurnTrace(
                    turn_idx=turn_idx,
                    tutor_state=tutor_state,
                    tutor_raw_output=tutor_raw_output,
                    tutor_visible_output=tutor_visible_output,
                    leaked=True,
                    student_output="",
                    judge_correct=False,
                    judge_feedback="",
                    progress=progress,
                    reward=reward,
                    reward_components={"leak": reward},
                    public_history_before=public_before,
                    public_history_after=public_before,
                )
                traces.append(trace)
                history.append(self._trace_to_history_record(trace, leak_result))
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
            student_answer, student_error = await self._run_student(student_state)
            student_answer = _strip_reasoning_for_context(student_answer)
            judge_result = self._score_aime_answer(task, ground_truth, student_answer)

            if judge_result.correct:
                progress = ProgressJudgment(
                    raw_output="",
                    label="improved",
                    confidence="high",
                    feedback="The student reached the correct final answer.",
                )
                reward = self.success_reward
                reward_components = {"success": self.success_reward}
                termination_reason = "success"
            else:
                progress = await self._run_progress_judge(
                    task=task,
                    ground_truth=ground_truth,
                    previous_student_answer=previous_student_output,
                    current_student_answer=student_answer,
                    tutor_visible_output=tutor_visible_output,
                )
                reward = self.progress_rewards.get(progress.label, 0.0)
                reward_components = {f"progress_{progress.label}": reward}
                termination_reason = "max_turns" if turn_idx == self.max_turns else "continue"

            results.append(self._response_to_tensordict(response, reward=reward))
            next_public_history = await self._run_public_summary_update(
                old_public_history=public_history,
                previous_student_answer=previous_student_output,
                tutor_visible_output=tutor_visible_output,
                current_student_answer=student_answer,
            )
            trace = TurnTrace(
                turn_idx=turn_idx,
                tutor_state=tutor_state,
                tutor_raw_output=tutor_raw_output,
                tutor_visible_output=tutor_visible_output,
                leaked=False,
                student_output=student_answer,
                judge_correct=judge_result.correct,
                judge_feedback=judge_result.feedback,
                progress=progress,
                reward=reward,
                reward_components=reward_components,
                public_history_before=public_before,
                public_history_after=next_public_history.summary,
            )
            traces.append(trace)
            history.append(self._trace_to_history_record(trace, None, student_error))

            public_history = next_public_history
            previous_tutor_visible_output = tutor_visible_output
            previous_student_output = student_answer
            previous_feedback = TutorPrivateFeedback(
                kind="student_judged",
                student_output=student_answer,
                judge_correct=judge_result.correct,
                judge_feedback=judge_result.feedback,
                progress_label=progress.label,
                progress_feedback=progress.feedback,
            )
            if judge_result.correct:
                break

        total_reward = float(sum(trace.reward for trace in traces))
        self.last_history = history
        self.last_traces = traces
        self.last_total_reward = total_reward
        self._log_rollout_stats(
            total_reward=total_reward,
            history=history,
            traces=traces,
            termination_reason=termination_reason,
            pre_success=False,
            leak_count=leak_count,
        )
        self._maybe_dump_debug_trace(
            task=task,
            ground_truth=ground_truth,
            initial_student_answer=initial_student_answer,
            latest_student_answer=previous_student_output,
            total_reward=total_reward,
            traces=traces,
            termination_reason=termination_reason,
            pre_success=False,
            leak_count=leak_count,
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

    def _response_to_tensordict(
        self, response: ModelResponse, *, reward: float
    ) -> dict[str, torch.Tensor]:
        full_ids = list(response.input_tokens) + list(response.output_tokens)
        output_logprobs = list(response.output_logprobs)
        if len(output_logprobs) < response.output_len:
            output_logprobs.extend([0.0] * (response.output_len - len(output_logprobs)))
        if len(output_logprobs) > response.output_len:
            output_logprobs = output_logprobs[: response.output_len]
        output_versions = list(response.output_versions)
        if len(output_versions) < response.output_len:
            output_versions.extend([0] * (response.output_len - len(output_versions)))
        if len(output_versions) > response.output_len:
            output_versions = output_versions[: response.output_len]
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(
                [0.0] * response.input_len + output_logprobs,
                dtype=torch.float32,
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                [0] * response.input_len + [1] * response.output_len,
                dtype=torch.long,
            ).unsqueeze(0),
            "versions": torch.tensor(
                [-1] * response.input_len + output_versions,
                dtype=torch.long,
            ).unsqueeze(0),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([float(reward)], dtype=torch.float32),
        }

    async def _run_student(
        self,
        state_or_task: StudentTurnState | str,
        teacher_action: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str | None]:
        if isinstance(state_or_task, StudentTurnState):
            prompt = self._build_student_prompt_from_state(state_or_task)
        else:
            prompt = self._build_student_prompt(
                str(state_or_task), teacher_action, history or []
            )
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
        return self._parse_leak_check_result(_strip_reasoning_for_context(raw_output))

    async def _run_progress_judge(
        self,
        *,
        task: str,
        ground_truth: str,
        previous_student_answer: str,
        current_student_answer: str,
        tutor_visible_output: str,
    ) -> ProgressJudgment:
        prompt = self._build_progress_judge_prompt(
            task=task,
            ground_truth=ground_truth,
            previous_student_answer=_strip_reasoning_for_context(previous_student_answer),
            current_student_answer=_strip_reasoning_for_context(current_student_answer),
            tutor_visible_output=_strip_reasoning_for_context(tutor_visible_output),
        )
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.progress_judge_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            return ProgressJudgment(
                raw_output="",
                label="unknown",
                confidence="low",
                feedback=f"Progress judge failed: {exc}",
                parse_error=str(exc),
                raw_result={},
            )
        return self._parse_progress_judgment(_strip_reasoning_for_context(raw_output))

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
            summary = self._parse_public_summary(raw_output)
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
            progress_label=feedback.progress_label,
            progress_feedback=feedback.progress_feedback or "(empty)",
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

    def _build_student_prompt(
        self,
        task: str,
        teacher_action: str | None,
        history: list[dict[str, Any]],
    ) -> str:
        public_summary = "\n".join(self._student_visible_history_summaries(history))
        return self._build_student_prompt_from_state(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(summary=public_summary),
                previous_student_output=self._latest_visible_student_answer(history),
                latest_tutor_visible_output=teacher_action
                or "(none, produce the first answer attempt)",
            )
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

    def _build_progress_judge_prompt(
        self,
        *,
        task: str,
        ground_truth: str,
        previous_student_answer: str,
        current_student_answer: str,
        tutor_visible_output: str,
    ) -> str:
        return render_prompt(
            PROGRESS_JUDGE_USER_TEMPLATE,
            task=task,
            ground_truth=ground_truth,
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

    def _parse_public_summary(self, raw_output: str) -> str:
        text = _strip_reasoning_for_context(raw_output)
        parsed, _ = parse_json_dict(text)
        if isinstance(parsed, dict):
            parts = []
            for key in [
                "student_progress",
                "visible_tutor_guidance",
                "student_current_misconception",
                "latest_student_state",
            ]:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(f"{key}: {value.strip()}")
            if parts:
                return "\n".join(parts)
        return text.strip()

    def _parse_progress_judgment(self, raw_output: str) -> ProgressJudgment:
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            lowered = raw_output.lower()
            if "improved" in lowered:
                label: ProgressLabel = "improved"
            elif "regressed" in lowered or "worse" in lowered:
                label = "regressed"
            elif "same" in lowered or "unchanged" in lowered:
                label = "same"
            else:
                label = "unknown"
            return ProgressJudgment(
                raw_output=raw_output,
                label=label,
                confidence="low",
                feedback="Failed to parse progress-judge output.",
                parse_error=parse_error,
                raw_result={},
            )
        label_value = parsed.get("label", parsed.get("progress", "unknown"))
        if label_value not in {"improved", "same", "regressed", "unknown"}:
            parse_error = join_errors(
                parse_error,
                '"label" must be one of improved, same, regressed, unknown',
            )
            label_value = "unknown"
        confidence_value = parsed.get("confidence", "low")
        if confidence_value not in {"high", "medium", "low"}:
            confidence_value = "low"
        feedback = parsed.get("feedback", "")
        if not isinstance(feedback, str):
            feedback = str(feedback)
        return ProgressJudgment(
            raw_output=raw_output,
            label=label_value,
            confidence=confidence_value,
            feedback=feedback or "No progress feedback provided.",
            parse_error=parse_error,
            raw_result=parsed,
        )

    def _parse_leak_check_result(self, raw_output: str) -> LeakCheckResult:
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            lowered = raw_output.lower()
            leaked = '"leaked": true' in lowered or re.search(r"\byes\b", lowered) is not None
            return LeakCheckResult(
                raw_output=raw_output,
                leaked=leaked,
                feedback="Failed to parse leak-check output.",
                parse_error=parse_error,
                raw_result={},
            )
        leaked = parsed.get("leaked")
        feedback = parsed.get("feedback", "")
        if not isinstance(leaked, bool):
            parse_error = join_errors(parse_error, '"leaked" must be a boolean')
            leaked = False
        if not isinstance(feedback, str):
            parse_error = join_errors(parse_error, '"feedback" must be a string')
            feedback = str(feedback)
        return LeakCheckResult(
            raw_output=raw_output,
            leaked=leaked,
            feedback=feedback or (
                "The teacher revealed the answer directly. The student did not see this turn."
                if leaked
                else "No answer leakage detected."
            ),
            parse_error=parse_error,
            raw_result=parsed,
        )

    def _score_aime_answer(
        self, task: str, ground_truth: str, student_answer: str
    ) -> JudgeResult:
        extracted_answer = _official_extract_aime_answer(student_answer)
        normalized_prediction = _official_strip_string(extracted_answer) if extracted_answer else ""
        normalized_target = _official_strip_string(ground_truth)
        correct = _official_is_equiv(extracted_answer, ground_truth)
        raw_result = {
            "method": "lm_eval_aime_exact_match",
            "task": task,
            "student_answer": _strip_reasoning_for_context(student_answer),
            "extracted_answer": extracted_answer,
            "normalized_prediction": normalized_prediction,
            "normalized_target": normalized_target,
        }
        return JudgeResult(
            raw_output=json.dumps(
                {
                    "correct": correct,
                    "feedback": "Correct." if correct else "Incorrect.",
                    "scoring": raw_result,
                },
                ensure_ascii=True,
                indent=2,
            ),
            correct=correct,
            feedback="Correct." if correct else "Incorrect.",
            parse_error=None,
            raw_result=raw_result,
        )

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

    def _trace_to_history_record(
        self,
        trace: TurnTrace,
        leak_result: LeakCheckResult | None,
        student_error: str | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "round_idx": trace.turn_idx,
            "teacher_raw_output": trace.tutor_raw_output,
            "teacher_action": trace.tutor_visible_output,
            "student_answer": trace.student_output,
            "student_error": student_error,
            "judge_feedback": trace.judge_feedback,
            "judge_correct": trace.judge_correct,
            "progress_label": trace.progress.label,
            "progress_feedback": trace.progress.feedback,
            "reward": trace.reward,
            "reward_components": dict(trace.reward_components),
            "leak_detected": trace.leaked,
            "public_history_before": trace.public_history_before,
            "public_history_after": trace.public_history_after,
        }
        if leak_result is not None:
            record["leak_feedback"] = leak_result.feedback
        return record

    def _log_rollout_stats(
        self,
        *,
        total_reward: float,
        history: list[dict[str, Any]],
        traces: list[TurnTrace],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        progress_counts = {"improved": 0, "same": 0, "regressed": 0, "unknown": 0}
        for trace in traces:
            progress_counts[trace.progress.label] += 1
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
            progress_improved=progress_counts["improved"],
            progress_same=progress_counts["same"],
            progress_regressed=progress_counts["regressed"],
            progress_unknown=progress_counts["unknown"],
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
                "turns": [self._trace_to_json(trace) for trace in traces],
            }
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")

    def _trace_to_json(self, trace: TurnTrace) -> dict[str, Any]:
        data = asdict(trace)
        data["tutor_state"]["ground_truth"] = trace.tutor_state.ground_truth
        return data

    def _student_visible_history_summaries(
        self, history: list[dict[str, Any]]
    ) -> list[str]:
        summaries: list[str] = []
        for record in history:
            if record.get("leak_detected"):
                continue
            summary = record.get("public_history_after") or record.get("student_visible_summary")
            if isinstance(summary, str) and summary.strip():
                summaries.append(_strip_reasoning_for_context(summary))
        return summaries

    def _latest_visible_student_answer(self, history: list[dict[str, Any]]) -> str:
        for record in reversed(history):
            if not record.get("leak_detected") and record.get("student_answer"):
                return _strip_reasoning_for_context(str(record["student_answer"]))
        return ""

    # Legacy transfer helpers are retained for manual/demo tooling, but training no longer calls them.
    async def _run_transfer_student(
        self,
        *,
        original_task: str,
        initial_student_answer: str,
        history: list[dict[str, Any]],
        transfer_task: str,
    ) -> tuple[str, str | None]:
        prompt = self._build_transfer_student_prompt(
            original_task=original_task,
            initial_student_answer=initial_student_answer,
            history=history,
            transfer_task=transfer_task,
        )
        try:
            return _strip_reasoning_for_context(
                await self._call_student_prompt(prompt)
            ), None
        except Exception as exc:
            return "", str(exc)

    async def _run_transfer_round(
        self,
        *,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        generation = await self._run_transfer_generation(task, ground_truth)
        payload: dict[str, Any] = {
            "transfer_triggered": True,
            "transfer_task": generation.task,
            "transfer_ground_truth": generation.ground_truth,
            "transfer_generation_error": generation.parse_error,
            "transfer_success": False,
        }
        if not generation.task or not generation.ground_truth:
            return payload
        answer, answer_error = await self._run_transfer_student(
            original_task=task,
            initial_student_answer=initial_student_answer,
            history=history,
            transfer_task=generation.task,
        )
        judge_result = self._score_aime_answer(
            generation.task, generation.ground_truth, answer
        )
        payload.update(
            {
                "transfer_student_answer": answer,
                "transfer_student_error": answer_error,
                "transfer_judge_feedback": judge_result.feedback,
                "transfer_success": judge_result.correct,
            }
        )
        return payload

    async def _run_transfer_generation(
        self, task: str, ground_truth: str
    ) -> GeneratedProblemResult:
        prompt = self._build_transfer_generation_prompt(task, ground_truth)
        try:
            raw_output = await self.aux_caller.call_text(
                [
                    {"role": "system", "content": self.generator_system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            error = f"Generator call failed: {exc}"
            return GeneratedProblemResult("", "", "", "", error, {})
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            return GeneratedProblemResult(raw_output, "", "", "", parse_error, {})
        task_value = parsed.get("task", "")
        gt_value = parsed.get("ground_truth", "")
        notes = parsed.get("similarity_notes", "")
        if not isinstance(task_value, str):
            parse_error = join_errors(parse_error, '"task" must be a string')
            task_value = ""
        if not isinstance(gt_value, str):
            if isinstance(gt_value, (int, float)) and not isinstance(gt_value, bool):
                gt_value = str(int(gt_value) if isinstance(gt_value, int) else gt_value)
            else:
                parse_error = join_errors(
                    parse_error, '"ground_truth" must be a string or number'
                )
                gt_value = ""
        if not isinstance(notes, str):
            notes = str(notes)
        return GeneratedProblemResult(
            raw_output=raw_output,
            task=task_value.strip(),
            ground_truth=gt_value.strip(),
            similarity_notes=notes.strip(),
            parse_error=parse_error,
            raw_result=parsed,
        )

    def _build_transfer_student_prompt(
        self,
        *,
        original_task: str,
        initial_student_answer: str,
        history: list[dict[str, Any]],
        transfer_task: str,
    ) -> str:
        return render_prompt(
            TRANSFER_STUDENT_USER_TEMPLATE,
            original_task=original_task,
            initial_student_answer=_strip_reasoning_for_context(initial_student_answer),
            visible_history=self._student_visible_history_summaries(history),
            transfer_task=transfer_task,
        )

    def _build_transfer_generation_prompt(self, task: str, ground_truth: str) -> str:
        return render_prompt(
            TRANSFER_GENERATION_USER_TEMPLATE,
            task=task,
            ground_truth=ground_truth,
        )


def _strip_reasoning_for_context(text: str) -> str:
    text = text or ""
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE).strip()


def _strip_think_tags(text: str) -> str:
    return _strip_reasoning_for_context(text)


def _compact_text(text: str, max_chars: int = 240) -> str:
    compact = " ".join(_strip_reasoning_for_context(text).split())
    if not compact:
        return "(empty)"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_string += "\\sqrt{" + a + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _official_strip_string(string: str) -> str:
    return _strip_string(string)


def _official_is_equiv(prediction: str, reference: str) -> bool:
    return _official_strip_string(prediction) == _official_strip_string(reference)


def _official_extract_aime_answer(response: str) -> str:
    matches = list(
        re.finditer(r"(?:^|[^0-9])([0-9]{1,4})(?:[^0-9]|$)", response or "")
    )
    if not matches:
        return (response or "").strip()
    return matches[-1].group(1)
