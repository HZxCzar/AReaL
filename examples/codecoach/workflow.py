from __future__ import annotations

import asyncio
import json
import logging as py_logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

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
            return type(
                "WorkflowContext",
                (), {"task_id": None, "is_eval": False, "lora_version": None},
            )()

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
from examples.common.files import make_run_dir
from examples.common.openai_utils import (
    AsyncLLMCaller,
    AuxModelConfig,
    make_teacher_client,
)
from examples.common.parsing import join_errors, parse_json_dict
from examples.math_tutor.core.callers import (
    ApiAuxiliaryCaller,
    AReaLEngineActorCaller,
    AReaLEngineAuxiliaryCaller,
    AReaLEngineChatCaller,
    ExternalActorCaller,
)
from examples.math_tutor.core.generation_budget import (
    CONTEXT_BUDGET_TERMINATION_REASON,
    ContextBudgetLimitExceeded,
)
from examples.math_tutor.core.tensors import response_to_tensordict
from examples.math_tutor.core.text import strip_reasoning_for_context

logger = logging.getLogger("CodeCoachWorkflow")

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are the teacher in a code-coaching environment. Give exactly one concise "
    "natural-language hint, correction, or debugging direction. Do not output code. "
    "Help the student improve the submitted program using the evaluator feedback."
)

DEFAULT_STUDENT_SYSTEM_PROMPT = (
    "You are a careful coding student. Output strict JSON only."
)

PairwiseOutcome = Literal["current", "reference", "tie", "failure"]


@dataclass(slots=True)
class StudentReply:
    raw_output: str
    message: str
    replace_code: bool
    code: str | None
    parse_error: str | None


@dataclass(slots=True)
class EvalResult:
    score: float
    target_ratio: float
    validity: float
    success: bool
    raw_result: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class PublicHistoryState:
    summary: str = ""
    turn_count: int = 0


@dataclass(slots=True)
class TeacherPrivateFeedback:
    kind: str = "initial"
    student_message: str = ""
    code_updated: bool = False
    error: str = ""
    eval_result: EvalResult | None = None
    best_eval: EvalResult | None = None
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TeacherTurnState:
    task_markdown: str
    current_code: str
    current_eval: EvalResult
    best_eval: EvalResult
    entry_function: str
    public_history: PublicHistoryState
    previous_teacher_visible_output: str
    previous_feedback: TeacherPrivateFeedback
    turn_idx: int
    max_turns: int


@dataclass(slots=True)
class StudentTurnState:
    task_markdown: str
    current_code: str
    teacher_visible_output: str
    entry_function: str
    public_history: PublicHistoryState
    turn_idx: int


@dataclass(slots=True)
class StudentRunResult:
    raw_output: str
    reply: StudentReply
    error: str | None


@dataclass(slots=True)
class TurnArtifact:
    turn_idx: int
    teacher_state: TeacherTurnState
    teacher_prompt: str
    teacher_response: ModelResponse
    teacher_raw_output: str
    teacher_visible_output: str
    student_state: StudentTurnState
    student_prompt: str
    student_raw_output: str
    student_reply: StudentReply
    code_updated: bool
    candidate_code: str
    eval_result: EvalResult
    best_eval_before: EvalResult
    best_eval_after: EvalResult
    public_history_before: str
    public_history_after: str
    error: str | None
    termination_feedback: str | None
    base_reward_components: dict[str, float]


@dataclass(slots=True)
class EpisodeArtifact:
    task_markdown: str
    initial_code: str
    initial_eval: EvalResult
    turns: list[TurnArtifact]
    termination_reason: str
    latest_code: str
    latest_eval: EvalResult
    best_eval: EvalResult


@dataclass(slots=True)
class RewardAssignment:
    reward: float
    reward_components: dict[str, float]


@dataclass(slots=True)
class TurnTrace:
    turn_idx: int
    teacher_raw_output: str
    teacher_visible_output: str
    student_message: str
    code_updated: bool
    eval_result: EvalResult
    best_eval: EvalResult
    error: str | None
    reward: float
    reward_components: dict[str, float]
    public_history_before: str
    public_history_after: str


@dataclass(slots=True)
class ReferenceTurnArtifact:
    turn_idx: int
    reference_version: int
    teacher_raw_output: str
    teacher_visible_output: str
    student_raw_output: str
    student_reply: StudentReply
    code_updated: bool
    candidate_code: str
    eval_result: EvalResult
    error: str | None


@dataclass(slots=True)
class PairwiseTurnResult:
    turn_idx: int
    reference_version: int
    outcome: PairwiseOutcome
    reward: float
    reason: str
    reference: ReferenceTurnArtifact | None = None


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class CodeCoachAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any | None = None,
        tokenizer: str | Any | None = None,
        max_turns: int = 6,
        enable_thinking: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        aux_mode: str = "api",
        aux_enable_thinking: bool = False,
        aux_base_url: str = "http://127.0.0.1:30000/v1",
        aux_model: str = "qwen-student",
        aux_api_key: str = "EMPTY",
        aux_timeout: int = 120,
        aux_max_tokens: int = 4096,
        aux_temperature: float = 0.7,
        aux_top_p: float | None = None,
        max_concurrent_aux_calls: int = 8,
        aux_request_params: dict[str, Any] | None = None,
        work_dir_root: str = "examples/codecoach/artifacts",
        max_train_sample_tokens: int | None = None,
        token_budget_penalty: float = -0.2,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
        pairwise_reward_enabled: bool = False,
        pairwise_reference_lag_steps: int = 5,
        pairwise_reward_scale: float = 0.05,
        pairwise_compare_all_turns: bool = True,
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 10,
    ):
        self.max_turns = int(max_turns)
        self.gconfig = gconfig
        self.temperature = gconfig.temperature if gconfig is not None else temperature
        self.top_p = gconfig.top_p if gconfig is not None else top_p
        self.max_completion_tokens = (
            gconfig.max_new_tokens if gconfig is not None else max_completion_tokens
        )
        self.enable_thinking = bool(enable_thinking)
        if aux_mode not in {"api", "self"}:
            raise ValueError(f"aux_mode must be 'api' or 'self', got {aux_mode!r}")
        self.aux_mode = aux_mode
        self.aux_enable_thinking = bool(aux_enable_thinking)
        self.aux_max_tokens = int(aux_max_tokens)
        self.aux_temperature = float(aux_temperature)
        self.aux_top_p = aux_top_p
        self.max_concurrent_aux_calls = int(max_concurrent_aux_calls)
        self._self_aux_semaphore = asyncio.Semaphore(
            max(1, self.max_concurrent_aux_calls)
        )
        self.work_dir_root = work_dir_root
        self.max_train_sample_tokens = max_train_sample_tokens
        self.token_budget_penalty = float(token_budget_penalty)
        self.context_window_margin = int(context_window_margin)
        self.pairwise_reward_enabled = bool(pairwise_reward_enabled)
        self.pairwise_reference_lag_steps = max(0, int(pairwise_reference_lag_steps))
        self.pairwise_reward_scale = float(pairwise_reward_scale)
        self.pairwise_compare_all_turns = bool(pairwise_compare_all_turns)
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
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
            request_params=aux_request_params or {},
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self.api_aux_caller = (
            ApiAuxiliaryCaller(AsyncLLMCaller(aux_config))
            if self.aux_mode == "api"
            else None
        )

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
            versions.add(max(0, actor_version - self.pairwise_reference_lag_steps))
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
        external_client: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if (engine is None) == (external_client is None):
            raise ValueError("Exactly one teacher generation source must be provided.")

        run_dir = make_run_dir(self.work_dir_root, "codecoach")
        task_markdown = str(data["task_markdown"])
        initial_code = str(data["initial_code"])
        evaluator_path = str(data["evaluator_path"])
        target_score = float(data["target_score"])
        entry_function = str(data["entry_function"])
        eval_timeout_sec = int(data.get("eval_timeout_sec", 120))

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
            self._make_engine_chat_caller(engine, enable_thinking=self.enable_thinking)
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

        initial_eval = self._evaluate_code(
            evaluator_path=evaluator_path,
            target_score=target_score,
            code=initial_code,
            eval_root=run_dir / "initial_eval",
            eval_timeout_sec=eval_timeout_sec,
        )
        current_code = initial_code
        current_eval = initial_eval
        best_eval = initial_eval
        best_ratio_history: list[float] = []
        turn_artifacts: list[TurnArtifact] = []
        termination_reason = "max_turns"
        public_history = PublicHistoryState(
            summary=self._build_initial_public_summary(initial_eval),
            turn_count=0,
        )
        previous_teacher_visible_output = ""
        previous_feedback = TeacherPrivateFeedback(
            kind="initial",
            eval_result=initial_eval,
            best_eval=best_eval,
        )

        for turn_idx in range(1, self.max_turns + 1):
            best_eval_before = best_eval
            teacher_state = TeacherTurnState(
                task_markdown=task_markdown,
                current_code=current_code,
                current_eval=current_eval,
                best_eval=best_eval,
                entry_function=entry_function,
                public_history=public_history,
                previous_teacher_visible_output=previous_teacher_visible_output,
                previous_feedback=previous_feedback,
                turn_idx=turn_idx,
                max_turns=self.max_turns,
            )
            public_before = public_history.summary
            try:
                teacher_response, teacher_raw_output = await self._generate_teacher_response(
                    teacher_state,
                    actor_caller=actor_caller,
                    lora_version=episode_lora_version,
                )
            except ContextBudgetLimitExceeded as exc:
                logger.info(
                    "Terminating CodeCoach episode at turn %s due to context budget: %s",
                    turn_idx,
                    exc,
                )
                termination_reason = CONTEXT_BUDGET_TERMINATION_REASON
                break
            teacher_visible_output = strip_reasoning_for_context(teacher_raw_output)

            student_state = StudentTurnState(
                task_markdown=task_markdown,
                current_code=current_code,
                teacher_visible_output=teacher_visible_output,
                entry_function=entry_function,
                public_history=public_history,
                turn_idx=turn_idx,
            )
            student_prompt = self._build_student_prompt(student_state)
            student_result = await self._run_student(
                student_state,
                aux_caller=aux_caller,
            )
            candidate_code, code_updated, update_error = self._apply_student_reply(
                current_code,
                student_result.reply,
                student_result.error,
            )

            eval_result = self._evaluate_code(
                evaluator_path=evaluator_path,
                target_score=target_score,
                code=candidate_code,
                eval_root=run_dir / f"round_{turn_idx:02d}_eval",
                eval_timeout_sec=eval_timeout_sec,
            )
            current_code = candidate_code
            current_eval = eval_result
            if self._compare_eval(eval_result, best_eval) > 0:
                best_eval = eval_result
            best_ratio_history.append(best_eval.target_ratio)
            base_reward = self._current_auc_gain(
                best_ratio_history,
                initial_eval.target_ratio,
            )
            base_components = {"progress_auc": base_reward}
            termination_feedback = None
            if teacher_response.stop_reason == "length":
                base_components["length_penalty"] = self.token_budget_penalty
                termination_feedback = "Teacher generation stopped by length limit."
                termination_reason = "length"
            error = join_errors(update_error, eval_result.error)
            if eval_result.target_ratio >= 1.0:
                termination_reason = "success"
            elif turn_idx == self.max_turns and termination_reason == "max_turns":
                termination_reason = "max_turns"
            elif termination_reason not in {"length", "success"}:
                termination_reason = "continue"

            next_public_history = self._run_public_summary_update(
                old_public_history=public_history,
                teacher_visible_output=teacher_visible_output,
                student_reply=student_result.reply,
                code_updated=code_updated,
                eval_result=eval_result,
                best_eval=best_eval,
                error=error,
            )
            artifact = TurnArtifact(
                turn_idx=turn_idx,
                teacher_state=teacher_state,
                teacher_prompt=self._build_teacher_prompt(teacher_state),
                teacher_response=teacher_response,
                teacher_raw_output=teacher_raw_output,
                teacher_visible_output=teacher_visible_output,
                student_state=student_state,
                student_prompt=student_prompt,
                student_raw_output=student_result.raw_output,
                student_reply=student_result.reply,
                code_updated=code_updated,
                candidate_code=candidate_code,
                eval_result=eval_result,
                best_eval_before=best_eval_before,
                best_eval_after=best_eval,
                public_history_before=public_before,
                public_history_after=next_public_history.summary,
                error=error,
                termination_feedback=termination_feedback,
                base_reward_components=base_components,
            )
            turn_artifacts.append(artifact)

            public_history = next_public_history
            previous_teacher_visible_output = teacher_visible_output
            previous_feedback = TeacherPrivateFeedback(
                kind="student_evaluated",
                student_message=student_result.reply.message,
                code_updated=code_updated,
                error=error or "",
                eval_result=eval_result,
                best_eval=best_eval,
                reward_components=dict(base_components),
            )
            if termination_reason in {"success", "length"}:
                break
            termination_reason = "max_turns" if turn_idx == self.max_turns else "continue"

        latest_eval = current_eval
        episode_artifact = EpisodeArtifact(
            task_markdown=task_markdown,
            initial_code=initial_code,
            initial_eval=initial_eval,
            turns=turn_artifacts,
            termination_reason=termination_reason,
            latest_code=current_code,
            latest_eval=latest_eval,
            best_eval=best_eval,
        )
        pairwise_rewards: dict[int, float] = {}
        if self._should_run_pairwise_reward(engine, turn_artifacts):
            pairwise_results = await self._run_pairwise_evaluation(
                episode_artifact,
                episode_lora_version=episode_lora_version,
                chat_caller=actor_chat_caller,
                aux_caller=aux_caller,
                evaluator_path=evaluator_path,
                target_score=target_score,
                eval_timeout_sec=eval_timeout_sec,
                run_dir=run_dir,
            )
            pairwise_rewards = {
                result.turn_idx: result.reward
                for result in pairwise_results
                if result.reward
            }

        assignments = self._assign_rewards(turn_artifacts, pairwise_rewards)
        traces = [
            self._artifact_to_trace(artifact, assignment)
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        history = [self._trace_to_history_record(trace) for trace in traces]
        results = [
            response_to_tensordict(artifact.teacher_response, reward=assignment.reward)
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        total_reward = float(sum(assignment.reward for assignment in assignments))
        self.last_history = history
        self.last_traces = traces
        self.last_total_reward = total_reward
        self._log_rollout_stats(
            total_reward=total_reward,
            num_turns=len(turn_artifacts),
            termination_reason=episode_artifact.termination_reason,
            latest_eval=latest_eval,
            best_eval=best_eval,
            initial_eval=initial_eval,
        )
        self._maybe_dump_debug_trace(episode_artifact, traces, total_reward)
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
        *,
        chat_caller: AReaLEngineChatCaller | None = None,
        external_client: Any | None = None,
    ) -> AReaLEngineActorCaller | ExternalActorCaller:
        if chat_caller is not None:
            return AReaLEngineActorCaller(
                chat_caller=chat_caller,
                gconfig=self._generation_config(),
                max_completion_tokens=self.max_completion_tokens,
                max_train_sample_tokens=self.max_train_sample_tokens,
            )
        if external_client is None:
            raise ValueError("Exactly one teacher generation source must be provided.")
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
        *,
        chat_caller: AReaLEngineChatCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller:
        if self.aux_mode == "api":
            if self.api_aux_caller is None:
                raise RuntimeError("API student caller is not initialized.")
            return self.api_aux_caller
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

    async def _generate_teacher_response(
        self,
        teacher_state: TeacherTurnState,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        lora_version: int | None,
    ) -> tuple[ModelResponse, str]:
        result = await actor_caller.generate(
            self._build_teacher_messages(teacher_state),
            lora_version=lora_version,
            rid_prefix=f"codecoach-teacher-{teacher_state.turn_idx}",
        )
        return result.response, result.raw_text

    async def _run_student(
        self,
        state: StudentTurnState,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
    ) -> StudentRunResult:
        result = await aux_caller.call_text(
            self._build_student_messages(state),
            rid_prefix=f"codecoach-student-{state.turn_idx}",
        )
        if result.error:
            reply = StudentReply("", "", False, None, result.error)
            return StudentRunResult(raw_output="", reply=reply, error=result.error)
        raw_output = result.raw_text or result.text
        reply = self._parse_student_reply(result.text)
        return StudentRunResult(raw_output=raw_output, reply=reply, error=None)

    def _build_teacher_messages(
        self, teacher_state: TeacherTurnState
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": DEFAULT_TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_teacher_prompt(teacher_state)},
        ]

    def _build_teacher_prompt(self, state: TeacherTurnState) -> str:
        feedback = state.previous_feedback
        feedback_eval = feedback.eval_result or state.current_eval
        feedback_best = feedback.best_eval or state.best_eval
        return dedent(
            f"""\
            Task:
            {state.task_markdown}

            Public coaching history:
            {state.public_history.summary or "No previous coaching history."}

            Current round: {state.turn_idx}/{state.max_turns}
            Remaining rounds including this one: {max(state.max_turns - state.turn_idx + 1, 0)}
            Required entry function: {state.entry_function}

            Current evaluation:
            - Score: {state.current_eval.score:.6f}
            - Target ratio: {state.current_eval.target_ratio:.6f}
            - Validity: {state.current_eval.validity:.6f}
            - Error: {state.current_eval.error or "None"}

            Best evaluation so far:
            - Score: {state.best_eval.score:.6f}
            - Target ratio: {state.best_eval.target_ratio:.6f}
            - Validity: {state.best_eval.validity:.6f}

            Previous teacher guidance:
            {state.previous_teacher_visible_output or "(none yet)"}

            Private feedback from the previous student/evaluator step:
            - Kind: {feedback.kind}
            - Student message: {feedback.student_message or "(empty)"}
            - Code updated: {feedback.code_updated}
            - Error: {feedback.error or "None"}
            - Previous score: {feedback_eval.score:.6f}
            - Previous target ratio: {feedback_eval.target_ratio:.6f}
            - Best target ratio after previous step: {feedback_best.target_ratio:.6f}

            Current code:
            ```python
            {state.current_code}
            ```

            Reply with one focused natural-language coaching message. Do not output code.
            """
        ).strip()

    def _build_student_messages(
        self, state: StudentTurnState
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": DEFAULT_STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_student_prompt(state)},
        ]

    def _build_student_prompt(self, state: StudentTurnState) -> str:
        return dedent(
            f"""\
            You are the student programmer.
            Output a single JSON object with this schema:
            {{
              "message": "string",
              "replace_code": true,
              "code": "full python file or null"
            }}

            Rules:
            - Output valid JSON only.
            - If you do not want to change the code, set "replace_code" to false and "code" to null.
            - If you change the code, "code" must be a complete executable Python file.
            - Keep the required entry function `{state.entry_function}`.

            Task:
            {state.task_markdown}

            Public coaching history:
            {state.public_history.summary or "No previous coaching history."}

            Latest teacher guidance:
            {state.teacher_visible_output or "(empty)"}

            Current code:
            ```python
            {state.current_code}
            ```
            """
        ).strip()

    def _parse_student_reply(self, raw_output: str) -> StudentReply:
        parsed, parse_error = parse_json_dict(raw_output)
        if parsed is None:
            return StudentReply(raw_output, "", False, None, parse_error)
        message = parsed.get("message", "")
        replace_code = parsed.get("replace_code", False)
        code = parsed.get("code")
        if not isinstance(message, str):
            parse_error = join_errors(parse_error, '"message" must be a string')
            message = str(message)
        if not isinstance(replace_code, bool):
            parse_error = join_errors(parse_error, '"replace_code" must be a boolean')
            replace_code = False
        if code is not None and not isinstance(code, str):
            parse_error = join_errors(parse_error, '"code" must be a string or null')
            code = None
        return StudentReply(raw_output, message.strip(), replace_code, code, parse_error)

    def _apply_student_reply(
        self,
        current_code: str,
        reply: StudentReply,
        call_error: str | None,
    ) -> tuple[str, bool, str | None]:
        update_error = join_errors(call_error, reply.parse_error)
        if not reply.replace_code:
            return current_code, False, update_error
        if not reply.code:
            return (
                current_code,
                False,
                join_errors(update_error, "replace_code=true but code is empty"),
            )
        return reply.code, True, update_error

    def _run_public_summary_update(
        self,
        *,
        old_public_history: PublicHistoryState,
        teacher_visible_output: str,
        student_reply: StudentReply,
        code_updated: bool,
        eval_result: EvalResult,
        best_eval: EvalResult,
        error: str | None,
    ) -> PublicHistoryState:
        entries = []
        existing_history = old_public_history.summary.strip()
        if existing_history:
            entries.append(existing_history)
        turn_idx = old_public_history.turn_count + 1
        entries.append(
            dedent(
                f"""\
                Round {turn_idx}:
                Teacher guidance: {teacher_visible_output or "(empty)"}
                Student message: {student_reply.message or "(empty)"}
                Code updated: {code_updated}
                Evaluator target ratio: {eval_result.target_ratio:.6f}
                Evaluator score: {eval_result.score:.6f}
                Evaluator validity: {eval_result.validity:.6f}
                Best target ratio: {best_eval.target_ratio:.6f}
                Error: {error or "None"}
                """
            ).strip()
        )
        return PublicHistoryState(
            summary="\n\n".join(entry for entry in entries if entry),
            turn_count=turn_idx,
        )

    def _build_initial_public_summary(self, initial_eval: EvalResult) -> str:
        return dedent(
            f"""\
            Initial evaluation:
            - Score: {initial_eval.score:.6f}
            - Target ratio: {initial_eval.target_ratio:.6f}
            - Validity: {initial_eval.validity:.6f}
            - Error: {initial_eval.error or "None"}
            """
        ).strip()

    def _assign_rewards(
        self,
        turn_artifacts: list[TurnArtifact],
        pairwise_rewards: dict[int, float],
    ) -> list[RewardAssignment]:
        assignments = []
        for artifact in turn_artifacts:
            components = dict(artifact.base_reward_components)
            pairwise_reward = pairwise_rewards.get(artifact.turn_idx, 0.0)
            if pairwise_reward:
                components["pairwise"] = float(pairwise_reward)
            assignments.append(
                RewardAssignment(
                    reward=float(sum(components.values())),
                    reward_components=components,
                )
            )
        return assignments

    def _artifact_to_trace(
        self,
        artifact: TurnArtifact,
        assignment: RewardAssignment,
    ) -> TurnTrace:
        return TurnTrace(
            turn_idx=artifact.turn_idx,
            teacher_raw_output=artifact.teacher_raw_output,
            teacher_visible_output=artifact.teacher_visible_output,
            student_message=artifact.student_reply.message,
            code_updated=artifact.code_updated,
            eval_result=artifact.eval_result,
            best_eval=artifact.best_eval_after,
            error=artifact.error,
            reward=assignment.reward,
            reward_components=assignment.reward_components,
            public_history_before=artifact.public_history_before,
            public_history_after=artifact.public_history_after,
        )

    def _trace_to_history_record(self, trace: TurnTrace) -> dict[str, Any]:
        return {
            "round_idx": trace.turn_idx,
            "teacher_raw_output": trace.teacher_raw_output,
            "teacher_action": trace.teacher_visible_output,
            "student_message": trace.student_message,
            "code_updated": trace.code_updated,
            "score": trace.eval_result.score,
            "target_ratio": trace.eval_result.target_ratio,
            "validity": trace.eval_result.validity,
            "best_score": trace.best_eval.score,
            "best_target_ratio": trace.best_eval.target_ratio,
            "error": trace.error,
            "reward": trace.reward,
            "reward_components": dict(trace.reward_components),
            "public_history_before": trace.public_history_before,
            "public_history_after": trace.public_history_after,
        }

    def _current_auc_gain(
        self, best_ratio_history: list[float], initial_ratio: float
    ) -> float:
        area = sum(best_ratio - initial_ratio for best_ratio in best_ratio_history)
        return area / float(max(1, self.max_turns))

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
        episode: EpisodeArtifact,
        *,
        episode_lora_version: int | None,
        chat_caller: AReaLEngineChatCaller | None,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        evaluator_path: str,
        target_score: float,
        eval_timeout_sec: int,
        run_dir: Path,
    ) -> list[PairwiseTurnResult]:
        if episode_lora_version is None or chat_caller is None:
            return []
        reference_version = max(
            0,
            int(episode_lora_version) - self.pairwise_reference_lag_steps,
        )
        turns = list(episode.turns)
        if not self.pairwise_compare_all_turns:
            turns = turns[-1:] if turns else []
        return [
            await self._run_pairwise_turn(
                turn,
                reference_version=reference_version,
                chat_caller=chat_caller,
                aux_caller=aux_caller,
                evaluator_path=evaluator_path,
                target_score=target_score,
                eval_timeout_sec=eval_timeout_sec,
                run_dir=run_dir,
            )
            for turn in turns
        ]

    async def _run_pairwise_turn(
        self,
        current_turn: TurnArtifact,
        *,
        reference_version: int,
        chat_caller: AReaLEngineChatCaller,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        evaluator_path: str,
        target_score: float,
        eval_timeout_sec: int,
        run_dir: Path,
    ) -> PairwiseTurnResult:
        try:
            reference_raw_output = await self._generate_reference_teacher_response(
                current_turn.teacher_state,
                chat_caller=chat_caller,
                reference_version=reference_version,
            )
        except Exception as exc:
            setattr(exc, "_fatal_rollout_error", True)
            raise
        reference_visible_output = strip_reasoning_for_context(reference_raw_output)
        reference_student_state = StudentTurnState(
            task_markdown=current_turn.student_state.task_markdown,
            current_code=current_turn.student_state.current_code,
            teacher_visible_output=reference_visible_output,
            entry_function=current_turn.student_state.entry_function,
            public_history=current_turn.student_state.public_history,
            turn_idx=current_turn.student_state.turn_idx,
        )
        reference_student = await self._run_student(
            reference_student_state,
            aux_caller=aux_caller,
        )
        candidate_code, code_updated, update_error = self._apply_student_reply(
            reference_student_state.current_code,
            reference_student.reply,
            reference_student.error,
        )
        eval_result = self._evaluate_code(
            evaluator_path=evaluator_path,
            target_score=target_score,
            code=candidate_code,
            eval_root=run_dir
            / f"pairwise_round_{current_turn.turn_idx:02d}_v{reference_version}",
            eval_timeout_sec=eval_timeout_sec,
        )
        error = join_errors(update_error, eval_result.error)
        reference = ReferenceTurnArtifact(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            teacher_raw_output=reference_raw_output,
            teacher_visible_output=reference_visible_output,
            student_raw_output=reference_student.raw_output,
            student_reply=reference_student.reply,
            code_updated=code_updated,
            candidate_code=candidate_code,
            eval_result=eval_result,
            error=error,
        )
        comparison = self._compare_eval(current_turn.eval_result, eval_result)
        if comparison > 0:
            outcome: PairwiseOutcome = "current"
            reward = self.pairwise_reward_scale
            reason = "evaluator_current_better"
        elif comparison < 0:
            outcome = "reference"
            reward = -self.pairwise_reward_scale
            reason = "evaluator_reference_better"
        else:
            outcome = "tie"
            reward = 0.0
            reason = "evaluator_tie"
        return PairwiseTurnResult(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            outcome=outcome,
            reward=reward,
            reason=reason,
            reference=reference,
        )

    async def _generate_reference_teacher_response(
        self,
        teacher_state: TeacherTurnState,
        *,
        chat_caller: AReaLEngineChatCaller,
        reference_version: int,
    ) -> str:
        result = await chat_caller.generate(
            self._build_teacher_messages(teacher_state),
            gconfig=self._generation_config(),
            max_completion_tokens=self.max_completion_tokens,
            max_train_sample_tokens=self.max_train_sample_tokens,
            metadata={"lora_version": int(reference_version)},
            rid_prefix="codecoach-reference-teacher",
        )
        return result.raw_text

    def _compare_eval(self, left: EvalResult, right: EvalResult) -> int:
        left_key = (float(left.target_ratio), float(left.validity), float(left.score))
        right_key = (float(right.target_ratio), float(right.validity), float(right.score))
        for left_value, right_value in zip(left_key, right_key, strict=True):
            if abs(left_value - right_value) <= 1e-9:
                continue
            return 1 if left_value > right_value else -1
        return 0

    def _evaluate_code(
        self,
        evaluator_path: str,
        target_score: float,
        code: str,
        eval_root: Path,
        eval_timeout_sec: int,
    ) -> EvalResult:
        eval_root.mkdir(parents=True, exist_ok=True)
        code_path = eval_root / "candidate.py"
        result_path = eval_root / "result.json"
        stdout_path = eval_root / "stdout.log"
        stderr_path = eval_root / "stderr.log"
        code_path.write_text(code, encoding="utf-8")
        result: dict[str, Any] | None = None
        error: str | None = None
        try:
            process = subprocess.run(
                [sys.executable, evaluator_path, str(code_path), str(result_path)],
                capture_output=True,
                text=True,
                timeout=eval_timeout_sec,
                check=False,
            )
            stdout_path.write_text(process.stdout or "", encoding="utf-8")
            stderr_path.write_text(process.stderr or "", encoding="utf-8")
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            if process.returncode != 0 and result is None:
                error = f"Evaluator exited with code {process.returncode}"
        except subprocess.TimeoutExpired:
            error = f"Evaluator timed out after {eval_timeout_sec}s"
        except Exception as exc:
            error = str(exc)

        if result is None:
            result = {}
        raw_score = result.get(
            "sum_radii",
            result.get(
                "score",
                result.get("eval_score", result.get("combined_score", 0.0)),
            ),
        )
        score = float(raw_score or 0.0)
        target_ratio = result.get("target_ratio")
        if target_ratio is None:
            target_ratio = score / target_score if target_score > 0 else 0.0
        validity = float(result.get("validity", 1.0 if score > 0 else 0.0))
        success = bool(result.get("success", validity > 0 and score > 0))
        error = error or result.get("error")
        return EvalResult(
            score=score,
            target_ratio=float(target_ratio or 0.0),
            validity=validity,
            success=success,
            raw_result=result,
            error=error,
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

    def _log_rollout_stats(
        self,
        *,
        total_reward: float,
        num_turns: int,
        termination_reason: str,
        latest_eval: EvalResult,
        best_eval: EvalResult,
        initial_eval: EvalResult,
    ) -> None:
        _safe_scalar(
            reward=float(total_reward),
            num_turns=int(num_turns),
            samples_per_episode=int(num_turns),
            score=float(latest_eval.score),
            target_ratio=float(latest_eval.target_ratio),
            best_score=float(best_eval.score),
            best_target_ratio=float(best_eval.target_ratio),
            auc_gain=float(best_eval.target_ratio - initial_eval.target_ratio),
            termination_success=float(termination_reason == "success"),
            termination_max_turns=float(termination_reason == "max_turns"),
            termination_length=float(termination_reason == "length"),
            termination_context_budget_limit=float(
                termination_reason == CONTEXT_BUDGET_TERMINATION_REASON
            ),
        )

    def _maybe_dump_debug_trace(
        self,
        episode: EpisodeArtifact,
        traces: list[TurnTrace],
        total_reward: float,
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
                "total_reward": total_reward,
                "termination_reason": episode.termination_reason,
                "initial_eval": episode.initial_eval,
                "latest_eval": episode.latest_eval,
                "best_eval": episode.best_eval,
                "traces": traces,
            }
            file_path.write_text(
                json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to dump CodeCoach debug trace: %s", exc)
