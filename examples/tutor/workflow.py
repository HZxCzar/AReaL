from __future__ import annotations

import asyncio
import hashlib
import json
import logging as py_logging
import operator
import os
import random
import re
import socket
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
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
from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    TutorStudentModelConfig,
)
from examples.tutor.core.callers import (
    ApiAuxiliaryCaller,
    AReaLEngineActorCaller,
    AReaLEngineAuxiliaryCaller,
    AReaLEngineChatCaller,
    ExternalActorCaller,
    TextCallResult,
    apply_chat_template,
)
from examples.tutor.core.confidence import compute_answer_token_confidence
from examples.tutor.core.generalization import load_student_generalize_bank
from examples.tutor.core.generation_budget import (
    CONTEXT_BUDGET_TERMINATION_REASON,
    ContextBudgetLimitExceeded,
)
from examples.tutor.core.history import (
    trace_to_history_record,
    trace_to_json,
)
from examples.tutor.core.parsers import (
    parse_leak_check_result,
    parse_staged_leak_check_result,
    parse_tagged_teacher_output,
)
from examples.tutor.core.rewards import EpisodeRewardComputer, artifact_to_trace
from examples.tutor.core.scoring import AnswerScorer, get_answer_scorer
from examples.tutor.core.semantic_similarity import (
    cosine_similarity,
    get_local_embedding_caller,
)
from examples.tutor.core.tensors import (
    response_to_tensordict,
    tokenize_teacher_forced_response,
)
from examples.tutor.core.text import (
    strip_reasoning_for_context as _strip_reasoning_for_context,
)
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    LeakHandlingMode,
    PromptPoolSelection,
    PublicHistoryState,
    RewardAssignment,
    StudentGeneralizeMode,
    StudentQuestionGenerationResult,
    StudentRequestJudgeResult,
    StudentTurnBehavior,
    StudentTurnState,
    TeacherGuidance,
    TeacherPreSolveAttempt,
    TeacherPreSolveResult,
    TeacherProgressJudgeResult,
    TurnArtifact,
    TurnTrace,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import (
    ANSWER_JUDGE_USER_TEMPLATE,
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    DEFAULT_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STAGED_LEAK_CHECK_SYSTEM_PROMPT,
    DEFAULT_STUDENT_REQUEST_JUDGE_SYSTEM_PROMPT,
    DEFAULT_TEACHER_PROGRESS_JUDGE_SYSTEM_PROMPT,
    DEFAULT_WORLD_MODEL_SYSTEM_PROMPT,
    EMPTY_PLACEHOLDER,
    FILTER_SOLVER_SYSTEM_PROMPT,
    FILTER_SOLVER_USER_TEMPLATE,
    INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
    LEAK_CHECK_DISABLED_FEEDBACK,
    LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE,
    LEAK_CHECK_PENDING_FEEDBACK,
    LEAK_CHECK_USER_TEMPLATE,
    NO_PREVIOUS_VISIBLE_TUTORING_HISTORY,
    NO_VISIBLE_TUTORING_HISTORY,
    NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT,
    NONE_PLACEHOLDER,
    NONE_YET_PLACEHOLDER,
    POLARIS_FILTER_SOLVER_USER_TEMPLATE,
    POLARIS_INSTRUCTION,
    PUBLIC_HISTORY_ENTRY_TEMPLATE,
    RAWBASE_LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    STAGED_LEAK_CHECK_USER_TEMPLATE,
    STUDENT_QUESTION_SYSTEM_PROMPT,
    STUDENT_QUESTION_USER_TEMPLATE,
    STUDENT_REQUEST_JUDGE_USER_TEMPLATE,
    INITIAL_ATTEMPT_WRAPPER,
    STUDENT_FINAL_SOLUTION_TEMPLATE,
    STUDENT_STATE_USER_TEMPLATE,
    STUDENT_TRANSFER_TURN_TEMPLATE,
    STUDENT_TRANSFER_USER_TEMPLATE,
    TASK_CONTEXT_TEMPLATE,
    TEACHER_ENV_FEEDBACK_TEMPLATE,
    TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE,
    TEACHER_ADAPTIVE_INSTRUCTION,
    TEACHER_ANTI_LEAK_INSTRUCTION,
    TEACHER_GUIDANCE_TAIL_TEMPLATE,
    TEACHER_MOVE_INSTRUCTIONS,
    TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE,
    TEACHER_REPAIR_INSTRUCTION,
    TEACHER_PROGRESS_JUDGE_USER_TEMPLATE,
    TEACHER_STATE_USER_TEMPLATE,
    WORLD_MODEL_USER_TEMPLATE,
    render_prompt,
)

logger = logging.getLogger("TutorWorkflow")

_POLARIS_SCORE_PROCESS: ContextVar[Any | None] = ContextVar(
    "tutor_polaris_score_process", default=None
)

_REWARD_COMPONENT_ALIASES = {
    "success_credit": "success",
}

LEAK_TERMINATION_REASON = "leak"
TEACHER_PRE_SKIPPED_TERMINATION_REASON = "pre_solve_skipped"
LEAK_HANDLING_MODES = {"disabled", "reward_only", "terminate"}


def load_prompt_pool(path: str, *, role: str) -> tuple[str, ...]:
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return ()

    file_path = Path(normalized_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{role} prompt pool file not found: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{role} prompt pool must be valid JSON: {file_path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"{role} prompt pool must be a non-empty JSON string array: {file_path}"
        )

    prompts: list[str] = []
    for index, raw_prompt in enumerate(payload):
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError(
                f"{role} prompt pool entry {index} must be a non-empty string: "
                f"{file_path}"
            )
        prompts.append(raw_prompt.strip())
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"{role} prompt pool entries must be unique: {file_path}")
    return tuple(prompts)


def load_student_turn_behaviors(path: str) -> tuple[StudentTurnBehavior, ...]:
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return ()

    file_path = Path(normalized_path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"student turn behavior pool file not found: {file_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"student turn behavior pool must be valid JSON: {file_path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise ValueError(
            f"student turn behavior pool must be a non-empty JSON array: {file_path}"
        )

    behaviors: list[StudentTurnBehavior] = []
    names: set[str] = set()
    base_entries = 0
    for index, raw_behavior in enumerate(payload):
        if not isinstance(raw_behavior, dict):
            raise ValueError(
                f"student turn behavior entry {index} must be an object: {file_path}"
            )
        name = str(raw_behavior.get("name") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(
                f"student turn behavior entry {index} has invalid name {name!r}: "
                f"{file_path}"
            )
        if name in names:
            raise ValueError(
                f"student turn behavior names must be unique: {name!r}: {file_path}"
            )
        names.add(name)

        raw_instruction = raw_behavior.get("instruction")
        if not isinstance(raw_instruction, str):
            raise ValueError(
                f"student turn behavior entry {index} instruction must be a string: "
                f"{file_path}"
            )
        instruction = raw_instruction.strip()
        base_entries += int(not instruction)

        raw_probability = raw_behavior.get("probability")
        if isinstance(raw_probability, bool) or not isinstance(
            raw_probability, (int, float)
        ):
            raise ValueError(
                f"student turn behavior entry {index} probability must be numeric: "
                f"{file_path}"
            )
        probability = float(raw_probability)
        if probability <= 0.0 or probability > 1.0:
            raise ValueError(
                f"student turn behavior entry {index} probability must be in "
                f"(0, 1]: {file_path}"
            )
        behaviors.append(
            StudentTurnBehavior(
                index=index,
                name=name,
                instruction=instruction,
                probability=probability,
            )
        )

    if base_entries != 1:
        raise ValueError(
            "student turn behavior pool must contain exactly one clean base entry "
            f"with an empty instruction: {file_path}"
        )
    probability_sum = sum(behavior.probability for behavior in behaviors)
    if abs(probability_sum - 1.0) > 1e-9:
        raise ValueError(
            "student turn behavior probabilities must sum to 1.0; "
            f"got {probability_sum:.12g}: {file_path}"
        )
    return tuple(behaviors)


def load_prompt_text(path: str, *, role: str) -> str:
    normalized_path = str(path or "").strip()
    if not normalized_path:
        return ""
    file_path = Path(normalized_path)
    try:
        prompt = file_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"{role} prompt file not found: {file_path}") from exc
    if not prompt:
        raise ValueError(f"{role} prompt file must be non-empty: {file_path}")
    return prompt


def _safe_scalar(**metrics: Any) -> None:
    try:
        stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
    except Exception:
        logger.debug("Skipping stats logging outside workflow context.")


def _safe_generalize_scalar(**metrics: Any) -> None:
    try:
        ctx = workflow_context.get()
        split = "test" if bool(getattr(ctx, "is_eval", False)) else "train"
        scoped_metrics = {f"{split}/{key}": value for key, value in metrics.items()}
        stats_tracker.get("generalize").scalar(**scoped_metrics)
    except Exception:
        logger.debug("Skipping generalize stats logging outside workflow context.")


def _binary_repeat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Binary repeat outcomes must not be empty.")
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError(f"Binary repeat outcomes must be 0 or 1, got {values!r}.")

    mean = sum(values) / len(values)
    variance = mean * (1.0 - mean)
    repeat_pairs = list(combinations(values, 2))
    agreement = (
        sum(float(left == right) for left, right in repeat_pairs) / len(repeat_pairs)
        if repeat_pairs
        else 1.0
    )
    return {
        "mean": mean,
        "variance": variance,
        "std": variance**0.5,
        "agreement": agreement,
        "disagreement": 1.0 - agreement,
        "all_equal": float(all(value == values[0] for value in values)),
        "any_success": float(any(values)),
        "all_success": float(all(values)),
        "success_set_intersection": float(all(values)),
        "success_set_union": float(any(values)),
    }


def _reward_component_key(name: str) -> str:
    return _REWARD_COMPONENT_ALIASES.get(name, name)


_STUDENT_GENERALIZE_LEVELS = ("level1", "level2")
# Re-test the ORIGINAL task as a full standalone solution. Off by default;
# an independent branch of the same chat, exactly like the transfer probes.
ORIGINAL_RETEST_LEVEL = "original"
_STUDENT_GENERALIZE_MODES = {"only_success", "always"}


@dataclass(slots=True)
class StudentGeneralizationCase:
    level: str
    task: str
    ground_truth: str
    reward: float


@dataclass(slots=True)
class StudentGeneralizationResult:
    level: str
    task: str = ""
    ground_truth: str = ""
    attempted: bool = False
    skipped: bool = False
    skip_reason: str = ""
    student_output: str = ""
    student_error: str | None = None
    judge_result: JudgeResult | None = None
    correctness_reward: float = 0.0
    replay_count: int = 0
    replay_correct: int = 0
    confidence: float = 0.0
    confidence_mean_logprob: float | None = None
    confidence_token_count: int = 0
    confidence_available: bool = False
    confidence_reason: str = ""
    confidence_backend: str = ""
    confidence_reward: float = 0.0
    reward: float = 0.0
    public_history: str = ""
    reward_turn_idx: int | None = None


@dataclass(slots=True)
class StudentGeneralizationAnchor:
    public_history: PublicHistoryState
    previous_student_output: str
    teacher_feedback: str
    reward_turn_idx: int | None


@dataclass(slots=True)
class WorldModelExample:
    input_ids: list[int] = field(default_factory=list)
    target_mask: list[int] = field(default_factory=list)
    skip_reason: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.input_ids) and bool(self.target_mask) and not self.skip_reason


@dataclass(slots=True)
class StudentModelRuntime:
    name: str
    model: str
    weight: float
    caller: ApiAuxiliaryCaller
    confidence_caller: ApiAuxiliaryCaller | None = None


@dataclass(slots=True)
class SelectedStudent:
    name: str
    model: str
    caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller
    confidence_caller: ApiAuxiliaryCaller | None = None


class TutorAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: Any | None = None,
        tokenizer: str | Any | None = None,
        dataset_type: str = "aime",
        answer_scorer: str = "auto",
        max_turns: int = 6,
        enable_thinking: bool = False,
        leak_handling_mode: LeakHandlingMode = "reward_only",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 512,
        tool_call_parser: str = "qwen25",
        reasoning_parser: str = "qwen3",
        aux_mode: str = "api",
        aux_enable_thinking: bool = False,
        aux_base_url: str = "https://choab9kmmqm8cbcbmqjbeg5jdej8ahaj.openapi-qb-ai.sii.edu.cn/v1",
        aux_model: str = "qwen3-4b",
        aux_api_key: str = "${oc.env:INF_API_KEY}",
        aux_timeout: int = 120,
        aux_max_tokens: int = 1024,
        aux_temperature: float = 0.7,
        aux_top_p: float | None = None,
        max_concurrent_aux_calls: int = 8,
        aux_request_params: dict[str, Any] | None = None,
        student_models: list[dict[str, Any]] | None = None,
        success_reward: float = 1.0,
        leak_penalty: float | None = -1.0,
        leak_penalty_mode: str = "binary",
        leak_penalty_final_answer: float | None = None,
        leak_penalty_compute: float | None = None,
        leak_penalty_formula: float | None = None,
        leak_penalty_aggregation: str = "turn",
        format_error_penalty: float = 0.0,
        leaked_success_reward_scale: float = 1.0,
        assign_success_reward: bool = False,
        outcome_prior_turn_weight: float = 0.1,
        outcome_credit_gamma: float = 0.9,
        early_success_bonus: float = 0.3,
        success_turn_shaping: dict[str, Any] | None = None,
        max_turn_penalty: float = 0.0,
        enable_turn_penalty: bool = False,
        turn_penalty: float = -0.01,
        length_penalty_threshold_chars: int = 1200,
        length_penalty_per_100_chars: float = -0.005,
        length_penalty_min: float = -0.1,
        zero_reward_on_length_stop: bool = False,
        teacher_diversity_reward: dict[str, Any] | None = None,
        teacher_context_reward: dict[str, Any] | None = None,
        teacher_progress_judge: dict[str, Any] | None = None,
        student_request_judge: dict[str, Any] | None = None,
        world_model: dict[str, Any] | None = None,
        guided_slots: dict[str, Any] | None = None,
        opd: dict[str, Any] | None = None,
        prompt_instruction: dict[str, Any] | None = None,
        local_advantage_turn_discount: float = 1.0,
        teacher_system_prompt: str = "",
        teacher_anti_leak_instruction_enabled: bool = False,
        teacher_adaptive_instruction_enabled: bool = False,
        teacher_prompt_pool_path: str = "",
        teacher_warmup_enabled: bool = False,
        teacher_warmup_prompt_path: str = "",
        teacher_warmup_steps: int = 0,
        teacher_user_prompt_template: str | None = None,
        teacher_show_ground_truth: bool = False,
        teacher_pre_enabled: bool = False,
        teacher_pre_mode: str = "filter_solver",
        teacher_pre_verify: bool = True,
        teacher_pre_attempts: int = 3,
        teacher_pre_max_tokens: int = 0,
        student_system_prompt: str = "",
        student_prompt_pool_path: str = "",
        student_heldout_prompt_pool_path: str = "",
        student_prompt_include_base: bool = False,
        student_turn_behavior_enabled: bool = False,
        student_turn_behavior_path: str = "",
        student_turn_behavior_separate_call_behavior_names: (
            list[str] | None
        ) = None,
        prompt_pool_seed: int = 0,
        leak_check_system_prompt: str = "",
        answer_judge_enabled: bool = False,
        answer_judge_max_tokens: int = 256,
        answer_judge_system_prompt: str = DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
        debug_trace_dir: str | None = None,
        debug_trace_every_n_rollouts: int = 1,
        max_train_sample_tokens: int | None = None,
        tokenizer_path: str | None = None,
        model_context_length: int | None = None,
        context_window_margin: int = 256,
        student_generalize_enabled: bool = False,
        student_generalize_mode: StudentGeneralizeMode | str = "only_success",
        student_generalize_source: str = "sidecar",
        student_generalize_path: str = "",
        student_generalize_replays: int = 1,
        student_generalize_retest_original: bool = False,
        student_generalize_level1_reward: float = 0.2,
        student_generalize_level2_reward: float = 0.5,
        student_generalize_confidence_enabled: bool = False,
        student_generalize_confidence_reward_scale: float = 0.25,
        eval_repeat_count: int = 1,
    ):
        self.eval_repeat_count = int(eval_repeat_count)
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be >= 1.")
        self._eval_repeat_outcomes: dict[int, list[float]] = {}
        self.max_turns = max_turns
        self.dataset_type = (dataset_type or "").strip().lower()
        if self.dataset_type not in {"aime", "math", "polaris"}:
            raise ValueError("dataset_type must be one of: 'aime', 'math', 'polaris'.")
        answer_scorer_name = (answer_scorer or "auto").strip().lower()
        if answer_scorer_name == "auto":
            answer_scorer_name = self.dataset_type
        elif answer_scorer_name != self.dataset_type:
            raise ValueError(
                "answer_scorer must be 'auto' or match dataset_type; "
                f"got dataset_type={self.dataset_type!r}, "
                f"answer_scorer={answer_scorer_name!r}."
            )
        self.answer_scorer_name = answer_scorer_name
        self.answer_scorer: AnswerScorer = get_answer_scorer(answer_scorer_name)
        self.enable_thinking = enable_thinking
        if leak_handling_mode not in LEAK_HANDLING_MODES:
            raise ValueError(
                "leak_handling_mode must be one of: 'disabled', "
                "'reward_only', or 'terminate'."
            )
        self.leak_handling_mode: LeakHandlingMode = leak_handling_mode
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
        self._student_model_configs = self._normalize_student_model_configs(
            student_models
        )
        self._self_aux_semaphore = asyncio.Semaphore(
            max(1, self.max_concurrent_aux_calls)
        )
        self.context_window_margin = int(context_window_margin)
        if leak_penalty_mode not in {"binary", "staged", "rawbase"}:
            raise ValueError(
                "leak_penalty_mode must be 'binary', 'staged', or 'rawbase'."
            )
        if leak_penalty_mode in {"binary", "rawbase"} and leak_penalty is None:
            raise ValueError(
                "leak_penalty must be set in binary/rawbase leak penalty mode."
            )
        staged_values = {
            "leak_penalty_final_answer": leak_penalty_final_answer,
            "leak_penalty_compute": leak_penalty_compute,
            "leak_penalty_formula": leak_penalty_formula,
        }
        if leak_penalty_mode == "staged":
            missing = [name for name, value in staged_values.items() if value is None]
            if missing:
                raise ValueError(
                    "staged leak penalty mode requires explicit values for "
                    f"{', '.join(missing)}."
                )
        if leak_penalty_aggregation not in {"turn", "episode"}:
            raise ValueError("leak_penalty_aggregation must be 'turn' or 'episode'.")
        if format_error_penalty > 0.0:
            raise ValueError("format_error_penalty must be <= 0.")
        if leaked_success_reward_scale < 0.0:
            raise ValueError("leaked_success_reward_scale must be >= 0.")

        self.success_reward = float(success_reward)
        self.leak_penalty_mode = leak_penalty_mode
        self.leak_penalty = float(leak_penalty) if leak_penalty is not None else 0.0
        self.leak_penalty_final_answer = (
            float(leak_penalty_final_answer)
            if leak_penalty_final_answer is not None
            else 0.0
        )
        self.leak_penalty_compute = (
            float(leak_penalty_compute) if leak_penalty_compute is not None else 0.0
        )
        self.leak_penalty_formula = (
            float(leak_penalty_formula) if leak_penalty_formula is not None else 0.0
        )
        self.leak_penalty_aggregation = leak_penalty_aggregation
        self.format_error_penalty = float(format_error_penalty)
        self.leaked_success_reward_scale = float(leaked_success_reward_scale)
        self.assign_success_reward = bool(assign_success_reward)
        self.outcome_prior_turn_weight = float(outcome_prior_turn_weight)
        self.outcome_credit_gamma = float(outcome_credit_gamma)
        self.early_success_bonus = float(early_success_bonus)
        success_turn_config = dict(success_turn_shaping or {})
        self.success_turn_shaping_enabled = bool(
            success_turn_config.get("enabled", False)
        )
        self.success_turn_shaping_min_reward = float(
            success_turn_config.get("min_reward", 1.0)
        )
        self.success_turn_shaping_max_reward = float(
            success_turn_config.get("max_reward", 1.0)
        )
        self.max_turn_penalty = float(max_turn_penalty)
        self.enable_turn_penalty = bool(enable_turn_penalty)
        self.turn_penalty = float(turn_penalty)
        self.length_penalty_threshold_chars = int(length_penalty_threshold_chars)
        self.length_penalty_per_100_chars = float(length_penalty_per_100_chars)
        self.length_penalty_min = float(length_penalty_min)
        self.zero_reward_on_length_stop = bool(zero_reward_on_length_stop)
        diversity_config = dict(teacher_diversity_reward or {})
        self.teacher_diversity_enabled = bool(diversity_config.get("enabled", False))
        self.teacher_diversity_weight = float(diversity_config.get("weight", 0.1))
        diversity_model_path = str(
            diversity_config.get("embedding_model_path", "") or ""
        ).strip()
        diversity_device = str(
            diversity_config.get("embedding_device", "cuda") or ""
        ).strip()
        diversity_dtype = str(
            diversity_config.get("embedding_dtype", "bfloat16") or ""
        ).strip()
        diversity_max_length = int(diversity_config.get("embedding_max_length", 8192))
        diversity_batch_wait_ms = float(
            diversity_config.get("embedding_batch_wait_ms", 2.0)
        )
        diversity_max_batch_texts = int(
            diversity_config.get("embedding_max_batch_texts", 64)
        )
        diversity_max_batch_tokens = int(
            diversity_config.get("embedding_max_batch_tokens", 32768)
        )
        if self.teacher_diversity_enabled and self.teacher_diversity_weight <= 0.0:
            raise ValueError("teacher diversity reward weight must be positive.")
        if self.teacher_diversity_enabled and not diversity_model_path:
            raise ValueError(
                "teacher diversity embedding_model_path is required when enabled."
            )
        if self.teacher_diversity_enabled and not diversity_device:
            raise ValueError(
                "teacher diversity embedding_device is required when enabled."
            )
        if diversity_max_length <= 0:
            raise ValueError("teacher diversity embedding_max_length must be positive.")
        self.teacher_diversity_caller = (
            get_local_embedding_caller(
                model_path=diversity_model_path,
                device=diversity_device,
                dtype=diversity_dtype,
                max_length=diversity_max_length,
                batch_wait_ms=diversity_batch_wait_ms,
                max_batch_texts=diversity_max_batch_texts,
                max_batch_tokens=diversity_max_batch_tokens,
            )
            if self.teacher_diversity_enabled
            else None
        )
        context_config = dict(teacher_context_reward or {})
        self.teacher_context_enabled = bool(context_config.get("enabled", False))
        self.teacher_context_apply_to_advantage = bool(
            context_config.get("apply_to_advantage", True)
        )
        self.teacher_context_weight = float(context_config.get("weight", 0.1))
        self.teacher_context_score_clip = float(context_config.get("score_clip", 5.0))
        if self.teacher_context_enabled and self.teacher_context_weight <= 0.0:
            raise ValueError("teacher context reward weight must be positive.")
        if self.teacher_context_enabled and self.teacher_context_score_clip <= 0.0:
            raise ValueError("teacher context reward score_clip must be positive.")
        world_model_config = dict(world_model or {})
        self.world_model_enabled = bool(world_model_config.get("enabled", False))
        self.world_model_loss_weight = float(
            world_model_config.get("loss_weight", 0.05)
        )
        self.world_model_system_prompt = str(
            world_model_config.get("system_prompt", DEFAULT_WORLD_MODEL_SYSTEM_PROMPT)
            or ""
        ).strip()
        paw_config = dict(world_model_config.get("paw") or {})
        self.world_model_paw_config = {
            "enabled": bool(paw_config.get("enabled", False)),
            "entropy_filter_enabled": bool(
                paw_config.get("entropy_filter_enabled", True)
            ),
            "entropy_keep_ratio": float(paw_config.get("entropy_keep_ratio", 0.75)),
            "cmae_enabled": bool(paw_config.get("cmae_enabled", True)),
            "confidence_threshold": float(paw_config.get("confidence_threshold", 0.2)),
            "reward_adaptive_enabled": bool(
                paw_config.get("reward_adaptive_enabled", False)
            ),
            "max_episode_return": float(paw_config.get("max_episode_return", 1.0)),
        }
        if self.world_model_enabled and self.world_model_loss_weight <= 0.0:
            raise ValueError("world model loss weight must be positive.")
        if self.world_model_enabled and not self.world_model_system_prompt:
            raise ValueError("world model system prompt is required.")
        if self.world_model_paw_config["enabled"] and not self.world_model_enabled:
            raise ValueError("world model must be enabled when PaW is enabled.")
        if not 0.0 < self.world_model_paw_config["entropy_keep_ratio"] <= 1.0:
            raise ValueError("PaW entropy keep ratio must be in (0, 1].")
        if not 0.0 < self.world_model_paw_config["confidence_threshold"] < 1.0:
            raise ValueError("PaW confidence threshold must be in (0, 1).")
        if self.world_model_paw_config["max_episode_return"] <= 0.0:
            raise ValueError("PaW max episode return must be positive.")
        guided_config = dict(guided_slots or {})
        self.guided_slots_enabled = bool(guided_config.get("enabled", False))
        self.guided_slots_count = int(guided_config.get("slots", 3))
        guided_moves = tuple(
            str(move).strip().upper()
            for move in (guided_config.get("moves") or ())
            if str(move).strip()
        )
        self.guided_slots_moves = guided_moves or ("DECOMPOSE", "REFRAME", "PROBE")
        self.guided_slots_turns = frozenset(
            int(turn) for turn in (guided_config.get("turns") or (1,))
        )
        self.guided_slots_rotate_by_task = bool(
            guided_config.get("rotate_by_task", True)
        )
        unknown_moves = [
            move
            for move in self.guided_slots_moves
            if move not in TEACHER_MOVE_INSTRUCTIONS
        ]
        if self.guided_slots_enabled and unknown_moves:
            raise ValueError(f"unknown guided slot moves: {unknown_moves}")
        if self.guided_slots_enabled and self.guided_slots_count < 1:
            raise ValueError("guided slot count must be at least 1 when enabled.")
        opd_config = dict(opd or {})
        self.opd_enabled = bool(opd_config.get("enabled", False))
        self.opd_loss_weight = float(opd_config.get("loss_weight", 1.0))
        self.opd_reward_clip = float(opd_config.get("reward_clip", 0.0))
        self.opd_instruction = (
            str(opd_config.get("instruction") or "").strip()
            or TEACHER_REPAIR_INSTRUCTION
        )
        self.opd_min_prior_failed_turns = int(
            opd_config.get("min_prior_failed_turns", 2)
        )
        self.opd_max_turns_per_episode = int(opd_config.get("max_turns_per_episode", 0))
        self.opd_skip_guided_rows = bool(opd_config.get("skip_guided_rows", True))
        self.opd_skip_leaked_rows = bool(opd_config.get("skip_leaked_rows", True))
        if self.opd_enabled and self.opd_loss_weight <= 0.0:
            raise ValueError("opd loss weight must be positive when enabled.")
        if self.opd_enabled and self.opd_reward_clip < 0.0:
            raise ValueError("opd reward clip must be non-negative (0 disables).")
        prompt_instr = dict(prompt_instruction or {})
        self.prompt_instruction_enabled = bool(prompt_instr.get("enabled", False))
        self.prompt_instruction_text = (
            str(prompt_instr.get("instruction") or "").strip()
            or TEACHER_REPAIR_INSTRUCTION
        )
        self.prompt_instruction_min_prior_failed_turns = int(
            prompt_instr.get("min_prior_failed_turns", 2)
        )
        if self.prompt_instruction_enabled and self.opd_enabled:
            raise ValueError(
                "prompt_instruction and opd are the two arms of one comparison; "
                "enabling both makes neither attributable."
            )
        progress_config = dict(teacher_progress_judge or {})
        self.teacher_progress_judge_enabled = bool(
            progress_config.get("enabled", False)
        )
        self.teacher_progress_judge_weight = float(progress_config.get("weight", 0.5))
        self.teacher_progress_judge_system_prompt = (
            DEFAULT_TEACHER_PROGRESS_JUDGE_SYSTEM_PROMPT
        )
        self.local_advantage_turn_discount = float(local_advantage_turn_discount)
        if (
            self.teacher_progress_judge_enabled
            and self.teacher_progress_judge_weight <= 0.0
        ):
            raise ValueError("teacher progress judge weight must be positive.")
        request_config = dict(student_request_judge or {})
        self.student_request_judge_enabled = bool(request_config.get("enabled", False))
        self.student_request_judge_weight = float(request_config.get("weight", 0.5))
        request_behavior_names = request_config.get("behavior_names", ["ask_question"])
        self.student_request_judge_behavior_names = frozenset(
            str(name).strip() for name in request_behavior_names if str(name).strip()
        )
        self.student_request_judge_system_prompt = (
            DEFAULT_STUDENT_REQUEST_JUDGE_SYSTEM_PROMPT
        )
        if (
            self.student_request_judge_enabled
            and self.student_request_judge_weight <= 0.0
        ):
            raise ValueError("student request judge weight must be positive.")
        if (
            self.student_request_judge_enabled
            and not self.student_request_judge_behavior_names
        ):
            raise ValueError("student request judge behavior_names must not be empty.")
        if self.local_advantage_turn_discount < 0.0:
            raise ValueError("local advantage turn discount must be non-negative.")
        self.teacher_anti_leak_instruction_enabled = bool(
            teacher_anti_leak_instruction_enabled
        )
        self.teacher_adaptive_instruction_enabled = bool(
            teacher_adaptive_instruction_enabled
        )
        self.teacher_system_prompt = self._resolve_teacher_system_prompt(
            teacher_system_prompt
        )
        self.teacher_prompt_pool = load_prompt_pool(
            teacher_prompt_pool_path, role="teacher"
        )
        self.teacher_warmup_enabled = bool(teacher_warmup_enabled)
        self.teacher_warmup_prompt_path = str(teacher_warmup_prompt_path or "").strip()
        self.teacher_warmup_steps = int(teacher_warmup_steps)
        if self.teacher_warmup_enabled and not self.teacher_warmup_prompt_path:
            raise ValueError(
                "teacher_warmup_prompt_path is required when teacher warm-up is "
                "enabled."
            )
        if self.teacher_warmup_enabled and self.teacher_warmup_steps <= 0:
            raise ValueError(
                "teacher_warmup_steps must be positive when teacher warm-up is enabled."
            )
        raw_teacher_warmup_prompt = (
            load_prompt_text(self.teacher_warmup_prompt_path, role="teacher warm-up")
            if self.teacher_warmup_enabled
            else ""
        )
        self.teacher_warmup_prompt = (
            self._resolve_teacher_system_prompt(raw_teacher_warmup_prompt)
            if raw_teacher_warmup_prompt
            else ""
        )
        self.teacher_user_prompt_template = (
            teacher_user_prompt_template or TEACHER_STATE_USER_TEMPLATE
        ).strip()
        self.teacher_show_ground_truth = bool(teacher_show_ground_truth)
        self.teacher_pre_enabled = bool(teacher_pre_enabled)
        self.teacher_pre_mode = (teacher_pre_mode or "filter_solver").strip()
        if self.teacher_pre_mode != "filter_solver":
            raise ValueError("teacher_pre_mode must be 'filter_solver'.")
        self.teacher_pre_verify = bool(teacher_pre_verify)
        self.teacher_pre_attempts = int(teacher_pre_attempts)
        if self.teacher_pre_attempts < 1:
            raise ValueError("teacher_pre_attempts must be >= 1.")
        self.teacher_pre_max_tokens = int(teacher_pre_max_tokens)
        self.student_system_prompt = student_system_prompt.strip()
        self.student_prompt_pool = load_prompt_pool(
            student_prompt_pool_path, role="student"
        )
        self.student_heldout_prompt_pool = load_prompt_pool(
            student_heldout_prompt_pool_path, role="held-out student"
        )
        self.student_prompt_include_base = bool(student_prompt_include_base)
        self.student_turn_behavior_enabled = bool(student_turn_behavior_enabled)
        if (
            self.student_turn_behavior_enabled
            and not str(student_turn_behavior_path or "").strip()
        ):
            raise ValueError(
                "student_turn_behavior_path is required when turn behaviors are "
                "enabled."
            )
        self.student_turn_behaviors = (
            load_student_turn_behaviors(student_turn_behavior_path)
            if self.student_turn_behavior_enabled
            else ()
        )
        self.student_turn_behavior_separate_call_behavior_names = frozenset(
            str(name).strip()
            for name in (student_turn_behavior_separate_call_behavior_names or [])
            if str(name).strip()
        )
        if (
            self.student_turn_behavior_separate_call_behavior_names
            and not self.student_turn_behavior_enabled
        ):
            raise ValueError(
                "separate-call student turn behaviors require student turn "
                "behaviors to be enabled."
            )
        configured_behavior_names = {
            behavior.name for behavior in self.student_turn_behaviors
        }
        unknown_separate_call_names = sorted(
            self.student_turn_behavior_separate_call_behavior_names
            - configured_behavior_names
        )
        if unknown_separate_call_names:
            raise ValueError(
                "separate-call behavior names are missing from the student turn "
                f"behavior pool: {unknown_separate_call_names}."
            )
        if self.student_request_judge_enabled:
            if not self.student_turn_behavior_enabled:
                raise ValueError(
                    "student request judge requires student turn behaviors to be "
                    "enabled."
                )
            unknown_behavior_names = sorted(
                self.student_request_judge_behavior_names - configured_behavior_names
            )
            if unknown_behavior_names:
                raise ValueError(
                    "student request judge behavior_names are missing from the "
                    f"student turn behavior pool: {unknown_behavior_names}."
                )
        self.prompt_pool_seed = int(prompt_pool_seed)
        self._teacher_prompt_pool_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:teacher:fallback"
        )
        self._teacher_warmup_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:teacher-warmup-gate:fallback"
        )
        self._student_prompt_pool_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:student:fallback"
        )
        self._student_turn_behavior_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:student-turn-behavior:fallback"
        )
        self.leak_check_system_prompt = self._resolve_leak_check_system_prompt(
            leak_check_system_prompt
        )
        self.answer_judge_enabled = bool(answer_judge_enabled)
        if self.dataset_type == "polaris":
            if self.leak_handling_mode != "disabled":
                raise ValueError(
                    "dataset_type='polaris' is incompatible with leak checks; "
                    "set leak_handling_mode='disabled'."
                )
            if self.answer_judge_enabled:
                raise ValueError(
                    "dataset_type='polaris' uses the Polaris rule judge and is "
                    "incompatible with answer_judge_enabled=true."
                )
        self.answer_judge_max_tokens = max(1, int(answer_judge_max_tokens))
        self.answer_judge_system_prompt = answer_judge_system_prompt.strip()
        self._answer_judge_cache: dict[tuple[str, str, str], JudgeResult] = {}
        self.debug_trace_dir = debug_trace_dir.strip() if debug_trace_dir else ""
        self.debug_trace_every_n_rollouts = max(1, int(debug_trace_every_n_rollouts))
        self.max_train_sample_tokens = max_train_sample_tokens
        self.student_generalize_enabled = bool(student_generalize_enabled)
        self.student_generalize_mode = (
            student_generalize_mode or "only_success"
        ).strip()
        if self.student_generalize_mode not in _STUDENT_GENERALIZE_MODES:
            raise ValueError(
                "student_generalize_mode must be 'only_success' or 'always'."
            )
        self.student_generalize_source = (
            student_generalize_source or "sidecar"
        ).strip()
        if self.student_generalize_source not in {"generated", "sidecar", "train"}:
            raise ValueError(
                "student_generalize_source must be "
                "'sidecar', 'train', or 'generated'."
            )
        self.student_generalize_path = student_generalize_path.strip()
        self.student_generalize_replays = max(
            1, int(student_generalize_replays)
        )
        self.student_generalize_retest_original = bool(
            student_generalize_retest_original
        )
        self.student_generalize_level_rewards = {
            "level1": float(student_generalize_level1_reward),
            "level2": float(student_generalize_level2_reward),
        }
        self.student_generalize_confidence_enabled = bool(
            student_generalize_confidence_enabled
        )
        self.student_generalize_confidence_reward_scale = float(
            student_generalize_confidence_reward_scale
        )
        if self.student_generalize_confidence_enabled:
            if not self.student_generalize_enabled:
                raise ValueError(
                    "student generalization confidence requires generalization to "
                    "be enabled."
                )
            if not 0.0 < self.student_generalize_confidence_reward_scale < 1.0:
                raise ValueError(
                    "student generalization confidence reward scale must be in (0, 1)."
                )
            if any(
                reward <= 0.0
                for reward in self.student_generalize_level_rewards.values()
            ):
                raise ValueError(
                    "student generalization rewards must be positive when confidence "
                    "reward is enabled."
                )
            if self.aux_mode != "api" and not self._student_model_configs:
                raise ValueError(
                    "student generalization confidence requires "
                    "auxiliary_model.mode='api' or a non-empty student_models API pool."
                )
        self.student_generalize_bank = (
            self._load_student_generalize_bank(
                self.student_generalize_path,
                source=self.student_generalize_source,
            )
            if self.student_generalize_enabled and self.student_generalize_path
            else {}
        )
        self.last_history: list[dict[str, Any]] = []
        self.last_traces: list[TurnTrace] = []
        self.last_student_generalization_results: list[StudentGeneralizationResult] = []
        self.last_teacher_pre_solve_result: TeacherPreSolveResult | None = None
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
        self.aux_caller = None
        self.confidence_aux_caller = None
        if self.aux_mode == "api":
            api_caller = AsyncLLMCaller(aux_config)
            self.aux_caller = ApiAuxiliaryCaller(api_caller)
            if self.student_generalize_confidence_enabled:
                self.confidence_aux_caller = ApiAuxiliaryCaller(
                    api_caller,
                    request_overrides={"logprobs": True},
                )
        self.student_model_runtimes = self._build_student_model_runtimes(
            tokenizer_path=tokenizer_path,
            context_length=model_context_length,
            context_window_margin=context_window_margin,
        )
        self.tokenizer_path = tokenizer_path
        self.model_context_length = model_context_length

    @staticmethod
    def _normalize_student_model_configs(
        student_models: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for raw_config in student_models or []:
            if isinstance(raw_config, TutorStudentModelConfig):
                config = raw_config
            elif isinstance(raw_config, dict):
                config = TutorStudentModelConfig(**raw_config)
            else:
                raise TypeError(
                    "student_models entries must be dictionaries or "
                    "TutorStudentModelConfig instances."
                )
            if config.name in seen_names:
                raise ValueError("student_models names must be unique.")
            seen_names.add(config.name)
            normalized.append(asdict(config))

        if normalized and not any(config["weight"] > 0.0 for config in normalized):
            raise ValueError(
                "student_models must contain at least one student with positive weight."
            )
        return normalized

    def _build_student_model_runtimes(
        self,
        *,
        tokenizer_path: str | None,
        context_length: int | None,
        context_window_margin: int,
    ) -> dict[str, StudentModelRuntime]:
        runtimes: dict[str, StudentModelRuntime] = {}
        for student in self._student_model_configs:
            config = AuxModelConfig(
                base_url=student["base_url"],
                model=student["model"],
                api_key=student["api_key"],
                timeout=student["timeout"],
                max_tokens=student["max_tokens"],
                temperature=student["temperature"],
                top_p=student["top_p"],
                max_concurrency=student["max_concurrent_calls"],
                request_params=student["request_params"],
                tokenizer_path=tokenizer_path,
                context_length=context_length,
                context_window_margin=context_window_margin,
            )
            api_caller = AsyncLLMCaller(config)
            runtimes[student["name"]] = StudentModelRuntime(
                name=student["name"],
                model=student["model"],
                weight=student["weight"],
                caller=ApiAuxiliaryCaller(api_caller),
                confidence_caller=(
                    ApiAuxiliaryCaller(
                        api_caller,
                        request_overrides={"logprobs": True},
                    )
                    if self.student_generalize_confidence_enabled
                    else None
                ),
            )
        return runtimes

    def _select_student(
        self,
        data: dict[str, Any],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
    ) -> SelectedStudent:
        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False
        forced_name = (
            str(data.get(TUTOR_EVAL_STUDENT_FIELD) or "").strip() if is_eval else ""
        )

        student_model_runtimes = getattr(self, "student_model_runtimes", {})
        if student_model_runtimes:
            if forced_name:
                runtime = student_model_runtimes.get(forced_name)
                if runtime is None:
                    raise ValueError(
                        f"Unknown forced evaluation student {forced_name!r}; expected "
                        f"one of {sorted(student_model_runtimes)}."
                    )
            else:
                runtimes = list(student_model_runtimes.values())
                runtime = random.choices(
                    runtimes,
                    weights=[item.weight for item in runtimes],
                    k=1,
                )[0]
            return SelectedStudent(
                name=runtime.name,
                model=runtime.model,
                caller=runtime.caller,
                confidence_caller=runtime.confidence_caller,
            )

        if forced_name:
            raise ValueError(
                "A forced evaluation student requires non-empty student_models."
            )
        legacy_name = (
            getattr(self, "aux_model", "legacy-student")
            if getattr(self, "aux_mode", "api") == "api"
            else "self"
        )
        return SelectedStudent(
            name=legacy_name,
            model=legacy_name,
            caller=aux_caller,
            confidence_caller=getattr(self, "confidence_aux_caller", None),
        )

    @staticmethod
    def _student_metric_name(name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-")
        return normalized or "unknown"

    def _resolve_leak_check_system_prompt(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if getattr(self, "leak_penalty_mode", "binary") != "staged":
            return prompt
        if not prompt or prompt == DEFAULT_LEAK_CHECK_SYSTEM_PROMPT:
            return DEFAULT_STAGED_LEAK_CHECK_SYSTEM_PROMPT
        return prompt

    def _resolve_teacher_system_prompt(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if (
            not self.enable_thinking
            and NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT not in prompt
        ):
            prompt = (
                f"{prompt}\n\n{NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT}"
                if prompt
                else NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT
            ).strip()
        instructions = []
        if getattr(self, "teacher_anti_leak_instruction_enabled", False):
            instructions.append(TEACHER_ANTI_LEAK_INSTRUCTION)
        if getattr(self, "teacher_adaptive_instruction_enabled", False):
            instructions.append(TEACHER_ADAPTIVE_INSTRUCTION)
        for instruction in instructions:
            if instruction not in prompt:
                prompt = f"{prompt}\n\n{instruction}" if prompt else instruction
        return prompt.strip()

    def _sample_prompt_pool(
        self, pool: tuple[str, ...], *, role: str, include_base: bool = False
    ) -> PromptPoolSelection | None:
        if not pool:
            return None
        try:
            ctx = workflow_context.get()
            if bool(getattr(ctx, "is_eval", False)):
                return None
            task_id = getattr(ctx, "task_id", None)
        except Exception:
            task_id = None

        if task_id is not None:
            rng = random.Random(f"{self.prompt_pool_seed}:{role}:{int(task_id)}")
        elif role == "teacher":
            rng = self._teacher_prompt_pool_fallback_rng
        else:
            rng = self._student_prompt_pool_fallback_rng
        index = rng.randrange(len(pool) + int(include_base))
        if index == len(pool):
            return None
        return PromptPoolSelection(
            index=index,
            suffix=pool[index],
            pool="seen" if role == "student" else "",
        )

    def _select_student_prompt(
        self, data: dict[str, Any]
    ) -> PromptPoolSelection | None:
        seen_pool = getattr(self, "student_prompt_pool", ())
        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False
        if not is_eval:
            return self._sample_prompt_pool(
                seen_pool,
                role="student",
                include_base=getattr(self, "student_prompt_include_base", False),
            )

        raw_index = data.get(TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD)
        if raw_index is None:
            return None
        if isinstance(raw_index, bool):
            raise ValueError(
                "Forced evaluation student prompt index must be an integer."
            )
        try:
            index = operator.index(raw_index)
        except TypeError as exc:
            raise ValueError(
                "Forced evaluation student prompt index must be an integer; "
                f"got {raw_index!r}."
            ) from exc
        raw_pool = str(
            data.get(TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD) or "seen"
        ).strip()
        pools = {
            "seen": seen_pool,
            "heldout": getattr(self, "student_heldout_prompt_pool", ()),
        }
        if raw_pool not in pools:
            raise ValueError(
                f"Unknown forced evaluation student prompt pool {raw_pool!r}; "
                "expected 'seen' or 'heldout'."
            )
        pool = pools[raw_pool]
        if not pool:
            raise ValueError(
                f"A forced evaluation student prompt requires a non-empty {raw_pool} "
                "student prompt pool."
            )
        if index < 0 or index >= len(pool):
            raise ValueError(
                f"Forced evaluation student prompt index {index} is out of range "
                f"for {len(pool)} prompts."
            )
        return PromptPoolSelection(index=index, suffix=pool[index], pool=raw_pool)

    def _select_student_turn_behavior(
        self, *, response_index: int
    ) -> StudentTurnBehavior | None:
        behaviors = getattr(self, "student_turn_behaviors", ())
        if not getattr(self, "student_turn_behavior_enabled", False) or not behaviors:
            return None
        try:
            ctx = workflow_context.get()
            if bool(getattr(ctx, "is_eval", False)):
                return None
            task_id = getattr(ctx, "task_id", None)
        except Exception:
            task_id = None

        response_index = int(response_index)
        if response_index < 0:
            raise ValueError("student response_index must be non-negative.")
        if task_id is not None:
            rng = random.Random(
                f"{self.prompt_pool_seed}:student-turn-behavior:"
                f"{int(task_id)}:{response_index}"
            )
        else:
            rng = self._student_turn_behavior_fallback_rng

        sample = rng.random()
        cumulative_probability = 0.0
        for behavior in behaviors:
            cumulative_probability += behavior.probability
            if sample < cumulative_probability:
                return behavior
        return behaviors[-1]

    def _teacher_warmup_probability(self, rollout_version: int | None) -> float:
        if not getattr(self, "teacher_warmup_enabled", False):
            return 0.0
        version = max(0, int(rollout_version or 0))
        return max(0.0, 1.0 - version / self.teacher_warmup_steps)

    def _sample_teacher_pool_with_base(
        self,
        pool: tuple[str, ...],
        *,
        rollout_version: int | None,
        warmup_probability: float,
        task_id: int | None,
    ) -> PromptPoolSelection:
        if task_id is not None:
            rng = random.Random(f"{self.prompt_pool_seed}:teacher:{int(task_id)}")
        else:
            rng = self._teacher_prompt_pool_fallback_rng
        # The final virtual item is the clean base prompt, with the same weight as
        # every suffix loaded from JSON.
        index = rng.randrange(len(pool) + 1)
        if index == len(pool):
            return PromptPoolSelection(
                index=-1,
                suffix="",
                source="pool_base",
                rollout_version=(
                    int(rollout_version) if rollout_version is not None else None
                ),
                warmup_probability=warmup_probability,
            )
        return PromptPoolSelection(
            index=index,
            suffix=pool[index],
            source="pool",
            rollout_version=(
                int(rollout_version) if rollout_version is not None else None
            ),
            warmup_probability=warmup_probability,
        )

    @property
    def wants_group_index(self) -> bool:
        """Ask GroupedRolloutWorkflow to tell each episode its slot in the group.

        Only guided slots need it, and only then, so every other configuration
        keeps receiving the exact dict it received before.
        """
        return bool(getattr(self, "guided_slots_enabled", False))

    def _is_guided_slot(self, group_index: int | None) -> bool:
        if not getattr(self, "guided_slots_enabled", False):
            return False
        if group_index is None:
            return False
        return int(group_index) < int(self.guided_slots_count)

    def _select_guidance(
        self,
        *,
        group_index: int | None,
        turn_idx: int,
        task: str,
    ) -> TeacherGuidance | None:
        """Instruction to append to this turn's teacher prompt, if any.

        Guided slots prescribe a teaching move so the group contains moves the
        collapsed policy would never sample. Evaluation never receives guidance:
        the whole point is that the deployed policy behaves this way unprompted.
        """
        # Control arm. Unlike a guided slot this is a deployment choice being
        # measured, so it applies to every rollout, is kept in the prompt the turn
        # is trained on, and stays on at evaluation. The turn gate matches
        # opd.min_prior_failed_turns so the two arms differ only in where the
        # instruction ends up.
        if getattr(self, "prompt_instruction_enabled", False):
            if (
                int(turn_idx) - 1
                < self.prompt_instruction_min_prior_failed_turns
            ):
                return None
            return TeacherGuidance(
                kind="prompt",
                name="repair",
                instruction=self.prompt_instruction_text,
            )

        if not self._is_guided_slot(group_index):
            return None
        if int(turn_idx) not in self.guided_slots_turns:
            return None
        try:
            if bool(getattr(workflow_context.get(), "is_eval", False)):
                return None
        except Exception:  # noqa: BLE001 - no context outside a rollout worker
            pass

        moves = self.guided_slots_moves
        offset = 0
        if self.guided_slots_rotate_by_task and moves:
            # With fewer slots than moves a fixed assignment would only ever
            # explore the first few moves. Rotating by task keeps the assignment
            # deterministic per task while covering every move across the dataset.
            offset = int(hashlib.sha1(task.encode("utf-8")).hexdigest()[:8], 16)
        name = moves[(int(group_index) + offset) % len(moves)]
        return TeacherGuidance(
            kind="move",
            name=name,
            instruction=TEACHER_MOVE_INSTRUCTIONS[name],
            slot=int(group_index),
        )

    def _select_teacher_prompt(
        self,
        pool: tuple[str, ...],
        *,
        rollout_version: int | None,
    ) -> PromptPoolSelection | None:
        try:
            ctx = workflow_context.get()
            if bool(getattr(ctx, "is_eval", False)):
                return None
            task_id = getattr(ctx, "task_id", None)
        except Exception:
            task_id = None

        warmup_probability = self._teacher_warmup_probability(rollout_version)
        if warmup_probability > 0.0:
            if task_id is not None:
                gate_rng = random.Random(
                    f"{self.prompt_pool_seed}:teacher-warmup-gate:{int(task_id)}"
                )
            else:
                gate_rng = self._teacher_warmup_fallback_rng
            if gate_rng.random() < warmup_probability:
                return PromptPoolSelection(
                    index=-1,
                    suffix="",
                    source="warmup_full",
                    rollout_version=int(rollout_version or 0),
                    warmup_probability=warmup_probability,
                    prompt_path=self.teacher_warmup_prompt_path,
                )

        if pool or getattr(self, "teacher_warmup_enabled", False):
            return self._sample_teacher_pool_with_base(
                pool,
                rollout_version=rollout_version,
                warmup_probability=warmup_probability,
                task_id=task_id,
            )
        return None

    @staticmethod
    def _append_prompt_pool_suffix(
        base_prompt: str, selection: PromptPoolSelection | None
    ) -> str:
        if selection is None:
            return base_prompt
        return f"{base_prompt.rstrip()}\n\n{selection.suffix}".strip()

    def _student_system_prompt_for_selection(
        self, selection: PromptPoolSelection | None
    ) -> str:
        return self._append_prompt_pool_suffix(self.student_system_prompt, selection)

    def _student_system_prompt_for_state(self, state: StudentTurnState) -> str:
        return self._student_system_prompt_for_selection(
            state.student_prompt_selection
        )

    def _student_question_system_prompt_for_state(
        self, state: StudentTurnState
    ) -> str:
        return self._append_prompt_pool_suffix(
            STUDENT_QUESTION_SYSTEM_PROMPT,
            state.student_prompt_selection,
        )

    def _is_student_separate_call_behavior(
        self, behavior: StudentTurnBehavior | None
    ) -> bool:
        return bool(
            behavior is not None
            and behavior.name
            in getattr(
                self,
                "student_turn_behavior_separate_call_behavior_names",
                frozenset(),
            )
        )

    def _student_turn_behavior_prompt(self, state: StudentTurnState) -> str:
        behavior = state.student_turn_behavior
        if (
            behavior is None
            or not behavior.instruction
            or self._is_student_separate_call_behavior(behavior)
        ):
            return ""
        return behavior.instruction.strip()

    def _teacher_system_prompt_for_selection(
        self, selection: PromptPoolSelection | None
    ) -> str:
        if selection is not None and selection.source == "warmup_full":
            return self.teacher_warmup_prompt
        return self._append_prompt_pool_suffix(self.teacher_system_prompt, selection)

    def _parse_tutor_visible_output(self, raw_output: str) -> tuple[str, str | None]:
        if getattr(self, "enable_thinking", False):
            return _strip_reasoning_for_context(raw_output), None
        output, parse_error = parse_tagged_teacher_output(raw_output)
        if output is None:
            return "", parse_error or "failed to parse tagged teacher output"
        return _strip_reasoning_for_context(output), None

    def _extract_tutor_visible_output(self, raw_output: str) -> str:
        output, _ = self._parse_tutor_visible_output(raw_output)
        return output

    async def arun_episode(self, engine, data: dict[str, Any]):
        return await self._run_episode_with_polaris_process(data, engine=engine)

    async def run(self, data: dict[str, Any], **extra_kwargs):
        teacher_client = make_teacher_client(extra_kwargs)
        await self._run_episode_with_polaris_process(
            data, external_client=teacher_client
        )
        return self.last_total_reward

    async def _run_episode_with_polaris_process(
        self,
        data: dict[str, Any],
        *,
        engine: Any | None = None,
        external_client: Any | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if getattr(self, "dataset_type", "aime") != "polaris":
            return await self._run_episode(
                data, engine=engine, external_client=external_client
            )

        from examples.tutor.core.polaris import PolarisScoreProcess

        score_process = PolarisScoreProcess()
        token = _POLARIS_SCORE_PROCESS.set(score_process)
        try:
            return await self._run_episode(
                data, engine=engine, external_client=external_client
            )
        finally:
            _POLARIS_SCORE_PROCESS.reset(token)
            await asyncio.to_thread(score_process.close)

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

        if self._should_skip_missing_generated_variants(data):
            self.last_history = []
            self.last_traces = []
            self.last_student_generalization_results = []
            self.last_teacher_pre_solve_result = None
            self.last_total_reward = 0.0
            return None

        task = str(data["task"])
        ground_truth = str(data["ground_truth"])
        # Attached by GroupedRolloutWorkflow only when wants_group_index is set.
        group_index = data.get("group_index")
        group_index = None if group_index is None else int(group_index)
        trajectory_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.last_student_generalization_results = []
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
        teacher_prompt_selection = self._select_teacher_prompt(
            getattr(self, "teacher_prompt_pool", ()),
            rollout_version=episode_lora_version,
        )
        student_prompt_selection = self._select_student_prompt(data)
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
        selected_student = self._select_student(data, aux_caller=aux_caller)
        student_caller = selected_student.caller
        student_generalize_caller = self._make_student_generalization_caller(
            aux_caller=student_caller,
            confidence_caller=selected_student.confidence_caller,
        )
        answer_judge_caller = self._make_answer_judge_caller(
            chat_caller=aux_chat_caller
        )
        teacher_pre_solve_result: TeacherPreSolveResult | None = None
        self.last_teacher_pre_solve_result = None
        if getattr(self, "teacher_pre_enabled", False):
            teacher_pre_solve_result = await self._run_teacher_pre_solve(
                task,
                ground_truth,
                actor_caller=actor_caller,
                answer_judge_caller=answer_judge_caller,
                lora_version=episode_lora_version,
            )
            self.last_teacher_pre_solve_result = teacher_pre_solve_result
            if not teacher_pre_solve_result.accepted:
                self.last_history = []
                self.last_traces = []
                self.last_student_generalization_results = []
                self.last_total_reward = 0.0
                completed_repeat_outcome = self._log_rollout_stats(
                    total_reward=0.0,
                    traces=[],
                    termination_reason=TEACHER_PRE_SKIPPED_TERMINATION_REASON,
                    pre_success=False,
                    leak_count=0,
                    teacher_pre_solve_result=teacher_pre_solve_result,
                    student_name=selected_student.name,
                    teacher_prompt_selection=teacher_prompt_selection,
                    student_prompt_selection=student_prompt_selection,
                )
                if completed_repeat_outcome is not None and self.debug_trace_dir:
                    await self._dump_eval_repeat_outcomes(*completed_repeat_outcome)
                await self._maybe_dump_debug_trace(
                    trajectory_id=trajectory_id,
                    task=task,
                    ground_truth=ground_truth,
                    initial_student_answer="",
                    latest_student_answer="",
                    total_reward=0.0,
                    traces=[],
                    termination_reason=TEACHER_PRE_SKIPPED_TERMINATION_REASON,
                    pre_success=False,
                    leak_count=0,
                    teacher_pre_solve_result=teacher_pre_solve_result,
                    student_name=selected_student.name,
                    student_model=selected_student.model,
                    teacher_prompt_selection=teacher_prompt_selection,
                    student_prompt_selection=student_prompt_selection,
                )
                return None

        initial_student_turn_behavior = self._select_student_turn_behavior(
            response_index=0
        )
        initial_student_state = StudentTurnState(
            task=task,
            public_history=PublicHistoryState(),
            previous_student_output="",
            latest_tutor_visible_output=INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
            student_prompt_selection=student_prompt_selection,
            student_turn_behavior=initial_student_turn_behavior,
        )
        initial_student_answer_raw, initial_student_error = await self._run_student(
            initial_student_state,
            aux_caller=student_caller,
        )
        initial_student_answer = _strip_reasoning_for_context(
            initial_student_answer_raw
        )
        initial_judge_result = await self._score_answer_async(
            task,
            ground_truth,
            initial_student_answer_raw,
            answer_judge_caller=answer_judge_caller,
        )
        (
            initial_student_answer,
            initial_effective_student_turn_behavior,
            initial_student_question_generation,
        ) = await self._maybe_append_student_question(
            initial_student_state,
            initial_student_answer,
            should_generate=(
                not initial_judge_result.correct and not initial_student_error
            ),
            aux_caller=student_caller,
        )
        if initial_judge_result.correct:
            self.last_history = []
            self.last_traces = []
            self.last_student_generalization_results = []
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
                teacher_pre_solve_result=teacher_pre_solve_result,
                student_name=selected_student.name,
                student_model=selected_student.model,
                teacher_prompt_selection=teacher_prompt_selection,
                student_prompt_selection=student_prompt_selection,
                initial_student_turn_behavior=initial_student_turn_behavior,
            )
            completed_repeat_outcome = self._log_rollout_stats(
                total_reward=0.0,
                traces=[],
                termination_reason=episode_artifact.termination_reason,
                pre_success=episode_artifact.pre_success,
                leak_count=episode_artifact.leak_count,
                teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
                student_name=selected_student.name,
                student_call_failed=bool(initial_student_error),
                teacher_prompt_selection=teacher_prompt_selection,
                student_prompt_selection=student_prompt_selection,
                initial_student_turn_behavior=initial_student_turn_behavior,
            )
            if completed_repeat_outcome is not None and self.debug_trace_dir:
                await self._dump_eval_repeat_outcomes(*completed_repeat_outcome)
            await self._maybe_dump_debug_trace(
                trajectory_id=trajectory_id,
                task=episode_artifact.task,
                ground_truth=episode_artifact.ground_truth,
                initial_student_answer=episode_artifact.initial_student_answer,
                latest_student_answer=episode_artifact.latest_student_answer,
                total_reward=0.0,
                traces=[],
                termination_reason=episode_artifact.termination_reason,
                pre_success=episode_artifact.pre_success,
                leak_count=episode_artifact.leak_count,
                teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
                student_name=selected_student.name,
                student_model=selected_student.model,
                teacher_prompt_selection=teacher_prompt_selection,
                student_prompt_selection=student_prompt_selection,
                initial_student_turn_behavior=initial_student_turn_behavior,
                initial_student_question_generation=(
                    initial_student_question_generation
                ),
            )
            return None

        initial_turns = self._initial_conversation(initial_student_answer)
        initial_turns[-1]["env"] = self._teacher_env_feedback(
            initial_judge_result, 0
        )
        public_history = PublicHistoryState(
            summary=self._build_initial_public_summary(initial_student_answer),
            turn_count=0,
            turns=initial_turns,
        )
        previous_tutor_visible_output = ""
        previous_student_output = initial_student_answer
        preceding_student_turn_behavior = initial_effective_student_turn_behavior
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
                teacher_pre_solve_result=teacher_pre_solve_result,
                teacher_prompt_selection=teacher_prompt_selection,
                student_reply_before_teacher=previous_student_output,
                preceding_student_turn_behavior=preceding_student_turn_behavior,
                guidance=self._select_guidance(
                    group_index=group_index,
                    turn_idx=turn_idx,
                    task=task,
                ),
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
            tutor_visible_output, tutor_format_error = self._parse_tutor_visible_output(
                tutor_raw_output
            )
            public_before = list(public_history.turns)
            leak_result = self._pending_leak_check_result()
            if self.leak_handling_mode == "terminate":
                leak_result = await self._run_optional_leak_check(
                    task,
                    ground_truth,
                    tutor_visible_output,
                    aux_caller=aux_caller,
                )
                if self.leak_handling_mode == "terminate" and leak_result.leaked:
                    termination_reason = LEAK_TERMINATION_REASON
                    turn_artifacts.append(
                        TurnArtifact(
                            turn_idx=turn_idx,
                            tutor_state=tutor_state,
                            tutor_messages=list(self._build_tutor_messages(tutor_state)),
                            tutor_response=response,
                            tutor_raw_output=tutor_raw_output,
                            tutor_visible_output=tutor_visible_output,
                            leak_result=leak_result,
                            public_history_before=public_before,
                            public_history_after=public_before,
                            tutor_format_error=tutor_format_error,
                        )
                    )
                    break

            student_turn_behavior = self._select_student_turn_behavior(
                response_index=turn_idx
            )
            student_state = StudentTurnState(
                task=task,
                public_history=public_history,
                previous_student_output=previous_student_output,
                latest_tutor_visible_output=tutor_visible_output,
                student_prompt_selection=student_prompt_selection,
                student_turn_behavior=student_turn_behavior,
            )
            student_prompt = self._build_student_prompt_from_state(student_state)
            student_answer_raw, student_error = await self._run_student(
                student_state,
                aux_caller=student_caller,
            )
            student_answer = _strip_reasoning_for_context(student_answer_raw)
            judge_result = await self._score_answer_async(
                task,
                ground_truth,
                student_answer_raw,
                answer_judge_caller=answer_judge_caller,
            )
            (
                student_answer,
                effective_student_turn_behavior,
                student_question_generation,
            ) = await self._maybe_append_student_question(
                student_state,
                student_answer,
                should_generate=(
                    not judge_result.correct
                    and not student_error
                    and turn_idx < self.max_turns
                ),
                aux_caller=student_caller,
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
                env_feedback=self._teacher_env_feedback(judge_result, turn_idx),
            )
            turn_artifacts.append(
                TurnArtifact(
                    turn_idx=turn_idx,
                    tutor_state=tutor_state,
                    tutor_messages=list(self._build_tutor_messages(tutor_state)),
                    tutor_response=response,
                    tutor_raw_output=tutor_raw_output,
                    tutor_visible_output=tutor_visible_output,
                    leak_result=leak_result,
                    public_history_before=public_before,
                    public_history_after=list(next_public_history.turns),
                    tutor_format_error=tutor_format_error,
                    student_state=student_state,
                    student_prompt=student_prompt,
                    student_output=student_answer,
                    student_error=student_error,
                    judge_result=judge_result,
                    student_question_generation=student_question_generation,
                )
            )

            public_history = next_public_history
            previous_tutor_visible_output = tutor_visible_output
            previous_student_output = student_answer
            preceding_student_turn_behavior = effective_student_turn_behavior
            previous_feedback = TutorPrivateFeedback(
                kind="student_judged",
                student_output=student_answer,
                judge_correct=judge_result.correct,
                judge_feedback=judge_result.feedback,
            )
            if judge_result.correct:
                break

        if self.leak_handling_mode == "reward_only":
            leak_count = await self._annotate_turn_leak_results(
                task,
                ground_truth,
                turn_artifacts,
                aux_caller=aux_caller,
            )
        else:
            leak_count = sum(
                1 for artifact in turn_artifacts if artifact.leak_result.leaked
            )
        await self._annotate_student_requests(
            turn_artifacts,
            aux_caller=aux_caller,
        )
        await self._annotate_teacher_progress(
            turn_artifacts,
            aux_caller=aux_caller,
        )
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
            teacher_pre_solve_result=teacher_pre_solve_result,
            student_name=selected_student.name,
            student_model=selected_student.model,
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
            initial_student_turn_behavior=initial_student_turn_behavior,
            initial_student_question_generation=(
                initial_student_question_generation
            ),
        )
        student_generalization_results = await self._run_student_generalization(
            data,
            episode_artifact,
            aux_caller=student_generalize_caller,
            answer_judge_caller=answer_judge_caller,
        )
        await self._annotate_teacher_diversity(turn_artifacts)
        reward_computer = EpisodeRewardComputer(
            success_reward=self.success_reward,
            leak_penalty=self.leak_penalty,
            leak_penalty_mode=getattr(self, "leak_penalty_mode", "binary"),
            leak_penalty_final_answer=getattr(self, "leak_penalty_final_answer", None),
            leak_penalty_compute=getattr(self, "leak_penalty_compute", None),
            leak_penalty_formula=getattr(self, "leak_penalty_formula", None),
            leak_penalty_aggregation=getattr(self, "leak_penalty_aggregation", "turn"),
            format_error_penalty=getattr(self, "format_error_penalty", 0.0),
            leaked_success_reward_scale=getattr(
                self, "leaked_success_reward_scale", 1.0
            ),
            assign_success_reward=self.assign_success_reward,
            outcome_prior_turn_weight=self.outcome_prior_turn_weight,
            outcome_credit_gamma=self.outcome_credit_gamma,
            early_success_bonus=self.early_success_bonus,
            success_turn_shaping_enabled=getattr(
                self, "success_turn_shaping_enabled", False
            ),
            success_turn_shaping_min_reward=getattr(
                self, "success_turn_shaping_min_reward", 1.0
            ),
            success_turn_shaping_max_reward=getattr(
                self, "success_turn_shaping_max_reward", 1.0
            ),
            max_turn_penalty=getattr(self, "max_turn_penalty", 0.0),
            enable_turn_penalty=self.enable_turn_penalty,
            turn_penalty=self.turn_penalty,
            length_penalty_threshold_chars=self.length_penalty_threshold_chars,
            length_penalty_per_100_chars=self.length_penalty_per_100_chars,
            length_penalty_min=self.length_penalty_min,
        )
        assignments = await reward_computer.compute(episode_artifact)
        self._apply_student_request_rewards(turn_artifacts, assignments)
        self._apply_teacher_progress_shaping(turn_artifacts, assignments)
        self._apply_student_generalization_rewards(
            turn_artifacts, assignments, student_generalization_results
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
        clean_teacher_inputs = [
            (
                self._clean_tutor_input_tokens(artifact)
                # A guided turn MUST get the override: without it the appended
                # instruction stays in the training prompt and the policy learns
                # to obey an instruction it will never see at eval, silently and
                # with no error. Control-arm ("prompt") guidance is the exact
                # opposite -- it is meant to stay -- so it asks not to be stripped.
                if artifact.tutor_state.teacher_prompt_selection is not None
                or (
                    artifact.tutor_state.guidance is not None
                    and artifact.tutor_state.guidance.strip_from_training
                )
                else None
            )
            for artifact in turn_artifacts
        ]
        training_teacher_inputs = [
            (
                list(clean_input)
                if clean_input is not None
                else list(artifact.tutor_response.input_tokens)
            )
            for artifact, clean_input in zip(
                turn_artifacts, clean_teacher_inputs, strict=True
            )
        ]
        preceding_teacher_inputs = [None, *training_teacher_inputs[:-1]]
        world_model_examples = await self._build_world_model_examples_async(
            turn_artifacts
        )
        opd_prompt_tokens = await self._build_opd_prompt_tokens_async(turn_artifacts)
        results = [
            response_to_tensordict(
                artifact.tutor_response,
                reward=assignment.reward,
                trajectory_id=trajectory_id,
                turn_idx=artifact.turn_idx,
                input_tokens_override=clean_input,
                zero_reward_on_length_stop=getattr(
                    self, "zero_reward_on_length_stop", False
                ),
                batch_centered_penalty_score=(
                    artifact.previous_teacher_similarity
                    if getattr(self, "teacher_diversity_enabled", False)
                    else None
                ),
                batch_centered_penalty_weight=(
                    getattr(self, "teacher_diversity_weight", 0.0)
                    if getattr(self, "teacher_diversity_enabled", False)
                    else None
                ),
                teacher_context_input_tokens=(
                    preceding_input
                    if getattr(self, "teacher_context_enabled", False)
                    else None
                ),
                teacher_context_reward_weight=(
                    getattr(self, "teacher_context_weight", 0.0)
                    if getattr(self, "teacher_context_enabled", False)
                    else None
                ),
                teacher_context_reward_score_clip=(
                    getattr(self, "teacher_context_score_clip", 5.0)
                    if getattr(self, "teacher_context_enabled", False)
                    else None
                ),
                teacher_context_reward_apply_to_advantage=(
                    getattr(self, "teacher_context_apply_to_advantage", True)
                    if getattr(self, "teacher_context_enabled", False)
                    else None
                ),
                world_model_input_tokens=(
                    world_model_example.input_ids
                    if world_model_example is not None
                    else None
                ),
                world_model_target_mask=(
                    world_model_example.target_mask
                    if world_model_example is not None
                    else None
                ),
                world_model_loss_weight=(
                    self.world_model_loss_weight
                    if world_model_example is not None
                    else None
                ),
                world_model_paw_config=(
                    self.world_model_paw_config
                    if world_model_example is not None
                    else None
                ),
                # Present on every row once OPD is on, so the padded batch stays
                # rectangular; rows that were not selected carry weight 0.
                opd_input_tokens=opd_prompt,
                opd_loss_weight=(
                    self.opd_loss_weight
                    if getattr(self, "opd_enabled", False)
                    else None
                ),
                opd_reward_clip=getattr(self, "opd_reward_clip", 0.0),
            )
            for (
                artifact,
                assignment,
                clean_input,
                preceding_input,
                world_model_example,
                opd_prompt,
            ) in zip(
                turn_artifacts,
                assignments,
                clean_teacher_inputs,
                preceding_teacher_inputs,
                world_model_examples,
                opd_prompt_tokens,
                strict=True,
            )
        ]
        total_reward = float(sum(assignment.reward for assignment in assignments))
        self.last_history = history
        self.last_traces = traces
        self.last_student_generalization_results = student_generalization_results
        self.last_total_reward = total_reward
        completed_repeat_outcome = self._log_rollout_stats(
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            pre_success=episode_artifact.pre_success,
            leak_count=episode_artifact.leak_count,
            student_generalization_results=student_generalization_results,
            teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
            student_name=selected_student.name,
            student_call_failed=bool(
                initial_student_error
                or any(artifact.student_error for artifact in turn_artifacts)
            ),
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
            initial_student_turn_behavior=initial_student_turn_behavior,
            inference_prompt_tokens=sum(
                artifact.tutor_response.input_len for artifact in turn_artifacts
            ),
            training_prompt_tokens=sum(
                len(clean_input)
                if clean_input is not None
                else artifact.tutor_response.input_len
                for artifact, clean_input in zip(
                    turn_artifacts, clean_teacher_inputs, strict=True
                )
            ),
        )
        if completed_repeat_outcome is not None and self.debug_trace_dir:
            await self._dump_eval_repeat_outcomes(*completed_repeat_outcome)
        await self._maybe_dump_debug_trace(
            trajectory_id=trajectory_id,
            task=episode_artifact.task,
            ground_truth=episode_artifact.ground_truth,
            initial_student_answer=episode_artifact.initial_student_answer,
            latest_student_answer=episode_artifact.latest_student_answer,
            total_reward=total_reward,
            traces=traces,
            termination_reason=episode_artifact.termination_reason,
            pre_success=episode_artifact.pre_success,
            leak_count=episode_artifact.leak_count,
            student_generalization_results=student_generalization_results,
            teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
            student_name=selected_student.name,
            student_model=selected_student.model,
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
            initial_student_turn_behavior=initial_student_turn_behavior,
            initial_student_question_generation=(
                initial_student_question_generation
            ),
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

    def _make_student_generalization_caller(
        self,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        confidence_caller: ApiAuxiliaryCaller | None = None,
    ) -> ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller:
        if not getattr(self, "student_generalize_confidence_enabled", False):
            return aux_caller
        confidence_caller = confidence_caller or getattr(
            self, "confidence_aux_caller", None
        )
        if confidence_caller is None:
            raise RuntimeError(
                "student generalization confidence requires an auxiliary API "
                "caller with token logprobs."
            )
        return confidence_caller

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

    def _teacher_pre_solve_tokens(self) -> int:
        if self.teacher_pre_max_tokens > 0:
            return self.teacher_pre_max_tokens
        return self.max_completion_tokens

    def _build_teacher_pre_solve_prompt(self, *, task: str) -> str:
        if getattr(self, "dataset_type", "aime") == "polaris":
            return POLARIS_FILTER_SOLVER_USER_TEMPLATE.format(task=task)
        return FILTER_SOLVER_USER_TEMPLATE.format(task=task)

    def _build_teacher_pre_solve_messages(self, *, task: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": FILTER_SOLVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_teacher_pre_solve_prompt(task=task),
            },
        ]

    async def _run_teacher_pre_solve(
        self,
        task: str,
        ground_truth: str,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        lora_version: int | None,
    ) -> TeacherPreSolveResult:
        attempts: list[TeacherPreSolveAttempt] = []
        messages = self._build_teacher_pre_solve_messages(task=task)
        max_completion_tokens = self._teacher_pre_solve_tokens()
        verification_enabled = bool(getattr(self, "teacher_pre_verify", True))
        attempt_count = self.teacher_pre_attempts if verification_enabled else 1
        for attempt_idx in range(1, attempt_count + 1):
            try:
                result = await actor_caller.generate(
                    messages,
                    lora_version=lora_version,
                    rid_prefix=f"teacher-pre-{attempt_idx}",
                    max_completion_tokens=max_completion_tokens,
                )
            except Exception as exc:
                attempts.append(
                    TeacherPreSolveAttempt(
                        attempt=attempt_idx,
                        raw_output="",
                        error=str(exc),
                        accepted=False,
                        judge_result=None,
                    )
                )
                continue

            raw_output = str(result.raw_text or "").strip()
            if not verification_enabled:
                attempts.append(
                    TeacherPreSolveAttempt(
                        attempt=attempt_idx,
                        raw_output=raw_output,
                        error=None,
                        accepted=True,
                        judge_result=None,
                    )
                )
                return TeacherPreSolveResult(
                    enabled=True,
                    mode=self.teacher_pre_mode,
                    accepted=True,
                    attempts=attempts,
                    raw_output=raw_output,
                    error=None,
                    verification_enabled=False,
                )
            judge_result = await self._score_answer_async(
                task,
                ground_truth,
                raw_output,
                answer_judge_caller=answer_judge_caller,
            )
            accepted = bool(judge_result.correct)
            attempts.append(
                TeacherPreSolveAttempt(
                    attempt=attempt_idx,
                    raw_output=raw_output,
                    error=None,
                    accepted=accepted,
                    judge_result=judge_result,
                )
            )
            if accepted:
                return TeacherPreSolveResult(
                    enabled=True,
                    mode=self.teacher_pre_mode,
                    accepted=True,
                    attempts=attempts,
                    raw_output=raw_output,
                    error=None,
                    verification_enabled=True,
                )

        return TeacherPreSolveResult(
            enabled=True,
            mode=self.teacher_pre_mode,
            accepted=False,
            attempts=attempts,
            raw_output="",
            error=(
                f"no correct teacher pre-solve after {attempt_count} attempts"
                if verification_enabled
                else "teacher pre-solve draft generation failed"
            ),
            verification_enabled=verification_enabled,
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
        input_token_reserve = self._clean_teacher_input_token_reserve(
            messages,
            tutor_state.teacher_prompt_selection,
            tutor_state.guidance,
        )
        if actor_caller is None:
            actor_caller = self._make_actor_caller(engine, external_client)
        result = await actor_caller.generate(
            messages,
            lora_version=lora_version,
            rid_prefix=f"{rid_prefix}-{tutor_state.turn_idx}",
            input_token_reserve=input_token_reserve,
        )
        return result.response, result.raw_text

    async def _run_student(
        self,
        state: StudentTurnState,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> tuple[str, str | None]:
        result = await self._call_auxiliary_messages(
            self._build_student_messages(state),
            aux_caller=aux_caller,
            rid_prefix=f"student-{state.public_history.turn_count}",
        )
        if result.error:
            return "", result.error
        return result.text, None

    def _build_student_question_prompt(
        self,
        state: StudentTurnState,
        student_answer: str,
    ) -> str:
        behavior = state.student_turn_behavior
        if not self._is_student_separate_call_behavior(behavior):
            raise ValueError(
                "student question generation requires a request behavior."
            )
        return render_prompt(
            STUDENT_QUESTION_USER_TEMPLATE,
            task=state.task,
            public_history=state.public_history.summary,
            teacher_feedback=state.latest_tutor_visible_output,
            student_answer=student_answer,
            question_instruction=behavior.instruction,
        )

    async def _run_student_question(
        self,
        state: StudentTurnState,
        student_answer: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> StudentQuestionGenerationResult:
        prompt = self._build_student_question_prompt(state, student_answer)
        result = await self._call_auxiliary_prompt(
            system_prompt=self._student_question_system_prompt_for_state(state),
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix=f"student-question-{state.public_history.turn_count}",
        )
        raw_output = result.raw_text or result.text
        if result.error:
            return StudentQuestionGenerationResult(
                prompt=prompt,
                raw_output=raw_output,
                question="",
                error=result.error,
            )
        question = _strip_reasoning_for_context(result.text).strip()
        error = None
        if not question:
            error = "student question generation returned an empty response."
        return StudentQuestionGenerationResult(
            prompt=prompt,
            raw_output=raw_output,
            question=question,
            error=error,
        )

    async def _maybe_append_student_question(
        self,
        state: StudentTurnState,
        student_answer: str,
        *,
        should_generate: bool,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> tuple[
        str,
        StudentTurnBehavior | None,
        StudentQuestionGenerationResult | None,
    ]:
        behavior = state.student_turn_behavior
        if not self._is_student_separate_call_behavior(behavior):
            return student_answer, behavior, None
        if not should_generate:
            return student_answer, None, None

        generation = await self._run_student_question(
            state,
            student_answer,
            aux_caller=aux_caller,
        )
        if generation.error or not generation.question:
            return student_answer, None, generation
        combined_answer = "\n\n".join(
            part for part in (student_answer.strip(), generation.question) if part
        )
        return combined_answer, behavior, generation

    @staticmethod
    def _pending_leak_check_result() -> LeakCheckResult:
        return LeakCheckResult(
            raw_output="",
            leaked=False,
            feedback=LEAK_CHECK_PENDING_FEEDBACK,
            parse_error=None,
            raw_result={"pending": True},
        )

    async def _annotate_turn_leak_results(
        self,
        task: str,
        ground_truth: str,
        turn_artifacts: list[TurnArtifact],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> int:
        if not turn_artifacts:
            return 0
        leak_results = await asyncio.gather(
            *(
                self._run_optional_leak_check(
                    task,
                    ground_truth,
                    artifact.tutor_visible_output,
                    aux_caller=aux_caller,
                )
                for artifact in turn_artifacts
            )
        )
        leak_count = 0
        for artifact, leak_result in zip(turn_artifacts, leak_results, strict=True):
            artifact.leak_result = leak_result
            leak_count += int(leak_result.leaked)
        return leak_count

    async def _run_optional_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> LeakCheckResult:
        if self.leak_handling_mode == "disabled":
            return LeakCheckResult(
                raw_output="",
                leaked=False,
                feedback=LEAK_CHECK_DISABLED_FEEDBACK,
                parse_error=None,
                raw_result={"disabled": True},
            )
        if getattr(self, "leak_penalty_mode", "binary") == "rawbase":
            return await self._run_rawbase_leak_check(
                task,
                ground_truth,
                teacher_action,
                aux_caller=aux_caller,
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
                feedback=LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE.format(error=result.error),
                parse_error=result.error,
                raw_result={},
                leak_level=(
                    1
                    if getattr(self, "leak_penalty_mode", "binary") == "staged"
                    else None
                ),
            )
        if getattr(self, "leak_penalty_mode", "binary") == "staged":
            return parse_staged_leak_check_result(result.text)
        return parse_leak_check_result(result.text)

    async def _run_rawbase_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
    ) -> LeakCheckResult:
        del task
        # The rollout passes an already-extracted student-visible action here.
        # Parsing it again would discard valid plain text in non-thinking mode.
        teacher_message = _strip_reasoning_for_context(teacher_action)
        prompt = render_prompt(
            RAWBASE_LEAK_CHECK_USER_TEMPLATE,
            ground_truth=ground_truth,
            teacher_action=teacher_message,
        )
        result = await self._call_auxiliary_prompt(
            system_prompt=RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix="rawbase-leak-check",
        )
        if result.error:
            return LeakCheckResult(
                raw_output="",
                leaked=True,
                feedback=RAWBASE_LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE.format(
                    error=result.error
                ),
                parse_error=result.error,
                raw_result={"method": "rawbase_llm"},
            )
        leak_result = parse_leak_check_result(result.text)
        leak_result.raw_result["method"] = "rawbase_llm"
        return leak_result

    async def _call_auxiliary_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        return await self._call_auxiliary_messages(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            aux_caller=aux_caller,
            rid_prefix=rid_prefix,
        )

    async def _call_auxiliary_messages(
        self,
        messages: list[dict[str, str]],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        caller = aux_caller or self._make_auxiliary_caller(engine=None)
        return await caller.call_text(messages, rid_prefix=rid_prefix)

    async def _run_public_summary_update(
        self,
        *,
        old_public_history: PublicHistoryState,
        previous_student_answer: str,
        tutor_visible_output: str,
        current_student_answer: str,
        env_feedback: str = "",
    ) -> PublicHistoryState:
        entries = []
        existing_history = old_public_history.summary.strip()
        if existing_history:
            entries.append(existing_history)
        else:
            entries.append(self._build_initial_public_summary(previous_student_answer))

        turn_idx = old_public_history.turn_count + 1
        entries.append(
            self._format_public_history_entry(
                "Tutor",
                turn_idx,
                tutor_visible_output,
            )
        )
        entries.append(
            self._format_public_history_entry(
                "Student",
                turn_idx,
                current_student_answer,
            )
        )
        turns = list(old_public_history.turns) or self._initial_conversation(
            previous_student_answer
        )
        turns.append({"role": "teacher", "content": tutor_visible_output})
        student_turn = {"role": "student", "content": current_student_answer}
        if env_feedback:
            student_turn["env"] = env_feedback
        turns.append(student_turn)
        return PublicHistoryState(
            summary="\n\n".join(entry for entry in entries if entry),
            turn_count=old_public_history.turn_count + 1,
            turns=turns,
        )

    def _build_tutor_prompt(self, state: TutorTurnState) -> str:
        feedback = state.previous_feedback
        prompt = render_prompt(
            self.teacher_user_prompt_template,
            task=state.task,
            ground_truth=state.ground_truth,
            show_ground_truth=self.teacher_show_ground_truth,
            public_history=state.public_history.summary or NO_VISIBLE_TUTORING_HISTORY,
            previous_tutor_output=state.previous_tutor_visible_output
            or NONE_YET_PLACEHOLDER,
            feedback_kind=feedback.kind,
            student_output=feedback.student_output or EMPTY_PLACEHOLDER,
            judge_correct=feedback.judge_correct,
            judge_feedback=feedback.judge_feedback or EMPTY_PLACEHOLDER,
            current_round=state.turn_idx,
            max_turns=state.max_turns,
            remaining_rounds=max(state.max_turns - state.turn_idx + 1, 0),
        )
        return self._append_teacher_pre_solve_context(
            prompt, state.teacher_pre_solve_result
        )

    def _append_teacher_pre_solve_context(
        self, prompt: str, teacher_pre_solve: TeacherPreSolveResult | None
    ) -> str:
        if not getattr(self, "teacher_pre_enabled", False) or teacher_pre_solve is None:
            return prompt
        raw_output = _strip_reasoning_for_context(
            str(teacher_pre_solve.raw_output or "")
        ).strip()
        if not teacher_pre_solve.accepted or not raw_output:
            return prompt
        context = render_prompt(
            TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE,
            raw_output=raw_output,
        )
        return f"{prompt.rstrip()}\n\n{context}\n"

    def _build_student_prompt_from_state(self, state: StudentTurnState) -> str:
        if (
            getattr(self, "dataset_type", "aime") == "polaris"
            and not state.public_history.summary
            and not state.previous_student_output
            and state.public_history.turn_count == 0
        ):
            prompt = f"{state.task}\n\n{POLARIS_INSTRUCTION}"
        else:
            prompt = render_prompt(
                STUDENT_STATE_USER_TEMPLATE,
                task=state.task,
                public_history=state.public_history.summary
                or NO_PREVIOUS_VISIBLE_TUTORING_HISTORY,
                previous_student_output=(
                    state.previous_student_output or EMPTY_PLACEHOLDER
                ),
                teacher_feedback=state.latest_tutor_visible_output
                or NONE_PLACEHOLDER,
            )
        behavior_prompt = self._student_turn_behavior_prompt(state)
        if behavior_prompt:
            return f"{prompt.rstrip()}\n\n{behavior_prompt}".strip()
        return prompt

    def _build_student_transfer_prompt(
        self,
        *,
        original_task: str,
        transfer_task: str,
        public_history: PublicHistoryState,
        previous_student_output: str,
        teacher_feedback: str,
    ) -> str:
        return render_prompt(
            STUDENT_TRANSFER_USER_TEMPLATE,
            original_task=original_task,
            public_history=public_history.summary
            or NO_PREVIOUS_VISIBLE_TUTORING_HISTORY,
            previous_student_output=previous_student_output or EMPTY_PLACEHOLDER,
            teacher_feedback=teacher_feedback or NONE_PLACEHOLDER,
            transfer_task=transfer_task,
        )

    def _build_student_probe_messages(
        self,
        *,
        episode_artifact: EpisodeArtifact,
        anchor: StudentGeneralizationAnchor,
        level: str,
        transfer_task: str,
    ) -> list[dict[str, str]]:
        """A probe continues the tutoring chat: same system prompt (carrying the
        ORIGINAL task), the same dialogue, then one new user turn.

        Each probe is built fresh from the anchor and never written back into
        `turns`, so level1 / level2 / original are independent branches that do
        not see each other's question or answer.
        """
        system = self._student_system_prompt_for_selection(
            episode_artifact.student_prompt_selection
        )
        system = f"{system.rstrip()}\n\n{self._task_context(episode_artifact.task)}"
        if level == ORIGINAL_RETEST_LEVEL:
            final_turn = render_prompt(STUDENT_FINAL_SOLUTION_TEMPLATE)
        else:
            final_turn = render_prompt(
                STUDENT_TRANSFER_TURN_TEMPLATE, transfer_task=transfer_task
            )
        return [
            {"role": "system", "content": system},
            *self._render_conversation(
                anchor.public_history.turns, speaker="student"
            ),
            {"role": "user", "content": final_turn},
        ]

    def _build_leak_check_prompt(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> str:
        template = (
            STAGED_LEAK_CHECK_USER_TEMPLATE
            if getattr(self, "leak_penalty_mode", "binary") == "staged"
            else LEAK_CHECK_USER_TEMPLATE
        )
        return render_prompt(
            template,
            task=task,
            ground_truth=ground_truth,
            teacher_action=_strip_reasoning_for_context(teacher_action),
        )

    def _build_initial_public_summary(self, initial_student_answer: str) -> str:
        return self._format_public_history_entry("Student", 0, initial_student_answer)

    def _probe_levels(self) -> tuple[str, ...]:
        """Probe branches run after tutoring. The original re-test is opt-in."""
        if getattr(self, "student_generalize_retest_original", False):
            return (ORIGINAL_RETEST_LEVEL, *_STUDENT_GENERALIZE_LEVELS)
        return _STUDENT_GENERALIZE_LEVELS

    def _teacher_env_feedback(self, judge_result: Any, turn_idx: int) -> str:
        """Grading + budget the teacher sees, as environment feedback."""
        if judge_result is None:
            return ""
        return render_prompt(
            TEACHER_ENV_FEEDBACK_TEMPLATE,
            judge_correct=bool(judge_result.correct),
            current_round=turn_idx,
            max_turns=self.max_turns,
            remaining_rounds=max(self.max_turns - turn_idx, 0),
        )

    @staticmethod
    def _initial_conversation(initial_student_answer: str) -> list[dict[str, str]]:
        return [
            {
                "role": "student",
                "content": render_prompt(
                    INITIAL_ATTEMPT_WRAPPER, attempt=initial_student_answer
                ),
            }
        ]

    def _format_public_history_entry(
        self, speaker: str, round_idx: int, text: str
    ) -> str:
        visible_text = _strip_reasoning_for_context(text)
        return PUBLIC_HISTORY_ENTRY_TEMPLATE.format(
            speaker=speaker,
            round_idx=round_idx,
            visible_text=visible_text,
        )

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
        if getattr(self, "dataset_type", "aime") == "polaris":
            from examples.tutor.core.polaris import score_polaris_answer_async

            exact_result = await score_polaris_answer_async(
                task,
                ground_truth,
                student_answer,
                score_process=_POLARIS_SCORE_PROCESS.get(),
            )
        else:
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

    def _build_student_request_judge_prompt(self, artifact: TurnArtifact) -> str:
        return render_prompt(
            STUDENT_REQUEST_JUDGE_USER_TEMPLATE,
            task=artifact.tutor_state.task,
            public_history=artifact.public_history_before,
            student_reply_before_teacher=(
                artifact.tutor_state.student_reply_before_teacher
            ),
            target_teacher_reply=artifact.tutor_visible_output,
        )

    @staticmethod
    def _parse_student_request_judge_result(
        judge_call: TextCallResult,
    ) -> StudentRequestJudgeResult:
        raw_output = judge_call.raw_text or judge_call.text
        if judge_call.error:
            return StudentRequestJudgeResult(
                raw_output=raw_output,
                score=None,
                reason="",
                parse_error=judge_call.error,
            )

        parsed, parse_error = parse_json_dict(judge_call.text)
        score = parsed.get("score") if isinstance(parsed, dict) else None
        reason = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        valid_score = (
            isinstance(score, int)
            and not isinstance(score, bool)
            and score in {-1, 1}
        )
        if valid_score:
            return StudentRequestJudgeResult(
                raw_output=raw_output,
                score=int(score),
                reason=str(reason),
                parse_error=parse_error,
            )

        # Preserve an unambiguous verdict when a model emits unescaped LaTeX
        # inside the reason and therefore breaks the surrounding JSON.
        score_match = re.search(r'"score"\s*:\s*(-1|1)(?!\d)', judge_call.text)
        if score_match is not None:
            return StudentRequestJudgeResult(
                raw_output=raw_output,
                score=int(score_match.group(1)),
                reason="",
                parse_error=parse_error,
            )

        error = parse_error or 'Student request judge field "score" must be -1 or 1.'
        return StudentRequestJudgeResult(
            raw_output=raw_output,
            score=None,
            reason="",
            parse_error=error,
        )

    async def _run_student_request_judge(
        self,
        artifact: TurnArtifact,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
    ) -> StudentRequestJudgeResult:
        judge_call = await self._call_auxiliary_prompt(
            system_prompt=self.student_request_judge_system_prompt,
            user_prompt=self._build_student_request_judge_prompt(artifact),
            aux_caller=aux_caller,
            rid_prefix="student-request-judge",
        )
        return self._parse_student_request_judge_result(judge_call)

    def _student_request_judge_targets(self, artifact: TurnArtifact) -> bool:
        behavior = artifact.tutor_state.preceding_student_turn_behavior
        return bool(
            behavior is not None
            and behavior.name in self.student_request_judge_behavior_names
        )

    async def _annotate_student_requests(
        self,
        turn_artifacts: list[TurnArtifact],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> None:
        if not getattr(self, "student_request_judge_enabled", False):
            return

        targets = [
            artifact
            for artifact in turn_artifacts
            if self._student_request_judge_targets(artifact)
        ]
        if not targets:
            return
        if aux_caller is None:
            raise RuntimeError("student request judge caller is unavailable.")

        async def judge(artifact: TurnArtifact) -> StudentRequestJudgeResult:
            if artifact.invalid_due_to_leak or artifact.leak_result.leaked:
                return StudentRequestJudgeResult(
                    raw_output="",
                    score=None,
                    reason="",
                    parse_error="skipped_leaked_turn",
                )
            if not artifact.tutor_state.student_reply_before_teacher.strip():
                return StudentRequestJudgeResult(
                    raw_output="",
                    score=None,
                    reason="",
                    parse_error="missing_generated_student_question",
                )
            return await self._run_student_request_judge(
                artifact,
                aux_caller=aux_caller,
            )

        results = await asyncio.gather(*(judge(artifact) for artifact in targets))
        for artifact, result in zip(targets, results, strict=True):
            artifact.student_request_judge_result = result

    def _apply_student_request_rewards(
        self,
        turn_artifacts: list[TurnArtifact],
        assignments: list[RewardAssignment],
    ) -> None:
        if not getattr(self, "student_request_judge_enabled", False):
            return
        if len(turn_artifacts) != len(assignments):
            raise ValueError(
                "Student request rewards require one reward assignment per turn."
            )

        weight = float(self.student_request_judge_weight)
        for artifact, assignment in zip(turn_artifacts, assignments, strict=True):
            result = artifact.student_request_judge_result
            reward = (
                weight * int(result.score)
                if result is not None and result.score is not None
                else 0.0
            )
            if result is not None:
                result.reward = float(reward)
            if reward:
                assignment.reward_components["student_request_fulfillment"] = float(
                    reward
                )
                assignment.reward = float(assignment.reward + reward)

    @staticmethod
    def _teacher_progress_reference_solution(artifact: TurnArtifact) -> str:
        result = artifact.tutor_state.teacher_pre_solve_result
        if result is None or not result.accepted:
            return ""
        return str(result.raw_output or "").strip()

    def _build_teacher_progress_judge_prompt(self, artifact: TurnArtifact) -> str:
        student_reply_before_teacher = (
            artifact.student_state.previous_student_output
            if artifact.student_state is not None
            else artifact.tutor_state.previous_feedback.student_output
        )
        return render_prompt(
            TEACHER_PROGRESS_JUDGE_USER_TEMPLATE,
            task=artifact.tutor_state.task,
            ground_truth=artifact.tutor_state.ground_truth,
            reference_solution=self._teacher_progress_reference_solution(artifact),
            public_history=artifact.public_history_before,
            student_reply_before_teacher=student_reply_before_teacher,
            target_teacher_reply=artifact.tutor_visible_output,
            student_reply_after_teacher=artifact.student_output,
        )

    @staticmethod
    def _parse_teacher_progress_judge_result(
        judge_call: TextCallResult,
    ) -> TeacherProgressJudgeResult:
        raw_output = judge_call.raw_text or judge_call.text
        if judge_call.error:
            return TeacherProgressJudgeResult(
                raw_output=raw_output,
                score=None,
                reason="",
                parse_error=judge_call.error,
            )

        parsed, parse_error = parse_json_dict(judge_call.text)
        score = parsed.get("score") if isinstance(parsed, dict) else None
        reason = parsed.get("reason", "") if isinstance(parsed, dict) else ""
        if isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 2:
            return TeacherProgressJudgeResult(
                raw_output=raw_output,
                score=score,
                reason=str(reason),
                parse_error=parse_error,
            )

        # Some models emit unescaped LaTeX in the reason, making the surrounding
        # JSON invalid even though the integer score is unambiguous.
        match = re.search(r'"score"\s*:\s*([012])', judge_call.text)
        if match is not None:
            return TeacherProgressJudgeResult(
                raw_output=raw_output,
                score=int(match.group(1)),
                reason="",
                parse_error=parse_error,
            )

        error = (
            parse_error or 'Teacher progress judge field "score" must be 0, 1, or 2.'
        )
        return TeacherProgressJudgeResult(
            raw_output=raw_output,
            score=None,
            reason="",
            parse_error=error,
        )

    async def _run_teacher_progress_judge(
        self,
        artifact: TurnArtifact,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
    ) -> TeacherProgressJudgeResult:
        judge_call = await self._call_auxiliary_prompt(
            system_prompt=self.teacher_progress_judge_system_prompt,
            user_prompt=self._build_teacher_progress_judge_prompt(artifact),
            aux_caller=aux_caller,
            rid_prefix="teacher-progress-judge",
        )
        return self._parse_teacher_progress_judge_result(judge_call)

    async def _annotate_teacher_progress(
        self,
        turn_artifacts: list[TurnArtifact],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> None:
        if not getattr(self, "teacher_progress_judge_enabled", False):
            return
        if aux_caller is None:
            raise RuntimeError("teacher progress judge caller is unavailable.")

        async def judge(artifact: TurnArtifact) -> TeacherProgressJudgeResult:
            if artifact.invalid_due_to_leak or artifact.leak_result.leaked:
                return TeacherProgressJudgeResult(
                    raw_output="",
                    score=None,
                    reason="",
                    parse_error="skipped_leaked_turn",
                )
            if artifact.student_error:
                return TeacherProgressJudgeResult(
                    raw_output="",
                    score=None,
                    reason="",
                    parse_error="skipped_student_error",
                )
            return await self._run_teacher_progress_judge(
                artifact,
                aux_caller=aux_caller,
            )

        results = await asyncio.gather(
            *(judge(artifact) for artifact in turn_artifacts)
        )
        for artifact, result in zip(turn_artifacts, results, strict=True):
            artifact.teacher_progress_judge_result = result

    def _apply_teacher_progress_shaping(
        self,
        turn_artifacts: list[TurnArtifact],
        assignments: list[RewardAssignment],
    ) -> None:
        if not getattr(self, "teacher_progress_judge_enabled", False):
            return
        if len(turn_artifacts) != len(assignments):
            raise ValueError(
                "Teacher progress shaping requires one reward assignment per turn."
            )

        weight = float(self.teacher_progress_judge_weight)
        targets = []
        for artifact in turn_artifacts:
            result = artifact.teacher_progress_judge_result
            target = (
                weight * (int(result.score) - 1)
                if result is not None and result.score is not None
                else 0.0
            )
            if result is not None:
                result.local_advantage = float(target)
            targets.append(float(target))

        gamma = float(self.local_advantage_turn_discount)
        for index, (artifact, assignment) in enumerate(
            zip(turn_artifacts, assignments, strict=True)
        ):
            next_target = targets[index + 1] if index + 1 < len(targets) else 0.0
            shaping = targets[index] - gamma * next_target
            result = artifact.teacher_progress_judge_result
            if result is not None:
                result.reward_shaping = float(shaping)
            if shaping:
                assignment.reward_components["teacher_progress_shaping"] = float(
                    shaping
                )
                assignment.reward = float(assignment.reward + shaping)

    @staticmethod
    def _load_student_generalize_bank(
        path: str,
        *,
        source: str = "sidecar",
    ) -> dict[str, Any]:
        return load_student_generalize_bank(path, source=source)

    def _student_generalization_cases(
        self, data: dict[str, Any]
    ) -> dict[str, StudentGeneralizationCase]:
        cases: dict[str, StudentGeneralizationCase] = {}
        if getattr(self, "student_generalize_retest_original", False):
            cases[ORIGINAL_RETEST_LEVEL] = StudentGeneralizationCase(
                level=ORIGINAL_RETEST_LEVEL,
                task=str(data.get("task", "")),
                ground_truth=str(data.get("ground_truth", "")),
                reward=0.0,
            )
        payload: Any | None = None
        sample_id = data.get("id")
        bank = getattr(self, "student_generalize_bank", {}) or {}
        if sample_id is not None and str(sample_id) in bank:
            payload = bank[str(sample_id)]
        else:
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                payload = metadata.get("student_generalize")

        if not isinstance(payload, dict):
            return cases

        rewards = getattr(self, "student_generalize_level_rewards", {}) or {}
        if getattr(self, "student_generalize_source", "sidecar") == "train":
            raw_samples = payload.get("samples")
            items = raw_samples if isinstance(raw_samples, list) else []
        else:
            items = [payload.get(level) for level in _STUDENT_GENERALIZE_LEVELS]

        for level, item in zip(_STUDENT_GENERALIZE_LEVELS, items, strict=False):
            if not isinstance(item, dict):
                continue
            task = item.get("task")
            ground_truth = item.get("ground_truth")
            if task is None or ground_truth is None:
                continue
            cases[level] = StudentGeneralizationCase(
                level=level,
                task=str(task),
                ground_truth=str(ground_truth),
                reward=float(rewards.get(level, 0.0)),
            )
        return cases

    def _should_skip_missing_generated_variants(
        self,
        data: dict[str, Any],
    ) -> bool:
        if not bool(getattr(self, "student_generalize_enabled", False)):
            return False
        if getattr(self, "student_generalize_source", "sidecar") != "generated":
            return False
        sample_id = data.get("id")
        bank = getattr(self, "student_generalize_bank", {}) or {}
        return sample_id is None or str(sample_id) not in bank

    def _success_turn(self, episode_artifact: EpisodeArtifact) -> TurnArtifact | None:
        if episode_artifact.termination_reason != "success":
            return None
        for artifact in episode_artifact.turns:
            if artifact.invalid_due_to_leak:
                continue
            if artifact.judge_result is not None and artifact.judge_result.correct:
                return artifact
        return None

    def _turn_generalization_anchor(
        self, artifact: TurnArtifact
    ) -> StudentGeneralizationAnchor:
        turn_count = 0
        summary = ""
        if artifact.student_state is not None:
            turn_count = artifact.student_state.public_history.turn_count + 1
            entries = [artifact.student_state.public_history.summary]
            entries.append(
                self._format_public_history_entry(
                    "Tutor", artifact.turn_idx, artifact.tutor_visible_output
                )
            )
            entries.append(
                self._format_public_history_entry(
                    "Student", artifact.turn_idx, artifact.student_output
                )
            )
            summary = "\n\n".join(entry for entry in entries if entry)
        return StudentGeneralizationAnchor(
            public_history=PublicHistoryState(
                summary=summary,
                turn_count=turn_count,
                turns=list(artifact.public_history_after),
            ),
            previous_student_output=artifact.student_output,
            teacher_feedback=artifact.tutor_visible_output,
            reward_turn_idx=int(artifact.turn_idx),
        )

    @staticmethod
    def _is_student_generalization_reward_turn(artifact: TurnArtifact) -> bool:
        return (
            artifact.student_state is not None
            and not artifact.invalid_due_to_leak
        )

    def _student_generalization_anchor(
        self,
        episode_artifact: EpisodeArtifact,
        *,
        allow_unsuccessful: bool = False,
    ) -> StudentGeneralizationAnchor | None:
        success_turn = self._success_turn(episode_artifact)
        if success_turn is not None:
            return self._turn_generalization_anchor(success_turn)

        if (
            not allow_unsuccessful
            and getattr(self, "student_generalize_mode", "only_success")
            == "only_success"
        ):
            return None

        for artifact in reversed(episode_artifact.turns):
            if self._is_student_generalization_reward_turn(artifact):
                return self._turn_generalization_anchor(artifact)

        return StudentGeneralizationAnchor(
            public_history=PublicHistoryState(
                summary=self._build_initial_public_summary(
                    episode_artifact.initial_student_answer
                ),
                turn_count=0,
                turns=self._initial_conversation(
                    episode_artifact.initial_student_answer
                ),
            ),
            previous_student_output=episode_artifact.initial_student_answer,
            teacher_feedback="",
            reward_turn_idx=None,
        )

    async def _run_student_generalization(
        self,
        data: dict[str, Any],
        episode_artifact: EpisodeArtifact,
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> list[StudentGeneralizationResult]:
        if not bool(getattr(self, "student_generalize_enabled", False)):
            return []

        transfer_anchor = self._student_generalization_anchor(episode_artifact)
        original_anchor = (
            self._student_generalization_anchor(
                episode_artifact, allow_unsuccessful=True
            )
            if getattr(self, "student_generalize_retest_original", False)
            else None
        )

        cases = self._student_generalization_cases(data)
        results: list[StudentGeneralizationResult] = []
        for level in self._probe_levels():
            anchor = (
                original_anchor if level == ORIGINAL_RETEST_LEVEL else transfer_anchor
            )
            if anchor is None:
                results.append(
                    StudentGeneralizationResult(
                        level=level,
                        skipped=True,
                        skip_reason="base_not_solved",
                    )
                )
                continue
            case = cases.get(level)
            if case is None:
                results.append(
                    StudentGeneralizationResult(
                        level=level,
                        skipped=True,
                        skip_reason="missing_variant",
                        public_history=anchor.public_history.summary,
                        reward_turn_idx=anchor.reward_turn_idx,
                    )
                )
                continue

            probe_messages = self._build_student_probe_messages(
                episode_artifact=episode_artifact,
                anchor=anchor,
                level=level,
                transfer_task=case.task,
            )
            replays = max(1, int(getattr(self, "student_generalize_replays", 1)))
            replay_results = await asyncio.gather(
                *[
                    self._call_auxiliary_messages(
                        probe_messages,
                        aux_caller=aux_caller,
                        rid_prefix=(
                            f"student-transfer-{level}-"
                            f"{anchor.public_history.turn_count}-r{replay_idx}"
                        ),
                    )
                    for replay_idx in range(replays)
                ]
            )
            # The first successful attempt is canonical for logging and
            # confidence. Falling back to attempt 1 keeps traces and confidence
            # semantics unchanged at replays=1, while making sure a single failed
            # call does not discard the replays that did succeed.
            student_result = next(
                (result for result in replay_results if not result.error),
                replay_results[0],
            )
            if student_result.error:
                student_output = ""
                student_output_raw = ""
                student_error = student_result.error
            else:
                student_output_raw = student_result.text
                student_error = None
            student_output = _strip_reasoning_for_context(student_output_raw)
            judge_result = None
            correctness_reward = 0.0
            confidence = 0.0
            confidence_mean_logprob = None
            confidence_token_count = 0
            confidence_available = False
            confidence_reason = ""
            confidence_backend = (
                "auxiliary_api"
                if getattr(self, "student_generalize_confidence_enabled", False)
                else ""
            )
            confidence_reward = 0.0
            replay_correct = 0
            replay_scored = 0
            if student_error is None:
                judge_results = await asyncio.gather(
                    *[
                        self._score_answer_async(
                            case.task,
                            case.ground_truth,
                            result.text,
                            answer_judge_caller=answer_judge_caller,
                        )
                        for result in replay_results
                        if not result.error
                    ]
                )
                judge_result = judge_results[0]
                replay_scored = len(judge_results)
                replay_correct = sum(
                    1 for judged in judge_results if judged.correct
                )
                # Fraction correct, so a variant the student gets right 3 times
                # out of 4 is worth more than one it gets right once.
                correctness_reward = case.reward * (
                    replay_correct / max(replay_scored, 1)
                )
                if getattr(self, "student_generalize_confidence_enabled", False):
                    if not student_result.token_logprobs:
                        raise RuntimeError(
                            "student generalization API response did not include "
                            "token logprobs."
                        )
                    confidence_result = compute_answer_token_confidence(
                        student_result.token_logprobs,
                    )
                    confidence = confidence_result.confidence
                    confidence_mean_logprob = confidence_result.mean_logprob
                    confidence_token_count = confidence_result.token_count
                    confidence_available = confidence_result.available
                    confidence_reason = confidence_result.reason
                    if judge_result.correct:
                        confidence_reward = (
                            case.reward
                            * self.student_generalize_confidence_reward_scale
                            * confidence
                        )
            reward = correctness_reward + confidence_reward
            results.append(
                StudentGeneralizationResult(
                    level=level,
                    task=case.task,
                    ground_truth=case.ground_truth,
                    attempted=True,
                    student_output=student_output,
                    student_error=student_error,
                    judge_result=judge_result,
                    correctness_reward=correctness_reward,
                    replay_count=replay_scored,
                    replay_correct=replay_correct,
                    confidence=confidence,
                    confidence_mean_logprob=confidence_mean_logprob,
                    confidence_token_count=confidence_token_count,
                    confidence_available=confidence_available,
                    confidence_reason=confidence_reason,
                    confidence_backend=confidence_backend,
                    confidence_reward=confidence_reward,
                    reward=reward,
                    public_history=anchor.public_history.summary,
                    reward_turn_idx=anchor.reward_turn_idx,
                )
            )
        return results

    def _apply_student_generalization_rewards(
        self,
        turn_artifacts: list[TurnArtifact],
        assignments: list[Any],
        student_generalization_results: list[StudentGeneralizationResult],
    ) -> None:
        if not student_generalization_results:
            return
        success_idx = next(
            (
                idx
                for idx, artifact in enumerate(turn_artifacts)
                if not artifact.invalid_due_to_leak
                and artifact.judge_result is not None
                and artifact.judge_result.correct
            ),
            None,
        )
        assignment_by_turn_idx = {
            int(artifact.turn_idx): assignment
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        }
        for result in student_generalization_results:
            if not result.reward:
                continue
            assignment = None
            if result.reward_turn_idx is not None:
                assignment = assignment_by_turn_idx.get(int(result.reward_turn_idx))
            if assignment is None and success_idx is not None:
                assignment = assignments[success_idx]
            if assignment is None:
                continue
            correctness_reward = float(result.correctness_reward)
            confidence_reward = float(result.confidence_reward)
            if not correctness_reward and not confidence_reward and result.reward:
                correctness_reward = float(result.reward)
            if correctness_reward:
                key = f"student_generalize_{result.level}"
                assignment.reward_components[key] = (
                    assignment.reward_components.get(key, 0.0) + correctness_reward
                )
            if confidence_reward:
                key = f"student_generalize_{result.level}_confidence"
                assignment.reward_components[key] = (
                    assignment.reward_components.get(key, 0.0) + confidence_reward
                )
            assignment.reward += correctness_reward + confidence_reward

    async def _annotate_teacher_diversity(
        self, turn_artifacts: list[TurnArtifact]
    ) -> None:
        if not getattr(self, "teacher_diversity_enabled", False):
            return
        if len(turn_artifacts) < 2:
            return

        caller = getattr(self, "teacher_diversity_caller", None)
        if caller is None:
            raise RuntimeError("teacher diversity reward has no embedding caller.")
        texts = [artifact.tutor_visible_output.strip() for artifact in turn_artifacts]
        unique_texts = list(dict.fromkeys(text for text in texts if text))
        if not unique_texts:
            for artifact in turn_artifacts[1:]:
                artifact.teacher_similarity_error = "empty_teacher_output"
            return

        try:
            embeddings = await caller.embed(unique_texts)
            embedding_by_text = dict(zip(unique_texts, embeddings, strict=True))
            for index in range(1, len(turn_artifacts)):
                previous_text = texts[index - 1]
                current_text = texts[index]
                artifact = turn_artifacts[index]
                if not previous_text or not current_text:
                    artifact.teacher_similarity_error = "empty_teacher_output"
                    continue
                artifact.previous_teacher_similarity = cosine_similarity(
                    embedding_by_text[previous_text],
                    embedding_by_text[current_text],
                )
        except Exception as exc:
            setattr(exc, "_fatal_rollout_error", True)
            logger.error(
                "Teacher diversity embedding or similarity computation failed; "
                "stopping the experiment.",
                exc_info=True,
            )
            raise

    def _log_generalize_stats(
        self,
        *,
        solved: bool,
        student_generalization_results: list[StudentGeneralizationResult] | None,
    ) -> None:
        if not bool(getattr(self, "student_generalize_enabled", False)):
            return

        try:
            is_eval = bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:
            is_eval = False

        metrics: dict[str, float] = {}
        if is_eval:
            metrics["teacher_success"] = float(solved)

        results = student_generalization_results or []
        for level in self._probe_levels():
            level_result = next(
                (result for result in results if result.level == level), None
            )
            attempted = bool(level_result is not None and level_result.attempted)
            correct = bool(
                attempted
                and level_result is not None
                and level_result.judge_result is not None
                and level_result.judge_result.correct
            )
            # With replays > 1 the headline series is the fraction correct, which
            # carries far less student-resampling noise than a single coin flip.
            # The binary series tracks attempt 1 only, so it stays comparable
            # with runs recorded before replays existed.
            replay_count = int(getattr(level_result, "replay_count", 0) or 0)
            replay_correct = int(getattr(level_result, "replay_correct", 0) or 0)
            score = (
                replay_correct / replay_count if replay_count > 0 else float(correct)
            )
            metrics[f"student_{level}_attempted"] = float(attempted)
            metrics[f"student_{level}_success"] = float(score)
            metrics[f"student_{level}_success_binary"] = float(correct)
            if attempted:
                metrics[f"student_{level}_correct_given_attempted"] = float(score)
                metrics[f"student_{level}_correct_given_attempted_binary"] = float(
                    correct
                )
                if replay_count > 0:
                    metrics[f"student_{level}_replay_count"] = float(replay_count)
                if (
                    getattr(self, "student_generalize_confidence_enabled", False)
                    and level_result is not None
                ):
                    metrics[f"student_{level}_confidence"] = float(
                        level_result.confidence
                    )
                    metrics[f"student_{level}_confidence_available"] = float(
                        level_result.confidence_available
                    )
                    metrics[f"student_{level}_confidence_token_count"] = float(
                        level_result.confidence_token_count
                    )
                    metrics[f"student_{level}_confidence_reward"] = float(
                        level_result.confidence_reward
                    )
                    if level_result.confidence_mean_logprob is not None:
                        metrics[f"student_{level}_answer_mean_logprob"] = float(
                            level_result.confidence_mean_logprob
                        )
            elif level_result is not None and level_result.skipped:
                metrics[f"student_{level}_skipped"] = 1.0

        if metrics:
            _safe_generalize_scalar(**metrics)

    @staticmethod
    def _student_generalization_result_to_json(
        result: StudentGeneralizationResult,
    ) -> dict[str, Any]:
        judge = result.judge_result
        return {
            "level": result.level,
            "task": result.task,
            "ground_truth": result.ground_truth,
            "attempted": bool(result.attempted),
            "skipped": bool(result.skipped),
            "skip_reason": result.skip_reason,
            "student_output": result.student_output,
            "student_error": result.student_error,
            "judge_correct": bool(judge.correct) if judge is not None else False,
            "judge_feedback": judge.feedback if judge is not None else "",
            # judge_correct is attempt 1 only; these are the whole replay set the
            # reward is actually computed from.
            "replay_count": int(result.replay_count),
            "replay_correct": int(result.replay_correct),
            "correctness_reward": float(result.correctness_reward),
            "confidence": float(result.confidence),
            "confidence_mean_logprob": result.confidence_mean_logprob,
            "confidence_token_count": int(result.confidence_token_count),
            "confidence_available": bool(result.confidence_available),
            "confidence_reason": result.confidence_reason,
            "confidence_backend": result.confidence_backend,
            "confidence_reward": float(result.confidence_reward),
            "reward": float(result.reward),
            "reward_turn_idx": result.reward_turn_idx,
            "public_history": result.public_history,
        }

    @staticmethod
    def _teacher_pre_solve_result_to_json(
        result: TeacherPreSolveResult,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(result.enabled),
            "mode": result.mode,
            "verification_enabled": bool(result.verification_enabled),
            "accepted": bool(result.accepted),
            "raw_output": result.raw_output,
            "error": result.error,
            "attempt_count": len(result.attempts),
            "attempts": [
                {
                    "attempt": attempt.attempt,
                    "raw_output": attempt.raw_output,
                    "error": attempt.error,
                    "accepted": bool(attempt.accepted),
                    "judge_correct": (
                        bool(attempt.judge_result.correct)
                        if attempt.judge_result is not None
                        else False
                    ),
                    "judge_feedback": (
                        attempt.judge_result.feedback
                        if attempt.judge_result is not None
                        else ""
                    ),
                    "judge_raw_result": (
                        dict(attempt.judge_result.raw_result)
                        if attempt.judge_result is not None
                        else {}
                    ),
                }
                for attempt in result.attempts
            ],
        }

    def _task_context(self, task: str, ground_truth: str = "") -> str:
        """Task block appended to a system prompt, ped-compare style."""
        block = render_prompt(TASK_CONTEXT_TEMPLATE, task=task)
        if ground_truth and self.teacher_show_ground_truth:
            block += "\n\n" + render_prompt(
                TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE, ground_truth=ground_truth
            )
        return block

    @staticmethod
    def _render_conversation(
        turns: list[dict[str, str]], *, speaker: str
    ) -> list[dict[str, str]]:
        """Shared dialogue seen from one side: own turns assistant, other user."""
        rendered = []
        for turn in turns:
            is_own = turn["role"] == speaker
            content = turn["content"]
            # Environment feedback rides on the other side's turn and is only
            # ever shown to the teacher.
            env = turn.get("env") if speaker == "teacher" and not is_own else None
            if env:
                content = f"{content}\n\n{env}"
            rendered.append(
                {"role": "assistant" if is_own else "user", "content": content}
            )
        return rendered

    def _teacher_system_for_state(
        self, tutor_state: TutorTurnState, *, clean: bool = False
    ) -> str:
        """Teacher system prompt. ``clean`` drops the prompt-pool suffix, which is
        the only difference between the rollout prompt and the training prompt."""
        base = (
            self.teacher_system_prompt
            if clean
            else self._teacher_system_prompt_for_selection(
                tutor_state.teacher_prompt_selection
            )
        )
        system = (
            f"{base.rstrip()}\n\n"
            f"{self._task_context(tutor_state.task, tutor_state.ground_truth)}"
        )
        return self._append_teacher_pre_solve_context(
            system, tutor_state.teacher_pre_solve_result
        )

    @staticmethod
    def _append_guidance_tail(
        messages: list[dict[str, str]], instruction: str
    ) -> list[dict[str, str]]:
        """Put the instruction at the very end of the prompt.

        Placement is load-bearing, not cosmetic: the identical text in the system
        prompt is followed far less often than when it sits immediately before
        generation. Measured in analysis/hazard_20260806.
        """
        # str.format, not render_prompt: render_prompt is Jinja2, so a "{...}"
        # placeholder would pass through untouched and the instruction would be
        # silently dropped. Plain formatting also keeps the instruction text out
        # of a template engine, where a stray "{{" would be interpreted.
        tail = TEACHER_GUIDANCE_TAIL_TEMPLATE.format(instruction=instruction.strip())
        if messages and messages[-1]["role"] == "user":
            merged = f"{messages[-1]['content'].rstrip()}\n\n{tail}\n"
            return [*messages[:-1], {**messages[-1], "content": merged}]
        return [*messages, {"role": "user", "content": f"{tail}\n"}]

    def _build_tutor_messages(
        self,
        tutor_state: TutorTurnState,
        *,
        clean: bool = False,
        include_guidance: bool = True,
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": self._teacher_system_for_state(tutor_state, clean=clean),
            },
            *self._render_conversation(
                tutor_state.public_history.turns, speaker="teacher"
            ),
        ]
        guidance = tutor_state.guidance
        if include_guidance and guidance is not None:
            messages = self._append_guidance_tail(messages, guidance.instruction)
        return messages

    def _build_student_messages(
        self, state: StudentTurnState
    ) -> list[dict[str, str]]:
        system = self._student_system_prompt_for_state(state)
        system = f"{system.rstrip()}\n\n{self._task_context(state.task)}"
        turns = list(state.public_history.turns)
        latest_teacher_output = state.latest_tutor_visible_output.strip()
        if latest_teacher_output:
            turns.append({"role": "teacher", "content": latest_teacher_output})
        messages = [
            {"role": "system", "content": system},
            *self._render_conversation(turns, speaker="student"),
        ]
        behavior_prompt = self._student_turn_behavior_prompt(state)
        if behavior_prompt:
            if messages[-1]["role"] == "user":
                messages[-1] = {
                    **messages[-1],
                    "content": f"{messages[-1]['content'].rstrip()}\n\n{behavior_prompt}",
                }
            else:
                messages.append({"role": "user", "content": behavior_prompt})
        return messages

    def _clean_tutor_input_tokens(self, artifact: TurnArtifact) -> list[int]:
        tokenizer = (
            getattr(artifact.tutor_response, "tokenizer", None) or self.tokenizer
        )
        return apply_chat_template(
            tokenizer,
            self._clean_tutor_messages(artifact),
            enable_thinking=self.enable_thinking,
        )

    def _clean_tutor_messages(self, artifact: TurnArtifact) -> list[dict[str, str]]:
        """The prompt this turn is trained on: rollout messages with the
        prompt-pool suffix stripped from the system turn and any guidance
        instruction stripped from the tail.

        Without guidance this rebuild is byte-identical to reusing
        ``artifact.tutor_messages[1:]`` verbatim, because everything after the
        system turn is a pure function of the state.
        """
        return self._build_tutor_messages(
            artifact.tutor_state, clean=True, include_guidance=False
        )

    def _opd_skip_reason(self, artifact: TurnArtifact) -> str:
        """Why this turn is not eligible for on-policy distillation, or ""."""
        state = artifact.tutor_state
        if state.guidance is not None and state.guidance.kind == "prompt":
            # Defensive: the configs forbid running both arms at once, and the
            # teacher would otherwise be handed the instruction twice.
            return "prompt_arm"
        if self.opd_skip_guided_rows and state.guidance is not None:
            # This row is already trained on a rewritten prompt. Stacking a second
            # perturbation on it makes neither effect attributable.
            return "guided"
        if self.opd_skip_leaked_rows and (
            artifact.leak_result.leaked or artifact.invalid_due_to_leak
        ):
            return "leak"
        # Episodes terminate on success, so every turn the tutor produced was
        # preceded by a wrong student answer: the number of tutor turns that have
        # already failed is turn_idx - 1. The repair instruction asserts that the
        # previous message did not get through, so it is only truthful once that
        # count reaches the threshold.
        if int(state.turn_idx) - 1 < self.opd_min_prior_failed_turns:
            return "too_early"
        if not list(getattr(artifact.tutor_response, "output_tokens", []) or []):
            return "empty"
        return ""

    def _build_opd_prompt_tokens(
        self, turn_artifacts: list[TurnArtifact]
    ) -> list[list[int] | None]:
        """Prompt tokens for the instructed teacher, one entry per turn.

        The teacher sees exactly the training prompt plus the repair instruction.
        The trainer then teacher-forces the policy's own output tokens under this
        prompt and pulls the policy toward the resulting distribution, so the
        instruction's effect ends up in the weights rather than in the prompt.
        """
        if not getattr(self, "opd_enabled", False):
            return [None] * len(turn_artifacts)
        try:
            if bool(workflow_context.get().is_eval):
                return [None] * len(turn_artifacts)
        except Exception:  # noqa: BLE001 - no context outside a rollout worker
            pass

        prompts: list[list[int] | None] = []
        skip_counts: dict[str, int] = {}
        selected = 0
        for artifact in turn_artifacts:
            reason = self._opd_skip_reason(artifact)
            if not reason and self.opd_max_turns_per_episode and (
                selected >= self.opd_max_turns_per_episode
            ):
                reason = "episode_cap"
            if not reason:
                tokenizer = (
                    getattr(artifact.tutor_response, "tokenizer", None) or self.tokenizer
                )
                try:
                    messages = self._append_guidance_tail(
                        self._clean_tutor_messages(artifact), self.opd_instruction
                    )
                    tokens = apply_chat_template(
                        tokenizer, messages, enable_thinking=self.enable_thinking
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping OPD sample at turn %s after tokenization error: %s",
                        artifact.turn_idx,
                        exc,
                    )
                    reason = "tokenization"
                    tokens = []
                if not reason:
                    output_len = len(artifact.tutor_response.output_tokens)
                    if (
                        self.max_train_sample_tokens is not None
                        and len(tokens) + output_len > self.max_train_sample_tokens
                    ):
                        reason = "overlength"
            if reason:
                artifact.opd_skip_reason = reason
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
                prompts.append(None)
                continue
            artifact.opd_prompt_tokens = tokens
            prompts.append(tokens)
            selected += 1

        metrics = {
            "opd/selected_turns": float(selected),
            "opd/selected_ratio": float(selected / max(1, len(turn_artifacts))),
        }
        for reason in (
            "guided",
            "prompt_arm",
            "leak",
            "too_early",
            "empty",
            "episode_cap",
            "tokenization",
            "overlength",
        ):
            metrics[f"opd/skipped_{reason}"] = float(skip_counts.get(reason, 0))
        _safe_scalar(**metrics)
        return prompts

    async def _build_opd_prompt_tokens_async(
        self, turn_artifacts: list[TurnArtifact]
    ) -> list[list[int] | None]:
        if not getattr(self, "opd_enabled", False):
            return [None] * len(turn_artifacts)
        return await asyncio.to_thread(self._build_opd_prompt_tokens, turn_artifacts)

    def _build_world_model_examples(
        self, turn_artifacts: list[TurnArtifact]
    ) -> list[WorldModelExample | None]:
        if not getattr(self, "world_model_enabled", False) or bool(
            workflow_context.get().is_eval
        ):
            return [None] * len(turn_artifacts)

        examples = [
            self._build_world_model_example(artifact) for artifact in turn_artifacts
        ]
        valid_examples = [example for example in examples if example.valid]
        skip_counts: dict[str, int] = {}
        for example in examples:
            if example.skip_reason:
                skip_counts[example.skip_reason] = (
                    skip_counts.get(example.skip_reason, 0) + 1
                )
        metrics = {
            "world_model/valid_responses": float(len(valid_examples)),
            "world_model/valid_ratio": float(
                len(valid_examples) / max(1, len(examples))
            ),
            "world_model/prompt_tokens": float(
                sum(
                    len(example.input_ids) - sum(example.target_mask)
                    for example in valid_examples
                )
            ),
            "world_model/target_tokens": float(
                sum(sum(example.target_mask) for example in valid_examples)
            ),
        }
        for reason in (
            "student_error",
            "empty",
            "leak",
            "overlength",
            "tokenization",
        ):
            metrics[f"world_model/skipped_{reason}"] = float(skip_counts.get(reason, 0))
        _safe_scalar(**metrics)
        return [
            example if example.valid else WorldModelExample() for example in examples
        ]

    async def _build_world_model_examples_async(
        self, turn_artifacts: list[TurnArtifact]
    ) -> list[WorldModelExample | None]:
        if not getattr(self, "world_model_enabled", False) or bool(
            workflow_context.get().is_eval
        ):
            return [None] * len(turn_artifacts)
        return await asyncio.to_thread(self._build_world_model_examples, turn_artifacts)

    def _build_world_model_example(self, artifact: TurnArtifact) -> WorldModelExample:
        if artifact.student_error:
            return WorldModelExample(skip_reason="student_error")
        if artifact.invalid_due_to_leak or artifact.leak_result.leaked:
            return WorldModelExample(skip_reason="leak")
        if not artifact.student_output.strip():
            return WorldModelExample(skip_reason="empty")

        teacher_system_prompt = self._teacher_system_prompt_for_selection(
            artifact.tutor_state.teacher_prompt_selection
        )
        user_prompt = render_prompt(
            WORLD_MODEL_USER_TEMPLATE,
            teacher_system_prompt=teacher_system_prompt,
            teacher_user_prompt=artifact.tutor_prompt,
            teacher_visible_output=artifact.tutor_visible_output,
        )
        messages = [
            {"role": "system", "content": self.world_model_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenizer = (
            getattr(artifact.tutor_response, "tokenizer", None) or self.tokenizer
        )
        try:
            input_ids, target_mask = tokenize_teacher_forced_response(
                tokenizer,
                messages,
                artifact.student_output,
                enable_thinking=False,
            )
        except Exception as exc:
            logger.warning(
                "Skipping World Model sample at turn %s after tokenization error: %s",
                artifact.turn_idx,
                exc,
            )
            return WorldModelExample(skip_reason="tokenization")
        if (
            self.max_train_sample_tokens is not None
            and len(input_ids) > self.max_train_sample_tokens
        ):
            return WorldModelExample(skip_reason="overlength")
        return WorldModelExample(input_ids=input_ids, target_mask=target_mask)

    def _clean_teacher_input_token_reserve(
        self,
        rollout_messages: list[dict[str, str]],
        selection: PromptPoolSelection | None,
        guidance: TeacherGuidance | None = None,
    ) -> int:
        """Extra generation budget to hold back so the training sample still fits.

        Only a training prompt that is *longer* than the rollout prompt needs a
        reserve. Guidance only ever makes the training prompt shorter, so it
        contributes nothing here and is accepted purely so the caller does not
        have to special-case it.
        """
        del guidance
        if selection is None or getattr(self, "max_train_sample_tokens", None) is None:
            return 0
        rollout_input_len = len(
            apply_chat_template(
                self.tokenizer,
                rollout_messages,
                enable_thinking=self.enable_thinking,
            )
        )
        clean_input_len = len(
            apply_chat_template(
                self.tokenizer,
                [
                    {"role": "system", "content": self.teacher_system_prompt},
                    *rollout_messages[1:],
                ],
                enable_thinking=self.enable_thinking,
            )
        )
        return max(0, clean_input_len - rollout_input_len)

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
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        student_name: str = "",
        student_call_failed: bool = False,
        teacher_prompt_selection: PromptPoolSelection | None = None,
        student_prompt_selection: PromptPoolSelection | None = None,
        initial_student_turn_behavior: StudentTurnBehavior | None = None,
        inference_prompt_tokens: int = 0,
        training_prompt_tokens: int = 0,
    ) -> tuple[int, list[float]] | None:
        success_round = next(
            (
                trace.turn_idx
                for trace in traces
                if trace.judge_correct and not trace.invalid_due_to_leak
            ),
            0,
        )
        invalid_success_due_to_leak = sum(
            1 for trace in traces if trace.invalid_due_to_leak and trace.judge_correct
        )
        solved = success_round > 0
        is_eval = bool(workflow_context.get().is_eval)
        is_forced_persona_eval = bool(is_eval and student_prompt_selection is not None)
        completed_repeat_outcome = None
        if is_eval and not is_forced_persona_eval:
            completed_repeat_outcome = self._record_eval_repeat_outcomes(
                final_correct=pre_success or solved,
            )
        metrics = {
            "reward": float(total_reward),
            "turns": len(traces),
            "leaks": int(leak_count),
            "format_errors": sum(
                int(bool(getattr(trace, "tutor_format_error", None)))
                for trace in traces
            ),
            "invalid_success_due_to_leak": int(invalid_success_due_to_leak),
            "pre_solved": float(pre_success),
            "solved": float(solved),
            "final_correct": float(pre_success or solved),
            "stop/max_turns": float(termination_reason == "max_turns"),
            "stop/context_limit": float(
                termination_reason == CONTEXT_BUDGET_TERMINATION_REASON
            ),
            "stop/leak": float(termination_reason == LEAK_TERMINATION_REASON),
            "stop/teacher_pre_skipped": float(
                termination_reason == TEACHER_PRE_SKIPPED_TERMINATION_REASON
            ),
        }
        if teacher_pre_solve_result is not None:
            metrics["teacher_pre/verification_enabled"] = float(
                teacher_pre_solve_result.verification_enabled
            )
            metrics["teacher_pre/accepted"] = float(teacher_pre_solve_result.accepted)
            metrics["teacher_pre/attempts"] = float(
                len(teacher_pre_solve_result.attempts)
            )
        if success_round > 0:
            metrics["solve_turn"] = int(success_round)

        if getattr(self, "prompt_instruction_enabled", False):
            n = sum(
                1
                for trace in traces
                if trace.tutor_state.guidance is not None
                and trace.tutor_state.guidance.kind == "prompt"
            )
            metrics["prompt_instruction/turns"] = float(n)
            metrics["prompt_instruction/share_of_turns"] = float(
                n / max(1, len(traces))
            )

        if getattr(self, "guided_slots_enabled", False):
            guided = [
                trace.tutor_state.guidance.name
                for trace in traces
                if trace.tutor_state.guidance is not None
                and trace.tutor_state.guidance.kind == "move"
            ]
            metrics["guided/turns"] = float(len(guided))
            metrics["guided/episode_guided"] = float(bool(guided))
            for move in self.guided_slots_moves:
                # A move is credited with the episode only when it was the move
                # played on the guided turn, so these read as "when we forced this
                # move here, how often did the episode end up solved".
                played = move in guided
                prefix = f"guided_move/{move}"
                metrics[f"{prefix}/played"] = float(played)
                metrics[f"{prefix}/solved"] = float(played and success_round > 0)
                metrics[f"{prefix}/reward"] = float(total_reward) if played else 0.0
                metrics[f"{prefix}/leaks"] = float(leak_count) if played else 0.0

        if teacher_prompt_selection is not None:
            selected_source = teacher_prompt_selection.source
            metrics["prompt_schedule/warmup_probability"] = float(
                teacher_prompt_selection.warmup_probability
            )
            metrics["prompt_schedule/rollout_version"] = float(
                teacher_prompt_selection.rollout_version or 0
            )
            metrics["prompt_schedule/warmup_selected"] = float(
                selected_source == "warmup_full"
            )
            metrics["prompt_schedule/inference_prompt_tokens"] = float(
                inference_prompt_tokens
            )
            metrics["prompt_schedule/training_prompt_tokens"] = float(
                training_prompt_tokens
            )
            metrics["prompt_schedule/prompt_token_delta"] = float(
                inference_prompt_tokens - training_prompt_tokens
            )
            teacher_output_chars = sum(
                len(trace.tutor_visible_output) for trace in traces
            )
            for source in ("warmup_full", "pool", "pool_base"):
                source_selected = source == selected_source
                prefix = f"prompt_source/{source}"
                metrics[f"{prefix}/selected"] = float(source_selected)
                metrics[f"{prefix}/solved"] = float(
                    source_selected and success_round > 0
                )
                metrics[f"{prefix}/reward"] = (
                    float(total_reward) if source_selected else 0.0
                )
                metrics[f"{prefix}/leaked"] = float(source_selected and leak_count > 0)
                metrics[f"{prefix}/solve_turn"] = (
                    float(success_round) if source_selected else 0.0
                )
                metrics[f"{prefix}/teacher_output_chars"] = (
                    float(teacher_output_chars) if source_selected else 0.0
                )

        if is_forced_persona_eval:
            metrics.clear()
        if not is_forced_persona_eval:
            configured_student_names = list(
                getattr(self, "student_model_runtimes", {}).keys()
            )
            if not configured_student_names and student_name:
                configured_student_names = [student_name]
            for configured_name in configured_student_names:
                metric_name = self._student_metric_name(configured_name)
                metrics[f"student/{metric_name}/selected"] = float(
                    configured_name == student_name
                )
            if student_name:
                metric_name = self._student_metric_name(student_name)
                prefix = f"student/{metric_name}"
                metrics[f"{prefix}/solved"] = float(success_round > 0)
                metrics[f"{prefix}/pre_solved"] = float(pre_success)
                metrics[f"{prefix}/final_correct"] = float(pre_success or solved)
                metrics[f"{prefix}/reward"] = float(total_reward)
                metrics[f"{prefix}/turns"] = float(len(traces))
                metrics[f"{prefix}/call_failed"] = float(student_call_failed)

        student_prompt_pools = {
            "seen": getattr(self, "student_prompt_pool", ()),
            "heldout": getattr(self, "student_heldout_prompt_pool", ()),
        }
        if any(student_prompt_pools.values()):
            selected_index = (
                student_prompt_selection.index
                if student_prompt_selection is not None
                else None
            )
            selected_pool = (
                student_prompt_selection.pool or "seen"
                if student_prompt_selection is not None
                else None
            )
            for pool_name, prompt_pool in student_prompt_pools.items():
                if not prompt_pool:
                    continue
                pool_selected = pool_name == selected_pool
                metrics[f"student_prompt/{pool_name}/selected"] = float(pool_selected)
                for prompt_index in range(len(prompt_pool)):
                    is_selected = pool_selected and prompt_index == selected_index
                    metrics[f"student_prompt/{pool_name}/{prompt_index}/selected"] = (
                        float(is_selected)
                    )
            if selected_index is not None:
                prefixes = [f"student_prompt/{selected_pool}"]
                prefixes.append(f"student_prompt/{selected_pool}/{selected_index}")
                for prefix in prefixes:
                    metrics[f"{prefix}/solved"] = float(success_round > 0)
                    metrics[f"{prefix}/pre_solved"] = float(pre_success)
                    metrics[f"{prefix}/reward"] = float(total_reward)
                    metrics[f"{prefix}/turns"] = float(len(traces))
                    metrics[f"{prefix}/call_failed"] = float(student_call_failed)

        student_turn_behaviors = getattr(self, "student_turn_behaviors", ())
        if student_turn_behaviors:
            selected_behaviors = [
                behavior
                for behavior in (
                    initial_student_turn_behavior,
                    *(trace.student_turn_behavior for trace in traces),
                )
                if behavior is not None
            ]
            total_behavior_responses = len(selected_behaviors)
            metrics["student_turn_behavior/total_responses"] = float(
                total_behavior_responses
            )
            for behavior in student_turn_behaviors:
                selected_count = sum(
                    selected.name == behavior.name for selected in selected_behaviors
                )
                prefix = f"student_turn_behavior/{behavior.name}"
                metrics[f"{prefix}/selected_count"] = float(selected_count)
                metrics[f"{prefix}/selected_fraction"] = (
                    float(selected_count / total_behavior_responses)
                    if total_behavior_responses
                    else 0.0
                )

        if not is_forced_persona_eval:
            similarities = [
                similarity
                for trace in traces
                if (similarity := getattr(trace, "previous_teacher_similarity", None))
                is not None
            ]
            if getattr(self, "teacher_diversity_enabled", False):
                metrics["teacher_diversity/valid_pairs"] = float(len(similarities))
                metrics["teacher_diversity/errors"] = float(
                    sum(
                        bool(getattr(trace, "teacher_similarity_error", None))
                        for trace in traces
                    )
                )
                if similarities:
                    metrics["teacher_diversity/mean_similarity"] = float(
                        sum(similarities) / len(similarities)
                    )
            if getattr(self, "teacher_progress_judge_enabled", False):
                progress_results = [
                    trace.teacher_progress_judge_result for trace in traces
                ]
                valid_progress = [
                    result
                    for result in progress_results
                    if result is not None and result.score is not None
                ]
                metrics["teacher_progress_judge/valid_turns"] = float(
                    len(valid_progress)
                )
                metrics["teacher_progress_judge/errors"] = float(
                    len(progress_results) - len(valid_progress)
                )
                if valid_progress:
                    metrics["teacher_progress_judge/mean_score"] = float(
                        sum(int(result.score) for result in valid_progress)
                        / len(valid_progress)
                    )
                    metrics["teacher_progress_judge/mean_local_advantage"] = float(
                        sum(result.local_advantage for result in valid_progress)
                        / len(valid_progress)
                    )
            if getattr(self, "student_request_judge_enabled", False):
                request_results = [
                    trace.student_request_judge_result
                    for trace in traces
                    if trace.student_request_judge_result is not None
                ]
                valid_requests = [
                    result for result in request_results if result.score is not None
                ]
                student_questions = valid_requests
                request_errors = [
                    result for result in request_results if result.score is None
                ]
                metrics["student_request_judge/targeted_turns"] = float(
                    len(request_results)
                )
                metrics["student_request_judge/student_question_turns"] = float(
                    len(student_questions)
                )
                # Kept for dashboard compatibility. A targeted turn now always
                # corresponds to a separately generated student question.
                metrics["student_request_judge/no_question_turns"] = 0.0
                metrics["student_request_judge/errors"] = float(len(request_errors))
                if valid_requests:
                    metrics["student_request_judge/mean_score"] = float(
                        sum(int(result.score) for result in valid_requests)
                        / len(valid_requests)
                    )
                    metrics["student_request_judge/mean_reward"] = float(
                        sum(result.reward for result in valid_requests)
                        / len(valid_requests)
                    )
                if student_questions:
                    metrics["student_request_judge/appropriate_answer_rate"] = float(
                        sum(result.score == 1 for result in student_questions)
                        / len(student_questions)
                    )
            metrics.update(self._reward_component_metrics(traces))
            self._log_generalize_stats(
                solved=success_round > 0,
                student_generalization_results=student_generalization_results,
            )
        _safe_scalar(**metrics)
        return completed_repeat_outcome

    def _record_eval_repeat_outcomes(
        self, *, final_correct: bool
    ) -> tuple[int, list[float]] | None:
        task_id = getattr(workflow_context.get(), "task_id", None)
        if task_id is None:
            return None

        pending_by_task = getattr(self, "_eval_repeat_outcomes", None)
        if pending_by_task is None:
            pending_by_task = {}
            self._eval_repeat_outcomes = pending_by_task
        task_outcomes = pending_by_task.setdefault(int(task_id), [])
        task_outcomes.append(float(final_correct))

        repeat_count = int(getattr(self, "eval_repeat_count", 1))
        if len(task_outcomes) < repeat_count:
            return None
        completed = pending_by_task.pop(int(task_id))
        self._log_eval_repeat_metrics(completed)
        return int(task_id), completed

    @staticmethod
    def _log_eval_repeat_metrics(values: list[float]) -> None:
        mean = sum(values) / len(values)
        sample_variance = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            if len(values) > 1
            else 0.0
        )
        _safe_scalar(
            **{"repeat/final_correct/mean_task_sample_variance": sample_variance}
        )

        repeat_pairs = list(combinations(values, 2))
        if not repeat_pairs:
            repeat_pairs = [(values[0], values[0])]
        for left, right in repeat_pairs:
            if left or right:
                _safe_scalar(
                    **{
                        "repeat/final_correct/pairwise_success_jaccard": float(
                            left and right
                        )
                    }
                )

    async def _dump_eval_repeat_outcomes(
        self, task_id: int, outcomes: list[float]
    ) -> None:
        ctx = workflow_context.get()

        try:
            out_dir = Path(self.debug_trace_dir) / "eval" / "repeat_outcomes"
            await aiofiles.os.makedirs(out_dir, exist_ok=True)
            shard_path = out_dir / (
                f"{socket.gethostname()}_{os.getpid()}_final_correct.jsonl"
            )
            payload = {
                "task_id": int(task_id),
                "lora_version": getattr(ctx, "lora_version", None),
                "final_correct": [int(value) for value in outcomes],
            }
            async with aiofiles.open(shard_path, "a", encoding="utf-8") as outcome_file:
                await outcome_file.write(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        except Exception:
            logger.exception("Failed to dump tutor eval repeat outcomes.")

    def _enabled_reward_component_keys(self) -> list[str]:
        keys = []
        if (
            getattr(self, "success_reward", 0.0)
            or getattr(self, "early_success_bonus", 0.0)
            or getattr(self, "success_turn_shaping_enabled", False)
        ):
            keys.append("success")
        leak_penalty_mode = getattr(self, "leak_penalty_mode", "binary")
        if leak_penalty_mode == "staged":
            for key, attr in (
                ("leak_final_answer", "leak_penalty_final_answer"),
                ("leak_compute", "leak_penalty_compute"),
                ("leak_formula", "leak_penalty_formula"),
            ):
                if getattr(self, attr, 0.0):
                    keys.append(key)
        elif getattr(self, "leak_penalty", 0.0):
            keys.append("leak")
        if not getattr(self, "enable_thinking", False) and getattr(
            self, "format_error_penalty", 0.0
        ):
            keys.append("format_error")
        if getattr(self, "enable_turn_penalty", False) and getattr(
            self, "turn_penalty", 0.0
        ):
            keys.append("turn_penalty")
        if getattr(self, "length_penalty_threshold_chars", 0) > 0 and getattr(
            self, "length_penalty_per_100_chars", 0.0
        ):
            keys.append("length_penalty")
        if getattr(self, "student_generalize_enabled", False):
            rewards = getattr(self, "student_generalize_level_rewards", {}) or {}
            for level in self._probe_levels():
                if rewards.get(level, 0.0):
                    keys.append(f"student_generalize_{level}")
                    if getattr(self, "student_generalize_confidence_enabled", False):
                        keys.append(f"student_generalize_{level}_confidence")
        if getattr(self, "teacher_progress_judge_enabled", False):
            keys.append("teacher_progress_shaping")
        if getattr(self, "student_request_judge_enabled", False):
            keys.append("student_request_fulfillment")
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

    async def _maybe_dump_debug_trace(
        self,
        *,
        trajectory_id: int,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        latest_student_answer: str,
        total_reward: float,
        traces: list[TurnTrace],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        student_name: str = "",
        student_model: str = "",
        teacher_prompt_selection: PromptPoolSelection | None = None,
        student_prompt_selection: PromptPoolSelection | None = None,
        initial_student_turn_behavior: StudentTurnBehavior | None = None,
        initial_student_question_generation: (
            StudentQuestionGenerationResult | None
        ) = None,
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
            await aiofiles.os.makedirs(out_dir, exist_ok=True)
            file_path = out_dir / f"task_{task_id:08d}_{int(time.time() * 1000)}.json"
            payload = {
                "task_id": task_id,
                "trajectory_id": int(trajectory_id),
                "is_eval": bool(ctx.is_eval),
                "termination_reason": termination_reason,
                "total_reward": float(total_reward),
                "num_turns": len(traces),
                "pre_success": bool(pre_success),
                "leak_count": int(leak_count),
                "student": {
                    "name": student_name,
                    "model": student_model,
                },
                "prompt_pool": {
                    "teacher": (
                        asdict(teacher_prompt_selection)
                        if teacher_prompt_selection is not None
                        else None
                    ),
                    "student": (
                        asdict(student_prompt_selection)
                        if student_prompt_selection is not None
                        else None
                    ),
                },
                "student_turn_behavior": {
                    "initial": (
                        asdict(initial_student_turn_behavior)
                        if initial_student_turn_behavior is not None
                        else None
                    ),
                    "turns": [
                        (
                            asdict(trace.student_turn_behavior)
                            if trace.student_turn_behavior is not None
                            else None
                        )
                        for trace in traces
                    ],
                },
                "task": task,
                "ground_truth": ground_truth,
                "initial_student_answer": initial_student_answer,
                "latest_student_answer": latest_student_answer,
                "turns": [trace_to_json(trace) for trace in traces],
                "teacher_pre_solve": (
                    self._teacher_pre_solve_result_to_json(teacher_pre_solve_result)
                    if teacher_pre_solve_result is not None
                    else None
                ),
                "student_generalization": [
                    self._student_generalization_result_to_json(result)
                    for result in (student_generalization_results or [])
                ],
            }
            if initial_student_question_generation is not None:
                payload["initial_student_question_generation"] = asdict(
                    initial_student_question_generation
                )
            async with aiofiles.open(file_path, "w", encoding="utf-8") as trace_file:
                await trace_file.write(
                    json.dumps(payload, ensure_ascii=False, indent=2)
                )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")
