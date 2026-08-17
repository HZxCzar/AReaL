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
from dataclasses import asdict, dataclass, field, replace
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
    STUDENT_MODE_CODE,
    STUDENT_MODE_TEXT,
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
from examples.tutor.core.code_exec import (
    CodeSession,
    is_constant_print,
    is_valid_python,
)
from examples.tutor.core.confidence import compute_answer_token_confidence
from examples.tutor.core.generalization import load_student_generalize_bank
from examples.tutor.core.repetition import depth_metrics
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
    CODE_STUDENT_NO_OUTPUT,
    CODE_STUDENT_NO_PROGRAM,
    CODE_STUDENT_TEACHER_VIEW_TEMPLATE,
    FREE_CHAT_CODE_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT,
    FREE_CHAT_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_STUDENT_SYSTEM_PROMPT,
    FREE_CHAT_TEACHER_OPEN_PROMPT,
    FREE_CHAT_TEACHER_SOLVE_PROMPT,
    FREE_CHAT_TEACHER_SYSTEM_PROMPT,
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
    TEACHER_HISTORY_MASKED_TEMPLATE,
    TEACHER_MOVE_INSTRUCTIONS,
    TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE,
    TEACHER_REPAIR_INSTRUCTION,
    resolve_teacher_instruction,
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
FORMAT_TERMINATION_REASON = "format_error"
TEACHER_PRE_SKIPPED_TERMINATION_REASON = "pre_solve_skipped"
LEAK_HANDLING_MODES = {"disabled", "reward_only", "terminate"}
TEACHER_PRE_ON_REJECT_MODES = {"skip", "continue"}
# Live entries in the pre-solve cache. One step needs
# train_dataset.batch_size (16) of them, so this is deep enough that nothing is
# evicted while its group is still in flight; it exists only to stop the dict
# growing for the length of a 500-step run.
TEACHER_PRE_CACHE_MAX_ENTRIES = 256


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


_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.S)


def extract_program(text: str) -> str | None:
    """Pull a runnable program out of a code student's reply, or None.

    Takes the first fenced block if there is one, otherwise the whole reply, and
    returns it only if it parses.

    WHY NOT PREFILL THE REPLY. Forcing generation to open inside a ```python
    fence is the stronger enforcement, but it needs SGLang's
    `continue_final_message`, and student_models entries may point at any
    OpenAI-compatible endpoint, so the training path must not depend on a
    backend-specific request field. Prefill also measurably costs accuracy (0.25
    against 0.45 on a solo re-test probe) because it removes the planning tokens
    before the code.

    Extraction plus regeneration gets the same guarantee more cheaply: mid
    conversation this student writes prose with a code block appended, and the
    block is the part that matters. Prose alone does not parse, so the caller
    regenerates rather than executing English as Python -- which is what made a
    third of turns look like crashes in an earlier harness.
    """
    if not text:
        return None
    blocks = _CODE_FENCE_RE.findall(text)
    candidates = [*blocks, text.replace("```python", "").replace("```", "")]
    for candidate in candidates:
        if candidate.strip() and is_valid_python(candidate):
            return candidate
    return None


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
# The same re-test run on the transcript as leak-terminate training would have
# left it: everything up to the last completed round before the first leak. Only
# ever produced at evaluation, only when the training arm terminates, and never
# rewarded -- it exists so the in-the-wild number and the train-consistent number
# come off one rollout instead of two eval passes.
PRELEAK_RETEST_LEVEL = "original_preleak"
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
    # Per-turn credit from the prefix re-tests, {turn_idx: reward}. The values
    # sum to correctness_reward, so this changes only which turn gets paid, never
    # the episode total.
    turn_credits: dict[int, float] | None = None


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
    # 'text' or 'code'. See TutorStudentModelConfig.mode. Carried on the runtime
    # rather than looked up by name so every downstream branch reads one field.
    mode: str = STUDENT_MODE_TEXT


@dataclass(slots=True)
class SelectedStudent:
    name: str
    model: str
    caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller
    confidence_caller: ApiAuxiliaryCaller | None = None
    mode: str = STUDENT_MODE_TEXT

    @property
    def is_code(self) -> bool:
        return self.mode == STUDENT_MODE_CODE


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
        turn_local_reward_components: tuple[str, ...] | list[str] = (),
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
        free_chat: dict[str, Any] | None = None,
        prompt_instruction: dict[str, Any] | None = None,
        teacher_history_tags: str = "stripped",
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
        teacher_pre_on_reject: str = "skip",
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
        student_generalize_turn_credit: bool = False,
        student_generalize_turn_credit_replays: int = 0,
        format_handling_mode: str = "continue",
        student_generalize_retest_original: bool = False,
        student_generalize_level1_enabled: bool = True,
        student_generalize_level2_enabled: bool = True,
        student_generalize_level1_reward: float = 0.2,
        student_generalize_level2_reward: float = 0.5,
        student_generalize_retest_reward: float = 0.0,
        eval_preleak_retest: bool = False,
        student_generalize_confidence_enabled: bool = False,
        student_generalize_confidence_reward_scale: float = 0.25,
        eval_repeat_count: int = 1,
    ):
        self.eval_repeat_count = int(eval_repeat_count)
        if self.eval_repeat_count < 1:
            raise ValueError("eval_repeat_count must be >= 1.")
        self._eval_repeat_outcomes: dict[int, list[float]] = {}
        self.max_turns = max_turns
        # Resolved here rather than with the other reward settings because the
        # budget overwrites max_turns, and max_turns is read by everything below.
        free_chat_config = dict(free_chat or {})
        self.free_chat_enabled = bool(free_chat_config.get("enabled", False))
        self.free_chat_budget = 0
        if self.free_chat_enabled:
            budget = int(free_chat_config.get("budget", 0) or 0)
            if budget > 0:
                self.max_turns = budget
            if int(self.max_turns) < 1:
                raise ValueError("free_chat needs a budget of at least 1 round.")
            self.free_chat_budget = int(self.max_turns)
        self.free_chat_no_teaching_baseline = bool(
            free_chat_config.get("no_teaching_baseline", False)
        )
        # task_id -> fraction the student solves unaided. The student never
        # changes, so this is estimated once and reused; re-estimating per group
        # would put sampling noise straight into the reward.
        self._no_teaching_baselines: dict[str, float] = {}
        self._no_teaching_baseline_lock = asyncio.Lock()
        # (problem, weight version) -> the in-flight or finished pre-solve for
        # that group. Keyed on the version because the draft is a function of the
        # teacher's weights: sharing it across the group is the point, sharing it
        # across an update would be teaching from a stale solution.
        self._teacher_pre_solve_cache: dict[
            tuple[str, int], "asyncio.Future[TeacherPreSolveResult]"
        ] = {}
        self._teacher_pre_solve_lock = asyncio.Lock()
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
        self.turn_local_reward_components = tuple(turn_local_reward_components)
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
        self.opd_instruction, self.opd_instruction_name = (
            resolve_teacher_instruction(opd_config.get("instruction"))
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
        if teacher_history_tags not in {"stripped", "masked", "unmasked"}:
            raise ValueError(
                "teacher_history_tags must be 'stripped', 'masked', or 'unmasked'."
            )
        self.teacher_history_tags = teacher_history_tags
        prompt_instr = dict(prompt_instruction or {})
        self.prompt_instruction_enabled = bool(prompt_instr.get("enabled", False))
        self.prompt_instruction_text, self.prompt_instruction_name = (
            resolve_teacher_instruction(prompt_instr.get("instruction"))
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
        self.teacher_pre_on_reject = (teacher_pre_on_reject or "skip").strip()
        if self.teacher_pre_on_reject not in TEACHER_PRE_ON_REJECT_MODES:
            raise ValueError(
                "teacher_pre_on_reject must be 'skip' or 'continue', got "
                f"{self.teacher_pre_on_reject!r}."
            )
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
        self.student_generalize_turn_credit = bool(student_generalize_turn_credit)
        # 0 means "same as the final re-test", which keeps S(t) and S(T) measured
        # the same way so their difference carries no systematic bias.
        self.student_generalize_turn_credit_replays = (
            int(student_generalize_turn_credit_replays)
            or self.student_generalize_replays
        )
        if self.student_generalize_turn_credit and not getattr(
            self, "free_chat_no_teaching_baseline", False
        ):
            raise ValueError(
                "student_generalize.turn_credit requires "
                "free_chat.no_teaching_baseline: per-turn credit is "
                "S(t) - S(t-1) and S(0) is that baseline."
            )
        self.format_handling_mode = str(format_handling_mode or "continue")
        self.student_generalize_retest_original = bool(
            student_generalize_retest_original
        )
        self.student_generalize_level1_enabled = bool(student_generalize_level1_enabled)
        self.student_generalize_level2_enabled = bool(student_generalize_level2_enabled)
        self.student_generalize_level_rewards = {
            "level1": float(student_generalize_level1_reward),
            "level2": float(student_generalize_level2_reward),
        }
        self.student_generalize_retest_reward = float(
            student_generalize_retest_reward
        )
        self.eval_preleak_retest = bool(eval_preleak_retest)
        if self.free_chat_enabled:
            # Nothing inside a free-chat episode is scored, so without the
            # re-test the episode carries no reward at all and every rollout
            # group would sit at zero advantage.
            if not self.student_generalize_enabled:
                raise ValueError(
                    "free_chat requires student_generalize.enabled=true: the "
                    "re-test is the only reward in the episode."
                )
            if not self.student_generalize_retest_original:
                raise ValueError(
                    "free_chat requires student_generalize.retest_original=true."
                )
            if self.student_generalize_retest_reward <= 0.0:
                raise ValueError(
                    "free_chat requires student_generalize.retest_reward > 0."
                )
            # Reaching the budget is the normal ending here, not a failure, so a
            # non-zero max_turn_penalty would charge every single episode.
            if self.max_turn_penalty:
                raise ValueError(
                    "free_chat requires reward.max_turn_penalty=0.0; every "
                    "episode now ends by reaching the budget."
                )
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
                mode=str(student.get("mode", STUDENT_MODE_TEXT)),
            )
        return runtimes

    def _group_rng(
        self, role: str, group_key: str, rollout_version: int | None
    ) -> random.Random:
        """A draw that is constant across one problem's rollouts in one step.

        `gconfig.n_samples` rollouts of a problem form the GRPO group, and
        `actor.group_baseline='episode'` subtracts that group's mean return from
        every member. Anything drawn independently per rollout therefore lands
        directly in the advantage: with two students it was worth 29-33% of the
        within-group spread on 20260815_065439, which the teacher cannot control
        and cannot even see at turn 1, because it speaks first.

        Seeding on the problem holds the draw fixed across the group; including
        the weight version lets it be redrawn on the next step, so a problem is
        not pinned to one student for all 10.5 epochs of a 500-step run.

        A None version -- the external-client path, where there are no local
        weights -- collapses to a single bucket, so the draw is fixed per problem
        for the whole run. That is what you want there: the teacher behind an API
        does not drift between steps, so there is nothing for a version to track.
        """
        seed = getattr(self, "prompt_pool_seed", 0)
        version = int(rollout_version) if rollout_version is not None else -1
        return random.Random(f"{seed}:{role}:{group_key}:{version}")

    def _select_student(
        self,
        data: dict[str, Any],
        *,
        aux_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller,
        group_key: str = "",
        rollout_version: int | None = None,
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
                # Group-scoped, not per-rollout: see _group_rng. With one student
                # at positive weight this is the same draw either way, so every
                # single-student arm is unaffected.
                rng = self._group_rng(
                    "student", group_key or str(data.get("id") or ""), rollout_version
                )
                runtime = rng.choices(
                    runtimes,
                    weights=[item.weight for item in runtimes],
                    k=1,
                )[0]
            return SelectedStudent(
                name=runtime.name,
                model=runtime.model,
                caller=runtime.caller,
                confidence_caller=runtime.confidence_caller,
                mode=runtime.mode,
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
                name=self.prompt_instruction_name,
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
        # Identifies the GRPO group: every rollout of one problem shares it, and
        # it is what the pre-solve cache and the student draw are scoped to. Same
        # key the no-teaching baseline uses, so the two agree on what a problem is.
        group_key = str(data.get("id") or task)
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
        selected_student = self._select_student(
            data,
            aux_caller=aux_caller,
            group_key=group_key,
            rollout_version=episode_lora_version,
        )
        student_caller = selected_student.caller
        # One interpreter per episode, so names persist across the conversation's
        # turns. None for a text student, which is what switches _run_student and
        # the re-test over to the prose path.
        code_session = CodeSession() if selected_student.is_code else None
        student_generalize_caller = self._make_student_generalization_caller(
            aux_caller=student_caller,
            confidence_caller=selected_student.confidence_caller,
        )
        answer_judge_caller = self._make_answer_judge_caller(
            chat_caller=aux_chat_caller
        )
        teacher_pre_solve_result: TeacherPreSolveResult | None = None
        teacher_pre_cache_hit = False
        self.last_teacher_pre_solve_result = None
        if getattr(self, "teacher_pre_enabled", False):
            (
                teacher_pre_solve_result,
                teacher_pre_cache_hit,
            ) = await self._teacher_pre_solve_for_group(
                task,
                ground_truth,
                actor_caller=actor_caller,
                answer_judge_caller=answer_judge_caller,
                lora_version=episode_lora_version,
                group_key=group_key,
            )
            self.last_teacher_pre_solve_result = teacher_pre_solve_result
            # 'continue' teaches this group without a draft instead of dropping
            # it. The draft is shared now, so a rejection is a property of the
            # group rather than of one rollout: under 'skip' the whole group goes,
            # which is what makes the pre-solve filter a filter on problems.
            # _free_chat_preamble and _append_teacher_pre_solve_context both gate
            # on `accepted`, so passing an unaccepted result through is already
            # the no-draft prompt.
            if (
                not teacher_pre_solve_result.accepted
                and getattr(self, "teacher_pre_on_reject", "skip") == "skip"
            ):
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
                    teacher_pre_cache_hit=teacher_pre_cache_hit,
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
                    student_mode=selected_student.mode,
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
            student_mode=selected_student.mode,
        )
        free_chat = bool(getattr(self, "free_chat_enabled", False))
        if free_chat:
            # The teacher opens, so there is no pre-attempt to open with and no
            # pre-solve check either: nothing is judged until the re-test.
            initial_student_answer_raw = ""
            initial_student_error = None
        else:
            (
                initial_student_answer_raw,
                initial_student_error,
            ) = await self._run_student(
                initial_student_state,
                aux_caller=student_caller,
                code_session=code_session,
            )
        initial_student_answer = _strip_reasoning_for_context(
            initial_student_answer_raw
        )
        initial_judge_result = (
            self._unscored_judge_result()
            if free_chat
            else await self._score_answer_async(
                task,
                ground_truth,
                initial_student_answer_raw,
                answer_judge_caller=answer_judge_caller,
            )
        )
        (
            initial_student_answer,
            initial_effective_student_turn_behavior,
            initial_student_question_generation,
        ) = await self._maybe_append_student_question(
            initial_student_state,
            initial_student_answer,
            should_generate=(
                not free_chat
                and not initial_judge_result.correct
                and not initial_student_error
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
                student_mode=selected_student.mode,
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
                student_mode=selected_student.mode,
                teacher_prompt_selection=teacher_prompt_selection,
                student_prompt_selection=student_prompt_selection,
                initial_student_turn_behavior=initial_student_turn_behavior,
                initial_student_question_generation=(
                    initial_student_question_generation
                ),
            )
            return None

        if free_chat:
            # Empty history: the teacher's first turn is generated from its
            # system prompt alone.
            initial_turns = []
            initial_summary = ""
        else:
            initial_turns = self._initial_conversation(initial_student_answer)
            initial_turns[-1]["env"] = self._teacher_env_feedback(
                initial_judge_result, 0
            )
            initial_summary = self._build_initial_public_summary(
                initial_student_answer
            )
        public_history = PublicHistoryState(
            summary=initial_summary,
            turn_count=0,
            turns=initial_turns,
        )
        previous_tutor_visible_output = ""
        previous_tutor_raw_outputs: tuple[str, ...] = ()
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
                previous_tutor_raw_outputs=previous_tutor_raw_outputs,
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
            if (
                tutor_format_error
                and getattr(self, "format_handling_mode", "continue") == "terminate"
            ):
                # Stop here rather than handing the student an empty message. The
                # parse failure leaves nothing to say, and continuing writes a
                # blank assistant turn into the tutor's own history, from which
                # 88-98% of following turns are also malformed. The turn keeps its
                # reward.format_error_penalty; nothing else is added, so set that
                # penalty to at least |max_turn_penalty| or this becomes the cheap
                # exit from a losing episode.
                termination_reason = FORMAT_TERMINATION_REASON
                turn_artifacts.append(
                    TurnArtifact(
                        turn_idx=turn_idx,
                        tutor_state=tutor_state,
                        tutor_messages=list(self._build_tutor_messages(tutor_state)),
                        tutor_response=response,
                        tutor_raw_output=tutor_raw_output,
                        tutor_visible_output=tutor_visible_output,
                        leak_result=self._pending_leak_check_result(),
                        public_history_before=public_before,
                        public_history_after=public_before,
                        tutor_format_error=tutor_format_error,
                    )
                )
                break
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
                student_mode=selected_student.mode,
            )
            student_prompt = self._build_student_prompt_from_state(student_state)
            student_answer_raw, student_error = await self._run_student(
                student_state,
                aux_caller=student_caller,
                code_session=code_session,
            )
            student_answer = _strip_reasoning_for_context(student_answer_raw)
            # Free chat scores nothing mid-episode. A per-turn judge would buy
            # only metrics here, and it is also the thing that ends the episode
            # early, which is exactly what this rollout removes.
            judge_result = (
                self._unscored_judge_result()
                if free_chat
                else await self._score_answer_async(
                    task,
                    ground_truth,
                    student_answer_raw,
                    answer_judge_caller=answer_judge_caller,
                )
            )
            (
                student_answer,
                effective_student_turn_behavior,
                student_question_generation,
            ) = await self._maybe_append_student_question(
                student_state,
                student_answer,
                should_generate=(
                    not free_chat
                    and not judge_result.correct
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
            previous_tutor_raw_outputs = (
                *previous_tutor_raw_outputs,
                tutor_raw_output if not tutor_format_error else "",
            )
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
            student_mode=selected_student.mode,
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
            initial_student_turn_behavior=initial_student_turn_behavior,
            initial_student_question_generation=(
                initial_student_question_generation
            ),
        )
        # Computed here rather than inside the probe runner so it never travels
        # through instance state: one workflow serves every concurrent episode.
        # Cached per problem, so the other seven rollouts of this group get it free.
        episode_no_teaching_baseline = await self._no_teaching_baseline(
            data,
            student_mode=selected_student.mode,
            aux_caller=student_generalize_caller,
            answer_judge_caller=answer_judge_caller,
        )
        student_generalization_results = await self._run_student_generalization(
            data,
            episode_artifact,
            aux_caller=student_generalize_caller,
            answer_judge_caller=answer_judge_caller,
            no_teaching_baseline=episode_no_teaching_baseline,
            code_session=code_session,
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
            turn_local_components=getattr(self, "turn_local_reward_components", ()),
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
                local_reward=(
                    assignment.local_reward
                    if getattr(self, "turn_local_reward_components", ())
                    else None
                ),
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
            teacher_pre_cache_hit=teacher_pre_cache_hit,
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
            no_teaching_baseline=episode_no_teaching_baseline,
            training_prompt_tokens=sum(
                len(clean_input)
                if clean_input is not None
                else artifact.tutor_response.input_len
                for artifact, clean_input in zip(
                    turn_artifacts, clean_teacher_inputs, strict=True
                )
            ),
            code_stats=code_session.stats() if code_session is not None else None,
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
            student_mode=selected_student.mode,
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
            initial_student_turn_behavior=initial_student_turn_behavior,
            initial_student_question_generation=(
                initial_student_question_generation
            ),
            code_stats=code_session.stats() if code_session is not None else None,
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

    def _build_teacher_pre_solve_messages(
        self, *, task: str, ground_truth: str | None = None
    ) -> list[dict[str, str]]:
        """Context the pre-solve is generated in.

        Under free chat these are exactly the first two messages of the context
        the draft will then sit in, so the draft is generated where it lives
        instead of being written somewhere else and pasted in. NOTE this is a
        different context from the dataset filter's, which is what
        `teacher_pre.mode='filter_solver'` used to mean and no longer does here:
        the accept rate and stop/teacher_pre_skipped are not comparable with runs
        before this change.

        Outside free chat the clean solver context is unchanged.
        """
        if getattr(self, "free_chat_enabled", False):
            return [
                {
                    "role": "system",
                    "content": self._free_chat_teacher_system(task, ground_truth),
                },
                {"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT},
            ]
        return [
            {"role": "system", "content": FILTER_SOLVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_teacher_pre_solve_prompt(task=task),
            },
        ]

    async def _teacher_pre_solve_for_group(
        self,
        task: str,
        ground_truth: str,
        *,
        actor_caller: AReaLEngineActorCaller | ExternalActorCaller,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        lora_version: int | None,
        group_key: str,
    ) -> tuple[TeacherPreSolveResult, bool]:
        """One pre-solve per problem per step, shared by that problem's group.

        Every rollout used to run its own. At `gconfig.n_samples` 8 with
        `teacher_pre.attempts` 3 that is up to 8 drafts and 24 judge calls where
        one draft was wanted, and worse than the cost: each rollout of the group
        taught from a DIFFERENT private solution, so the eight episodes whose mean
        `actor.group_baseline` subtracts were not answering the same question. The
        draft is a function of (teacher weights, problem), so it is cached on
        exactly that pair -- shared across the group, and dropped by the next
        weight update rather than going stale.

        Returns the result and whether this rollout reused another's draft, which
        is what `teacher_pre/cache_hit` reports. Expect (n_samples - 1)/n_samples
        once this is working, lower only where a group straddles a weight update.
        """
        key = (group_key, int(lora_version) if lora_version is not None else -1)
        async with self._teacher_pre_solve_lock:
            pending = self._teacher_pre_solve_cache.get(key)
            cache_hit = pending is not None
            if pending is None:
                pending = asyncio.ensure_future(
                    self._run_teacher_pre_solve(
                        task,
                        ground_truth,
                        actor_caller=actor_caller,
                        answer_judge_caller=answer_judge_caller,
                        lora_version=lora_version,
                    )
                )
                self._teacher_pre_solve_cache[key] = pending
                while len(self._teacher_pre_solve_cache) > TEACHER_PRE_CACHE_MAX_ENTRIES:
                    # dicts are insertion-ordered, so this drops the oldest group.
                    self._teacher_pre_solve_cache.pop(
                        next(iter(self._teacher_pre_solve_cache))
                    )
        try:
            # Shielded because the group's other rollouts are waiting on this same
            # future: whichever rollout happened to create it must not take the
            # draft down with it if it is cancelled.
            result = await asyncio.shield(pending)
        except Exception:
            # Do not let one transient failure poison the whole group for this
            # step -- drop the entry so the next sibling retries.
            async with self._teacher_pre_solve_lock:
                if self._teacher_pre_solve_cache.get(key) is pending:
                    del self._teacher_pre_solve_cache[key]
            raise
        return result, cache_hit

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
        messages = self._build_teacher_pre_solve_messages(
            task=task, ground_truth=ground_truth
        )
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
        code_session: CodeSession | None = None,
    ) -> tuple[str, str | None]:
        """One student turn. Returns (visible_turn_text, error).

        A code student's turn is a program plus what running it produced, and that
        pair is what goes into the public history -- see
        CODE_STUDENT_TEACHER_VIEW_TEMPLATE for why one rendering serves both the
        teacher's view and the student's own.
        """
        messages = self._build_student_messages(state)
        rid = f"student-{state.public_history.turn_count}"
        if (
            getattr(state, "student_mode", STUDENT_MODE_TEXT) == STUDENT_MODE_CODE
            and code_session is not None
        ):
            program, _raw, _attempts, error = await self._call_student_for_program(
                messages, aux_caller=aux_caller, rid_prefix=rid
            )
            if program is None:
                if error and error != "no parseable program":
                    # An endpoint failure, not the student's doing.
                    return "", error
                code_session.note_missing_program()
                return CODE_STUDENT_NO_PROGRAM, None
            cell = await code_session.run(program)
            return (
                render_prompt(
                    CODE_STUDENT_TEACHER_VIEW_TEMPLATE,
                    program=program.strip(),
                    result=cell.output.strip() or CODE_STUDENT_NO_OUTPUT,
                ),
                None,
            )
        result = await self._call_auxiliary_messages(
            messages,
            aux_caller=aux_caller,
            rid_prefix=rid,
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

    async def _call_student_for_program(
        self,
        messages: list[dict[str, str]],
        *,
        aux_caller: Any,
        rid_prefix: str,
        retries: int = 3,
    ) -> tuple[str | None, str, int, str | None]:
        """Get one runnable program out of a code student.

        Returns (program, raw_reply, attempts_used, error). program is None when
        every attempt failed to yield parseable Python, which the caller records
        as a turn that produced nothing rather than executing prose.
        """
        raw = ""
        error: str | None = None
        for attempt in range(max(1, retries)):
            result = await self._call_auxiliary_messages(
                messages,
                aux_caller=aux_caller,
                rid_prefix=f"{rid_prefix}-a{attempt}",
            )
            if result.error:
                error = result.error
                continue
            raw = result.raw_text or result.text or ""
            error = None
            program = extract_program(raw)
            if program is not None:
                return program, raw, attempt + 1, None
        return None, raw, max(1, retries), error

    async def _code_student_answer(
        self,
        messages: list[dict[str, str]],
        *,
        session: CodeSession,
        aux_caller: Any,
        rid_prefix: str,
        keep: bool,
    ) -> tuple[str, str, str, str | None]:
        """One code-student act: write a program, run it, report what it produced.

        Returns (answer_text, program, status, error). `answer_text` is what gets
        judged or shown, and it is the program's OUTPUT -- so a code student's
        score is the same quantity as a text student's boxed answer, measured
        through a different channel and scored by the same judge.

        `keep` is False for the re-test and the no-teaching baseline: those run on
        a branch and must not change the session the conversation built.
        """
        program, _raw, _attempts, error = await self._call_student_for_program(
            messages, aux_caller=aux_caller, rid_prefix=rid_prefix
        )
        if program is None:
            return "", "", "crash", error or "no parseable program"
        result = await (session.run(program) if keep else session.peek(program))
        return result.output, program, result.status, None

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
        free_chat = bool(getattr(self, "free_chat_enabled", False))
        if existing_history:
            entries.append(existing_history)
        elif not free_chat:
            # Both this and the turns fallback below reconstruct the student's
            # opening attempt when the history is empty. Free chat has no opening
            # attempt -- the teacher speaks first into an empty history -- so
            # synthesising one writes a student turn that never happened into the
            # teacher's own context and into the re-test transcript.
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
        turns = list(old_public_history.turns)
        if not turns and not free_chat:
            turns = self._initial_conversation(previous_student_answer)
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
        free_chat = bool(getattr(self, "free_chat_enabled", False))
        code_student = (
            getattr(episode_artifact, "student_mode", STUDENT_MODE_TEXT)
            == STUDENT_MODE_CODE
        )
        if free_chat:
            # The same one-line system prompt the conversation used, so the
            # re-test is the same student. The task appears for the first time in
            # the final user turn below.
            system = (
                FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT
                if code_student
                else FREE_CHAT_STUDENT_SYSTEM_PROMPT
            )
        else:
            system = self._student_system_prompt_for_selection(
                episode_artifact.student_prompt_selection
            )
            system = (
                f"{system.rstrip()}\n\n"
                f"{self._task_context(episode_artifact.task)}"
            )
        if level in (ORIGINAL_RETEST_LEVEL, PRELEAK_RETEST_LEVEL):
            final_turn = (
                render_prompt(
                    FREE_CHAT_CODE_STUDENT_RETEST_TEMPLATE
                    if code_student
                    else FREE_CHAT_STUDENT_RETEST_TEMPLATE,
                    task=episode_artifact.task,
                )
                if free_chat
                else render_prompt(STUDENT_FINAL_SOLUTION_TEMPLATE)
            )
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

    def _transfer_probe_levels(self) -> tuple[str, ...]:
        """The transfer levels that are switched on, in their fixed order.

        Order matters: a 'train' sidecar stores cases positionally and
        _student_generalization_cases zips them against _STUDENT_GENERALIZE_LEVELS.
        """
        return tuple(
            level
            for level, on in (
                ("level1", getattr(self, "student_generalize_level1_enabled", True)),
                ("level2", getattr(self, "student_generalize_level2_enabled", True)),
            )
            if on
        )

    def _probe_levels(self) -> tuple[str, ...]:
        """Probe branches run after tutoring, each switched on separately.

        The re-test costs one student call and needs nothing from the bank;
        level1 and level2 each cost a call and each need a variant. They are
        independent because the common case is wanting the re-test without
        paying for variants nobody is measuring.
        """
        levels = self._transfer_probe_levels()
        if not getattr(self, "student_generalize_retest_original", False):
            return levels
        if self._preleak_retest_active():
            return (ORIGINAL_RETEST_LEVEL, PRELEAK_RETEST_LEVEL, *levels)
        return (ORIGINAL_RETEST_LEVEL, *levels)

    def _preleak_retest_active(self) -> bool:
        """Whether to also re-test the pre-leak prefix. Evaluation only.

        Training already re-tests the truncated transcript, because it really did
        terminate. This is for the eval pass that deliberately did not.
        """
        if not getattr(self, "eval_preleak_retest", False):
            return False
        try:
            return bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:  # noqa: BLE001 - no context outside a rollout worker
            return False

    @staticmethod
    def _unscored_judge_result() -> JudgeResult:
        """Placeholder for a student reply free chat deliberately does not judge.

        `correct=False` is what keeps the rest of the loop on its normal path:
        the success branch, the early break and the pre-solved short circuit are
        all driven by this flag, so none of them fire and the episode runs the
        full budget.
        """
        return JudgeResult(
            raw_output="",
            correct=False,
            feedback="",
            parse_error=None,
            raw_result={},
        )

    def _teacher_env_feedback(self, judge_result: Any, turn_idx: int) -> str:
        """Grading + budget the teacher sees, as environment feedback."""
        # Free chat judges nothing mid-episode, so there is no correctness to
        # report, and the budget is already in the teacher's system prompt.
        if getattr(self, "free_chat_enabled", False):
            return ""
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

    async def _score_code_output(
        self,
        task: str,
        ground_truth: str,
        output: str,
        *,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
    ) -> JudgeResult:
        """Score what a code student's program printed.

        There is no extraction step here, and that is the point. A text student
        marks its answer with \\boxed{}; a code student's answer is simply what it
        printed. Running the boxed extractor over program output returns an empty
        string, so the judge gets asked whether nothing equals the ground truth --
        which is why the first two-student run credited 0.3% of code re-tests when
        a crude match on the same outputs credited 10.9%.

        An exact match on the whole output short-circuits without a judge call.
        Everything else goes to the judge WITH THE OUTPUT VERBATIM and the ground
        truth beside it, which is the thing best placed to decide whether a
        program that printed a search log arrived at the right value. No harness
        guess about which line holds the answer.

        Empty output stays wrong, deliberately. A program that printed nothing has
        not answered, and that is the student's failure, not the harness's -- the
        teacher already sees "(no output)" in the result block and is the one
        positioned to tell it to print. Short-circuited so it costs no judge call.
        """
        text = str(output or "").strip()
        # Reuse the deterministic scorer for the clean case by presenting the whole
        # output as the answer. Exact, not a guess: it matches only when the
        # program printed the answer and nothing else.
        exact_result = self._score_answer(task, ground_truth, f"\\boxed{{{text}}}")
        if not text:
            return exact_result
        if exact_result.correct or not self.answer_judge_enabled:
            return exact_result

        # What the judge is shown IS the output, capped so a runaway log cannot
        # crowd out the ground truth in the prompt.
        shown = text if len(text) <= 2000 else text[:2000] + "\n...[truncated]"
        cache_key = (str(task), str(ground_truth), shown)
        cached = self._answer_judge_cache.get(cache_key)
        if cached is not None:
            return cached
        if answer_judge_caller is None:
            result = self._answer_judge_failed_result(
                exact_result, error="answer judge caller is unavailable"
            )
            self._answer_judge_cache[cache_key] = result
            return result

        judge_call = await self._call_auxiliary_prompt(
            system_prompt=self.answer_judge_system_prompt,
            user_prompt=self._build_answer_judge_prompt(task, ground_truth, shown),
            aux_caller=answer_judge_caller,
            rid_prefix="answer-judge-code",
        )
        if judge_call.error:
            result = self._answer_judge_failed_result(
                exact_result, error=judge_call.error
            )
        else:
            result = self._parse_answer_judge_result(exact_result, judge_call)
        self._answer_judge_cache[cache_key] = result
        return result

    async def _score_answer_async(
        self,
        task: str,
        ground_truth: str,
        student_answer: str,
        *,
        answer_judge_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None,
        code_output: bool = False,
    ) -> JudgeResult:
        if code_output:
            return await self._score_code_output(
                task,
                ground_truth,
                student_answer,
                answer_judge_caller=answer_judge_caller,
            )
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
                # 0.0 keeps the historical behaviour where the re-test is scored
                # and logged but never rewarded. free_chat sets it and makes this
                # the entire episode reward.
                reward=float(getattr(self, "student_generalize_retest_reward", 0.0)),
            )
            if self._preleak_retest_active():
                # reward 0.0, always: this is a diagnostic on an eval rollout and
                # must not move any advantage.
                cases[PRELEAK_RETEST_LEVEL] = StudentGeneralizationCase(
                    level=PRELEAK_RETEST_LEVEL,
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

        active_levels = self._transfer_probe_levels()
        for level, item in zip(_STUDENT_GENERALIZE_LEVELS, items, strict=False):
            if level not in active_levels:
                continue
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
        if not self._transfer_probe_levels():
            # Nothing reads the variants, so a row without them is not defective
            # and the rollout should not be thrown away.
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

    def _preleak_generalization_anchor(
        self, episode_artifact: EpisodeArtifact
    ) -> StudentGeneralizationAnchor | None:
        """The transcript as leak-terminate training would have left it.

        Terminate mode stops at the leaking turn and stores that turn with
        ``public_history_after = public_history_before``, so its re-test anchor is
        the last COMPLETED round before the leak. Here the episode ran on past the
        leak, so the same prefix is recovered by walking to the round before the
        first leaked one.

        None when the first round already leaked: terminate mode would have had no
        completed round to re-test, and the metric records that as a zero rather
        than inventing a transcript.
        """
        completed: TurnArtifact | None = None
        for artifact in episode_artifact.turns:
            if artifact.leak_result.leaked:
                break
            if self._is_student_generalization_reward_turn(artifact):
                completed = artifact
        if completed is None:
            # The first round leaked, so terminate mode would have left the
            # student with nothing. That is the empty transcript, not a zero.
            return self._empty_generalization_anchor(episode_artifact)
        return self._turn_generalization_anchor(completed)

    def _empty_generalization_anchor(
        self, episode_artifact: EpisodeArtifact
    ) -> StudentGeneralizationAnchor | None:
        """Re-test with no conversation at all: the student solves it alone.

        None only when the episode produced no turn whatsoever -- there is then
        nothing to attach a reward to and nothing that happened to measure.

        Worth knowing: on these episodes the score is the student's unaided
        ability on that problem, so they double as the no-teaching baseline for
        whichever problems they land on.
        """
        if not episode_artifact.turns:
            return None
        return StudentGeneralizationAnchor(
            public_history=PublicHistoryState(summary="", turn_count=0, turns=[]),
            previous_student_output="",
            teacher_feedback="",
            reward_turn_idx=int(episode_artifact.turns[-1].turn_idx),
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

        if getattr(self, "free_chat_enabled", False):
            # No completed round -- a leak terminated the very first turn. Re-test
            # on the empty transcript rather than scoring 0: "the student got no
            # help" is measurable, and assuming it is worth zero charges the
            # episode by the problem's difficulty instead of by the teacher's
            # behaviour. reward_turn_idx points at the one turn that exists, or
            # the attachment loop drops the result and the re-test contributes
            # nothing at all.
            return self._empty_generalization_anchor(episode_artifact)

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
        no_teaching_baseline: float | None = None,
        code_session: CodeSession | None = None,
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
        preleak_anchor = (
            self._preleak_generalization_anchor(episode_artifact)
            if self._preleak_retest_active()
            else None
        )

        cases = self._student_generalization_cases(data)
        results: list[StudentGeneralizationResult] = []
        # No leak means terminate mode would have run the identical conversation,
        # so its score is the in-the-wild score and re-running the probe would
        # only resample it. Reuse instead; only leaked episodes pay twice.
        episode_leaked = any(
            artifact.leak_result.leaked for artifact in episode_artifact.turns
        )
        for level in self._probe_levels():
            if level == PRELEAK_RETEST_LEVEL and not episode_leaked:
                original_result = next(
                    (r for r in results if r.level == ORIGINAL_RETEST_LEVEL), None
                )
                if original_result is not None:
                    results.append(
                        replace(
                            original_result,
                            level=PRELEAK_RETEST_LEVEL,
                            reward=0.0,
                            correctness_reward=0.0,
                            confidence_reward=0.0,
                            reward_turn_idx=None,
                        )
                    )
                    continue
            if level == ORIGINAL_RETEST_LEVEL:
                anchor = original_anchor
            elif level == PRELEAK_RETEST_LEVEL:
                anchor = preleak_anchor
            else:
                anchor = transfer_anchor
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
            if code_session is not None:
                # The re-test runs on a branch: `peek` so the conversation's
                # namespace is visible to the program but nothing the re-test does
                # is written back, and the replays stay independent of each other.
                # What gets judged is the program's OUTPUT, so the code student's
                # score is the same quantity as the text student's boxed answer
                # through the same judge.
                code_answers = await asyncio.gather(
                    *[
                        self._code_student_answer(
                            probe_messages,
                            session=code_session,
                            aux_caller=aux_caller,
                            rid_prefix=(
                                f"student-transfer-{level}-"
                                f"{anchor.public_history.turn_count}-r{replay_idx}"
                            ),
                            keep=False,
                        )
                        for replay_idx in range(replays)
                    ]
                )
                replay_results = [
                    TextCallResult(
                        # The raw program output. _score_answer_async is told it
                        # is program output and shows it to the judge verbatim.
                        text=answer if status == "ok" else "",
                        raw_text=program,
                        error=error,
                    )
                    for answer, program, status, error in code_answers
                ]
            else:
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
            turn_credits = None
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
                            code_output=code_session is not None,
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
                replay_fraction = replay_correct / max(replay_scored, 1)
                scored_fraction = replay_fraction
                if no_teaching_baseline is not None:
                    # The gain over no teaching, which is what the episode is
                    # being asked to produce. Can go negative: a conversation
                    # that leaves the student worse off than the bare problem
                    # statement should cost something.
                    scored_fraction = replay_fraction - no_teaching_baseline
                correctness_reward = case.reward * scored_fraction
                if (
                    getattr(self, "student_generalize_turn_credit", False)
                    and level == ORIGINAL_RETEST_LEVEL
                    and no_teaching_baseline is not None
                    and len(episode_artifact.turns) > 1
                    # Eval reports the episode metric and trains nothing, so the
                    # per-turn split has no consumer there. Skipping it keeps the
                    # eval pass at one re-test per episode instead of one per turn.
                    and not self._in_eval_rollout()
                ):
                    prefix_scores = await self._score_prefix_retests(
                        episode_artifact=episode_artifact,
                        anchor=anchor,
                        aux_caller=aux_caller,
                        answer_judge_caller=answer_judge_caller,
                    )
                    turn_credits = self._turn_credits_from_prefixes(
                        turn_artifacts=episode_artifact.turns,
                        prefix_scores=prefix_scores,
                        final_fraction=replay_fraction,
                        baseline=no_teaching_baseline,
                        scale=case.reward,
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
                    turn_credits=turn_credits,
                )
            )
        return results

    @staticmethod
    def _in_eval_rollout() -> bool:
        """True inside the eval pass; False when there is no rollout context."""
        try:
            return bool(getattr(workflow_context.get(), "is_eval", False))
        except Exception:  # noqa: BLE001 - no context outside a rollout worker
            return False

    async def _score_prefix_retests(
        self,
        *,
        episode_artifact: EpisodeArtifact,
        anchor: StudentGeneralizationAnchor,
        aux_caller: Any,
        answer_judge_caller: Any,
    ) -> dict[int, float]:
        """S(t) at every turn boundary except the last.

        Cuts the public history after the student's reply to turn t -- which is
        exactly ``public_history_after`` -- and runs the same solo re-test the
        episode ends with. The final boundary is skipped because it is the
        episode's own re-test, already measured.
        """
        replays = max(1, int(self.student_generalize_turn_credit_replays))
        prefixes = [
            (int(artifact.turn_idx), list(artifact.public_history_after or []))
            for artifact in episode_artifact.turns[:-1]
        ]
        prefixes = [(idx, history) for idx, history in prefixes if history]
        if not prefixes:
            return {}

        async def score(turn_idx, history):
            probe_anchor = StudentGeneralizationAnchor(
                public_history=PublicHistoryState(
                    summary=anchor.public_history.summary,
                    turn_count=len(history),
                    turns=history,
                ),
                previous_student_output=anchor.previous_student_output,
                teacher_feedback=anchor.teacher_feedback,
                reward_turn_idx=anchor.reward_turn_idx,
            )
            messages = self._build_student_probe_messages(
                episode_artifact=episode_artifact,
                anchor=probe_anchor,
                level=ORIGINAL_RETEST_LEVEL,
                transfer_task="",
            )
            replies = await asyncio.gather(
                *[
                    self._call_auxiliary_messages(
                        messages,
                        aux_caller=aux_caller,
                        rid_prefix=f"turn-credit-t{turn_idx}-r{replay_idx}",
                    )
                    for replay_idx in range(replays)
                ]
            )
            usable = [reply for reply in replies if not reply.error]
            if not usable:
                return turn_idx, None
            judged = await asyncio.gather(
                *[
                    self._score_answer_async(
                        episode_artifact.task,
                        episode_artifact.ground_truth,
                        reply.text,
                        answer_judge_caller=answer_judge_caller,
                    )
                    for reply in usable
                ]
            )
            return turn_idx, sum(1 for j in judged if j.correct) / len(judged)

        scored = await asyncio.gather(
            *[score(idx, history) for idx, history in prefixes]
        )
        return {idx: value for idx, value in scored if value is not None}

    def _turn_credits_from_prefixes(
        self,
        *,
        turn_artifacts: list[TurnArtifact],
        prefix_scores: dict[int, float],
        final_fraction: float,
        baseline: float,
        scale: float,
    ) -> dict[int, float]:
        """Split scale * (S(T) - S(0)) into per-turn marginals S(t) - S(t-1).

        The marginals telescope, so the episode reward is exactly what it was
        before this setting existed -- only the turn it lands on changes. A
        prefix whose re-test failed to score is carried forward rather than
        dropped, which keeps that property intact.
        """
        credits: dict[int, float] = {}
        previous = float(baseline)
        last_position = len(turn_artifacts) - 1
        for position, artifact in enumerate(turn_artifacts):
            turn_idx = int(artifact.turn_idx)
            if position == last_position:
                current = float(final_fraction)
            else:
                scored = prefix_scores.get(turn_idx)
                if scored is None:
                    continue
                current = float(scored)
            credits[turn_idx] = scale * (current - previous)
            previous = current
        return credits

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
            if result.turn_credits:
                # Each turn is paid for the re-test gain it produced. The credits
                # already sum to correctness_reward, so the episode total is
                # unchanged; only the allocation across turns differs. Checked
                # before the `not result.reward` guard below because the credits
                # can be non-zero while summing to zero.
                key = f"student_generalize_{result.level}"
                for turn_idx, credit in result.turn_credits.items():
                    credited = assignment_by_turn_idx.get(int(turn_idx))
                    if credited is None or not credit:
                        continue
                    credited.reward_components[key] = (
                        credited.reward_components.get(key, 0.0) + float(credit)
                    )
                    credited.reward += float(credit)
                if result.confidence_reward and result.reward_turn_idx is not None:
                    tail = assignment_by_turn_idx.get(int(result.reward_turn_idx))
                    if tail is not None:
                        ckey = f"student_generalize_{result.level}_confidence"
                        tail.reward_components[ckey] = (
                            tail.reward_components.get(ckey, 0.0)
                            + float(result.confidence_reward)
                        )
                        tail.reward += float(result.confidence_reward)
                continue
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
        solved: float,
        student_generalization_results: list[StudentGeneralizationResult] | None,
    ) -> None:
        """``solved`` is the episode outcome score, not a flag: 0/1 in the
        answer-attempt loop, and the re-test fraction under free chat."""
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
        turns: list[dict[str, str]],
        *,
        speaker: str,
        own_turn_template: str | None = None,
        own_turn_raw_outputs: tuple[str, ...] | None = None,
    ) -> list[dict[str, str]]:
        """Shared dialogue seen from one side: own turns assistant, other user.

        ``own_turn_template`` re-wraps the speaker's OWN turns, and is how the
        teacher gets the shape of its replies back after `public_history`
        stripped the tags off them. It takes a single ``{visible}`` field.
        ``own_turn_raw_outputs`` takes precedence when an aligned non-empty raw
        reply exists; this is the teacher-only unmasked history. Only the teacher
        passes either argument. The student's view stays plain text because its
        state contains only the public transcript.
        """
        rendered = []
        own_turn_idx = 0
        for turn in turns:
            is_own = turn["role"] == speaker
            content = turn["content"]
            if is_own:
                raw_content = (
                    own_turn_raw_outputs[own_turn_idx]
                    if own_turn_raw_outputs is not None
                    and own_turn_idx < len(own_turn_raw_outputs)
                    else ""
                )
                own_turn_idx += 1
                if raw_content:
                    content = raw_content
                elif own_turn_template is not None:
                    content = own_turn_template.format(visible=content)
            # Environment feedback rides on the other side's turn and is only
            # ever shown to the teacher.
            env = turn.get("env") if speaker == "teacher" and not is_own else None
            if env:
                content = f"{content}\n\n{env}"
            rendered.append(
                {"role": "assistant" if is_own else "user", "content": content}
            )
        return rendered

    def _free_chat_teacher_system(
        self, task: str, ground_truth: str | None = None
    ) -> str:
        """Teacher system prompt for the free-chat rollout: the setting only.

        Budget, task, and what the student will be tested on, plus the
        ground-truth key when `teacher_show_ground_truth` is on. The task rides
        inside the first block instead of being appended by `_task_context`,
        because this is the only copy of it in the episode -- the student is not
        given the task until the re-test.

        Everything about how to REPLY has moved out of here and into
        FREE_CHAT_TEACHER_OPEN_PROMPT, one turn later: the format contract, the
        anti-leak clause and the adaptive clause are all directives about a
        message to the student, and the pre-solve reply is not one. See the
        prompts.py note on FREE_CHAT_TEACHER_SOLVE_PROMPT for why the ordering is
        the whole point. What is left here is state, not instruction, so it is
        also what the pre-solve call is conditioned on.

        Takes the task rather than a TutorTurnState because the pre-solve runs
        before any turn state exists.

        There is no prompt-pool suffix, so the rollout prompt and the training
        prompt are the same string and `clean` has nothing to drop.
        """
        system = render_prompt(
            FREE_CHAT_TEACHER_SYSTEM_PROMPT,
            budget=int(getattr(self, "free_chat_budget", 0) or self.max_turns),
            task=task,
        )
        if ground_truth and self.teacher_show_ground_truth:
            system = f"{system}\n\n" + render_prompt(
                TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE,
                ground_truth=ground_truth,
            )
        return system

    def _free_chat_open_prompt(self) -> str:
        """The user turn that starts the conversation and carries its directives.

        Same blocks in the same order they had at the end of the system prompt,
        so the only thing that changed is which turn they are in.
        """
        parts = [FREE_CHAT_TEACHER_OPEN_PROMPT]
        if not self.enable_thinking:
            parts.append(NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT)
        if getattr(self, "teacher_anti_leak_instruction_enabled", False):
            parts.append(TEACHER_ANTI_LEAK_INSTRUCTION)
        if getattr(self, "teacher_adaptive_instruction_enabled", False):
            parts.append(TEACHER_ADAPTIVE_INSTRUCTION)
        return "\n\n".join(parts)

    def _free_chat_preamble(self, tutor_state: TutorTurnState) -> list[dict[str, str]]:
        """Everything before the first teacher reply, after the system turn.

        Two messages when the pre-solve ran and was accepted -- the request and
        the draft that answered it -- then the turn that opens the conversation.
        With `teacher_pre.enabled` off the first two are simply absent and
        nothing else moves, which is what makes the no-pre-solve arm a control
        rather than a different prompt.
        """
        messages: list[dict[str, str]] = []
        pre_solve = tutor_state.teacher_pre_solve_result
        if (
            getattr(self, "teacher_pre_enabled", False)
            and pre_solve is not None
            and pre_solve.accepted
        ):
            raw_output = _strip_reasoning_for_context(
                str(pre_solve.raw_output or "")
            ).strip()
            if raw_output:
                messages.append(
                    {"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT}
                )
                messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": self._free_chat_open_prompt()})
        return messages

    def _teacher_system_for_state(
        self, tutor_state: TutorTurnState, *, clean: bool = False
    ) -> str:
        """Teacher system prompt. ``clean`` drops the prompt-pool suffix, which is
        the only difference between the rollout prompt and the training prompt."""
        if getattr(self, "free_chat_enabled", False):
            return self._free_chat_teacher_system(
                tutor_state.task, tutor_state.ground_truth
            )
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
    def _append_guidance_to_system(system: str, instruction: str) -> str:
        """Put a per-reply instruction at the end of the system prompt.

        It cannot go at the end of the message list. The teacher's prompt is a
        real conversation -- its own turns are assistant, the student's are user
        -- so appending a directive there puts words in the student's mouth, and
        from turn 3 on some student turns would carry an instruction while
        earlier ones did not. The system turn is where a directive belongs.

        An earlier measurement found tail placement followed more reliably than
        the system prompt, but that was taken against the single-user-message
        state template, where the "tail" is the end of a state description rather
        than someone's utterance. Generation here uses the chat format, so that
        result does not transfer and the effect size measured under it is not
        guaranteed to carry over.
        """
        # str.format, not render_prompt: render_prompt is Jinja2, so a "{...}"
        # placeholder would pass through untouched and the instruction would be
        # silently dropped. Plain formatting also keeps the instruction text out
        # of a template engine, where a stray "{{" would be interpreted.
        block = TEACHER_GUIDANCE_TAIL_TEMPLATE.format(instruction=instruction.strip())
        return f"{system.rstrip()}\n\n{block}"

    def _build_tutor_messages(
        self,
        tutor_state: TutorTurnState,
        *,
        clean: bool = False,
        include_guidance: bool = True,
        guidance_override: TeacherGuidance | None = None,
    ) -> list[dict[str, str]]:
        """Messages for one teacher turn.

        ``guidance_override`` lets a caller ask for an instruction the rollout did
        not carry, which is how the OPD teacher is built: the row is unguided, and
        the teacher differs from it only by this instruction.

        Under free chat a preamble sits between the system turn and the
        conversation: the pre-solve exchange when there is one, then the turn that
        opens the conversation. It is a pure function of the state, so it is
        identical in the rollout prompt and in the training prompt, and it lands
        entirely on the prompt side of the split -- the draft is never trained on.
        """
        system = self._teacher_system_for_state(tutor_state, clean=clean)
        guidance = (
            guidance_override if guidance_override is not None else tutor_state.guidance
        )
        if include_guidance and guidance is not None:
            system = self._append_guidance_to_system(system, guidance.instruction)
        preamble = (
            self._free_chat_preamble(tutor_state)
            if getattr(self, "free_chat_enabled", False)
            else []
        )
        return [
            {"role": "system", "content": system},
            *preamble,
            *self._render_conversation(
                tutor_state.public_history.turns,
                speaker="teacher",
                own_turn_template=(
                    TEACHER_HISTORY_MASKED_TEMPLATE
                    if getattr(self, "teacher_history_tags", "stripped")
                    in {"masked", "unmasked"}
                    else None
                ),
                own_turn_raw_outputs=(
                    tutor_state.previous_tutor_raw_outputs
                    if getattr(self, "teacher_history_tags", "stripped")
                    == "unmasked"
                    else None
                ),
            ),
        ]

    def _build_student_messages(
        self, state: StudentTurnState
    ) -> list[dict[str, str]]:
        code_student = (
            getattr(state, "student_mode", STUDENT_MODE_TEXT) == STUDENT_MODE_CODE
        )
        if getattr(self, "free_chat_enabled", False):
            # No task, no subject, no instruction about what to do. The student
            # only ever learns what this is about from what the teacher says,
            # which is the point: telling it to solve the task on every turn is
            # what made every reply an answer attempt.
            system = (
                FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT
                if code_student
                else FREE_CHAT_STUDENT_SYSTEM_PROMPT
            )
        else:
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
        # A behaviour prompt is prose telling the student how to reply, which is
        # meaningless to one whose only reply is a program, and it would be a
        # second instruction competing with the format the channel enforces.
        behavior_prompt = (
            "" if code_student else self._student_turn_behavior_prompt(state)
        )
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
        # already failed is turn_idx - 1. The gate exists because the repair
        # instruction asserts that the previous message did not get through, and
        # that is only true once the count reaches the threshold. An instruction
        # that makes no claim about the history sets this to 0 and supervises from
        # turn 1, which is where most episodes actually end.
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
                    # Exactly the training prompt plus the instruction, so the
                    # teacher and the policy differ by nothing else.
                    messages = self._build_tutor_messages(
                        artifact.tutor_state,
                        clean=True,
                        guidance_override=TeacherGuidance(
                            kind="opd",
                            name=self.opd_instruction_name,
                            instruction=self.opd_instruction,
                        ),
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

    async def _no_teaching_baseline(
        self,
        data: dict[str, Any],
        *,
        aux_caller: Any,
        answer_judge_caller: Any,
        student_mode: str = STUDENT_MODE_TEXT,
    ) -> float | None:
        """What this problem is worth with no teaching, cached per problem AND mode.

        The same messages the re-test uses, with an empty transcript: the student
        gets the task and nothing else. None when the setting is off or every
        student call failed, and the caller then leaves the reward alone rather
        than subtracting a fabricated zero.

        KEYED BY MODE, not by problem alone. A code student and a text student
        solve the same problem unaided at different rates, so one shared number
        would charge one of them the other's baseline and be measured as teaching.
        """
        if not getattr(self, "free_chat_no_teaching_baseline", False):
            return None
        task = str(data.get("task", ""))
        ground_truth = str(data.get("ground_truth", ""))
        code_student = student_mode == STUDENT_MODE_CODE
        key = f"{data.get('id', task)}::{student_mode}"
        cached = self._no_teaching_baselines.get(key)
        if cached is not None:
            return cached
        async with self._no_teaching_baseline_lock:
            # Re-check: another episode of the same group may have filled it in
            # while this one waited.
            cached = self._no_teaching_baselines.get(key)
            if cached is not None:
                return cached
            # The same two messages _build_student_probe_messages produces for an
            # empty transcript. It has to be the same prompt: the baseline is
            # subtracted from that probe's score, so any difference here would be
            # measured as teaching.
            if getattr(self, "free_chat_enabled", False):
                system = (
                    FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT
                    if code_student
                    else FREE_CHAT_STUDENT_SYSTEM_PROMPT
                )
                final_turn = render_prompt(
                    FREE_CHAT_CODE_STUDENT_RETEST_TEMPLATE
                    if code_student
                    else FREE_CHAT_STUDENT_RETEST_TEMPLATE,
                    task=task,
                )
            else:
                system = self._student_system_prompt_for_selection(None)
                system = f"{system.rstrip()}\n\n{self._task_context(task)}"
                final_turn = render_prompt(STUDENT_FINAL_SOLUTION_TEMPLATE)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": final_turn},
            ]
            replays = max(1, int(getattr(self, "student_generalize_replays", 1)))
            if code_student:
                # Each replay gets its own empty session: no teaching means no
                # prior cells either, so the namespace starts bare.
                answers = await asyncio.gather(
                    *[
                        self._code_student_answer(
                            messages,
                            session=CodeSession(),
                            aux_caller=aux_caller,
                            rid_prefix=f"no-teaching-baseline-r{index}",
                            keep=False,
                        )
                        for index in range(replays)
                    ]
                )
                # A program that crashed or printed nothing is a failed attempt,
                # not a missing measurement: unaided, this student could not
                # produce an answer.
                texts = [
                    answer for answer, _program, status, _err in answers
                    if status == "ok"
                ]
                attempted = sum(
                    1 for _a, _p, _s, err in answers if err != "no parseable program"
                )
                if not attempted:
                    return None
                judged = await asyncio.gather(
                    *[
                        self._score_answer_async(
                            task,
                            ground_truth,
                            text,
                            answer_judge_caller=answer_judge_caller,
                            code_output=True,
                        )
                        for text in texts
                        if text.strip()
                    ]
                )
                correct = sum(1 for item in judged if item.correct)
                baseline = correct / max(attempted, 1)
                self._no_teaching_baselines[key] = baseline
                return baseline
            attempts = await asyncio.gather(
                *[
                    self._call_auxiliary_messages(
                        messages,
                        aux_caller=aux_caller,
                        rid_prefix=f"no-teaching-baseline-r{index}",
                    )
                    for index in range(replays)
                ]
            )
            usable = [attempt for attempt in attempts if not attempt.error]
            if not usable:
                return None
            judged = await asyncio.gather(
                *[
                    self._score_answer_async(
                        task,
                        ground_truth,
                        attempt.text,
                        answer_judge_caller=answer_judge_caller,
                    )
                    for attempt in usable
                ]
            )
            baseline = sum(1 for item in judged if item.correct) / len(judged)
            self._no_teaching_baselines[key] = baseline
            return baseline

    @staticmethod
    def _level_retest_score(
        results: list[StudentGeneralizationResult] | None, level: str
    ) -> float | None:
        """Fraction of re-test replays correct for one probe level.

        None when that level was not run at all, which is how the caller knows to
        leave the metric out rather than log a misleading zero.
        """
        for result in results or []:
            if result.level != level:
                continue
            if result.skipped or not result.attempted:
                return 0.0
            replay_count = int(getattr(result, "replay_count", 0) or 0)
            if replay_count > 0:
                replay_correct = int(getattr(result, "replay_correct", 0) or 0)
                return replay_correct / replay_count
            judge = result.judge_result
            return float(bool(judge is not None and judge.correct))
        return None

    @staticmethod
    def _free_chat_outcome_score(
        results: list[StudentGeneralizationResult] | None,
    ) -> float:
        """The episode's outcome in free chat: the solo re-test score.

        Nothing inside the conversation is judged, so `success_round` is always 0
        and every in-chat success series would read as a flat zero. The quantity
        that decides a free-chat episode is the same one evaluation reports -- the
        fraction of re-test replays the student got right -- so the batch mean of
        this is the re-test success rate.

        A skipped re-test scores 0. That happens when a leak terminated the very
        first turn, leaving no conversation that could have taught anything, so 0
        is the honest reading rather than a missing value.
        """
        for result in results or []:
            if result.level != ORIGINAL_RETEST_LEVEL:
                continue
            if result.skipped or not result.attempted:
                return 0.0
            replay_count = int(getattr(result, "replay_count", 0) or 0)
            if replay_count > 0:
                replay_correct = int(getattr(result, "replay_correct", 0) or 0)
                return replay_correct / replay_count
            judge = result.judge_result
            return float(bool(judge is not None and judge.correct))
        return 0.0

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
        teacher_pre_cache_hit: bool = False,
        student_name: str = "",
        student_call_failed: bool = False,
        teacher_prompt_selection: PromptPoolSelection | None = None,
        student_prompt_selection: PromptPoolSelection | None = None,
        initial_student_turn_behavior: StudentTurnBehavior | None = None,
        inference_prompt_tokens: int = 0,
        training_prompt_tokens: int = 0,
        no_teaching_baseline: float | None = None,
        code_stats: dict[str, int] | None = None,
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
        # What "this episode ended up solved" means. Identical to float(solved)
        # unless free chat is on, where nothing in the conversation is judged and
        # the re-test is the outcome. Every success series below uses this, so the
        # in-chat and re-test regimes cannot report different things under the same
        # metric name.
        outcome_score = float(solved)
        if getattr(self, "free_chat_enabled", False):
            outcome_score = self._free_chat_outcome_score(
                student_generalization_results
            )
        final_correct_score = max(float(pre_success), outcome_score)
        is_eval = bool(workflow_context.get().is_eval)
        is_forced_persona_eval = bool(is_eval and student_prompt_selection is not None)
        completed_repeat_outcome = None
        if is_eval and not is_forced_persona_eval:
            completed_repeat_outcome = self._record_eval_repeat_outcomes(
                final_correct=final_correct_score,
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
            "solved": outcome_score,
            "final_correct": final_correct_score,
            # Whatever the teacher gave away, the episode still counts here. The
            # two punished readings are added below.
            "final_correct_leak_gated": (
                0.0 if leak_count else final_correct_score
            ),
            "stop/max_turns": float(termination_reason == "max_turns"),
            "stop/context_limit": float(
                termination_reason == CONTEXT_BUDGET_TERMINATION_REASON
            ),
            "stop/leak": float(termination_reason == LEAK_TERMINATION_REASON),
            "stop/format_error": float(
                termination_reason == FORMAT_TERMINATION_REASON
            ),
            "stop/teacher_pre_skipped": float(
                termination_reason == TEACHER_PRE_SKIPPED_TERMINATION_REASON
            ),
        }
        # The code student's channel health. Present only for a code student, so
        # the series is never padded with zeros that mean "not measured".
        #
        # This is a VALIDITY check, not performance telemetry. A rise in the code
        # student's re-test score is ambiguous without it: either the teaching
        # improved, or the teacher learned to hand over an answer the student just
        # prints, in which case constant_prints climbs, the two students converge,
        # and the arm has stopped testing that their demands differ. The mirror
        # case is a fall in score that is really silent_cells -- a channel fault
        # rather than a teaching one. The two need opposite fixes.
        if code_stats is not None:
            turns = max(len(traces), 1)
            for name, value in code_stats.items():
                metrics[f"code/{name}"] = float(value)
                metrics[f"code/{name}_per_turn"] = float(value) / turns
            productive = turns - int(code_stats.get("crashes", 0)) - int(
                code_stats.get("silent_cells", 0)
            ) - int(code_stats.get("no_program", 0))
            # Turns that ran and actually told the teacher something.
            metrics["code/productive_per_turn"] = max(0.0, productive / turns)
        if teacher_pre_solve_result is not None:
            metrics["teacher_pre/verification_enabled"] = float(
                teacher_pre_solve_result.verification_enabled
            )
            metrics["teacher_pre/accepted"] = float(teacher_pre_solve_result.accepted)
            metrics["teacher_pre/attempts"] = float(
                len(teacher_pre_solve_result.attempts)
            )
            # Whether this rollout reused its group's draft instead of generating
            # its own. Expect (n_samples - 1)/n_samples = 0.875 at n_samples 8; a
            # flat 0.0 means the sharing is not happening and every rollout is
            # paying for its own pre-solve again.
            #
            # NOTE this changes how `attempts` above reads. A hit returns the
            # group's result object, so all n_samples rollouts report that one
            # draft's attempt count -- the series still means "attempts per
            # accepted draft", but it is no longer a count of generations this
            # rollout paid for. Multiply by (1 - cache_hit) for the call volume.
            metrics["teacher_pre/cache_hit"] = float(bool(teacher_pre_cache_hit))
        if success_round > 0:
            metrics["solve_turn"] = int(success_round)

        # What leak-terminate training would have scored on this same rollout.
        # Only present on an eval pass of a terminate arm, where the conversation
        # deliberately ran past the leak; absent otherwise, so the series is never
        # padded with zeros that mean "not measured".
        baseline = no_teaching_baseline
        if baseline is not None:
            in_the_wild = self._level_retest_score(
                student_generalization_results, ORIGINAL_RETEST_LEVEL
            )
            metrics["retest/no_teaching_baseline"] = float(baseline)
            if in_the_wild is not None:
                # The headline stays the raw fraction; this is the gain over a
                # student that was handed the problem and no conversation.
                metrics["retest/improvement"] = float(in_the_wild - baseline)
                metrics["retest/improved"] = float(in_the_wild > baseline)
                metrics["retest/made_it_worse"] = float(in_the_wild < baseline)

        preleak_score = self._level_retest_score(
            student_generalization_results, PRELEAK_RETEST_LEVEL
        )
        if preleak_score is not None:
            metrics["final_correct_preleak"] = preleak_score
            metrics["final_correct_preleak_delta"] = (
                final_correct_score - preleak_score
            )

        # Depth counters. The objective is that episodes where the student is
        # still wrong after two explanations still end solved -- depth/stuck_*
        # is that number, and the per-bucket hazard says where it is lost.
        # All COUNTS: divide the batch means, never average the per-episode
        # rate. See core/repetition.py.
        metrics.update(
            depth_metrics(
                [trace.tutor_visible_output for trace in traces],
                [bool(trace.judge_correct) for trace in traces],
            )
        )

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
                metrics[f"{prefix}/solved"] = outcome_score if played else 0.0
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
                metrics[f"{prefix}/solved"] = (
                    outcome_score if source_selected else 0.0
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
                metrics[f"{prefix}/solved"] = outcome_score
                metrics[f"{prefix}/pre_solved"] = float(pre_success)
                metrics[f"{prefix}/final_correct"] = final_correct_score
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
                    metrics[f"{prefix}/solved"] = outcome_score
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
                solved=outcome_score,
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
        student_mode: str = STUDENT_MODE_TEXT,
        teacher_prompt_selection: PromptPoolSelection | None = None,
        student_prompt_selection: PromptPoolSelection | None = None,
        initial_student_turn_behavior: StudentTurnBehavior | None = None,
        initial_student_question_generation: (
            StudentQuestionGenerationResult | None
        ) = None,
        code_stats: dict[str, int] | None = None,
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
                    # Recorded so trace analysis can split by action space instead
                    # of guessing it from the name string.
                    "mode": student_mode,
                },
                "code_session": code_stats,
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
