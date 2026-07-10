from __future__ import annotations

import asyncio
import json
import logging as py_logging
import os
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
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
from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_FIELD,
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
from examples.tutor.core.generation_budget import (
    CONTEXT_BUDGET_TERMINATION_REASON,
    ContextBudgetLimitExceeded,
)
from examples.tutor.core.history import (
    trace_to_history_record,
    trace_to_json,
)
from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.parsers import (
    parse_leak_check_result,
    parse_staged_leak_check_result,
)
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
    LeakHandlingMode,
    PromptPoolSelection,
    PublicHistoryState,
    StudentGeneralizeMode,
    StudentTurnState,
    TeacherPreSolveAttempt,
    TeacherPreSolveResult,
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
    EMPTY_PLACEHOLDER,
    FEEDBACK_LEAK_CHECK_SYSTEM_PROMPT_SUFFIX,
    FILTER_SOLVER_SYSTEM_PROMPT,
    FILTER_SOLVER_USER_TEMPLATE,
    INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
    LEAK_CHECK_DISABLED_FEEDBACK,
    LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE,
    LEAK_CHECK_NO_DETAIL_FEEDBACK,
    LEAK_CHECK_PENDING_FEEDBACK,
    LEAK_CHECK_USER_TEMPLATE,
    NO_PREVIOUS_VISIBLE_TUTORING_HISTORY,
    NO_VISIBLE_TUTORING_HISTORY,
    NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT,
    NONE_PLACEHOLDER,
    NONE_YET_PLACEHOLDER,
    POLARIS_FILTER_SOLVER_USER_TEMPLATE,
    POLARIS_INSTRUCTION,
    PRIVATE_LEAK_FEEDBACK_TEMPLATE,
    PRIVATE_LEAK_LEVEL_SUFFIX_TEMPLATE,
    PUBLIC_HISTORY_ENTRY_TEMPLATE,
    RAWBASE_LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    STAGED_LEAK_CHECK_USER_TEMPLATE,
    STUDENT_STATE_USER_TEMPLATE,
    STUDENT_TRANSFER_USER_TEMPLATE,
    TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE,
    TEACHER_STATE_USER_TEMPLATE,
    render_prompt,
)

logger = logging.getLogger("TutorWorkflow")

_REWARD_COMPONENT_ALIASES = {
    "success_credit": "success",
}

LEAK_TERMINATION_REASON = "leak"
TEACHER_PRE_SKIPPED_TERMINATION_REASON = "pre_solve_skipped"
LEAK_HANDLING_MODES = {"disabled", "reward_only", "terminate", "feedback"}


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


def _reward_component_key(name: str) -> str:
    return _REWARD_COMPONENT_ALIASES.get(name, name)


_STUDENT_GENERALIZE_LEVELS = ("level1", "level2")
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
        leaked_success_reward_scale: float = 1.0,
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
        teacher_prompt_pool_path: str = "",
        teacher_user_prompt_template: str | None = None,
        teacher_show_ground_truth: bool = False,
        teacher_pre_enabled: bool = False,
        teacher_pre_mode: str = "filter_solver",
        teacher_pre_attempts: int = 3,
        teacher_pre_max_tokens: int = 0,
        student_system_prompt: str = "",
        student_prompt_pool_path: str = "",
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
        student_generalize_path: str = "",
        student_generalize_level1_reward: float = 0.2,
        student_generalize_level2_reward: float = 0.5,
        student_generalize_confidence_enabled: bool = False,
        student_generalize_confidence_reward_scale: float = 0.25,
        pairwise_reward_enabled: bool = False,
        pairwise_reference_lag_steps: int = 5,
        pairwise_reward_scale: float = 0.05,
        pairwise_compare_all_turns: bool = True,
        pairwise_judge_both_incorrect: bool = True,
    ):
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
                "'reward_only', 'terminate', or 'feedback'."
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
        self.leaked_success_reward_scale = float(leaked_success_reward_scale)
        self.assign_success_reward = bool(assign_success_reward)
        self.outcome_prior_turn_weight = float(outcome_prior_turn_weight)
        self.outcome_credit_gamma = float(outcome_credit_gamma)
        self.early_success_bonus = float(early_success_bonus)
        self.enable_turn_penalty = bool(enable_turn_penalty)
        self.turn_penalty = float(turn_penalty)
        self.length_penalty_threshold_chars = int(length_penalty_threshold_chars)
        self.length_penalty_per_100_chars = float(length_penalty_per_100_chars)
        self.length_penalty_min = float(length_penalty_min)
        self.teacher_system_prompt = self._resolve_teacher_system_prompt(
            teacher_system_prompt
        )
        self.teacher_prompt_pool = load_prompt_pool(
            teacher_prompt_pool_path, role="teacher"
        )
        self.teacher_user_prompt_template = (
            teacher_user_prompt_template or TEACHER_STATE_USER_TEMPLATE
        ).strip()
        self.teacher_show_ground_truth = bool(teacher_show_ground_truth)
        self.teacher_pre_enabled = bool(teacher_pre_enabled)
        self.teacher_pre_mode = (teacher_pre_mode or "filter_solver").strip()
        if self.teacher_pre_mode != "filter_solver":
            raise ValueError("teacher_pre_mode must be 'filter_solver'.")
        self.teacher_pre_attempts = int(teacher_pre_attempts)
        if self.teacher_pre_attempts < 1:
            raise ValueError("teacher_pre_attempts must be >= 1.")
        self.teacher_pre_max_tokens = int(teacher_pre_max_tokens)
        self.student_system_prompt = student_system_prompt.strip()
        self.student_prompt_pool = load_prompt_pool(
            student_prompt_pool_path, role="student"
        )
        self.prompt_pool_seed = int(prompt_pool_seed)
        self._teacher_prompt_pool_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:teacher:fallback"
        )
        self._student_prompt_pool_fallback_rng = random.Random(
            f"{self.prompt_pool_seed}:student:fallback"
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
        self.student_generalize_path = student_generalize_path.strip()
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
                    "student generalization level rewards must be positive when "
                    "confidence reward is enabled."
                )
            if self.aux_mode != "api" and not self._student_model_configs:
                raise ValueError(
                    "student generalization confidence requires "
                    "auxiliary_model.mode='api' or a non-empty student_models API pool."
                )
        self.student_generalize_bank = (
            self._load_student_generalize_bank(self.student_generalize_path)
            if self.student_generalize_enabled and self.student_generalize_path
            else {}
        )
        self.pairwise_reward_enabled = bool(pairwise_reward_enabled)
        self.pairwise_reference_lag_steps = max(0, int(pairwise_reference_lag_steps))
        self.pairwise_reward_scale = float(pairwise_reward_scale)
        self.pairwise_compare_all_turns = bool(pairwise_compare_all_turns)
        self.pairwise_judge_both_incorrect = bool(pairwise_judge_both_incorrect)
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

    def _leak_check_system_prompt_for_current_mode(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if getattr(self, "leak_handling_mode", "reward_only") != "feedback":
            return prompt
        if FEEDBACK_LEAK_CHECK_SYSTEM_PROMPT_SUFFIX in prompt:
            return prompt
        return f"{prompt}\n\n{FEEDBACK_LEAK_CHECK_SYSTEM_PROMPT_SUFFIX}".strip()

    def _resolve_teacher_system_prompt(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if self.enable_thinking:
            return prompt
        if NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT in prompt:
            return prompt
        return (
            f"{prompt}\n\n{NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT}"
            if prompt
            else NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT
        ).strip()

    def _sample_prompt_pool(
        self, pool: tuple[str, ...], *, role: str
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
        index = rng.randrange(len(pool))
        return PromptPoolSelection(index=index, suffix=pool[index])

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

    def _extract_tutor_visible_output(self, raw_output: str) -> str:
        if getattr(self, "enable_thinking", False):
            return _strip_reasoning_for_context(raw_output)
        parsed, _ = parse_json_dict(raw_output)
        output = parsed.get("output") if isinstance(parsed, dict) else None
        if isinstance(output, str):
            return _strip_reasoning_for_context(output)
        return _strip_reasoning_for_context(raw_output)

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
        teacher_prompt_selection = self._sample_prompt_pool(
            getattr(self, "teacher_prompt_pool", ()), role="teacher"
        )
        student_prompt_selection = self._sample_prompt_pool(
            getattr(self, "student_prompt_pool", ()), role="student"
        )
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
                self._log_rollout_stats(
                    total_reward=0.0,
                    traces=[],
                    termination_reason=TEACHER_PRE_SKIPPED_TERMINATION_REASON,
                    pre_success=False,
                    leak_count=0,
                    teacher_pre_solve_result=teacher_pre_solve_result,
                    student_name=selected_student.name,
                )
                self._maybe_dump_debug_trace(
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

        initial_student_answer_raw, initial_student_error = await self._run_student(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(),
                previous_student_output="",
                latest_tutor_visible_output=INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
                student_prompt_selection=student_prompt_selection,
            ),
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
            )
            self._log_rollout_stats(
                total_reward=0.0,
                traces=[],
                termination_reason=episode_artifact.termination_reason,
                pre_success=episode_artifact.pre_success,
                leak_count=episode_artifact.leak_count,
                teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
                student_name=selected_student.name,
                student_call_failed=bool(initial_student_error),
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
                teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
                student_name=selected_student.name,
                student_model=selected_student.model,
                teacher_prompt_selection=teacher_prompt_selection,
                student_prompt_selection=student_prompt_selection,
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
                teacher_pre_solve_result=teacher_pre_solve_result,
                teacher_prompt_selection=teacher_prompt_selection,
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
            tutor_visible_output = self._extract_tutor_visible_output(tutor_raw_output)
            public_before = public_history.summary
            leak_result = self._pending_leak_check_result()
            if self.leak_handling_mode in {"terminate", "feedback"}:
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
                            tutor_prompt=self._build_tutor_prompt(tutor_state),
                            tutor_response=response,
                            tutor_raw_output=tutor_raw_output,
                            tutor_visible_output=tutor_visible_output,
                            leak_result=leak_result,
                            public_history_before=public_before,
                            public_history_after=public_before,
                        )
                    )
                    break

            student_state = StudentTurnState(
                task=task,
                public_history=public_history,
                previous_student_output=previous_student_output,
                latest_tutor_visible_output=tutor_visible_output,
                student_prompt_selection=student_prompt_selection,
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
            invalid_due_to_leak = (
                self.leak_handling_mode == "feedback" and leak_result.leaked
            )

            if judge_result.correct and not invalid_due_to_leak:
                termination_reason = "success"
            else:
                termination_reason = (
                    "max_turns" if turn_idx == self.max_turns else "continue"
                )

            if invalid_due_to_leak:
                next_public_history = PublicHistoryState(
                    summary=public_history.summary,
                    turn_count=public_history.turn_count,
                )
            else:
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
                    invalid_due_to_leak=invalid_due_to_leak,
                )
            )

            if invalid_due_to_leak:
                leak_feedback = self._private_leak_feedback(turn_idx, leak_result)
                leak_history = self._private_leak_history(turn_artifacts)
                previous_feedback = TutorPrivateFeedback(
                    kind="leak",
                    leak_feedback=leak_feedback,
                    leak_history=leak_history,
                )
            else:
                public_history = next_public_history
                previous_tutor_visible_output = tutor_visible_output
                previous_student_output = student_answer
                previous_feedback = TutorPrivateFeedback(
                    kind="student_judged",
                    student_output=student_answer,
                    judge_correct=judge_result.correct,
                    judge_feedback=judge_result.feedback,
                    leak_history=self._private_leak_history(turn_artifacts),
                )
            if judge_result.correct and not invalid_due_to_leak:
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
        )
        student_generalization_results = await self._run_student_generalization(
            data,
            episode_artifact,
            aux_caller=student_generalize_caller,
            answer_judge_caller=answer_judge_caller,
        )
        reward_computer = EpisodeRewardComputer(
            success_reward=self.success_reward,
            leak_penalty=self.leak_penalty,
            leak_penalty_mode=getattr(self, "leak_penalty_mode", "binary"),
            leak_penalty_final_answer=getattr(self, "leak_penalty_final_answer", None),
            leak_penalty_compute=getattr(self, "leak_penalty_compute", None),
            leak_penalty_formula=getattr(self, "leak_penalty_formula", None),
            leak_penalty_aggregation=getattr(self, "leak_penalty_aggregation", "turn"),
            leaked_success_reward_scale=getattr(
                self, "leaked_success_reward_scale", 1.0
            ),
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
                student_caller=student_caller,
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
        results = [
            response_to_tensordict(
                artifact.tutor_response,
                reward=assignment.reward,
                trajectory_id=trajectory_id,
                turn_idx=artifact.turn_idx,
                input_tokens_override=(
                    self._clean_tutor_input_tokens(artifact)
                    if artifact.tutor_state.teacher_prompt_selection is not None
                    else None
                ),
            )
            for artifact, assignment in zip(turn_artifacts, assignments, strict=True)
        ]
        total_reward = float(sum(assignment.reward for assignment in assignments))
        self.last_history = history
        self.last_traces = traces
        self.last_student_generalization_results = student_generalization_results
        self.last_total_reward = total_reward
        self._log_rollout_stats(
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
            student_generalization_results=student_generalization_results,
            teacher_pre_solve_result=episode_artifact.teacher_pre_solve_result,
            student_name=selected_student.name,
            student_model=selected_student.model,
            teacher_prompt_selection=teacher_prompt_selection,
            student_prompt_selection=student_prompt_selection,
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
        for attempt_idx in range(1, self.teacher_pre_attempts + 1):
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
                )

        return TeacherPreSolveResult(
            enabled=True,
            mode=self.teacher_pre_mode,
            accepted=False,
            attempts=attempts,
            raw_output="",
            error=(
                "no correct teacher pre-solve after "
                f"{self.teacher_pre_attempts} attempts"
            ),
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
            system_prompt=self._student_system_prompt_for_selection(
                state.student_prompt_selection
            ),
            user_prompt=prompt,
            aux_caller=aux_caller,
            rid_prefix=f"student-{state.public_history.turn_count}",
        )
        if result.error:
            return "", result.error
        return result.text, None

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
            system_prompt=self._leak_check_system_prompt_for_current_mode(
                self.leak_check_system_prompt
            ),
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
        teacher_message = self._extract_tutor_visible_output(teacher_action)
        prompt = render_prompt(
            RAWBASE_LEAK_CHECK_USER_TEMPLATE,
            ground_truth=ground_truth,
            teacher_action=teacher_message,
        )
        result = await self._call_auxiliary_prompt(
            system_prompt=self._leak_check_system_prompt_for_current_mode(
                RAWBASE_LEAK_CHECK_SYSTEM_PROMPT
            ),
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

    @staticmethod
    def _compact_private_feedback(value: Any) -> str:
        if isinstance(value, bool) or value is None:
            return ""
        if isinstance(value, str):
            text = value
        elif isinstance(value, (int, float)):
            text = str(value)
        elif isinstance(value, (list, tuple)):
            parts = [
                TutorAgentWorkflow._compact_private_feedback(item) for item in value
            ]
            text = ", ".join(part for part in parts if part)
        elif isinstance(value, dict):
            parts = [
                TutorAgentWorkflow._compact_private_feedback(item)
                for item in value.values()
            ]
            text = ", ".join(part for part in parts if part)
        else:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 240:
            return f"{text[:237].rstrip()}..."
        return text

    def _leak_feedback_text(self, leak_result: LeakCheckResult) -> str:
        feedback = self._compact_private_feedback(leak_result.feedback)
        feedback_lower = feedback.lower()
        if feedback and "failed" not in feedback_lower:
            return feedback
        return LEAK_CHECK_NO_DETAIL_FEEDBACK

    def _private_leak_feedback(
        self, turn_idx: int, leak_result: LeakCheckResult
    ) -> str:
        level = (
            PRIVATE_LEAK_LEVEL_SUFFIX_TEMPLATE.format(leak_level=leak_result.leak_level)
            if leak_result.leak_level is not None
            else ""
        )
        feedback = self._leak_feedback_text(leak_result)
        return PRIVATE_LEAK_FEEDBACK_TEMPLATE.format(
            turn_idx=turn_idx,
            level=level,
            feedback=feedback,
        )

    def _private_leak_history(self, turn_artifacts: list[TurnArtifact]) -> str:
        entries = [
            self._private_leak_feedback(artifact.turn_idx, artifact.leak_result)
            for artifact in turn_artifacts
            if artifact.invalid_due_to_leak
        ]
        return "\n".join(entries)

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
        return PublicHistoryState(
            summary="\n\n".join(entry for entry in entries if entry),
            turn_count=old_public_history.turn_count + 1,
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
            leak_feedback=feedback.leak_feedback or "",
            leak_history=feedback.leak_history or "",
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
            return f"{state.task}\n\n{POLARIS_INSTRUCTION}"
        return render_prompt(
            STUDENT_STATE_USER_TEMPLATE,
            task=state.task,
            public_history=state.public_history.summary
            or NO_PREVIOUS_VISIBLE_TUTORING_HISTORY,
            previous_student_output=state.previous_student_output or EMPTY_PLACEHOLDER,
            teacher_feedback=state.latest_tutor_visible_output or NONE_PLACEHOLDER,
        )

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
                task, ground_truth, student_answer
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

    @staticmethod
    def _load_student_generalize_bank(path: str) -> dict[str, Any]:
        if not path:
            return {}
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                f"student_generalize sidecar must be a JSON object: {file_path}"
            )
        return payload

    def _student_generalization_cases(
        self, data: dict[str, Any]
    ) -> dict[str, StudentGeneralizationCase]:
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
            return {}

        rewards = getattr(self, "student_generalize_level_rewards", {}) or {}
        cases: dict[str, StudentGeneralizationCase] = {}
        for level in _STUDENT_GENERALIZE_LEVELS:
            item = payload.get(level)
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

    def _success_turn(self, episode_artifact: EpisodeArtifact) -> TurnArtifact | None:
        if episode_artifact.termination_reason != "success":
            return None
        for artifact in episode_artifact.turns:
            if artifact.invalid_due_to_leak:
                continue
            if artifact.judge_result is not None and artifact.judge_result.correct:
                return artifact
        return None

    @staticmethod
    def _turn_generalization_anchor(
        artifact: TurnArtifact,
    ) -> StudentGeneralizationAnchor:
        turn_count = 0
        if artifact.student_state is not None:
            turn_count = artifact.student_state.public_history.turn_count + 1
        return StudentGeneralizationAnchor(
            public_history=PublicHistoryState(
                summary=artifact.public_history_after,
                turn_count=turn_count,
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
            and not artifact.leak_result.leaked
        )

    def _student_generalization_anchor(
        self, episode_artifact: EpisodeArtifact
    ) -> StudentGeneralizationAnchor | None:
        success_turn = self._success_turn(episode_artifact)
        if success_turn is not None:
            return self._turn_generalization_anchor(success_turn)

        if getattr(self, "student_generalize_mode", "only_success") == "only_success":
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

        anchor = self._student_generalization_anchor(episode_artifact)
        if anchor is None:
            return []

        cases = self._student_generalization_cases(data)
        results: list[StudentGeneralizationResult] = []
        for level in _STUDENT_GENERALIZE_LEVELS:
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

            transfer_prompt = self._build_student_transfer_prompt(
                original_task=episode_artifact.task,
                transfer_task=case.task,
                public_history=anchor.public_history,
                previous_student_output=anchor.previous_student_output,
                teacher_feedback=anchor.teacher_feedback,
            )
            student_result = await self._call_auxiliary_prompt(
                system_prompt=self._student_system_prompt_for_selection(
                    episode_artifact.student_prompt_selection
                ),
                user_prompt=transfer_prompt,
                aux_caller=aux_caller,
                rid_prefix=f"student-transfer-{level}-{anchor.public_history.turn_count}",
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
            if student_error is None:
                judge_result = await self._score_answer_async(
                    case.task,
                    case.ground_truth,
                    student_output_raw,
                    answer_judge_caller=answer_judge_caller,
                )
                if judge_result.correct:
                    correctness_reward = case.reward
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
        for level in _STUDENT_GENERALIZE_LEVELS:
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
            metrics[f"student_{level}_attempted"] = float(attempted)
            metrics[f"student_{level}_success"] = float(correct)
            if attempted:
                metrics[f"student_{level}_correct_given_attempted"] = float(correct)
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
        student_caller: ApiAuxiliaryCaller | AReaLEngineAuxiliaryCaller | None = None,
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
            run_student=lambda state: self._run_student(
                state, aux_caller=student_caller or aux_caller
            ),
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
            visible_output_extractor=self._extract_tutor_visible_output,
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
            {
                "role": "system",
                "content": self._append_prompt_pool_suffix(
                    self.teacher_system_prompt,
                    tutor_state.teacher_prompt_selection,
                ),
            },
            {"role": "user", "content": self._build_tutor_prompt(tutor_state)},
        ]

    def _clean_tutor_input_tokens(self, artifact: TurnArtifact) -> list[int]:
        tokenizer = (
            getattr(artifact.tutor_response, "tokenizer", None) or self.tokenizer
        )
        return apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": self.teacher_system_prompt},
                {"role": "user", "content": artifact.tutor_prompt},
            ],
            enable_thinking=self.enable_thinking,
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
        traces: list[TurnTrace],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        student_name: str = "",
        student_call_failed: bool = False,
    ) -> None:
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
        metrics = {
            "reward": float(total_reward),
            "turns": len(traces),
            "leaks": int(leak_count),
            "invalid_success_due_to_leak": int(invalid_success_due_to_leak),
            "pre_solved": float(pre_success),
            "solved": float(success_round > 0),
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
            metrics["teacher_pre/accepted"] = float(teacher_pre_solve_result.accepted)
            metrics["teacher_pre/attempts"] = float(
                len(teacher_pre_solve_result.attempts)
            )
        if success_round > 0:
            metrics["solve_turn"] = int(success_round)

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
            metrics[f"{prefix}/reward"] = float(total_reward)
            metrics[f"{prefix}/turns"] = float(len(traces))
            metrics[f"{prefix}/call_failed"] = float(student_call_failed)

        metrics.update(self._reward_component_metrics(traces))
        self._log_generalize_stats(
            solved=success_round > 0,
            student_generalization_results=student_generalization_results,
        )
        _safe_scalar(**metrics)

    def _enabled_reward_component_keys(self) -> list[str]:
        keys = []
        if getattr(self, "success_reward", 0.0) or getattr(
            self, "early_success_bonus", 0.0
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
        if getattr(self, "enable_turn_penalty", False) and getattr(
            self, "turn_penalty", 0.0
        ):
            keys.append("turn_penalty")
        if getattr(self, "length_penalty_threshold_chars", 0) > 0 and getattr(
            self, "length_penalty_per_100_chars", 0.0
        ):
            keys.append("length_penalty")
        if getattr(self, "pairwise_reward_enabled", False) and getattr(
            self, "pairwise_reward_scale", 0.0
        ):
            keys.append("pairwise")
        if getattr(self, "student_generalize_enabled", False):
            rewards = getattr(self, "student_generalize_level_rewards", {}) or {}
            for level in _STUDENT_GENERALIZE_LEVELS:
                if rewards.get(level, 0.0):
                    keys.append(f"student_generalize_{level}")
                    if getattr(self, "student_generalize_confidence_enabled", False):
                        keys.append(f"student_generalize_{level}_confidence")
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
        student_generalization_results: list[StudentGeneralizationResult] | None = None,
        teacher_pre_solve_result: TeacherPreSolveResult | None = None,
        student_name: str = "",
        student_model: str = "",
        teacher_prompt_selection: PromptPoolSelection | None = None,
        student_prompt_selection: PromptPoolSelection | None = None,
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
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Tutor debug trace dumped to %s", os.fspath(file_path))
        except Exception:
            logger.exception("Failed to dump tutor debug trace.")
