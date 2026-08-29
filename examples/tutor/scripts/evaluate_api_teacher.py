from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from tqdm import tqdm

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled with a runtime error
    AsyncOpenAI = None

from examples.tutor import train as tutor_train
from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    TutorConfig,
)
from examples.tutor.core.history import trace_to_json
from examples.tutor.workflow import TutorAgentWorkflow, _binary_repeat_summary

from areal import workflow_context
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.workflow_context import WorkflowContext
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorApiTeacherEval")

_T = TypeVar("_T")

DEFAULT_CONFIG_PATH = (
    # Default re-pointed when the answer-attempt config trees were deleted; this
    # file carries the same endpoint and model blocks the script reads.
    "examples/tutor/configs/math/0810/2gpu/base.yaml"
)
DEFAULT_DEEPSEEK_MODEL = "DeepSeek-V3.2"
DEEPSEEK_TEMPERATURE = 1.0
DEEPSEEK_TOP_P = 0.95

_RESERVED_REQUEST_FIELDS = {
    "messages",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
}
_PROXY_ENV_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


@dataclass(frozen=True, slots=True)
class PresolveMode:
    name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    mode: PresolveMode
    dataset_index: int
    attempt: int
    row: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.mode.name}:{self.dataset_index}:{self.attempt}"


@dataclass(slots=True)
class EpisodeResult:
    key: str
    mode: str
    presolve_enabled: bool
    dataset_index: int
    attempt: int
    item_id: str
    student_name: str
    student_model: str
    termination_reason: str
    error: str | None
    pre_solved: bool
    taught_success: bool
    final_correct: bool
    num_turns: int
    solve_turn: int | None
    leak_count: int
    leak_check_failed_count: int
    format_error_count: int
    student_call_failed: bool
    answer_judge_used_count: int
    answer_judge_failed_count: int
    answer_judge_override_correct_count: int
    total_reward: float
    teacher_pre_accepted: bool | None
    teacher_pre_attempts: int
    teacher_pre_error_count: int
    generalization: dict[str, dict[str, Any]]
    latest_student_answer_preview: str
    trace_path: str | None
    duration_seconds: float
    student_prompt_pool: str = ""
    student_prompt_index: int | None = None
    free_chat_enabled: bool = False
    outcome_score: float | None = None
    final_correct_score: float | None = None
    no_teaching_baseline: float | None = None
    code_stats: dict[str, int] | None = None
    # Per-episode personality evidence. Keeping the counts here (rather than only
    # in debug traces) lets full evaluations save traces only for errors without
    # losing the turn-1 and post-complaint adaptation measurements.
    personality_gate: dict[str, Any] = field(default_factory=dict)


class RecordingTutorWorkflow(TutorAgentWorkflow):
    """Tutor workflow that captures the existing eval outputs without trainers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.captured_stats: dict[str, Any] | None = None
        self.captured_trace: dict[str, Any] | None = None
        self.leak_check_failed_count = 0
        self.answer_judge_used_count = 0
        self.answer_judge_failed_count = 0
        self.answer_judge_override_correct_count = 0
        self.extra_api_callers: list[Any] = []
        super().__init__(*args, **kwargs)

    def _log_rollout_stats(self, **kwargs: Any) -> None:
        self.captured_stats = dict(kwargs)

    async def _maybe_dump_debug_trace(self, **kwargs: Any) -> None:
        self.captured_trace = dict(kwargs)

    def _make_answer_judge_caller(self, **kwargs: Any) -> Any:
        caller = super()._make_answer_judge_caller(**kwargs)
        if caller is not None and hasattr(caller, "caller"):
            self.extra_api_callers.append(caller)
        return caller

    async def _run_optional_leak_check(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._run_optional_leak_check(*args, **kwargs)
        if result.parse_error:
            self.leak_check_failed_count += 1
        return result

    async def _score_answer_async(self, *args: Any, **kwargs: Any) -> Any:
        result = await super()._score_answer_async(*args, **kwargs)
        answer_judge = result.raw_result.get("answer_judge")
        if isinstance(answer_judge, dict) and answer_judge.get("enabled") is True:
            if answer_judge.get("used") is True:
                self.answer_judge_used_count += 1
                if (
                    result.correct
                    and result.raw_result.get("exact_match_correct") is False
                ):
                    self.answer_judge_override_correct_count += 1
            elif answer_judge.get("error"):
                self.answer_judge_failed_count += 1
        return result


class ApiTeacherClient:
    """Force a configured model while retaining the workflow's token budget."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        max_retries: int,
        request_params: dict[str, Any],
        client: Any | None = None,
    ) -> None:
        if client is None:
            if AsyncOpenAI is None:
                raise RuntimeError("The openai package is required for API evaluation.")
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "EMPTY",
                timeout=timeout,
                max_retries=max(0, int(max_retries)),
            )
        self.model = model
        self.request_params = deepcopy(request_params)
        self._client = client
        self.chat = SimpleNamespace(completions=_ApiTeacherCompletions(self))

    async def list_models(self) -> list[str]:
        response = await self._client.models.list()
        return [str(item.id) for item in response.data]

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result


class _ApiTeacherCompletions:
    def __init__(self, parent: ApiTeacherClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        request = merge_dicts(self._parent.request_params, kwargs)
        request["model"] = self._parent.model
        return await self._parent._client.chat.completions.create(**request)


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


@contextmanager
def without_proxy_environment(enabled: bool = True) -> Any:
    """Temporarily avoid broken global proxies, then restore them exactly."""

    if not enabled:
        yield
        return
    saved = {name: os.environ[name] for name in _PROXY_ENV_VARS if name in os.environ}
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in _PROXY_ENV_VARS:
            os.environ.pop(name, None)
        os.environ.update(saved)


async def run_without_proxy_environment(
    operation: Callable[[], Awaitable[_T]],
    *,
    enabled: bool = True,
) -> _T:
    """Keep proxy variables cleared for an entire asynchronous API phase."""

    with without_proxy_environment(enabled=enabled):
        return await operation()


@contextmanager
def without_config_snapshot_writes() -> Any:
    """Load standalone-eval configs without writing trainer log snapshots."""

    previous_rank = os.environ.get("RANK")
    os.environ["RANK"] = "1"
    try:
        yield
    finally:
        if previous_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = previous_rank


def deepseek_non_thinking_params(seed: int) -> dict[str, Any]:
    """Request fields understood by local SGLang DeepSeek-V3.2 servers."""

    return {
        "seed": int(seed),
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
    }


def prepare_episode_workflow_kwargs(
    workflow_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Copy mutable role settings without changing configured request seeds."""

    resolved = dict(workflow_kwargs)
    resolved["student_models"] = deepcopy(workflow_kwargs["student_models"])
    resolved["aux_request_params"] = deepcopy(
        workflow_kwargs.get("aux_request_params") or {}
    )
    return resolved


def load_json_object(value: str, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_request_params(
    value: str,
    path: Path | None,
    *,
    label: str,
) -> dict[str, Any]:
    params = load_json_object(value, label=label)
    if path is not None:
        file_params = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(file_params, dict):
            raise ValueError(f"{label}-file must contain a JSON object.")
        params = merge_dicts(params, file_params)
    reserved = sorted(_RESERVED_REQUEST_FIELDS.intersection(params))
    if reserved:
        raise ValueError(
            f"{label} cannot set dedicated fields {reserved}; use the matching CLI "
            "options instead."
        )
    return params


def normalize_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("--teacher-base-url is required (or set DEEPSEEK_BASE_URL).")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def resolve_presolve_modes(choice: str, config_enabled: bool) -> list[PresolveMode]:
    if choice == "both":
        return [
            PresolveMode(name="presolve_off", enabled=False),
            PresolveMode(name="presolve_on", enabled=True),
        ]
    enabled = config_enabled if choice == "config" else choice == "on"
    return [
        PresolveMode(
            name="presolve_on" if enabled else "presolve_off",
            enabled=enabled,
        )
    ]


def resolve_generalization(choice: str, config_enabled: bool) -> bool:
    if choice == "config":
        return bool(config_enabled)
    return choice == "on"


def snapshot_student_models(config_path: str) -> list[dict[str, Any]]:
    """Resolve students before any role-specific command-line overrides."""

    # StatsLogger imports HTTP clients while loading config. Do not let a global
    # SOCKS proxy make this purely local operation require optional socksio.
    with without_proxy_environment(), without_config_snapshot_writes():
        baseline, _ = load_expr_config(["--config", config_path], TutorConfig)
    students = [asdict(student) for student in baseline.student_models]
    if not students:
        raise ValueError(
            "The source config must define at least one student_models entry."
        )
    return students


def load_experiment_config(
    config_path: str,
    overrides: list[str],
) -> tuple[TutorConfig, list[dict[str, Any]]]:
    students = snapshot_student_models(config_path)
    with without_proxy_environment(), without_config_snapshot_writes():
        config, _ = load_expr_config(
            ["--config", config_path, *overrides],
            TutorConfig,
        )
    return config, students


def select_student_models(
    student_models: list[dict[str, Any]], requested_names: list[str] | None
) -> list[dict[str, Any]]:
    """Keep an explicit eval subset without mutating the source config."""

    if not requested_names:
        return deepcopy(student_models)
    names = list(dict.fromkeys(str(name) for name in requested_names))
    available = {str(student["name"]): student for student in student_models}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            "Unknown --student-name value(s): "
            f"{missing}. Available students: {sorted(available)}"
        )
    return [deepcopy(available[name]) for name in names]


def effective_eval_presolve_enabled(config: TutorConfig) -> bool:
    override = config.evaluator.teacher_pre_enabled
    return bool(config.teacher_pre.enabled if override is None else override)


def configured_generalization_levels(
    workflow_kwargs: dict[str, Any],
) -> tuple[str, ...]:
    levels: list[str] = []
    if workflow_kwargs.get("student_generalize_retest_original"):
        levels.append("original")
    if workflow_kwargs.get("eval_preleak_retest"):
        levels.append("original_preleak")
    if workflow_kwargs.get("student_generalize_level1_enabled"):
        levels.append("level1")
    if workflow_kwargs.get("student_generalize_level2_enabled"):
        levels.append("level2")
    return tuple(levels)


def validate_effective_eval_semantics(
    *,
    config: TutorConfig,
    workflow_kwargs: dict[str, Any],
    student_models: list[dict[str, Any]],
) -> None:
    """Fail before API calls if offline eval drifted from regular validation."""

    if workflow_kwargs.get("aux_mode") != "api":
        raise ValueError(
            "The standalone API evaluator has no AReaL inference engine, so "
            "auxiliary_model.mode='self' cannot run here. Use "
            "--self-aux-via-teacher when the external teacher endpoint serves the "
            "same actor checkpoint, or evaluate with an explicit API auxiliary "
            "model."
        )
    if not student_models:
        raise ValueError("Evaluation needs at least one selected student.")
    configured_names = [str(student["name"]) for student in student_models]
    effective_names = [
        str(student["name"]) for student in workflow_kwargs["student_models"]
    ]
    if effective_names != configured_names:
        raise ValueError(
            "Effective eval students differ from the requested subset: "
            f"requested={configured_names}, effective={effective_names}."
        )

    expected_personality = asdict(config.personality)
    effective_personality = dict(workflow_kwargs.get("personality") or {})
    if effective_personality != expected_personality:
        demanded = sorted(
            {
                str(student.get("personality") or "")
                for student in student_models
                if str(student.get("personality") or "") not in {"", "none"}
            }
        )
        raise ValueError(
            "Offline evaluator personality settings drifted from regular Eval: "
            f"demanded={demanded}, effective={effective_personality}, "
            f"expected={expected_personality}."
        )

    if not config.free_chat.enabled:
        return
    free_chat = dict(workflow_kwargs.get("free_chat") or {})
    errors: list[str] = []
    if not free_chat.get("enabled"):
        errors.append("free_chat.enabled is false")
    if int(free_chat.get("budget", 0) or 0) != int(config.free_chat.budget):
        errors.append(
            "free_chat.budget is "
            f"{free_chat.get('budget')}, expected {config.free_chat.budget}"
        )
    if not workflow_kwargs.get("student_generalize_enabled"):
        errors.append("student_generalize is disabled")
    if not workflow_kwargs.get("student_generalize_retest_original"):
        errors.append("student_generalize.retest_original is false")
    if int(workflow_kwargs.get("student_generalize_replays", 0)) != int(
        config.student_generalize.replays
    ):
        errors.append(
            "student_generalize.replays is "
            f"{workflow_kwargs.get('student_generalize_replays')}, expected "
            f"{config.student_generalize.replays}"
        )
    if bool(workflow_kwargs.get("student_generalize_level1_enabled")) != bool(
        config.student_generalize.level1_enabled
    ):
        errors.append("student_generalize.level1_enabled drifted")
    if bool(workflow_kwargs.get("student_generalize_level2_enabled")) != bool(
        config.student_generalize.level2_enabled
    ):
        errors.append("student_generalize.level2_enabled drifted")
    if errors:
        raise ValueError(
            "Offline evaluator does not match the regular free-chat evaluation: "
            + "; ".join(errors)
        )


def build_eval_workflow_kwargs(
    *,
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    presolve_enabled: bool,
    external_self_aux: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    teacher_pre = config.teacher_pre
    student_generalize = config.student_generalize
    base_eval_gconfig = config.eval_gconfig or config.gconfig
    eval_gconfig = base_eval_gconfig.new(
        n_samples=1,
        temperature=float(args.teacher_temperature),
        top_p=args.teacher_top_p,
        max_new_tokens=int(args.teacher_max_tokens),
    )
    presolve_attempts = (
        int(args.presolve_attempts)
        if int(args.presolve_attempts) > 0
        else int(teacher_pre.attempts)
    )
    presolve_max_tokens = (
        int(args.presolve_max_tokens)
        if args.presolve_max_tokens is not None
        else int(teacher_pre.max_tokens)
    )

    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": tokenizer,
        "dataset_type": config.dataset_type,
        "answer_scorer": config.answer_scorer,
        "max_turns": config.max_turns,
        "enable_thinking": config.enable_thinking,
        "leak_handling_mode": config.leak_handling_mode,
        "aux_mode": auxiliary_model.mode,
        "aux_enable_thinking": auxiliary_model.enable_thinking,
        "aux_base_url": auxiliary_model.base_url,
        "aux_model": auxiliary_model.model,
        "aux_api_key": auxiliary_model.api_key,
        "aux_timeout": auxiliary_model.timeout,
        "aux_max_tokens": auxiliary_model.max_tokens,
        "aux_temperature": auxiliary_model.temperature,
        "aux_top_p": auxiliary_model.top_p,
        "max_concurrent_aux_calls": auxiliary_model.max_concurrent_calls,
        "aux_request_params": deepcopy(auxiliary_model.request_params),
        "student_models": deepcopy(student_models),
        "success_reward": reward.success,
        "leak_penalty": reward.leak_penalty,
        "leak_penalty_mode": reward.leak_penalty_mode,
        "leak_penalty_final_answer": reward.leak_penalty_final_answer,
        "leak_penalty_compute": reward.leak_penalty_compute,
        "leak_penalty_formula": reward.leak_penalty_formula,
        "leak_penalty_aggregation": reward.leak_penalty_aggregation,
        "turn_local_reward_components": tuple(reward.turn_local_components),
        "turn_local_reward_component_placements": dict(
            reward.turn_local_component_placements
        ),
        "turn_local_reward_default_placement": (
            "group_norm"
            if config.actor.group_baseline_local_reward_mode == "include"
            else "pre_std"
        ),
        "format_error_penalty": reward.format_error_penalty,
        "personality_gate_terminate_penalty": (
            reward.personality_gate_terminate_penalty
        ),
        "personality_gate_fail_penalty": reward.personality_gate_fail_penalty,
        "leaked_success_reward_scale": reward.leaked_success_reward_scale,
        "assign_success_reward": reward.assign_success_reward,
        "outcome_prior_turn_weight": reward.outcome_prior_turn_weight,
        "outcome_credit_gamma": reward.outcome_credit_gamma,
        "early_success_bonus": reward.early_success_bonus,
        "success_turn_shaping": asdict(reward.success_turn_shaping),
        "max_turn_penalty": reward.max_turn_penalty,
        "enable_turn_penalty": reward.enable_turn_penalty,
        "turn_penalty": reward.turn_penalty,
        "length_penalty_threshold_chars": reward.length_penalty_threshold_chars,
        "length_penalty_per_100_chars": reward.length_penalty_per_100_chars,
        "length_penalty_min": reward.length_penalty_min,
        "zero_reward_on_length_stop": reward.zero_reward_on_length_stop,
        "teacher_diversity_reward": asdict(reward.teacher_diversity),
        "teacher_context_reward": asdict(reward.teacher_context),
        "teacher_progress_judge": asdict(reward.teacher_progress_judge),
        "student_request_judge": asdict(reward.student_request_judge),
        "world_model": asdict(config.world_model),
        "guided_slots": asdict(config.guided_slots),
        "opd": asdict(config.opd),
        "prompt_instruction": asdict(config.prompt_instruction),
        "free_chat": asdict(config.free_chat),
        # The gate is part of the student's deployment behavior, not a training-only
        # reward. Regular validation forwards this block in train.py; the standalone
        # evaluator must do the same or demanding personality cells cannot run.
        "personality": asdict(config.personality),
        "student_type_probe": asdict(config.student_type_probe),
        "cross_eval": asdict(config.cross_eval),
        "teacher_history_tags": config.teacher_history_tags,
        "teacher_private_visibility": config.teacher_private_visibility,
        "local_advantage_turn_discount": config.actor.turn_discount,
        "teacher_system_prompt": config.teacher_system_prompt,
        "teacher_anti_leak_instruction_enabled": (
            config.teacher_anti_leak_instruction_enabled
        ),
        "teacher_adaptive_instruction_enabled": (
            config.teacher_adaptive_instruction_enabled
        ),
        "teacher_prompt_pool_path": config.prompt_pool.teacher_path,
        "teacher_warmup_enabled": config.prompt_pool.teacher_warmup.enabled,
        "teacher_warmup_prompt_path": (config.prompt_pool.teacher_warmup.prompt_path),
        "teacher_warmup_steps": config.prompt_pool.teacher_warmup.steps,
        "teacher_user_prompt_template": config.teacher_user_prompt_template,
        "teacher_show_ground_truth": config.teacher_show_ground_truth,
        "format_handling_mode": config.format_handling_mode,
        "teacher_pre_enabled": teacher_pre.enabled,
        "teacher_pre_mode": teacher_pre.mode,
        "teacher_pre_verify": teacher_pre.verify,
        "teacher_pre_attempts": teacher_pre.attempts,
        "teacher_pre_max_tokens": teacher_pre.max_tokens,
        "teacher_pre_visibility": teacher_pre.visibility,
        "teacher_pre_on_reject": teacher_pre.on_reject,
        "teacher_pre_share_per_group": teacher_pre.share_per_group,
        "student_system_prompt": config.student_system_prompt,
        "student_prompt_pool_path": config.prompt_pool.student_train_path,
        "student_heldout_prompt_pool_path": "",
        "student_prompt_include_base": config.prompt_pool.include_base,
        "student_turn_behavior_enabled": (
            config.prompt_pool.student_turn_behavior.enabled
        ),
        "student_turn_behavior_path": config.prompt_pool.student_turn_behavior.path,
        "student_turn_behavior_separate_call_behavior_names": (
            config.prompt_pool.student_turn_behavior.separate_call_behavior_names
        ),
        "prompt_pool_seed": config.seed,
        "leak_check_system_prompt": config.leak_check_system_prompt,
        "answer_judge_enabled": auxiliary_model.answer_judge_enabled,
        "answer_judge_max_tokens": auxiliary_model.answer_judge_max_tokens,
        "answer_judge_system_prompt": config.answer_judge_system_prompt,
        "debug_trace_dir": config.debug_trace_dir or None,
        "debug_trace_every_n_rollouts": config.debug_trace_every_n_rollouts,
        "max_train_sample_tokens": config.gconfig.max_tokens,
        "tokenizer_path": config.tokenizer_path,
        "model_context_length": config.sglang.context_length,
        "student_generalize_enabled": student_generalize.enabled,
        "student_generalize_mode": student_generalize.mode,
        "student_generalize_source": student_generalize.source,
        "student_generalize_path": student_generalize.path,
        "student_generalize_replays": student_generalize.replays,
        "student_generalize_turn_credit": student_generalize.turn_credit,
        "student_generalize_turn_credit_replays": (
            student_generalize.turn_credit_replays
        ),
        "student_generalize_retest_original": student_generalize.retest_original,
        "student_generalize_level1_enabled": student_generalize.level1_enabled,
        "student_generalize_level2_enabled": student_generalize.level2_enabled,
        "student_generalize_level1_reward": student_generalize.level1_reward,
        "student_generalize_level2_reward": student_generalize.level2_reward,
        "student_generalize_retest_reward": student_generalize.retest_reward,
        "student_generalize_confidence_enabled": (
            student_generalize.confidence.enabled
        ),
        "student_generalize_confidence_reward_scale": (
            student_generalize.confidence.reward_scale
        ),
    }
    if external_self_aux is not None:
        if auxiliary_model.mode != "self":
            raise ValueError(
                "--self-aux-via-teacher requires auxiliary_model.mode='self'; "
                f"the loaded config uses {auxiliary_model.mode!r}."
            )
        required = {"base_url", "model", "api_key", "request_params"}
        missing = sorted(required - set(external_self_aux))
        if missing:
            raise ValueError(
                "external self-auxiliary bridge is missing: " + ", ".join(missing)
            )
        # Regular validation's 'self' auxiliary uses the current actor adapter.
        # A standalone evaluator has no AReaL engine, so reproduce that path with
        # the same OpenAI endpoint and the same per-request lora_path as the
        # external teacher. Keeping the source auxiliary sampling settings while
        # merging the teacher request body preserves its judge/leak-call semantics.
        workflow_kwargs["aux_mode"] = "api"
        workflow_kwargs["aux_base_url"] = str(external_self_aux["base_url"])
        workflow_kwargs["aux_model"] = str(external_self_aux["model"])
        workflow_kwargs["aux_api_key"] = str(external_self_aux["api_key"])
        workflow_kwargs["aux_request_params"] = merge_dicts(
            deepcopy(auxiliary_model.request_params),
            dict(external_self_aux["request_params"]),
        )
    # This is the canonical conversion used by TutorPPOTrainer for its regular
    # validation pass. Keeping it here prevents the API evaluator from silently
    # falling back to answer-attempt defaults for free-chat experiments.
    eval_workflow_kwargs = tutor_train._build_eval_workflow_kwargs(
        workflow_kwargs, config
    )
    eval_workflow_kwargs["gconfig"] = eval_gconfig
    eval_workflow_kwargs["teacher_pre_enabled"] = bool(presolve_enabled)
    eval_workflow_kwargs["teacher_pre_attempts"] = presolve_attempts
    eval_workflow_kwargs["teacher_pre_max_tokens"] = presolve_max_tokens
    # Traces are owned by this standalone evaluator's --save-traces path.
    eval_workflow_kwargs["debug_trace_dir"] = None
    validate_effective_eval_semantics(
        config=config,
        workflow_kwargs=eval_workflow_kwargs,
        student_models=student_models,
    )
    return eval_workflow_kwargs


def prepare_test_dataset(
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    *,
    tokenizer: Any,
    limit: int,
    stratified_max_samples: int = 0,
    student_prompts: tuple[tutor_train.EvalStudentPrompt, ...] | None = None,
) -> Any:
    valid_config = tutor_train._without_remote_dataset_loading(config.valid_dataset)
    dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_config,
        tokenizer=tokenizer,
    )
    student_generalize = getattr(config, "student_generalize", None)
    required_levels = (
        student_generalize.transfer_levels()
        if student_generalize is not None and student_generalize.enabled
        else ()
    )
    if required_levels:
        bank = tutor_train.load_student_generalize_bank(
            student_generalize.path,
            source=student_generalize.source,
        )
        dataset = tutor_train._filter_student_generalize_dataset(
            dataset,
            bank=bank,
            split_name="test",
            required_levels=required_levels,
            sample_count=(
                tutor_train.MATH_GENERALIZATION_SAMPLE_COUNT
                if student_generalize.source == "train"
                else None
            ),
        )
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if stratified_max_samples > 0 and eval_max_samples > 0:
            raise ValueError(
                "Use either evaluator.max_samples or --stratified-max-samples, "
                "not both."
            )
        if 0 < eval_max_samples < len(dataset):
            rng = random.Random(config.seed)
            indices = sorted(rng.sample(range(len(dataset)), k=eval_max_samples))
            dataset = dataset.select(indices)
    if 0 < stratified_max_samples < len(dataset):
        dataset = dataset.select(
            stratified_math_subset_indices(
                dataset,
                sample_count=stratified_max_samples,
                seed=int(config.seed),
            )
        )
    if limit > 0 and limit < len(dataset):
        dataset = dataset.select(range(limit))
    dataset = tutor_train._expand_eval_dataset_for_students(
        dataset,
        [str(student["name"]) for student in student_models],
    )
    if student_prompts is None:
        student_prompts = tutor_train._load_eval_student_prompts(config)
    dataset = tutor_train._expand_eval_dataset_for_student_prompts(
        dataset,
        student_prompts,
    )
    return dataset


def stratified_math_subset_indices(
    dataset: Any,
    *,
    sample_count: int,
    seed: int,
) -> list[int]:
    """Select a deterministic proportional subset over MATH type x level."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    dataset_size = len(dataset)
    if sample_count >= dataset_size:
        return list(range(dataset_size))

    strata: dict[tuple[str, str], list[int]] = {}
    for index in range(dataset_size):
        row = dataset[index]
        metadata = row.get("metadata") if hasattr(row, "get") else None
        if not isinstance(metadata, dict):
            raise ValueError(
                "Stratified evaluation requires every row to have metadata."
            )
        math_type = str(metadata.get("type") or "").strip()
        level = str(metadata.get("level") or "").strip()
        if not math_type or not level:
            raise ValueError(
                "Stratified evaluation requires metadata.type and metadata.level."
            )
        strata.setdefault((math_type, level), []).append(index)

    allocations: dict[tuple[str, str], int] = {}
    remainders: dict[tuple[str, str], int] = {}
    for key, indices in strata.items():
        numerator = sample_count * len(indices)
        allocations[key], remainders[key] = divmod(numerator, dataset_size)
    unallocated = sample_count - sum(allocations.values())
    remainder_order = sorted(strata, key=lambda key: (-remainders[key], key))
    for key in remainder_order[:unallocated]:
        allocations[key] += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for key in sorted(strata):
        selected.extend(rng.sample(strata[key], allocations[key]))
    selected.sort()
    if len(selected) != sample_count or len(set(selected)) != sample_count:
        raise RuntimeError("Stratified subset selection produced invalid indices.")
    return selected


def dataset_sha256(dataset: Any) -> str:
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        payload = json.dumps(
            dict(dataset[index]),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        digest.update(payload.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def preview_text(value: Any, max_chars: int = 500) -> str:
    text = str(value or "")
    return text[: max(0, int(max_chars))]


def serialize_generalization(
    workflow: RecordingTutorWorkflow,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for result in workflow.last_student_generalization_results:
        judge_result = result.judge_result
        replay_count = int(result.replay_count or 0)
        replay_correct = int(result.replay_correct or 0)
        score = None
        if result.attempted and not result.skipped:
            score = (
                replay_correct / replay_count
                if replay_count > 0
                else float(bool(judge_result is not None and judge_result.correct))
            )
        payload[str(result.level)] = {
            "attempted": bool(result.attempted),
            "skipped": bool(result.skipped),
            "skip_reason": str(result.skip_reason or ""),
            "correct": bool(judge_result.correct) if judge_result is not None else None,
            "student_error": result.student_error,
            "replay_count": replay_count,
            "replay_correct": replay_correct,
            "score": score,
            "confidence": float(result.confidence),
        }
    return payload


def summarize_personality_gate(
    workflow: RecordingTutorWorkflow, *, student_name: str
) -> dict[str, Any]:
    """Keep the gate sequence needed to distinguish blind and adaptive teaching.

    Compliance is always computed over sampled gate calls. An unsampled turn is
    neither a pass nor a failure. "Post complaint" starts after the first gated
    turn, and the first-post value is the next *sampled* check; this avoids calling
    an unaudited turn compliant merely because the student was allowed through.
    """

    personality = "none"
    for runtime in workflow.student_model_runtimes.values():
        if runtime.name == student_name:
            personality = str(runtime.personality or "none")
            break

    traces = list(workflow.last_traces)
    gate_traces = [
        trace for trace in traces if trace.personality_gate_result is not None
    ]
    active = personality not in {"", "none"}
    if not active:
        return {
            "personality": "none",
            "active": False,
            "eligible_turn_count": 0,
            "sampled_turn_count": 0,
            "passed_turn_count": 0,
            "gated_turn_count": 0,
            "gate_error_count": 0,
            "compliance": None,
            "turn1_sampled": False,
            "turn1_passed": None,
            "first_complaint_turn": None,
            "first_complaint_kind": None,
            "first_post_complaint_sampled_turn": None,
            "first_post_complaint_passed": None,
            "post_complaint_sampled_turn_count": 0,
            "post_complaint_passed_turn_count": 0,
            "post_complaint_compliance": None,
            "post_complaint_all_passed": None,
            "bare_complaint_count": 0,
            "explain_complaint_count": 0,
            "unknown_complaint_count": 0,
        }

    sampled = [
        trace for trace in gate_traces if bool(trace.personality_gate_result.sampled)
    ]
    passed = [trace for trace in sampled if bool(trace.personality_gate_result.passed)]
    gated = [trace for trace in gate_traces if bool(trace.personality_gated)]
    errors = [trace for trace in sampled if bool(trace.personality_gate_result.error)]

    turn1 = next(
        (
            trace
            for trace in gate_traces
            if trace.turn_idx == 1 and trace.personality_gate_result.sampled
        ),
        None,
    )
    first_complaint = gated[0] if gated else None
    post_sampled = (
        [trace for trace in sampled if trace.turn_idx > first_complaint.turn_idx]
        if first_complaint is not None
        else []
    )
    post_passed = [
        trace for trace in post_sampled if bool(trace.personality_gate_result.passed)
    ]
    first_post = post_sampled[0] if post_sampled else None

    bare_complaints = set(workflow.personality_complaints_bare)
    explain_complaints = set(
        workflow.personality_complaints_explain.get(personality, ())
    )

    def complaint_kind(trace: Any) -> str:
        complaint = str(trace.student_output or "").strip()
        if complaint in explain_complaints:
            return "explain"
        if complaint in bare_complaints:
            return "bare"
        return "unknown"

    complaint_kinds = [complaint_kind(trace) for trace in gated]
    payload = {
        "personality": personality,
        "active": True,
        "eligible_turn_count": len(gate_traces),
        "sampled_turn_count": len(sampled),
        "passed_turn_count": len(passed),
        "gated_turn_count": len(gated),
        "gate_error_count": len(errors),
        "compliance": _rate(len(passed), len(sampled)),
        "turn1_sampled": turn1 is not None,
        "turn1_passed": (
            bool(turn1.personality_gate_result.passed) if turn1 is not None else None
        ),
        "first_complaint_turn": (
            int(first_complaint.turn_idx) if first_complaint is not None else None
        ),
        "first_complaint_kind": (complaint_kinds[0] if complaint_kinds else None),
        "first_post_complaint_sampled_turn": (
            int(first_post.turn_idx) if first_post is not None else None
        ),
        "first_post_complaint_passed": (
            bool(first_post.personality_gate_result.passed)
            if first_post is not None
            else None
        ),
        "post_complaint_sampled_turn_count": len(post_sampled),
        "post_complaint_passed_turn_count": len(post_passed),
        "post_complaint_compliance": _rate(len(post_passed), len(post_sampled)),
        "post_complaint_all_passed": (
            len(post_passed) == len(post_sampled) if post_sampled else None
        ),
        "bare_complaint_count": complaint_kinds.count("bare"),
        "explain_complaint_count": complaint_kinds.count("explain"),
        "unknown_complaint_count": complaint_kinds.count("unknown"),
    }
    classification_labels = Counter(
        str(trace.personality_gate_result.classification_label)
        for trace in sampled
        if trace.personality_gate_result.classification_label is not None
    )
    if classification_labels:
        payload["classification_label_counts"] = dict(
            sorted(classification_labels.items())
        )
    classification_distributions = [
        trace.personality_gate_result.classification_probabilities
        for trace in sampled
        if trace.personality_gate_result.classification_probabilities is not None
    ]
    if classification_distributions:
        labels = sorted(
            {
                label
                for distribution in classification_distributions
                for label in distribution
            }
        )
        payload["classification_distribution_count"] = len(classification_distributions)
        payload["mean_classification_probabilities"] = {
            label: sum(
                float(distribution.get(label, 0.0))
                for distribution in classification_distributions
            )
            / len(classification_distributions)
            for label in labels
        }
        margins = [
            float(trace.personality_gate_result.classification_margin)
            for trace in sampled
            if trace.personality_gate_result.classification_margin is not None
        ]
        if margins:
            payload["mean_classification_margin"] = sum(margins) / len(margins)
    return payload


def build_trace_payload(
    *,
    workflow: RecordingTutorWorkflow,
    spec: EpisodeSpec,
    result: EpisodeResult,
) -> dict[str, Any]:
    captured = workflow.captured_trace or {}
    teacher_pre = workflow.last_teacher_pre_solve_result
    return {
        "result": asdict(result),
        "dataset_row": spec.row,
        "task": captured.get("task", spec.row.get("task", "")),
        "ground_truth": captured.get("ground_truth", spec.row.get("ground_truth", "")),
        "initial_student_answer": captured.get("initial_student_answer", ""),
        "latest_student_answer": captured.get("latest_student_answer", ""),
        "turns": [trace_to_json(trace) for trace in workflow.last_traces],
        "history": workflow.last_history,
        "teacher_pre_solve": asdict(teacher_pre) if teacher_pre is not None else None,
        "student_generalization": [
            asdict(item) for item in workflow.last_student_generalization_results
        ],
    }


def result_from_workflow(
    *,
    workflow: RecordingTutorWorkflow,
    spec: EpisodeSpec,
    duration_seconds: float,
) -> EpisodeResult:
    stats = workflow.captured_stats
    if stats is None:
        raise RuntimeError("Tutor workflow did not emit episode statistics.")
    termination_reason = str(stats.get("termination_reason") or "unknown")
    traces = workflow.last_traces
    solve_turn = next(
        (
            int(trace.turn_idx)
            for trace in traces
            if trace.judge_correct and not trace.invalid_due_to_leak
        ),
        None,
    )
    captured = workflow.captured_trace or {}
    student_call_failed = bool(stats.get("student_call_failed"))
    teacher_pre = workflow.last_teacher_pre_solve_result
    teacher_pre_errors = (
        [attempt.error for attempt in teacher_pre.attempts if attempt.error]
        if teacher_pre is not None
        else []
    )
    all_teacher_pre_attempts_failed = bool(
        teacher_pre is not None
        and teacher_pre.attempts
        and len(teacher_pre_errors) == len(teacher_pre.attempts)
    )
    pre_solved = bool(stats.get("pre_success"))
    free_chat_enabled = bool(getattr(workflow, "free_chat_enabled", False))
    outcome_score = (
        float(
            workflow._free_chat_outcome_score(
                workflow.last_student_generalization_results
            )
        )
        if free_chat_enabled
        else float(termination_reason == "success")
    )
    taught_success = bool(outcome_score > 0.0)
    final_correct_score = max(float(pre_solved), outcome_score)
    no_teaching_baseline = stats.get("no_teaching_baseline")
    code_stats = stats.get("code_stats")
    return EpisodeResult(
        key=spec.key,
        mode=spec.mode.name,
        presolve_enabled=spec.mode.enabled,
        dataset_index=spec.dataset_index,
        attempt=spec.attempt,
        item_id=str(spec.row.get("id", spec.dataset_index)),
        student_name=str(stats.get("student_name") or ""),
        student_model=str(captured.get("student_model") or ""),
        termination_reason=termination_reason,
        error=(
            "TeacherPreSolveError: all presolve attempts failed: "
            + " | ".join(teacher_pre_errors)
            if all_teacher_pre_attempts_failed
            else None
        ),
        pre_solved=pre_solved,
        taught_success=taught_success,
        final_correct=bool(final_correct_score > 0.0),
        num_turns=len(traces),
        solve_turn=solve_turn,
        leak_count=int(stats.get("leak_count") or 0),
        leak_check_failed_count=workflow.leak_check_failed_count,
        format_error_count=sum(bool(trace.tutor_format_error) for trace in traces),
        student_call_failed=student_call_failed,
        answer_judge_used_count=workflow.answer_judge_used_count,
        answer_judge_failed_count=workflow.answer_judge_failed_count,
        answer_judge_override_correct_count=(
            workflow.answer_judge_override_correct_count
        ),
        total_reward=float(stats.get("total_reward") or 0.0),
        teacher_pre_accepted=(
            bool(teacher_pre.accepted) if teacher_pre is not None else None
        ),
        teacher_pre_attempts=(
            len(teacher_pre.attempts) if teacher_pre is not None else 0
        ),
        teacher_pre_error_count=len(teacher_pre_errors),
        generalization=serialize_generalization(workflow),
        latest_student_answer_preview=preview_text(
            captured.get("latest_student_answer", "")
        ),
        trace_path=None,
        duration_seconds=float(duration_seconds),
        student_prompt_pool=str(
            spec.row.get(TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD) or ""
        ),
        student_prompt_index=(
            int(spec.row[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD])
            if spec.row.get(TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD) is not None
            else None
        ),
        free_chat_enabled=free_chat_enabled,
        outcome_score=outcome_score,
        final_correct_score=final_correct_score,
        no_teaching_baseline=(
            float(no_teaching_baseline) if no_teaching_baseline is not None else None
        ),
        code_stats=(
            {str(key): int(value) for key, value in code_stats.items()}
            if isinstance(code_stats, dict)
            else None
        ),
        personality_gate=summarize_personality_gate(
            workflow,
            student_name=str(stats.get("student_name") or ""),
        ),
    )


def error_result(
    spec: EpisodeSpec,
    error: Exception,
    *,
    duration_seconds: float,
) -> EpisodeResult:
    return EpisodeResult(
        key=spec.key,
        mode=spec.mode.name,
        presolve_enabled=spec.mode.enabled,
        dataset_index=spec.dataset_index,
        attempt=spec.attempt,
        item_id=str(spec.row.get("id", spec.dataset_index)),
        student_name=str(spec.row.get(TUTOR_EVAL_STUDENT_FIELD) or ""),
        student_model="",
        termination_reason="error",
        error=f"{type(error).__name__}: {error}",
        pre_solved=False,
        taught_success=False,
        final_correct=False,
        num_turns=0,
        solve_turn=None,
        leak_count=0,
        leak_check_failed_count=0,
        format_error_count=0,
        student_call_failed=False,
        answer_judge_used_count=0,
        answer_judge_failed_count=0,
        answer_judge_override_correct_count=0,
        total_reward=0.0,
        teacher_pre_accepted=None,
        teacher_pre_attempts=0,
        teacher_pre_error_count=0,
        generalization={},
        latest_student_answer_preview="",
        trace_path=None,
        duration_seconds=float(duration_seconds),
        student_prompt_pool=str(
            spec.row.get(TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD) or ""
        ),
        student_prompt_index=(
            int(spec.row[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD])
            if spec.row.get(TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD) is not None
            else None
        ),
    )


async def close_workflow_api_clients(workflow: RecordingTutorWorkflow) -> None:
    """Close per-episode auxiliary/student clients created by the workflow."""

    wrappers = [
        workflow.aux_caller,
        workflow.confidence_aux_caller,
        *workflow.extra_api_callers,
    ]
    for runtime in workflow.student_model_runtimes.values():
        wrappers.extend([runtime.caller, runtime.confidence_caller])

    clients: dict[int, Any] = {}
    for wrapper in wrappers:
        caller = getattr(wrapper, "caller", None)
        client = getattr(caller, "_client", None)
        if client is not None:
            clients[id(client)] = client
    for client in clients.values():
        close = getattr(client, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if asyncio.iscoroutine(result):
                async with asyncio.timeout(10.0):
                    await result
        except TimeoutError:
            logger.warning("Timed out closing an episode API client after 10s.")
        except Exception as exc:  # pragma: no cover - transport-specific cleanup
            logger.warning("Failed to close an episode API client: %s", exc)


def safe_path_token(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    return safe.strip("._") or "item"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()


def rewrite_results_jsonl(path: Path, results: list[EpisodeResult]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_existing_results(path: Path) -> list[EpisodeResult]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty_line_numbers = [
        line_number for line_number, line in enumerate(lines, 1) if line.strip()
    ]
    last_nonempty_line = nonempty_line_numbers[-1] if nonempty_line_numbers else 0
    results_by_key: dict[str, EpisodeResult] = {}
    needs_rewrite = False
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            result = EpisodeResult(**payload)
        except json.JSONDecodeError as exc:
            if line_number == last_nonempty_line:
                logger.warning(
                    "Ignoring a truncated final resume record at %s:%s: %s",
                    path,
                    line_number,
                    exc,
                )
                needs_rewrite = True
                break
            raise ValueError(
                f"Invalid resume record at {path}:{line_number}: {exc}"
            ) from exc
        except TypeError as exc:
            raise ValueError(
                f"Invalid resume record at {path}:{line_number}: {exc}"
            ) from exc
        if result.key in results_by_key:
            logger.warning(
                "Duplicate resume key %s; keeping the latest record.", result.key
            )
            needs_rewrite = True
        results_by_key[result.key] = result
    results = list(results_by_key.values())
    if needs_rewrite:
        rewrite_results_jsonl(path, results)
    return results


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def latest_results(results: list[EpisodeResult]) -> list[EpisodeResult]:
    """Keep the last record for each episode key, matching JSONL resume semantics."""

    by_key: dict[str, EpisodeResult] = {}
    for result in results:
        by_key[result.key] = result
    return list(by_key.values())


def result_needs_retry(
    result: EpisodeResult,
    *,
    retry_errors: bool = True,
    retry_diagnostic_failures: bool = False,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
) -> bool:
    """Return whether resume should replace an unreliable episode record."""

    return bool(
        result_retry_reasons(
            result,
            retry_errors=retry_errors,
            retry_diagnostic_failures=retry_diagnostic_failures,
            generalization_levels=generalization_levels,
            expected_generalization_replays=expected_generalization_replays,
        )
    )


def result_retry_reasons(
    result: EpisodeResult,
    *,
    retry_errors: bool = True,
    retry_diagnostic_failures: bool = False,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
) -> list[str]:
    """Describe infrastructure failures that make an episode unsafe to score."""

    reasons: list[str] = []
    if retry_errors and result.error is not None:
        reasons.append(f"error={result.error}")
    if not retry_diagnostic_failures:
        return reasons
    if result.student_call_failed:
        reasons.append("student_call_failed")
    if result.leak_check_failed_count:
        reasons.append(f"leak_check_failed={result.leak_check_failed_count}")
    if result.answer_judge_failed_count:
        reasons.append(f"answer_judge_failed={result.answer_judge_failed_count}")
    if result.teacher_pre_error_count:
        reasons.append(f"teacher_pre_errors={result.teacher_pre_error_count}")
    gate = result.personality_gate or {}
    if int(gate.get("gate_error_count", 0) or 0):
        reasons.append(f"personality_gate_errors={gate['gate_error_count']}")
    expected_replays = max(0, int(expected_generalization_replays))
    if expected_replays:
        generalization = result.generalization or {}
        for level in generalization_levels:
            replay = generalization.get(level)
            if not isinstance(replay, dict):
                reasons.append(f"{level}_retest=missing")
                continue
            actual_replays = int(replay.get("replay_count", -1) or 0)
            if actual_replays != expected_replays:
                reasons.append(f"{level}_replays={actual_replays}/{expected_replays}")
            if replay.get("score") is None:
                reasons.append(f"{level}_score=missing")
            if replay.get("student_error"):
                reasons.append(f"{level}_student_error={replay['student_error']}")
    return reasons


def aggregate_mode(
    results: list[EpisodeResult],
    *,
    expected: int,
    generalization_enabled: bool = False,
    generalization_levels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    results = latest_results(results)
    completed = [result for result in results if result.error is None]
    pre_solved = sum(result.pre_solved for result in completed)
    taught_success = sum(result.taught_success for result in completed)
    outcome_scores = [
        float(
            result.outcome_score
            if result.outcome_score is not None
            else result.taught_success
        )
        for result in completed
    ]
    final_correct_scores = [
        float(
            result.final_correct_score
            if result.final_correct_score is not None
            else result.final_correct
        )
        for result in completed
    ]
    clean_taught_success = sum(
        result.taught_success and not result.student_call_failed for result in completed
    )
    clean_aux_taught_success = sum(
        result.taught_success
        and result.leak_check_failed_count == 0
        and result.answer_judge_failed_count == 0
        for result in completed
    )
    final_correct = sum(result.final_correct for result in completed)
    pre_solve_skipped = sum(
        result.termination_reason == "pre_solve_skipped" for result in completed
    )
    observed_initially_unsolved = sum(
        not result.pre_solved and result.termination_reason != "pre_solve_skipped"
        for result in completed
    )
    clean_initially_unsolved = sum(
        not result.pre_solved
        and result.termination_reason != "pre_solve_skipped"
        and not result.student_call_failed
        for result in completed
    )
    clean_aux_initially_unsolved = sum(
        not result.pre_solved
        and result.termination_reason != "pre_solve_skipped"
        and result.leak_check_failed_count == 0
        and result.answer_judge_failed_count == 0
        for result in completed
    )
    covered = len(completed) - pre_solve_skipped
    solve_turns = [
        result.solve_turn for result in completed if result.solve_turn is not None
    ]
    presolve_results = [result for result in completed if result.presolve_enabled]
    presolve_accepted = sum(
        result.teacher_pre_accepted is True for result in presolve_results
    )
    presolve_enabled = any(result.presolve_enabled for result in results)
    generalization: dict[str, dict[str, Any]] = {}
    levels = {level for result in completed for level in result.generalization}
    if generalization_levels is not None:
        levels.update(generalization_levels)
    elif generalization_enabled:
        levels.update(("level1", "level2"))
    levels = sorted(levels)
    for level in levels:
        level_results = [
            (result, result.generalization[level])
            for result in completed
            if level in result.generalization
        ]
        attempted = [pair for pair in level_results if pair[1].get("attempted")]
        evaluable = [pair for pair in attempted if not pair[1].get("student_error")]
        correct = sum(item.get("correct") is True for _, item in attempted)
        evaluable_correct = sum(item.get("correct") is True for _, item in evaluable)
        taught_attempted = [pair for pair in attempted if pair[0].taught_success]
        taught_correct = sum(
            item.get("correct") is True for _, item in taught_attempted
        )
        student_errors = sum(
            bool(item.get("student_error")) for _, item in level_results
        )
        replay_count = sum(int(item.get("replay_count") or 0) for _, item in attempted)
        replay_correct = sum(
            int(item.get("replay_correct") or 0) for _, item in attempted
        )

        def _score(item: dict[str, Any]) -> float:
            if item.get("score") is not None:
                return float(item["score"])
            item_replays = int(item.get("replay_count") or 0)
            if item_replays > 0:
                return int(item.get("replay_correct") or 0) / item_replays
            return float(item.get("correct") is True)

        attempted_score = sum(_score(item) for _, item in attempted)
        evaluable_score = sum(_score(item) for _, item in evaluable)
        taught_score = sum(_score(item) for _, item in taught_attempted)
        generalization[level] = {
            "present_episode_count": len(level_results),
            "attempted": len(attempted),
            "evaluable": len(evaluable),
            "correct": correct,
            "student_error_count": student_errors,
            "replay_attempt_count": replay_count,
            "replay_correct_count": replay_correct,
            "accuracy_on_replays": _rate(replay_correct, replay_count),
            "mean_episode_score_on_attempted": _rate(attempted_score, len(attempted)),
            "mean_episode_score_on_evaluable": _rate(evaluable_score, len(evaluable)),
            "mean_episode_score_full_set": _rate(attempted_score, expected),
            "conditional_accuracy_on_attempted": _rate(correct, len(attempted)),
            "conditional_accuracy_on_evaluable": _rate(
                evaluable_correct, len(evaluable)
            ),
            "attempt_coverage_on_taught_success": _rate(
                len(taught_attempted), taught_success
            ),
            "attempt_coverage_full_set": _rate(len(attempted), expected),
            "end_to_end_correct_rate_on_taught_success": _rate(
                taught_correct, taught_success
            ),
            "end_to_end_correct_rate_full_set": _rate(correct, expected),
            "end_to_end_score_on_taught_success": _rate(taught_score, taught_success),
        }

    code_results = [
        result for result in completed if isinstance(result.code_stats, dict)
    ]
    code_stat_names = sorted(
        {name for result in code_results for name in (result.code_stats or {})}
    )
    code_totals = {
        name: sum(
            int((result.code_stats or {}).get(name, 0)) for result in code_results
        )
        for name in code_stat_names
    }
    code_turns = sum(result.num_turns for result in code_results)
    code_channel = {
        "episode_count": len(code_results),
        "turn_count": code_turns,
        "totals": code_totals,
        "per_turn": {
            name: _rate(value, code_turns) for name, value in code_totals.items()
        },
    }
    if code_results:
        unproductive = sum(
            code_totals.get(name, 0)
            for name in ("crashes", "silent_cells", "no_program")
        )
        code_channel["productive_rate_per_turn"] = _rate(
            max(0, code_turns - unproductive), code_turns
        )

    baseline_pairs = [
        (float(result.no_teaching_baseline), outcome_score)
        for result, outcome_score in zip(completed, outcome_scores, strict=True)
        if result.no_teaching_baseline is not None
    ]

    gate_payloads = [
        result.personality_gate or {}
        for result in completed
        if isinstance(result.personality_gate, dict)
    ]
    active_gates = [gate for gate in gate_payloads if gate.get("active") is True]
    sampled_turns = sum(
        int(gate.get("sampled_turn_count", 0) or 0) for gate in active_gates
    )
    passed_turns = sum(
        int(gate.get("passed_turn_count", 0) or 0) for gate in active_gates
    )
    gated_turns = sum(
        int(gate.get("gated_turn_count", 0) or 0) for gate in active_gates
    )
    turn1_sampled = [gate for gate in active_gates if gate.get("turn1_sampled")]
    complaint_episodes = [
        gate for gate in active_gates if gate.get("first_complaint_turn") is not None
    ]
    first_post_checked = [
        gate
        for gate in complaint_episodes
        if gate.get("first_post_complaint_passed") is not None
    ]
    post_sampled_turns = sum(
        int(gate.get("post_complaint_sampled_turn_count", 0) or 0)
        for gate in complaint_episodes
    )
    post_passed_turns = sum(
        int(gate.get("post_complaint_passed_turn_count", 0) or 0)
        for gate in complaint_episodes
    )
    post_sustained = [
        gate
        for gate in complaint_episodes
        if gate.get("post_complaint_all_passed") is not None
    ]
    episode_compliances = [
        float(gate["compliance"])
        for gate in active_gates
        if gate.get("compliance") is not None
    ]
    personality_gate = {
        "active_episode_count": len(active_gates),
        "ungated_episode_count": len(gate_payloads) - len(active_gates),
        "personalities": dict(
            sorted(
                Counter(
                    str(gate.get("personality") or "none") for gate in gate_payloads
                ).items()
            )
        ),
        "eligible_turn_count": sum(
            int(gate.get("eligible_turn_count", 0) or 0) for gate in active_gates
        ),
        "sampled_turn_count": sampled_turns,
        "passed_turn_count": passed_turns,
        "gated_turn_count": gated_turns,
        "gate_error_count": sum(
            int(gate.get("gate_error_count", 0) or 0) for gate in active_gates
        ),
        "gate_error_episode_count": sum(
            int(gate.get("gate_error_count", 0) or 0) > 0 for gate in active_gates
        ),
        "micro_compliance": _rate(passed_turns, sampled_turns),
        "mean_episode_compliance": (
            sum(episode_compliances) / len(episode_compliances)
            if episode_compliances
            else None
        ),
        "turn1_sampled_episode_count": len(turn1_sampled),
        "turn1_passed_episode_count": sum(
            gate.get("turn1_passed") is True for gate in turn1_sampled
        ),
        "turn1_compliance": _rate(
            sum(gate.get("turn1_passed") is True for gate in turn1_sampled),
            len(turn1_sampled),
        ),
        "complaint_episode_count": len(complaint_episodes),
        "complaint_episode_rate": _rate(len(complaint_episodes), len(active_gates)),
        "bare_complaint_count": sum(
            int(gate.get("bare_complaint_count", 0) or 0) for gate in active_gates
        ),
        "explain_complaint_count": sum(
            int(gate.get("explain_complaint_count", 0) or 0) for gate in active_gates
        ),
        "unknown_complaint_count": sum(
            int(gate.get("unknown_complaint_count", 0) or 0) for gate in active_gates
        ),
        "first_post_complaint_checked_episode_count": len(first_post_checked),
        "first_post_complaint_passed_episode_count": sum(
            gate.get("first_post_complaint_passed") is True
            for gate in first_post_checked
        ),
        "first_post_complaint_compliance": _rate(
            sum(
                gate.get("first_post_complaint_passed") is True
                for gate in first_post_checked
            ),
            len(first_post_checked),
        ),
        "post_complaint_sampled_turn_count": post_sampled_turns,
        "post_complaint_passed_turn_count": post_passed_turns,
        "post_complaint_micro_compliance": _rate(post_passed_turns, post_sampled_turns),
        "post_complaint_sustained_episode_count": sum(
            gate.get("post_complaint_all_passed") is True for gate in post_sustained
        ),
        "post_complaint_sustained_rate": _rate(
            sum(
                gate.get("post_complaint_all_passed") is True for gate in post_sustained
            ),
            len(post_sustained),
        ),
    }
    classification_label_counts = Counter()
    for gate in active_gates:
        classification_label_counts.update(
            {
                str(label): int(count)
                for label, count in (
                    gate.get("classification_label_counts") or {}
                ).items()
            }
        )
    if classification_label_counts:
        personality_gate["classification_label_counts"] = dict(
            sorted(classification_label_counts.items())
        )
    classification_distribution_count = sum(
        int(gate.get("classification_distribution_count", 0) or 0)
        for gate in active_gates
    )
    if classification_distribution_count:
        probability_sums: Counter[str] = Counter()
        margin_sum = 0.0
        margin_count = 0
        for gate in active_gates:
            count = int(gate.get("classification_distribution_count", 0) or 0)
            if count <= 0:
                continue
            probability_sums.update(
                {
                    str(label): float(probability) * count
                    for label, probability in (
                        gate.get("mean_classification_probabilities") or {}
                    ).items()
                }
            )
            if gate.get("mean_classification_margin") is not None:
                margin_sum += float(gate["mean_classification_margin"]) * count
                margin_count += count
        personality_gate["classification_distribution_count"] = (
            classification_distribution_count
        )
        personality_gate["mean_classification_probabilities"] = {
            label: total / classification_distribution_count
            for label, total in sorted(probability_sums.items())
        }
        if margin_count:
            personality_gate["mean_classification_margin"] = margin_sum / margin_count

    return {
        "expected_attempts": int(expected),
        "recorded_attempts": len(results),
        "completed_attempts": len(completed),
        "error_count": len(results) - len(completed),
        "recorded_coverage_rate": _rate(len(results), expected),
        "execution_coverage_rate": _rate(len(completed), expected),
        "termination_histogram": dict(
            sorted(Counter(result.termination_reason for result in results).items())
        ),
        "pre_solved_count": pre_solved,
        "pre_solved_rate_on_covered": _rate(pre_solved, covered),
        "taught_success_count": taught_success,
        "teaching_solve_rate_observed_initially_unsolved": _rate(
            taught_success, observed_initially_unsolved
        ),
        "teaching_solve_rate_clean_student_calls": _rate(
            clean_taught_success, clean_initially_unsolved
        ),
        "teaching_solve_rate_clean_aux_calls": _rate(
            clean_aux_taught_success, clean_aux_initially_unsolved
        ),
        "teaching_solve_rate_full_set_conservative": _rate(
            taught_success, max(0, expected - pre_solved)
        ),
        "teaching_lift_full_set": _rate(taught_success, expected),
        "outcome_score_sum": sum(outcome_scores),
        "outcome_score_mean_completed": _rate(sum(outcome_scores), len(completed)),
        "outcome_score_mean_full_set": _rate(sum(outcome_scores), expected),
        "final_correct_count": final_correct,
        "full_set_final_correct_rate": _rate(final_correct, expected),
        "completed_final_correct_rate": _rate(final_correct, len(completed)),
        "final_correct_score_sum": sum(final_correct_scores),
        "regular_eval_score_mean_completed": _rate(
            sum(final_correct_scores), len(completed)
        ),
        "regular_eval_score_mean_full_set": _rate(sum(final_correct_scores), expected),
        "no_teaching_baseline_mean": (
            _rate(sum(pair[0] for pair in baseline_pairs), len(baseline_pairs))
            if baseline_pairs
            else None
        ),
        "improvement_over_no_teaching_baseline_mean": (
            _rate(
                sum(outcome - baseline for baseline, outcome in baseline_pairs),
                len(baseline_pairs),
            )
            if baseline_pairs
            else None
        ),
        "workflow_covered_count": covered,
        "workflow_coverage_rate_full_set": _rate(covered, expected),
        "presolve_covered_count": covered if presolve_enabled else None,
        "presolve_coverage_rate": (
            _rate(covered, len(completed)) if presolve_enabled else None
        ),
        "presolve_coverage_rate_full_set": (
            _rate(covered, expected) if presolve_enabled else None
        ),
        "presolve_skipped_count": pre_solve_skipped,
        "presolve_accepted_count": presolve_accepted,
        "presolve_acceptance_rate": _rate(presolve_accepted, len(presolve_results)),
        "presolve_acceptance_rate_full_set": (
            _rate(presolve_accepted, expected) if presolve_enabled else None
        ),
        "teacher_pre_error_count": sum(
            result.teacher_pre_error_count for result in results
        ),
        "leaked_episode_count": sum(result.leak_count > 0 for result in completed),
        "clean_nonleaky_taught_success_count": sum(
            result.taught_success and result.leak_count == 0 for result in completed
        ),
        "leak_check_failed_episode_count": sum(
            result.leak_check_failed_count > 0 for result in completed
        ),
        "answer_judge_used_count": sum(
            result.answer_judge_used_count for result in completed
        ),
        "answer_judge_failed_episode_count": sum(
            result.answer_judge_failed_count > 0 for result in completed
        ),
        "answer_judge_override_correct_count": sum(
            result.answer_judge_override_correct_count for result in completed
        ),
        "format_error_episode_count": sum(
            result.format_error_count > 0 for result in completed
        ),
        "student_call_failed_count": sum(
            result.student_call_failed for result in completed
        ),
        "avg_turns_completed": (
            sum(result.num_turns for result in completed) / len(completed)
            if completed
            else None
        ),
        "avg_solve_turn": (
            sum(solve_turns) / len(solve_turns) if solve_turns else None
        ),
        "free_chat_episode_count": sum(
            result.free_chat_enabled for result in completed
        ),
        "personality_gate": personality_gate,
        "code_channel": code_channel,
        "generalization": generalization,
    }


def aggregate_repeat_metrics(
    results: list[EpisodeResult],
    *,
    expected_items: int,
    attempts: int,
) -> dict[str, Any]:
    """Match the grouped evaluator's per-task repeat stability metrics."""

    expected_attempts = set(range(1, attempts + 1))
    by_item: dict[int, dict[int, EpisodeResult]] = {}
    for result in latest_results(results):
        by_item.setdefault(result.dataset_index, {})[result.attempt] = result

    complete_items = [
        [attempt_results[attempt] for attempt in range(1, attempts + 1)]
        for attempt_results in by_item.values()
        if set(attempt_results) == expected_attempts
        and all(result.error is None for result in attempt_results.values())
    ]
    error_items = sum(
        any(result.error is not None for result in attempt_results.values())
        for attempt_results in by_item.values()
    )

    def summarize(outcome: str) -> dict[str, float | None]:
        values_by_item = [
            [
                float(
                    result.taught_success
                    if outcome == "solved"
                    else result.final_correct
                )
                for result in item_results
            ]
            for item_results in complete_items
        ]
        if not values_by_item:
            return {
                key: None
                for key in (
                    "mean",
                    "variance",
                    "std",
                    "agreement",
                    "disagreement",
                    "all_equal",
                    "any_success",
                    "all_success",
                    "success_set_intersection",
                    "success_set_union",
                    "success_set_jaccard",
                    "pairwise_success_set_jaccard",
                )
            }

        per_item = [_binary_repeat_summary(values) for values in values_by_item]
        summary = {
            key: sum(item[key] for item in per_item) / len(per_item)
            for key in (
                "mean",
                "variance",
                "std",
                "agreement",
                "disagreement",
                "all_equal",
                "any_success",
                "all_success",
                "success_set_intersection",
                "success_set_union",
            )
        }
        success_union = sum(item["success_set_union"] for item in per_item)
        summary["success_set_jaccard"] = (
            sum(item["success_set_intersection"] for item in per_item) / success_union
            if success_union
            else None
        )

        pairwise_intersection = 0.0
        pairwise_union = 0.0
        for values in values_by_item:
            repeat_pairs = list(combinations(values, 2))
            if not repeat_pairs:
                repeat_pairs = [(values[0], values[0])]
            for left, right in repeat_pairs:
                if left or right:
                    pairwise_intersection += float(left and right)
                    pairwise_union += 1.0
        summary["pairwise_success_set_jaccard"] = (
            pairwise_intersection / pairwise_union if pairwise_union else None
        )
        return summary

    complete_item_count = len(complete_items)
    return {
        "expected_item_count": int(expected_items),
        "recorded_item_count": len(by_item),
        "complete_item_count": complete_item_count,
        "incomplete_item_count": max(0, int(expected_items) - complete_item_count),
        "error_item_count": error_items,
        "complete_item_coverage_rate": _rate(complete_item_count, expected_items),
        "solved": summarize("solved"),
        "final_correct": summarize("final_correct"),
    }


def aggregate_report(
    results: list[EpisodeResult],
    *,
    modes: list[PresolveMode],
    dataset_size: int,
    attempts: int,
    generalization_enabled: bool = False,
    generalization_levels: tuple[str, ...] | None = None,
    student_prompt_rows: dict[tuple[str, int], int] | None = None,
) -> dict[str, Any]:
    results = latest_results(results)
    persona_row_count = sum((student_prompt_rows or {}).values())
    base_prompt_rows = dataset_size - persona_row_count
    if base_prompt_rows < 0:
        raise ValueError(
            "Student prompt row counts exceed the evaluation dataset size."
        )
    expected_per_mode = base_prompt_rows * attempts
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        all_mode_results = [result for result in results if result.mode == mode.name]
        mode_results = (
            [
                result
                for result in all_mode_results
                if result.student_prompt_index is None
            ]
            if student_prompt_rows
            else all_mode_results
        )
        mode_summary = {
            **aggregate_mode(
                mode_results,
                expected=expected_per_mode,
                generalization_enabled=generalization_enabled,
                generalization_levels=generalization_levels,
            ),
            **aggregate_items(mode_results, expected_items=base_prompt_rows),
            "repeat": aggregate_repeat_metrics(
                mode_results,
                expected_items=base_prompt_rows,
                attempts=attempts,
            ),
        }
        if student_prompt_rows:
            pool_summaries: dict[str, dict[str, Any]] = {}
            for (pool, prompt_index), row_count in sorted(student_prompt_rows.items()):
                prompt_results = [
                    result
                    for result in all_mode_results
                    if result.student_prompt_pool == pool
                    and result.student_prompt_index == prompt_index
                ]
                pool_summaries.setdefault(pool, {})[str(prompt_index)] = {
                    "dataset_rows": row_count,
                    **aggregate_mode(
                        prompt_results,
                        expected=row_count * attempts,
                        generalization_enabled=generalization_enabled,
                        generalization_levels=generalization_levels,
                    ),
                    **aggregate_items(prompt_results, expected_items=row_count),
                    "repeat": aggregate_repeat_metrics(
                        prompt_results,
                        expected_items=row_count,
                        attempts=attempts,
                    ),
                }
            mode_summary["student_prompts"] = pool_summaries
        mode_summaries[mode.name] = mode_summary

    return {
        "dataset_rows": dataset_size,
        "base_prompt_rows": base_prompt_rows,
        "attempts_per_row": attempts,
        "expected_total_attempts": dataset_size * attempts * len(modes),
        "recorded_total_attempts": len(results),
        "modes": mode_summaries,
    }


def aggregate_items(
    results: list[EpisodeResult], *, expected_items: int
) -> dict[str, Any]:
    """Report empirical any-success rates when multiple attempts are requested."""

    by_item: dict[int, list[EpisodeResult]] = {}
    for result in latest_results(results):
        by_item.setdefault(result.dataset_index, []).append(result)
    any_taught = sum(
        any(result.taught_success for result in item_results)
        for item_results in by_item.values()
    )
    any_final = sum(
        any(result.final_correct for result in item_results)
        for item_results in by_item.values()
    )
    return {
        "recorded_item_count": len(by_item),
        "item_any_taught_success_count": any_taught,
        "item_any_taught_success_rate_full_set": _rate(any_taught, expected_items),
        "item_any_final_correct_count": any_final,
        "item_any_final_correct_rate_full_set": _rate(any_final, expected_items),
    }


def student_prompt_row_counts(dataset: Any) -> dict[tuple[str, int], int]:
    """Count expanded validation rows for each forced student prompt."""

    required = {
        TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
        TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    }
    if not required.issubset(dataset.column_names):
        return {}
    return dict(
        sorted(
            Counter(
                (str(pool), int(prompt_index))
                for pool, prompt_index in zip(
                    dataset[TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD],
                    dataset[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD],
                    strict=True,
                )
                if prompt_index is not None
            ).items()
        )
    )


def redact_sensitive(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(
        token in lowered
        for token in (
            "api_key",
            "api-key",
            "authorization",
            "inference-key",
            "secret",
        )
    ):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            item_key: redact_sensitive(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def build_run_signature(
    *,
    args: argparse.Namespace,
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    modes: list[PresolveMode],
    dataset_size: int,
    dataset_hash: str,
    attempts: int,
    teacher_base_url: str,
    teacher_request_params: dict[str, Any],
    workflow_kwargs_by_mode: dict[str, dict[str, Any]],
    student_prompts: tuple[tutor_train.EvalStudentPrompt, ...] = (),
) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    effective_kwargs = next(iter(workflow_kwargs_by_mode.values()))
    return redact_sensitive(
        {
            "config": str(config_path),
            "config_sha256": config_hash,
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "overrides": list(args.overrides),
            "dataset_size": dataset_size,
            "dataset_sha256": dataset_hash,
            "dataset_selection": {
                "strategy": (
                    "math_type_level_stratified"
                    if int(args.stratified_max_samples) > 0
                    else "config"
                ),
                "stratified_max_samples": int(args.stratified_max_samples),
            },
            "attempts": attempts,
            "modes": [asdict(mode) for mode in modes],
            "presolve": {
                "verify": bool(effective_kwargs["teacher_pre_verify"]),
                "attempts": int(effective_kwargs["teacher_pre_attempts"]),
                "max_tokens": int(effective_kwargs["teacher_pre_max_tokens"]),
            },
            "teacher": {
                "base_url": teacher_base_url,
                "model": args.teacher_model,
                "temperature": args.teacher_temperature,
                "top_p": args.teacher_top_p,
                "max_tokens": args.teacher_max_tokens,
                "timeout": args.teacher_timeout,
                "request_params": teacher_request_params,
                "test_flow_budget": {
                    "tokenizer_path": config.tokenizer_path,
                    "model_context_length": config.sglang.context_length,
                    "max_train_sample_tokens": config.gconfig.max_tokens,
                    "note": (
                        "Preserved from the original test flow; token counts use "
                        "the experiment tokenizer rather than the API model tokenizer."
                    ),
                },
            },
            "reliability": {
                "episode_error_retries": int(args.episode_error_retries),
                "episode_error_retry_backoff_seconds": float(
                    args.episode_error_retry_backoff_seconds
                ),
                "retry_diagnostic_failures": bool(args.retry_diagnostic_failures),
            },
            "auxiliary": {
                "source_mode": config.auxiliary_model.mode,
                "effective_mode": effective_kwargs["aux_mode"],
                "base_url": effective_kwargs["aux_base_url"],
                "model": effective_kwargs["aux_model"],
                "temperature": config.auxiliary_model.temperature,
                "top_p": config.auxiliary_model.top_p,
                "max_tokens": config.auxiliary_model.max_tokens,
                "timeout": config.auxiliary_model.timeout,
                "max_concurrent_calls": effective_kwargs["max_concurrent_aux_calls"],
                "request_params": effective_kwargs["aux_request_params"],
            },
            "students": student_models,
            "test_semantics": {
                "dataset_type": config.dataset_type,
                "answer_scorer": config.answer_scorer,
                "max_turns": effective_kwargs["max_turns"],
                "leak_handling_mode": effective_kwargs["leak_handling_mode"],
                "format_handling_mode": effective_kwargs["format_handling_mode"],
                "free_chat": effective_kwargs["free_chat"],
                "teacher_history_tags": effective_kwargs["teacher_history_tags"],
                "teacher_show_ground_truth": config.teacher_show_ground_truth,
                "teacher_anti_leak_instruction_enabled": (
                    config.teacher_anti_leak_instruction_enabled
                ),
                "teacher_adaptive_instruction_enabled": (
                    config.teacher_adaptive_instruction_enabled
                ),
                "personality": effective_kwargs["personality"],
                "student_generalize_enabled": effective_kwargs[
                    "student_generalize_enabled"
                ],
                "student_generalize_mode": effective_kwargs["student_generalize_mode"],
                "student_generalize_source": effective_kwargs[
                    "student_generalize_source"
                ],
                "student_generalize_replays": effective_kwargs[
                    "student_generalize_replays"
                ],
                "student_generalize_retest_original": effective_kwargs[
                    "student_generalize_retest_original"
                ],
                "student_generalize_level1_enabled": effective_kwargs[
                    "student_generalize_level1_enabled"
                ],
                "student_generalize_level2_enabled": effective_kwargs[
                    "student_generalize_level2_enabled"
                ],
                "generalization_levels": list(
                    configured_generalization_levels(effective_kwargs)
                ),
                **(
                    {
                        "student_prompt_pools": {
                            "paths": config.prompt_pool.student_eval_paths,
                            "count": len(student_prompts),
                            "sha256": hashlib.sha256(
                                json.dumps(
                                    [asdict(prompt) for prompt in student_prompts],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        }
                    }
                    if student_prompts
                    else {}
                ),
            },
        }
    )


def resolve_output_dir(args: argparse.Namespace, config: TutorConfig) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return (
        Path(config.cluster.fileroot)
        / "api_teacher_eval"
        / safe_path_token(config.experiment_name)
        / safe_path_token(config.trial_name)
        / timestamp
    ).resolve()


def prepare_output_dir(
    output_dir: Path,
    *,
    signature: dict[str, Any],
    resume: bool,
    allow_evaluator_code_change: bool = False,
) -> None:
    signature_path = output_dir / "run_config.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. Use --resume or choose "
                "a new directory."
            )
        if not signature_path.exists():
            raise ValueError(f"Resume directory is missing {signature_path.name}.")
        previous = json.loads(signature_path.read_text(encoding="utf-8"))
        previous_signature = previous.get("signature")
        if previous_signature != signature:
            compatible_code_change = False
            if allow_evaluator_code_change and isinstance(previous_signature, dict):
                previous_without_code = dict(previous_signature)
                current_without_code = dict(signature)
                previous_hash = previous_without_code.pop("evaluator_sha256", None)
                current_hash = current_without_code.pop("evaluator_sha256", None)
                compatible_code_change = (
                    previous_hash is not None
                    and current_hash is not None
                    and previous_hash != current_hash
                    and previous_without_code == current_without_code
                )
            if not compatible_code_change:
                raise ValueError(
                    "Resume settings differ from the existing run_config.json."
                )
            logger.warning(
                "Resuming across an explicitly allowed evaluator code change; "
                "all other run signature fields match exactly (old_sha256=%s, "
                "new_sha256=%s).",
                previous_hash,
                current_hash,
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        signature_path,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "signature": signature,
        },
    )


async def run_episode(
    *,
    spec: EpisodeSpec,
    workflow_kwargs: dict[str, Any],
    teacher_client: ApiTeacherClient,
    output_dir: Path,
    save_traces: str,
    keep_env_proxy: bool,
    execution_try: int = 1,
    generalization_levels: tuple[str, ...] = (),
    expected_generalization_replays: int = 0,
    episode_timeout_seconds: float = 300.0,
) -> EpisodeResult:
    started = time.monotonic()
    workflow: RecordingTutorWorkflow | None = None
    try:
        episode_workflow_kwargs = prepare_episode_workflow_kwargs(workflow_kwargs)
        with without_proxy_environment(enabled=not keep_env_proxy):
            workflow = RecordingTutorWorkflow(**episode_workflow_kwargs)
        workflow_context.set(
            WorkflowContext(
                is_eval=True,
                task_id=spec.dataset_index,
                lora_version=None,
            )
        )
        # Client-level timeouts do not protect the whole workflow: one episode
        # fans out into teacher, student, judge, replay, and code-execution calls.
        # Keep a hard outer deadline so one transport task can never hold a cell
        # (and the other GPU pair at its barrier) indefinitely.
        async with asyncio.timeout(float(episode_timeout_seconds)):
            await workflow._run_episode(
                dict(spec.row),
                external_client=teacher_client,
            )
        result = result_from_workflow(
            workflow=workflow,
            spec=spec,
            duration_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        result = error_result(
            spec,
            exc,
            duration_seconds=time.monotonic() - started,
        )
    finally:
        if workflow is not None:
            await close_workflow_api_clients(workflow)

    should_trace = save_traces == "all" or (
        save_traces == "errors"
        and result_needs_retry(
            result,
            retry_errors=True,
            retry_diagnostic_failures=True,
            generalization_levels=generalization_levels,
            expected_generalization_replays=expected_generalization_replays,
        )
    )
    if should_trace:
        retry_suffix = "" if execution_try <= 1 else f"_retry_{execution_try - 1:02d}"
        trace_path = (
            output_dir
            / "traces"
            / spec.mode.name
            / (
                f"row_{spec.dataset_index:05d}_id_{safe_path_token(result.item_id)}_"
                f"attempt_{spec.attempt:02d}{retry_suffix}.json"
            )
        )
        if result.error is None and workflow is not None:
            trace_payload = build_trace_payload(
                workflow=workflow,
                spec=spec,
                result=result,
            )
        else:
            trace_payload = {
                "result": asdict(result),
                "dataset_row": spec.row,
            }
        result.trace_path = str(trace_path)
        trace_payload["result"]["trace_path"] = str(trace_path)
        await asyncio.to_thread(write_json, trace_path, trace_payload)
    return result


async def preflight_teacher(client: ApiTeacherClient, model: str) -> None:
    models = await client.list_models()
    normalized = {item.lower() for item in models}
    if model.lower() not in normalized:
        raise ValueError(
            f"Teacher model {model!r} was not returned by /v1/models: {models}."
        )
    logger.info("DeepSeek endpoint ready; available models=%s", models)


async def run_all(
    *,
    specs: list[EpisodeSpec],
    completed_keys: set[str],
    workflow_kwargs_by_mode: dict[str, dict[str, Any]],
    teacher_client: ApiTeacherClient,
    output_dir: Path,
    save_traces: str,
    concurrency: int,
    log_every: int,
    keep_env_proxy: bool,
    error_retries: int = 0,
    retry_backoff_seconds: float = 1.0,
    retry_diagnostic_failures: bool = False,
    episode_timeout_seconds: float = 300.0,
) -> list[EpisodeResult]:
    pending = [spec for spec in specs if spec.key not in completed_keys]
    if not pending:
        logger.info(
            "All %s attempts are already present; nothing to resume.", len(specs)
        )
        return []

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    write_lock = asyncio.Lock()
    results_path = output_dir / "results.jsonl"
    retry_events_path = output_dir / "retry_events.jsonl"
    previous_results = (
        [
            result
            for result in latest_results(load_existing_results(results_path))
            if result.key in completed_keys
        ]
        if results_path.exists()
        else []
    )
    processed = 0
    error_count = sum(result.error is not None for result in previous_results)
    diagnostic_failure_count = sum(
        result.student_call_failed
        or result.leak_check_failed_count > 0
        or result.answer_judge_failed_count > 0
        or result.teacher_pre_error_count > 0
        for result in previous_results
    )
    mode_counts: Counter[str] = Counter(result.mode for result in previous_results)
    progress = tqdm(
        total=len(specs),
        initial=len(specs) - len(pending),
        desc="Tutor API eval",
        unit="episode",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    async def _run(spec: EpisodeSpec) -> EpisodeResult:
        nonlocal diagnostic_failure_count, error_count, processed
        result: EpisodeResult | None = None
        total_execution_tries = max(0, int(error_retries)) + 1
        workflow_kwargs = workflow_kwargs_by_mode[spec.mode.name]
        generalization_levels = configured_generalization_levels(workflow_kwargs)
        expected_generalization_replays = int(
            workflow_kwargs.get("student_generalize_replays", 0) or 0
        )
        async with semaphore:
            for execution_try in range(1, total_execution_tries + 1):
                result = await run_episode(
                    spec=spec,
                    workflow_kwargs=workflow_kwargs,
                    teacher_client=teacher_client,
                    output_dir=output_dir,
                    save_traces=save_traces,
                    keep_env_proxy=keep_env_proxy,
                    execution_try=execution_try,
                    generalization_levels=generalization_levels,
                    expected_generalization_replays=expected_generalization_replays,
                    episode_timeout_seconds=episode_timeout_seconds,
                )
                reasons = result_retry_reasons(
                    result,
                    retry_errors=True,
                    retry_diagnostic_failures=retry_diagnostic_failures,
                    generalization_levels=generalization_levels,
                    expected_generalization_replays=expected_generalization_replays,
                )
                if not reasons:
                    break

                will_retry = execution_try < total_execution_tries
                retry_event = {
                    "attempt": spec.attempt,
                    "dataset_index": spec.dataset_index,
                    "episode_key": spec.key,
                    "execution_try": execution_try,
                    "item_id": result.item_id,
                    "max_execution_tries": total_execution_tries,
                    "reasons": reasons,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "will_retry": will_retry,
                }
                async with write_lock:
                    await asyncio.to_thread(
                        append_jsonl, retry_events_path, retry_event
                    )
                if not will_retry:
                    logger.error(
                        "Episode %s exhausted %s execution tries; recording it for "
                        "later backfill and continuing: %s",
                        spec.key,
                        total_execution_tries,
                        "; ".join(reasons),
                    )
                    break

                delay = max(0.0, float(retry_backoff_seconds)) * (
                    2 ** (execution_try - 1)
                )
                logger.warning(
                    "Episode %s failed execution try %s/%s; retrying in %.1fs: %s",
                    spec.key,
                    execution_try,
                    total_execution_tries,
                    delay,
                    "; ".join(reasons),
                )
                if delay:
                    await asyncio.sleep(delay)
        assert result is not None
        async with write_lock:
            await asyncio.to_thread(append_jsonl, results_path, asdict(result))
            processed += 1
            error_count += int(result.error is not None)
            diagnostic_failure_count += int(
                result.student_call_failed
                or result.leak_check_failed_count > 0
                or result.answer_judge_failed_count > 0
                or result.teacher_pre_error_count > 0
            )
            mode_counts[result.mode] += 1
            progress.update(1)
            progress.set_postfix(
                errors=error_count,
                diagnostic=diagnostic_failure_count,
                off=mode_counts["presolve_off"],
                on=mode_counts["presolve_on"],
                refresh=False,
            )
            if processed == len(pending) or processed % max(1, log_every) == 0:
                logger.info(
                    "Completed %s/%s pending attempts (termination=%s, id=%s)",
                    processed,
                    len(pending),
                    result.termination_reason,
                    result.item_id,
                )
        return result

    try:
        return list(await asyncio.gather(*[_run(spec) for spec in pending]))
    finally:
        progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TutorAgentWorkflow test split without training, using an "
            "OpenAI-compatible DeepSeek teacher while preserving the source config's "
            "Qwen auxiliary model and API student."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--teacher-base-url",
        default=os.getenv("DEEPSEEK_BASE_URL", ""),
        help="DeepSeek OpenAI-compatible endpoint, with or without the /v1 suffix.",
    )
    parser.add_argument("--teacher-model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument(
        "--api-key",
        default="",
        help="Prefer the DEEPSEEK_API_KEY environment variable over this option.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=None,
        help="Defaults to eval_gconfig.temperature from --config.",
    )
    parser.add_argument(
        "--teacher-top-p",
        type=float,
        default=None,
        help="Defaults to eval_gconfig.top_p from --config.",
    )
    parser.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=None,
        help="Defaults to eval_gconfig.max_new_tokens from --config.",
    )
    parser.add_argument("--teacher-timeout", type=float, default=300.0)
    parser.add_argument("--teacher-request-params", default="")
    parser.add_argument("--teacher-request-params-file", type=Path, default=None)
    parser.add_argument(
        "--self-aux-via-teacher",
        action="store_true",
        help=(
            "For a source config with auxiliary_model.mode=self, reproduce regular "
            "validation by routing auxiliary calls to this external teacher "
            "endpoint with the same request params (including lora_path). Required "
            "when evaluating an actor checkpoint whose self judges must use that "
            "same checkpoint."
        ),
    )
    parser.add_argument(
        "--teacher-presolve",
        choices=["config", "off", "on", "both"],
        default="config",
    )
    parser.add_argument("--presolve-attempts", type=int, default=0)
    parser.add_argument("--presolve-max-tokens", type=int, default=None)
    parser.add_argument(
        "--student-generalization",
        choices=["config", "off", "on"],
        default="config",
        help="'config' preserves the configured original retest and transfer probes.",
    )
    parser.add_argument(
        "--student-name",
        action="append",
        default=[],
        help=(
            "Evaluate only this configured student; repeat for multiple students. "
            "By default all configured students are evaluated."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=0,
        help="Attempts per test row; 0 uses evaluator.average_rollouts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Base validation-item limit before student and prompt expansion.",
    )
    parser.add_argument(
        "--stratified-max-samples",
        type=int,
        default=0,
        help=(
            "Select this many base rows proportionally by metadata.type x "
            "metadata.level using the config seed; 0 disables stratification."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument(
        "--episode-error-retries",
        type=int,
        default=0,
        help=(
            "Retry a whole episode this many additional times after an API or "
            "diagnostic infrastructure failure."
        ),
    )
    parser.add_argument(
        "--episode-error-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial whole-episode retry delay; subsequent delays double.",
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "Hard wall-clock limit for one whole episode, including teaching, "
            "judging, and replays. A timeout is handled by the normal whole-episode "
            "retry/backfill policy."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-evaluator-code-change-on-resume",
        action="store_true",
        help=(
            "Allow --resume only when evaluator_sha256 is the sole run-signature "
            "difference. All dataset, model, prompt, and generation settings must "
            "still match exactly."
        ),
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On resume, rerun error records by default; use --no-retry-errors to "
            "keep them."
        ),
    )
    parser.add_argument(
        "--retry-diagnostic-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retry whole episodes with student, leak-check, answer-judge, or "
            "teacher-presolve call failures, and incomplete generalization "
            "replays, both within a run and on resume. Disabled by default to "
            "avoid conditional resampling."
        ),
    )
    parser.add_argument(
        "--save-traces",
        choices=["all", "errors", "none"],
        default="all",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--keep-env-proxy",
        action="store_true",
        help=(
            "Let OpenAI clients inherit HTTP(S)/ALL_PROXY. By default clients are "
            "created with those variables temporarily cleared, then the environment "
            "is restored."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label in (
        "teacher_max_tokens",
        "concurrency",
    ):
        if int(getattr(args, label)) <= 0:
            raise ValueError(f"--{label.replace('_', '-')} must be positive.")
    if args.attempts < 0:
        raise ValueError("--attempts must be non-negative.")
    if args.stratified_max_samples < 0:
        raise ValueError("--stratified-max-samples must be non-negative.")
    if args.stratified_max_samples > 0 and args.limit > 0:
        raise ValueError("Use either --stratified-max-samples or --limit, not both.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")
    if args.episode_error_retries < 0:
        raise ValueError("--episode-error-retries must be non-negative.")
    if args.episode_error_retry_backoff_seconds < 0:
        raise ValueError("--episode-error-retry-backoff-seconds must be non-negative.")
    if args.episode_timeout_seconds <= 0:
        raise ValueError("--episode-timeout-seconds must be positive.")
    if args.presolve_attempts < 0:
        raise ValueError("--presolve-attempts must be non-negative.")
    if args.presolve_max_tokens is not None and args.presolve_max_tokens < 0:
        raise ValueError("--presolve-max-tokens must be non-negative.")
    if float(args.teacher_temperature) < 0.0:
        raise ValueError("--teacher-temperature must be non-negative.")
    if args.teacher_top_p is not None and not 0.0 < float(args.teacher_top_p) <= 1.0:
        raise ValueError("--teacher-top-p must be in (0, 1].")


def resolve_teacher_generation_args(
    args: argparse.Namespace, config: TutorConfig
) -> None:
    eval_gconfig = config.eval_gconfig or config.gconfig
    if args.teacher_temperature is None:
        args.teacher_temperature = float(eval_gconfig.temperature)
    if args.teacher_top_p is None:
        args.teacher_top_p = eval_gconfig.top_p
    if args.teacher_max_tokens is None:
        args.teacher_max_tokens = int(eval_gconfig.max_new_tokens)


async def main_async(args: argparse.Namespace) -> None:
    teacher_base_url = normalize_base_url(args.teacher_base_url)
    env_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    teacher_api_key = env_api_key or args.api_key or "EMPTY"

    config, student_models = load_experiment_config(args.config, args.overrides)
    tutor_train._apply_eval_average_rollouts(config)
    resolve_teacher_generation_args(args, config)
    validate_args(args)
    student_models = select_student_models(student_models, args.student_name)
    config.student_generalize.enabled = resolve_generalization(
        args.student_generalization,
        config.student_generalize.enabled,
    )
    if config.student_generalize.enabled:
        tutor_train._prepare_math_generalization_data(config)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    # No pre-flight validation any more. prepare_test_dataset below filters the
    # split to the rows carrying every enabled transfer level, and raises only if
    # that leaves nothing.

    student_prompts = tutor_train._load_eval_student_prompts(config)
    dataset = prepare_test_dataset(
        config,
        student_models,
        tokenizer=tokenizer,
        limit=max(0, int(args.limit)),
        stratified_max_samples=int(args.stratified_max_samples),
        student_prompts=student_prompts,
    )
    prompt_row_counts = student_prompt_row_counts(dataset)
    modes = resolve_presolve_modes(
        args.teacher_presolve,
        effective_eval_presolve_enabled(config),
    )
    attempts = (
        int(args.attempts)
        if int(args.attempts) > 0
        else max(1, int(config.evaluator.average_rollouts))
    )

    teacher_request_params = merge_dicts(
        deepseek_non_thinking_params(config.seed),
        load_request_params(
            args.teacher_request_params,
            args.teacher_request_params_file,
            label="--teacher-request-params",
        ),
    )

    external_self_aux = (
        {
            "base_url": teacher_base_url,
            "model": args.teacher_model,
            "api_key": teacher_api_key,
            "request_params": teacher_request_params,
        }
        if args.self_aux_via_teacher
        else None
    )
    workflow_kwargs_by_mode = {
        mode.name: build_eval_workflow_kwargs(
            config=config,
            student_models=student_models,
            tokenizer=tokenizer,
            args=args,
            presolve_enabled=mode.enabled,
            external_self_aux=external_self_aux,
        )
        for mode in modes
    }
    retry_workflow_kwargs = next(iter(workflow_kwargs_by_mode.values()))
    retry_generalization_levels = configured_generalization_levels(
        retry_workflow_kwargs
    )
    retry_generalization_replays = int(
        retry_workflow_kwargs.get("student_generalize_replays", 0) or 0
    )
    signature = build_run_signature(
        args=args,
        config=config,
        student_models=student_models,
        modes=modes,
        dataset_size=len(dataset),
        dataset_hash=dataset_sha256(dataset),
        attempts=attempts,
        teacher_base_url=teacher_base_url,
        teacher_request_params=teacher_request_params,
        workflow_kwargs_by_mode=workflow_kwargs_by_mode,
        student_prompts=student_prompts,
    )
    output_dir = resolve_output_dir(args, config)
    prepare_output_dir(
        output_dir,
        signature=signature,
        resume=args.resume,
        allow_evaluator_code_change=args.allow_evaluator_code_change_on_resume,
    )
    results_path = output_dir / "results.jsonl"
    existing_results = load_existing_results(results_path) if args.resume else []
    completed_keys = {
        result.key
        for result in existing_results
        if not result_needs_retry(
            result,
            retry_errors=args.retry_errors,
            retry_diagnostic_failures=args.retry_diagnostic_failures,
            generalization_levels=retry_generalization_levels,
            expected_generalization_replays=retry_generalization_replays,
        )
    }

    specs = [
        EpisodeSpec(
            mode=mode,
            dataset_index=index,
            attempt=attempt,
            row=dict(dataset[index]),
        )
        for mode in modes
        for index in range(len(dataset))
        for attempt in range(1, attempts + 1)
    ]

    async def _run_api_phase() -> list[EpisodeResult]:
        teacher_client = ApiTeacherClient(
            base_url=teacher_base_url,
            api_key=teacher_api_key,
            model=args.teacher_model,
            timeout=args.teacher_timeout,
            max_retries=args.max_retries,
            request_params=teacher_request_params,
        )
        try:
            if not args.skip_preflight:
                await preflight_teacher(teacher_client, args.teacher_model)
            return await run_all(
                specs=specs,
                completed_keys=completed_keys,
                workflow_kwargs_by_mode=workflow_kwargs_by_mode,
                teacher_client=teacher_client,
                output_dir=output_dir,
                save_traces=args.save_traces,
                concurrency=args.concurrency,
                log_every=args.log_every,
                keep_env_proxy=args.keep_env_proxy,
                error_retries=args.episode_error_retries,
                retry_backoff_seconds=args.episode_error_retry_backoff_seconds,
                retry_diagnostic_failures=args.retry_diagnostic_failures,
                episode_timeout_seconds=args.episode_timeout_seconds,
            )
        finally:
            await teacher_client.close()

    new_results = await run_without_proxy_environment(
        _run_api_phase,
        enabled=not args.keep_env_proxy,
    )

    all_results = latest_results([*existing_results, *new_results])
    pending_backfill = [
        result
        for result in all_results
        if result_needs_retry(
            result,
            retry_errors=True,
            retry_diagnostic_failures=True,
            generalization_levels=retry_generalization_levels,
            expected_generalization_replays=retry_generalization_replays,
        )
    ]
    pending_backfill_path = output_dir / "pending_backfill.jsonl"
    rewrite_results_jsonl(pending_backfill_path, pending_backfill)
    report = aggregate_report(
        all_results,
        modes=modes,
        dataset_size=len(dataset),
        attempts=attempts,
        generalization_enabled=config.student_generalize.enabled,
        generalization_levels=configured_generalization_levels(
            next(iter(workflow_kwargs_by_mode.values()))
        ),
        student_prompt_rows=prompt_row_counts,
    )
    if prompt_row_counts:
        report["student_prompt_pools"] = {
            "paths": config.prompt_pool.student_eval_paths,
            "row_counts": {
                f"{pool}/{index}": count
                for (pool, index), count in prompt_row_counts.items()
            },
            "prompts": [asdict(prompt) for prompt in student_prompts],
        }
    report["output_dir"] = str(output_dir)
    report["pending_backfill"] = {
        "count": len(pending_backfill),
        "path": str(pending_backfill_path),
    }
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output_dir / "summary.json", report)
    logger.info("API teacher evaluation complete: %s", output_dir)
    logger.info("Summary: %s", json.dumps(report["modes"], ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
