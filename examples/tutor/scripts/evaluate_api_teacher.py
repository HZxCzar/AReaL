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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
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
from examples.tutor.workflow import TutorAgentWorkflow

from areal import workflow_context
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.workflow_context import WorkflowContext
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorApiTeacherEval")

_T = TypeVar("_T")

DEFAULT_CONFIG_PATH = (
    "examples/tutor/configs/math/july/pass@2/qwen8b-qwen1.7b-math-baseline.yaml"
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
    with without_proxy_environment():
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
    with without_proxy_environment():
        config, _ = load_expr_config(
            ["--config", config_path, *overrides],
            TutorConfig,
        )
    return config, students


def build_eval_workflow_kwargs(
    *,
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    presolve_enabled: bool,
) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    teacher_pre = config.teacher_pre
    student_generalize = config.student_generalize
    base_eval_gconfig = config.eval_gconfig or config.gconfig
    eval_gconfig = base_eval_gconfig.new(
        n_samples=1,
        temperature=float(args.teacher_temperature),
        top_p=float(args.teacher_top_p),
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

    return {
        "gconfig": eval_gconfig,
        "tokenizer": tokenizer,
        "dataset_type": config.dataset_type,
        "answer_scorer": config.answer_scorer,
        "max_turns": config.max_turns,
        "enable_thinking": False,
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
        "format_error_penalty": reward.format_error_penalty,
        "leaked_success_reward_scale": reward.leaked_success_reward_scale,
        "assign_success_reward": reward.assign_success_reward,
        "outcome_prior_turn_weight": reward.outcome_prior_turn_weight,
        "outcome_credit_gamma": reward.outcome_credit_gamma,
        "early_success_bonus": reward.early_success_bonus,
        "max_turn_penalty": reward.max_turn_penalty,
        "enable_turn_penalty": reward.enable_turn_penalty,
        "turn_penalty": reward.turn_penalty,
        "length_penalty_threshold_chars": reward.length_penalty_threshold_chars,
        "length_penalty_per_100_chars": reward.length_penalty_per_100_chars,
        "length_penalty_min": reward.length_penalty_min,
        "zero_reward_on_length_stop": reward.zero_reward_on_length_stop,
        "teacher_system_prompt": config.teacher_system_prompt,
        "teacher_anti_leak_instruction_enabled": (
            config.teacher_anti_leak_instruction_enabled
        ),
        # Evaluation never samples teacher prompts. Student prompts are loaded only
        # when the config explicitly requests exhaustive prompt coverage.
        "teacher_prompt_pool_path": "",
        "teacher_warmup_enabled": False,
        "teacher_warmup_prompt_path": "",
        "teacher_warmup_steps": 0,
        "teacher_user_prompt_template": config.teacher_user_prompt_template,
        "teacher_show_ground_truth": config.teacher_show_ground_truth,
        "teacher_pre_enabled": presolve_enabled,
        "teacher_pre_mode": teacher_pre.mode,
        "teacher_pre_attempts": presolve_attempts,
        "teacher_pre_max_tokens": presolve_max_tokens,
        "student_system_prompt": config.student_system_prompt,
        "student_prompt_pool_path": config.prompt_pool.student_eval_paths.get(
            "seen", ""
        ),
        "student_heldout_prompt_pool_path": (
            config.prompt_pool.student_eval_paths.get("heldout", "")
        ),
        "student_prompt_include_base": config.prompt_pool.include_base,
        "prompt_pool_seed": config.seed,
        "leak_check_system_prompt": config.leak_check_system_prompt,
        "answer_judge_enabled": auxiliary_model.answer_judge_enabled,
        "answer_judge_max_tokens": auxiliary_model.answer_judge_max_tokens,
        "answer_judge_system_prompt": config.answer_judge_system_prompt,
        # Keep the original experiment's total sample and context budgets.
        "debug_trace_dir": None,
        "debug_trace_every_n_rollouts": 1,
        "max_train_sample_tokens": config.gconfig.max_tokens,
        "tokenizer_path": config.tokenizer_path,
        "model_context_length": config.sglang.context_length,
        "student_generalize_enabled": student_generalize.enabled,
        "student_generalize_mode": student_generalize.mode,
        "student_generalize_source": student_generalize.source,
        "student_generalize_path": student_generalize.path,
        "student_generalize_level1_reward": student_generalize.level1_reward,
        "student_generalize_level2_reward": student_generalize.level2_reward,
        "student_generalize_confidence_enabled": (
            student_generalize.confidence.enabled
        ),
        "student_generalize_confidence_reward_scale": (
            student_generalize.confidence.reward_scale
        ),
    }


def prepare_test_dataset(
    config: TutorConfig,
    student_models: list[dict[str, Any]],
    *,
    tokenizer: Any,
    limit: int,
    student_prompts: tuple[tutor_train.EvalStudentPrompt, ...] | None = None,
) -> Any:
    valid_config = tutor_train._without_remote_dataset_loading(config.valid_dataset)
    dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_config,
        tokenizer=tokenizer,
    )
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if 0 < eval_max_samples < len(dataset):
            rng = random.Random(config.seed)
            indices = sorted(rng.sample(range(len(dataset)), k=eval_max_samples))
            dataset = dataset.select(indices)
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
        payload[str(result.level)] = {
            "attempted": bool(result.attempted),
            "skipped": bool(result.skipped),
            "skip_reason": str(result.skip_reason or ""),
            "correct": bool(judge_result.correct) if judge_result is not None else None,
            "student_error": result.student_error,
            "confidence": float(result.confidence),
        }
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
    taught_success = termination_reason == "success"
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
        final_correct=pre_solved or taught_success,
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
                await result
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
) -> bool:
    """Return whether resume should replace an unreliable episode record."""

    if result.error is not None:
        return retry_errors
    if not retry_diagnostic_failures:
        return False
    return bool(
        result.student_call_failed
        or result.leak_check_failed_count
        or result.answer_judge_failed_count
        or result.teacher_pre_error_count
    )


def aggregate_mode(
    results: list[EpisodeResult],
    *,
    expected: int,
    generalization_enabled: bool = False,
) -> dict[str, Any]:
    results = latest_results(results)
    completed = [result for result in results if result.error is None]
    pre_solved = sum(result.pre_solved for result in completed)
    taught_success = sum(result.taught_success for result in completed)
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
    if generalization_enabled:
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
        generalization[level] = {
            "present_episode_count": len(level_results),
            "attempted": len(attempted),
            "evaluable": len(evaluable),
            "correct": correct,
            "student_error_count": student_errors,
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
        }

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
        "final_correct_count": final_correct,
        "full_set_final_correct_rate": _rate(final_correct, expected),
        "completed_final_correct_rate": _rate(final_correct, len(completed)),
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
        "generalization": generalization,
    }


def aggregate_report(
    results: list[EpisodeResult],
    *,
    modes: list[PresolveMode],
    dataset_size: int,
    attempts: int,
    generalization_enabled: bool = False,
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
            ),
            **aggregate_items(mode_results, expected_items=base_prompt_rows),
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
                    ),
                    **aggregate_items(prompt_results, expected_items=row_count),
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
    student_prompts: tuple[tutor_train.EvalStudentPrompt, ...] = (),
) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return redact_sensitive(
        {
            "config": str(config_path),
            "config_sha256": config_hash,
            "overrides": list(args.overrides),
            "dataset_size": dataset_size,
            "dataset_sha256": dataset_hash,
            "attempts": attempts,
            "modes": [asdict(mode) for mode in modes],
            "presolve": {
                "attempts": (
                    int(args.presolve_attempts)
                    if int(args.presolve_attempts) > 0
                    else int(config.teacher_pre.attempts)
                ),
                "max_tokens": (
                    int(args.presolve_max_tokens)
                    if args.presolve_max_tokens is not None
                    else int(config.teacher_pre.max_tokens)
                ),
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
            "auxiliary": {
                "base_url": config.auxiliary_model.base_url,
                "model": config.auxiliary_model.model,
                "temperature": config.auxiliary_model.temperature,
                "top_p": config.auxiliary_model.top_p,
                "max_tokens": config.auxiliary_model.max_tokens,
                "timeout": config.auxiliary_model.timeout,
                "max_concurrent_calls": (config.auxiliary_model.max_concurrent_calls),
                "request_params": config.auxiliary_model.request_params,
            },
            "students": student_models,
            "test_semantics": {
                "dataset_type": config.dataset_type,
                "answer_scorer": config.answer_scorer,
                "max_turns": config.max_turns,
                "leak_handling_mode": config.leak_handling_mode,
                "teacher_show_ground_truth": config.teacher_show_ground_truth,
                "student_generalize_enabled": config.student_generalize.enabled,
                "student_generalize_source": config.student_generalize.source,
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
        if previous.get("signature") != signature:
            raise ValueError(
                "Resume settings differ from the existing run_config.json."
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
        save_traces == "errors" and result.error is not None
    )
    if should_trace:
        trace_path = (
            output_dir
            / "traces"
            / spec.mode.name
            / (
                f"row_{spec.dataset_index:05d}_id_{safe_path_token(result.item_id)}_"
                f"attempt_{spec.attempt:02d}.json"
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
        async with semaphore:
            result = await run_episode(
                spec=spec,
                workflow_kwargs=workflow_kwargs_by_mode[spec.mode.name],
                teacher_client=teacher_client,
                output_dir=output_dir,
                save_traces=save_traces,
                keep_env_proxy=keep_env_proxy,
            )
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
        "--teacher-temperature", type=float, default=DEEPSEEK_TEMPERATURE
    )
    parser.add_argument("--teacher-top-p", type=float, default=DEEPSEEK_TOP_P)
    parser.add_argument("--teacher-max-tokens", type=int, default=4096)
    parser.add_argument("--teacher-timeout", type=float, default=300.0)
    parser.add_argument("--teacher-request-params", default="")
    parser.add_argument("--teacher-request-params-file", type=Path, default=None)
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
        help="'config' preserves the original two transfer probes after success.",
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
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
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
            "On resume, optionally rerun whole episodes with student, leak-check, "
            "answer-judge, or teacher-presolve call failures. Disabled by default "
            "to avoid conditional resampling."
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
    if args.presolve_attempts < 0:
        raise ValueError("--presolve-attempts must be non-negative.")
    if args.presolve_max_tokens is not None and args.presolve_max_tokens < 0:
        raise ValueError("--presolve-max-tokens must be non-negative.")


async def main_async(args: argparse.Namespace) -> None:
    validate_args(args)
    teacher_base_url = normalize_base_url(args.teacher_base_url)
    env_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    teacher_api_key = env_api_key or args.api_key or "EMPTY"

    config, student_models = load_experiment_config(args.config, args.overrides)
    tutor_train._apply_eval_average_rollouts(config)
    config.student_generalize.enabled = resolve_generalization(
        args.student_generalization,
        config.student_generalize.enabled,
    )
    if config.student_generalize.enabled:
        tutor_train._prepare_math_generalization_data(config)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    if config.student_generalize.enabled:
        tutor_train._validate_student_generalize_datasets(config, tokenizer)

    student_prompts = tutor_train._load_eval_student_prompts(config)
    dataset = prepare_test_dataset(
        config,
        student_models,
        tokenizer=tokenizer,
        limit=max(0, int(args.limit)),
        student_prompts=student_prompts,
    )
    prompt_row_counts = student_prompt_row_counts(dataset)
    modes = resolve_presolve_modes(
        args.teacher_presolve,
        config.teacher_pre.enabled,
    )
    attempts = (
        int(args.attempts)
        if int(args.attempts) > 0
        else max(1, int(config.evaluator.average_rollouts))
    )

    teacher_request_params = merge_dicts(
        load_request_params(
            args.teacher_request_params,
            args.teacher_request_params_file,
            label="--teacher-request-params",
        ),
        deepseek_non_thinking_params(config.seed),
    )

    workflow_kwargs_by_mode = {
        mode.name: build_eval_workflow_kwargs(
            config=config,
            student_models=student_models,
            tokenizer=tokenizer,
            args=args,
            presolve_enabled=mode.enabled,
        )
        for mode in modes
    }
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
        student_prompts=student_prompts,
    )
    output_dir = resolve_output_dir(args, config)
    prepare_output_dir(output_dir, signature=signature, resume=args.resume)
    results_path = output_dir / "results.jsonl"
    existing_results = load_existing_results(results_path) if args.resume else []
    completed_keys = {
        result.key
        for result in existing_results
        if not result_needs_retry(
            result,
            retry_errors=args.retry_errors,
            retry_diagnostic_failures=args.retry_diagnostic_failures,
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
            )
        finally:
            await teacher_client.close()

    new_results = await run_without_proxy_environment(
        _run_api_phase,
        enabled=not args.keep_env_proxy,
    )

    all_results = latest_results([*existing_results, *new_results])
    report = aggregate_report(
        all_results,
        modes=modes,
        dataset_size=len(dataset),
        attempts=attempts,
        generalization_enabled=config.student_generalize.enabled,
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
    report["finished_at"] = datetime.now(UTC).isoformat()
    write_json(output_dir / "summary.json", report)
    logger.info("API teacher evaluation complete: %s", output_dir)
    logger.info("Summary: %s", json.dumps(report["modes"], ensure_ascii=False))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
