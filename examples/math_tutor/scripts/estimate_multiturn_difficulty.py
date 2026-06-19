from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import asyncio
import csv
import json
import logging
import os
import pathlib
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TUTOR_DIR = _THIS_DIR.parent
_REPO_ROOT = _TUTOR_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TUTOR_DIR))

from examples.math_tutor.core.history import trace_to_json
from examples.math_tutor.core.text import strip_reasoning_for_context


logger = logging.getLogger("TutorMultiturnDifficulty")

DEFAULT_CONFIG_PATH = "examples/tutor/configs/math/baseline-overfit-2.yaml"
DEFAULT_BASE_URL = "http://127.0.0.1:30008/v1"
DEFAULT_MODEL = "default"
DEFAULT_ATTEMPTS = 3
DEFAULT_OUTPUT_ROOT = Path(
    "/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output"
)


@dataclass(slots=True)
class EpisodeSummary:
    termination_reason: str
    total_reward: float
    num_turns: int
    pre_success: bool
    leak_count: int
    success: bool
    latest_student_answer: str


@dataclass(slots=True)
class AttemptResult:
    attempt: int
    termination_reason: str
    success: bool
    pre_success: bool
    num_turns: int
    leak_count: int
    total_reward: float
    latest_student_answer: str
    trace_path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class RowDifficultyResult:
    index: int
    item_id: int | str
    attempts: list[AttemptResult]
    error: str | None = None


class DifficultyTutorWorkflow:
    """Mixin-style subclass target that records a compact episode summary."""

    last_episode_summary: EpisodeSummary | None
    last_episode_trace_payload: dict[str, Any] | None

    def _record_episode_summary(
        self,
        *,
        total_reward: float,
        traces: list[Any],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        latest_student_answer = ""
        if traces:
            latest_student_answer = str(getattr(traces[-1], "student_output", "") or "")
        self.last_episode_summary = EpisodeSummary(
            termination_reason=str(termination_reason),
            total_reward=float(total_reward),
            num_turns=len(traces),
            pre_success=bool(pre_success),
            leak_count=int(leak_count),
            success=str(termination_reason) in {"success", "pre_solved"},
            latest_student_answer=latest_student_answer,
        )

    def _maybe_dump_debug_trace(
        self,
        *,
        task: str,
        ground_truth: str,
        initial_student_answer: str,
        latest_student_answer: str,
        total_reward: float,
        traces: list[Any],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        self.last_episode_trace_payload = {
            "termination_reason": str(termination_reason),
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
        super()._maybe_dump_debug_trace(  # type: ignore[misc]
            task=task,
            ground_truth=ground_truth,
            initial_student_answer=initial_student_answer,
            latest_student_answer=latest_student_answer,
            total_reward=total_reward,
            traces=traces,
            termination_reason=termination_reason,
            pre_success=pre_success,
            leak_count=leak_count,
        )

    def _log_rollout_stats(
        self,
        *,
        total_reward: float,
        traces: list[Any],
        termination_reason: str,
        pre_success: bool,
        leak_count: int,
    ) -> None:
        self._record_episode_summary(
            total_reward=total_reward,
            traces=traces,
            termination_reason=termination_reason,
            pre_success=pre_success,
            leak_count=leak_count,
        )
        super()._log_rollout_stats(  # type: ignore[misc]
            total_reward=total_reward,
            traces=traces,
            termination_reason=termination_reason,
            pre_success=pre_success,
            leak_count=leak_count,
        )


class ApiTutorClient:
    """OpenAI-compatible client wrapper for offline tutor actor calls."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        request_params: dict[str, Any],
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for tutor API calls.")
        self.model = model
        self.request_params = dict(request_params)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=0,
        )
        self.chat = _ApiTutorChat(self)


class _ApiTutorChat:
    def __init__(self, parent: ApiTutorClient) -> None:
        self.completions = _ApiTutorCompletions(parent)


class _ApiTutorCompletions:
    def __init__(self, parent: ApiTutorClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        request = merge_dicts(self._parent.request_params, kwargs)
        request["model"] = self._parent.model
        return await self._parent._client.chat.completions.create(**request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate tutor task difficulty by running full multi-turn Tutor episodes "
            "multiple times per row and writing success-rate metadata."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for per-attempt trace JSON files when --trace-dir is not "
            f"set. Defaults to {DEFAULT_OUTPUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help=(
            "Directory for full per-attempt trace JSON files. Defaults to "
            "<trace-root>/tutor/multiturn_difficulty_traces/"
            "<experiment_name>/<trial_name>."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Maximum concurrent dataset rows. Defaults to auxiliary_model.max_concurrent_calls.",
    )
    parser.add_argument(
        "--thinking",
        choices=["config", "on", "off", "unset"],
        default="config",
        help="Whether to send extra_body.chat_template_kwargs.enable_thinking. 'config' uses config.enable_thinking for tutor and auxiliary_model.enable_thinking for student/judge.",
    )
    parser.add_argument(
        "--request-params",
        default="",
        help="Additional chat.completions.create kwargs as a JSON object for API calls.",
    )
    parser.add_argument("--request-params-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-on-error", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument(
        "--partial-every",
        type=int,
        default=10,
        help=(
            "Write partial JSON/CSV summaries every N completed rows. "
            "Set to 0 to disable partial outputs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_object_arg(value: str, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_request_params(args: argparse.Namespace) -> dict[str, Any]:
    params = load_json_object_arg(args.request_params, label="--request-params")
    if args.request_params_file is not None:
        file_params = json.loads(args.request_params_file.read_text(encoding="utf-8"))
        if not isinstance(file_params, dict):
            raise ValueError("--request-params-file must contain a JSON object.")
        params = merge_dicts(params, file_params)
    return params


def resolve_thinking(choice: str, default: bool) -> bool | None:
    if choice == "unset":
        return None
    if choice == "on":
        return True
    if choice == "off":
        return False
    return bool(default)


def with_thinking_param(
    params: dict[str, Any],
    enable_thinking: bool | None,
) -> dict[str, Any]:
    if enable_thinking is None:
        return dict(params)
    return merge_dicts(
        params,
        {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                }
            }
        },
    )


def build_tutor_request_params(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    params = load_request_params(args)
    return with_thinking_param(
        params,
        resolve_thinking(args.thinking, bool(config.enable_thinking)),
    )


def build_aux_request_params(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    params = merge_dicts(dict(auxiliary_model.request_params), load_request_params(args))
    return with_thinking_param(
        params,
        resolve_thinking(args.thinking, bool(auxiliary_model.enable_thinking)),
    )


def item_id_from_row(row: dict[str, Any], index: int) -> int | str:
    value = row.get("id", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def resolve_splits(requested: list[str], dataset: Any) -> list[str]:
    if "all" in requested:
        return list(dataset.keys())
    missing = [split for split in requested if split not in dataset]
    if missing:
        raise ValueError(
            f"Unknown split(s) {missing}; available splits are {list(dataset.keys())}"
        )
    return requested


def build_workflow(
    config: Any,
    args: argparse.Namespace,
    *,
    aux_request_params: dict[str, Any],
    max_concurrency: int,
) -> Any:
    from workflow import TutorAgentWorkflow

    class OfflineDifficultyWorkflow(DifficultyTutorWorkflow, TutorAgentWorkflow):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.last_episode_summary = None
            self.last_episode_trace_payload = None
            super().__init__(*args, **kwargs)

    auxiliary_model = config.auxiliary_model
    reward = config.reward
    pairwise = reward.pairwise
    aux_thinking = resolve_thinking(args.thinking, bool(auxiliary_model.enable_thinking))
    return OfflineDifficultyWorkflow(
        tokenizer=config.tokenizer_path,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=config.max_turns,
        answer_scorer=config.answer_scorer,
        enable_thinking=config.enable_thinking,
        enable_leak_check=config.enable_leak_check,
        aux_mode="api",
        aux_enable_thinking=bool(aux_thinking),
        aux_base_url=args.base_url or auxiliary_model.base_url,
        aux_model=args.model or auxiliary_model.model,
        aux_api_key=args.api_key or auxiliary_model.api_key,
        aux_timeout=int(args.timeout or auxiliary_model.timeout),
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=max_concurrency,
        aux_request_params=aux_request_params,
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        assign_success_reward=reward.assign_success_reward,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        enable_turn_penalty=reward.enable_turn_penalty,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        teacher_system_prompt=config.teacher_system_prompt,
        teacher_user_prompt_template=config.teacher_user_prompt_template,
        teacher_show_ground_truth=config.teacher_show_ground_truth,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        answer_judge_system_prompt=config.answer_judge_system_prompt,
        summary_system_prompt=config.summary_system_prompt,
        max_train_sample_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        pairwise_reward_enabled=pairwise.enabled,
        pairwise_reference_lag_steps=pairwise.reference_lag_steps,
        pairwise_reward_scale=pairwise.scale,
        pairwise_compare_all_turns=pairwise.compare_all_turns,
        pairwise_judge_both_incorrect=pairwise.judge_both_incorrect,
    )


def safe_path_token(value: Any) -> str:
    token = str(value)
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in token)
    return safe.strip("._") or "item"


def default_trace_dir(config: Any, trace_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return (
        trace_root
        / "tutor"
        / "multiturn_difficulty_traces"
        / safe_path_token(config.experiment_name)
        / safe_path_token(config.trial_name)
    )


def resolve_trace_dir(args: argparse.Namespace, config: Any) -> Path:
    if args.trace_dir is not None:
        return args.trace_dir.expanduser().resolve()
    return default_trace_dir(config, args.trace_root.expanduser().resolve())


def write_attempt_trace(
    *,
    trace_dir: Path,
    split_name: str,
    row: dict[str, Any],
    index: int,
    item_id: int | str,
    attempt_idx: int,
    payload: dict[str, Any],
) -> Path:
    split_dir = trace_dir / safe_path_token(split_name)
    split_dir.mkdir(parents=True, exist_ok=True)
    file_path = split_dir / (
        f"row_{index:08d}_id_{safe_path_token(item_id)}"
        f"_attempt_{attempt_idx:02d}.json"
    )
    trace_payload = dict(payload)
    trace_payload.update(
        {
            "split": split_name,
            "index": int(index),
            "id": item_id,
            "attempt": int(attempt_idx),
            "row_metadata": row.get("metadata") or {},
        }
    )
    file_path.write_text(
        json.dumps(trace_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path


async def run_row_attempts(
    *,
    workflow_factory: Callable[[], Any],
    tutor_client: Any,
    row: dict[str, Any],
    index: int,
    attempts: int,
    split_name: str,
    trace_dir: Path,
) -> RowDifficultyResult:
    item_id = item_id_from_row(row, index)
    attempt_rows: list[AttemptResult] = []
    row_error: str | None = None

    for attempt_idx in range(1, attempts + 1):
        workflow = workflow_factory()
        try:
            await workflow._run_episode(dict(row), external_client=tutor_client)
            summary = workflow.last_episode_summary
            if summary is None:
                raise RuntimeError("Tutor workflow did not record an episode summary.")
            trace_path = None
            if workflow.last_episode_trace_payload is not None:
                trace_path = os.fspath(
                    write_attempt_trace(
                        trace_dir=trace_dir,
                        split_name=split_name,
                        row=row,
                        index=index,
                        item_id=item_id,
                        attempt_idx=attempt_idx,
                        payload=workflow.last_episode_trace_payload,
                    )
                )
            attempt_rows.append(
                AttemptResult(
                    attempt=attempt_idx,
                    termination_reason=summary.termination_reason,
                    success=summary.success,
                    pre_success=summary.pre_success,
                    num_turns=summary.num_turns,
                    leak_count=summary.leak_count,
                    total_reward=summary.total_reward,
                    latest_student_answer=summary.latest_student_answer,
                    trace_path=trace_path,
                    error=None,
                )
            )
        except Exception as exc:
            error = str(exc)
            row_error = error
            attempt_rows.append(
                AttemptResult(
                    attempt=attempt_idx,
                    termination_reason="error",
                    success=False,
                    pre_success=False,
                    num_turns=0,
                    leak_count=0,
                    total_reward=0.0,
                    latest_student_answer="",
                    error=error,
                )
            )

    if any(attempt.error is None for attempt in attempt_rows):
        row_error = None
    return RowDifficultyResult(
        index=index,
        item_id=item_id,
        attempts=attempt_rows,
        error=row_error,
    )


async def run_split(
    *,
    workflow_factory: Callable[[], Any],
    tutor_client: Any,
    dataset: Any,
    split_name: str,
    attempts: int,
    concurrency: int,
    limit: int,
    trace_dir: Path,
    on_result: Callable[[RowDifficultyResult, int, int], None] | None = None,
    log_every: int = 10,
) -> list[RowDifficultyResult]:
    size = len(dataset) if limit <= 0 else min(limit, len(dataset))
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    processed = 0
    log_lock = asyncio.Lock()

    async def _run(index: int) -> RowDifficultyResult:
        nonlocal processed
        async with semaphore:
            result = await run_row_attempts(
                workflow_factory=workflow_factory,
                tutor_client=tutor_client,
                row=dict(dataset[index]),
                index=index,
                attempts=attempts,
                split_name=split_name,
                trace_dir=trace_dir,
            )
        async with log_lock:
            processed += 1
            if on_result is not None:
                on_result(result, processed, size)
            if processed == size or processed % max(1, log_every) == 0:
                logger.info(
                    "Estimated multi-turn difficulty for %s/%s rows in '%s'",
                    processed,
                    size,
                    split_name,
                )
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


def aggregate_attempts(attempts: list[AttemptResult]) -> dict[str, Any]:
    completed = [attempt for attempt in attempts if attempt.error is None]
    success_count = sum(1 for attempt in completed if attempt.success)
    success_rate = (
        float(success_count / len(completed)) if completed else None
    )
    avg_turns = (
        float(sum(attempt.num_turns for attempt in completed) / len(completed))
        if completed
        else None
    )
    avg_leaks = (
        float(sum(attempt.leak_count for attempt in completed) / len(completed))
        if completed
        else None
    )
    histogram = Counter(attempt.termination_reason for attempt in attempts)
    return {
        "multiturn_attempts": len(attempts),
        "multiturn_completed_attempts": len(completed),
        "multiturn_success_count": success_count,
        "multiturn_success_rate": success_rate,
        "multiturn_avg_turns": avg_turns,
        "multiturn_avg_leak_count": avg_leaks,
        "multiturn_error_count": len(attempts) - len(completed),
        "multiturn_termination_histogram": dict(sorted(histogram.items())),
        "multiturn_difficulty_source": "sampled_multiturn",
        "multiturn_difficulty_confidence": difficulty_confidence(len(completed)),
    }


def difficulty_confidence(completed_attempts: int) -> str:
    if completed_attempts >= 3:
        return "medium"
    if completed_attempts == 2:
        return "low"
    if completed_attempts == 1:
        return "very_low"
    return "error"


def result_metadata(result: RowDifficultyResult) -> dict[str, Any]:
    return aggregate_attempts(result.attempts)


def merge_result_metadata(row: dict[str, Any], result: RowDifficultyResult) -> dict[str, Any]:
    new_row = dict(row)
    metadata = dict(new_row.get("metadata") or {})
    metadata.update(result_metadata(result))
    new_row["metadata"] = metadata
    return new_row


def attempt_to_report(
    attempt: AttemptResult,
    *,
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    payload = asdict(attempt)
    if save_outputs:
        return payload
    answer = str(payload.pop("latest_student_answer") or "")
    payload["latest_student_answer_preview"] = strip_reasoning_for_context(answer)[
        : max(0, int(preview_chars))
    ]
    payload["latest_student_answer_chars"] = len(answer)
    return payload


def result_to_report(
    result: RowDifficultyResult,
    *,
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    return {
        "index": result.index,
        "id": result.item_id,
        "error": result.error,
        **result_metadata(result),
        "attempts": [
            attempt_to_report(
                attempt,
                save_outputs=save_outputs,
                preview_chars=preview_chars,
            )
            for attempt in result.attempts
        ],
    }


def split_report(
    *,
    input_rows: int,
    results: list[RowDifficultyResult],
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    success_rates = [
        result_metadata(result)["multiturn_success_rate"]
        for result in results
        if result_metadata(result)["multiturn_success_rate"] is not None
    ]
    return {
        "input_rows": input_rows,
        "evaluated": len(results),
        "rows_with_errors": sum(1 for result in results if result.error is not None),
        "avg_success_rate": (
            float(sum(success_rates) / len(success_rates)) if success_rates else None
        ),
        "rows": [
            result_to_report(
                result,
                save_outputs=save_outputs,
                preview_chars=preview_chars,
            )
            for result in results
        ],
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_multiturn_difficulty")


def default_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}_multiturn_difficulty_report.json")


def default_csv_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}_multiturn_difficulty.csv")


def partial_path(path: Path) -> Path:
    suffix = path.suffix
    if suffix:
        return path.with_name(f"{path.stem}.partial{suffix}")
    return path.with_name(f"{path.name}.partial")


def write_csv(path: Path, *, split_results: dict[str, list[RowDifficultyResult]]) -> None:
    fieldnames = [
        "split",
        "index",
        "id",
        "multiturn_attempts",
        "multiturn_completed_attempts",
        "multiturn_success_count",
        "multiturn_success_rate",
        "multiturn_avg_turns",
        "multiturn_avg_leak_count",
        "multiturn_error_count",
        "multiturn_difficulty_confidence",
        "termination_histogram",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, results in split_results.items():
            for result in results:
                metadata = result_metadata(result)
                writer.writerow(
                    {
                        "split": split_name,
                        "index": result.index,
                        "id": result.item_id,
                        "termination_histogram": json.dumps(
                            metadata.pop("multiturn_termination_histogram"),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        **{
                            key: value
                            for key, value in metadata.items()
                            if key in fieldnames
                        },
                    }
                )


async def main_async(args: argparse.Namespace) -> None:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to estimate tutor difficulty."
        ) from exc
    from configs import TutorConfig

    from areal.api.cli_args import load_expr_config

    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    input_path = Path(args.input or config.train_dataset.path).resolve()
    output_path = (
        Path(args.output).resolve() if args.output else default_output_path(input_path)
    )
    report_path = (
        Path(args.report).resolve()
        if args.report
        else default_report_path(output_path)
    )
    csv_path = Path(args.csv).resolve() if args.csv else default_csv_path(output_path)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    attempts = max(1, int(args.attempts))
    max_concurrency = max(
        1,
        int(args.concurrency)
        if args.concurrency and args.concurrency > 0
        else int(config.auxiliary_model.max_concurrent_calls),
    )
    aux_request_params = build_aux_request_params(args, config)
    tutor_request_params = build_tutor_request_params(args, config)
    trace_dir = resolve_trace_dir(args, config)

    def workflow_factory() -> Any:
        return build_workflow(
            config,
            args,
            aux_request_params=aux_request_params,
            max_concurrency=max_concurrency,
        )

    tutor_client = ApiTutorClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=float(args.timeout),
        request_params=tutor_request_params,
    )

    loaded = load_from_disk(str(input_path))
    is_dataset_dict = isinstance(loaded, DatasetDict)
    dataset = loaded if is_dataset_dict else DatasetDict({"train": loaded})
    selected_splits = resolve_splits(args.splits, dataset)

    output_splits: dict[str, Any] = {}
    split_results: dict[str, list[RowDifficultyResult]] = {}
    report: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "config": str(Path(args.config).resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "attempts": attempts,
        "max_concurrency": max_concurrency,
        "selected_splits": selected_splits,
        "teacher_show_ground_truth": bool(config.teacher_show_ground_truth),
        "teacher_prompt": config.teacher_system_prompt,
        "teacher_user_prompt_template": config.teacher_user_prompt_template,
        "tutor_request_params": tutor_request_params,
        "auxiliary_request_params": aux_request_params,
        "trace_dir": str(trace_dir),
        "save_outputs": bool(args.save_outputs),
        "preview_chars": int(args.preview_chars),
        "splits": {},
    }
    partial_every = max(0, int(args.partial_every))
    partial_report_path = partial_path(report_path)
    partial_csv_path = partial_path(csv_path)

    for split_name, split_dataset in dataset.items():
        if split_name not in selected_splits:
            output_splits[split_name] = split_dataset
            report["splits"][split_name] = {
                "input_rows": len(split_dataset),
                "copied_unchanged": True,
            }
            continue

        logger.info(
            "Estimating multi-turn difficulty for split '%s' with %s rows",
            split_name,
            len(split_dataset),
        )
        split_partial_results: list[RowDifficultyResult] = []

        def _write_partial(
            result: RowDifficultyResult,
            processed: int,
            total: int,
        ) -> None:
            split_partial_results.append(result)
            if partial_every <= 0:
                return
            if processed != total and processed % partial_every != 0:
                return
            partial_report = dict(report)
            partial_report["partial"] = True
            partial_report["splits"] = dict(report["splits"])
            partial_summary = split_report(
                input_rows=len(split_dataset),
                results=split_partial_results,
                save_outputs=bool(args.save_outputs),
                preview_chars=int(args.preview_chars),
            )
            partial_summary["partial"] = True
            partial_summary["processed"] = processed
            partial_summary["total_to_evaluate"] = total
            if args.limit and args.limit > 0:
                partial_summary["debug_limit"] = int(args.limit)
            partial_report["splits"][split_name] = partial_summary
            partial_report_path.parent.mkdir(parents=True, exist_ok=True)
            partial_report_path.write_text(
                json.dumps(partial_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            write_csv(
                partial_csv_path,
                split_results={split_name: split_partial_results},
            )
            logger.info(
                "Wrote partial results for %s/%s rows to %s and %s",
                processed,
                total,
                partial_report_path,
                partial_csv_path,
            )

        results = await run_split(
            workflow_factory=workflow_factory,
            tutor_client=tutor_client,
            dataset=split_dataset,
            split_name=split_name,
            attempts=attempts,
            concurrency=max_concurrency,
            limit=max(0, int(args.limit)),
            trace_dir=trace_dir,
            on_result=_write_partial,
        )
        all_error_ids = [result.item_id for result in results if result.error is not None]
        if all_error_ids and not args.keep_on_error:
            raise RuntimeError(
                "All attempts failed for row ids "
                f"{all_error_ids[:10]}. Pass --keep-on-error to write partial results."
            )
        result_by_index = {result.index: result for result in results}

        def _map_row(row: dict[str, Any], index: int) -> dict[str, Any]:
            result = result_by_index.get(index)
            if result is None:
                return row
            return merge_result_metadata(row, result)

        output_splits[split_name] = split_dataset.map(_map_row, with_indices=True)
        split_results[split_name] = results
        summary = split_report(
            input_rows=len(split_dataset),
            results=results,
            save_outputs=bool(args.save_outputs),
            preview_chars=int(args.preview_chars),
        )
        if args.limit and args.limit > 0:
            summary["debug_limit"] = int(args.limit)
        report["splits"][split_name] = summary
        logger.info(
            "Split '%s': evaluated=%s avg_success_rate=%s rows_with_errors=%s",
            split_name,
            summary["evaluated"],
            summary["avg_success_rate"],
            summary["rows_with_errors"],
        )

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataset = DatasetDict(output_splits) if is_dataset_dict else output_splits["train"]
    output_dataset.save_to_disk(str(output_path))
    logger.info("Saved metadata-labeled dataset to %s", output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved report to %s", report_path)
    write_csv(csv_path, split_results=split_results)
    logger.info("Saved CSV to %s", csv_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

