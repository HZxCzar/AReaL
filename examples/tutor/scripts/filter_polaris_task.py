from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import asyncio
import json
import logging
import os
import pathlib
import shutil
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None

if TYPE_CHECKING:
    from configs import TutorConfig


logger = logging.getLogger("PolarisTaskFilter")
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TUTOR_DIR = _THIS_DIR.parent
_REPO_ROOT = _TUTOR_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TUTOR_DIR))

from examples.tutor.core.polaris import score_polaris_answer_async
from examples.tutor.prompts import POLARIS_FILTER_SOLVER_USER_TEMPLATE


DEFAULT_CONFIG_PATH = (
    "examples/tutor/configs/polaris/qwen8b-qwen1.7b-polaris-baseline.yaml"
)
DEFAULT_OUTPUT_PATH = (
    "examples/tutor/data/polaris_dataset_qwen1.7b_student2_qwen8b_teacher_filter_task"
)
PROXY_ENV_VARS = (
    "ALL_PROXY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "http_proxy",
    "https_proxy",
)


@dataclass(slots=True)
class RoleConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    temperature: float
    top_p: float | None
    concurrency: int
    request_params: dict[str, Any]


@dataclass(slots=True)
class GenerationResult:
    answer: str
    usage: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class ScoreOutcome:
    correct: bool
    feedback: str
    raw_result: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class AttemptRecord:
    role: str
    attempt: int
    answer: str
    usage: dict[str, Any]
    correct: bool
    feedback: str
    raw_result: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class FilterResult:
    index: int
    item_id: int | str
    status: str
    kept: bool
    student_attempts: list[AttemptRecord]
    teacher_attempts: list[AttemptRecord]
    error: str | None = None


ScoreFn = Callable[[str, str, str], Awaitable[ScoreOutcome]]


class OpenAISolverClient:
    """OpenAI-compatible solver using the fixed official Polaris user prompt."""

    def __init__(self, config: RoleConfig) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for Polaris filtering.")
        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key or "EMPTY",
            timeout=config.timeout,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def solve(self, prompt: str, *, seed_offset: int = 0) -> GenerationResult:
        request_params = dict(self._config.request_params)
        extra_body = request_params.pop("extra_body", None)
        if "seed" in request_params:
            request_params["seed"] = int(request_params["seed"]) + seed_offset
        request_params.setdefault("max_completion_tokens", self._config.max_tokens)
        request_params.setdefault("temperature", self._config.temperature)
        if self._config.top_p is not None:
            request_params.setdefault("top_p", self._config.top_p)
        request_params = {
            key: value for key, value in request_params.items() if value is not None
        }
        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=extra_body,
                    **request_params,
                )
        except Exception as exc:
            return GenerationResult(answer="", usage={}, error=str(exc))

        content = response.choices[0].message.content or ""
        return GenerationResult(
            answer=content.strip(),
            usage=_response_usage_to_dict(response),
        )


def render_official_prompt(task: str) -> str:
    """Render the fixed prompt used by the official Polaris dataset."""
    return POLARIS_FILTER_SOLVER_USER_TEMPLATE.format(task=task)


async def score_generated_answer(
    task: str,
    ground_truth: str,
    answer: str,
    *,
    max_workers: int,
) -> ScoreOutcome:
    result = await score_polaris_answer_async(
        task,
        ground_truth,
        answer,
        max_workers=max_workers,
    )
    raw_result = dict(result.raw_result)
    scoring_error = raw_result.get("scoring_error")
    return ScoreOutcome(
        correct=bool(result.correct),
        feedback=result.feedback,
        raw_result=raw_result,
        error=str(scoring_error) if scoring_error else None,
    )


async def classify_polaris_row(
    *,
    row: dict[str, Any],
    index: int,
    student_client: Any,
    teacher_client: Any,
    score_fn: ScoreFn,
    student_attempts: int = 2,
    teacher_attempts: int = 1,
) -> FilterResult:
    """Keep a row only when the student always fails and the teacher succeeds."""
    if student_attempts < 1:
        raise ValueError("student_attempts must be at least 1.")
    if teacher_attempts < 1:
        raise ValueError("teacher_attempts must be at least 1.")

    item_id = _item_id_from_row(row, index)
    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    prompt = render_official_prompt(task)
    student_records: list[AttemptRecord] = []

    for attempt in range(1, student_attempts + 1):
        generation = await student_client.solve(
            prompt,
            seed_offset=index * student_attempts + attempt - 1,
        )
        if generation.error is not None:
            student_records.append(_failed_attempt("student", attempt, generation))
            continue

        score = await score_fn(task, ground_truth, generation.answer)
        student_records.append(_scored_attempt("student", attempt, generation, score))
        if score.correct:
            return FilterResult(
                index=index,
                item_id=item_id,
                status="student_solved",
                kept=False,
                student_attempts=student_records,
                teacher_attempts=[],
            )
        if score.error is not None:
            continue

    student_errors = [record.error for record in student_records if record.error]
    if student_errors:
        return FilterResult(
            index=index,
            item_id=item_id,
            status="student_error",
            kept=False,
            student_attempts=student_records,
            teacher_attempts=[],
            error="; ".join(student_errors),
        )

    teacher_records: list[AttemptRecord] = []
    for attempt in range(1, teacher_attempts + 1):
        generation = await teacher_client.solve(
            prompt,
            seed_offset=index * teacher_attempts + attempt - 1,
        )
        if generation.error is not None:
            teacher_records.append(_failed_attempt("teacher", attempt, generation))
            continue

        score = await score_fn(task, ground_truth, generation.answer)
        teacher_records.append(_scored_attempt("teacher", attempt, generation, score))
        if score.correct:
            return FilterResult(
                index=index,
                item_id=item_id,
                status="teacher_solved",
                kept=True,
                student_attempts=student_records,
                teacher_attempts=teacher_records,
            )
        if score.error is not None:
            continue

    teacher_errors = [record.error for record in teacher_records if record.error]
    if teacher_errors:
        return FilterResult(
            index=index,
            item_id=item_id,
            status="teacher_error",
            kept=False,
            student_attempts=student_records,
            teacher_attempts=teacher_records,
            error="; ".join(teacher_errors),
        )
    return FilterResult(
        index=index,
        item_id=item_id,
        status="teacher_unsolved",
        kept=False,
        student_attempts=student_records,
        teacher_attempts=teacher_records,
    )


async def filter_split(
    *,
    dataset: Any,
    split_name: str,
    student_client: Any,
    teacher_client: Any,
    score_fn: ScoreFn,
    student_attempts: int,
    teacher_attempts: int,
    row_concurrency: int,
    limit: int = 0,
) -> list[FilterResult]:
    size = len(dataset) if limit <= 0 else min(limit, len(dataset))
    semaphore = asyncio.Semaphore(max(1, row_concurrency))
    completed = 0
    log_lock = asyncio.Lock()

    async def _run(index: int) -> FilterResult:
        nonlocal completed
        async with semaphore:
            result = await classify_polaris_row(
                row=dict(dataset[index]),
                index=index,
                student_client=student_client,
                teacher_client=teacher_client,
                score_fn=score_fn,
                student_attempts=student_attempts,
                teacher_attempts=teacher_attempts,
            )
        async with log_lock:
            completed += 1
            if completed == size or completed % 50 == 0:
                logger.info("Filtered %s/%s rows in '%s'", completed, size, split_name)
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Polaris tasks by dropping any task solved by the student in "
            "two attempts and any remaining task not solved by the teacher."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", default="")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--student-base-url", default="")
    parser.add_argument("--teacher-base-url", default="")
    parser.add_argument("--student-api-key", default="")
    parser.add_argument("--teacher-api-key", default="")
    parser.add_argument("--student-model", default="")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--student-attempts", type=int, default=2)
    parser.add_argument("--teacher-attempts", type=int, default=1)
    parser.add_argument("--student-concurrency", type=int, default=0)
    parser.add_argument("--teacher-concurrency", type=int, default=0)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--student-max-tokens", type=int, default=0)
    parser.add_argument("--teacher-max-tokens", type=int, default=0)
    parser.add_argument("--student-temperature", type=float, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--student-top-p", type=float, default=None)
    parser.add_argument("--teacher-top-p", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-env-proxy", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_role_configs(
    config: TutorConfig, args: argparse.Namespace
) -> tuple[RoleConfig, RoleConfig]:
    if config.dataset_type != "polaris" or config.answer_scorer != "polaris":
        raise ValueError(
            "Polaris filtering requires dataset_type=polaris and answer_scorer=polaris."
        )
    if len(config.student_models) != 1:
        raise ValueError("Polaris filtering requires exactly one configured student.")

    student = config.student_models[0]
    auxiliary = config.auxiliary_model
    student_config = RoleConfig(
        base_url=args.student_base_url or student.base_url,
        api_key=_resolve_api_key(args.student_api_key, student.api_key),
        model=args.student_model or student.model,
        timeout=float(student.timeout),
        max_tokens=args.student_max_tokens or int(student.max_tokens),
        temperature=(
            float(student.temperature)
            if args.student_temperature is None
            else args.student_temperature
        ),
        top_p=(
            None
            if student.top_p is None and args.student_top_p is None
            else (
                float(student.top_p)
                if args.student_top_p is None
                else args.student_top_p
            )
        ),
        concurrency=args.student_concurrency or int(student.max_concurrent_calls),
        request_params=dict(student.request_params),
    )

    teacher_request_params = dict(auxiliary.request_params)
    extra_body = dict(teacher_request_params.get("extra_body") or {})
    top_k = int(config.gconfig.top_k)
    if top_k < int(1e8):
        extra_body["top_k"] = top_k
    else:
        extra_body.pop("top_k", None)
    extra_body.pop("min_p", None)
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs["enable_thinking"] = bool(config.enable_thinking)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    teacher_request_params["extra_body"] = extra_body
    teacher_config = RoleConfig(
        base_url=args.teacher_base_url or auxiliary.base_url,
        api_key=_resolve_api_key(args.teacher_api_key, auxiliary.api_key),
        model=args.teacher_model or auxiliary.model,
        timeout=float(auxiliary.timeout),
        max_tokens=args.teacher_max_tokens or int(config.gconfig.max_new_tokens),
        temperature=(
            float(config.gconfig.temperature)
            if args.teacher_temperature is None
            else args.teacher_temperature
        ),
        top_p=(
            float(config.gconfig.top_p)
            if args.teacher_top_p is None
            else args.teacher_top_p
        ),
        concurrency=args.teacher_concurrency or int(auxiliary.max_concurrent_calls),
        request_params=teacher_request_params,
    )
    for role, role_config in (("student", student_config), ("teacher", teacher_config)):
        if not role_config.base_url:
            raise ValueError(f"{role} base URL is not configured.")
        if role_config.concurrency < 1:
            raise ValueError(f"{role} concurrency must be at least 1.")
        if role_config.max_tokens < 1:
            raise ValueError(f"{role} max_tokens must be at least 1.")
    return student_config, teacher_config


async def main_async(args: argparse.Namespace) -> None:
    _validate_args(args)
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to run the Polaris filter."
        ) from exc
    from configs import TutorConfig

    from areal.api.cli_args import load_expr_config

    config, _ = load_expr_config(
        ["--config", args.config, *args.overrides], TutorConfig
    )
    student_config, teacher_config = build_role_configs(config, args)
    student_client = OpenAISolverClient(student_config)
    teacher_client = OpenAISolverClient(teacher_config)
    input_path = Path(args.input or config.train_dataset.path).resolve()
    output_path = Path(args.output).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.with_name(f"{output_path.name}_filter_task_report.json")
    )

    loaded = load_from_disk(str(input_path))
    is_dataset_dict = isinstance(loaded, DatasetDict)
    dataset = loaded if is_dataset_dict else DatasetDict({"train": loaded})
    selected_splits = _resolve_splits(args.splits, dataset)
    filtered_splits: dict[str, Any] = {}
    report: dict[str, Any] = {
        "input": str(input_path),
        "output": None if args.dry_run else str(output_path),
        "config": str(Path(args.config).resolve()),
        "filter_rule": {
            "student_attempts": args.student_attempts,
            "drop_if_any_student_attempt_correct": True,
            "teacher_attempts": args.teacher_attempts,
            "keep_only_if_any_teacher_attempt_correct": True,
            "attempt_seed_policy": (
                "configured seed + row index * role attempts + attempt index"
            ),
        },
        "prompt_template": POLARIS_FILTER_SOLVER_USER_TEMPLATE,
        "scorer": "examples.tutor.core.polaris.score_polaris_answer_async",
        "student": _role_report(student_config),
        "teacher": _role_report(teacher_config),
        "selected_splits": selected_splits,
        "splits": {},
        "dry_run": bool(args.dry_run),
    }

    async def _score(task: str, ground_truth: str, answer: str) -> ScoreOutcome:
        return await score_generated_answer(
            task,
            ground_truth,
            answer,
            max_workers=max(1, int(args.score_workers)),
        )

    row_concurrency = max(student_config.concurrency, teacher_config.concurrency)
    for split_name, split_dataset in dataset.items():
        if split_name not in selected_splits:
            filtered_splits[split_name] = split_dataset
            report["splits"][split_name] = {
                "input_rows": len(split_dataset),
                "kept": len(split_dataset),
                "copied_unchanged": True,
            }
            continue

        results = await filter_split(
            dataset=split_dataset,
            split_name=split_name,
            student_client=student_client,
            teacher_client=teacher_client,
            score_fn=_score,
            student_attempts=int(args.student_attempts),
            teacher_attempts=int(args.teacher_attempts),
            row_concurrency=row_concurrency,
            limit=max(0, int(args.limit)),
        )
        keep_indices = [result.index for result in results if result.kept]
        filtered_splits[split_name] = split_dataset.select(keep_indices)
        report["splits"][split_name] = _summarize_results(
            input_rows=len(split_dataset),
            results=results,
            keep_indices=keep_indices,
            save_outputs=bool(args.save_outputs),
            preview_chars=max(0, int(args.preview_chars)),
        )
        if args.limit > 0:
            report["splits"][split_name]["debug_limit"] = int(args.limit)
        logger.info(
            "Split '%s': kept=%s student_solved=%s teacher_unsolved=%s errors=%s",
            split_name,
            len(keep_indices),
            sum(result.status == "student_solved" for result in results),
            sum(result.status == "teacher_unsolved" for result in results),
            sum(result.status.endswith("_error") for result in results),
        )

    if not args.dry_run:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dataset = (
            DatasetDict(filtered_splits)
            if is_dataset_dict
            else filtered_splits["train"]
        )
        output_dataset.save_to_disk(str(output_path))
        logger.info("Saved filtered Polaris dataset to %s", output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved Polaris filter report to %s", report_path)


def _failed_attempt(
    role: str, attempt: int, generation: GenerationResult
) -> AttemptRecord:
    return AttemptRecord(
        role=role,
        attempt=attempt,
        answer="",
        usage=generation.usage,
        correct=False,
        feedback=f"{role.capitalize()} API call failed.",
        raw_result={},
        error=generation.error,
    )


def _scored_attempt(
    role: str,
    attempt: int,
    generation: GenerationResult,
    score: ScoreOutcome,
) -> AttemptRecord:
    return AttemptRecord(
        role=role,
        attempt=attempt,
        answer=generation.answer,
        usage=generation.usage,
        correct=score.correct,
        feedback=score.feedback,
        raw_result=score.raw_result,
        error=score.error,
    )


def _item_id_from_row(row: dict[str, Any], index: int) -> int | str:
    value = row.get("id", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _resolve_splits(requested: list[str], dataset: Any) -> list[str]:
    if "all" in requested:
        return list(dataset.keys())
    missing = [split for split in requested if split not in dataset]
    if missing:
        raise ValueError(
            f"Unknown split(s) {missing}; available splits are {list(dataset.keys())}."
        )
    return requested


def _validate_args(args: argparse.Namespace) -> None:
    if args.student_attempts < 1:
        raise ValueError("--student-attempts must be at least 1.")
    if args.teacher_attempts < 1:
        raise ValueError("--teacher-attempts must be at least 1.")
    if args.score_workers < 1:
        raise ValueError("--score-workers must be at least 1.")


def _resolve_api_key(cli_value: str, config_value: str) -> str:
    return (
        cli_value
        or config_value
        or os.getenv("INF_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )


def _response_usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if isinstance(usage, dict):
        return dict(usage)
    return dict(getattr(usage, "__dict__", {}))


def _role_report(config: RoleConfig) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "model": config.model,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "concurrency": config.concurrency,
        "request_params": config.request_params,
    }


def _attempt_report(
    attempt: AttemptRecord, *, save_outputs: bool, preview_chars: int
) -> dict[str, Any]:
    report = {
        "role": attempt.role,
        "attempt": attempt.attempt,
        "correct": attempt.correct,
        "error": attempt.error,
        "feedback": attempt.feedback,
        "usage": attempt.usage,
        "extracted_answer": attempt.raw_result.get("extracted_answer"),
        "target_answers": attempt.raw_result.get("target_answers"),
        "mathd_correct": attempt.raw_result.get("mathd_correct"),
        "sympy_correct": attempt.raw_result.get("sympy_correct"),
    }
    if save_outputs:
        report["answer"] = attempt.answer
    else:
        report["answer_preview"] = attempt.answer[:preview_chars]
        report["answer_chars"] = len(attempt.answer)
    return report


def _summarize_results(
    *,
    input_rows: int,
    results: list[FilterResult],
    keep_indices: list[int],
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    statuses = {
        status: sum(result.status == status for result in results)
        for status in (
            "student_solved",
            "teacher_solved",
            "teacher_unsolved",
            "student_error",
            "teacher_error",
        )
    }
    return {
        "input_rows": input_rows,
        "evaluated": len(results),
        "kept": len(keep_indices),
        **statuses,
        "kept_original_indices": keep_indices,
        "kept_ids": [result.item_id for result in results if result.kept],
        "rows": [
            {
                "index": result.index,
                "id": result.item_id,
                "status": result.status,
                "kept": result.kept,
                "error": result.error,
                "student_attempts": [
                    _attempt_report(
                        attempt,
                        save_outputs=save_outputs,
                        preview_chars=preview_chars,
                    )
                    for attempt in result.student_attempts
                ],
                "teacher_attempts": [
                    _attempt_report(
                        attempt,
                        save_outputs=save_outputs,
                        preview_chars=preview_chars,
                    )
                    for attempt in result.teacher_attempts
                ],
            }
            for result in results
        ],
    }


def main() -> None:
    load_dotenv(_REPO_ROOT / ".env", override=False)
    args = parse_args()
    if not args.keep_env_proxy:
        for name in PROXY_ENV_VARS:
            os.environ.pop(name, None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
