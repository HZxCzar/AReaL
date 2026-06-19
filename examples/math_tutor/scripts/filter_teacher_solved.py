from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None


logger = logging.getLogger("TutorTeacherSolvedFilter")
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TUTOR_DIR = _THIS_DIR.parent
_REPO_ROOT = _TUTOR_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TUTOR_DIR))


DEFAULT_CONFIG_PATH = "examples/math_tutor/configs/math/baseline.yaml"
DEFAULT_TEACHER_BASE_URL = "http://127.0.0.1:30008/v1"
DEFAULT_SOLVER_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem independently. "
    "Show your reasoning if useful. Put the final answer in the last "
    "\\boxed{...}. Do not use any answer key or hidden solution."
)

DEFAULT_SOLVER_USER_TEMPLATE = """\
Task:
{task}

Solve the problem. Put your final answer in \\boxed{{}}.
"""


@dataclass(slots=True)
class TeacherAttempt:
    attempt: int
    answer: str
    error: str | None
    correct: bool
    feedback: str
    raw_result: dict[str, Any]
    usage: dict[str, Any]


@dataclass(slots=True)
class ClassifiedRow:
    index: int
    item_id: int | str
    status: str
    kept: bool
    attempts: list[TeacherAttempt]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep tutor dataset rows that an external teacher model can solve "
            "without student interaction."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--input",
        default="",
        help="Input HuggingFace dataset path. Defaults to config.train_dataset.path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output HuggingFace dataset path. Defaults to "
            "'<input>_teacher_solved'."
        ),
    )
    parser.add_argument(
        "--report",
        default="",
        help=(
            "JSON report path. Defaults to "
            "'<output>_teacher_solved_filter_report.json', or the same suffix "
            "next to --input when --output is omitted."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to filter, or 'all'. Unselected splits are copied unchanged.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Keep a row if any teacher attempt solves it.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent teacher calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug limit per filtered split. Output contains only evaluated rows.",
    )
    parser.add_argument(
        "--keep-on-error",
        action="store_true",
        help="Keep rows whose teacher classification failed for every attempt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    parser.add_argument(
        "--teacher-base-url",
        default=DEFAULT_TEACHER_BASE_URL,
        help=(
            "OpenAI-compatible teacher base URL. Defaults to "
            f"{DEFAULT_TEACHER_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--teacher-api-key",
        default="",
        help="Teacher API key. Defaults to OPENAI_API_KEY, then EMPTY.",
    )
    parser.add_argument(
        "--teacher-model",
        default="default",
        help="Model name sent in chat.completions.create.",
    )
    parser.add_argument(
        "--teacher-timeout",
        type=float,
        default=120.0,
        help="Teacher request timeout in seconds.",
    )
    parser.add_argument(
        "--teacher-max-tokens",
        type=int,
        default=0,
        help="Max completion tokens. Defaults to config.gconfig.max_new_tokens.",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=None,
        help="Teacher sampling temperature. Defaults to config.gconfig.temperature.",
    )
    parser.add_argument(
        "--teacher-top-p",
        type=float,
        default=None,
        help="Teacher top_p. Defaults to config.gconfig.top_p.",
    )
    parser.add_argument(
        "--thinking",
        choices=["config", "on", "off", "unset"],
        default="config",
        help=(
            "Whether to send extra_body.chat_template_kwargs.enable_thinking. "
            "'config' uses config.enable_thinking; 'unset' sends no value."
        ),
    )
    parser.add_argument(
        "--request-params",
        default="",
        help=(
            "Additional chat.completions.create kwargs as a JSON object. "
            "Use this for seed, extra_body, or backend-specific parameters."
        ),
    )
    parser.add_argument(
        "--request-params-file",
        type=Path,
        default=None,
        help="JSON file containing additional chat.completions.create kwargs.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional file overriding the teacher solver system prompt.",
    )
    parser.add_argument(
        "--user-template-file",
        type=Path,
        default=None,
        help="Optional format template with a '{task}' placeholder.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Store full teacher outputs in the JSON report.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Answer preview length when --save-outputs is not set.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_json_object_arg(value: str, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_request_params(args: argparse.Namespace) -> dict[str, Any]:
    params = load_json_object_arg(args.request_params, label="--request-params")
    if args.request_params_file is not None:
        file_params = json.loads(args.request_params_file.read_text(encoding="utf-8"))
        if not isinstance(file_params, dict):
            raise ValueError("--request-params-file must contain a JSON object.")
        params = merge_dicts(params, file_params)
    return params


def build_request_params(
    *,
    args: argparse.Namespace,
    config: Any,
) -> dict[str, Any]:
    params = load_request_params(args)
    max_tokens = (
        int(args.teacher_max_tokens)
        if args.teacher_max_tokens and args.teacher_max_tokens > 0
        else int(config.gconfig.max_new_tokens)
    )
    temperature = (
        float(args.teacher_temperature)
        if args.teacher_temperature is not None
        else float(config.gconfig.temperature)
    )
    top_p = (
        float(args.teacher_top_p)
        if args.teacher_top_p is not None
        else float(config.gconfig.top_p)
    )
    params.setdefault("temperature", temperature)
    params.setdefault("top_p", top_p)
    if "max_tokens" in params and "max_completion_tokens" not in params:
        params["max_completion_tokens"] = int(params.pop("max_tokens"))
    params.setdefault("max_completion_tokens", max_tokens)

    if args.thinking != "unset":
        enable_thinking = bool(config.enable_thinking)
        if args.thinking == "on":
            enable_thinking = True
        elif args.thinking == "off":
            enable_thinking = False
        params = merge_dicts(
            params,
            {
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking,
                    }
                }
            },
        )
    return params


def request_params_for_create(params: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    request_kwargs = {
        key: value for key, value in params.items() if key != "extra_body"
    }
    extra_body = params.get("extra_body")
    return request_kwargs, extra_body


def resolve_splits(requested: list[str], dataset: Any) -> list[str]:
    if "all" in requested:
        return list(dataset.keys())
    missing = [split for split in requested if split not in dataset]
    if missing:
        raise ValueError(
            f"Unknown split(s) {missing}; available splits are {list(dataset.keys())}"
        )
    return requested


def item_id_from_row(row: dict[str, Any], index: int) -> int | str:
    value = row.get("id", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def format_user_prompt(template: str, task: str) -> str:
    try:
        return template.format(task=task)
    except KeyError as exc:
        raise ValueError(
            "User template may only reference the '{task}' placeholder."
        ) from exc


def response_usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if isinstance(usage, dict):
        return dict(usage)
    return dict(getattr(usage, "__dict__", {}))


class TeacherSolverClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        request_params: dict[str, Any],
        concurrency: int,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for teacher calls.")
        self.model = model
        self.request_params = request_params
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def solve(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, dict[str, Any], str | None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_kwargs, extra_body = request_params_for_create(self.request_params)
        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body=extra_body,
                    **request_kwargs,
                )
        except Exception as exc:
            return "", {}, str(exc)
        content = response.choices[0].message.content or ""
        return content.strip(), response_usage_to_dict(response), None


async def classify_row(
    *,
    client: TeacherSolverClient,
    scorer: Any,
    system_prompt: str,
    user_template: str,
    row: dict[str, Any],
    index: int,
    attempts: int,
    keep_on_error: bool,
) -> ClassifiedRow:
    item_id = item_id_from_row(row, index)
    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    attempt_rows: list[TeacherAttempt] = []
    saw_successful_call = False

    for attempt_idx in range(1, attempts + 1):
        answer, usage, error = await client.solve(
            system_prompt=system_prompt,
            user_prompt=format_user_prompt(user_template, task),
        )
        if error is not None:
            attempt_rows.append(
                TeacherAttempt(
                    attempt=attempt_idx,
                    answer="",
                    error=error,
                    correct=False,
                    feedback="Teacher call failed.",
                    raw_result={},
                    usage=usage,
                )
            )
            continue

        saw_successful_call = True
        judge_result = scorer(task, ground_truth, answer)
        attempt_rows.append(
            TeacherAttempt(
                attempt=attempt_idx,
                answer=answer,
                error=None,
                correct=bool(judge_result.correct),
                feedback=judge_result.feedback,
                raw_result=judge_result.raw_result,
                usage=usage,
            )
        )
        if judge_result.correct:
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                status="teacher_solved",
                kept=True,
                attempts=attempt_rows,
            )

    if saw_successful_call:
        return ClassifiedRow(
            index=index,
            item_id=item_id,
            status="teacher_unsolved",
            kept=False,
            attempts=attempt_rows,
        )
    return ClassifiedRow(
        index=index,
        item_id=item_id,
        status="error",
        kept=keep_on_error,
        attempts=attempt_rows,
        error="All teacher attempts failed.",
    )


async def classify_split(
    *,
    client: TeacherSolverClient,
    scorer: Any,
    system_prompt: str,
    user_template: str,
    dataset: Any,
    split_name: str,
    attempts: int,
    keep_on_error: bool,
    limit: int,
    log_every: int = 10,
) -> list[ClassifiedRow]:
    size = len(dataset) if limit <= 0 else min(limit, len(dataset))
    processed = 0
    log_lock = asyncio.Lock()

    async def _run(index: int) -> ClassifiedRow:
        nonlocal processed
        result = await classify_row(
            client=client,
            scorer=scorer,
            system_prompt=system_prompt,
            user_template=user_template,
            row=dict(dataset[index]),
            index=index,
            attempts=attempts,
            keep_on_error=keep_on_error,
        )
        async with log_lock:
            processed += 1
            if processed == size or processed % max(1, log_every) == 0:
                logger.info(
                    "Classified %s/%s rows in split '%s'",
                    processed,
                    size,
                    split_name,
                )
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


def attempt_to_report(
    attempt: TeacherAttempt,
    *,
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    row = {
        "attempt": attempt.attempt,
        "error": attempt.error,
        "correct": attempt.correct,
        "feedback": attempt.feedback,
        "extracted_answer": attempt.raw_result.get("extracted_answer"),
        "target_answer": attempt.raw_result.get("target_answer"),
        "normalized_prediction": attempt.raw_result.get("normalized_prediction"),
        "normalized_target": attempt.raw_result.get("normalized_target"),
        "usage": attempt.usage,
    }
    if save_outputs:
        row["answer"] = attempt.answer
    else:
        row["answer_preview"] = attempt.answer[: max(0, int(preview_chars))]
        row["answer_chars"] = len(attempt.answer)
    return row


def summarize_results(
    results: list[ClassifiedRow],
    *,
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    kept = [result for result in results if result.kept]
    solved = [result for result in results if result.status == "teacher_solved"]
    unsolved = [result for result in results if result.status == "teacher_unsolved"]
    errors = [result for result in results if result.status == "error"]
    return {
        "evaluated": len(results),
        "kept": len(kept),
        "teacher_solved": len(solved),
        "teacher_unsolved": len(unsolved),
        "errors": len(errors),
        "kept_ids": [result.item_id for result in kept],
        "teacher_solved_ids": [result.item_id for result in solved],
        "teacher_unsolved_ids": [result.item_id for result in unsolved],
        "error_ids": [result.item_id for result in errors],
        "rows": [
            {
                "index": result.index,
                "id": result.item_id,
                "status": result.status,
                "kept": result.kept,
                "error": result.error,
                "attempts": [
                    attempt_to_report(
                        attempt,
                        save_outputs=save_outputs,
                        preview_chars=preview_chars,
                    )
                    for attempt in result.attempts
                ],
            }
            for result in results
        ],
    }


def default_report_path(input_path: Path, output_path: Path | None) -> Path:
    base = output_path if output_path is not None else input_path
    return base.with_name(f"{base.name}_teacher_solved_filter_report.json")


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_teacher_solved")


def load_prompt(path: Path | None, default: str) -> str:
    if path is None:
        return default
    return path.read_text(encoding="utf-8").strip()


async def main_async(args: argparse.Namespace) -> None:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to run the teacher-solved filter."
        ) from exc
    from examples.math_tutor.configs import TutorConfig
    from examples.math_tutor.core.scoring import get_answer_scorer

    from areal.api.cli_args import load_expr_config

    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    input_path = Path(args.input or config.train_dataset.path).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(input_path).resolve()
    )
    report_path = (
        Path(args.report).resolve()
        if args.report
        else default_report_path(input_path, output_path)
    )
    base_url = args.teacher_base_url or os.getenv("OPENAI_BASE_URL", "")
    if not base_url:
        raise ValueError("Pass --teacher-base-url or set OPENAI_BASE_URL.")
    api_key = args.teacher_api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    attempts = max(1, int(args.attempts))
    max_concurrency = max(1, int(args.concurrency))
    request_params = build_request_params(args=args, config=config)
    system_prompt = load_prompt(args.system_prompt_file, DEFAULT_SOLVER_SYSTEM_PROMPT)
    user_template = load_prompt(args.user_template_file, DEFAULT_SOLVER_USER_TEMPLATE)

    loaded = load_from_disk(str(input_path))
    is_dataset_dict = isinstance(loaded, DatasetDict)
    dataset = loaded if is_dataset_dict else DatasetDict({"train": loaded})
    selected_splits = resolve_splits(args.splits, dataset)
    scorer = get_answer_scorer(config.answer_scorer)
    client = TeacherSolverClient(
        base_url=base_url,
        api_key=api_key,
        model=args.teacher_model,
        timeout=float(args.teacher_timeout),
        request_params=request_params,
        concurrency=max_concurrency,
    )

    filtered_splits: dict[str, Any] = {}
    report: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "config": str(Path(args.config).resolve()),
        "answer_scorer": config.answer_scorer,
        "teacher": {
            "base_url": base_url,
            "model": args.teacher_model,
            "timeout": float(args.teacher_timeout),
            "request_params": request_params,
            "max_concurrency": max_concurrency,
        },
        "prompt": {
            "system_prompt": system_prompt,
            "user_template": user_template,
        },
        "splits": {},
        "attempts": attempts,
        "keep_on_error": bool(args.keep_on_error),
        "save_outputs": bool(args.save_outputs),
        "preview_chars": int(args.preview_chars),
    }

    for split_name, split_dataset in dataset.items():
        if split_name not in selected_splits:
            filtered_splits[split_name] = split_dataset
            report["splits"][split_name] = {
                "evaluated": 0,
                "kept": len(split_dataset),
                "copied_unchanged": True,
            }
            continue

        logger.info(
            "Filtering split '%s' with %s rows", split_name, len(split_dataset)
        )
        results = await classify_split(
            client=client,
            scorer=scorer,
            system_prompt=system_prompt,
            user_template=user_template,
            dataset=split_dataset,
            split_name=split_name,
            attempts=attempts,
            keep_on_error=bool(args.keep_on_error),
            limit=max(0, int(args.limit)),
        )
        keep_indices = [result.index for result in results if result.kept]
        filtered_splits[split_name] = split_dataset.select(keep_indices)
        split_summary = summarize_results(
            results,
            save_outputs=bool(args.save_outputs),
            preview_chars=int(args.preview_chars),
        )
        if args.limit and args.limit > 0:
            split_summary["debug_limit"] = int(args.limit)
        report["splits"][split_name] = split_summary
        logger.info(
            "Split '%s': kept=%s teacher_solved=%s unsolved=%s errors=%s",
            split_name,
            split_summary["kept"],
            split_summary["teacher_solved"],
            split_summary["teacher_unsolved"],
            split_summary["errors"],
        )

    if output_path is not None:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dataset = (
            DatasetDict(filtered_splits) if is_dataset_dict else filtered_splits["train"]
        )
        output_dataset.save_to_disk(str(output_path))
        logger.info("Saved teacher-solved dataset to %s", output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved filter report to %s", report_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()

