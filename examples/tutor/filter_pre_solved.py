from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

    from configs import TutorConfig
    from workflow import TutorAgentWorkflow


logger = logging.getLogger("TutorPreSolveFilter")
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.append(str(_THIS_DIR))
sys.path.append(str(_REPO_ROOT))

from examples.tutor.core.types import PublicHistoryState, StudentTurnState


@dataclass(slots=True)
class ClassifiedRow:
    index: int
    item_id: int | str
    status: str
    kept: bool
    attempts: list[dict[str, Any]]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter tutor dataset rows that the auxiliary student can solve before "
            "receiving any tutor feedback."
        )
    )
    parser.add_argument("--config", default="examples/tutor/config.yaml")
    parser.add_argument(
        "--input",
        default="",
        help="Input HuggingFace dataset path. Defaults to config.train_dataset.path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output HuggingFace dataset path. If omitted, only the report is written.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="JSON report path. Defaults to '<output>_pre_solve_filter_report.json'.",
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
        help="Drop a row if any initial student attempt solves it.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help=(
            "Maximum concurrent student calls. Defaults to "
            "config.auxiliary_model.max_concurrent_calls."
        ),
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
        help="Keep rows whose pre-solve classification call failed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_workflow(config: TutorConfig, max_concurrency: int) -> TutorAgentWorkflow:
    from workflow import TutorAgentWorkflow

    auxiliary_model = config.auxiliary_model
    return TutorAgentWorkflow(
        max_turns=config.max_turns,
        answer_scorer=config.answer_scorer,
        aux_mode=auxiliary_model.mode,
        aux_enable_thinking=auxiliary_model.enable_thinking,
        aux_base_url=auxiliary_model.base_url,
        aux_model=auxiliary_model.model,
        aux_api_key=auxiliary_model.api_key,
        aux_timeout=auxiliary_model.timeout,
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=max_concurrency,
        api_params_config_path=auxiliary_model.api_params_config_path or None,
        api_params_key=auxiliary_model.api_params_key or None,
        student_system_prompt=config.student_system_prompt,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
    )


def resolve_splits(requested: list[str], dataset: DatasetDict) -> list[str]:
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


async def classify_row(
    *,
    workflow: TutorAgentWorkflow,
    row: dict[str, Any],
    index: int,
    attempts: int,
    keep_on_error: bool,
) -> ClassifiedRow:
    item_id = item_id_from_row(row, index)
    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    attempt_rows: list[dict[str, Any]] = []

    for attempt_idx in range(1, attempts + 1):
        answer, error = await workflow._run_student(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(),
                previous_student_output="",
                latest_tutor_visible_output="(none, produce the first answer attempt)",
            )
        )
        judge_result = workflow._score_answer(task, ground_truth, answer)
        attempt_rows.append(
            {
                "attempt": attempt_idx,
                "answer": answer,
                "student_error": error,
                "correct": bool(judge_result.correct),
                "feedback": judge_result.feedback,
                "extracted_answer": judge_result.raw_result.get("extracted_answer"),
            }
        )
        if error is not None:
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                status="error",
                kept=keep_on_error,
                attempts=attempt_rows,
                error=error,
            )
        if judge_result.correct:
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                status="pre_solved",
                kept=False,
                attempts=attempt_rows,
            )
    return ClassifiedRow(
        index=index,
        item_id=item_id,
        status="kept",
        kept=True,
        attempts=attempt_rows,
    )


async def classify_split(
    *,
    workflow: TutorAgentWorkflow,
    dataset: Dataset,
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
            workflow=workflow,
            row=dict(dataset[index]),
            index=index,
            attempts=attempts,
            keep_on_error=keep_on_error,
        )
        async with log_lock:
            processed += 1
            if processed == size or processed % max(1, log_every) == 0:
                logger.info(
                    "Classified %s/%s rows in split '%s'", processed, size, split_name
                )
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


def summarize_results(results: list[ClassifiedRow]) -> dict[str, Any]:
    kept = [result for result in results if result.kept]
    pre_solved = [result for result in results if result.status == "pre_solved"]
    errors = [result for result in results if result.status == "error"]
    return {
        "evaluated": len(results),
        "kept": len(kept),
        "dropped_pre_solved": len(pre_solved),
        "errors": len(errors),
        "kept_ids": [result.item_id for result in kept],
        "pre_solved_ids": [result.item_id for result in pre_solved],
        "error_ids": [result.item_id for result in errors],
        "rows": [
            {
                "index": result.index,
                "id": result.item_id,
                "status": result.status,
                "kept": result.kept,
                "error": result.error,
                "attempts": result.attempts,
            }
            for result in results
        ],
    }


async def main_async(args: argparse.Namespace) -> None:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to run the tutor pre-solve filter."
        ) from exc
    from configs import TutorConfig

    from areal.api.cli_args import load_expr_config

    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    input_path = Path(args.input or config.train_dataset.path).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    report_path = (
        Path(args.report).resolve()
        if args.report
        else (
            output_path.with_name(f"{output_path.name}_pre_solve_filter_report.json")
            if output_path is not None
            else None
        )
    )
    if output_path is None and report_path is None:
        raise ValueError("Pass --output and/or --report.")
    attempts = max(1, int(args.attempts))
    max_concurrency = max(
        1,
        int(args.concurrency)
        if args.concurrency and args.concurrency > 0
        else int(config.auxiliary_model.max_concurrent_calls),
    )

    loaded = load_from_disk(str(input_path))
    dataset = loaded if isinstance(loaded, DatasetDict) else DatasetDict({"train": loaded})
    selected_splits = resolve_splits(args.splits, dataset)
    workflow = build_workflow(config, max_concurrency=max_concurrency)

    filtered_splits: dict[str, Dataset] = {}
    report: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "config": str(Path(args.config).resolve()),
        "splits": {},
        "attempts": attempts,
        "keep_on_error": bool(args.keep_on_error),
        "max_concurrency": max_concurrency,
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
        logger.info("Filtering split '%s' with %s rows", split_name, len(split_dataset))
        results = await classify_split(
            workflow=workflow,
            dataset=split_dataset,
            split_name=split_name,
            attempts=attempts,
            keep_on_error=bool(args.keep_on_error),
            limit=max(0, int(args.limit)),
        )
        keep_indices = [result.index for result in results if result.kept]
        filtered_splits[split_name] = split_dataset.select(keep_indices)
        split_summary = summarize_results(results)
        if args.limit and args.limit > 0:
            split_summary["debug_limit"] = int(args.limit)
        report["splits"][split_name] = split_summary
        logger.info(
            "Split '%s': kept=%s dropped_pre_solved=%s errors=%s",
            split_name,
            split_summary["kept"],
            split_summary["dropped_pre_solved"],
            split_summary["errors"],
        )

    if output_path is not None:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. Pass --overwrite to replace it."
                )
            shutil.rmtree(output_path)
        DatasetDict(filtered_splits).save_to_disk(str(output_path))
        logger.info("Saved filtered dataset to %s", output_path)

    if report_path is not None:
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
