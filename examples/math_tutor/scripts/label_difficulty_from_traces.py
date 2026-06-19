from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET = Path("examples/tutor/data/baseline-qwen3-8B")
DEFAULT_OUTPUT = Path("examples/tutor/data/baseline-qwen3-8B-difficulty")
DEFAULT_REPORT = Path("examples/tutor/report/baseline-qwen3-8B_difficulty_report.json")
DEFAULT_CSV = Path("examples/tutor/report/baseline-qwen3-8B_difficulty.csv")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TAGENT_ROOT = _REPO_ROOT.parent


def default_trace_dirs() -> tuple[Path, Path]:
    trace_root = _TAGENT_ROOT / "output/tutor/debug_traces/tutor-math-baseline"
    return (
        trace_root / "20260611_011611_qwen8b-thinking-self-turn10-baseline/train",
        trace_root / "20260610_232916_qwen8b-thinking-self-turn10-baseline/train",
    )
CONTEXT_LIMIT_REASONS = {"context_limit", "context_budget_limit"}
NOISY_LEAK_THRESHOLD = 2.0


@dataclass
class DifficultyStats:
    difficulty_label: str
    difficulty_score: float
    difficulty_source: str
    difficulty_confidence: str
    trace_count: int
    trace_success_rate: float | None
    trace_avg_reward: float | None
    trace_avg_turns: float | None
    trace_avg_leak_count: float | None
    trace_max_turns_rate: float | None
    trace_context_limit_rate: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label tutor train-set difficulty from prior baseline debug traces. "
            "The test split is copied unchanged."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Input HuggingFace dataset path. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to label. Other splits are copied unchanged. Default: train.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Train debug trace directory. Can be passed multiple times. "
            "Defaults to the two latest long baseline runs under ../output."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Per-question CSV path. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Dataset row id field. Default: id.",
    )
    parser.add_argument(
        "--task-field",
        default="task",
        help="Dataset task text field used to match traces. Default: task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute labels and write reports without saving the output dataset.",
    )
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def trace_dirs(args: argparse.Namespace) -> list[Path]:
    dirs = args.trace_dir if args.trace_dir is not None else list(default_trace_dirs())
    resolved = [path.expanduser().resolve() for path in dirs]
    missing = [str(path) for path in resolved if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Trace dir(s) not found: {missing}")
    return resolved


def row_id(row: dict[str, Any], id_field: str) -> str:
    if id_field not in row:
        raise KeyError(f"Dataset row does not contain id field '{id_field}'.")
    return str(row[id_field])


def row_task(row: dict[str, Any], task_field: str) -> str:
    if task_field not in row:
        raise KeyError(f"Dataset row does not contain task field '{task_field}'.")
    return str(row[task_field])


def build_task_index(dataset: Any, *, id_field: str, task_field: str) -> dict[str, str]:
    task_to_id: dict[str, str] = {}
    duplicate_tasks: list[str] = []
    for row in dataset:
        task = row_task(row, task_field)
        item_id = row_id(row, id_field)
        if task in task_to_id:
            duplicate_tasks.append(task[:120])
        else:
            task_to_id[task] = item_id
    if duplicate_tasks:
        examples = duplicate_tasks[:5]
        raise ValueError(
            "Cannot match traces by task because duplicate task text exists. "
            f"Examples: {examples}"
        )
    return task_to_id


def collect_traces(
    *,
    dirs: list[Path],
    task_to_id: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    traces_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_reports: list[dict[str, Any]] = []
    total_files = 0
    total_loaded = 0
    total_matched = 0

    for trace_dir in dirs:
        files = sorted(trace_dir.glob("*.json"))
        matched_ids: Counter[str] = Counter()
        loaded = 0
        bad_json = 0
        unmatched = 0
        for path in files:
            total_files += 1
            trace = load_trace(path)
            if trace is None:
                bad_json += 1
                continue
            loaded += 1
            total_loaded += 1
            task = str(trace.get("task", ""))
            item_id = task_to_id.get(task)
            if item_id is None:
                unmatched += 1
                continue
            traces_by_id[item_id].append(trace)
            matched_ids[item_id] += 1
            total_matched += 1

        run_reports.append(
            {
                "trace_dir": str(trace_dir),
                "files": len(files),
                "loaded": loaded,
                "bad_json": bad_json,
                "matched_traces": sum(matched_ids.values()),
                "matched_unique_items": len(matched_ids),
                "unmatched_traces": unmatched,
                "repeat_count_histogram": dict(sorted(Counter(matched_ids.values()).items())),
            }
        )

    trace_report = {
        "trace_dirs": [str(path) for path in dirs],
        "total_files": total_files,
        "total_loaded": total_loaded,
        "total_matched_traces": total_matched,
        "matched_unique_items": len(traces_by_id),
        "runs": run_reports,
    }
    return traces_by_id, trace_report


def termination_reason(trace: dict[str, Any]) -> str:
    return str(trace.get("termination_reason") or "")


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def trace_score(trace_list: list[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    count = len(trace_list)
    success_rate = mean(
        [1.0 if termination_reason(trace) == "success" else 0.0 for trace in trace_list]
    )
    max_turns_rate = mean(
        [1.0 if termination_reason(trace) == "max_turns" else 0.0 for trace in trace_list]
    )
    context_limit_rate = mean(
        [
            1.0 if termination_reason(trace) in CONTEXT_LIMIT_REASONS else 0.0
            for trace in trace_list
        ]
    )
    avg_turns = mean([float(trace.get("num_turns") or 0.0) for trace in trace_list])
    avg_leak_count = mean(
        [float(trace.get("leak_count") or 0.0) for trace in trace_list]
    )
    avg_reward = mean([float(trace.get("total_reward") or 0.0) for trace in trace_list])
    score = 100.0 * (
        0.45 * (1.0 - success_rate)
        + 0.25 * min(avg_turns / 10.0, 1.0)
        + 0.20 * max_turns_rate
        + 0.10 * min(avg_leak_count / 3.0, 1.0)
    )
    metrics = {
        "trace_count": float(count),
        "trace_success_rate": success_rate,
        "trace_avg_reward": avg_reward,
        "trace_avg_turns": avg_turns,
        "trace_avg_leak_count": avg_leak_count,
        "trace_max_turns_rate": max_turns_rate,
        "trace_context_limit_rate": context_limit_rate,
    }
    return score, metrics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile from an empty list.")
    index = round(float(fraction) * (len(values) - 1))
    index = max(0, min(len(values) - 1, index))
    return float(sorted(values)[index])


def level_number(row: dict[str, Any]) -> int | None:
    metadata = row.get("metadata") or {}
    level = str(metadata.get("level", ""))
    if not level.startswith("Level "):
        return None
    try:
        return int(level.split()[-1])
    except ValueError:
        return None


def fallback_label(row: dict[str, Any]) -> tuple[str, float]:
    level = level_number(row)
    if level is None:
        return "medium", 50.0
    if level <= 2:
        return "easy", 25.0
    if level <= 4:
        return "medium", 50.0
    return "hard", 75.0


def confidence(trace_count: int, source: str) -> str:
    if source == "metadata_fallback":
        return "fallback"
    if trace_count >= 3:
        return "high"
    if trace_count == 2:
        return "medium"
    return "low"


def build_labels(
    dataset: Any,
    *,
    traces_by_id: dict[str, list[dict[str, Any]]],
    id_field: str,
) -> tuple[dict[str, DifficultyStats], dict[str, Any]]:
    trace_scores: dict[str, tuple[float, dict[str, float], bool]] = {}
    non_noisy_scores: list[float] = []

    for row in dataset:
        item_id = row_id(row, id_field)
        trace_list = traces_by_id.get(item_id, [])
        if not trace_list:
            continue
        score, metrics = trace_score(trace_list)
        is_noisy = (
            metrics["trace_context_limit_rate"] > 0.0
            or metrics["trace_avg_leak_count"] >= NOISY_LEAK_THRESHOLD
        )
        trace_scores[item_id] = (score, metrics, is_noisy)
        if not is_noisy:
            non_noisy_scores.append(score)

    easy_threshold = percentile(non_noisy_scores, 0.33)
    hard_threshold = percentile(non_noisy_scores, 0.67)

    labels: dict[str, DifficultyStats] = {}
    for row in dataset:
        item_id = row_id(row, id_field)
        if item_id not in trace_scores:
            label, score = fallback_label(row)
            labels[item_id] = DifficultyStats(
                difficulty_label=label,
                difficulty_score=score,
                difficulty_source="metadata_fallback",
                difficulty_confidence="fallback",
                trace_count=0,
                trace_success_rate=None,
                trace_avg_reward=None,
                trace_avg_turns=None,
                trace_avg_leak_count=None,
                trace_max_turns_rate=None,
                trace_context_limit_rate=None,
            )
            continue

        score, metrics, is_noisy = trace_scores[item_id]
        if is_noisy:
            label = "noisy"
        elif score <= easy_threshold:
            label = "easy"
        elif score <= hard_threshold:
            label = "medium"
        else:
            label = "hard"
        trace_count = int(metrics["trace_count"])
        labels[item_id] = DifficultyStats(
            difficulty_label=label,
            difficulty_score=score,
            difficulty_source="trace",
            difficulty_confidence=confidence(trace_count, "trace"),
            trace_count=trace_count,
            trace_success_rate=metrics["trace_success_rate"],
            trace_avg_reward=metrics["trace_avg_reward"],
            trace_avg_turns=metrics["trace_avg_turns"],
            trace_avg_leak_count=metrics["trace_avg_leak_count"],
            trace_max_turns_rate=metrics["trace_max_turns_rate"],
            trace_context_limit_rate=metrics["trace_context_limit_rate"],
        )

    threshold_report = {
        "binning": "relative_quantile",
        "easy_threshold_q33": easy_threshold,
        "hard_threshold_q67": hard_threshold,
        "non_noisy_trace_labeled_items": len(non_noisy_scores),
        "noisy_rule": {
            "context_limit_rate_gt": 0.0,
            "avg_leak_count_gte": NOISY_LEAK_THRESHOLD,
        },
        "score_formula": (
            "100 * (0.45*(1-success_rate) + 0.25*min(avg_turns/10,1) "
            "+ 0.20*max_turns_rate + 0.10*min(avg_leak_count/3,1))"
        ),
    }
    return labels, threshold_report


def with_difficulty_metadata(
    row: dict[str, Any],
    *,
    id_field: str,
    labels: dict[str, DifficultyStats],
) -> dict[str, Any]:
    item_id = row_id(row, id_field)
    stats = labels[item_id]
    new_row = dict(row)
    metadata = dict(new_row.get("metadata") or {})
    metadata.update(asdict(stats))
    new_row["metadata"] = metadata
    return new_row


def label_distribution_report(
    dataset: Any,
    *,
    labels: dict[str, DifficultyStats],
    id_field: str,
) -> dict[str, Any]:
    by_label: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_confidence: Counter[str] = Counter()
    by_level: dict[str, Counter[str]] = defaultdict(Counter)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in dataset:
        item_id = row_id(row, id_field)
        stats = labels[item_id]
        metadata = row.get("metadata") or {}
        label = stats.difficulty_label
        by_label[label] += 1
        by_source[stats.difficulty_source] += 1
        by_confidence[stats.difficulty_confidence] += 1
        by_level[str(metadata.get("level", "unknown"))][label] += 1
        by_type[str(metadata.get("type", "unknown"))][label] += 1

    return {
        "by_label": dict(sorted(by_label.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_level": {
            level: dict(sorted(counter.items()))
            for level, counter in sorted(by_level.items())
        },
        "by_type": {
            problem_type: dict(sorted(counter.items()))
            for problem_type, counter in sorted(by_type.items())
        },
    }


def write_csv(
    path: Path,
    *,
    dataset: Any,
    labels: dict[str, DifficultyStats],
    id_field: str,
) -> None:
    fieldnames = [
        "id",
        "level",
        "type",
        "difficulty_label",
        "difficulty_score",
        "difficulty_source",
        "difficulty_confidence",
        "trace_count",
        "trace_success_rate",
        "trace_avg_reward",
        "trace_avg_turns",
        "trace_avg_leak_count",
        "trace_max_turns_rate",
        "trace_context_limit_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            item_id = row_id(row, id_field)
            metadata = row.get("metadata") or {}
            stats = asdict(labels[item_id])
            writer.writerow(
                {
                    "id": item_id,
                    "level": metadata.get("level", ""),
                    "type": metadata.get("type", ""),
                    **stats,
                }
            )


def main() -> None:
    args = parse_args()

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to label tutor datasets."
        ) from exc

    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve()

    loaded = load_from_disk(str(dataset_path))
    if not isinstance(loaded, DatasetDict):
        raise ValueError(f"Expected DatasetDict at {dataset_path}")
    if args.split not in loaded:
        raise ValueError(
            f"Split '{args.split}' not found at {dataset_path}; available: {list(loaded)}"
        )

    target_split = loaded[args.split]
    task_to_id = build_task_index(
        target_split,
        id_field=args.id_field,
        task_field=args.task_field,
    )
    traces_by_id, trace_report = collect_traces(
        dirs=trace_dirs(args),
        task_to_id=task_to_id,
    )
    labels, threshold_report = build_labels(
        target_split,
        traces_by_id=traces_by_id,
        id_field=args.id_field,
    )

    split_reports: dict[str, Any] = {}
    for split_name, split_dataset in loaded.items():
        if split_name == args.split:
            split_reports[split_name] = {
                "input_rows": len(split_dataset),
                "output_rows": len(split_dataset),
                "labeled": True,
                **label_distribution_report(
                    split_dataset,
                    labels=labels,
                    id_field=args.id_field,
                ),
            }
        else:
            split_reports[split_name] = {
                "input_rows": len(split_dataset),
                "output_rows": len(split_dataset),
                "labeled": False,
                "copied_unchanged": True,
            }

    report = {
        "input_dataset": str(dataset_path),
        "output_dataset": str(output_path),
        "split": args.split,
        "trace_report": trace_report,
        "thresholds": threshold_report,
        "splits": split_reports,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, dataset=target_split, labels=labels, id_field=args.id_field)

    if args.dry_run:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Dry run: wrote report {report_path} and CSV {csv_path}; dataset not saved.")
        return

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    output_splits: dict[str, Any] = {}
    for split_name, split_dataset in loaded.items():
        if split_name == args.split:
            output_splits[split_name] = split_dataset.map(
                lambda row: with_difficulty_metadata(
                    row,
                    id_field=args.id_field,
                    labels=labels,
                ),
                desc=f"Label {split_name} difficulty",
            )
        else:
            output_splits[split_name] = split_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(output_splits).save_to_disk(str(output_path))
    print(
        f"Wrote {output_path}: "
        + ", ".join(f"{split}={len(dataset)}" for split, dataset in output_splits.items())
    )
    print(f"Wrote report {report_path}")
    print(f"Wrote CSV {csv_path}")


if __name__ == "__main__":
    main()
