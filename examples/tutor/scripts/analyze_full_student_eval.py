#!/usr/bin/env python3
"""Summarize one full-student teacher against the eight standard 0818 students."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MECHANISMS = ("original", "student_fade", "teacher_fade", "long_drop")
DISPLAY_MECHANISMS = ("original", "student_fade", "long_drop", "teacher_fade")
SPLIT_BEHAVIOR = {"id": "text", "ood": "code"}


@dataclass(frozen=True)
class Observation:
    item_id: str
    score: float
    preleak_score: float
    leaked: bool
    format_error: bool
    call_failed: bool


def parse_student_name(name: str) -> tuple[str, str]:
    for behavior in SPLIT_BEHAVIOR.values():
        marker = f"-{behavior}-"
        if marker in name:
            mechanism = name.rsplit(marker, 1)[1]
            if mechanism in MECHANISMS:
                return behavior, mechanism
    raise ValueError(f"Cannot parse standard student name {name!r}.")


def score(record: dict[str, Any], level: str) -> float:
    value = (record.get("generalization") or {}).get(level)
    if not isinstance(value, dict) or value.get("score") is None:
        raise ValueError(f"missing generalization.{level}.score")
    return float(value["score"])


def load_split(path: Path, expected_behavior: str) -> dict[str, list[Observation]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Episode keys identify dataset item/mode/attempt, but they are intentionally
    # reused by different singleton student cells after those cells are merged.
    # De-duplicate retries within a student, never across students.
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        student_name = str(record.get("student_name") or "")
        episode_key = str(record.get("key") or f"line-{line_number}")
        latest[(student_name, episode_key)] = record

    by_mechanism: dict[str, list[Observation]] = defaultdict(list)
    errors: list[str] = []
    for (student_name, episode_key), record in latest.items():
        key = f"{student_name}/{episode_key}"
        if record.get("mode") not in {None, "presolve_on"}:
            continue
        if record.get("error"):
            errors.append(f"{key}: {record['error']}")
            continue
        try:
            behavior, mechanism = parse_student_name(str(record["student_name"]))
            if behavior != expected_behavior:
                raise ValueError(
                    f"expected {expected_behavior} student, found {behavior}"
                )
            by_mechanism[mechanism].append(
                Observation(
                    item_id=str(record.get("item_id") or record["dataset_index"]),
                    score=score(record, "original"),
                    preleak_score=score(record, "original_preleak"),
                    leaked=int(record.get("leak_count", 0) or 0) > 0,
                    format_error=int(record.get("format_error_count", 0) or 0) > 0,
                    call_failed=bool(
                        record.get("student_call_failed")
                        or int(record.get("leak_check_failed_count", 0) or 0) > 0
                        or int(record.get("answer_judge_failed_count", 0) or 0) > 0
                        or int(record.get("teacher_pre_error_count", 0) or 0) > 0
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        raise ValueError(
            f"{path}: {len(errors)} invalid/error record(s): " + "; ".join(errors[:3])
        )
    missing = sorted(set(MECHANISMS) - set(by_mechanism))
    if missing:
        raise ValueError(f"{path}: missing mechanisms {missing}.")
    return dict(by_mechanism)


def item_means(observations: list[Observation], field: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        values[observation.item_id].append(float(getattr(observation, field)))
    return {item: statistics.fmean(xs) for item, xs in values.items()}


def boolean_rate(observations: list[Observation], field: str) -> float:
    values = [bool(getattr(observation, field)) for observation in observations]
    return statistics.fmean(values) if values else math.nan


def analyze_split(eval_root: Path, split: str) -> dict[str, Any]:
    behavior = SPLIT_BEHAVIOR[split]
    path = eval_root / "teachers" / "full" / split / "results.jsonl"
    loaded = load_split(path, behavior)
    score_items = {
        mechanism: item_means(loaded[mechanism], "score")
        for mechanism in MECHANISMS
    }
    preleak_items = {
        mechanism: item_means(loaded[mechanism], "preleak_score")
        for mechanism in MECHANISMS
    }
    item_sets = [set(score_items[mechanism]) for mechanism in MECHANISMS]
    complete_items = sorted(set.intersection(*item_sets))
    counts = {mechanism: len(score_items[mechanism]) for mechanism in MECHANISMS}
    if not complete_items or any(set(items) != set(complete_items) for items in score_items.values()):
        raise ValueError(
            f"{split}: student cells do not contain the identical item set; counts={counts}."
        )
    if any(set(preleak_items[mechanism]) != set(complete_items) for mechanism in MECHANISMS):
        raise ValueError(f"{split}: pre-leak item coverage differs from original coverage.")

    original = {
        mechanism: statistics.fmean(
            score_items[mechanism][item] for item in complete_items
        )
        for mechanism in MECHANISMS
    }
    preleak = {
        mechanism: statistics.fmean(
            preleak_items[mechanism][item] for item in complete_items
        )
        for mechanism in MECHANISMS
    }
    return {
        "behavior": behavior,
        "items_per_student": len(complete_items),
        "record_counts": {
            mechanism: len(loaded[mechanism]) for mechanism in MECHANISMS
        },
        "original_score": original,
        "original_preleak_score": preleak,
        "leak_episode_rate": {
            mechanism: boolean_rate(loaded[mechanism], "leaked")
            for mechanism in MECHANISMS
        },
        "format_error_episode_rate": {
            mechanism: boolean_rate(loaded[mechanism], "format_error")
            for mechanism in MECHANISMS
        },
        "diagnostic_call_failure_rate": {
            mechanism: boolean_rate(loaded[mechanism], "call_failed")
            for mechanism in MECHANISMS
        },
    }


def markdown_row(title: str, values: dict[str, float]) -> str:
    header = "| full teacher \\ student | " + " | ".join(DISPLAY_MECHANISMS) + " |"
    separator = "|---|" + "---:|" * len(DISPLAY_MECHANISMS)
    row = f"| {title} | " + " | ".join(
        f"{values[mechanism]:.4f}" for mechanism in DISPLAY_MECHANISMS
    ) + " |"
    return "\n".join((header, separator, row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = {
        "teacher": "full",
        **{
            split: analyze_split(args.eval_root, split)
            for split in SPLIT_BEHAVIOR
        },
    }
    output = args.output or args.eval_root / "matrix_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split, behavior in SPLIT_BEHAVIOR.items():
        result = summary[split]
        print(f"\n[{split.upper()} / {behavior}] items={result['items_per_student']}")
        print(markdown_row("original score", result["original_score"]))
        print()
        print(markdown_row("pre-leak score", result["original_preleak_score"]))
        print()
        print(markdown_row("leak episode rate", result["leak_episode_rate"]))
        print()
        print(
            markdown_row(
                "diagnostic call failure rate",
                result["diagnostic_call_failure_rate"],
            )
        )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
