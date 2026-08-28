#!/usr/bin/env python3
"""Build live and final summaries for the 0825 full-student preference run."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _load_latest_results(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    latest: dict[str, dict[str, Any]] = {}
    parse_errors = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            key = str(payload.get("key") or "")
            if not key:
                parse_errors += 1
                continue
            latest[key] = payload
    return list(latest.values()), parse_errors


def _retest(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    attempted_episodes = 0
    replay_count = 0
    replay_correct = 0
    score_sum = 0.0
    for record in records:
        original = (record.get("generalization") or {}).get("original") or {}
        if not original.get("attempted") or original.get("skipped"):
            continue
        attempted_episodes += 1
        item_replays = int(original.get("replay_count", 0) or 0)
        item_correct = int(original.get("replay_correct", 0) or 0)
        replay_count += item_replays
        replay_correct += item_correct
        if original.get("score") is not None:
            score_sum += float(original["score"])
        elif item_replays:
            score_sum += item_correct / item_replays
        else:
            score_sum += float(original.get("correct") is True)
    return {
        "attempted_episode_count": attempted_episodes,
        "replay_count": replay_count,
        "replay_correct_count": replay_correct,
        "accuracy_on_replays": _rate(replay_correct, replay_count),
        "mean_episode_score": _rate(score_sum, attempted_episodes),
    }


def _outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(records),
        "taught_success_count": sum(
            record.get("taught_success") is True for record in records
        ),
        "taught_success_rate": _rate(
            sum(record.get("taught_success") is True for record in records),
            len(records),
        ),
        "original_retest": _retest(records),
    }


def _cell_summary(
    cell_dir: Path,
    *,
    preference: str,
    expected: int,
) -> dict[str, Any]:
    records, parse_errors = _load_latest_results(cell_dir / "results.jsonl")
    completed = [record for record in records if record.get("error") is None]
    gates = [
        record.get("personality_gate") or {}
        for record in completed
        if isinstance(record.get("personality_gate"), dict)
    ]
    active_gates = [gate for gate in gates if gate.get("active") is True]
    sampled_turns = sum(
        int(gate.get("sampled_turn_count", 0) or 0) for gate in active_gates
    )
    passed_turns = sum(
        int(gate.get("passed_turn_count", 0) or 0) for gate in active_gates
    )
    turn1_checked = [gate for gate in active_gates if gate.get("turn1_sampled")]
    complaint_keys = {
        str(record.get("key"))
        for record in completed
        if (record.get("personality_gate") or {}).get("first_complaint_turn")
        is not None
    }
    complaint_records = [
        record for record in completed if str(record.get("key")) in complaint_keys
    ]
    no_complaint_records = [
        record for record in completed if str(record.get("key")) not in complaint_keys
    ]
    complaint_gates = [
        record.get("personality_gate") or {} for record in complaint_records
    ]
    first_post_checked = [
        gate
        for gate in complaint_gates
        if gate.get("first_post_complaint_passed") is not None
    ]
    post_sampled = sum(
        int(gate.get("post_complaint_sampled_turn_count", 0) or 0)
        for gate in complaint_gates
    )
    post_passed = sum(
        int(gate.get("post_complaint_passed_turn_count", 0) or 0)
        for gate in complaint_gates
    )
    sustained_checked = [
        gate
        for gate in complaint_gates
        if gate.get("post_complaint_all_passed") is not None
    ]
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
    gate_summary = {
        "active_episode_count": len(active_gates),
        "sampled_turn_count": sampled_turns,
        "passed_turn_count": passed_turns,
        "micro_compliance": _rate(passed_turns, sampled_turns),
        "turn1_checked_episode_count": len(turn1_checked),
        "turn1_compliance": _rate(
            sum(gate.get("turn1_passed") is True for gate in turn1_checked),
            len(turn1_checked),
        ),
        "gate_error_count": sum(
            int(gate.get("gate_error_count", 0) or 0) for gate in active_gates
        ),
        "complaint_episode_count": len(complaint_records),
        "complaint_episode_rate": _rate(len(complaint_records), len(active_gates)),
        "first_post_complaint_checked_episode_count": len(first_post_checked),
        "first_post_complaint_compliance": _rate(
            sum(
                gate.get("first_post_complaint_passed") is True
                for gate in first_post_checked
            ),
            len(first_post_checked),
        ),
        "post_complaint_sampled_turn_count": post_sampled,
        "post_complaint_micro_compliance": _rate(post_passed, post_sampled),
        "post_complaint_sustained_rate": _rate(
            sum(
                gate.get("post_complaint_all_passed") is True
                for gate in sustained_checked
            ),
            len(sustained_checked),
        ),
    }
    if classification_label_counts:
        gate_summary["classification_label_counts"] = dict(
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
        gate_summary["classification_distribution_count"] = (
            classification_distribution_count
        )
        gate_summary["mean_classification_probabilities"] = {
            label: total / classification_distribution_count
            for label, total in sorted(probability_sums.items())
        }
        if margin_count:
            gate_summary["mean_classification_margin"] = margin_sum / margin_count
    return {
        "preference": preference,
        "results_path": str((cell_dir / "results.jsonl").resolve()),
        "expected_episode_count": expected,
        "recorded_episode_count": len(records),
        "completed_episode_count": len(completed),
        "error_count": len(records) - len(completed),
        "jsonl_parse_error_count": parse_errors,
        "progress": _rate(len(records), expected),
        "teaching": _outcomes(completed),
        "gate": gate_summary,
        "after_complaint": _outcomes(complaint_records),
        "without_complaint": _outcomes(no_complaint_records),
        "diagnostics": {
            "leaked_episode_count": sum(
                int(record.get("leak_count", 0) or 0) > 0 for record in completed
            ),
            "format_error_episode_count": sum(
                int(record.get("format_error_count", 0) or 0) > 0
                for record in completed
            ),
            "student_call_failed_count": sum(
                record.get("student_call_failed") is True for record in completed
            ),
            "answer_judge_failed_episode_count": sum(
                int(record.get("answer_judge_failed_count", 0) or 0) > 0
                for record in completed
            ),
        },
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teacher = str(manifest.get("teacher") or "none")
    students = list(manifest.get("students") or [])
    expected = int((manifest.get("dataset") or {}).get("phase_rows", 0) or 0)
    cells: dict[str, Any] = {}
    for student in students:
        preference = str(student["preference"])
        name = str(student["name"])
        cells[preference] = _cell_summary(
            run_dir / "cells" / teacher / name,
            preference=preference,
            expected=expected,
        )
    total_expected = expected * len(students)
    total_recorded = sum(cell["recorded_episode_count"] for cell in cells.values())
    overlap_config = manifest.get("gate_overlap_audit") or {}
    overlap_enabled = overlap_config.get("enabled") is True
    overlap_path = run_dir / "gate_overlap" / "summary.json"
    overlap_summary: dict[str, Any] | None = None
    if overlap_enabled and overlap_path.is_file():
        loaded_overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        slices = {}
        for name, value in (loaded_overlap.get("slices") or {}).items():
            compact = dict(value)
            compact.pop("multi_pass_examples", None)
            slices[name] = compact
        overlap_summary = {
            "summary_path": str(overlap_path.resolve()),
            "complete": loaded_overlap.get("complete") is True,
            "progress": loaded_overlap.get("progress"),
            "teacher_reply_count": loaded_overlap.get("teacher_reply_count"),
            "first_turn_reply_count": loaded_overlap.get("first_turn_reply_count"),
            "expected_decision_count": loaded_overlap.get("expected_decision_count"),
            "recorded_decision_count": loaded_overlap.get("recorded_decision_count"),
            "applicability": loaded_overlap.get("applicability"),
            "gates": loaded_overlap.get("gates"),
            "slices": slices,
        }
    elif overlap_enabled:
        overlap_summary = {
            "summary_path": str(overlap_path.resolve()),
            "complete": False,
            "progress": 0.0,
        }
    cells_complete = bool(cells) and all(
        cell["recorded_episode_count"] >= expected for cell in cells.values()
    )
    overlap_complete = not overlap_enabled or bool(
        overlap_summary and overlap_summary.get("complete") is True
    )
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "teacher": teacher,
        "adapter": manifest.get("adapter"),
        "dataset": manifest.get("dataset"),
        "total_expected_episode_count": total_expected,
        "total_recorded_episode_count": total_recorded,
        "overall_progress": _rate(total_recorded, total_expected),
        "complete": cells_complete and overlap_complete,
        "gate_overlap_audit": overlap_summary,
        "cells": cells,
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _tsv(report: dict[str, Any]) -> str:
    header = (
        "preference\tprogress\trecorded\terrors\tgate\tturn1\tcomplaint_rate\t"
        "post1\tpost_micro\tsustained\tteach_success\tretest\t"
        "complaint_teach_success\tcomplaint_retest\tleak\tformat"
    )
    rows = [header]
    for preference, cell in report["cells"].items():
        gate = cell["gate"]
        teaching = cell["teaching"]
        after = cell["after_complaint"]
        diagnostics = cell["diagnostics"]
        rows.append(
            "\t".join(
                (
                    preference,
                    _fmt(cell["progress"]),
                    str(cell["recorded_episode_count"]),
                    str(cell["error_count"]),
                    _fmt(gate["micro_compliance"]),
                    _fmt(gate["turn1_compliance"]),
                    _fmt(gate["complaint_episode_rate"]),
                    _fmt(gate["first_post_complaint_compliance"]),
                    _fmt(gate["post_complaint_micro_compliance"]),
                    _fmt(gate["post_complaint_sustained_rate"]),
                    _fmt(teaching["taught_success_rate"]),
                    _fmt(teaching["original_retest"]["accuracy_on_replays"]),
                    _fmt(after["taught_success_rate"]),
                    _fmt(after["original_retest"]["accuracy_on_replays"]),
                    str(diagnostics["leaked_episode_count"]),
                    str(diagnostics["format_error_episode_count"]),
                )
            )
        )
    return "\n".join(rows) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_report(run_dir: Path) -> dict[str, Any]:
    report = build_report(run_dir)
    _atomic_write(
        run_dir / "live_summary.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(run_dir / "live_summary.tsv", _tsv(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    while True:
        report = write_report(run_dir)
        if not args.watch or report["complete"]:
            print(_tsv(report), end="")
            print(f"[analysis] wrote {run_dir / 'live_summary.json'}")
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
