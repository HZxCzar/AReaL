#!/usr/bin/env python3
"""Combine the selected personality evaluation cells into one readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _cell_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing cell summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    modes = summary.get("modes") or {}
    if set(modes) != {"presolve_on"}:
        raise ValueError(f"{summary_path}: expected only presolve_on, got {sorted(modes)}")
    mode = modes["presolve_on"]
    generalization = mode.get("generalization") or {}
    original = generalization.get("original") or {}
    preleak = generalization.get("original_preleak") or {}
    gate = mode.get("personality_gate") or {}
    expected = int(mode.get("expected_attempts", 0) or 0)
    completed = int(mode.get("completed_attempts", 0) or 0)
    pending_info = summary.get("pending_backfill") or {}
    return {
        "dataset_rows": int(summary.get("dataset_rows", 0) or 0),
        "expected_attempts": expected,
        "completed_attempts": completed,
        "error_count": int(mode.get("error_count", 0) or 0),
        "pending_backfill_count": int(pending_info.get("count", 0) or 0),
        "execution_coverage_rate": mode.get("execution_coverage_rate"),
        "regular_eval_score": original.get("accuracy_on_replays"),
        "preleak_eval_score": preleak.get("accuracy_on_replays"),
        "retest_replay_count": int(original.get("replay_attempt_count", 0) or 0),
        "leaked_episode_count": int(mode.get("leaked_episode_count", 0) or 0),
        "leaked_episode_rate": (
            int(mode.get("leaked_episode_count", 0) or 0) / completed
            if completed
            else None
        ),
        "format_error_episode_count": int(
            mode.get("format_error_episode_count", 0) or 0
        ),
        "format_error_episode_rate": (
            int(mode.get("format_error_episode_count", 0) or 0) / completed
            if completed
            else None
        ),
        "student_call_failed_count": int(
            mode.get("student_call_failed_count", 0) or 0
        ),
        "gate": {
            "micro_compliance": gate.get("micro_compliance"),
            "turn1_compliance": gate.get("turn1_compliance"),
            "complaint_episode_rate": gate.get("complaint_episode_rate"),
            "first_post_complaint_compliance": gate.get(
                "first_post_complaint_compliance"
            ),
            "post_complaint_micro_compliance": gate.get(
                "post_complaint_micro_compliance"
            ),
            "post_complaint_sustained_rate": gate.get(
                "post_complaint_sustained_rate"
            ),
            "sampled_turn_count": int(gate.get("sampled_turn_count", 0) or 0),
            "gated_turn_count": int(gate.get("gated_turn_count", 0) or 0),
            "gate_error_count": int(gate.get("gate_error_count", 0) or 0),
            "first_post_complaint_checked_episode_count": int(
                gate.get("first_post_complaint_checked_episode_count", 0) or 0
            ),
        },
    }


def build_report(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teachers = list(manifest.get("teachers") or [])
    students = list(manifest.get("students") or [])
    if not teachers or not students:
        raise ValueError("manifest must describe at least one teacher and student")
    if len(teachers) != len(set(map(str, teachers))):
        raise ValueError("manifest contains duplicate teachers")
    student_names = [str(student["name"]) for student in students]
    if len(student_names) != len(set(student_names)):
        raise ValueError("manifest contains duplicate students")

    report: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "phase": manifest.get("phase"),
        "sample_count_per_cell": manifest.get("eval_max_samples"),
        "generalization_replays": manifest.get("student_generalize_replays"),
        "adapters": manifest.get("adapters"),
        "teachers": {},
    }
    for teacher in teachers:
        teacher_name = str(teacher)
        cells: dict[str, Any] = {}
        for student in students:
            personality = str(student["personality"])
            student_name = str(student["name"])
            split = str(student["split"])
            cell = _cell_summary(run_dir / "cells" / teacher_name / student_name)
            cell["split"] = split
            cell["student_name"] = student_name
            cells[personality] = cell

        by_split: dict[str, Any] = {}
        for split in ("id", "ood"):
            selected = [cell for cell in cells.values() if cell["split"] == split]
            by_split[split] = {
                "cell_count": len(selected),
                "macro_regular_eval_score": _mean(
                    [cell["regular_eval_score"] for cell in selected]
                ),
                "macro_preleak_eval_score": _mean(
                    [cell["preleak_eval_score"] for cell in selected]
                ),
                "macro_gate_compliance": _mean(
                    [cell["gate"]["micro_compliance"] for cell in selected]
                ),
                "macro_turn1_compliance": _mean(
                    [cell["gate"]["turn1_compliance"] for cell in selected]
                ),
                "macro_first_post_complaint_compliance": _mean(
                    [
                        cell["gate"]["first_post_complaint_compliance"]
                        for cell in selected
                    ]
                ),
                "macro_post_complaint_sustained_rate": _mean(
                    [
                        cell["gate"]["post_complaint_sustained_rate"]
                        for cell in selected
                    ]
                ),
            }
        report["teachers"][teacher_name] = {"splits": by_split, "cells": cells}
    return report


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def print_table(report: dict[str, Any]) -> None:
    print(
        "teacher\tstudent\tsplit\tscore\tgate\tturn1\tpost1\tsustained\tleak\tformat"
    )
    for teacher, payload in report["teachers"].items():
        for personality, cell in payload["cells"].items():
            gate = cell["gate"]
            print(
                "\t".join(
                    (
                        teacher,
                        personality,
                        cell["split"],
                        _fmt(cell["regular_eval_score"]),
                        _fmt(gate["micro_compliance"]),
                        _fmt(gate["turn1_compliance"]),
                        _fmt(gate["first_post_complaint_compliance"]),
                        _fmt(gate["post_complaint_sustained_rate"]),
                        _fmt(cell["leaked_episode_rate"]),
                        _fmt(cell["format_error_episode_rate"]),
                    )
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    report = build_report(run_dir)
    output = args.output or (run_dir / "matrix_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print_table(report)
    print(f"[analysis] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
