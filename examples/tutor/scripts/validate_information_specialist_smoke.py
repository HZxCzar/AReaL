#!/usr/bin/env python3
"""Fail-closed post-run checks for the 0818 information-specialist smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MECHANISMS = ("original", "student_fade", "teacher_fade", "long_drop")
TEACHERS = MECHANISMS
BEHAVIORS = ("text", "code")
RETEST_LEVELS = ("original", "original_preleak")
DIAGNOSTIC_COUNT_FIELDS = (
    "teacher_pre_error_count",
    "leak_check_failed_count",
    "answer_judge_failed_count",
)


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing results file: {path}")
    latest: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        key = str(record.get("key") or "")
        if not key:
            raise ValueError(f"{path}:{line_number}: result has no key")
        latest[key] = record
    return latest


def _trace_path(cell_dir: Path, raw_path: Any) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ValueError(f"{cell_dir}: result has no trace_path")
    path = Path(text)
    return path if path.is_absolute() else cell_dir / path


def _validate_level(
    *,
    cell_dir: Path,
    key: str,
    level: str,
    payload: Any,
    replays: int,
) -> int:
    if not isinstance(payload, dict):
        raise ValueError(f"{cell_dir}: {key} has no {level} result")
    if not payload.get("attempted") or payload.get("skipped"):
        raise ValueError(f"{cell_dir}: {key} did not attempt {level}")
    if payload.get("student_error") is not None:
        raise ValueError(
            f"{cell_dir}: {key} {level} student error: {payload['student_error']}"
        )
    replay_count = int(payload.get("replay_count", -1))
    replay_correct = int(payload.get("replay_correct", -1))
    if replay_count != replays or not 0 <= replay_correct <= replay_count:
        raise ValueError(
            f"{cell_dir}: {key} {level} has replay_correct/count "
            f"{replay_correct}/{replay_count}, expected count {replays}"
        )
    score = payload.get("score")
    expected_score = replay_correct / replay_count
    if score is None or not math.isclose(
        float(score), expected_score, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{cell_dir}: {key} {level} score {score!r} does not match "
            f"{replay_correct}/{replay_count}"
        )
    return replay_correct


def validate_smoke(
    run_dir: Path,
    *,
    expected_rows: int,
    replays: int,
    diagonal_only: bool = False,
) -> dict[str, Any]:
    """Validate the selected smoke cells and reject a collapsed behavior."""

    if not 1 <= expected_rows <= 8:
        raise ValueError(f"smoke expected_rows must be in [1, 8], got {expected_rows}")
    if replays < 1:
        raise ValueError(f"replays must be positive, got {replays}")

    axis_totals = {
        (behavior, mechanism): {"records": 0, "replays": 0, "correct": 0}
        for behavior in BEHAVIORS
        for mechanism in MECHANISMS
    }
    cell_reports: dict[str, dict[str, int]] = {}

    for teacher in TEACHERS:
        for behavior in BEHAVIORS:
            for mechanism in MECHANISMS:
                if diagonal_only and teacher != mechanism:
                    continue
                student = f"qwen3-1.7b-{behavior}-{mechanism}"
                cell_dir = run_dir / "cells" / teacher / student
                records = _latest_records(cell_dir / "results.jsonl")
                if len(records) != expected_rows:
                    raise ValueError(
                        f"{cell_dir}: found {len(records)} latest records, "
                        f"expected {expected_rows}"
                    )

                cell_correct = 0
                for key, record in records.items():
                    if record.get("error") is not None:
                        raise ValueError(f"{cell_dir}: {key} error: {record['error']}")
                    if record.get("student_name") != student:
                        raise ValueError(
                            f"{cell_dir}: {key} belongs to {record.get('student_name')}"
                        )
                    if record.get("student_call_failed"):
                        raise ValueError(
                            f"{cell_dir}: {key} has a student call failure"
                        )
                    failed_counts = {
                        field: int(record.get(field, 0) or 0)
                        for field in DIAGNOSTIC_COUNT_FIELDS
                        if int(record.get(field, 0) or 0) != 0
                    }
                    if failed_counts:
                        raise ValueError(
                            f"{cell_dir}: {key} diagnostic failures {failed_counts}"
                        )
                    trace_path = _trace_path(cell_dir, record.get("trace_path"))
                    if not trace_path.is_file():
                        raise ValueError(
                            f"{cell_dir}: {key} missing trace {trace_path}"
                        )
                    if behavior == "code" and not isinstance(
                        record.get("code_stats"), dict
                    ):
                        raise ValueError(f"{cell_dir}: {key} has no code_stats")

                    generalization = record.get("generalization") or {}
                    for level in RETEST_LEVELS:
                        correct = _validate_level(
                            cell_dir=cell_dir,
                            key=key,
                            level=level,
                            payload=generalization.get(level),
                            replays=replays,
                        )
                        if level == "original":
                            cell_correct += correct

                expected_replays = expected_rows * replays
                cell_key = f"{teacher}/{student}"
                cell_reports[cell_key] = {
                    "records": len(records),
                    "replays": expected_replays,
                    "correct": cell_correct,
                }
                totals = axis_totals[(behavior, mechanism)]
                totals["records"] += len(records)
                totals["replays"] += expected_replays
                totals["correct"] += cell_correct

    if diagonal_only:
        zero_behaviors = [
            behavior
            for behavior in BEHAVIORS
            if sum(
                axis_totals[(behavior, mechanism)]["correct"]
                for mechanism in MECHANISMS
            )
            == 0
        ]
        if zero_behaviors:
            raise ValueError(
                "all original re-test replays were wrong for behaviors: "
                + ", ".join(zero_behaviors)
            )
    else:
        zero_axes = [
            f"{behavior}/{mechanism}"
            for (behavior, mechanism), totals in axis_totals.items()
            if totals["correct"] == 0
        ]
        if zero_axes:
            raise ValueError(
                "all original re-test replays were wrong for behavior/mask axes: "
                + ", ".join(zero_axes)
            )

    axes: dict[str, dict[str, dict[str, float | int]]] = {
        behavior: {} for behavior in BEHAVIORS
    }
    for (behavior, mechanism), totals in axis_totals.items():
        axes[behavior][mechanism] = {
            **totals,
            "accuracy": totals["correct"] / totals["replays"],
        }
    return {
        "status": "pass",
        "expected_cells": (
            len(BEHAVIORS) * len(MECHANISMS)
            if diagonal_only
            else len(TEACHERS) * len(BEHAVIORS) * len(MECHANISMS)
        ),
        "matrix_scope": "diagonal" if diagonal_only else "full",
        "expected_rows_per_cell": expected_rows,
        "replays_per_retest": replays,
        "fresh_code_session_regression": "passed_before_gpu_launch",
        "trace_policy": "all",
        "axes": axes,
        "cells": cell_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--diagonal-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_smoke(
            args.run_dir,
            expected_rows=args.expected_rows,
            replays=args.replays,
            diagonal_only=args.diagonal_only,
        )
    except ValueError as exc:
        raise SystemExit(f"smoke gate failed: {exc}") from exc

    output = args.output or args.run_dir / "smoke_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    for behavior in BEHAVIORS:
        for mechanism in MECHANISMS:
            totals = report["axes"][behavior][mechanism]
            print(
                f"[smoke-ok] {behavior}/{mechanism}: "
                f"{totals['correct']}/{totals['replays']} correct replays"
            )
    print(
        f"[smoke-ok] {report['expected_cells']} cells complete, "
        "no call failures, all traces present"
    )
    print(f"[smoke-ok] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
