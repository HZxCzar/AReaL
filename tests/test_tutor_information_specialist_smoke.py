from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.tutor.scripts.validate_information_specialist_smoke import (
    BEHAVIORS,
    MECHANISMS,
    TEACHERS,
    validate_smoke,
)


def _write_smoke_matrix(
    root: Path,
    *,
    zero_axis: tuple[str, str] | None = None,
    failed_student: str | None = None,
) -> None:
    for teacher in TEACHERS:
        for behavior in BEHAVIORS:
            for mechanism in MECHANISMS:
                student = f"qwen3-1.7b-{behavior}-{mechanism}"
                cell = root / "cells" / teacher / student
                trace_dir = cell / "traces" / "presolve_on"
                trace_dir.mkdir(parents=True, exist_ok=True)
                records = []
                for row in range(2):
                    trace = trace_dir / f"row_{row:05d}.json"
                    trace.write_text("{}\n", encoding="utf-8")
                    correct = int(
                        teacher == "original"
                        and row == 0
                        and zero_axis != (behavior, mechanism)
                    )
                    level = {
                        "attempted": True,
                        "skipped": False,
                        "student_error": None,
                        "replay_count": 8,
                        "replay_correct": correct,
                        "score": correct / 8,
                    }
                    records.append(
                        {
                            "key": f"presolve_on:{row}:1:{student}",
                            "student_name": student,
                            "error": None,
                            "student_call_failed": student == failed_student,
                            "teacher_pre_error_count": 0,
                            "leak_check_failed_count": 0,
                            "answer_judge_failed_count": 0,
                            "trace_path": str(trace),
                            "code_stats": {} if behavior == "code" else None,
                            "generalization": {
                                "original": dict(level),
                                "original_preleak": dict(level),
                            },
                        }
                    )
                (cell / "results.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )


def test_validate_smoke_accepts_complete_nonzero_matrix(tmp_path: Path) -> None:
    """All 32 healthy cells and every behavior/mask axis pass the gate."""

    _write_smoke_matrix(tmp_path)

    report = validate_smoke(tmp_path, expected_rows=2, replays=8)

    assert report["status"] == "pass"
    assert len(report["cells"]) == 32
    assert report["axes"]["code"]["student_fade"]["correct"] == 1
    assert report["axes"]["text"]["long_drop"]["replays"] == 64


@pytest.mark.parametrize("behavior", BEHAVIORS)
def test_validate_smoke_rejects_all_zero_axis(tmp_path: Path, behavior: str) -> None:
    """A channel/mask-wide zero is a failed smoke, not a plausible full run."""

    _write_smoke_matrix(tmp_path, zero_axis=(behavior, "long_drop"))

    with pytest.raises(ValueError, match=rf"{behavior}/long_drop"):
        validate_smoke(tmp_path, expected_rows=2, replays=8)


def test_validate_smoke_rejects_diagnostic_call_failure(tmp_path: Path) -> None:
    """A completed results file cannot hide a student endpoint failure."""

    student = "qwen3-1.7b-code-original"
    _write_smoke_matrix(tmp_path, failed_student=student)

    with pytest.raises(ValueError, match="student call failure"):
        validate_smoke(tmp_path, expected_rows=2, replays=8)


def test_launcher_wires_both_pre_full_smoke_gates() -> None:
    """The public smoke command runs mechanics before GPU and health after it."""

    source = Path("examples/tutor/scripts/eval_0818_step50_specialists.sh").read_text(
        encoding="utf-8"
    )

    regression_call = '"$PYTHON" -B -m pytest -q -p no:cacheprovider'
    validator_call = '"$PYTHON" -B "$SMOKE_VALIDATOR"'
    server_call = 'setsid "$PYTHON" -m sglang.launch_server'
    assert regression_call in source
    assert validator_call in source
    assert source.index(regression_call) < source.index(server_call)
    assert source.index(validator_call) > source.index(server_call)
    assert "smoke requires SAVE_TRACES=all" in source
    assert "smoke refuses SKIP_LIVENESS=1" in source


def test_launcher_is_teacher_major_code_first_and_benchmarks_h200_concurrency() -> None:
    """Private matrix scheduling prioritizes useful partial results and saturation."""

    source = Path("examples/tutor/scripts/eval_0818_step50_specialists.sh").read_text(
        encoding="utf-8"
    )

    assert "for teacher_index in \"${!MECHANISMS[@]}\"" in source
    assert "for behavior in code text" in source
    assert 'barrier_wait "teacher-${teacher_index}-${behavior}"' in source
    assert 'write_teacher_progress "$teacher"' in source
    assert "BENCHMARK_CONCURRENCIES:-3,6,12,16" in source
    assert source.count('--max-running-requests "$SERVER_MAX_RUNNING_REQUESTS"') == 2
    assert '"student_axes.0.template.max_concurrent_calls=$CALLER_MAX_CONCURRENT"' in source
    assert "benchmark--concurrency-$concurrency.gpu.csv" in source
