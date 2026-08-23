from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.tutor.scripts.analyze_information_specialist_matrix import (
    MECHANISMS,
    analyze,
    parse_student_name,
)


def _write_split(root: Path, split: str, behavior: str) -> None:
    teachers = ("baseline", *MECHANISMS)
    for teacher in teachers:
        output = root / "teachers" / teacher / split / "results.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for mechanism in MECHANISMS:
            for item in range(12):
                if teacher == "baseline":
                    score = 0.4
                elif split == "id":
                    score = 0.8 if teacher == mechanism else 0.2
                else:
                    score = 0.5
                records.append(
                    {
                        "answer_judge_failed_count": 0,
                        "attempt": 1,
                        "dataset_index": item,
                        "error": None,
                        "format_error_count": 0,
                        "generalization": {
                            "original": {"score": score},
                            "original_preleak": {"score": score},
                        },
                        "item_id": f"item-{item}",
                        "leak_check_failed_count": 0,
                        "leak_count": 0,
                        "mode": "presolve_on",
                        "student_call_failed": False,
                        "student_name": f"qwen3-1.7b-{behavior}-{mechanism}",
                        "teacher_pre_error_count": 0,
                    }
                )
        output.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


def test_parse_student_name_reads_both_axes() -> None:
    assert parse_student_name("qwen3-1.7b-text-teacher_fade") == (
        "text",
        "teacher_fade",
    )
    assert parse_student_name("qwen3-1.7b-code-long_drop") == ("code", "long_drop")


def test_analyze_detects_id_specialization_and_ood_equivalence(tmp_path: Path) -> None:
    _write_split(tmp_path, "id", "text")
    _write_split(tmp_path, "ood", "code")

    summary = analyze(tmp_path, ("id", "ood"), bootstrap_samples=200, seed=7)

    assert summary["id"]["matched_minus_mismatched"] == pytest.approx(0.6)
    assert summary["id"]["preleak_matched_minus_mismatched"] == pytest.approx(0.6)
    assert summary["id"]["label_permutation_exact_p"] == pytest.approx(1.0 / 24.0)
    assert summary["id"]["diagonal_teacher_rank_in_column"] == [1, 1, 1, 1]
    assert summary["id"]["specialization_screen_passed"] is True
    assert summary["ood"]["matched_minus_mismatched"] == pytest.approx(0.0)
    assert summary["ood"]["ood_equivalence_screen_passed"] is True


def test_analyze_without_common_baseline(tmp_path: Path) -> None:
    _write_split(tmp_path, "id", "text")
    _write_split(tmp_path, "ood", "code")
    for split in ("id", "ood"):
        (tmp_path / "teachers" / "baseline" / split / "results.jsonl").unlink()

    summary = analyze(
        tmp_path,
        ("id", "ood"),
        bootstrap_samples=200,
        seed=7,
        include_baseline=False,
    )

    assert summary["id"]["matched_minus_mismatched"] == pytest.approx(0.6)
    assert summary["id"]["common_baseline_vector"] is None
    assert summary["id"]["delta_from_common_baseline_matrix"] is None
