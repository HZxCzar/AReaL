#!/usr/bin/env python3
"""Analyze the paired 4x4 information-specialist screening matrix.

Expected layout::

    EVAL_ROOT/teachers/{baseline,original,student_fade,teacher_fade,long_drop}/
        {id,ood}/results.jsonl

The primary contrast is matched minus mismatched performance. It gives every
teacher row and every student column equal weight, so teacher-quality and
student-difficulty main effects cancel without assuming what a good teaching
"pattern" should look like. The baseline adapter is used only to report changes
from the common branch point; it cancels algebraically from the primary contrast.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MECHANISMS = ("original", "student_fade", "teacher_fade", "long_drop")
TEACHERS = MECHANISMS
SPLIT_BEHAVIOR = {"id": "text", "ood": "code"}


@dataclass(frozen=True)
class Observation:
    item_id: str
    attempt: int
    score: float
    preleak_score: float | None
    leaked: bool
    format_error: bool
    call_failed: bool


def parse_student_name(name: str) -> tuple[str, str]:
    for behavior in SPLIT_BEHAVIOR.values():
        marker = f"-{behavior}-"
        if marker not in name:
            continue
        mechanism = name.rsplit(marker, 1)[1]
        if mechanism in MECHANISMS:
            return behavior, mechanism
    raise ValueError(
        f"Cannot read behavior/information axes from student name {name!r}."
    )


def _optional_score(record: dict[str, Any], level: str) -> float | None:
    level_record = (record.get("generalization") or {}).get(level)
    if not isinstance(level_record, dict) or level_record.get("score") is None:
        return None
    return float(level_record["score"])


def load_results(path: Path, expected_behavior: str) -> dict[str, list[Observation]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_mechanism: dict[str, list[Observation]] = defaultdict(list)
    errors: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if record.get("mode") not in {None, "presolve_on"}:
            continue
        if record.get("error"):
            errors.append(f"line {line_number}: {record['error']}")
            continue
        behavior, mechanism = parse_student_name(str(record["student_name"]))
        if behavior != expected_behavior:
            raise ValueError(
                f"{path}: expected {expected_behavior} students, found {behavior} "
                f"on line {line_number}."
            )
        score = _optional_score(record, "original")
        if score is None:
            errors.append(f"line {line_number}: missing generalization.original.score")
            continue
        by_mechanism[mechanism].append(
            Observation(
                item_id=str(record.get("item_id") or record["dataset_index"]),
                attempt=int(record.get("attempt", 1)),
                score=score,
                preleak_score=_optional_score(record, "original_preleak"),
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
    if errors:
        preview = "; ".join(errors[:3])
        raise ValueError(f"{path}: {len(errors)} invalid/error records ({preview}).")
    missing = sorted(set(MECHANISMS) - set(by_mechanism))
    if missing:
        raise ValueError(f"{path}: missing student mechanisms {missing}.")
    return dict(by_mechanism)


def item_means(
    observations: Iterable[Observation], field: str = "score"
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        value = getattr(observation, field)
        if value is not None:
            values[observation.item_id].append(float(value))
    return {item_id: statistics.fmean(xs) for item_id, xs in values.items()}


def rate(observations: Iterable[Observation], field: str) -> float:
    xs = [bool(getattr(observation, field)) for observation in observations]
    return statistics.fmean(xs) if xs else math.nan


def matched_contrast(
    matrix: list[list[float]], permutation: tuple[int, ...] | None = None
) -> float:
    size = len(matrix)
    permutation = permutation or tuple(range(size))
    selected = {(row, permutation[row]) for row in range(size)}
    matched = [matrix[row][column] for row, column in selected]
    mismatched = [
        matrix[row][column]
        for row in range(size)
        for column in range(size)
        if (row, column) not in selected
    ]
    return statistics.fmean(matched) - statistics.fmean(mismatched)


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return percentile(draws, 0.025), percentile(draws, 0.975)


def markdown_matrix(title: str, matrix: list[list[float]]) -> str:
    lines = [title, "| teacher \\ student | " + " | ".join(MECHANISMS) + " |"]
    lines.append("|---|" + "---:|" * len(MECHANISMS))
    for teacher, row in zip(TEACHERS, matrix, strict=True):
        lines.append(
            f"| {teacher} | " + " | ".join(f"{value:.4f}" for value in row) + " |"
        )
    return "\n".join(lines)


def _complete_items(
    per_item: dict[str, dict[str, dict[str, float]]],
    teacher_names: Iterable[str],
) -> tuple[list[str], dict[str, int]]:
    sets = {
        f"{teacher}/{mechanism}": set(per_item[teacher][mechanism])
        for teacher in teacher_names
        for mechanism in MECHANISMS
    }
    complete = sorted(set.intersection(*sets.values()))
    return complete, {cell: len(items) for cell, items in sets.items()}


def analyze_split(
    eval_root: Path,
    split: str,
    *,
    bootstrap_samples: int,
    seed: int,
    min_complete_fraction: float,
    equivalence_margin: float,
    include_baseline: bool = True,
) -> dict[str, Any]:
    behavior = SPLIT_BEHAVIOR[split]
    teacher_names = ("baseline", *TEACHERS) if include_baseline else TEACHERS
    loaded: dict[str, dict[str, list[Observation]]] = {}
    for teacher in teacher_names:
        path = eval_root / "teachers" / teacher / split / "results.jsonl"
        loaded[teacher] = load_results(path, behavior)

    per_item = {
        teacher: {
            mechanism: item_means(loaded[teacher][mechanism])
            for mechanism in MECHANISMS
        }
        for teacher in teacher_names
    }
    preleak_per_item = {
        teacher: {
            mechanism: item_means(loaded[teacher][mechanism], field="preleak_score")
            for mechanism in MECHANISMS
        }
        for teacher in teacher_names
    }
    complete_items, cell_counts = _complete_items(per_item, teacher_names)
    complete_preleak_items, preleak_cell_counts = _complete_items(
        preleak_per_item, teacher_names
    )
    complete_items = sorted(set(complete_items).intersection(complete_preleak_items))
    maximum_items = max(cell_counts.values())
    complete_fraction = len(complete_items) / maximum_items if maximum_items else 0.0
    if not complete_items or complete_fraction < min_complete_fraction:
        raise ValueError(
            f"{split}: only {len(complete_items)}/{maximum_items} items are complete "
            f"across every cell; required fraction is {min_complete_fraction:.2f}."
        )

    score_matrix = [
        [
            statistics.fmean(
                per_item[teacher][mechanism][item] for item in complete_items
            )
            for mechanism in MECHANISMS
        ]
        for teacher in TEACHERS
    ]
    baseline_vector = (
        [
            statistics.fmean(
                per_item["baseline"][mechanism][item] for item in complete_items
            )
            for mechanism in MECHANISMS
        ]
        if include_baseline
        else None
    )
    delta_matrix = (
        [
            [score_matrix[row][column] - baseline_vector[column] for column in range(4)]
            for row in range(4)
        ]
        if baseline_vector is not None
        else None
    )
    preleak_score_matrix = [
        [
            statistics.fmean(
                preleak_per_item[teacher][mechanism][item] for item in complete_items
            )
            for mechanism in MECHANISMS
        ]
        for teacher in TEACHERS
    ]
    preleak_baseline_vector = (
        [
            statistics.fmean(
                preleak_per_item["baseline"][mechanism][item] for item in complete_items
            )
            for mechanism in MECHANISMS
        ]
        if include_baseline
        else None
    )
    preleak_delta_matrix = (
        [
            [
                preleak_score_matrix[row][column] - preleak_baseline_vector[column]
                for column in range(4)
            ]
            for row in range(4)
        ]
        if preleak_baseline_vector is not None
        else None
    )

    item_contrasts = []
    for item in complete_items:
        item_matrix = [
            [per_item[teacher][mechanism][item] for mechanism in MECHANISMS]
            for teacher in TEACHERS
        ]
        item_contrasts.append(matched_contrast(item_matrix))
    contrast = statistics.fmean(item_contrasts)
    ci_low, ci_high = bootstrap_mean_ci(
        item_contrasts, samples=bootstrap_samples, seed=seed
    )
    preleak_item_contrasts = []
    for item in complete_items:
        item_matrix = [
            [preleak_per_item[teacher][mechanism][item] for mechanism in MECHANISMS]
            for teacher in TEACHERS
        ]
        preleak_item_contrasts.append(matched_contrast(item_matrix))
    preleak_contrast = statistics.fmean(preleak_item_contrasts)
    preleak_ci_low, preleak_ci_high = bootstrap_mean_ci(
        preleak_item_contrasts,
        samples=bootstrap_samples,
        seed=seed + 10_000,
    )

    permutations = list(itertools.permutations(range(4)))
    permutation_values = [
        matched_contrast(score_matrix, permutation) for permutation in permutations
    ]
    exact_p = sum(value >= contrast - 1.0e-12 for value in permutation_values) / len(
        permutation_values
    )
    identity_rank = 1 + sum(value > contrast + 1.0e-12 for value in permutation_values)
    preleak_permutation_values = [
        matched_contrast(preleak_score_matrix, permutation)
        for permutation in permutations
    ]
    preleak_exact_p = sum(
        value >= preleak_contrast - 1.0e-12 for value in preleak_permutation_values
    ) / len(preleak_permutation_values)
    preleak_identity_rank = 1 + sum(
        value > preleak_contrast + 1.0e-12 for value in preleak_permutation_values
    )

    diagonal_margins = []
    diagonal_ranks = []
    preleak_diagonal_margins = []
    preleak_diagonal_ranks = []
    for column in range(4):
        diagonal = score_matrix[column][column]
        others = [score_matrix[row][column] for row in range(4) if row != column]
        diagonal_margins.append(diagonal - max(others))
        diagonal_ranks.append(1 + sum(value > diagonal for value in others))
        preleak_diagonal = preleak_score_matrix[column][column]
        preleak_others = [
            preleak_score_matrix[row][column] for row in range(4) if row != column
        ]
        preleak_diagonal_margins.append(preleak_diagonal - max(preleak_others))
        preleak_diagonal_ranks.append(
            1 + sum(value > preleak_diagonal for value in preleak_others)
        )

    leak_matrix = [
        [rate(loaded[teacher][mechanism], "leaked") for mechanism in MECHANISMS]
        for teacher in TEACHERS
    ]
    format_matrix = [
        [rate(loaded[teacher][mechanism], "format_error") for mechanism in MECHANISMS]
        for teacher in TEACHERS
    ]
    call_failure_matrix = [
        [rate(loaded[teacher][mechanism], "call_failed") for mechanism in MECHANISMS]
        for teacher in TEACHERS
    ]
    column_spreads = [
        max(score_matrix[row][column] for row in range(4))
        - min(score_matrix[row][column] for row in range(4))
        for column in range(4)
    ]

    mean_leak = statistics.fmean(value for row in leak_matrix for value in row)
    mean_format = statistics.fmean(value for row in format_matrix for value in row)
    max_call_failure = max(max(row) for row in call_failure_matrix)
    safety_ok = mean_leak <= 0.15 and mean_format <= 0.05 and max_call_failure <= 0.01
    specialization_screen = bool(
        split == "id"
        and ci_low > 0.0
        and preleak_ci_low > 0.0
        and exact_p <= 1.0 / 24.0 + 1.0e-12
        and preleak_exact_p <= 1.0 / 24.0 + 1.0e-12
        and all(rank == 1 for rank in diagonal_ranks)
        and all(rank == 1 for rank in preleak_diagonal_ranks)
        and safety_ok
    )
    equivalence_screen = bool(
        split == "ood"
        and ci_low >= -equivalence_margin
        and ci_high <= equivalence_margin
        and safety_ok
    )

    return {
        "behavior": behavior,
        "complete_items": len(complete_items),
        "complete_fraction": complete_fraction,
        "cell_item_counts": cell_counts,
        "preleak_cell_item_counts": preleak_cell_counts,
        "score_matrix": score_matrix,
        "common_baseline_vector": baseline_vector,
        "delta_from_common_baseline_matrix": delta_matrix,
        "matched_minus_mismatched": contrast,
        "bootstrap_95_ci": [ci_low, ci_high],
        "label_permutation_exact_p": exact_p,
        "label_permutation_rank": identity_rank,
        "preleak_score_matrix": preleak_score_matrix,
        "preleak_common_baseline_vector": preleak_baseline_vector,
        "preleak_delta_from_common_baseline_matrix": preleak_delta_matrix,
        "preleak_matched_minus_mismatched": preleak_contrast,
        "preleak_bootstrap_95_ci": [preleak_ci_low, preleak_ci_high],
        "preleak_label_permutation_exact_p": preleak_exact_p,
        "preleak_label_permutation_rank": preleak_identity_rank,
        "diagonal_margin_over_best_other_teacher": diagonal_margins,
        "diagonal_teacher_rank_in_column": diagonal_ranks,
        "preleak_diagonal_margin_over_best_other_teacher": (preleak_diagonal_margins),
        "preleak_diagonal_teacher_rank_in_column": preleak_diagonal_ranks,
        "teacher_spread_by_student_column": column_spreads,
        "leak_episode_rate_matrix": leak_matrix,
        "format_error_episode_rate_matrix": format_matrix,
        "diagnostic_call_failure_rate_matrix": call_failure_matrix,
        "mean_leak_episode_rate": mean_leak,
        "mean_format_error_episode_rate": mean_format,
        "max_diagnostic_call_failure_rate": max_call_failure,
        "safety_gate_passed": safety_ok,
        "specialization_screen_passed": specialization_screen,
        "equivalence_margin": equivalence_margin,
        "ood_equivalence_screen_passed": equivalence_screen,
    }


def analyze(
    eval_root: Path,
    splits: Iterable[str],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    min_complete_fraction: float = 0.95,
    equivalence_margin: float = 0.05,
    include_baseline: bool = True,
) -> dict[str, Any]:
    return {
        split: analyze_split(
            eval_root,
            split,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index,
            min_complete_fraction=min_complete_fraction,
            equivalence_margin=equivalence_margin,
            include_baseline=include_baseline,
        )
        for index, split in enumerate(splits)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=tuple(SPLIT_BEHAVIOR),
        help="Repeat to select splits; default: id and ood.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-complete-fraction", type=float, default=0.95)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Analyze only the four specialist rows; baseline-derived fields are null.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = args.splits or list(SPLIT_BEHAVIOR)
    summary = analyze(
        args.eval_root,
        splits,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        min_complete_fraction=args.min_complete_fraction,
        equivalence_margin=args.equivalence_margin,
        include_baseline=not args.no_baseline,
    )
    output = args.output or args.eval_root / "matrix_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for split in splits:
        result = summary[split]
        print(f"\n[{split.upper()} / {result['behavior']}]")
        print(markdown_matrix("score", result["score_matrix"]))
        if result["delta_from_common_baseline_matrix"] is not None:
            print()
            print(
                markdown_matrix(
                    "delta from common branch point",
                    result["delta_from_common_baseline_matrix"],
                )
            )
        print()
        print(markdown_matrix("pre-leak score", result["preleak_score_matrix"]))
        low, high = result["bootstrap_95_ci"]
        preleak_low, preleak_high = result["preleak_bootstrap_95_ci"]
        print(
            f"\nmatched-minus-mismatched = {result['matched_minus_mismatched']:.4f} "
            f"(task bootstrap 95% CI [{low:.4f}, {high:.4f}]); "
            f"label-permutation p={result['label_permutation_exact_p']:.4f}, "
            f"rank={result['label_permutation_rank']}/24"
        )
        print(
            f"pre-leak matched-minus-mismatched = "
            f"{result['preleak_matched_minus_mismatched']:.4f} "
            f"(task bootstrap 95% CI [{preleak_low:.4f}, {preleak_high:.4f}]); "
            f"label-permutation p={result['preleak_label_permutation_exact_p']:.4f}, "
            f"rank={result['preleak_label_permutation_rank']}/24"
        )
        print(
            "diagonal ranks by student column: "
            + ", ".join(
                f"{mechanism}={rank}"
                for mechanism, rank in zip(
                    MECHANISMS,
                    result["diagonal_teacher_rank_in_column"],
                    strict=True,
                )
            )
        )
        print(
            f"safety_gate={result['safety_gate_passed']} "
            f"specialization_screen={result['specialization_screen_passed']} "
            f"ood_equivalence_screen={result['ood_equivalence_screen_passed']}"
        )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
