#!/usr/bin/env python3
"""Reconstruct turn-micro preference compliance from sampled training traces.

The legacy training metric averages one compliance ratio per episode.  This
script instead pools eligible gate passes and calls within 25-update windows.
For attempt-diagnosis, gates before the first real student response are
ineligible because there is no attempt to diagnose yet.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import orjson


RUNS = {
    "ALL": "20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu",
    "Attempt single": (
        "20260901_182004_0901-preference-v3-reward-v3-attempt-diagnosis-8gpu"
    ),
    "Subgoal single": (
        "20260901_182422_0901-preference-v3-reward-v3-subgoal-decomposition-8gpu"
    ),
}
PERSONALITIES = (
    "attempt-diagnosis",
    "subgoal-decomposition",
    "contrastive-comparison",
)
TRACE_ROOT = Path(
    "/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/"
    "tutor/debug_traces/tutor-math-baseline"
)
LOG_ROOT = Path(
    "/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/"
    "tutor/logs/root/tutor-math-baseline"
)
TRACE_NAME_RE = re.compile(r"^task_(\d+)_(\d+)\.json$")


def _metric_boundaries(run: str, max_step: int) -> tuple[list[float], list[int]]:
    wall_times: list[float] = []
    steps: list[int] = []
    with (LOG_ROOT / run / "metrics.jsonl").open("rb") as handle:
        for line in handle:
            record = orjson.loads(line)
            step = int(record["global_step"])
            if step > max_step:
                break
            wall_times.append(float(record["wall_time"]))
            steps.append(step)
    if not wall_times or steps[-1] < max_step:
        raise RuntimeError(
            f"{run} has no complete metrics history through step {max_step}"
        )
    return wall_times, steps


def _personality(student_name: str) -> str | None:
    for personality in PERSONALITIES:
        if student_name.endswith(personality):
            return personality
    return None


def _real_student_response(turn: dict) -> bool:
    """Mirror the rollout branch that updates last_real_student_output."""
    return bool(str(turn.get("student_output") or "").strip()) and not bool(
        turn.get("personality_gated")
    ) and not bool(turn.get("leak_masked"))


def _read_run(
    item: tuple[str, str, int, int, int],
) -> tuple[str, dict[tuple[str, int], tuple[int, int]], int, int]:
    label, run, max_step, bin_size, task_modulus = item
    wall_times, steps = _metric_boundaries(run, max_step)
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    parsed = 0
    skipped = 0

    trace_dir = TRACE_ROOT / run / "train"
    for entry in os.scandir(trace_dir):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        match = TRACE_NAME_RE.match(entry.name)
        if match is None:
            skipped += 1
            continue
        if int(match.group(1)) % task_modulus:
            continue
        timestamp = int(match.group(2)) / 1000.0
        metric_index = bisect.bisect_left(wall_times, timestamp)
        if metric_index >= len(wall_times):
            continue
        step = steps[metric_index]
        if step > max_step:
            continue

        try:
            with open(entry.path, "rb") as handle:
                record = orjson.loads(handle.read())
        except (OSError, orjson.JSONDecodeError):
            skipped += 1
            continue
        personality = _personality(str(record.get("student", {}).get("name", "")))
        if personality is None:
            continue

        has_real_student_turn = bool(
            str(record.get("initial_student_answer") or "").strip()
        )
        bin_end = ((step // bin_size) + 1) * bin_size
        for turn in record.get("turns", []):
            gate = turn.get("personality_gate_result")
            sampled = bool(gate is not None and gate.get("sampled", True))
            eligible = personality != "attempt-diagnosis" or has_real_student_turn
            if sampled and eligible:
                bucket = counts[(personality, bin_end)]
                bucket[1] += 1
                bucket[0] += int(bool(gate.get("passed")))
            if _real_student_response(turn):
                has_real_student_turn = True
        parsed += 1

    frozen = {key: (value[0], value[1]) for key, value in counts.items()}
    return label, frozen, parsed, skipped


def _series(
    counts: dict[tuple[str, int], tuple[int, int]],
    personalities: tuple[str, ...],
    max_step: int,
    bin_size: int,
) -> tuple[list[int], list[float], list[int], list[int]]:
    xs: list[int] = []
    rates: list[float] = []
    passes: list[int] = []
    calls: list[int] = []
    for bin_end in range(bin_size, max_step + 2, bin_size):
        passed = sum(counts.get((p, bin_end), (0, 0))[0] for p in personalities)
        called = sum(counts.get((p, bin_end), (0, 0))[1] for p in personalities)
        if called:
            xs.append(bin_end)
            rates.append(100.0 * passed / called)
            passes.append(passed)
            calls.append(called)
    return xs, rates, passes, calls


def _write_csv(
    path: Path,
    all_counts: dict[str, dict[tuple[str, int], tuple[int, int]]],
    max_step: int,
    bin_size: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run", "preference", "bin_end_step", "passes", "calls", "compliance"]
        )
        for label, counts in all_counts.items():
            for personality in PERSONALITIES:
                xs, rates, passes, calls = _series(
                    counts, (personality,), max_step, bin_size
                )
                for x, rate, passed, called in zip(
                    xs, rates, passes, calls, strict=True
                ):
                    writer.writerow(
                        [label, personality, x, passed, called, f"{rate / 100.0:.8f}"]
                    )


def _plot(
    path: Path,
    all_counts: dict[str, dict[tuple[str, int], tuple[int, int]]],
    max_step: int,
    bin_size: int,
    task_modulus: int,
) -> None:
    colors = {
        "ALL": "#2563eb",
        "Attempt single": "#dc2626",
        "Subgoal single": "#059669",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)

    aggregate_personalities = {
        "ALL": PERSONALITIES,
        "Attempt single": ("attempt-diagnosis",),
        "Subgoal single": ("subgoal-decomposition",),
    }
    for label, personalities in aggregate_personalities.items():
        xs, rates, _passes, _calls = _series(
            all_counts[label], personalities, max_step, bin_size
        )
        axes[0].plot(xs, rates, color=colors[label], linewidth=2.0, label=label)
    axes[0].set_title("Run-level training compliance")
    axes[0].legend(frameon=False, loc="lower right")

    for label in ("ALL", "Attempt single"):
        xs, rates, _passes, _calls = _series(
            all_counts[label], ("attempt-diagnosis",), max_step, bin_size
        )
        axes[1].plot(xs, rates, color=colors[label], linewidth=2.0, label=label)
    axes[1].set_title("Attempt diagnosis\n(after first real student response)")
    axes[1].legend(frameon=False, loc="lower right")

    for label in ("ALL", "Subgoal single"):
        xs, rates, _passes, _calls = _series(
            all_counts[label], ("subgoal-decomposition",), max_step, bin_size
        )
        axes[2].plot(xs, rates, color=colors[label], linewidth=2.0, label=label)
    axes[2].set_title("Subgoal decomposition")
    axes[2].legend(frameon=False, loc="lower right")

    for axis in axes:
        axis.set_xlim(bin_size, max_step + 1)
        axis.set_ylim(0, 100)
        axis.set_xlabel("Training updates")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Eligible gate pass rate (%)")
    fig.suptitle(
        f"0901 turn-micro compliance ({bin_size}-update pooled bins; "
        f"task_id mod {task_modulus} sample)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-step", type=int, default=999)
    parser.add_argument("--bin-size", type=int, default=25)
    parser.add_argument(
        "--task-modulus",
        type=int,
        default=50,
        help="Read traces whose task id is divisible by this value (traces exist every 10).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("0901_training_compliance_micro_step1000"),
    )
    args = parser.parse_args()
    if args.max_step < 0 or args.bin_size <= 0 or args.task_modulus <= 0:
        parser.error(
            "--max-step must be >= 0; --bin-size and --task-modulus must be > 0"
        )

    jobs = [
        (label, run, args.max_step, args.bin_size, args.task_modulus)
        for label, run in RUNS.items()
    ]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(_read_run, jobs))

    all_counts: dict[str, dict[tuple[str, int], tuple[int, int]]] = {}
    for label, counts, parsed, skipped in results:
        all_counts[label] = counts
        print(f"{label}: parsed {parsed:,} traces; skipped {skipped:,}")

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")
    _write_csv(csv_path, all_counts, args.max_step, args.bin_size)
    _plot(
        png_path,
        all_counts,
        args.max_step,
        args.bin_size,
        args.task_modulus,
    )
    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
