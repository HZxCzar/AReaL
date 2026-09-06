#!/usr/bin/env python3
"""Plot pooled 20-step subgoal gate compliance for the selected V3/V4 runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


LOG_ROOT = Path(
    "/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/"
    "tutor/logs/root/tutor-math-baseline"
)
OUTPUT = Path(__file__).resolve().parents[3] / "v4_all_vs_v3_subgoal_compliance_1500.png"
RUNS = (
    (
        "V4 all (episode normalization)",
        "20260905_043603_0901-preference-v3-reward-v4-all-id-8gpu",
        "#1769aa",
        "o",
    ),
    (
        "V3 all (turn normalization)",
        "20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu",
        "#6a3d9a",
        "s",
    ),
    (
        "V3 subgoal single (turn normalization)",
        "20260901_182422_0901-preference-v3-reward-v3-subgoal-decomposition-8gpu",
        "#ef6c00",
        "o",
    ),
    (
        "V4 subgoal single (episode normalization)",
        "20260905_143037_0901-preference-v3-reward-v4-subgoal-decomposition-8gpu",
        "#159447",
        "D",
    ),
)
PREFIX = "rollout/personality/subgoal-decomposition"
BIN_SIZE = 20


def load_series(run: str) -> tuple[list[float], list[float], int]:
    bins: dict[int, list[float]] = {}
    last_step = -1
    with (LOG_ROOT / run / "metrics.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            step = int(record["global_step"])
            stats = record["stats"]
            gated = stats.get(f"{PREFIX}/gated_turns")
            calls = stats.get(f"{PREFIX}/gate_calls")
            if gated is None or calls is None:
                continue
            bucket = bins.setdefault(step // BIN_SIZE, [0.0, 0.0, 0.0, 0.0])
            bucket[0] += step
            bucket[1] += 1
            bucket[2] += float(gated)
            bucket[3] += float(calls)
            last_step = max(last_step, step)
    xs, ys = [], []
    for step_sum, count, gated_sum, call_sum in bins.values():
        if call_sum > 0:
            xs.append(step_sum / count)
            ys.append(1.0 - gated_sum / call_sum)
    return xs, ys, last_step


def main() -> None:
    fig, axis = plt.subplots(figsize=(14, 6.7))
    endpoints = []
    for label, run, color, marker in RUNS:
        xs, ys, last_step = load_series(run)
        endpoints.append(last_step)
        axis.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=2.2,
            label=f"{label} (shown through step {last_step})",
        )

    x_max = max(1500, ((max(endpoints) + 99) // 100) * 100)
    axis.set_title(
        "Subgoal new compliance: V4 all vs V3 all vs V3/V4 subgoal single "
        f"(0–{x_max} steps)"
    )
    axis.set_xlabel("Training step (20-step bins)")
    axis.set_ylabel("Subgoal compliance (turn-micro)")
    axis.set_xlim(0, x_max)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(OUTPUT)


if __name__ == "__main__":
    main()
