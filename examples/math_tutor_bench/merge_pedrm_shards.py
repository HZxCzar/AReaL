#!/usr/bin/env python3
"""Merge data-parallel Ped-RM shards into the standard result layout."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from score_pedrm import PEDAGOGY_TASKS, mean, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    tasks_root = args.tasks_root.resolve()
    shards_root = args.shards_root.resolve()
    output = args.output.resolve()
    aggregate = {}

    for task in PEDAGOGY_TASKS:
        data = json.loads(
            (tasks_root / task / "generations.json").read_text(encoding="utf-8")
        )
        scores = {}
        for shard_index in range(args.num_shards):
            path = shards_root / f"{task}-{shard_index}.json"
            if not path.is_file():
                raise SystemExit(f"missing Ped-RM shard: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = (task, shard_index, args.num_shards, len(data))
            actual = (
                payload.get("task"),
                payload.get("shard_index"),
                payload.get("num_shards"),
                payload.get("total_task_samples"),
            )
            if actual != expected:
                raise SystemExit(f"incompatible Ped-RM shard {path}: {actual}")
            for record in payload["records"]:
                index = int(record["index"])
                if index in scores:
                    raise SystemExit(f"duplicate index {index} for {task}")
                scores[index] = (
                    float(record["candidate_score"]),
                    float(record["reference_score"]),
                )

        if set(scores) != set(range(len(data))):
            raise SystemExit(f"incomplete Ped-RM shards for {task}")

        candidate_scores = [scores[index][0] for index in range(len(data))]
        reference_scores = [scores[index][1] for index in range(len(data))]
        margins = [
            candidate - reference
            for candidate, reference in zip(candidate_scores, reference_scores)
        ]
        metrics = {
            "win_rate": sum(margin > 0 for margin in margins) / len(margins),
            "score": mean(candidate_scores),
            "baseline_score": mean(reference_scores),
            "mean_margin": mean(margins),
            "total_samples": len(margins),
        }
        enriched = deepcopy(data)
        for index, (candidate, reference) in scores.items():
            enriched[index]["chosen_score"] = candidate
            enriched[index]["rejected_score"] = reference

        task_output = output / task
        task_output.mkdir(parents=True, exist_ok=True)
        write_json(task_output / "metrics.json", metrics)
        write_json(task_output / "enriched_generations.json", enriched)
        aggregate[task] = metrics
        print(f"[pedrm] {task}: win_rate={metrics['win_rate']:.6f}")

    write_json(output / "pedrm_metrics.json", aggregate)


if __name__ == "__main__":
    main()
