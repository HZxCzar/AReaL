"""Merge disjoint evaluator shards without changing their saved trajectories."""

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _comparable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                str(Path(item).resolve())
                if key == "lora_path" and isinstance(item, str) and item
                else _comparable(item)
            )
            for key, item in value.items()
            if key != "base_url"
        }
    if isinstance(value, list):
        return [_comparable(item) for item in value]
    return value


def merge_cell(cell_dir: str | Path, shard_count: int) -> dict[str, Any]:
    from examples.tutor.scripts.evaluate_api_teacher import (
        EpisodeResult,
        PresolveMode,
        aggregate_report,
        result_needs_retry,
        rewrite_results_jsonl,
        write_json,
    )

    cell_dir = Path(cell_dir).resolve()
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    reference = None
    canonical = None
    results = []
    expected_keys: set[str] = set()
    seen: set[str] = set()
    for shard_index in range(shard_count):
        shard_dir = cell_dir / "shards" / str(shard_index)
        signature = json.loads((shard_dir / "run_config.json").read_text())["signature"]
        if signature.get("shard") != {"count": shard_count, "index": shard_index}:
            raise ValueError(f"Unexpected shard identity: {shard_dir}")
        comparable = deepcopy(signature)
        comparable["shard"].pop("index")
        comparable = _comparable(comparable)
        if reference is None:
            reference = signature
            canonical = comparable
            expected_keys = {
                f"{mode['name']}:{index}:{attempt}"
                for mode in signature["modes"]
                for index in range(int(signature["dataset_size"]))
                for attempt in range(1, int(signature["attempts"]) + 1)
            }
        elif comparable != canonical:
            raise ValueError(f"Evaluation signatures differ: {shard_dir}")
        # Resume journals may contain older copies; take the latest without
        # rewriting the original shard artifacts.
        latest = {}
        for line in (shard_dir / "results.jsonl").read_text().splitlines():
            if line.strip():
                result = EpisodeResult(**json.loads(line))
                latest[result.key] = result
        for result in latest.values():
            correct_key = f"{result.mode}:{result.dataset_index}:{result.attempt}"
            if (
                result.key != correct_key
                or result.key not in expected_keys
                or result.dataset_index % shard_count != shard_index
                or result.key in seen
            ):
                raise ValueError(f"Unexpected or overlapping episode: {result.key}")
            if result.student_prompt_index is not None:
                raise ValueError("This merger supports base-prompt evaluations only")
            if not result.trace_path or not Path(result.trace_path).is_file():
                raise ValueError(f"Missing full trajectory for {result.key}")
            seen.add(result.key)
            results.append(result)
    if seen != expected_keys:
        raise ValueError(
            f"Incomplete shards: missing {len(expected_keys - seen)} episodes"
        )
    assert reference is not None
    results.sort(key=lambda result: (result.mode, result.dataset_index, result.attempt))
    semantics = reference["test_semantics"]
    levels = tuple(semantics.get("generalization_levels", ()))
    pending = [
        result
        for result in results
        if result_needs_retry(
            result,
            retry_errors=True,
            retry_diagnostic_failures=True,
            generalization_levels=levels,
            expected_generalization_replays=int(
                semantics["student_generalize_replays"]
            ),
        )
    ]
    report = aggregate_report(
        results,
        modes=[PresolveMode(**mode) for mode in reference["modes"]],
        dataset_size=int(reference["dataset_size"]),
        attempts=int(reference["attempts"]),
        generalization_enabled=bool(semantics["student_generalize_enabled"]),
        generalization_levels=levels,
    )
    report.update(
        output_dir=str(cell_dir),
        pending_backfill={
            "count": len(pending),
            "path": str(cell_dir / "pending_backfill.jsonl"),
        },
        finished_at=datetime.now(UTC).isoformat(),
        merged_shards=shard_count,
    )
    merged_signature = deepcopy(reference)
    merged_signature.pop("shard")
    rewrite_results_jsonl(cell_dir / "results.jsonl", results)
    rewrite_results_jsonl(cell_dir / "pending_backfill.jsonl", pending)
    write_json(
        cell_dir / "run_config.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "signature": merged_signature,
            "merged_shards": shard_count,
        },
    )
    write_json(cell_dir / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_dir", type=Path)
    parser.add_argument("shard_count", type=int)
    args = parser.parse_args()
    report = merge_cell(args.cell_dir, args.shard_count)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "dataset_rows",
                    "recorded_total_attempts",
                    "pending_backfill",
                    "merged_shards",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
