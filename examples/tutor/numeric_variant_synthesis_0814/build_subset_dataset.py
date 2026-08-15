#!/usr/bin/env python3
"""Build a dialogue-variant/original-retest DatasetDict from accepted variants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

SCRIPT_DIR = Path(__file__).resolve().parent
TUTOR_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE = TUTOR_DIR / "data/math_1.7b_8b/math_pass@2"
DEFAULT_BANK = SCRIPT_DIR / "runs/full_train_test/variant_bank.jsonl"
DEFAULT_OUTPUT = TUTOR_DIR / "data/math_1.7b_8b/numeric_variant_retest_original_0814"
SPLITS = ("train", "test")


def read_bank(path: Path) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            source_id = str(item["source_id"])
            if source_id in bank:
                raise ValueError(
                    f"duplicate source_id {source_id!r} at line {line_number}"
                )
            split = str(item["split"])
            if split not in SPLITS:
                raise ValueError(f"unsupported split {split!r} for {source_id}")
            required = (
                "teacher_variant_task",
                "teacher_variant_answer",
                "original_retest_task",
                "original_retest_ground_truth",
            )
            missing = [name for name in required if not str(item.get(name, ""))]
            if missing:
                raise ValueError(f"{source_id} is missing required fields {missing}")
            checks = item.get("mechanical_checks")
            if not isinstance(checks, dict) or not checks or not all(checks.values()):
                raise ValueError(f"{source_id} did not preserve all mechanical checks")
            bank[source_id] = item
    if not bank:
        raise ValueError(f"accepted variant bank is empty: {path}")
    return bank


def build_row(source: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    source_id = str(source["id"])
    original_task = str(source["task"])
    original_ground_truth = str(source["ground_truth"])
    if str(item["original_retest_task"]) != original_task:
        raise ValueError(
            f"{source_id}: bank original task does not match source dataset"
        )
    if str(item["original_retest_ground_truth"]) != original_ground_truth:
        raise ValueError(
            f"{source_id}: bank original ground truth does not match source dataset"
        )

    variant_task = str(item["teacher_variant_task"])
    variant_answer = str(item["teacher_variant_answer"])
    if variant_task == original_task:
        raise ValueError(f"{source_id}: accepted variant task equals original task")

    row = dict(source)
    metadata = dict(row.get("metadata") or {})
    original_solution = str(metadata.get("solution", ""))
    # The source solution solves the re-test problem, not the dialogue variant.
    # Keep it explicitly with the re-test and make the generic solution slot
    # empty so no future reader can accidentally hand the wrong derivation to
    # the teacher.
    metadata["solution"] = ""
    row.update(
        {
            "task": variant_task,
            "ground_truth": variant_answer,
            "problem": variant_task,
            "answer": variant_answer,
            "prompt": variant_task,
            "metadata": metadata,
            "retest_task": original_task,
            "retest_ground_truth": original_ground_truth,
            "retest_reference_solution": original_solution,
            "variant_edits_json": json.dumps(
                item.get("edits", []), ensure_ascii=False, sort_keys=True
            ),
            "variant_mechanical_checks_json": json.dumps(
                item.get("mechanical_checks", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "variant_semantic_audit_json": json.dumps(
                item.get("semantic_audit", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "variant_answer_audit_json": json.dumps(
                item.get("answer_audit", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "variant_teacher_solve_answers_json": json.dumps(
                item.get("teacher_solve_answers", []),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "variant_student_solved": bool(
                (item.get("student_bonus") or {}).get("solved", False)
            ),
        }
    )
    return row


def build_dataset(source_path: Path, bank_path: Path) -> DatasetDict:
    source = load_from_disk(str(source_path))
    if not isinstance(source, DatasetDict):
        raise TypeError(f"source must be a DatasetDict: {source_path}")
    bank = read_bank(bank_path)
    consumed: set[str] = set()
    output: dict[str, Dataset] = {}

    for split in SPLITS:
        if split not in source:
            raise ValueError(f"source DatasetDict is missing split {split!r}")
        rows = []
        for raw in source[split]:
            source_id = str(raw["id"])
            item = bank.get(source_id)
            if item is None or str(item["split"]) != split:
                continue
            rows.append(build_row(dict(raw), item))
            consumed.add(source_id)
        output[split] = Dataset.from_list(rows)

    missing = sorted(set(bank).difference(consumed))
    if missing:
        raise ValueError(
            f"{len(missing)} accepted variants were not found: {missing[:5]}"
        )
    return DatasetDict(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    bank = args.bank.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(source, bank)
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    dataset.save_to_disk(str(temporary))
    manifest = {
        "format": "dialogue_variant_original_retest_v1",
        "source_dataset": str(source),
        "accepted_variant_bank": str(bank),
        "output_dataset": str(output),
        "split_counts": {split: len(dataset[split]) for split in SPLITS},
        "dialogue_fields": {"task": "variant", "ground_truth": "variant_answer"},
        "retest_fields": {
            "retest_task": "original",
            "retest_ground_truth": "original_ground_truth",
        },
    }
    with (temporary / "variant_subset_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
