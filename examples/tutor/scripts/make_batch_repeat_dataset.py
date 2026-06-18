"""Create a train split where each unique task fills one full batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def build_batch_repeat_dataset(
    source: Path,
    output: Path,
    *,
    id_field: str = "id",
    repeat_count: int = 16,
) -> None:
    """Write a dataset with unique train rows repeated in contiguous blocks."""
    if repeat_count <= 0:
        raise ValueError(f"repeat_count must be positive, got {repeat_count}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    dataset = load_from_disk(str(source))
    if "train" not in dataset:
        raise ValueError(f"source dataset is missing a train split: {source}")

    train = dataset["train"]
    if id_field not in train.column_names:
        raise ValueError(
            f"train split is missing id field '{id_field}'. "
            f"Available columns: {train.column_names}"
        )

    seen_ids = set()
    unique_indices = []
    for idx, row in enumerate(train):
        row_id = row[id_field]
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        unique_indices.append(idx)

    if not unique_indices:
        raise ValueError("source train split has no rows")

    repeated_blocks = []
    for idx in unique_indices:
        row_dataset = train.select([idx])
        repeated_blocks.extend([row_dataset] * repeat_count)

    output_dataset = DatasetDict({"train": concatenate_datasets(repeated_blocks)})
    for split_name, split_dataset in dataset.items():
        if split_name != "train":
            output_dataset[split_name] = split_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(output))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a saved HuggingFace dataset whose train split contains each "
            "unique task repeated contiguously repeat-count times. Non-train "
            "splits are copied unchanged."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--repeat-count", type=int, default=16)
    args = parser.parse_args()

    build_batch_repeat_dataset(
        source=args.source,
        output=args.output,
        id_field=args.id_field,
        repeat_count=args.repeat_count,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
