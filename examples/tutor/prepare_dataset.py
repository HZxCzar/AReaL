import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Tutor manifest must be a list")

    rows = []
    for item in manifest:
        rows.append(
            {
                "id": int(item["id"]),
                "task": str(item["task"]),
                "ground_truth": str(item["ground_truth"]),
                "metadata": item.get("metadata") or {},
            }
        )

    split_at = max(1, len(rows) - max(1, args.test_size))
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(rows[:split_at]),
            "test": Dataset.from_list(rows[split_at:]),
        }
    )
    dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()
