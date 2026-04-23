import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_manifest(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Tutor manifest must be a list")
    return manifest


def normalize_item_id(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"Cannot parse item id from {value!r}")
    return int(match.group(1))


def load_split_ids(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Split file must be a JSON list: {path}")
    resolved = set()
    for item in payload:
        if isinstance(item, dict):
            raw = item.get("item_id", item.get("id"))
        else:
            raw = item
        resolved.add(normalize_item_id(raw))
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--train-ids", type=Path, default=None)
    parser.add_argument("--test-ids", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    manifest = load_manifest(manifest_path)
    train_ids = load_split_ids(args.train_ids.resolve() if args.train_ids else None)
    test_ids = load_split_ids(args.test_ids.resolve() if args.test_ids else None)

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

    if train_ids is not None or test_ids is not None:
        train_rows = [row for row in rows if train_ids is None or row["id"] in train_ids]
        test_rows = [row for row in rows if test_ids is None or row["id"] in test_ids]
        if train_ids is not None and not train_rows:
            raise ValueError("No tutor train rows matched --train-ids")
        if test_ids is not None and not test_rows:
            raise ValueError("No tutor test rows matched --test-ids")
        if train_ids is None:
            excluded = {row["id"] for row in test_rows}
            train_rows = [row for row in rows if row["id"] not in excluded]
        if test_ids is None:
            excluded = {row["id"] for row in train_rows}
            test_rows = [row for row in rows if row["id"] not in excluded]
        if not train_rows or not test_rows:
            raise ValueError("Tutor split by ids produced an empty train or test split")
    else:
        split_at = max(1, len(rows) - max(1, args.test_size))
        train_rows = rows[:split_at]
        test_rows = rows[split_at:]

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    dataset.save_to_disk(str(output_path))


if __name__ == "__main__":
    main()
