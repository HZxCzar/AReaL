import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalize_item_id(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"Cannot parse item id from {value!r}")
    return int(match.group(1))


def load_split_ids(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Split file must be a JSON list: {path}")
    resolved = []
    for item in payload:
        if isinstance(item, dict):
            raw = item.get("item_id", item.get("id"))
        else:
            raw = item
        resolved.append(normalize_item_id(raw))
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("CodeCoach manifest must be a list")
    train_ids = load_split_ids(args.train_ids.resolve() if args.train_ids else None)
    test_ids = load_split_ids(args.test_ids.resolve() if args.test_ids else None)

    base_dir = manifest_path.parent
    rows = []
    for item in manifest:
        task_path = resolve_path(base_dir, item["task_markdown_path"])
        code_path = resolve_path(base_dir, item["initial_code_path"])
        evaluator_path = resolve_path(base_dir, item["evaluator_path"])
        rows.append(
            {
                "id": int(item["id"]),
                "task_markdown": task_path.read_text(encoding="utf-8"),
                "initial_code": code_path.read_text(encoding="utf-8"),
                "evaluator_path": str(evaluator_path),
                "target_score": float(item["target_score"]),
                "entry_function": str(item["entry_function"]),
                "eval_timeout_sec": int(item.get("eval_timeout_sec", 120)),
                "metadata": item.get("metadata") or {},
            }
        )

    if train_ids is not None or test_ids is not None:
        rows_by_id: dict[int, dict] = {row["id"]: row for row in rows}

        def materialize(ids: list[int] | None) -> list[dict]:
            if ids is None:
                return []
            materialized = []
            for item_id in ids:
                if item_id not in rows_by_id:
                    raise ValueError(f"CodeCoach item id {item_id} not found in manifest")
                materialized.append(dict(rows_by_id[item_id]))
            return materialized

        train_rows = materialize(train_ids)
        test_rows = materialize(test_ids)
        if train_ids is None:
            excluded = {row["id"] for row in test_rows}
            train_rows = [row for row in rows if row["id"] not in excluded]
        if test_ids is None:
            excluded = {row["id"] for row in train_rows}
            test_rows = [row for row in rows if row["id"] not in excluded]
        if not train_rows or not test_rows:
            raise ValueError("CodeCoach split by ids produced an empty train or test split")
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
