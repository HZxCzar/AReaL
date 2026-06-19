from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Tutor manifest must be a list")
    return manifest


def normalize_item_id(value: Any) -> int:
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


def manifest_item_to_tutor_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(item["id"]),
        "task": str(item["task"]),
        "ground_truth": str(item["ground_truth"]),
        "metadata": item.get("metadata") or {},
    }


def build_aime_splits(
    *,
    manifest_path: Path,
    test_size: int,
    train_ids_path: Path | None = None,
    test_ids_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_manifest(manifest_path)
    rows = [manifest_item_to_tutor_row(item) for item in manifest]
    train_ids = load_split_ids(train_ids_path)
    test_ids = load_split_ids(test_ids_path)

    if train_ids is not None or test_ids is not None:
        train_rows = [
            row for row in rows if train_ids is None or row["id"] in train_ids
        ]
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
        return train_rows, test_rows

    split_at = max(1, len(rows) - max(1, test_size))
    return rows[:split_at], rows[split_at:]
