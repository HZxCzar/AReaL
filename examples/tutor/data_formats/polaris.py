from __future__ import annotations

from typing import Any

DEFAULT_POLARIS_DATASET = "POLARIS-Project/Polaris-Dataset-53K"


def polaris_item_to_tutor_row(
    item: dict[str, Any],
    *,
    fallback_id: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": "polaris"}
    if item.get("difficulty") is not None:
        metadata["difficulty"] = str(item["difficulty"])
    return {
        "id": str(item.get("id", fallback_id)),
        "task": str(item["problem"]),
        "ground_truth": str(item["answer"]),
        "metadata": metadata,
    }


def build_polaris_splits_from_rows(
    rows: list[dict[str, Any]],
    *,
    test_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Polaris dataset is empty.")
    resolved_test_size = max(1, int(test_size))
    if resolved_test_size >= len(rows):
        raise ValueError("Polaris test_size must be smaller than the dataset size.")
    split_at = len(rows) - resolved_test_size
    return rows[:split_at], rows[split_at:]


def build_polaris_splits(
    *,
    dataset_path: str = DEFAULT_POLARIS_DATASET,
    test_size: int = 1024,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to load the Polaris dataset."
        ) from exc

    dataset = load_dataset(dataset_path, split="train")
    rows = [
        polaris_item_to_tutor_row(dict(item), fallback_id=f"polaris-{idx}")
        for idx, item in enumerate(dataset)
    ]
    return build_polaris_splits_from_rows(rows, test_size=test_size)
