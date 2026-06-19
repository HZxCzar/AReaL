from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_BAD_QUESTION_IDS = frozenset(
    {
        "train-334",
        "train-504",
        "train-3460",
        "train-3694",
        "train-3997",
        "train-6780",
        "train-6816",
        "train-6843",
        "train-6922",
        "train-7225",
        "train-7321",
        "train-4253",
        "train-7417",
        "train-7493",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove known bad tutor dataset rows from a HuggingFace dataset saved "
            "with save_to_disk()."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input filtered dataset directory created by filter_pre_solved.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the pruned dataset.",
    )
    parser.add_argument(
        "--report",
        default="",
        help=(
            "Optional JSON report path. Defaults to "
            "'<output>_prune_bad_questions_report.json'."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help=(
            "Dataset splits to prune, or 'all'. "
            "Unselected splits are copied unchanged."
        ),
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Dataset column containing row ids.",
    )
    parser.add_argument(
        "--bad-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional bad id file. Supports a JSON list, a JSON object with "
            "'bad_ids' or 'ids', or a text file with one id per line."
        ),
    )
    parser.add_argument(
        "--bad-id",
        action="append",
        default=[],
        help="Additional bad id to remove. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-default-bad-ids",
        action="store_true",
        help="Use only ids from --bad-ids-file and --bad-id.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested bad id is not found in the selected split(s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output dataset directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and write the report without saving a new dataset.",
    )
    return parser.parse_args()


def load_bad_ids_file(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return set()

    if stripped[0] in "[{":
        data = json.loads(stripped)
        if isinstance(data, list):
            return {str(item) for item in data}
        if isinstance(data, dict):
            for key in ("bad_ids", "ids", "remove_ids"):
                value = data.get(key)
                if value is not None:
                    if not isinstance(value, list):
                        raise ValueError(f"{path}: JSON field '{key}' must be a list.")
                    return {str(item) for item in value}
        raise ValueError(
            f"{path}: expected a JSON list or object with 'bad_ids', "
            "'ids', or 'remove_ids'."
        )

    ids: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.add(line)
    return ids


def collect_bad_ids(args: argparse.Namespace) -> set[str]:
    bad_ids: set[str] = set()
    if not args.no_default_bad_ids:
        bad_ids.update(DEFAULT_BAD_QUESTION_IDS)
    if args.bad_ids_file is not None:
        bad_ids.update(load_bad_ids_file(args.bad_ids_file))
    bad_ids.update(str(item) for item in args.bad_id)
    if not bad_ids:
        raise ValueError("No bad ids were provided.")
    return bad_ids


def resolve_splits(requested: list[str], split_names: list[str]) -> set[str]:
    if "all" in requested:
        return set(split_names)
    missing = sorted(set(requested) - set(split_names))
    if missing:
        raise ValueError(
            f"Unknown split(s) {missing}; available splits are {split_names}"
        )
    return set(requested)


def row_id(row: dict[str, Any], id_field: str) -> str:
    if id_field not in row:
        raise KeyError(f"Dataset row does not contain id field '{id_field}'.")
    return str(row[id_field])


def prune_split(
    dataset: Any, *, split_name: str, id_field: str, bad_ids: set[str]
) -> tuple[Any, dict[str, Any]]:
    keep_indices: list[int] = []
    removed_ids: list[str] = []
    removed_indices: list[int] = []

    for index, row in enumerate(dataset):
        item_id = row_id(row, id_field)
        if item_id in bad_ids:
            removed_ids.append(item_id)
            removed_indices.append(index)
        else:
            keep_indices.append(index)

    return dataset.select(keep_indices), {
        "split": split_name,
        "input_rows": len(dataset),
        "output_rows": len(keep_indices),
        "removed": len(removed_ids),
        "removed_ids": removed_ids,
        "removed_indices": removed_indices,
    }


def main() -> None:
    args = parse_args()

    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to prune tutor datasets."
        ) from exc

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = (
        Path(args.report).resolve()
        if args.report
        else output_path.with_name(
            f"{output_path.name}_prune_bad_questions_report.json"
        )
    )
    bad_ids = collect_bad_ids(args)

    loaded = load_from_disk(str(input_path))
    is_dataset_dict = isinstance(loaded, DatasetDict)
    dataset = loaded if is_dataset_dict else DatasetDict({"train": loaded})
    selected_splits = resolve_splits(args.splits, list(dataset.keys()))

    pruned_splits: dict[str, Any] = {}
    split_reports: dict[str, Any] = {}
    found_ids: set[str] = set()

    for split_name, split_dataset in dataset.items():
        if split_name not in selected_splits:
            pruned_splits[split_name] = split_dataset
            split_reports[split_name] = {
                "input_rows": len(split_dataset),
                "output_rows": len(split_dataset),
                "removed": 0,
                "copied_unchanged": True,
            }
            continue

        pruned_split, split_report = prune_split(
            split_dataset,
            split_name=split_name,
            id_field=args.id_field,
            bad_ids=bad_ids,
        )
        pruned_splits[split_name] = pruned_split
        split_reports[split_name] = split_report
        found_ids.update(split_report["removed_ids"])

    missing_ids = sorted(bad_ids - found_ids)
    report = {
        "input": str(input_path),
        "output": None if args.dry_run else str(output_path),
        "id_field": args.id_field,
        "selected_splits": sorted(selected_splits),
        "bad_ids_requested": sorted(bad_ids),
        "bad_ids_removed": sorted(found_ids),
        "bad_ids_missing": missing_ids,
        "splits": split_reports,
        "dry_run": bool(args.dry_run),
    }

    if args.strict and missing_ids:
        raise ValueError(f"Bad ids not found in selected split(s): {missing_ids}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.dry_run:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. "
                    "Pass --overwrite to replace it."
                )
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dataset = (
            DatasetDict(pruned_splits) if is_dataset_dict else pruned_splits["train"]
        )
        output_dataset.save_to_disk(str(output_path))

    total_removed = sum(item.get("removed", 0) for item in split_reports.values())
    print(
        f"Removed {total_removed} row(s) from {input_path}; "
        f"report written to {report_path}."
    )
    if not args.dry_run:
        print(f"Saved pruned dataset to {output_path}.")


if __name__ == "__main__":
    main()
