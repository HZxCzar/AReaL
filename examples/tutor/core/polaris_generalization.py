import json
import random
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm

POLARIS_GENERALIZATION_SPLITS = (
    "train",
    "test",
    "train_generalize_pool",
    "test_generalize_pool",
)


@dataclass(frozen=True)
class PolarisGeneralizationResult:
    dataset_path: Path
    sidecar_path: Path
    manifest_path: Path
    train_size: int
    test_size: int
    train_generalize_pool_size: int
    test_generalize_pool_size: int
    reused: bool = False


def default_polaris_generalization_output_path(
    source_path: str | Path,
    *,
    seed: int,
    train_ratio: float,
    generalize_ratio: float,
) -> Path:
    source = Path(source_path)
    suffix = (
        f"polaris_generalize_seed{seed}_"
        f"train{_ratio_label(train_ratio)}_gen{_ratio_label(generalize_ratio)}"
    )
    return source.with_name(f"{source.name}_{suffix}")


def prepare_polaris_generalization_dataset(
    *,
    source_path: str | Path,
    output_path: str | Path,
    seed: int,
    train_ratio: float,
    generalize_ratio: float,
    reuse_generalize_tasks: bool,
    overwrite: bool,
    sidecar_filename: str = "student_generalize.json",
    show_progress: bool = True,
) -> PolarisGeneralizationResult:
    source = Path(source_path)
    output = Path(output_path)
    sidecar = output / sidecar_filename
    manifest = output / "polaris_processing_manifest.json"
    config_payload = {
        "source_path": str(source),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "generalize_ratio": float(generalize_ratio),
        "reuse_generalize_tasks": bool(reuse_generalize_tasks),
        "sidecar_filename": sidecar_filename,
        "difficulty_rule": {
            "level1_preferred": ["d-2", "d-1"],
            "level1_fallback": ["d"],
            "level2_preferred": ["d+1", "d+2"],
            "level2_fallback": ["d"],
        },
    }

    with tqdm(
        total=5,
        desc="正在处理数据",
        disable=not show_progress,
    ) as progress:
        reused = _maybe_reuse_existing(
            output=output,
            sidecar=sidecar,
            manifest=manifest,
            expected_config=config_payload,
            overwrite=overwrite,
        )
        if reused is not None:
            progress.update(5)
            return reused
        progress.update(1)

        dataset = _load_source_as_single_dataset(source)
        rows = [dict(row) for row in dataset]
        if len(rows) < 4:
            raise ValueError(
                "Polaris generalization processing requires at least 4 source rows."
            )
        _validate_rows(rows)
        progress.update(1)

        train_rows, test_rows = _stratified_split(
            rows,
            keep_ratio=float(train_ratio),
            seed=int(seed),
            label="train/test",
        )
        train_main, train_pool = _stratified_split(
            train_rows,
            keep_ratio=1.0 - float(generalize_ratio),
            seed=int(seed) + 1,
            label="train/generalize",
        )
        test_main, test_pool = _stratified_split(
            test_rows,
            keep_ratio=1.0 - float(generalize_ratio),
            seed=int(seed) + 2,
            label="test/generalize",
        )
        progress.update(1)

        bank: dict[str, Any] = {}
        bank.update(
            _build_generalization_bank(
                main_rows=train_main,
                pool_rows=train_pool,
                seed=int(seed) + 3,
                reuse_generalize_tasks=bool(reuse_generalize_tasks),
                split_name="train",
            )
        )
        bank.update(
            _build_generalization_bank(
                main_rows=test_main,
                pool_rows=test_pool,
                seed=int(seed) + 4,
                reuse_generalize_tasks=bool(reuse_generalize_tasks),
                split_name="test",
            )
        )
        progress.update(1)

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            shutil.rmtree(output)
        derived = DatasetDict(
            {
                "train": Dataset.from_list(train_main),
                "test": Dataset.from_list(test_main),
                "train_generalize_pool": Dataset.from_list(train_pool),
                "test_generalize_pool": Dataset.from_list(test_pool),
            }
        )
        derived.save_to_disk(str(output))
        sidecar.write_text(
            json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "config": config_payload,
                    "sizes": {
                        "train": len(train_main),
                        "test": len(test_main),
                        "train_generalize_pool": len(train_pool),
                        "test_generalize_pool": len(test_pool),
                        "sidecar_entries": len(bank),
                    },
                    "difficulty_counts": {
                        "train": _difficulty_counts(train_main),
                        "test": _difficulty_counts(test_main),
                        "train_generalize_pool": _difficulty_counts(train_pool),
                        "test_generalize_pool": _difficulty_counts(test_pool),
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        progress.update(1)

    return PolarisGeneralizationResult(
        dataset_path=output,
        sidecar_path=sidecar,
        manifest_path=manifest,
        train_size=len(train_main),
        test_size=len(test_main),
        train_generalize_pool_size=len(train_pool),
        test_generalize_pool_size=len(test_pool),
    )


def _ratio_label(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _maybe_reuse_existing(
    *,
    output: Path,
    sidecar: Path,
    manifest: Path,
    expected_config: dict[str, Any],
    overwrite: bool,
) -> PolarisGeneralizationResult | None:
    if not output.exists():
        return None
    if overwrite:
        return None
    if not manifest.exists() or not sidecar.exists():
        raise ValueError(
            f"Derived Polaris dataset already exists at {output}, but manifest or "
            "sidecar is missing. Set polaris_processing.overwrite=true to rebuild."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("config") != expected_config:
        raise ValueError(
            f"Derived Polaris dataset already exists at {output} with different "
            "processing settings. Set polaris_processing.overwrite=true or choose "
            "a different output_path."
        )
    sizes = payload.get("sizes") or {}
    return PolarisGeneralizationResult(
        dataset_path=output,
        sidecar_path=sidecar,
        manifest_path=manifest,
        train_size=int(sizes.get("train", 0)),
        test_size=int(sizes.get("test", 0)),
        train_generalize_pool_size=int(sizes.get("train_generalize_pool", 0)),
        test_generalize_pool_size=int(sizes.get("test_generalize_pool", 0)),
        reused=True,
    )


def _load_source_as_single_dataset(source: Path) -> Dataset:
    dataset = load_from_disk(str(source))
    if isinstance(dataset, DatasetDict):
        splits = [dataset[name] for name in dataset.keys()]
        if not splits:
            raise ValueError(f"Empty Polaris DatasetDict at {source}")
        return concatenate_datasets(splits)
    if isinstance(dataset, Dataset):
        return dataset
    raise ValueError(f"Unsupported Polaris dataset object at {source}: {type(dataset)}")


def _validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        for key in ("id", "task", "ground_truth", "metadata"):
            if key not in row:
                raise ValueError(
                    f"Polaris row {index} is missing required field {key!r}."
                )
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(
                f"Duplicate Polaris sample id after source merge: {sample_id}"
            )
        seen.add(sample_id)
        _difficulty(row)


def _stratified_split(
    rows: Sequence[dict[str, Any]],
    *,
    keep_ratio: float,
    seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError(f"Cannot split empty Polaris rows for {label}.")
    if not 0.0 < keep_ratio < 1.0:
        raise ValueError(f"{label} keep_ratio must be in (0, 1), got {keep_ratio}.")

    rng = random.Random(seed)
    keep: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for _, bucket in _rows_by_difficulty(rows).items():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        keep_count = _bounded_ratio_count(len(shuffled), keep_ratio)
        keep.extend(shuffled[:keep_count])
        heldout.extend(shuffled[keep_count:])

    rng.shuffle(keep)
    rng.shuffle(heldout)
    if not keep or not heldout:
        raise ValueError(
            f"Polaris stratified split {label} produced an empty side: "
            f"keep={len(keep)}, heldout={len(heldout)}."
        )
    return keep, heldout


def _bounded_ratio_count(size: int, ratio: float) -> int:
    count = round(size * ratio)
    if size <= 1:
        return size
    return min(size - 1, max(1, count))


def _rows_by_difficulty(
    rows: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_difficulty: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_difficulty[_difficulty(row)].append(row)
    return dict(sorted(by_difficulty.items()))


def _build_generalization_bank(
    *,
    main_rows: Sequence[dict[str, Any]],
    pool_rows: Sequence[dict[str, Any]],
    seed: int,
    reuse_generalize_tasks: bool,
    split_name: str,
) -> dict[str, Any]:
    if not main_rows:
        return {}
    if not pool_rows:
        raise ValueError(f"{split_name} generalization pool is empty.")

    rng = random.Random(seed)
    by_difficulty = _rows_by_difficulty(pool_rows)
    min_difficulty = min(by_difficulty)
    max_difficulty = max(by_difficulty)
    used: set[str] = set()
    bank: dict[str, Any] = {}

    for row in main_rows:
        sample_id = str(row["id"])
        difficulty = _difficulty(row)
        level1 = _pick_probe(
            by_difficulty=by_difficulty,
            rng=rng,
            difficulty=difficulty,
            preferred_difficulties=[
                value
                for value in (difficulty - 2, difficulty - 1)
                if min_difficulty <= value <= max_difficulty
            ],
            fallback_difficulties=[difficulty],
            exclude_ids={sample_id},
            used_ids=used,
            reuse_generalize_tasks=reuse_generalize_tasks,
            level="level1",
            split_name=split_name,
        )
        level2 = _pick_probe(
            by_difficulty=by_difficulty,
            rng=rng,
            difficulty=difficulty,
            preferred_difficulties=[
                value
                for value in (difficulty + 1, difficulty + 2)
                if min_difficulty <= value <= max_difficulty
            ],
            fallback_difficulties=[difficulty],
            exclude_ids={sample_id, str(level1["id"])},
            used_ids=used,
            reuse_generalize_tasks=reuse_generalize_tasks,
            level="level2",
            split_name=split_name,
        )
        bank[sample_id] = {
            "level1": _probe_payload(level1),
            "level2": _probe_payload(level2),
        }
        if not reuse_generalize_tasks:
            used.add(str(level1["id"]))
            used.add(str(level2["id"]))

    return bank


def _pick_probe(
    *,
    by_difficulty: dict[int, list[dict[str, Any]]],
    rng: random.Random,
    difficulty: int,
    preferred_difficulties: Sequence[int],
    fallback_difficulties: Sequence[int],
    exclude_ids: set[str],
    used_ids: set[str],
    reuse_generalize_tasks: bool,
    level: str,
    split_name: str,
) -> dict[str, Any]:
    candidates = _candidate_rows(
        by_difficulty=by_difficulty,
        difficulties=preferred_difficulties,
        exclude_ids=exclude_ids,
        used_ids=used_ids,
        reuse_generalize_tasks=reuse_generalize_tasks,
    )
    if not candidates:
        candidates = _candidate_rows(
            by_difficulty=by_difficulty,
            difficulties=fallback_difficulties,
            exclude_ids=exclude_ids,
            used_ids=used_ids,
            reuse_generalize_tasks=reuse_generalize_tasks,
        )
    if not candidates:
        raise ValueError(
            f"Could not match {level} Polaris generalization probe for "
            f"{split_name} sample with difficulty {difficulty}."
        )
    return rng.choice(candidates)


def _candidate_rows(
    *,
    by_difficulty: dict[int, list[dict[str, Any]]],
    difficulties: Sequence[int],
    exclude_ids: set[str],
    used_ids: set[str],
    reuse_generalize_tasks: bool,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for difficulty in difficulties:
        for row in by_difficulty.get(difficulty, []):
            sample_id = str(row["id"])
            if sample_id in exclude_ids:
                continue
            if not reuse_generalize_tasks and sample_id in used_ids:
                continue
            candidates.append(row)
    return candidates


def _probe_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "task": str(row["task"]),
        "ground_truth": str(row["ground_truth"]),
        "difficulty": _difficulty(row),
    }


def _difficulty(row: dict[str, Any]) -> int:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Polaris row {row.get('id')!r} has invalid metadata.")
    raw = metadata.get("difficulty")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        head = raw.split("/", 1)[0].strip()
        if head:
            try:
                return int(head)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Polaris difficulty {raw!r} for row {row.get('id')!r}."
                ) from exc
    raise ValueError(f"Missing Polaris difficulty for row {row.get('id')!r}.")


def _difficulty_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(_difficulty(row))] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))
