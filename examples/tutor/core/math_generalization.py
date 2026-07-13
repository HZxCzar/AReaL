import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

MATH_GENERALIZATION_SAMPLE_COUNT = 2


@dataclass(frozen=True)
class MathGeneralizationResult:
    output_path: Path
    sidecar_path: Path
    manifest_path: Path
    sample_count: int
    main_size: int
    train_pool_size: int
    reused: bool = False


def default_math_generalization_output_path(
    train_dataset_path: str | Path,
    *,
    seed: int,
    sample_count: int,
) -> Path:
    source = Path(train_dataset_path)
    suffix = f"math_generalize_seed{seed}_samples{sample_count}"
    return source.with_name(f"{source.name}_{suffix}")


def prepare_math_generalization_sidecar(
    *,
    train_dataset_path: str | Path,
    valid_dataset_path: str | Path | None,
    output_path: str | Path,
    seed: int,
    sample_count: int = MATH_GENERALIZATION_SAMPLE_COUNT,
    overwrite: bool = False,
    sidecar_filename: str = "student_generalize.json",
    show_progress: bool = True,
) -> MathGeneralizationResult:
    train_source = Path(train_dataset_path)
    valid_source = Path(valid_dataset_path) if valid_dataset_path else None
    output = Path(output_path)
    sidecar = output / sidecar_filename
    manifest = output / "math_generalization_manifest.json"
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("Math generalization sample_count must be positive.")

    config_payload = {
        "train_dataset_path": str(train_source),
        "valid_dataset_path": str(valid_source) if valid_source else None,
        "seed": int(seed),
        "sample_count": sample_count,
        "sidecar_filename": sidecar_filename,
    }

    with tqdm(total=4, desc="正在处理数据", disable=not show_progress) as progress:
        reused = _maybe_reuse_existing(
            output=output,
            sidecar=sidecar,
            manifest=manifest,
            expected_config=config_payload,
            overwrite=overwrite,
        )
        if reused is not None:
            progress.update(4)
            return reused
        progress.update(1)

        train_rows = _load_split(train_source, "train")
        train_pool = _distinct_questions(train_rows)
        valid_rows = (
            _load_split(valid_source, "test") if valid_source is not None else []
        )
        main_rows = [*train_rows, *valid_rows]
        _validate_rows(train_pool, main_rows, sample_count=sample_count)
        progress.update(1)

        bank = _build_generalization_bank(
            main_rows=main_rows,
            train_rows=train_pool,
            seed=int(seed),
            sample_count=sample_count,
        )
        progress.update(1)

        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        sidecar.write_text(
            json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest.write_text(
            json.dumps(
                {
                    "config": config_payload,
                    "sizes": {
                        "main": len(main_rows),
                        "train_pool": len(train_pool),
                        "sidecar_entries": len(bank),
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

    return MathGeneralizationResult(
        output_path=output,
        sidecar_path=sidecar,
        manifest_path=manifest,
        sample_count=sample_count,
        main_size=len(main_rows),
        train_pool_size=len(train_pool),
    )


def _load_split(source: Path, split: str) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(source))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Dataset at {source} does not contain split {split!r}.")
        selected = dataset[split]
    elif isinstance(dataset, Dataset):
        selected = dataset
    else:
        raise ValueError(
            f"Unsupported math dataset object at {source}: {type(dataset)}"
        )
    return [dict(row) for row in selected]


def _distinct_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    distinct: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in rows:
        task = str(row.get("task", ""))
        if task in seen_tasks:
            continue
        seen_tasks.add(task)
        distinct.append(row)
    return distinct


def _validate_rows(
    train_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    *,
    sample_count: int,
) -> None:
    if len(train_rows) <= sample_count:
        raise ValueError(
            "Math generalization requires more train rows than sample_count; "
            f"got {len(train_rows)} rows and sample_count={sample_count}."
        )

    seen_ids: set[str] = set()
    for index, row in enumerate(main_rows):
        for field in ("id", "task", "ground_truth"):
            if row.get(field) in (None, ""):
                raise ValueError(
                    f"Math generalization row {index} is missing field {field!r}."
                )
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate math sample id across train/test splits: {sample_id}"
            )
        seen_ids.add(sample_id)


def _build_generalization_bank(
    *,
    main_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    seed: int,
    sample_count: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    train_indices_by_task = {
        str(row["task"]): index for index, row in enumerate(train_rows)
    }
    bank: dict[str, Any] = {}
    pool_size = len(train_rows)

    for row in main_rows:
        sample_id = str(row["id"])
        excluded_index = train_indices_by_task.get(str(row["task"]))
        selected_indices: list[int] = []
        seen_indices: set[int] = set()
        while len(selected_indices) < sample_count:
            candidate_index = rng.randrange(pool_size)
            if candidate_index == excluded_index or candidate_index in seen_indices:
                continue
            selected_indices.append(candidate_index)
            seen_indices.add(candidate_index)
        bank[sample_id] = {
            "samples": [
                _sample_payload(train_rows[index]) for index in selected_indices
            ]
        }
    return bank


def _sample_payload(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row["id"]),
        "task": str(row["task"]),
        "ground_truth": str(row["ground_truth"]),
    }


def _maybe_reuse_existing(
    *,
    output: Path,
    sidecar: Path,
    manifest: Path,
    expected_config: dict[str, Any],
    overwrite: bool,
) -> MathGeneralizationResult | None:
    if not output.exists() or overwrite:
        return None
    if not manifest.exists() or not sidecar.exists():
        raise ValueError(
            f"Math generalization output exists at {output}, but its manifest or "
            "sidecar is missing. Set student_generalize.overwrite=true to rebuild."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("config") != expected_config:
        raise ValueError(
            f"Math generalization output exists at {output} with different "
            "processing settings. Set student_generalize.overwrite=true or choose "
            "a different output_path."
        )
    sizes = payload.get("sizes") or {}
    return MathGeneralizationResult(
        output_path=output,
        sidecar_path=sidecar,
        manifest_path=manifest,
        sample_count=int(expected_config["sample_count"]),
        main_size=int(sizes.get("main", 0)),
        train_pool_size=int(sizes.get("train_pool", 0)),
        reused=True,
    )
