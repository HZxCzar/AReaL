import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REQUIRED_STUDENT_GENERALIZE_LEVELS = ("level1", "level2")
REQUIRED_STUDENT_GENERALIZE_FIELDS = ("task", "ground_truth")


def load_student_generalize_bank(path: str) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"student_generalize sidecar must be a JSON object: {file_path}"
        )
    return payload


def validate_student_generalize_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    split_name: str,
    bank: Mapping[str, Any],
) -> None:
    missing: list[str] = []
    total = 0
    for index, sample in enumerate(dataset):
        total += 1
        sample_missing = _missing_generalize_fields(sample, bank)
        if sample_missing:
            sample_id = sample.get("id")
            label = str(sample_id) if sample_id is not None else f"index {index}"
            missing.append(f"{label}: {', '.join(sample_missing)}")

    if not missing:
        return

    preview = "; ".join(missing[:10])
    if len(missing) > 10:
        preview += f"; ... and {len(missing) - 10} more"
    raise ValueError(
        "student_generalize.enabled=true but "
        f"{split_name} split has {len(missing)}/{total} samples without complete "
        "student generalization cases. Each sample must provide level1 and level2 "
        "with task and ground_truth, either in the sidecar keyed by id or in "
        f"metadata.student_generalize. Missing: {preview}"
    )


def _missing_generalize_fields(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
) -> list[str]:
    payload: Any | None = None
    sample_id = sample.get("id")
    if sample_id is not None and str(sample_id) in bank:
        payload = bank[str(sample_id)]
    else:
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping):
            payload = metadata.get("student_generalize")

    if not isinstance(payload, Mapping):
        return ["student_generalize"]

    missing: list[str] = []
    for level in REQUIRED_STUDENT_GENERALIZE_LEVELS:
        item = payload.get(level)
        if not isinstance(item, Mapping):
            missing.append(level)
            continue
        for field in REQUIRED_STUDENT_GENERALIZE_FIELDS:
            if item.get(field) in (None, ""):
                missing.append(f"{level}.{field}")
    return missing
