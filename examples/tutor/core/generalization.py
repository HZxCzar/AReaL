import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REQUIRED_STUDENT_GENERALIZE_LEVELS = ("level1", "level2")
REQUIRED_STUDENT_GENERALIZE_FIELDS = ("task", "ground_truth")
GENERATED_VARIANT_TO_LEVEL = {
    "variant1": "level1",
    "variant2": "level2",
}


def load_student_generalize_bank(
    path: str,
    *,
    source: str = "sidecar",
) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"student_generalize sidecar must be a JSON object: {file_path}"
        )
    if source == "generated":
        return _load_generated_variant_bank(payload, file_path=file_path)
    return payload


def _load_generated_variant_bank(
    payload: Mapping[str, Any],
    *,
    file_path: Path,
) -> dict[str, Any]:
    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise ValueError(
            "student_generalize.source='generated' expects a JSON object with a "
            f"'triples' list: {file_path}"
        )

    bank: dict[str, Any] = {}
    for index, triple in enumerate(triples):
        if not isinstance(triple, Mapping):
            raise ValueError(
                f"Generated triple at index {index} must be a JSON object: {file_path}"
            )
        source_id = triple.get("source_id")
        if source_id in (None, ""):
            raise ValueError(
                f"Generated triple at index {index} is missing source_id: {file_path}"
            )
        sample_id = str(source_id)
        if sample_id in bank:
            raise ValueError(
                f"Duplicate generated source_id {sample_id!r}: {file_path}"
            )

        cases: dict[str, dict[str, str]] = {}
        for variant_name, level in GENERATED_VARIANT_TO_LEVEL.items():
            variant = triple.get(variant_name)
            if not isinstance(variant, Mapping):
                raise ValueError(
                    f"Generated triple {sample_id!r} is missing {variant_name}: "
                    f"{file_path}"
                )
            task = variant.get("task")
            ground_truth = variant.get("ground_truth")
            if task in (None, "") or ground_truth in (None, ""):
                raise ValueError(
                    f"Generated triple {sample_id!r} has incomplete {variant_name}; "
                    f"task and ground_truth are required: {file_path}"
                )
            cases[level] = {
                "task": str(task),
                "ground_truth": str(ground_truth),
            }
        bank[sample_id] = cases
    return bank


def has_complete_generalize_case(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    sample_count: int | None = None,
    required_levels: Iterable[str] = REQUIRED_STUDENT_GENERALIZE_LEVELS,
) -> bool:
    """Can this row serve every transfer level that is switched on?

    required_levels is the caller's set of enabled transfer levels. Empty means
    no probe reads the bank -- the original re-test is built from the row's own
    task -- so every row qualifies.

    The case may come from the sidecar keyed by id or from
    metadata.student_generalize; both forms are accepted, and for the
    positional 'samples' form a level maps to its index in
    REQUIRED_STUDENT_GENERALIZE_LEVELS.
    """
    required_levels = tuple(required_levels)
    if not required_levels:
        return True
    if not isinstance(sample, Mapping):
        return False
    return not _missing_generalize_fields(
        sample,
        bank,
        sample_count=sample_count,
        required_levels=required_levels,
    )


def _missing_generalize_fields(
    sample: Mapping[str, Any],
    bank: Mapping[str, Any],
    *,
    sample_count: int | None = None,
    required_levels: tuple[str, ...] = REQUIRED_STUDENT_GENERALIZE_LEVELS,
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

    samples = payload.get("samples")
    if isinstance(samples, list):
        missing: list[str] = []
        if required_levels == REQUIRED_STUDENT_GENERALIZE_LEVELS:
            # Unchanged path. Every level is in use, so the sidecar must hold
            # exactly the expected number of cases and all of them complete.
            expected_count = sample_count if sample_count is not None else len(samples)
            if len(samples) != expected_count:
                missing.append(
                    f"samples(count={len(samples)}, expected={expected_count})"
                )
            wanted: Iterable[int] = range(len(samples))
        else:
            # A subset of levels reads a subset of positions -- the case builder
            # zips levels against this list positionally -- so only those
            # positions have to be there and the total stops mattering.
            wanted = [
                REQUIRED_STUDENT_GENERALIZE_LEVELS.index(level)
                for level in required_levels
            ]
        for index in wanted:
            item = samples[index] if index < len(samples) else None
            if not isinstance(item, Mapping):
                missing.append(f"samples[{index}]")
                continue
            for field in REQUIRED_STUDENT_GENERALIZE_FIELDS:
                if item.get(field) in (None, ""):
                    missing.append(f"samples[{index}].{field}")
        return missing

    missing: list[str] = []
    for level in required_levels:
        item = payload.get(level)
        if not isinstance(item, Mapping):
            missing.append(level)
            continue
        for field in REQUIRED_STUDENT_GENERALIZE_FIELDS:
            if item.get(field) in (None, ""):
                missing.append(f"{level}.{field}")
    return missing
