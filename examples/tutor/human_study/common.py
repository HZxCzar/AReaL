"""Small, independent helpers for reproducible human-study generation."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    """Atomic replacement; never leave a half-written result after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config must be a YAML mapping")
    return value


def load_dataset(path):
    dataset = read_json(path)
    if dataset.get("schema_version") != 2:
        raise ValueError(
            "Expected rule-based dataset v2; regenerate old drafts with prepare_data"
        )
    content = {k: v for k, v in dataset.items() if k != "fingerprint"}
    if digest(content) != dataset.get("fingerprint"):
        raise ValueError("Dataset changed; regenerate it with prepare_data")
    if not dataset["cases"]:
        raise ValueError("No included cases")
    ids = [c["id"] for c in dataset["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    return dataset
