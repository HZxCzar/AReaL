from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def make_run_dir(root: str | Path | None, prefix: str) -> Path:
    base = ensure_dir(root or Path.cwd() / "artifacts" / prefix)
    run_dir = base / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
