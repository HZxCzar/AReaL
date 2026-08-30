#!/usr/bin/env python3
"""Protect committed AReaL recover generations while training is running.

The watcher polls each trial's atomic ``recover/current.json`` pointer. Once a
complete generation becomes current, it hard-links that immutable generation
into ``recover_history`` outside AReaL's built-in cleanup directory. This makes
the watcher safe to start against already-running jobs and avoids copying the
large distributed checkpoint shards.
"""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("recover-history-watcher")
ARCHIVE_DIR_NAME = "recover_history"
EVENTS_FILE_NAME = "events.jsonl"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _read_current_generation(pointer_path: Path) -> str:
    with pointer_path.open(encoding="utf-8") as source:
        generation = json.load(source)["generation"]
    if (
        not isinstance(generation, str)
        or not generation
        or generation in {".", ".."}
        or Path(generation).name != generation
    ):
        raise ValueError(f"invalid recovery generation: {generation!r}")
    return generation


def _load_step_info(generation_path: Path) -> dict[str, Any]:
    step_info_path = generation_path / "recover_info" / "step_info.json"
    checkpoint_path = generation_path / "checkpoints"
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"missing checkpoint directory: {checkpoint_path}")
    with step_info_path.open(encoding="utf-8") as source:
        step_info = json.load(source)
    int(step_info["global_step"])
    return step_info


def _generation_timestamp_ns(generation: str) -> int:
    try:
        return int(generation.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _append_event(archive_root: Path, payload: dict[str, Any]) -> None:
    payload = {
        "recorded_at": time.time(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    with (archive_root / EVENTS_FILE_NAME).open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


def _archive_generation(
    trial_root: Path,
    generation: str,
    source_path: Path,
    step_info: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    archive_root = trial_root / ARCHIVE_DIR_NAME
    destination = archive_root / generation
    if destination.is_dir():
        return False
    if destination.exists():
        raise RuntimeError(
            f"archive target exists but is not a directory: {destination}"
        )

    if dry_run:
        LOGGER.info(
            "[dry-run] would archive trial=%s generation=%s global_step=%s",
            trial_root.name,
            generation,
            step_info["global_step"],
        )
        return False

    archive_root.mkdir(parents=True, exist_ok=True)
    if source_path.stat().st_dev != archive_root.stat().st_dev:
        raise RuntimeError(
            "hard-link archive must be on the same filesystem as the recovery "
            f"generation: {source_path} -> {archive_root}"
        )

    temporary = archive_root / f".tmp-{generation}-{os.getpid()}-{time.time_ns()}"
    try:
        shutil.copytree(
            source_path,
            temporary,
            copy_function=os.link,
            symlinks=True,
        )
        manifest = {
            "generation": generation,
            "global_step": int(step_info["global_step"]),
            "source": str(source_path),
            "trial_name": trial_root.name,
        }
        with (temporary / "archive_manifest.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    _append_event(
        archive_root,
        {
            "action": "archive",
            "generation": generation,
            "global_step": int(step_info["global_step"]),
        },
    )
    LOGGER.info(
        "Archived trial=%s generation=%s global_step=%s",
        trial_root.name,
        generation,
        step_info["global_step"],
    )
    return True


def _archive_sort_key(path: Path) -> tuple[int, int, str]:
    timestamp_ns = _generation_timestamp_ns(path.name)
    try:
        step_info = _load_step_info(path)
        global_step = int(step_info["global_step"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        global_step = -1
    if timestamp_ns == 0:
        timestamp_ns = path.stat().st_mtime_ns
    return timestamp_ns, global_step, path.name


def _prune_archives(
    trial_root: Path,
    current_generation: str,
    keep_last: int,
    *,
    dry_run: bool,
) -> None:
    archive_root = trial_root / ARCHIVE_DIR_NAME
    if not archive_root.is_dir():
        return
    generations = sorted(
        (
            path
            for path in archive_root.iterdir()
            if path.is_dir() and not path.name.startswith(".tmp-")
        ),
        key=_archive_sort_key,
        reverse=True,
    )
    current = next(
        (path for path in generations if path.name == current_generation), None
    )
    retained: list[Path] = []
    if current is not None:
        retained.append(current)
    for path in generations:
        if path == current or len(retained) >= keep_last:
            continue
        retained.append(path)
    retained_names = {path.name for path in retained}

    for path in generations:
        if path.name in retained_names:
            continue
        if dry_run:
            LOGGER.info("[dry-run] would prune archived generation %s", path)
            continue
        shutil.rmtree(path)
        _append_event(
            archive_root,
            {"action": "prune", "generation": path.name},
        )
        LOGGER.info("Pruned archived generation %s", path)


def _matches_trial(trial_name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(trial_name, pattern) for pattern in patterns)


def _scan_once(
    checkpoint_root: Path,
    trial_patterns: list[str],
    keep_last: int,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    matched = 0
    archived = 0
    for pointer_path in sorted(checkpoint_root.glob("*/recover/current.json")):
        trial_root = pointer_path.parent.parent
        if not _matches_trial(trial_root.name, trial_patterns):
            continue
        matched += 1
        try:
            generation = _read_current_generation(pointer_path)
            source_path = trial_root / "recover" / "generations" / generation
            step_info = _load_step_info(source_path)
            archived += int(
                _archive_generation(
                    trial_root,
                    generation,
                    source_path,
                    step_info,
                    dry_run=dry_run,
                )
            )
            _prune_archives(
                trial_root,
                generation,
                keep_last,
                dry_run=dry_run,
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not archive trial=%s: %s", trial_root.name, exc)
        except Exception:
            LOGGER.exception("Unexpected archive failure for trial=%s", trial_root.name)
    return matched, archived


def _acquire_lock(checkpoint_root: Path):
    lock_path = checkpoint_root / ".recover_history_watcher.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(
            f"another recover history watcher already holds {lock_path}"
        ) from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"pid={os.getpid()} started_at={datetime.now(UTC).isoformat()}\n")
    lock_file.flush()
    return lock_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint_root",
        type=Path,
        help="Experiment checkpoint directory whose immediate children are trials.",
    )
    parser.add_argument(
        "--keep-last",
        type=_positive_int,
        default=3,
        help="Archived generations to retain per trial (default: 3).",
    )
    parser.add_argument(
        "--poll-secs",
        type=_positive_float,
        default=30.0,
        help="Seconds between scans (default: 30).",
    )
    parser.add_argument(
        "--trial-glob",
        action="append",
        default=[],
        help="Trial-name glob to include; repeat for multiple patterns (default: '*').",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional file that receives the same logs printed to stdout.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan once and exit instead of watching continuously.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report archive and prune actions without changing recovery files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_file)
    checkpoint_root = args.checkpoint_root.resolve()
    if not checkpoint_root.is_dir():
        LOGGER.error("Checkpoint root is not a directory: %s", checkpoint_root)
        return 2
    trial_patterns = args.trial_glob or ["*"]

    try:
        lock_file = _acquire_lock(checkpoint_root)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 2

    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    LOGGER.info(
        "Watching checkpoint_root=%s keep_last=%d poll_secs=%.1f trial_globs=%s",
        checkpoint_root,
        args.keep_last,
        args.poll_secs,
        trial_patterns,
    )
    try:
        while True:
            matched, archived = _scan_once(
                checkpoint_root,
                trial_patterns,
                args.keep_last,
                dry_run=args.dry_run,
            )
            if matched == 0:
                LOGGER.warning("No matching recover/current.json files were found.")
            elif archived:
                LOGGER.info(
                    "Scan complete: matched=%d newly_archived=%d", matched, archived
                )
            if args.once or stop_event.wait(args.poll_secs):
                break
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    LOGGER.info("Recover history watcher stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
