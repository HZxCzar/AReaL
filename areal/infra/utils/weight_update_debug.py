# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from typing import Any

from areal.api import WeightUpdateMeta


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _truncate(value: Any, limit: int = 4096) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...<truncated {len(value) - limit} chars>"
    if isinstance(value, Mapping):
        return {str(k): _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, limit) for v in value]
    return value


def meta_summary(meta: WeightUpdateMeta | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    fields = (
        "type",
        "path",
        "use_lora",
        "lora_name",
        "lora_int_id",
        "base_model_name",
        "clear_checkpoint_after_load",
        "version",
        "weight_chunked_mem_mb",
    )
    return {name: getattr(meta, name, None) for name in fields}


def _cuda_summary() -> dict[str, Any] | None:
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return {"torch_loaded": False}
        if not torch.cuda.is_available():
            return {"torch_loaded": True, "available": False}
        if not torch.cuda.is_initialized():
            return {
                "torch_loaded": True,
                "available": True,
                "initialized": False,
            }
        device = torch.cuda.current_device()
        return {
            "torch_loaded": True,
            "available": True,
            "initialized": True,
            "device_count": torch.cuda.device_count(),
            "current_device": device,
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
        }
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def debug_log_path(
    meta: WeightUpdateMeta | None = None,
    experiment_name: str | None = None,
    trial_name: str | None = None,
) -> str:
    """Return the shared JSONL debug log path for weight-update diagnostics."""
    env_path = os.environ.get("AREAL_WEIGHT_UPDATE_DEBUG_LOG")
    if env_path:
        return env_path

    if meta is not None and meta.path:
        return os.path.join(os.path.dirname(meta.path), "weight_update_debug.jsonl")

    if experiment_name and trial_name:
        return os.path.join(
            "/tmp",
            "areal_weight_update_debug_"
            f"{_safe_name(experiment_name)}_{_safe_name(trial_name)}.jsonl",
        )

    return "/tmp/areal_weight_update_debug.jsonl"


def log_weight_update_debug(
    event: str,
    *,
    meta: WeightUpdateMeta | None = None,
    experiment_name: str | None = None,
    trial_name: str | None = None,
    **fields: Any,
) -> str | None:
    """Append one JSONL event for debugging disk/LoRA weight updates.

    This must never affect training. All exceptions are swallowed deliberately.
    """
    try:
        path = debug_log_path(meta, experiment_name, trial_name)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        record = {
            "time": time.time(),
            "time_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "event": event,
            "log_path": path,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
            "hostname": socket.gethostname(),
            "thread": threading.current_thread().name,
            "argv": sys.argv,
            "env": {
                name: os.environ.get(name)
                for name in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "NCCL_DEBUG",
                    "TORCH_DISTRIBUTED_DEBUG",
                )
                if os.environ.get(name) is not None
            },
            "cuda": _cuda_summary(),
            "meta": meta_summary(meta),
        }
        record.update(_truncate(fields))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return path
    except Exception:
        return None


def exception_fields(exc: BaseException) -> dict[str, Any]:
    return {
        "exception_type": exc.__class__.__name__,
        "exception": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }
