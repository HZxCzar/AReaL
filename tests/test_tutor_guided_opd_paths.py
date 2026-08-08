#!/usr/bin/env python3
"""Resolve every file path each config references and report what is missing.

A fresh git worktree only contains tracked files, so gitignored data directories
and the venv have to be symlinked in by hand. Missing ones do not surface until
train.py is several seconds into startup, after the job has already been
scheduled. This checks all of them up front, for all four arms.
"""
from __future__ import annotations

import os
import sys

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import TutorConfig

BASE = "examples/tutor/configs/math/0805/v2"
STEM = "qwen8b-train-qwen1.7b-eval3-math-pre-aleak-generated-g8k4-leakt"
ARMS = [("baseline", ""), ("opd", "-opd"), ("guided3", "-guided3"),
        ("guided3-opd", "-guided3-opd")]

missing_any = False

print("\nworktree symlinks")
for path in (".env", ".venv", "examples/tutor/data", "examples/tutor/prompt_pools"):
    ok = os.path.exists(path)
    missing_any = missing_any or not ok
    target = os.readlink(path) if os.path.islink(path) else "(real dir)"
    print(f"  {'ok  ' if ok else 'MISS'}  {path:<32} {target}")

for label, suffix in ARMS:
    config, _ = load_expr_config([
        "--config", f"{BASE}/{STEM}{suffix}.yaml"
    ], TutorConfig)
    candidates = {
        "train_dataset.path": getattr(config.train_dataset, "path", ""),
        "valid_dataset.path": getattr(config.valid_dataset, "path", ""),
        "student_generalize.path": config.student_generalize.path,
        "actor.path": config.actor.path,
        "tokenizer_path": getattr(config, "tokenizer_path", ""),
        "prompt_pool.teacher_path": config.prompt_pool.teacher_path,
        "prompt_pool.student_seen_path": config.prompt_pool.student_seen_path,
        "prompt_pool.student_heldout_path": config.prompt_pool.student_heldout_path,
        "prompt_pool.teacher_warmup.prompt_path": (
            config.prompt_pool.teacher_warmup.prompt_path
        ),
        "prompt_pool.student_turn_behavior.path": (
            config.prompt_pool.student_turn_behavior.path
        ),
    }
    bad = [
        f"{key} = {value}"
        for key, value in candidates.items()
        if value and not os.path.exists(value)
    ]
    print(f"\n{label}")
    if bad:
        missing_any = True
        for entry in bad:
            print(f"  MISS  {entry}")
    else:
        checked = sum(1 for v in candidates.values() if v)
        print(f"  ok    all {checked} referenced paths exist")

print()
if missing_any:
    print("MISSING PATHS -- a run would die during startup")
    sys.exit(1)
print("every path referenced by all four arms resolves")
