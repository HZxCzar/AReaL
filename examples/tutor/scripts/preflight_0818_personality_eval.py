#!/usr/bin/env python3
"""Read-only semantic and dataset preflight for the 0818 personality matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from examples.tutor import train as tutor_train
from examples.tutor.scripts.evaluate_api_teacher import (
    build_eval_workflow_kwargs,
    dataset_sha256,
    effective_eval_presolve_enabled,
    load_experiment_config,
    prepare_test_dataset,
    resolve_teacher_generation_args,
)
from examples.tutor.workflow import (
    load_personality_complaints,
    load_personality_prompts,
)

from areal.utils.hf_utils import load_hf_tokenizer

PERSONALITIES = (
    ("none", "id", "qwen3-1.7b-text-original"),
    ("executive", "id", "qwen3-1.7b-text-original-executive"),
    ("surface", "id", "qwen3-1.7b-text-original-surface"),
    (
        "contrasting_cases",
        "id",
        "qwen3-1.7b-text-original-contrasting_cases",
    ),
    ("analogy", "ood", "qwen3-1.7b-text-original-analogy"),
    ("error_focused", "ood", "qwen3-1.7b-text-original-error_focused"),
    ("example_first", "ood", "qwen3-1.7b-text-original-example_first"),
    ("rule_first", "ood", "qwen3-1.7b-text-original-rule_first"),
)
TEACHERS = ("none", "surface", "executive", "contrasting_cases", "full")


def _eval_args() -> Namespace:
    return Namespace(
        teacher_temperature=None,
        teacher_top_p=None,
        teacher_max_tokens=None,
        presolve_attempts=0,
        presolve_max_tokens=None,
    )


def _parse_adapter(raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--adapter must be LABEL=/checkpoint/path")
    return label, Path(path).expanduser().resolve()


def _validate_adapters(
    adapters: dict[str, Path], teacher_model_path: Path
) -> dict[str, int]:
    completed_steps: dict[str, int] = {}
    for label, adapter in adapters.items():
        model_file = adapter / "adapter_model.safetensors"
        config_file = adapter / "adapter_config.json"
        if not model_file.is_file() or not config_file.is_file():
            raise ValueError(f"{label}: incomplete LoRA checkpoint: {adapter}")
        manifest = json.loads(config_file.read_text(encoding="utf-8"))
        if manifest.get("peft_type") != "LORA" or int(manifest.get("r", 0)) != 16:
            raise ValueError(f"{label}: expected a rank-16 LoRA checkpoint")
        configured_base = Path(str(manifest.get("base_model_name_or_path", "")))
        if configured_base.resolve() != teacher_model_path.resolve():
            raise ValueError(
                f"{label}: adapter base model differs from {teacher_model_path}"
            )
        marker = adapter.name.rsplit("globalstep", 1)
        if len(marker) != 2 or not marker[1].isdigit():
            raise ValueError(f"{label}: checkpoint directory has no globalstep: {adapter}")
        completed_steps[label] = int(marker[1]) + 1
    return completed_steps


def run_preflight(args: argparse.Namespace) -> dict[str, object]:
    config, students = load_experiment_config(str(args.config), [])
    tutor_train._apply_eval_average_rollouts(config)
    expected_names = [name for _, _, name in PERSONALITIES]
    actual_names = [str(student["name"]) for student in students]
    if actual_names != expected_names:
        raise ValueError(
            f"personality pool drifted: actual={actual_names}, expected={expected_names}"
        )

    selected_names = args.personality or [row[0] for row in PERSONALITIES]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError(f"duplicate selected personalities: {selected_names}")
    personality_rows = {row[0]: row for row in PERSONALITIES}
    selected_rows = [personality_rows[name] for name in selected_names]
    students_by_name = {str(student["name"]): student for student in students}
    selected_students = [students_by_name[row[2]] for row in selected_rows]

    prompts = load_personality_prompts(config.personality.prompts_path)
    bare, explained = load_personality_complaints(
        config.personality.complaints_path
    )
    demanding = [personality for personality, _, _ in PERSONALITIES if personality != "none"]
    missing_prompts = [name for name in demanding if name not in prompts]
    missing_complaints = [name for name in demanding if name not in explained]
    if missing_prompts or missing_complaints or not bare:
        raise ValueError(
            "incomplete personality prompt/complaint bank: "
            f"prompts={missing_prompts}, complaints={missing_complaints}, bare={len(bare)}"
        )

    eval_args = _eval_args()
    resolve_teacher_generation_args(eval_args, config)
    for student in selected_students:
        effective = build_eval_workflow_kwargs(
            config=config,
            student_models=[student],
            tokenizer=object(),
            args=eval_args,
            presolve_enabled=effective_eval_presolve_enabled(config),
            external_self_aux={
                "base_url": "http://127.0.0.1:1/v1",
                "model": "qwen3-8b",
                "api_key": "EMPTY",
                "request_params": {},
            },
        )
        checks = {
            "presolve": effective["teacher_pre_enabled"] is True,
            "no_verify": effective["teacher_pre_verify"] is False,
            "leak_continues": effective["leak_handling_mode"] == "reward_only",
            "format_continues": effective["format_handling_mode"] == "continue",
            "original_retest": effective["student_generalize_retest_original"] is True,
            "preleak_retest": effective["eval_preleak_retest"] is True,
            "replays": int(effective["student_generalize_replays"]) == args.replays,
            "personality_forwarded": bool(effective["personality"]),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{student['name']}: semantic preflight failed: {failed}")

    if config.student_generalize.enabled:
        tutor_train._prepare_math_generalization_data(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    student_prompts = tutor_train._load_eval_student_prompts(config)
    if student_prompts:
        raise ValueError("unexpected forced student prompt rows in personality eval")
    saved_max_samples = config.evaluator.max_samples
    config.evaluator.max_samples = None
    full_dataset = prepare_test_dataset(
        config,
        [selected_students[0]],
        tokenizer=tokenizer,
        limit=0,
        student_prompts=student_prompts,
    )
    config.evaluator.max_samples = args.max_samples or saved_max_samples
    runtime_dataset = prepare_test_dataset(
        config,
        [selected_students[0]],
        tokenizer=tokenizer,
        limit=0,
        student_prompts=student_prompts,
    )
    expected_runtime_rows = (
        min(args.max_samples, len(full_dataset)) if args.max_samples else len(full_dataset)
    )
    if len(runtime_dataset) != expected_runtime_rows:
        raise ValueError(
            f"sample selection produced {len(runtime_dataset)}, expected {expected_runtime_rows}"
        )

    adapters = dict(args.adapter)
    if not adapters or len(adapters) != len(args.adapter):
        raise ValueError("expected one or more distinct adapter labels")
    unknown_adapters = sorted(set(adapters) - set(TEACHERS))
    if unknown_adapters:
        raise ValueError(f"unknown teacher adapter labels: {unknown_adapters}")
    completed_steps = _validate_adapters(adapters, args.teacher_model_path)
    return {
        "students": [
            {"personality": personality, "split": split, "name": name}
            for personality, split, name in selected_rows
        ],
        "teachers": list(adapters),
        "adapters": {label: str(path) for label, path in adapters.items()},
        "full_dataset_rows": len(full_dataset),
        "runtime_dataset_rows": len(runtime_dataset),
        "runtime_dataset_sha256": dataset_sha256(runtime_dataset),
        "seed": config.seed,
        "completed_steps": completed_steps,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(
                asdict(config),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(
            Path(config.personality.prompts_path).read_bytes()
        ).hexdigest(),
        "complaints_sha256": hashlib.sha256(
            Path(config.personality.complaints_path).read_bytes()
        ).hexdigest(),
        "semantics": {
            "presolve": True,
            "verify": False,
            "leak_handling_mode": "reward_only",
            "format_handling_mode": "continue",
            "gate_sample_rate": config.personality.gate_sample_rate,
            "explain_ratio": config.personality.explain_ratio,
            "generalization_replays": args.replays,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=_parse_adapter, action="append", default=[])
    parser.add_argument(
        "--personality",
        choices=[row[0] for row in PERSONALITIES],
        action="append",
        default=[],
        help="evaluate only this personality; repeat to select a subset",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--replays", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples < 0 or args.replays < 1:
        raise ValueError("--max-samples must be non-negative and --replays positive")
    report = run_preflight(args)
    split_counts = {
        split: sum(student["split"] == split for student in report["students"])
        for split in ("id", "ood")
    }
    print("[preflight] standard personality Eval semantics: PASS")
    print(
        f"[preflight] students={len(report['students'])} "
        f"(ID={split_counts['id']}, OOD={split_counts['ood']}), "
        f"full_rows={report['full_dataset_rows']}, "
        f"phase_rows={report['runtime_dataset_rows']}"
    )
    print(
        f"[preflight] dataset_sha256={report['runtime_dataset_sha256']}, "
        f"completed_steps={report['completed_steps']}"
    )
    print("PREFLIGHT_JSON=" + json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
