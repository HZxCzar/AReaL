#!/usr/bin/env python3
"""Read-only preflight for the 0825 six-preference full-student evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

from areal.utils.hf_utils import load_hf_tokenizer

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


PREFERENCES = (
    ("none", "qwen3-1.7b-text-original"),
    ("feedback", "qwen3-1.7b-text-original-feedback"),
    ("hinting", "qwen3-1.7b-text-original-hinting"),
    ("instructing", "qwen3-1.7b-text-original-instructing"),
    ("explaining", "qwen3-1.7b-text-original-explaining"),
    ("modeling", "qwen3-1.7b-text-original-modeling"),
    ("questioning", "qwen3-1.7b-text-original-questioning"),
)


def _eval_args() -> Namespace:
    return Namespace(
        teacher_temperature=None,
        teacher_top_p=None,
        teacher_max_tokens=None,
        presolve_attempts=0,
        presolve_max_tokens=None,
    )


def _validate_adapter(adapter: Path, teacher_model_path: Path) -> int:
    model_file = adapter / "adapter_model.safetensors"
    config_file = adapter / "adapter_config.json"
    if not model_file.is_file() or not config_file.is_file():
        raise ValueError(f"incomplete LoRA checkpoint: {adapter}")
    manifest = json.loads(config_file.read_text(encoding="utf-8"))
    if manifest.get("peft_type") != "LORA" or int(manifest.get("r", 0)) != 16:
        raise ValueError(f"expected a rank-16 LoRA checkpoint: {adapter}")
    configured_base = Path(str(manifest.get("base_model_name_or_path", "")))
    if configured_base.resolve() != teacher_model_path.resolve():
        raise ValueError(
            "adapter base model differs from configured teacher model: "
            f"{configured_base} != {teacher_model_path}"
        )
    marker = adapter.name.rsplit("globalstep", 1)
    if len(marker) != 2 or not marker[1].isdigit():
        raise ValueError(f"checkpoint directory has no globalstep: {adapter}")
    return int(marker[1]) + 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: actual={actual!r}, expected={expected!r}")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config, students = load_experiment_config(str(args.config), [])
    tutor_train._apply_eval_average_rollouts(config)

    expected_names = [name for _, name in PREFERENCES]
    actual_names = [str(student["name"]) for student in students]
    _require_equal("student preference pool", actual_names, expected_names)
    for student in students:
        _require_equal(
            f"{student['name']} model",
            str(student.get("model") or "").lower(),
            "qwen3-1.7b",
        )

    _require_equal(
        "personality.gate_prompt_version", config.personality.gate_prompt_version, "v2"
    )
    _require_equal(
        "personality.gate_decision_mode",
        config.personality.gate_decision_mode,
        args.gate_decision_mode,
    )
    _require_equal(
        "personality.gate_sample_rate", float(config.personality.gate_sample_rate), 1.0
    )
    _require_equal(
        "personality.gated_turn_visibility",
        config.personality.gated_turn_visibility,
        "teacher_only",
    )
    _require_equal(
        "auxiliary model",
        str(config.auxiliary_model.model or "").lower(),
        "qwen3-8b",
    )

    prompts = load_personality_prompts(config.personality.prompts_path)
    bare, explained = load_personality_complaints(config.personality.complaints_path)
    demanding = [preference for preference, _ in PREFERENCES if preference != "none"]
    missing_prompts = [
        preference for preference in demanding if preference not in prompts
    ]
    missing_complaints = [
        preference for preference in demanding if preference not in explained
    ]
    if missing_prompts or missing_complaints or not bare:
        raise ValueError(
            "incomplete personality prompt/complaint bank: "
            f"prompts={missing_prompts}, complaints={missing_complaints}, "
            f"bare={len(bare)}"
        )

    eval_args = _eval_args()
    resolve_teacher_generation_args(eval_args, config)
    if args.require_base_aux and config.auxiliary_model.mode != "api":
        raise ValueError(
            "base-aux evaluation requires auxiliary_model.mode='api', got "
            f"{config.auxiliary_model.mode!r}"
        )
    external_self_aux = (
        {
            "base_url": "http://127.0.0.1:1/v1",
            "model": "qwen3-8b",
            "api_key": "EMPTY",
            "request_params": {},
        }
        if config.auxiliary_model.mode == "self"
        else None
    )
    effective_aux_mode = ""
    effective_aux_has_lora = False
    for student in students:
        effective = build_eval_workflow_kwargs(
            config=config,
            student_models=[student],
            tokenizer=object(),
            args=eval_args,
            presolve_enabled=effective_eval_presolve_enabled(config),
            external_self_aux=external_self_aux,
        )
        effective_aux_mode = str(effective["aux_mode"])
        effective_aux_has_lora = bool(
            (effective.get("aux_request_params", {}).get("extra_body") or {}).get(
                "lora_path"
            )
        )
        checks = {
            "presolve": effective["teacher_pre_enabled"] is True,
            "no_verify": effective["teacher_pre_verify"] is False,
            "leak_continues": effective["leak_handling_mode"] == "reward_only",
            "format_continues": effective["format_handling_mode"] == "continue",
            "gate_v2": effective["personality"].get("gate_prompt_version") == "v2",
            "gate_decision_mode": effective["personality"].get("gate_decision_mode")
            == args.gate_decision_mode,
            "original_retest": effective["student_generalize_retest_original"] is True,
            "preleak_retest": effective["eval_preleak_retest"] is True,
            "replays": int(effective["student_generalize_replays"]) == args.replays,
        }
        if args.require_base_aux:
            checks.update(
                {
                    "auxiliary_api": effective_aux_mode == "api",
                    "auxiliary_without_lora": not effective_aux_has_lora,
                }
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"{student['name']}: eval semantics failed: {failed}")

    if config.student_generalize.enabled:
        tutor_train._prepare_math_generalization_data(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    student_prompts = tutor_train._load_eval_student_prompts(config)
    if student_prompts:
        raise ValueError("unexpected scripted eval-student prompt rows")

    saved_max_samples = config.evaluator.max_samples
    config.evaluator.max_samples = None
    full_dataset = prepare_test_dataset(
        config,
        [students[0]],
        tokenizer=tokenizer,
        limit=0,
        stratified_max_samples=0,
        student_prompts=student_prompts,
    )
    config.evaluator.max_samples = saved_max_samples
    runtime_dataset = prepare_test_dataset(
        config,
        [students[0]],
        tokenizer=tokenizer,
        limit=0,
        stratified_max_samples=args.stratified_max_samples,
        student_prompts=student_prompts,
    )
    expected_rows = (
        len(full_dataset)
        if args.stratified_max_samples == 0
        else min(args.stratified_max_samples, len(full_dataset))
    )
    if len(runtime_dataset) != expected_rows:
        raise ValueError(
            f"stratified selection produced {len(runtime_dataset)}, expected {expected_rows}"
        )

    if args.teacher_variant == "trained":
        if args.adapter is None:
            raise ValueError("trained teacher preflight requires --adapter")
        completed_steps: int | None = _validate_adapter(
            args.adapter, args.teacher_model_path
        )
        adapter: str | None = str(args.adapter)
    else:
        if args.adapter is not None:
            raise ValueError("untrained teacher preflight must not receive --adapter")
        completed_steps = None
        adapter = None
    return {
        "teacher": args.teacher_variant,
        "students": [
            {"preference": preference, "name": name} for preference, name in PREFERENCES
        ],
        "adapter": adapter,
        "completed_steps": completed_steps,
        "full_dataset_rows": len(full_dataset),
        "runtime_dataset_rows": len(runtime_dataset),
        "runtime_dataset_sha256": dataset_sha256(runtime_dataset),
        "seed": int(config.seed),
        "dataset_selection": {
            "strategy": (
                "full"
                if args.stratified_max_samples == 0
                else "math_type_level_stratified"
            ),
            "requested_rows": args.stratified_max_samples,
            "strata": ["metadata.type", "metadata.level"],
        },
        "config_sha256": _sha256(args.config),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(
                asdict(config),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "prompts_sha256": _sha256(Path(config.personality.prompts_path)),
        "complaints_sha256": _sha256(Path(config.personality.complaints_path)),
        "semantics": {
            "auxiliary_source_mode": config.auxiliary_model.mode,
            "auxiliary_effective_mode": effective_aux_mode,
            "auxiliary_has_lora_path": effective_aux_has_lora,
            "gate_prompt_version": config.personality.gate_prompt_version,
            "gate_decision_mode": config.personality.gate_decision_mode,
            "gate_sample_rate": float(config.personality.gate_sample_rate),
            "gated_turn_visibility": config.personality.gated_turn_visibility,
            "explain_ratio": float(config.personality.explain_ratio),
            "presolve": True,
            "verify": False,
            "leak_handling_mode": "reward_only",
            "format_handling_mode": "continue",
            "generalization_replays": args.replays,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--teacher-variant",
        choices=("trained", "untrained"),
        default="trained",
    )
    parser.add_argument("--stratified-max-samples", type=int, default=48)
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument(
        "--gate-decision-mode",
        choices=("binary", "classifier", "classifier_logits"),
        default="classifier",
    )
    parser.add_argument("--require-base-aux", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.teacher_model_path = args.teacher_model_path.expanduser().resolve()
    if args.adapter is not None:
        args.adapter = args.adapter.expanduser().resolve()
    if args.stratified_max_samples < 0 or args.replays < 1:
        raise ValueError("sample count must be non-negative and replay count positive")
    report = run_preflight(args)
    print(
        "[preflight] 0825 full-student semantics: PASS; "
        f"students={len(report['students'])}, "
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
