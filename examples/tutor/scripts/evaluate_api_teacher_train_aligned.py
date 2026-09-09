"""Offline-only compatibility entrypoint; leaves the shared evaluator unchanged."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.tutor.scripts import evaluate_api_teacher as evaluator
from examples.tutor.scripts.evaluate_api_teacher import (
    dataset_sha256,
    effective_eval_presolve_enabled,
    load_experiment_config,
    prepare_test_dataset,
    resolve_teacher_generation_args,
)

_original_build = evaluator.build_eval_workflow_kwargs
_original_signature = evaluator.build_run_signature

__all__ = [
    "build_eval_workflow_kwargs",
    "dataset_sha256",
    "effective_eval_presolve_enabled",
    "load_experiment_config",
    "prepare_test_dataset",
    "resolve_teacher_generation_args",
]


def build_eval_workflow_kwargs(**kwargs: Any) -> dict[str, Any]:
    config = kwargs["config"]
    effective = _original_build(**kwargs)
    # These are forwarded by train.main but omitted by the older API evaluator.
    effective.update(
        length_retry_enabled=config.length_retry.enabled,
        length_retry_attempts=config.length_retry.attempts,
        soft_overlong_penalty=asdict(config.reward.soft_overlong),
        student_generalize_gate_pass_credit_only=(
            config.student_generalize.gate_pass_credit_only
        ),
        student_sampling=asdict(config.student_sampling),
    )
    return effective


def build_run_signature(**kwargs: Any) -> dict[str, Any]:
    signature = _original_signature(**kwargs)
    effective = next(iter(kwargs["workflow_kwargs_by_mode"].values()))
    signature["train_aligned_eval"] = {
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **{
            key: effective[key]
            for key in (
                "teacher_end_enabled",
                "length_retry_enabled",
                "length_retry_attempts",
                "soft_overlong_penalty",
                "student_generalize_gate_pass_credit_only",
                "student_sampling",
            )
        },
    }
    return signature


def main() -> None:
    # Change only this standalone process, never the shared source or training.
    evaluator.build_eval_workflow_kwargs = build_eval_workflow_kwargs
    evaluator.build_run_signature = build_run_signature
    evaluator.main()


if __name__ == "__main__":
    main()
