from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.tutor import train as tutor_train
from examples.tutor.scripts.eval_checkpoints import (
    Checkpoint,
    checkpoint_request_params,
)
from examples.tutor.scripts.evaluate_api_teacher import (
    EpisodeResult,
    aggregate_mode,
    build_eval_workflow_kwargs,
    configured_generalization_levels,
    effective_eval_presolve_enabled,
    load_experiment_config,
    resolve_teacher_generation_args,
    select_student_models,
    serialize_generalization,
)

CONFIG_PATH = "examples/tutor/configs/math/0810/4gpu/two-student-text-code-eval.yaml"


def _load_free_chat_config(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "TUTOR_QWEN3_8B_BASE_URL",
        "TUTOR_QWEN3_1_7B_BASE_URL",
        "TUTOR_LLAMA31_8B_BASE_URL",
        "TUTOR_GEMMA3_1B_BASE_URL",
    ):
        monkeypatch.setenv(name, "http://127.0.0.1:1/v1")
    monkeypatch.setenv("INF_API_KEY", "test")
    config, students = load_experiment_config(CONFIG_PATH, [])
    tutor_train._apply_eval_average_rollouts(config)
    return config, students


def _args() -> Namespace:
    return Namespace(
        teacher_temperature=None,
        teacher_top_p=None,
        teacher_max_tokens=None,
        presolve_attempts=0,
        presolve_max_tokens=None,
    )


def test_api_eval_uses_regular_free_chat_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, students = _load_free_chat_config(monkeypatch)
    args = _args()
    resolve_teacher_generation_args(args, config)
    code_student = select_student_models(students, ["qwen3-1.7b-code"])

    kwargs = build_eval_workflow_kwargs(
        config=config,
        student_models=code_student,
        tokenizer=object(),
        args=args,
        presolve_enabled=effective_eval_presolve_enabled(config),
    )

    assert kwargs["free_chat"]["enabled"] is True
    assert kwargs["free_chat"]["budget"] == 5
    assert kwargs["student_generalize_enabled"] is True
    assert kwargs["student_generalize_retest_original"] is True
    assert kwargs["student_generalize_replays"] == 4
    assert kwargs["student_generalize_level1_enabled"] is False
    assert kwargs["student_generalize_level2_enabled"] is False
    assert configured_generalization_levels(kwargs) == (
        "original",
        "original_preleak",
    )
    assert kwargs["leak_handling_mode"] == "reward_only"
    assert kwargs["eval_preleak_retest"] is True
    assert kwargs["format_handling_mode"] == "continue"
    assert kwargs["teacher_pre_verify"] is False
    assert [student["name"] for student in kwargs["student_models"]] == [
        "qwen3-1.7b-code"
    ]
    assert args.teacher_temperature == config.eval_gconfig.temperature
    assert args.teacher_top_p == config.eval_gconfig.top_p
    assert args.teacher_max_tokens == config.eval_gconfig.max_new_tokens


def test_select_student_models_rejects_unknown_student() -> None:
    with pytest.raises(ValueError, match="Unknown --student-name"):
        select_student_models([{"name": "text"}], ["code"])


def test_generalization_serialization_preserves_replay_score() -> None:
    result = SimpleNamespace(
        level="original",
        attempted=True,
        skipped=False,
        skip_reason="",
        judge_result=SimpleNamespace(correct=True),
        student_error=None,
        replay_count=4,
        replay_correct=3,
        confidence=0.0,
    )
    workflow = SimpleNamespace(last_student_generalization_results=[result])

    payload = serialize_generalization(workflow)

    assert payload["original"]["replay_count"] == 4
    assert payload["original"]["replay_correct"] == 3
    assert payload["original"]["score"] == 0.75


def test_aggregate_mode_reports_regular_score_and_code_health() -> None:
    result = EpisodeResult(
        key="presolve_on:0:1",
        mode="presolve_on",
        presolve_enabled=True,
        dataset_index=0,
        attempt=1,
        item_id="test-0",
        student_name="qwen3-1.7b-code",
        student_model="qwen3-1.7b",
        termination_reason="max_turns",
        error=None,
        pre_solved=False,
        taught_success=True,
        final_correct=True,
        num_turns=5,
        solve_turn=None,
        leak_count=0,
        leak_check_failed_count=0,
        format_error_count=0,
        student_call_failed=False,
        answer_judge_used_count=4,
        answer_judge_failed_count=0,
        answer_judge_override_correct_count=0,
        total_reward=0.5,
        teacher_pre_accepted=True,
        teacher_pre_attempts=1,
        teacher_pre_error_count=0,
        generalization={
            "original": {
                "attempted": True,
                "skipped": False,
                "correct": True,
                "student_error": None,
                "replay_count": 4,
                "replay_correct": 3,
                "score": 0.75,
            }
        },
        latest_student_answer_preview="program output",
        trace_path=None,
        duration_seconds=1.0,
        free_chat_enabled=True,
        outcome_score=0.75,
        final_correct_score=0.75,
        no_teaching_baseline=0.25,
        code_stats={
            "crashes": 0,
            "silent_cells": 0,
            "constant_prints": 0,
            "no_program": 0,
        },
    )

    summary = aggregate_mode([result], expected=1, generalization_levels=("original",))

    assert summary["regular_eval_score_mean_full_set"] == 0.75
    assert summary["no_teaching_baseline_mean"] == 0.25
    assert summary["improvement_over_no_teaching_baseline_mean"] == 0.5
    assert summary["generalization"]["original"]["accuracy_on_replays"] == 0.75
    assert summary["code_channel"]["totals"]["no_program"] == 0
    assert summary["code_channel"]["productive_rate_per_turn"] == 1.0


def test_checkpoint_request_params_keeps_qwen_template_and_adds_lora() -> None:
    checkpoint = Checkpoint(step=299, path=Path("/checkpoints/step299"))

    request = checkpoint_request_params(
        '{"seed":42,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}',
        checkpoint,
    )

    extra_body = request["extra_body"]
    assert isinstance(extra_body, dict)
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra_body["lora_path"] == "/checkpoints/step299"
