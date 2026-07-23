from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset

from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
)
from examples.tutor.scripts.evaluate_api_teacher import (
    DEFAULT_CONFIG_PATH,
    ApiTeacherClient,
    EpisodeResult,
    EpisodeSpec,
    PresolveMode,
    aggregate_mode,
    aggregate_repeat_metrics,
    aggregate_report,
    append_jsonl,
    build_eval_workflow_kwargs,
    close_workflow_api_clients,
    deepseek_non_thinking_params,
    latest_results,
    load_existing_results,
    load_experiment_config,
    merge_dicts,
    prepare_episode_workflow_kwargs,
    prepare_test_dataset,
    resolve_presolve_modes,
    result_from_workflow,
    result_needs_retry,
    run_episode,
    run_without_proxy_environment,
    student_prompt_row_counts,
    without_config_snapshot_writes,
)
from examples.tutor.train import EvalStudentPrompt

CONFIG_PATH = Path(DEFAULT_CONFIG_PATH)
PERSONA_CONFIG_PATH = CONFIG_PATH.with_name("persona-v1") / (
    "qwen8b-qwen1.7b-math-student5.yaml"
)
HELDOUT_CONFIG_PATH = CONFIG_PATH.with_name("persona-v1") / (
    "qwen8b-qwen1.7b-math-student5-heldout5.yaml"
)

PROXY_ENV_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


def test_run_without_proxy_environment_spans_await_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazily created API clients cannot inherit proxies after an await boundary."""

    expected = {name: f"proxy-for-{name}" for name in PROXY_ENV_VARS}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    async def operation() -> str:
        await asyncio.sleep(0)
        assert all(name not in os.environ for name in PROXY_ENV_VARS)
        return "ok"

    result = asyncio.run(run_without_proxy_environment(operation))

    assert result == "ok"
    assert {name: os.environ.get(name) for name in PROXY_ENV_VARS} == expected


def test_without_config_snapshot_writes_restores_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone config loading must not leak its synthetic rank setting."""

    monkeypatch.setenv("RANK", "7")

    with without_config_snapshot_writes():
        assert os.environ["RANK"] == "1"

    assert os.environ["RANK"] == "7"


def test_default_config_targets_requested_pre_aleak_experiment() -> None:
    """The standalone evaluator should be safe against accidental non-aleak runs."""

    assert CONFIG_PATH.name == "qwen8b-qwen1.7b-math-pre-aleak.yaml"
    assert CONFIG_PATH.exists()


def _args(**overrides: Any) -> SimpleNamespace:
    values = {
        "teacher_temperature": 1.0,
        "teacher_top_p": 0.95,
        "teacher_max_tokens": 4096,
        "presolve_attempts": 0,
        "presolve_max_tokens": None,
        "concurrency": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(
    *,
    key: str,
    mode: str = "presolve_off",
    termination_reason: str,
    error: str | None = None,
    pre_solved: bool = False,
    taught_success: bool = False,
    teacher_pre_accepted: bool | None = None,
    student_call_failed: bool = False,
    leak_check_failed_count: int = 0,
    answer_judge_failed_count: int = 0,
    teacher_pre_error_count: int = 0,
    generalization: dict[str, dict[str, Any]] | None = None,
    dataset_index: int = 0,
    attempt: int = 1,
    student_prompt_pool: str = "",
    student_prompt_index: int | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        key=key,
        mode=mode,
        presolve_enabled=mode == "presolve_on",
        dataset_index=dataset_index,
        attempt=attempt,
        item_id=key,
        student_name="qwen3-1.7b",
        student_model="qwen3-1.7b",
        termination_reason=termination_reason,
        error=error,
        pre_solved=pre_solved,
        taught_success=taught_success,
        final_correct=pre_solved or taught_success,
        num_turns=1 if taught_success else 0,
        solve_turn=1 if taught_success else None,
        leak_count=0,
        leak_check_failed_count=leak_check_failed_count,
        format_error_count=0,
        student_call_failed=student_call_failed,
        answer_judge_used_count=0,
        answer_judge_failed_count=answer_judge_failed_count,
        answer_judge_override_correct_count=0,
        total_reward=0.0,
        teacher_pre_accepted=teacher_pre_accepted,
        teacher_pre_attempts=1 if teacher_pre_accepted is not None else 0,
        teacher_pre_error_count=teacher_pre_error_count,
        generalization=generalization or {},
        latest_student_answer_preview="",
        trace_path=None,
        duration_seconds=0.1,
        student_prompt_pool=student_prompt_pool,
        student_prompt_index=student_prompt_index,
    )


def test_deepseek_non_thinking_params_override_conflicting_request() -> None:
    """The requested evaluation profile always keeps DeepSeek in chat mode."""

    user_params = {
        "seed": 7,
        "extra_body": {
            "chat_template_kwargs": {"thinking": True},
            "custom": "kept",
        },
    }

    resolved = merge_dicts(user_params, deepseek_non_thinking_params(seed=42))

    assert resolved["seed"] == 42
    assert resolved["extra_body"]["chat_template_kwargs"]["thinking"] is False
    assert resolved["extra_body"]["custom"] == "kept"


@pytest.mark.parametrize(
    "overrides",
    [
        {"error": "timeout"},
        {"student_call_failed": True},
        {"leak_check_failed_count": 1},
        {"answer_judge_failed_count": 1},
        {"teacher_pre_error_count": 1},
    ],
)
def test_result_needs_retry_for_execution_and_diagnostic_failures(
    overrides: dict[str, Any],
) -> None:
    result = _result(key="retry", termination_reason="max_turns", **overrides)

    assert result_needs_retry(
        result,
        retry_diagnostic_failures=result.error is None,
    )


def test_result_needs_retry_respects_resume_opt_outs() -> None:
    error = _result(key="error", termination_reason="error", error="timeout")
    diagnostic = _result(
        key="diagnostic",
        termination_reason="leak",
        leak_check_failed_count=1,
    )
    clean = _result(key="clean", termination_reason="success", taught_success=True)

    assert not result_needs_retry(error, retry_errors=False)
    assert not result_needs_retry(
        diagnostic,
        retry_diagnostic_failures=False,
    )
    assert not result_needs_retry(clean)


def test_api_teacher_client_forces_model_and_preserves_budget() -> None:
    """The workflow's reduced token budget wins while the real model is forced."""

    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"ok": True}

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    )
    client = ApiTeacherClient(
        base_url="http://unused/v1",
        api_key="EMPTY",
        model="DeepSeek-V3.2",
        timeout=1.0,
        max_retries=0,
        request_params={
            "seed": 42,
            "extra_body": {"chat_template_kwargs": {"thinking": False}},
        },
        client=fake_client,
    )

    asyncio.run(
        client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": "hello"}],
            temperature=1.0,
            top_p=0.95,
            max_completion_tokens=123,
        )
    )

    assert calls[0]["model"] == "DeepSeek-V3.2"
    assert calls[0]["max_completion_tokens"] == 123
    assert calls[0]["extra_body"]["chat_template_kwargs"]["thinking"] is False


def test_load_experiment_config_freezes_original_student(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auxiliary overrides cannot redirect the configured Qwen student."""

    monkeypatch.setenv("INF_API_KEY", "test-key")

    config, students = load_experiment_config(
        str(CONFIG_PATH),
        [
            "auxiliary_model.base_url=http://deepseek.example/v1",
            "auxiliary_model.model=DeepSeek-V3.2",
        ],
    )

    assert config.auxiliary_model.model == "DeepSeek-V3.2"
    assert students[0]["name"] == "qwen3-1.7b"
    assert students[0]["model"] == "qwen3-1.7b"
    assert "deepseek.example" not in students[0]["base_url"]


def test_build_eval_workflow_kwargs_preserves_role_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teacher, auxiliary, and student retain their independent settings."""

    monkeypatch.setenv("INF_API_KEY", "test-key")
    config, students = load_experiment_config(str(CONFIG_PATH), [])
    config.teacher_pre.verify = False

    kwargs = build_eval_workflow_kwargs(
        config=config,
        student_models=students,
        tokenizer=object(),
        args=_args(),
        presolve_enabled=True,
    )

    assert kwargs["gconfig"].temperature == pytest.approx(1.0)
    assert kwargs["gconfig"].top_p == pytest.approx(0.95)
    assert kwargs["gconfig"].max_new_tokens == 4096
    assert kwargs["aux_model"] == "qwen3-8b"
    assert kwargs["aux_base_url"] == config.auxiliary_model.base_url
    assert kwargs["aux_api_key"] == "test-key"
    assert kwargs["aux_temperature"] == pytest.approx(0.0)
    assert kwargs["aux_top_p"] == pytest.approx(1.0)
    assert kwargs["aux_max_tokens"] == 1024
    assert kwargs["aux_timeout"] == 120
    assert kwargs["max_concurrent_aux_calls"] == 4
    assert kwargs["aux_request_params"] == config.auxiliary_model.request_params
    assert kwargs["aux_request_params"]["seed"] == 42
    assert kwargs["aux_request_params"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "response_format" not in kwargs["aux_request_params"]
    assert kwargs["student_models"][0]["model"] == "qwen3-1.7b"
    assert kwargs["teacher_pre_enabled"] is True
    assert kwargs["teacher_pre_verify"] is False
    assert kwargs["teacher_pre_attempts"] == 3
    assert kwargs["teacher_pre_max_tokens"] == 0
    assert kwargs["teacher_anti_leak_instruction_enabled"] is True
    assert kwargs["teacher_prompt_pool_path"] == ""
    assert kwargs["student_prompt_pool_path"] == ""
    assert kwargs["student_generalize_enabled"] is True


def test_build_eval_workflow_kwargs_evaluates_seen_student_prompt_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone evaluator exhaustively covers the configured seen pool."""

    monkeypatch.setenv("INF_API_KEY", "test-key")
    config, students = load_experiment_config(str(PERSONA_CONFIG_PATH), [])

    kwargs = build_eval_workflow_kwargs(
        config=config,
        student_models=students,
        tokenizer=object(),
        args=_args(),
        presolve_enabled=False,
    )

    assert kwargs["student_prompt_pool_path"] == config.prompt_pool.student_seen_path
    assert kwargs["student_heldout_prompt_pool_path"] == ""


def test_build_eval_workflow_kwargs_loads_heldout_student_prompt_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The standalone evaluator keeps seen and held-out pools separate."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    config, students = load_experiment_config(str(HELDOUT_CONFIG_PATH), [])

    kwargs = build_eval_workflow_kwargs(
        config=config,
        student_models=students,
        tokenizer=object(),
        args=_args(),
        presolve_enabled=False,
    )

    assert kwargs["student_prompt_pool_path"] == config.prompt_pool.student_seen_path
    assert (
        kwargs["student_heldout_prompt_pool_path"]
        == config.prompt_pool.student_heldout_path
    )


def test_prepare_test_dataset_limits_base_items_before_full_prompt_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smoke-test limit covers the base prompt and every student persona."""

    import examples.tutor.scripts.evaluate_api_teacher as module

    source = Dataset.from_dict({"id": [10, 20], "task": ["a", "b"]})
    config = SimpleNamespace(
        valid_dataset=SimpleNamespace(scheduling_spec="remote"),
        evaluator=SimpleNamespace(max_samples=None),
        seed=42,
    )
    students = [{"name": "student-a"}, {"name": "student-b"}]
    monkeypatch.setattr(module, "get_custom_dataset", lambda **_kwargs: source)
    monkeypatch.setattr(
        module.tutor_train,
        "_load_eval_student_prompts",
        lambda _config: tuple(
            EvalStudentPrompt(pool="seen", index=index, suffix=f"prompt-{index}")
            for index in range(3)
        ),
    )

    dataset = prepare_test_dataset(
        config,
        students,
        tokenizer=object(),
        limit=1,
    )

    combinations = set(
        zip(
            dataset["id"],
            dataset[TUTOR_EVAL_STUDENT_FIELD],
            dataset[TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD],
            dataset[TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD],
            strict=True,
        )
    )
    assert len(dataset) == 8
    assert combinations == {
        (10, student_name, None, None) for student_name in ("student-a", "student-b")
    } | {
        (10, student_name, "seen", prompt_index)
        for student_name in ("student-a", "student-b")
        for prompt_index in range(3)
    }
    assert student_prompt_row_counts(dataset) == {
        ("seen", 0): 2,
        ("seen", 1): 2,
        ("seen", 2): 2,
    }


def test_aggregate_mode_separates_initial_and_teaching_success() -> None:
    """Teaching rate excludes initial solves, errors, and presolve skips."""

    results = [
        _result(key="pre", termination_reason="pre_solved", pre_solved=True),
        _result(key="taught", termination_reason="success", taught_success=True),
        _result(key="failed", termination_reason="max_turns"),
        _result(key="error", termination_reason="error", error="boom"),
    ]

    summary = aggregate_mode(results, expected=4)

    assert summary["pre_solved_count"] == 1
    assert summary["taught_success_count"] == 1
    assert summary["teaching_solve_rate_observed_initially_unsolved"] == pytest.approx(
        0.5
    )
    assert summary["full_set_final_correct_rate"] == pytest.approx(0.5)
    assert summary["completed_final_correct_rate"] == pytest.approx(2 / 3)
    assert summary["teaching_solve_rate_full_set_conservative"] == pytest.approx(1 / 3)
    assert summary["teaching_lift_full_set"] == pytest.approx(0.25)
    assert summary["execution_coverage_rate"] == pytest.approx(0.75)


def test_aggregate_mode_reports_presolve_coverage() -> None:
    """Skipped presolve rows remain visible in the full-set denominator."""

    results = [
        _result(
            key="pre",
            mode="presolve_on",
            termination_reason="pre_solved",
            pre_solved=True,
            teacher_pre_accepted=True,
        ),
        _result(
            key="taught",
            mode="presolve_on",
            termination_reason="success",
            taught_success=True,
            teacher_pre_accepted=True,
        ),
        _result(
            key="skip",
            mode="presolve_on",
            termination_reason="pre_solve_skipped",
            teacher_pre_accepted=False,
        ),
        _result(
            key="failed",
            mode="presolve_on",
            termination_reason="max_turns",
            teacher_pre_accepted=True,
        ),
    ]

    summary = aggregate_mode(results, expected=4)

    assert summary["presolve_covered_count"] == 3
    assert summary["presolve_coverage_rate"] == pytest.approx(0.75)
    assert summary["presolve_coverage_rate_full_set"] == pytest.approx(0.75)
    assert summary["presolve_acceptance_rate"] == pytest.approx(0.75)
    assert summary["full_set_final_correct_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("choice", "config_enabled", "expected"),
    [
        ("config", False, [PresolveMode("presolve_off", False)]),
        ("config", True, [PresolveMode("presolve_on", True)]),
        ("off", True, [PresolveMode("presolve_off", False)]),
        ("on", False, [PresolveMode("presolve_on", True)]),
        (
            "both",
            False,
            [
                PresolveMode("presolve_off", False),
                PresolveMode("presolve_on", True),
            ],
        ),
    ],
)
def test_resolve_presolve_modes_returns_expected_variants(
    choice: str,
    config_enabled: bool,
    expected: list[PresolveMode],
) -> None:
    """Presolve mode selection is explicit and deterministic."""

    assert resolve_presolve_modes(choice, config_enabled) == expected


def test_load_existing_results_tolerates_truncated_tail_and_deduplicates(
    tmp_path: Path,
) -> None:
    """Resume survives an interrupted final write and keeps the latest duplicate."""

    first = _result(key="same", termination_reason="max_turns")
    latest = _result(key="same", termination_reason="success", taught_success=True)
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(asdict(first)),
                json.dumps(asdict(latest)),
                '{"key":',
            ]
        ),
        encoding="utf-8",
    )

    results = load_existing_results(path)

    assert len(results) == 1
    assert results[0].termination_reason == "success"

    appended = _result(key="next", termination_reason="max_turns", dataset_index=1)
    append_jsonl(path, asdict(appended))
    reloaded = load_existing_results(path)
    assert [result.key for result in reloaded] == ["same", "next"]


def test_load_existing_results_rejects_structurally_invalid_final_record(
    tmp_path: Path,
) -> None:
    """Valid JSON with missing EpisodeResult fields is not mistaken for truncation."""

    path = tmp_path / "results.jsonl"
    path.write_text('{"key": "incomplete"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid resume record"):
        load_existing_results(path)


def test_load_existing_results_accepts_records_before_prompt_index_field(
    tmp_path: Path,
) -> None:
    """Adding persona metadata does not invalidate older resumable results."""

    payload = asdict(_result(key="old", termination_reason="max_turns"))
    payload.pop("student_prompt_index")
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    results = load_existing_results(path)

    assert len(results) == 1
    assert results[0].student_prompt_index is None


def test_api_teacher_client_preserves_fixed_config_seed_for_every_call() -> None:
    """The API adapter must not invent per-episode or per-call seed semantics."""

    calls: list[dict[str, Any]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"ok": True}

    client = ApiTeacherClient(
        base_url="http://unused/v1",
        api_key="EMPTY",
        model="DeepSeek-V3.2",
        timeout=1.0,
        max_retries=0,
        request_params={"seed": 42},
        client=SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    async def make_calls() -> None:
        await client.chat.completions.create(messages=[])
        await client.chat.completions.create(messages=[])
        await client.chat.completions.create(messages=[])

    asyncio.run(make_calls())

    assert [call["seed"] for call in calls] == [42, 42, 42]


def test_prepare_episode_workflow_kwargs_preserves_fixed_role_seeds() -> None:
    """Episode construction must copy, not rewrite, student or auxiliary seeds."""

    base = {
        "student_models": [
            {
                "request_params": {
                    "seed": 42,
                    "extra_body": {"top_k": 20, "min_p": 0},
                }
            }
        ],
        "aux_request_params": {
            "seed": 42,
            "extra_body": {"top_k": 20, "min_p": 0},
        },
    }

    episode = prepare_episode_workflow_kwargs(base)

    assert episode == base
    assert episode is not base
    assert episode["student_models"] is not base["student_models"]
    assert episode["aux_request_params"] is not base["aux_request_params"]


def test_clean_rates_filter_success_numerators_consistently() -> None:
    """Student or auxiliary failures cannot make a clean rate exceed one."""

    results = [
        _result(
            key="recovered_student",
            termination_reason="success",
            taught_success=True,
            student_call_failed=True,
        ),
        _result(key="clean_failure", termination_reason="max_turns"),
        _result(
            key="recovered_aux",
            termination_reason="success",
            taught_success=True,
            leak_check_failed_count=1,
        ),
    ]

    summary = aggregate_mode(results, expected=3)

    assert summary["teaching_solve_rate_clean_student_calls"] == pytest.approx(0.5)
    assert summary["teaching_solve_rate_clean_aux_calls"] == pytest.approx(0.5)
    assert summary["leak_check_failed_episode_count"] == 1


def test_generalization_reports_conditional_and_full_set_rates() -> None:
    """Always-mode probes on failed originals cannot inflate taught-success rates."""

    correct_probe = {"attempted": True, "correct": True, "student_error": None}
    results = [
        _result(
            key="taught",
            termination_reason="success",
            taught_success=True,
            generalization={"level1": correct_probe},
        ),
        _result(
            key="original_failed",
            termination_reason="max_turns",
            generalization={"level1": correct_probe},
            dataset_index=1,
        ),
    ]

    summary = aggregate_mode(results, expected=2, generalization_enabled=True)

    level1 = summary["generalization"]["level1"]
    assert level1["attempt_coverage_on_taught_success"] == pytest.approx(1.0)
    assert level1["end_to_end_correct_rate_on_taught_success"] == pytest.approx(1.0)
    assert level1["end_to_end_correct_rate_full_set"] == pytest.approx(1.0)
    assert summary["generalization"]["level2"]["attempted"] == 0


def test_aggregate_report_adds_item_any_success_for_multiple_attempts() -> None:
    """Per-item any-success stays separate from flattened per-attempt accuracy."""

    results = [
        _result(
            key="off:0:1",
            termination_reason="max_turns",
            dataset_index=0,
            attempt=1,
        ),
        _result(
            key="off:0:2",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
            attempt=2,
        ),
        _result(
            key="off:1:1",
            termination_reason="max_turns",
            dataset_index=1,
            attempt=1,
        ),
        _result(
            key="off:1:2",
            termination_reason="max_turns",
            dataset_index=1,
            attempt=2,
        ),
    ]

    report = aggregate_report(
        results,
        modes=[PresolveMode("presolve_off", False)],
        dataset_size=2,
        attempts=2,
    )

    mode = report["modes"]["presolve_off"]
    assert mode["teaching_lift_full_set"] == pytest.approx(0.25)
    assert mode["item_any_taught_success_rate_full_set"] == pytest.approx(0.5)


def test_aggregate_repeat_metrics_matches_grouped_eval_stability_semantics() -> None:
    """Repeat statistics are computed within each task before test-set averaging."""

    results = [
        _result(
            key="off:0:1",
            termination_reason="max_turns",
            dataset_index=0,
            attempt=1,
        ),
        _result(
            key="off:0:2",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
            attempt=2,
        ),
        _result(
            key="off:0:3",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
            attempt=3,
        ),
        _result(
            key="off:1:1",
            termination_reason="success",
            taught_success=True,
            dataset_index=1,
            attempt=1,
        ),
        _result(
            key="off:1:2",
            termination_reason="success",
            taught_success=True,
            dataset_index=1,
            attempt=2,
        ),
        _result(
            key="off:1:3",
            termination_reason="success",
            taught_success=True,
            dataset_index=1,
            attempt=3,
        ),
    ]

    repeat = aggregate_repeat_metrics(results, expected_items=2, attempts=3)

    assert repeat["complete_item_count"] == 2
    assert repeat["complete_item_coverage_rate"] == pytest.approx(1.0)
    solved = repeat["solved"]
    assert solved["mean"] == pytest.approx(5 / 6)
    assert solved["variance"] == pytest.approx(1 / 9)
    assert solved["std"] == pytest.approx((2 / 9) ** 0.5 / 2)
    assert solved["agreement"] == pytest.approx(2 / 3)
    assert solved["disagreement"] == pytest.approx(1 / 3)
    assert solved["all_equal"] == pytest.approx(0.5)
    assert solved["any_success"] == pytest.approx(1.0)
    assert solved["all_success"] == pytest.approx(0.5)
    assert solved["success_set_jaccard"] == pytest.approx(0.5)
    assert solved["pairwise_success_set_jaccard"] == pytest.approx(2 / 3)
    assert repeat["final_correct"] == solved


def test_aggregate_repeat_metrics_excludes_incomplete_or_error_items() -> None:
    """A failed attempt must not be mistaken for a valid low-variance task."""

    results = [
        _result(
            key="off:0:1",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
            attempt=1,
        ),
        _result(
            key="off:0:2",
            termination_reason="error",
            error="timeout",
            dataset_index=0,
            attempt=2,
        ),
        _result(
            key="off:1:1",
            termination_reason="success",
            taught_success=True,
            dataset_index=1,
            attempt=1,
        ),
    ]

    repeat = aggregate_repeat_metrics(results, expected_items=2, attempts=3)

    assert repeat["recorded_item_count"] == 2
    assert repeat["complete_item_count"] == 0
    assert repeat["incomplete_item_count"] == 2
    assert repeat["error_item_count"] == 1
    assert repeat["complete_item_coverage_rate"] == pytest.approx(0.0)
    assert repeat["solved"]["variance"] is None
    assert repeat["final_correct"]["success_set_jaccard"] is None


def test_aggregate_report_separates_student_prompt_results() -> None:
    """The main summary stays base-only while personas report independently."""

    results = [
        _result(
            key="off:base:1",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
        ),
        _result(
            key="off:0:1",
            termination_reason="success",
            taught_success=True,
            dataset_index=0,
            student_prompt_pool="seen",
            student_prompt_index=0,
        ),
        _result(
            key="off:1:1",
            termination_reason="max_turns",
            dataset_index=1,
            student_prompt_pool="heldout",
            student_prompt_index=1,
        ),
    ]

    report = aggregate_report(
        results,
        modes=[PresolveMode("presolve_off", False)],
        dataset_size=3,
        attempts=1,
        student_prompt_rows={("seen", 0): 1, ("heldout", 1): 1},
    )

    mode = report["modes"]["presolve_off"]
    assert report["base_prompt_rows"] == 1
    assert mode["recorded_attempts"] == 1
    assert mode["taught_success_count"] == 1
    assert mode["teaching_lift_full_set"] == pytest.approx(1.0)
    assert mode["item_any_taught_success_rate_full_set"] == pytest.approx(1.0)
    prompts = report["modes"]["presolve_off"]["student_prompts"]
    assert prompts["seen"]["0"]["taught_success_count"] == 1
    assert prompts["seen"]["0"]["teaching_lift_full_set"] == pytest.approx(1.0)
    assert prompts["heldout"]["1"]["taught_success_count"] == 0
    assert prompts["heldout"]["1"]["teaching_lift_full_set"] == pytest.approx(0.0)


def test_aggregate_mode_deduplicates_retry_records() -> None:
    """A successful retry replaces its prior error instead of inflating counts."""

    failed = _result(key="same", termination_reason="error", error="temporary")
    retried = _result(key="same", termination_reason="success", taught_success=True)

    summary = aggregate_mode([failed, retried], expected=1)

    assert summary["recorded_attempts"] == 1
    assert summary["completed_attempts"] == 1
    assert summary["taught_success_count"] == 1
    assert latest_results([failed, retried]) == [retried]


def test_close_workflow_api_clients_includes_tracked_answer_judge() -> None:
    """The separately created answer-judge pool is closed exactly once."""

    class FakeClient:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1

    client = FakeClient()
    wrapper = SimpleNamespace(caller=SimpleNamespace(_client=client))
    workflow = SimpleNamespace(
        aux_caller=wrapper,
        confidence_aux_caller=None,
        extra_api_callers=[wrapper],
        student_model_runtimes={},
    )

    asyncio.run(close_workflow_api_clients(workflow))

    assert client.close_count == 1


def test_run_episode_records_workflow_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One bad per-episode constructor cannot abort the complete evaluation gather."""

    import examples.tutor.scripts.evaluate_api_teacher as module

    class BrokenWorkflow:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            raise RuntimeError("constructor failed")

    fake_teacher = ApiTeacherClient(
        base_url="http://unused/v1",
        api_key="EMPTY",
        model="DeepSeek-V3.2",
        timeout=1.0,
        max_retries=0,
        request_params={"seed": 42},
        client=SimpleNamespace(),
    )
    spec = EpisodeSpec(
        mode=PresolveMode("presolve_off", False),
        dataset_index=0,
        attempt=1,
        row={"id": "sample", TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 2},
    )
    monkeypatch.setattr(module, "RecordingTutorWorkflow", BrokenWorkflow)

    result = asyncio.run(
        run_episode(
            spec=spec,
            workflow_kwargs={"student_models": [], "aux_request_params": {}},
            teacher_client=fake_teacher,
            output_dir=tmp_path,
            save_traces="none",
            keep_env_proxy=False,
        )
    )

    assert result.termination_reason == "error"
    assert result.error == "RuntimeError: constructor failed"
    assert result.student_prompt_index == 2


@pytest.mark.parametrize(
    ("termination_reason", "pre_success", "expected_pre", "expected_taught"),
    [
        ("pre_solved", True, True, False),
        ("success", False, False, True),
        ("pre_solve_skipped", False, False, False),
        ("leak", False, False, False),
        ("context_limit", False, False, False),
        ("max_turns", False, False, False),
    ],
)
def test_result_from_workflow_maps_test_termination_semantics(
    termination_reason: str,
    pre_success: bool,
    expected_pre: bool,
    expected_taught: bool,
) -> None:
    """Episode labels match the existing workflow's test termination contract."""

    workflow = SimpleNamespace(
        captured_stats={
            "termination_reason": termination_reason,
            "pre_success": pre_success,
            "student_call_failed": False,
            "student_name": "qwen3-1.7b",
            "total_reward": 0.0,
            "leak_count": int(termination_reason == "leak"),
        },
        captured_trace={"student_model": "qwen3-1.7b"},
        last_traces=[],
        last_teacher_pre_solve_result=None,
        last_student_generalization_results=[],
        leak_check_failed_count=0,
        answer_judge_used_count=0,
        answer_judge_failed_count=0,
        answer_judge_override_correct_count=0,
    )
    spec = EpisodeSpec(
        mode=PresolveMode("presolve_off", False),
        dataset_index=0,
        attempt=1,
        row={"id": "sample", TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 2},
    )

    result = result_from_workflow(workflow=workflow, spec=spec, duration_seconds=0.1)

    assert result.pre_solved is expected_pre
    assert result.taught_success is expected_taught
    assert result.final_correct is (expected_pre or expected_taught)
    assert result.student_prompt_index == 2


def test_result_from_workflow_marks_all_presolve_transport_failures_retryable() -> None:
    """Endpoint failures are execution errors, not ordinary presolve rejections."""

    teacher_pre = SimpleNamespace(
        accepted=False,
        attempts=[
            SimpleNamespace(error="timeout"),
            SimpleNamespace(error="server overloaded"),
        ],
    )
    workflow = SimpleNamespace(
        captured_stats={
            "termination_reason": "pre_solve_skipped",
            "pre_success": False,
            "student_call_failed": False,
            "student_name": "qwen3-1.7b",
            "total_reward": 0.0,
            "leak_count": 0,
        },
        captured_trace={"student_model": "qwen3-1.7b"},
        last_traces=[],
        last_teacher_pre_solve_result=teacher_pre,
        last_student_generalization_results=[],
        leak_check_failed_count=0,
        answer_judge_used_count=0,
        answer_judge_failed_count=0,
        answer_judge_override_correct_count=0,
    )
    spec = EpisodeSpec(
        mode=PresolveMode("presolve_on", True),
        dataset_index=0,
        attempt=1,
        row={"id": "sample"},
    )

    result = result_from_workflow(workflow=workflow, spec=spec, duration_seconds=0.1)

    assert result.error is not None
    assert result.error.startswith("TeacherPreSolveError:")
    assert result.teacher_pre_error_count == 2
