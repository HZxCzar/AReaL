"""Budget tests use synthetic usage only; no paid API requests."""

import pytest

from examples.tutor.core.api_eval_budget import TeacherBudget


def test_budget_stop_skips_pending_episodes_and_preserves_results(
    monkeypatch, tmp_path
):
    """Stop launching queued episodes and stop retries, but retain finished work."""
    import asyncio
    from types import SimpleNamespace

    from examples.tutor.scripts import evaluate_api_teacher as evaluator

    calls = []
    specs = [
        evaluator.EpisodeSpec(
            mode=evaluator.PresolveMode(name="presolve_on", enabled=True),
            dataset_index=i,
            attempt=1,
            row={"id": str(i)},
        )
        for i in range(3)
    ]

    async def run_episode(**kwargs):
        calls.append(kwargs["spec"].key)
        return evaluator.error_result(
            kwargs["spec"], RuntimeError("budget stop"), duration_seconds=0
        )

    monkeypatch.setattr(evaluator, "run_episode", run_episode)
    results = asyncio.run(
        evaluator.run_all(
            specs=specs,
            completed_keys=set(),
            workflow_kwargs_by_mode={"presolve_on": {}},
            teacher_client=SimpleNamespace(stop_requested=lambda: bool(calls)),
            output_dir=tmp_path,
            save_traces="none",
            concurrency=1,
            log_every=1,
            keep_env_proxy=False,
            error_retries=3,
            retry_backoff_seconds=0,
        )
    )
    assert len(calls) == len(results) == 1
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 1


def test_cached_input_and_reasoning_are_charged_once(tmp_path):
    """Total completion includes thinking; cache tokens use their own rates."""
    budget = TeacherBudget(tmp_path / "teacher_usage.jsonl", 25, [0.2, 0.02, 0.25, 1.2])
    budget.record(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "prompt_tokens_details": {
                    "cached_tokens": 600,
                    "cache_write_tokens": 100,
                },
                "completion_tokens_details": {"reasoning_tokens": 150},
            }
        }
    )
    assert budget.spent == pytest.approx(
        (300 * 0.2 + 600 * 0.02 + 100 * 0.25 + 200 * 1.2) / 1e6
    )
    resumed = TeacherBudget(budget.path, 25, budget.rates)
    assert resumed.spent == budget.spent


def test_threshold_stops_and_resume_does_not_reset_spend(tmp_path):
    """Previously dispatched calls are still recorded after the threshold."""
    budget = TeacherBudget(tmp_path / "teacher_usage.jsonl", 1, [1, 1, 1, 1])
    budget.record({"usage": {"prompt_tokens": 1000000, "completion_tokens": 0}})
    with pytest.raises(RuntimeError, match="budget stopped"):
        budget.check()
    budget.record({"usage": {"prompt_tokens": 100000, "completion_tokens": 0}})
    resumed = TeacherBudget(budget.path, 1, budget.rates)
    assert resumed.stopped()
    assert resumed.spent == pytest.approx(1.1)


@pytest.mark.parametrize("completion,total", [(402, 1182), (1034, 1182)])
def test_total_tokens_accounts_for_thinking_on_resume(tmp_path, completion, total):
    """Gemini visible-only and OpenAI thinking-inclusive usage cost the same."""
    budget = TeacherBudget(
        tmp_path / "teacher_usage.jsonl", 0.003, [0.75, 0.075, 0.75, 3.75]
    )
    budget.record(
        {
            "usage": {
                "prompt_tokens": 148,
                "completion_tokens": completion,
                "total_tokens": total,
            }
        }
    )
    expected = (148 * 0.75 + 1034 * 3.75) / 1e6
    assert budget.spent == pytest.approx(expected)
    assert budget.stopped()
    resumed = TeacherBudget(budget.path, budget.limit, budget.rates)
    assert resumed.spent == pytest.approx(expected)
    assert resumed.stopped()


@pytest.mark.parametrize("total", [float("nan"), float("inf"), -1])
def test_invalid_total_tokens_stops_budget(tmp_path, total):
    """Invalid totals must not bypass accounting for hidden output tokens."""
    budget = TeacherBudget(tmp_path / "teacher_usage.jsonl", 2, [1] * 4)
    budget.record(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": total}}
    )
    assert budget.stopped()


def test_missing_usage_stops_persistently(tmp_path):
    """An unknown bill must not silently become a zero-dollar response."""
    budget = TeacherBudget(tmp_path / "teacher_usage.jsonl", 25, [1] * 4)
    budget.record({"usage": None})
    assert budget.stopped()
    assert TeacherBudget(budget.path, 25, budget.rates).stopped()


def test_unknown_reservation_counts_toward_cap_and_survives_resume(
    monkeypatch, tmp_path
):
    """Missing usage remains visible and charged against the cap, never free."""
    monkeypatch.setenv("TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD", "0.1")
    budget = TeacherBudget(tmp_path / "teacher_usage.jsonl", 0.25, [1] * 4)
    budget.record({"usage": {"prompt_tokens": 100000, "completion_tokens": 0}})
    budget.record({"usage": None})
    assert budget.unknown and not budget.stopped()
    resumed = TeacherBudget(budget.path, 0.25, budget.rates)
    assert resumed.spent == pytest.approx(0.1)
    assert resumed.unknown_count == 1
    resumed.record({"usage": None})
    assert resumed.stopped()
    with pytest.raises(RuntimeError):
        resumed.check()
    monkeypatch.setenv("TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD", "0.01")
    assert TeacherBudget(budget.path, 0.25, budget.rates).stopped()


@pytest.mark.parametrize("reserve", ["-1", "nan", "inf"])
def test_invalid_unknown_reserve_rejected(monkeypatch, tmp_path, reserve):
    """Invalid reservations must fail before any requests are sent."""
    monkeypatch.setenv("TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD", reserve)
    with pytest.raises(ValueError):
        TeacherBudget(tmp_path / "usage.jsonl", 25, [1] * 4)


@pytest.mark.parametrize(
    "limit,rates", [(0, [1] * 4), (float("nan"), [1] * 4), (25, [-1] * 4), (25, [1])]
)
def test_invalid_budget_is_rejected(tmp_path, limit, rates):
    """Reject non-finite or nonsensical budget settings before API calls."""
    with pytest.raises(ValueError):
        TeacherBudget(tmp_path / "usage.jsonl", limit, rates)
