import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from examples.tutor import workflow as tutor_workflow


def _metrics_by_key(calls):
    values = defaultdict(list)
    for call in calls:
        for key, value in call.items():
            values[key].append(value)
    return values


def test_binary_repeat_summary_reports_per_task_stability():
    summary = tutor_workflow._binary_repeat_summary([1.0, 0.0, 1.0, 0.0])

    assert summary["mean"] == pytest.approx(0.5)
    assert summary["variance"] == pytest.approx(0.25)
    assert summary["std"] == pytest.approx(0.5)
    assert summary["agreement"] == pytest.approx(1.0 / 3.0)
    assert summary["disagreement"] == pytest.approx(2.0 / 3.0)
    assert summary["all_equal"] == pytest.approx(0.0)
    assert summary["any_success"] == pytest.approx(1.0)
    assert summary["all_success"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 0.0, 0.0], 1.0 / 3.0),
        ([1.0, 0.0, 1.0, 0.0], 1.0 / 3.0),
        ([1.0, 1.0, 1.0, 1.0], 0.0),
    ],
)
def test_eval_repeat_sample_variance_supports_any_repeat_count(
    monkeypatch, values, expected
):
    """Test sample variance remains comparable across repeat counts."""
    calls = []
    monkeypatch.setattr(tutor_workflow, "_safe_scalar", lambda **x: calls.append(x))

    tutor_workflow.TutorAgentWorkflow._log_eval_repeat_metrics(values)

    metrics = _metrics_by_key(calls)
    assert metrics["repeat/final_correct/mean_task_sample_variance"] == pytest.approx(
        [expected]
    )


def test_eval_repeat_metrics_wait_for_all_outcomes_and_report_core_metrics(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(tutor_workflow, "_safe_scalar", lambda **x: calls.append(x))
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=True, task_id=7),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.eval_repeat_count = 4
    workflow._eval_repeat_outcomes = {}

    for final_correct in (True, False, True):
        completed = workflow._record_eval_repeat_outcomes(final_correct=final_correct)
        assert completed is None
    assert calls == []

    completed = workflow._record_eval_repeat_outcomes(final_correct=False)

    values = _metrics_by_key(calls)
    assert values["repeat/final_correct/mean_task_sample_variance"] == pytest.approx(
        [1.0 / 3.0]
    )
    pairwise_jaccards = values["repeat/final_correct/pairwise_success_jaccard"]
    assert len(pairwise_jaccards) == 5
    assert sum(pairwise_jaccards) / len(pairwise_jaccards) == pytest.approx(0.2)
    assert set(values) == {
        "repeat/final_correct/mean_task_sample_variance",
        "repeat/final_correct/pairwise_success_jaccard",
    }
    assert workflow._eval_repeat_outcomes == {}
    assert completed == (7, [1.0, 0.0, 1.0, 0.0])


def test_single_eval_repeat_reports_zero_sample_variance(monkeypatch):
    calls = []
    monkeypatch.setattr(tutor_workflow, "_safe_scalar", lambda **x: calls.append(x))
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=True, task_id=11),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.eval_repeat_count = 1
    workflow._eval_repeat_outcomes = {}

    completed = workflow._record_eval_repeat_outcomes(final_correct=True)

    values = _metrics_by_key(calls)
    assert values["repeat/final_correct/mean_task_sample_variance"] == pytest.approx(
        [0.0]
    )
    assert values["repeat/final_correct/pairwise_success_jaccard"] == pytest.approx(
        [1.0]
    )
    assert completed == (11, [1.0])


@pytest.mark.asyncio
async def test_eval_repeat_outcomes_dump_compact_task_record(monkeypatch, tmp_path):
    """Test completed repeats are persisted without prompt or response text."""
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: SimpleNamespace(is_eval=True, task_id=7, lora_version=20),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.debug_trace_dir = str(tmp_path)

    await workflow._dump_eval_repeat_outcomes(7, [1.0, 0.0, 1.0])

    shard_paths = list((tmp_path / "eval" / "repeat_outcomes").glob("*.jsonl"))
    assert len(shard_paths) == 1
    payload = json.loads(shard_paths[0].read_text(encoding="utf-8"))
    assert payload == {
        "task_id": 7,
        "lora_version": 20,
        "final_correct": [1, 0, 1],
    }
