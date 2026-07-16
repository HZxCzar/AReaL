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


def test_eval_repeat_metrics_wait_for_all_outcomes_and_report_jaccards(monkeypatch):
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

    for solved in (True, False, True):
        workflow._record_eval_repeat_outcomes(
            solved=solved,
            final_correct=solved,
        )
    assert calls == []

    workflow._record_eval_repeat_outcomes(solved=False, final_correct=False)

    values = _metrics_by_key(calls)
    assert values["repeat/solved/variance"] == pytest.approx([0.25])
    assert values["repeat/solved/agreement"] == pytest.approx([1.0 / 3.0])
    assert values["repeat/solved/success_set_jaccard"] == pytest.approx([0.0])
    pairwise_jaccards = values["repeat/solved/pairwise_success_set_jaccard"]
    assert len(pairwise_jaccards) == 5
    assert sum(pairwise_jaccards) / len(pairwise_jaccards) == pytest.approx(0.2)
    assert workflow._eval_repeat_outcomes == {}


def test_single_eval_repeat_reports_degenerate_stability_metrics(monkeypatch):
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

    workflow._record_eval_repeat_outcomes(solved=True, final_correct=True)

    values = _metrics_by_key(calls)
    assert values["repeat/solved/variance"] == pytest.approx([0.0])
    assert values["repeat/solved/std"] == pytest.approx([0.0])
    assert values["repeat/solved/agreement"] == pytest.approx([1.0])
    assert values["repeat/solved/disagreement"] == pytest.approx([0.0])
    assert values["repeat/solved/all_equal"] == pytest.approx([1.0])
    assert values["repeat/solved/success_set_jaccard"] == pytest.approx([1.0])
    assert values["repeat/solved/pairwise_success_set_jaccard"] == pytest.approx([1.0])
