from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from examples.tutor import workflow as workflow_module
from examples.tutor.configs import TUTOR_TRAIN_STUDENT_FIELD
from examples.tutor.workflow import TutorAgentWorkflow

from areal.api import RolloutWorkflow
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_executor import WorkflowExecutor


class _BatchHookWorkflow(RolloutWorkflow):
    def prepare_rollout_batch(
        self, data: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        return [{**item, "prepared": True} for item in data]

    async def arun_episode(self, engine, data):
        raise AssertionError("this test only exercises batch preparation")


def _sampler(*, weights: list[float] | None = None) -> TutorAgentWorkflow:
    weights = weights or [1.0] * 8
    workflow = object.__new__(TutorAgentWorkflow)
    workflow.student_sampling_strategy = "stratified"
    workflow._student_model_configs = [
        {"name": f"student-{index}", "weight": weight}
        for index, weight in enumerate(weights)
    ]
    workflow._stratified_student_scores = {
        config["name"]: 0.0
        for config in workflow._student_model_configs
        if config["weight"] > 0.0
    }
    workflow._stratified_student_batch_index = 0
    workflow.prompt_pool_seed = 42
    return workflow


def test_equal_students_are_exactly_balanced_in_divisible_batch() -> None:
    workflow = _sampler()
    original = [{"id": str(index)} for index in range(16)]

    prepared = workflow.prepare_rollout_batch(original)

    counts = Counter(item[TUTOR_TRAIN_STUDENT_FIELD] for item in prepared)
    assert counts == {f"student-{index}": 2 for index in range(8)}
    assert all(TUTOR_TRAIN_STUDENT_FIELD not in item for item in original)


def test_uneven_remainder_rotates_between_batches() -> None:
    workflow = _sampler()

    first = workflow.prepare_rollout_batch([{"id": str(i)} for i in range(5)])
    second = workflow.prepare_rollout_batch([{"id": str(i)} for i in range(5, 8)])

    names = [item[TUTOR_TRAIN_STUDENT_FIELD] for item in first + second]
    assert Counter(names) == {f"student-{index}": 1 for index in range(8)}


def test_stratification_follows_declared_weights() -> None:
    workflow = _sampler(weights=[1.0, 3.0])

    prepared = workflow.prepare_rollout_batch([{"id": str(i)} for i in range(8)])

    assert Counter(item[TUTOR_TRAIN_STUDENT_FIELD] for item in prepared) == {
        "student-0": 2,
        "student-1": 6,
    }


def test_weighted_random_default_is_identity() -> None:
    workflow = _sampler()
    workflow.student_sampling_strategy = "weighted_random"
    batch = [{"id": "one"}]

    assert workflow.prepare_rollout_batch(batch) is batch


def test_reserved_training_field_is_rejected() -> None:
    workflow = _sampler()

    with pytest.raises(ValueError, match="reserved field"):
        workflow.prepare_rollout_batch(
            [{"id": "one", TUTOR_TRAIN_STUDENT_FIELD: "student-0"}]
        )


def test_prepared_assignment_forces_group_student() -> None:
    workflow = _sampler()
    runtimes = {}
    for config in workflow._student_model_configs:
        name = config["name"]
        runtimes[name] = SimpleNamespace(
            name=name,
            model="model",
            caller=object(),
            confidence_caller=None,
            mask={"mode": "full"},
            mode="text",
            weight=config["weight"],
        )
    workflow.student_model_runtimes = runtimes

    selected = workflow._select_student(
        {TUTOR_TRAIN_STUDENT_FIELD: "student-3"},
        aux_caller=object(),
        group_key="problem",
        rollout_version=1,
    )

    assert selected.name == "student-3"


def test_selected_student_gets_retest_improvement_metrics(monkeypatch) -> None:
    workflow = object.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.student_generalize_enabled = False
    workflow.student_model_runtimes = {
        "student-a": object(),
        "student-b": object(),
    }
    captured: dict[str, float] = {}
    monkeypatch.setattr(
        workflow_module,
        "workflow_context",
        SimpleNamespace(get=lambda: SimpleNamespace(is_eval=False)),
    )
    monkeypatch.setattr(
        workflow_module,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    result = SimpleNamespace(
        level="original",
        skipped=False,
        attempted=True,
        replay_correct=3,
        replay_count=4,
        judge_result=None,
    )

    workflow._log_rollout_stats(
        total_reward=1.0,
        traces=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        student_generalization_results=[result],
        student_name="student-a",
        no_teaching_baseline=0.25,
    )

    assert captured["retest/improvement"] == 0.5
    assert captured["student/student-a/retest/no_teaching_baseline"] == 0.25
    assert captured["student/student-a/retest/improvement"] == 0.5
    assert captured["student/student-a/retest/improved"] == 1.0
    assert captured["student/student-a/retest/made_it_worse"] == 0.0
    assert "student/student-b/retest/improvement" not in captured


def test_executor_batch_hook_reaches_workflow_through_group_wrapper() -> None:
    grouped = GroupedRolloutWorkflow(
        _BatchHookWorkflow(), group_size=8, logger=SimpleNamespace()
    )

    prepared = WorkflowExecutor._prepare_rollout_batch([{"id": "one"}], grouped)

    assert prepared == [{"id": "one", "prepared": True}]
