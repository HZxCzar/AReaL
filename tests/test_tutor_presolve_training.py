"""Fixed-count pre-solve RL is opt-in and does not train shared drafts twice."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from examples.tutor.configs import TutorTeacherPreConfig
from examples.tutor.core.types import JudgeResult
from examples.tutor.workflow import TutorAgentWorkflow


def _workflow(correct, train=True):
    workflow = object.__new__(TutorAgentWorkflow)
    workflow.teacher_pre_train = train
    workflow.teacher_pre_attempts = 4
    workflow.teacher_pre_verify = True
    workflow.teacher_pre_mode = "filter_solver"
    workflow.teacher_pre_max_tokens = 4096
    workflow.teacher_pre_share_per_group = True
    workflow._teacher_pre_solve_shared = {}
    workflow._teacher_pre_solve_shared_lock = asyncio.Lock()
    workflow._build_teacher_pre_solve_messages = lambda **kwargs: [
        {"role": "user", "content": "solve"}
    ]
    workflow._presolve_training_active = lambda: train
    calls = []

    async def generate(messages, **kwargs):
        i = len(calls)
        calls.append((messages, kwargs))
        response = SimpleNamespace(
            input_tokens=[10],
            output_tokens=[20 + i, 2],
            output_logprobs=[-0.2, -0.3],
            output_versions=[7, 7],
            stop_reason="stop",
        )
        return SimpleNamespace(response=response, raw_text=str(i))

    async def score(task, truth, text, **kwargs):
        return JudgeResult("", correct[int(text)], "", None, {})

    workflow._score_answer_async = score
    return workflow, SimpleNamespace(generate=generate), calls


def _run(workflow, caller):
    return asyncio.run(
        workflow._run_teacher_pre_solve(
            "task",
            "answer",
            actor_caller=caller,
            answer_judge_caller=None,
            lora_version=7,
        )
    )


@pytest.mark.parametrize(
    "correct",
    [
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, True],
        [True] * 4,
        [False] * 4,
    ],
)
def test_fixed_candidates_and_binary_rloo(correct):
    workflow, caller, calls = _workflow(correct)
    result = _run(workflow, caller)
    assert len(calls) == 4
    assert all(c[0] == calls[0][0] for c in calls)
    assert all(c[1]["lora_version"] == 7 for c in calls)
    assert result.accepted == any(correct)
    assert result.raw_output == (str(correct.index(True)) if any(correct) else "")
    rows = workflow._presolve_training_rows(result, SimpleNamespace(name="student"))
    expected = torch.tensor([float(r) - (sum(correct) - r) / 3 for r in correct])
    torch.testing.assert_close(
        torch.cat([r["presolve_advantage"] for r in rows]), expected
    )
    assert len({r["trajectory_id"].item() for r in rows}) == 4
    assert all(r["loss_mask"].tolist() == [[0, 1, 1]] for r in rows)
    assert all(r["presolve_mask"].item() for r in rows)


def test_disabled_keeps_first_success_retry_and_no_responses():
    workflow, caller, calls = _workflow([False, True, False, False], train=False)
    result = _run(workflow, caller)
    assert len(calls) == 2
    assert result.raw_output == "1"
    assert all(a.response is None for a in result.attempts)


def test_eight_siblings_share_one_fixed_sampling_group():
    workflow, caller, calls = _workflow([True, False, True, False])

    async def run():
        return await asyncio.gather(
            *[
                workflow._teacher_pre_solve_for_group(
                    "task",
                    "answer",
                    actor_caller=caller,
                    answer_judge_caller=None,
                    lora_version=7,
                    group_key="problem",
                )
                for _ in range(8)
            ]
        )

    results = asyncio.run(run())
    assert len(calls) == 4
    assert sum(hit for _, hit in results) == 7
    assert all(result is results[0][0] for result, _ in results)
    rows = [
        workflow._presolve_training_rows(
            result, SimpleNamespace(name="student"), group_index=i
        )
        for i, (result, _) in enumerate(results)
    ]
    assert [len(r) for r in rows] == [4, 0, 0, 0, 0, 0, 0, 0]


def test_all_wrong_owner_exports_rows_even_without_teaching():
    workflow, caller, _ = _workflow([False] * 4)
    result = _run(workflow, caller)
    rows = workflow._presolve_training_rows(result, SimpleNamespace(name="student"))
    batch = workflow._finish_training_rows([], rows)
    assert batch["presolve_mask"].tolist() == [True] * 4
    assert batch["presolve_advantage"].count_nonzero() == 0
    assert workflow._finish_training_rows([], []) is None


def test_reward_v4_rows_have_matching_metadata_for_concatenation():
    from examples.tutor.core.tensors import response_to_tensordict

    workflow, caller, _ = _workflow([True, False, True, False])
    workflow.student_model_runtimes = {"student": None}
    workflow.turn_local_reward_components = ("format_error", "soft_overlong")
    workflow.student_generalize_gate_pass_credit_only = True
    result = _run(workflow, caller)
    solve_rows = workflow._presolve_training_rows(
        result, SimpleNamespace(name="student")
    )
    teaching = response_to_tensordict(
        result.attempts[0].response,
        reward=0.5,
        local_reward=-0.1,
        local_reward_by_placement={"post_std": -0.1},
        gate_masked_reward=0.5,
        gate_credit_mask=True,
        student_environment_id=0,
        student_environment_names={"student": 0, "presolve": 1},
        trajectory_id=123,
        turn_idx=1,
    )
    batch = workflow._finish_training_rows([teaching], solve_rows)
    assert batch["presolve_mask"].tolist() == [False, True, True, True, True]
    torch.testing.assert_close(
        batch["local_rewards_post_std"], torch.tensor([-0.1, 0, 0, 0, 0])
    )
    assert batch["gate_credit_mask"].tolist() == [True, False, False, False, False]


def test_generation_failure_is_not_a_negative_reward():
    workflow, caller, _ = _workflow([True] * 4)

    async def broken(*args, **kwargs):
        raise RuntimeError("server unavailable")

    caller.generate = broken
    with pytest.raises(RuntimeError, match="server unavailable"):
        _run(workflow, caller)


def test_train_switch_defaults_off_and_rejects_invalid_modes():
    assert not TutorTeacherPreConfig().train
    with pytest.raises(ValueError, match="requires"):
        TutorTeacherPreConfig(train=True)
    assert TutorTeacherPreConfig(train=True, enabled=True, attempts=4).train
