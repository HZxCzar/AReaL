"""Unit tests for dialogue-variant/original-retest dataset rows."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    PublicHistoryState,
)
from examples.tutor.workflow import (
    ORIGINAL_RETEST_LEVEL,
    StudentGeneralizationAnchor,
    TutorAgentWorkflow,
    _retest_problem,
)

VARIANT_TASK = "What is 8 + 7?"
VARIANT_ANSWER = "15"
ORIGINAL_TASK = "What is 3 + 4?"
ORIGINAL_ANSWER = "7"


def make_workflow(**overrides: object) -> TutorAgentWorkflow:
    workflow = object.__new__(TutorAgentWorkflow)
    attrs = {
        "free_chat_enabled": True,
        "student_generalize_retest_original": True,
        "student_generalize_retest_reward": 1.0,
        "student_generalize_level1_enabled": False,
        "student_generalize_level2_enabled": False,
        "student_generalize_bank": {},
        "student_generalize_source": "sidecar",
        "student_generalize_level_rewards": {},
        "eval_preleak_retest": False,
    }
    attrs.update(overrides)
    for name, value in attrs.items():
        setattr(workflow, name, value)
    return workflow


def row_with_retest() -> dict[str, str]:
    return {
        "id": "train-example",
        "task": VARIANT_TASK,
        "ground_truth": VARIANT_ANSWER,
        "retest_task": ORIGINAL_TASK,
        "retest_ground_truth": ORIGINAL_ANSWER,
    }


def episode() -> EpisodeArtifact:
    return EpisodeArtifact(
        task=VARIANT_TASK,
        ground_truth=VARIANT_ANSWER,
        initial_student_answer="",
        initial_student_error=None,
        initial_judge_result=JudgeResult(
            raw_output="",
            correct=False,
            feedback="",
            parse_error=None,
            raw_result={},
        ),
        turns=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        latest_student_answer="",
    )


def anchor() -> StudentGeneralizationAnchor:
    return StudentGeneralizationAnchor(
        public_history=PublicHistoryState(
            summary="",
            turn_count=1,
            turns=[
                {"role": "teacher", "content": "Let us compare two examples."},
                {"role": "student", "content": "I see the pattern."},
            ],
        ),
        previous_student_output="I see the pattern.",
        teacher_feedback="Let us compare two examples.",
        reward_turn_idx=1,
    )


def test_retest_case_uses_designated_original_problem() -> None:
    case = make_workflow()._student_generalization_cases(row_with_retest())[
        ORIGINAL_RETEST_LEVEL
    ]
    assert case.task == ORIGINAL_TASK
    assert case.ground_truth == ORIGINAL_ANSWER
    assert case.reward == 1.0


def test_retest_prompt_uses_original_not_dialogue_variant() -> None:
    messages = make_workflow()._build_student_probe_messages(
        episode_artifact=episode(),
        anchor=anchor(),
        level=ORIGINAL_RETEST_LEVEL,
        transfer_task=ORIGINAL_TASK,
    )
    final_turn = messages[-1]["content"]
    assert ORIGINAL_TASK in final_turn
    assert VARIANT_TASK not in final_turn


def test_legacy_rows_fall_back_to_task_and_ground_truth() -> None:
    legacy = {"task": VARIANT_TASK, "ground_truth": VARIANT_ANSWER}
    assert _retest_problem(legacy) == (VARIANT_TASK, VARIANT_ANSWER)
    case = make_workflow()._student_generalization_cases(legacy)[ORIGINAL_RETEST_LEVEL]
    assert (case.task, case.ground_truth) == (VARIANT_TASK, VARIANT_ANSWER)


@pytest.mark.parametrize(
    "partial",
    [
        {
            "task": VARIANT_TASK,
            "ground_truth": VARIANT_ANSWER,
            "retest_task": ORIGINAL_TASK,
        },
        {
            "task": VARIANT_TASK,
            "ground_truth": VARIANT_ANSWER,
            "retest_ground_truth": ORIGINAL_ANSWER,
        },
    ],
)
def test_partial_retest_mapping_is_rejected(partial: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must be set together"):
        _retest_problem(partial)


def test_no_teaching_baseline_uses_original_problem() -> None:
    workflow = make_workflow(
        free_chat_no_teaching_baseline=True,
        student_generalize_replays=1,
    )
    workflow._no_teaching_baselines = {}
    workflow._no_teaching_baseline_lock = asyncio.Lock()
    captured_messages: list[list[dict[str, str]]] = []
    captured_scores: list[tuple[str, str]] = []

    async def fake_call(messages, **_kwargs):
        captured_messages.append(messages)
        return SimpleNamespace(error=None, text="\\boxed{7}")

    async def fake_score(task, ground_truth, _answer, **_kwargs):
        captured_scores.append((task, ground_truth))
        return JudgeResult(
            raw_output="",
            correct=True,
            feedback="",
            parse_error=None,
            raw_result={},
        )

    workflow._call_auxiliary_messages = fake_call
    workflow._score_answer_async = fake_score
    score = asyncio.run(
        workflow._no_teaching_baseline(
            row_with_retest(),
            aux_caller=object(),
            answer_judge_caller=object(),
        )
    )

    assert score == 1.0
    assert ORIGINAL_TASK in captured_messages[0][-1]["content"]
    assert VARIANT_TASK not in captured_messages[0][-1]["content"]
    assert captured_scores == [(ORIGINAL_TASK, ORIGINAL_ANSWER)]
