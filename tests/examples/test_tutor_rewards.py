from __future__ import annotations

import asyncio

import pytest

from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.rewards import EpisodeRewardComputer
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PublicHistoryState,
    StudentTurnState,
    TurnArtifact,
    TutorPrivateFeedback,
    TutorTurnState,
)


def _judge(correct: bool) -> JudgeResult:
    return JudgeResult(
        raw_output="",
        correct=correct,
        feedback="ok" if correct else "no",
        parse_error=None,
        raw_result={},
    )


def _leak(leaked: bool) -> LeakCheckResult:
    return LeakCheckResult(
        raw_output="",
        leaked=leaked,
        feedback="leaked" if leaked else "ok",
        parse_error=None,
        raw_result={},
    )


def _turn(
    turn_idx: int,
    *,
    max_turns: int = 10,
    leaked: bool = False,
    correct: bool = False,
    tutor_output: str = "hint",
) -> TurnArtifact:
    tutor_state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState(),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=turn_idx,
        max_turns=max_turns,
    )
    return TurnArtifact(
        turn_idx=turn_idx,
        tutor_state=tutor_state,
        tutor_prompt="prompt",
        tutor_response=None,
        tutor_raw_output=tutor_output,
        tutor_visible_output=tutor_output,
        leak_result=_leak(leaked),
        public_history_before="",
        public_history_after="",
        student_state=StudentTurnState(
            task="task",
            public_history=PublicHistoryState(),
            previous_student_output="prev",
            latest_tutor_visible_output=tutor_output,
        )
        if not leaked
        else None,
        student_prompt="student prompt" if not leaked else "",
        student_output="student answer" if not leaked else "",
        judge_result=_judge(correct) if not leaked else None,
    )


def _episode(turns: list[TurnArtifact], termination_reason: str) -> EpisodeArtifact:
    return EpisodeArtifact(
        task="task",
        ground_truth="42",
        initial_student_answer="initial",
        initial_student_error=None,
        initial_judge_result=_judge(False),
        turns=turns,
        termination_reason=termination_reason,
        pre_success=False,
        leak_count=sum(1 for turn in turns if turn.leak_result.leaked),
        latest_student_answer="latest",
    )


def _outcome_computer(**overrides) -> EpisodeRewardComputer:
    params = {
        "success_reward": 1.0,
        "leak_penalty": -1.0,
        "outcome_prior_turn_weight": 0.1,
        "outcome_credit_gamma": 0.9,
        "early_success_bonus": 0.3,
        "turn_penalty": -0.01,
        "length_penalty_threshold_chars": 1200,
        "length_penalty_per_100_chars": -0.005,
        "length_penalty_min": -0.1,
    }
    params.update(overrides)
    return EpisodeRewardComputer(**params)


def test_default_success_reward_goes_to_success_turn_only():
    turns = [_turn(1), _turn(2), _turn(3, correct=True)]
    assignments = asyncio.run(
        _outcome_computer().compute(_episode(turns, termination_reason="success"))
    )

    expected_budget = 1.0 + 0.3 * (10 - 3) / (10 - 1)
    assert assignments[0].reward_components == {}
    assert assignments[1].reward_components == {}
    assert assignments[2].reward_components["success_credit"] == pytest.approx(
        expected_budget
    )
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [0.0, 0.0, expected_budget]
    )


def test_assign_success_reward_distributes_credit_from_final_outcome():
    turns = [_turn(1), _turn(2), _turn(3, correct=True)]
    assignments = asyncio.run(
        _outcome_computer(assign_success_reward=True).compute(
            _episode(turns, termination_reason="success")
        )
    )

    success_credits = [
        assignment.reward_components["success_credit"] for assignment in assignments
    ]
    expected_budget = 1.0 + 0.3 * (10 - 3) / (10 - 1)
    assert sum(success_credits) == pytest.approx(expected_budget)
    assert success_credits[2] > success_credits[1] > success_credits[0]


def test_leak_turn_gets_no_success_credit():
    turns = [_turn(1, leaked=True), _turn(2, correct=True)]
    assignments = asyncio.run(
        _outcome_computer().compute(_episode(turns, termination_reason="success"))
    )

    assert "success_credit" not in assignments[0].reward_components
    assert assignments[0].reward_components == {"leak": -1.0}
    assert assignments[0].reward == pytest.approx(-1.0)
    assert assignments[1].reward_components["success_credit"] > 0.0


def test_failed_episode_has_no_positive_success_credit_or_turn_penalty_by_default():
    turns = [_turn(1), _turn(2)]
    assignments = asyncio.run(
        _outcome_computer().compute(_episode(turns, termination_reason="max_turns"))
    )

    assert [item.reward_components for item in assignments] == [{}, {}]
    assert [item.reward for item in assignments] == pytest.approx([0.0, 0.0])


def test_turn_penalty_applies_only_when_enabled():
    turns = [_turn(1), _turn(2)]
    assignments = asyncio.run(
        _outcome_computer(enable_turn_penalty=True).compute(
            _episode(turns, termination_reason="max_turns")
        )
    )

    assert all(
        assignment.reward_components == {"turn_penalty": -0.01}
        for assignment in assignments
    )
    assert [item.reward for item in assignments] == pytest.approx([-0.01, -0.01])


def test_length_penalty_is_capped():
    turns = [_turn(1, tutor_output="x" * 1000)]
    assignments = asyncio.run(
        _outcome_computer(
            turn_penalty=0.0,
            length_penalty_threshold_chars=10,
            length_penalty_per_100_chars=-0.5,
            length_penalty_min=-0.1,
        ).compute(_episode(turns, termination_reason="max_turns"))
    )

    assert assignments[0].reward_components == {"length_penalty": -0.1}
    assert assignments[0].reward == pytest.approx(-0.1)


def test_pairwise_can_skip_judge_when_both_answers_are_incorrect():
    class RewardCaller:
        calls = 0

        async def call_text(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("pairwise judge should not be called")

    async def generate_reference_tutor(tutor_state, reference_version):
        del tutor_state, reference_version
        return "reference hint"

    async def run_student(state):
        del state
        return "reference student answer", None

    async def run_leak_check(task, ground_truth, teacher_action):
        del task, ground_truth, teacher_action
        return _leak(False)

    def score_answer(task, ground_truth, student_output):
        del task, ground_truth, student_output
        return _judge(False)

    reward_caller = RewardCaller()
    evaluator = PairwiseTutorEvaluator(
        reward_scale=0.05,
        reward_caller=reward_caller,
        generate_reference_tutor=generate_reference_tutor,
        run_student=run_student,
        run_leak_check=run_leak_check,
        score_answer=score_answer,
        judge_both_incorrect=False,
    )
    turn = _turn(1, correct=False)
    result = asyncio.run(
        evaluator.evaluate_turn(
            _episode([turn], termination_reason="max_turns"),
            turn,
            reference_version=0,
        )
    )

    assert result.outcome == "tie"
    assert result.reason == "both_incorrect"
    assert result.reward == pytest.approx(0.0)
    assert reward_caller.calls == 0
