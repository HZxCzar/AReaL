from __future__ import annotations

import asyncio
import types

import pytest

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import (
    TutorConfig,
    TutorRewardConfig,
    TutorStudentGeneralizeConfig,
)
from examples.tutor.core.callers import TextCallResult
from examples.tutor.core.math import score_math_answer
from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.parsers import parse_staged_leak_check_result
from examples.tutor.core.rewards import EpisodeRewardComputer
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PublicHistoryState,
    StudentTurnState,
    TeacherPreSolveAttempt,
    TeacherPreSolveResult,
    TurnArtifact,
    TurnTrace,
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


def _leak(
    leaked: bool,
    leak_level: int | None = None,
    *,
    feedback: str | None = None,
) -> LeakCheckResult:
    return LeakCheckResult(
        raw_output="",
        leaked=leaked,
        feedback=feedback if feedback is not None else ("leaked" if leaked else "ok"),
        parse_error=None,
        raw_result={},
        leak_level=leak_level,
    )


def _turn(
    turn_idx: int,
    *,
    max_turns: int = 10,
    leaked: bool = False,
    leak_level: int | None = None,
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
        leak_result=_leak(leaked, leak_level),
        public_history_before="",
        public_history_after="",
        student_state=StudentTurnState(
            task="task",
            public_history=PublicHistoryState(),
            previous_student_output="prev",
            latest_tutor_visible_output=tutor_output,
        ),
        student_prompt="student prompt",
        student_output="student answer",
        judge_result=_judge(correct),
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


def test_student_generalize_config_defaults_to_only_success():
    assert TutorStudentGeneralizeConfig().mode == "only_success"


def test_student_generalize_config_rejects_invalid_mode():
    with pytest.raises(ValueError, match="student_generalize.mode"):
        TutorStudentGeneralizeConfig(mode="sometimes")


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


def test_tutor_config_rejects_invalid_leak_handling_mode():
    with pytest.raises(ValueError, match="leak_handling_mode"):
        TutorConfig(dataset_type="math", leak_handling_mode="invalid")


def test_workflow_rejects_invalid_leak_handling_mode():
    with pytest.raises(ValueError, match="leak_handling_mode"):
        tutor_workflow.TutorAgentWorkflow(leak_handling_mode="invalid")


def test_teacher_pre_context_strips_hidden_thinking():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_pre_enabled = True
    teacher_pre_solve = TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=True,
        raw_output="<think>hidden final-answer reasoning</think>\nVisible solution plan.",
    )

    prompt = workflow._append_teacher_pre_solve_context(
        "Base tutor prompt.", teacher_pre_solve
    )

    assert "teacher_private_solution_draft" in prompt
    assert "Visible solution plan." in prompt
    assert "<think>" not in prompt
    assert "hidden final-answer reasoning" not in prompt


def test_teacher_pre_context_skips_empty_visible_draft():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_pre_enabled = True
    teacher_pre_solve = TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=True,
        raw_output="<think>only hidden reasoning</think>",
    )

    prompt = workflow._append_teacher_pre_solve_context(
        "Base tutor prompt.", teacher_pre_solve
    )

    assert prompt == "Base tutor prompt."


def test_tutor_reward_config_validates_leak_penalty_modes():
    config = TutorRewardConfig()

    assert config.leak_penalty_mode == "binary"
    assert config.leak_penalty == pytest.approx(-1.0)
    assert config.leaked_success_reward_scale == pytest.approx(1.0)
    assert config.leak_penalty_aggregation == "turn"

    with pytest.raises(ValueError, match="leak_penalty"):
        TutorRewardConfig(leak_penalty=None)

    with pytest.raises(ValueError, match="leaked_success_reward_scale"):
        TutorRewardConfig(leaked_success_reward_scale=-0.1)

    with pytest.raises(ValueError, match="leak_penalty_aggregation"):
        TutorRewardConfig(leak_penalty_aggregation="invalid")

    with pytest.raises(ValueError, match="leak_penalty_formula"):
        TutorRewardConfig(
            leak_penalty_mode="staged",
            leak_penalty_final_answer=-1.0,
            leak_penalty_compute=-0.5,
        )

    staged = TutorRewardConfig(
        leak_penalty_mode="staged",
        leak_penalty_final_answer=-1.0,
        leak_penalty_compute=-0.5,
        leak_penalty_formula=-0.1,
    )
    assert staged.leak_penalty_formula == pytest.approx(-0.1)

    rawbase = TutorRewardConfig(leak_penalty_mode="rawbase", leak_penalty=-0.05)
    assert rawbase.leak_penalty_mode == "rawbase"
    assert rawbase.leak_penalty == pytest.approx(-0.05)

    with pytest.raises(ValueError, match="leak_penalty"):
        TutorRewardConfig(leak_penalty_mode="rawbase", leak_penalty=None)


@pytest.mark.parametrize(
    ("level", "expected_leaked"),
    [(1, True), (2, True), (3, True), (4, False)],
)
def test_parse_staged_leak_check_result_maps_levels(level, expected_leaked):
    result = parse_staged_leak_check_result(
        f'{{"level": {level}, "feedback": "level {level}"}}'
    )

    assert result.leak_level == level
    assert result.leaked is expected_leaked
    assert result.feedback == f"level {level}"
    assert result.parse_error is None


def test_parse_staged_leak_check_result_fails_closed_to_level_one():
    invalid_level = parse_staged_leak_check_result('{"level": 5, "feedback": "bad"}')
    invalid_json = parse_staged_leak_check_result("not json")

    assert invalid_level.leaked is True
    assert invalid_level.leak_level == 1
    assert invalid_level.feedback == "Failed to parse staged leak-check output."
    assert invalid_level.parse_error is not None
    assert invalid_json.leaked is True
    assert invalid_json.leak_level == 1
    assert invalid_json.parse_error is not None


def test_workflow_staged_leak_check_uses_staged_prompt_and_parser(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_penalty_mode = "staged"
    workflow.leak_check_system_prompt = "staged system"
    captured = {}

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return TextCallResult(text='{"level": 2, "feedback": "computed"}')

    monkeypatch.setattr(workflow, "_call_auxiliary_prompt", call_auxiliary_prompt)

    result = asyncio.run(workflow._run_leak_check("task", "42", "6 * 7 = 42"))

    assert result.leaked is True
    assert result.leak_level == 2
    assert result.feedback == "computed"
    assert captured["system_prompt"] == "staged system"
    assert "Choose exactly one level" in captured["user_prompt"]
    assert "choose the most severe level" in captured["user_prompt"]
    assert '"evidence"' not in captured["user_prompt"]


def test_workflow_staged_mode_replaces_binary_default_system_prompt():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_penalty_mode = "staged"

    prompt = workflow._resolve_leak_check_system_prompt(
        tutor_workflow.DEFAULT_LEAK_CHECK_SYSTEM_PROMPT
    )

    assert prompt == tutor_workflow.DEFAULT_STAGED_LEAK_CHECK_SYSTEM_PROMPT
    assert "choose the most severe level" in prompt


def test_feedback_mode_only_adds_private_feedback_instruction():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_handling_mode = "reward_only"

    prompt = workflow._leak_check_system_prompt_for_current_mode("base prompt")
    assert prompt == "base prompt"

    workflow.leak_handling_mode = "feedback"
    prompt = workflow._leak_check_system_prompt_for_current_mode("base prompt")

    assert prompt.startswith("base prompt")
    assert tutor_workflow.FEEDBACK_LEAK_CHECK_SYSTEM_PROMPT_SUFFIX in prompt
    assert "Do not add new JSON keys" in prompt
    assert "evidence" not in prompt


def test_non_thinking_teacher_system_prompt_requires_json_output():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = False

    prompt = workflow._resolve_teacher_system_prompt("base prompt")

    assert prompt.startswith("base prompt")
    assert tutor_workflow.NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT in prompt
    assert '"reasoning"' in prompt
    assert '"output"' in prompt


def test_thinking_teacher_system_prompt_keeps_existing_format():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = True

    prompt = workflow._resolve_teacher_system_prompt("base prompt")

    assert prompt == "base prompt"


def test_non_thinking_tutor_visible_output_uses_json_output_only():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = False

    visible = workflow._extract_tutor_visible_output(
        r'{"reasoning": "secret \boxed{42}", "output": "Try isolating x first."}'
    )

    assert visible == "Try isolating x first."


def test_non_thinking_tutor_visible_output_falls_back_for_invalid_json():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = False

    visible = workflow._extract_tutor_visible_output(
        "not json <think>private</think> visible hint"
    )

    assert visible == "not json  visible hint"


def test_rawbase_leak_check_asks_llm_about_visible_output(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_handling_mode = "reward_only"
    workflow.leak_penalty_mode = "rawbase"
    captured = {}

    def fail_score_answer(*args, **kwargs):
        raise AssertionError("rawbase should not use rule-based answer extraction")

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return TextCallResult(text='{"leaked": true, "feedback": ""}')

    monkeypatch.setattr(workflow, "_score_answer", fail_score_answer)
    monkeypatch.setattr(workflow, "_call_auxiliary_prompt", call_auxiliary_prompt)

    result = asyncio.run(
        workflow._run_optional_leak_check(
            "task",
            "42",
            "The value is six times seven, so continue from there.",
        )
    )

    assert result.leaked is True
    assert result.raw_result["method"] == "rawbase_llm"
    assert captured["system_prompt"] == tutor_workflow.RAWBASE_LEAK_CHECK_SYSTEM_PROMPT
    assert "Ground Truth:\n42" in captured["user_prompt"]
    assert "The value is six times seven" in captured["user_prompt"]
    assert '"evidence"' not in captured["user_prompt"]
    assert captured["rid_prefix"] == "rawbase-leak-check"


def test_rawbase_leak_check_strips_private_reasoning_json_output(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = False
    workflow.leak_handling_mode = "reward_only"
    workflow.leak_penalty_mode = "rawbase"
    captured = {}

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return TextCallResult(text='{"leaked": false, "feedback": ""}')

    monkeypatch.setattr(workflow, "_call_auxiliary_prompt", call_auxiliary_prompt)

    result = asyncio.run(
        workflow._run_rawbase_leak_check(
            "task",
            "42",
            r'{"reasoning": "the answer is 42", "output": "Try factoring first."}',
        )
    )

    assert result.leaked is False
    assert result.raw_result["method"] == "rawbase_llm"
    assert "Try factoring first." in captured["user_prompt"]
    assert "the answer is 42" not in captured["user_prompt"]


def test_rawbase_leak_check_strips_thinking_output(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.enable_thinking = True
    workflow.leak_handling_mode = "reward_only"
    workflow.leak_penalty_mode = "rawbase"
    captured = {}

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return TextCallResult(text='{"leaked": false, "feedback": ""}')

    monkeypatch.setattr(workflow, "_call_auxiliary_prompt", call_auxiliary_prompt)

    result = asyncio.run(
        workflow._run_rawbase_leak_check(
            "task",
            "42",
            "<think>the answer is 42</think>Try factoring first.",
        )
    )

    assert result.leaked is False
    assert result.raw_result["method"] == "rawbase_llm"
    assert "Try factoring first." in captured["user_prompt"]
    assert "the answer is 42" not in captured["user_prompt"]


def test_initial_public_summary_uses_student_round_zero():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )

    assert (
        workflow._build_initial_public_summary("initial wrong answer")
        == "Student round 0:\ninitial wrong answer"
    )


def test_teacher_pre_solve_context_is_hidden_in_tutor_prompt():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_pre_enabled = True
    workflow.teacher_show_ground_truth = False
    workflow.teacher_user_prompt_template = tutor_workflow.TEACHER_STATE_USER_TEMPLATE

    result = TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=True,
        attempts=[
            TeacherPreSolveAttempt(
                attempt=1,
                raw_output="private solution with \\boxed{42}",
                error=None,
                accepted=True,
                judge_result=_judge(True),
            )
        ],
        raw_output="private solution with \\boxed{42}",
    )
    state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState("Student round 0:\nwrong", 0),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(
            kind="student_judged",
            student_output="wrong",
            judge_correct=False,
            judge_feedback="Incorrect.",
        ),
        turn_idx=1,
        max_turns=3,
        teacher_pre_solve_result=result,
    )

    prompt = workflow._build_tutor_prompt(state)

    assert "Private teacher solution draft hidden from the student" in prompt
    assert '<teacher_private_solution_draft mode="filter_solver">' in prompt
    assert "private solution with \\boxed{42}" in prompt


def test_teacher_pre_solve_retries_until_correct(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_pre_mode = "filter_solver"
    workflow.teacher_pre_attempts = 3
    workflow.teacher_pre_max_tokens = 128
    workflow.max_completion_tokens = 512

    outputs = ["wrong", "solution \\boxed{42}"]
    calls = []

    class FakeActor:
        async def generate(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return types.SimpleNamespace(raw_text=outputs[len(calls) - 1])

    async def score_answer(task, ground_truth, answer, *, answer_judge_caller):
        del task, ground_truth, answer_judge_caller
        return _judge("\\boxed{42}" in answer)

    monkeypatch.setattr(workflow, "_score_answer_async", score_answer)

    result = asyncio.run(
        workflow._run_teacher_pre_solve(
            "task",
            "42",
            actor_caller=FakeActor(),
            answer_judge_caller=None,
            lora_version=7,
        )
    )

    assert result.accepted is True
    assert result.raw_output == "solution \\boxed{42}"
    assert [attempt.accepted for attempt in result.attempts] == [False, True]
    assert len(calls) == 2
    assert calls[0][1]["max_completion_tokens"] == 128
    assert calls[0][1]["lora_version"] == 7


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


def test_leak_turn_can_receive_success_credit_with_penalty():
    turns = [_turn(1, leaked=True, correct=True)]
    assignments = asyncio.run(
        _outcome_computer().compute(_episode(turns, termination_reason="success"))
    )

    assert assignments[0].reward_components["leak"] == pytest.approx(-1.0)
    assert assignments[0].reward_components["success_credit"] > 0.0
    assert assignments[0].reward == pytest.approx(0.3)


def test_leaked_success_reward_scale_can_remove_success_credit():
    turns = [_turn(1, leaked=True), _turn(2, correct=True)]
    assignments = asyncio.run(
        _outcome_computer(leaked_success_reward_scale=0.0).compute(
            _episode(turns, termination_reason="success")
        )
    )

    assert assignments[0].reward_components == {"leak": -1.0}
    assert "success_credit" not in assignments[1].reward_components
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [-1.0, 0.0]
    )


def test_leaked_success_reward_scale_does_not_affect_clean_success():
    turns = [_turn(1), _turn(2, correct=True)]
    assignments = asyncio.run(
        _outcome_computer(leaked_success_reward_scale=0.0).compute(
            _episode(turns, termination_reason="success")
        )
    )

    expected_budget = 1.0 + 0.3 * (10 - 2) / (10 - 1)
    assert assignments[1].reward_components["success_credit"] == pytest.approx(
        expected_budget
    )


def test_episode_leak_penalty_aggregation_applies_once():
    turns = [_turn(1, leaked=True), _turn(2, leaked=True), _turn(3)]
    assignments = asyncio.run(
        _outcome_computer(leak_penalty_aggregation="episode").compute(
            _episode(turns, termination_reason="max_turns")
        )
    )

    assert assignments[0].reward_components == {"leak": -1.0}
    assert assignments[1].reward_components == {}
    assert assignments[2].reward_components == {}
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [-1.0, 0.0, 0.0]
    )


def test_staged_leak_penalty_uses_level_specific_components():
    turns = [
        _turn(1, leaked=True, leak_level=1),
        _turn(2, leaked=True, leak_level=2),
        _turn(3, leaked=True, leak_level=3),
        _turn(4, leaked=False, leak_level=4),
    ]
    assignments = asyncio.run(
        _outcome_computer(
            leak_penalty_mode="staged",
            leak_penalty_final_answer=-1.0,
            leak_penalty_compute=-0.5,
            leak_penalty_formula=-0.1,
        ).compute(_episode(turns, termination_reason="max_turns"))
    )

    assert assignments[0].reward_components == {"leak_final_answer": -1.0}
    assert assignments[1].reward_components == {"leak_compute": -0.5}
    assert assignments[2].reward_components == {"leak_formula": -0.1}
    assert assignments[3].reward_components == {}
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [-1.0, -0.5, -0.1, 0.0]
    )


def test_episode_staged_leak_penalty_uses_most_severe_component_once():
    turns = [
        _turn(1, leaked=True, leak_level=3),
        _turn(2, leaked=True, leak_level=1),
        _turn(3, leaked=True, leak_level=2),
    ]
    assignments = asyncio.run(
        _outcome_computer(
            leak_penalty_mode="staged",
            leak_penalty_final_answer=-1.0,
            leak_penalty_compute=-0.5,
            leak_penalty_formula=-0.1,
            leak_penalty_aggregation="episode",
        ).compute(_episode(turns, termination_reason="max_turns"))
    )

    assert assignments[0].reward_components == {}
    assert assignments[1].reward_components == {"leak_final_answer": -1.0}
    assert assignments[2].reward_components == {}
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [0.0, -1.0, 0.0]
    )


def test_rawbase_leak_penalty_uses_binary_component():
    turns = [_turn(1, leaked=True), _turn(2, leaked=False)]
    assignments = asyncio.run(
        _outcome_computer(
            leak_penalty_mode="rawbase",
            leak_penalty=-0.05,
        ).compute(_episode(turns, termination_reason="max_turns"))
    )

    assert assignments[0].reward_components == {"leak": -0.05}
    assert assignments[1].reward_components == {}
    assert [assignment.reward for assignment in assignments] == pytest.approx(
        [-0.05, 0.0]
    )


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


def test_optional_leak_check_disabled_returns_clean_result_without_checker(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_handling_mode = "disabled"

    async def fail_leak_check(*args, **kwargs):
        raise AssertionError("leak checker should not be called")

    monkeypatch.setattr(workflow, "_run_leak_check", fail_leak_check)

    result = asyncio.run(
        workflow._run_optional_leak_check("task", "42", "the answer is 42")
    )

    assert result.leaked is False
    assert result.feedback == "Leak check disabled."
    assert result.parse_error is None
    assert result.raw_result == {"disabled": True}


def test_pairwise_leak_check_disabled_uses_clean_reference_result(monkeypatch):
    captured = {}

    class FakePairwiseEvaluator:
        def __init__(self, **kwargs):
            self.run_leak_check = kwargs["run_leak_check"]

        async def evaluate(self, episode_artifact, *, reference_version):
            captured["reference_version"] = reference_version
            captured["leak_result"] = await self.run_leak_check(
                episode_artifact.task,
                episode_artifact.ground_truth,
                "reference answer is 42",
            )
            return []

    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.leak_handling_mode = "disabled"
    workflow.pairwise_reference_lag_steps = 2
    workflow.pairwise_reward_scale = 0.05
    workflow.pairwise_compare_all_turns = True
    workflow.pairwise_judge_both_incorrect = True

    async def fail_leak_check(*args, **kwargs):
        raise AssertionError("leak checker should not be called")

    async def unused_student(*args, **kwargs):
        raise AssertionError("student should not be called")

    monkeypatch.setattr(workflow, "_run_leak_check", fail_leak_check)
    monkeypatch.setattr(workflow, "_run_student", unused_student)
    monkeypatch.setattr(tutor_workflow, "PairwiseTutorEvaluator", FakePairwiseEvaluator)

    results = asyncio.run(
        workflow._run_pairwise_evaluation(
            _episode([_turn(1)], termination_reason="max_turns"),
            episode_lora_version=5,
            chat_caller=object(),
            aux_caller=object(),
            answer_judge_caller=None,
        )
    )

    assert results == []
    assert captured["reference_version"] == 3
    assert captured["leak_result"].leaked is False
    assert captured["leak_result"].raw_result == {"disabled": True}


class FakeAnswerJudgeCaller:
    def __init__(self, *outputs: str):
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    async def call_text(self, messages, *, rid_prefix="auxiliary"):
        self.calls.append({"messages": messages, "rid_prefix": rid_prefix})
        if not self.outputs:
            raise AssertionError("unexpected answer judge call")
        output = self.outputs.pop(0)
        return TextCallResult(text=output, raw_text=output, error=None)


def _answer_judge_workflow(*, enabled: bool = True):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.answer_scorer = score_math_answer
    workflow.answer_judge_enabled = enabled
    workflow.answer_judge_system_prompt = "judge system"
    workflow._answer_judge_cache = {}
    return workflow


def test_answer_judge_exact_match_skips_llm_call():
    workflow = _answer_judge_workflow(enabled=True)
    caller = FakeAnswerJudgeCaller('{"correct": false}')

    result = asyncio.run(
        workflow._score_answer_async(
            "task",
            r"\frac{x+5}{6}",
            r"$$\boxed{\frac{x+5}{6}}$$",
            answer_judge_caller=caller,
        )
    )

    assert result.correct is True
    assert caller.calls == []


def test_answer_judge_marks_equivalent_function_answer_correct():
    workflow = _answer_judge_workflow(enabled=True)
    caller = FakeAnswerJudgeCaller('{"correct": true}')

    result = asyncio.run(
        workflow._score_answer_async(
            "Find the inverse.",
            r"\frac{x+5}{6}",
            r"$$\boxed{h^{-1}(x)=\frac{x+5}{6}}$$",
            answer_judge_caller=caller,
        )
    )

    assert result.correct is True
    assert result.raw_result["extracted_answer"] == r"h^{-1}(x)=\frac{x+5}{6}"
    assert result.raw_result["answer_judge"]["used"] is True
    assert caller.calls[0]["rid_prefix"] == "answer-judge"
    user_prompt = caller.calls[0]["messages"][1]["content"]
    assert "Find the inverse." in user_prompt
    assert r"\frac{x+5}{6}" in user_prompt
    assert r"h^{-1}(x)=\frac{x+5}{6}" in user_prompt


def test_answer_judge_keeps_non_equivalent_answer_incorrect():
    workflow = _answer_judge_workflow(enabled=True)
    caller = FakeAnswerJudgeCaller('{"correct": false}')

    result = asyncio.run(
        workflow._score_answer_async(
            "task",
            r"\frac{x+5}{6}",
            r"$$\boxed{\frac{x-5}{6}}$$",
            answer_judge_caller=caller,
        )
    )

    assert result.correct is False
    assert result.feedback == "Incorrect."
    assert result.raw_result["answer_judge"]["correct"] is False


def test_answer_judge_parse_failure_fails_closed():
    workflow = _answer_judge_workflow(enabled=True)
    caller = FakeAnswerJudgeCaller("not json")

    result = asyncio.run(
        workflow._score_answer_async(
            "task",
            r"\frac{x+5}{6}",
            r"$$\boxed{h^{-1}(x)=\frac{x+5}{6}}$$",
            answer_judge_caller=caller,
        )
    )

    assert result.correct is False
    assert result.feedback == "Incorrect."
    assert result.raw_result["answer_judge"]["used"] is False
    assert "error" in result.raw_result["answer_judge"]


def test_answer_judge_caches_repeated_comparisons():
    workflow = _answer_judge_workflow(enabled=True)
    caller = FakeAnswerJudgeCaller('{"correct": true}')
    args = (
        "task",
        r"\frac{x+5}{6}",
        r"$$\boxed{h^{-1}(x)=\frac{x+5}{6}}$$",
    )

    first = asyncio.run(workflow._score_answer_async(*args, answer_judge_caller=caller))
    second = asyncio.run(
        workflow._score_answer_async(*args, answer_judge_caller=caller)
    )

    assert first.correct is True
    assert second.correct is True
    assert len(caller.calls) == 1


def _minimal_episode_workflow(**overrides):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    params = {
        "max_turns": 1,
        "success_reward": 1.0,
        "leak_penalty": -1.0,
        "leak_penalty_mode": "binary",
        "leak_penalty_final_answer": 0.0,
        "leak_penalty_compute": 0.0,
        "leak_penalty_formula": 0.0,
        "assign_success_reward": False,
        "outcome_prior_turn_weight": 0.1,
        "outcome_credit_gamma": 0.9,
        "early_success_bonus": 0.0,
        "enable_turn_penalty": False,
        "turn_penalty": -0.01,
        "length_penalty_threshold_chars": 0,
        "length_penalty_per_100_chars": 0.0,
        "length_penalty_min": 0.0,
        "pairwise_reward_enabled": False,
        "leak_handling_mode": "reward_only",
    }
    params.update(overrides)
    for name, value in params.items():
        setattr(workflow, name, value)
    return workflow


def test_workflow_teacher_pre_solve_failure_skips_sample(monkeypatch):
    workflow = _minimal_episode_workflow(
        teacher_pre_enabled=True,
        student_generalize_enabled=True,
        student_generalize_mode="always",
    )
    stats = {}
    pre_result = TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=False,
        attempts=[
            TeacherPreSolveAttempt(
                attempt=1,
                raw_output="wrong",
                error=None,
                accepted=False,
                judge_result=_judge(False),
            )
        ],
        error="no correct teacher pre-solve after 1 attempts",
    )

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._log_rollout_stats = lambda **kwargs: stats.update(kwargs)
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def run_teacher_pre_solve(*_args, **_kwargs):
        return pre_result

    async def fail_run_student(*_args, **_kwargs):
        raise AssertionError("student should not run after failed teacher pre-solve")

    async def fail_generalization(*_args, **_kwargs):
        raise AssertionError("generalization should not run after teacher pre-skip")

    monkeypatch.setattr(workflow, "_run_teacher_pre_solve", run_teacher_pre_solve)
    monkeypatch.setattr(workflow, "_run_student", fail_run_student)
    monkeypatch.setattr(workflow, "_run_student_generalization", fail_generalization)

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is None
    assert workflow.last_teacher_pre_solve_result is pre_result
    assert workflow.last_history == []
    assert workflow.last_traces == []
    assert workflow.last_total_reward == pytest.approx(0.0)
    assert stats["termination_reason"] == "pre_solve_skipped"
    assert stats["teacher_pre_solve_result"] is pre_result


def test_workflow_pre_solved_sample_skips_always_generalization(monkeypatch):
    workflow = _minimal_episode_workflow(
        student_generalize_enabled=True,
        student_generalize_mode="always",
    )
    stats = {}
    student_calls = []

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._log_rollout_stats = lambda **kwargs: stats.update(kwargs)
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        return "already solved", None

    async def score_answer(*_args, **_kwargs):
        return _judge(True)

    async def fail_generalization(*_args, **_kwargs):
        raise AssertionError("generalization should not run after pre-solved sample")

    monkeypatch.setattr(workflow, "_run_student", run_student)
    monkeypatch.setattr(workflow, "_score_answer_async", score_answer)
    monkeypatch.setattr(workflow, "_run_student_generalization", fail_generalization)

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is None
    assert len(student_calls) == 1
    assert workflow.last_student_generalization_results == []
    assert workflow.last_total_reward == pytest.approx(0.0)
    assert stats["termination_reason"] == "pre_solved"
    assert stats["pre_success"] is True


def test_workflow_reward_only_leak_does_not_skip_student_or_success(monkeypatch):
    workflow = _minimal_episode_workflow()

    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[0],
        input_len=1,
        output_len=1,
    )
    student_calls = []
    leak_calls = []
    judge_results = [_judge(False), _judge(True)]

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._build_tutor_prompt = lambda state: f"tutor prompt {state.turn_idx}"
    workflow._build_student_prompt_from_state = lambda state: "student prompt"
    workflow._log_rollout_stats = lambda **_kwargs: None
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def generate_tutor_response(*_args, **_kwargs):
        return response, "the answer is 42"

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        if len(student_calls) == 1:
            return "initial wrong answer", None
        return "student copies 42", None

    async def score_answer(*_args, **_kwargs):
        return judge_results.pop(0)

    async def run_leak_check(task, ground_truth, teacher_action, **_kwargs):
        leak_calls.append((task, ground_truth, teacher_action))
        return _leak(True)

    async def update_history(**_kwargs):
        return PublicHistoryState(summary="teacher and student kept", turn_count=1)

    workflow._generate_tutor_response = generate_tutor_response
    workflow._run_student = run_student
    workflow._score_answer_async = score_answer
    workflow._run_optional_leak_check = run_leak_check
    workflow._run_public_summary_update = update_history

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is not None
    assert len(student_calls) == 2
    assert len(leak_calls) == 1
    assert leak_calls[0][2] == "the answer is 42"
    assert workflow.last_traces[0].leaked is True
    assert workflow.last_traces[0].judge_correct is True
    assert workflow.last_traces[0].student_output == "student copies 42"
    assert workflow.last_traces[0].public_history_after == "teacher and student kept"
    assert workflow.last_traces[0].reward_components["leak"] == pytest.approx(-1.0)
    assert workflow.last_traces[0].reward_components["success_credit"] == pytest.approx(
        1.0
    )
    assert workflow.last_total_reward == pytest.approx(0.0)


def test_workflow_leak_termination_skips_student_and_assigns_penalty(monkeypatch):
    workflow = _minimal_episode_workflow(leak_handling_mode="terminate")

    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[0],
        input_len=1,
        output_len=1,
    )
    student_calls = []
    score_calls = []
    leak_calls = []
    stats = {}

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._build_tutor_prompt = lambda state: f"tutor prompt {state.turn_idx}"
    workflow._build_student_prompt_from_state = lambda state: "student prompt"
    workflow._log_rollout_stats = lambda **kwargs: stats.update(kwargs)
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def generate_tutor_response(*_args, **_kwargs):
        return response, "the answer is 42"

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        return "initial wrong answer", None

    async def score_answer(*_args, **_kwargs):
        score_calls.append(_args)
        return _judge(False)

    async def run_leak_check(task, ground_truth, teacher_action, **_kwargs):
        leak_calls.append((task, ground_truth, teacher_action))
        return _leak(True)

    async def fail_update_history(**_kwargs):
        raise AssertionError("history should not update after leak termination")

    async def fail_annotate(*_args, **_kwargs):
        raise AssertionError("leak results should already be annotated")

    workflow._generate_tutor_response = generate_tutor_response
    workflow._run_student = run_student
    workflow._score_answer_async = score_answer
    workflow._run_optional_leak_check = run_leak_check
    workflow._run_public_summary_update = fail_update_history
    workflow._annotate_turn_leak_results = fail_annotate

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is not None
    assert len(student_calls) == 1
    assert len(score_calls) == 1
    assert len(leak_calls) == 1
    assert leak_calls[0][2] == "the answer is 42"
    assert stats["termination_reason"] == "leak"
    assert stats["leak_count"] == 1
    assert workflow.last_traces[0].leaked is True
    assert workflow.last_traces[0].judge_correct is False
    assert workflow.last_traces[0].student_output == ""
    assert (
        workflow.last_traces[0].public_history_before
        == "Student round 0:\ninitial wrong answer"
    )
    assert (
        workflow.last_traces[0].public_history_after
        == workflow.last_traces[0].public_history_before
    )
    assert workflow.last_traces[0].reward_components == {"leak": -1.0}
    assert workflow.last_total_reward == pytest.approx(-1.0)


def test_workflow_leak_termination_reuses_immediate_clean_checks(monkeypatch):
    workflow = _minimal_episode_workflow(max_turns=2, leak_handling_mode="terminate")

    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[0],
        input_len=1,
        output_len=1,
    )
    student_calls = []
    leak_results = [_leak(False), _leak(True)]
    tutor_outputs = ["first hint", "the answer is 42"]

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._build_tutor_prompt = lambda state: f"tutor prompt {state.turn_idx}"
    workflow._build_student_prompt_from_state = lambda state: "student prompt"
    workflow._log_rollout_stats = lambda **_kwargs: None
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def generate_tutor_response(*_args, **_kwargs):
        return response, tutor_outputs.pop(0)

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        if len(student_calls) == 1:
            return "initial wrong answer", None
        return "still wrong", None

    async def score_answer(*_args, **_kwargs):
        return _judge(False)

    async def run_leak_check(*_args, **_kwargs):
        return leak_results.pop(0)

    async def update_history(**_kwargs):
        return PublicHistoryState(summary="first turn kept", turn_count=1)

    async def fail_annotate(*_args, **_kwargs):
        raise AssertionError("immediate leak checks should not be rerun post-hoc")

    workflow._generate_tutor_response = generate_tutor_response
    workflow._run_student = run_student
    workflow._score_answer_async = score_answer
    workflow._run_optional_leak_check = run_leak_check
    workflow._run_public_summary_update = update_history
    workflow._annotate_turn_leak_results = fail_annotate

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is not None
    assert len(student_calls) == 2
    assert [trace.leaked for trace in workflow.last_traces] == [False, True]
    assert workflow.last_traces[0].student_output == "still wrong"
    assert workflow.last_traces[0].public_history_after == "first turn kept"
    assert workflow.last_traces[1].student_output == ""
    assert workflow.last_traces[1].public_history_before == "first turn kept"
    assert workflow.last_traces[1].public_history_after == "first turn kept"
    assert workflow.last_traces[1].reward_components == {"leak": -1.0}
    assert workflow.last_total_reward == pytest.approx(-1.0)


def test_workflow_feedback_mode_invalidates_leaked_student_success_and_persists_private_history(
    monkeypatch,
):
    workflow = _minimal_episode_workflow(max_turns=3, leak_handling_mode="feedback")

    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[0],
        input_len=1,
        output_len=1,
    )
    student_calls = []
    tutor_states = []
    update_calls = []
    stats = {}
    leak_results = [
        _leak(True, feedback="The tutor directly stated the final answer."),
        _leak(True, feedback="The tutor computed the final value for the student."),
        _leak(False),
    ]
    tutor_outputs = ["the answer is 42", "42 is the value", "check your algebra"]
    judge_results = [_judge(False), _judge(True), _judge(False), _judge(True)]

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._build_tutor_prompt = lambda state: f"tutor prompt {state.turn_idx}"
    workflow._build_student_prompt_from_state = lambda state: "student prompt"
    workflow._log_rollout_stats = lambda **kwargs: stats.update(kwargs)
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def generate_tutor_response(tutor_state, **_kwargs):
        tutor_states.append(tutor_state)
        return response, tutor_outputs.pop(0)

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        outputs = [
            "initial wrong answer",
            "student copies 42",
            "student still contaminated",
            "clean student solved",
        ]
        return outputs[len(student_calls) - 1], None

    async def score_answer(*_args, **_kwargs):
        return judge_results.pop(0)

    async def run_leak_check(*_args, **_kwargs):
        return leak_results.pop(0)

    async def update_history(**kwargs):
        update_calls.append(kwargs)
        return PublicHistoryState(summary="clean turn kept", turn_count=1)

    async def fail_annotate(*_args, **_kwargs):
        raise AssertionError("feedback mode should not rerun post-hoc leak checks")

    workflow._generate_tutor_response = generate_tutor_response
    workflow._run_student = run_student
    workflow._score_answer_async = score_answer
    workflow._run_optional_leak_check = run_leak_check
    workflow._run_public_summary_update = update_history
    workflow._annotate_turn_leak_results = fail_annotate

    result = asyncio.run(
        workflow._run_episode(
            {"task": "task", "ground_truth": "42"},
            external_client=object(),
        )
    )

    assert result is not None
    assert len(student_calls) == 4
    assert len(update_calls) == 1
    assert stats["termination_reason"] == "success"
    assert stats["leak_count"] == 2
    assert stats["traces"][0].judge_correct is True
    assert stats["traces"][0].invalid_due_to_leak is True
    assert stats["traces"][1].invalid_due_to_leak is True
    assert stats["traces"][2].invalid_due_to_leak is False
    assert stats["traces"][0].reward_components == {"leak": -1.0}
    assert stats["traces"][1].reward_components == {"leak": -1.0}
    assert stats["traces"][2].reward_components["success_credit"] == pytest.approx(1.0)
    assert workflow.last_total_reward == pytest.approx(-1.0)

    first_public_history = "Student round 0:\ninitial wrong answer"
    assert student_calls[1].previous_student_output == "initial wrong answer"
    assert student_calls[2].previous_student_output == "initial wrong answer"
    assert student_calls[3].previous_student_output == "initial wrong answer"
    assert student_calls[1].public_history.summary == first_public_history
    assert student_calls[2].public_history.summary == first_public_history
    assert student_calls[3].public_history.summary == first_public_history
    assert update_calls[0]["previous_student_answer"] == "initial wrong answer"
    assert update_calls[0]["current_student_answer"] == "clean student solved"

    assert tutor_states[1].previous_feedback.kind == "leak"
    assert (
        "Turn 1: tutor output was rejected for answer leakage. Leak feedback: "
        "The tutor directly stated the final answer."
        in tutor_states[1].previous_feedback.leak_history
    )
    assert "student copies 42" not in tutor_states[1].previous_feedback.leak_history
    assert (
        "Turn 1: tutor output was rejected for answer leakage. Leak feedback: "
        "The tutor directly stated the final answer."
        in tutor_states[2].previous_feedback.leak_history
    )
    assert (
        "Turn 2: tutor output was rejected for answer leakage. Leak feedback: "
        "The tutor computed the final value for the student."
        in tutor_states[2].previous_feedback.leak_history
    )
    assert workflow.last_history[0]["invalid_due_to_leak"] is True
    assert workflow.last_history[0]["judge_correct"] is True


def test_workflow_student_generalize_runs_after_success_from_same_context(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.max_turns = 1
    workflow.leak_handling_mode = "reward_only"
    workflow.success_reward = 1.0
    workflow.leak_penalty = 0.0
    workflow.assign_success_reward = False
    workflow.outcome_prior_turn_weight = 0.1
    workflow.outcome_credit_gamma = 0.9
    workflow.early_success_bonus = 0.0
    workflow.enable_turn_penalty = False
    workflow.turn_penalty = 0.0
    workflow.length_penalty_threshold_chars = 0
    workflow.length_penalty_per_100_chars = 0.0
    workflow.length_penalty_min = 0.0
    workflow.pairwise_reward_enabled = False
    workflow.student_generalize_enabled = True
    workflow.student_generalize_level_rewards = {"level1": 0.2, "level2": 0.5}
    workflow.student_generalize_bank = {}
    workflow.student_system_prompt = ""

    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[0],
        input_len=1,
        output_len=1,
    )
    student_calls = []
    transfer_calls = []
    score_calls = []
    judge_results = [_judge(False), _judge(True), _judge(True), _judge(False)]

    workflow._make_actor_caller = lambda **_kwargs: object()
    workflow._make_auxiliary_caller = lambda **_kwargs: object()
    workflow._make_answer_judge_caller = lambda **_kwargs: None
    workflow._build_tutor_prompt = lambda state: f"tutor prompt {state.turn_idx}"
    workflow._build_student_prompt_from_state = lambda state: "student prompt"
    workflow._log_rollout_stats = lambda **_kwargs: None
    workflow._maybe_dump_debug_trace = lambda **_kwargs: None

    async def generate_tutor_response(*_args, **_kwargs):
        return response, "last tutor hint"

    async def run_student(state, **_kwargs):
        student_calls.append(state)
        outputs = [
            "initial wrong",
            "original solved",
        ]
        return outputs[len(student_calls) - 1], None

    async def call_auxiliary_prompt(**kwargs):
        transfer_calls.append(kwargs)
        outputs = ["level1 solved", "level2 wrong"]
        output = outputs[len(transfer_calls) - 1]
        return TextCallResult(text=output, raw_text=output, error=None)

    async def score_answer(task, ground_truth, student_output, **_kwargs):
        score_calls.append((task, ground_truth, student_output))
        return judge_results.pop(0)

    async def run_leak_check(*_args, **_kwargs):
        return _leak(False)

    async def update_history(**_kwargs):
        return PublicHistoryState(summary="success context", turn_count=1)

    workflow._generate_tutor_response = generate_tutor_response
    workflow._run_student = run_student
    workflow._call_auxiliary_prompt = call_auxiliary_prompt
    workflow._score_answer_async = score_answer
    workflow._run_optional_leak_check = run_leak_check
    workflow._run_public_summary_update = update_history

    result = asyncio.run(
        workflow._run_episode(
            {
                "id": "sample-1",
                "task": "original task",
                "ground_truth": "42",
                "metadata": {
                    "student_generalize": {
                        "level1": {
                            "task": "level1 task",
                            "ground_truth": "43",
                        },
                        "level2": {
                            "task": "level2 task",
                            "ground_truth": "44",
                        },
                    }
                },
            },
            external_client=object(),
        )
    )

    assert result is not None
    assert [state.task for state in student_calls] == [
        "original task",
        "original task",
    ]
    assert len(transfer_calls) == 2
    assert transfer_calls[0]["system_prompt"] == ""
    assert "success context" in transfer_calls[0]["user_prompt"]
    assert "success context" in transfer_calls[1]["user_prompt"]
    assert "level1 task" in transfer_calls[0]["user_prompt"]
    assert "level2 task" in transfer_calls[1]["user_prompt"]
    assert "original solved" in transfer_calls[0]["user_prompt"]
    assert "last tutor hint" in transfer_calls[1]["user_prompt"]
    assert score_calls[-2:] == [
        ("level1 task", "43", "level1 solved"),
        ("level2 task", "44", "level2 wrong"),
    ]

    generalization = workflow.last_student_generalization_results
    assert [item.level for item in generalization] == ["level1", "level2"]
    assert [item.attempted for item in generalization] == [True, True]
    assert [item.reward for item in generalization] == pytest.approx([0.2, 0.0])
    assert workflow.last_traces[0].reward_components["success_credit"] == pytest.approx(
        1.0
    )
    assert workflow.last_traces[0].reward_components[
        "student_generalize_level1"
    ] == pytest.approx(0.2)
    assert "student_generalize_level2" not in workflow.last_traces[0].reward_components
    assert workflow.last_total_reward == pytest.approx(1.2)


def test_student_generalize_only_success_skips_failed_rollout(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_enabled = True
    workflow.student_generalize_mode = "only_success"
    workflow.student_generalize_level_rewards = {"level1": 0.2, "level2": 0.5}
    workflow.student_generalize_bank = {}

    async def fail_call(*args, **kwargs):
        raise AssertionError("generalization should not run after failed rollout")

    workflow._call_auxiliary_prompt = fail_call
    results = asyncio.run(
        workflow._run_student_generalization(
            {
                "id": "sample-1",
                "metadata": {
                    "student_generalize": {
                        "level1": {"task": "level1", "ground_truth": "1"},
                        "level2": {"task": "level2", "ground_truth": "2"},
                    }
                },
            },
            _episode([_turn(1, correct=False)], termination_reason="max_turns"),
            aux_caller=object(),
            answer_judge_caller=None,
        )
    )

    assert results == []


def test_student_generalize_always_rewards_final_valid_failed_turn(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_enabled = True
    workflow.student_generalize_mode = "always"
    workflow.student_generalize_level_rewards = {"level1": 0.2, "level2": 0.5}
    workflow.student_generalize_bank = {}
    workflow.student_system_prompt = ""

    turn1 = _turn(1, correct=False, tutor_output="first hint")
    turn1.public_history_after = "turn 1 context"
    turn1.student_output = "turn 1 wrong"
    turn2 = _turn(2, correct=False, tutor_output="final hint")
    turn2.student_state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(summary="turn 1 context", turn_count=1),
        previous_student_output="turn 1 wrong",
        latest_tutor_visible_output="final hint",
    )
    turn2.public_history_after = "turn 2 context"
    turn2.student_output = "turn 2 wrong"
    episode = _episode([turn1, turn2], termination_reason="max_turns")

    transfer_calls = []
    judge_results = [_judge(True), _judge(False)]

    async def call_auxiliary_prompt(**kwargs):
        transfer_calls.append(kwargs)
        output = "level1 solved" if len(transfer_calls) == 1 else "level2 wrong"
        return TextCallResult(text=output, raw_text=output, error=None)

    async def score_answer(*_args, **_kwargs):
        return judge_results.pop(0)

    workflow._call_auxiliary_prompt = call_auxiliary_prompt
    workflow._score_answer_async = score_answer

    results = asyncio.run(
        workflow._run_student_generalization(
            {
                "metadata": {
                    "student_generalize": {
                        "level1": {"task": "level1 task", "ground_truth": "1"},
                        "level2": {"task": "level2 task", "ground_truth": "2"},
                    }
                }
            },
            episode,
            aux_caller=object(),
            answer_judge_caller=None,
        )
    )
    assignments = [
        types.SimpleNamespace(reward=0.0, reward_components={}),
        types.SimpleNamespace(reward=0.0, reward_components={}),
    ]
    workflow._apply_student_generalization_rewards([turn1, turn2], assignments, results)

    assert [result.reward_turn_idx for result in results] == [2, 2]
    assert [result.reward for result in results] == pytest.approx([0.2, 0.0])
    assert "turn 2 context" in transfer_calls[0]["user_prompt"]
    assert "turn 2 wrong" in transfer_calls[0]["user_prompt"]
    assert "final hint" in transfer_calls[0]["user_prompt"]
    assert assignments[0].reward == pytest.approx(0.0)
    assert assignments[1].reward == pytest.approx(0.2)
    assert assignments[1].reward_components["student_generalize_level1"] == (
        pytest.approx(0.2)
    )


def test_student_generalize_always_uses_previous_valid_turn_before_leak():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_mode = "always"
    turn1 = _turn(1, correct=False, tutor_output="safe hint")
    turn1.public_history_after = "safe context"
    turn1.student_output = "safe wrong"
    leaked_turn = _turn(2, leaked=True, tutor_output="leaked hint")
    leaked_turn.student_state = None
    leaked_turn.student_output = ""
    leaked_turn.judge_result = None
    leaked_turn.public_history_after = "safe context"

    anchor = workflow._student_generalization_anchor(
        _episode([turn1, leaked_turn], termination_reason="leak")
    )

    assert anchor is not None
    assert anchor.reward_turn_idx == 1
    assert anchor.public_history.summary == "safe context"
    assert anchor.previous_student_output == "safe wrong"
    assert anchor.teacher_feedback == "safe hint"


def test_student_generalize_always_without_valid_turn_uses_initial_log_only_context():
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_mode = "always"

    anchor = workflow._student_generalization_anchor(
        _episode(
            [], termination_reason=tutor_workflow.CONTEXT_BUDGET_TERMINATION_REASON
        )
    )

    assert anchor is not None
    assert anchor.reward_turn_idx is None
    assert "Student round 0" in anchor.public_history.summary
    assert "initial" in anchor.public_history.summary
    assert anchor.previous_student_output == "initial"


def test_student_generalize_missing_variants_skips_without_student_call(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_enabled = True
    workflow.student_generalize_level_rewards = {"level1": 0.2, "level2": 0.5}
    workflow.student_generalize_bank = {}
    turn = _turn(1, correct=True)
    turn.public_history_after = "success context"
    episode = _episode([turn], termination_reason="success")

    async def fail_student(*args, **kwargs):
        raise AssertionError("student should not be called without variants")

    workflow._run_student = fail_student

    results = asyncio.run(
        workflow._run_student_generalization(
            {"id": "missing", "metadata": {}},
            episode,
            aux_caller=object(),
            answer_judge_caller=None,
        )
    )

    assert [result.level for result in results] == ["level1", "level2"]
    assert all(result.skipped for result in results)
    assert all(result.skip_reason == "missing_variant" for result in results)


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

    async def score_answer(task, ground_truth, student_output):
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


def test_pairwise_current_leak_still_compares_reference():
    calls = {"reference": 0, "student": 0, "leak": 0, "score": 0}

    class RewardCaller:
        async def call_text(self, *args, **kwargs):
            raise AssertionError("pairwise judge should not be needed")

    async def generate_reference_tutor(tutor_state, reference_version):
        del tutor_state, reference_version
        calls["reference"] += 1
        return "reference hint"

    async def run_student(state):
        del state
        calls["student"] += 1
        return "reference student answer", None

    async def run_leak_check(task, ground_truth, teacher_action):
        del task, ground_truth, teacher_action
        calls["leak"] += 1
        return _leak(False)

    async def score_answer(task, ground_truth, student_output):
        del task, ground_truth, student_output
        calls["score"] += 1
        return _judge(False)

    evaluator = PairwiseTutorEvaluator(
        reward_scale=0.05,
        reward_caller=RewardCaller(),
        generate_reference_tutor=generate_reference_tutor,
        run_student=run_student,
        run_leak_check=run_leak_check,
        score_answer=score_answer,
    )
    turn = _turn(1, leaked=True, correct=True)

    result = asyncio.run(
        evaluator.evaluate_turn(
            _episode([turn], termination_reason="success"),
            turn,
            reference_version=0,
        )
    )

    assert result.outcome == "current"
    assert result.reason == "exact_correctness"
    assert result.reward == pytest.approx(0.05)
    assert calls == {"reference": 1, "student": 1, "leak": 1, "score": 1}


def test_pairwise_invalid_current_leak_is_skipped():
    async def fail_reference(*args, **kwargs):
        raise AssertionError("invalid current leak should skip reference generation")

    async def fail_student(*args, **kwargs):
        raise AssertionError("student should not be called")

    async def fail_leak_check(*args, **kwargs):
        raise AssertionError("leak checker should not be called")

    async def fail_score(*args, **kwargs):
        raise AssertionError("answer scorer should not be called")

    class RewardCaller:
        async def call_text(self, *args, **kwargs):
            raise AssertionError("pairwise judge should not be called")

    evaluator = PairwiseTutorEvaluator(
        reward_scale=0.05,
        reward_caller=RewardCaller(),
        generate_reference_tutor=fail_reference,
        run_student=fail_student,
        run_leak_check=fail_leak_check,
        score_answer=fail_score,
    )
    turn = _turn(1, leaked=True, correct=True)
    turn.invalid_due_to_leak = True

    result = asyncio.run(
        evaluator.evaluate_turn(
            _episode([turn], termination_reason="max_turns"),
            turn,
            reference_version=0,
        )
    )

    assert result.outcome == "skipped"
    assert result.reason == "current_turn_invalid_due_to_leak"
    assert result.reward == pytest.approx(0.0)


def test_pairwise_reference_leak_still_runs_student():
    calls = {"student": 0, "score": 0}

    class RewardCaller:
        async def call_text(self, *args, **kwargs):
            raise AssertionError("pairwise judge should not be needed")

    async def generate_reference_tutor(tutor_state, reference_version):
        del tutor_state, reference_version
        return "reference leaks 42"

    async def run_student(state):
        del state
        calls["student"] += 1
        return "reference student answer", None

    async def run_leak_check(task, ground_truth, teacher_action):
        del task, ground_truth, teacher_action
        return _leak(True)

    async def score_answer(task, ground_truth, student_output):
        del task, ground_truth, student_output
        calls["score"] += 1
        return _judge(True)

    evaluator = PairwiseTutorEvaluator(
        reward_scale=0.05,
        reward_caller=RewardCaller(),
        generate_reference_tutor=generate_reference_tutor,
        run_student=run_student,
        run_leak_check=run_leak_check,
        score_answer=score_answer,
    )
    turn = _turn(1, correct=False)

    result = asyncio.run(
        evaluator.evaluate_turn(
            _episode([turn], termination_reason="max_turns"),
            turn,
            reference_version=0,
        )
    )

    assert result.outcome == "reference"
    assert result.reason == "exact_correctness"
    assert result.reference is not None
    assert result.reference.leak_result.leaked is True
    assert result.reward == pytest.approx(-0.05)
    assert calls == {"student": 1, "score": 1}


def _trace(
    turn_idx: int,
    reward_components: dict[str, float],
    *,
    leaked: bool = False,
    leak_level: int | None = None,
    correct: bool = False,
) -> TurnTrace:
    turn = _turn(turn_idx, leaked=leaked, leak_level=leak_level, correct=correct)
    return TurnTrace(
        turn_idx=turn.turn_idx,
        tutor_state=turn.tutor_state,
        tutor_raw_output=turn.tutor_raw_output,
        tutor_visible_output=turn.tutor_visible_output,
        leaked=turn.leak_result.leaked,
        student_output=turn.student_output,
        judge_correct=bool(turn.judge_result and turn.judge_result.correct),
        judge_feedback=turn.judge_result.feedback if turn.judge_result else "",
        reward=sum(reward_components.values()),
        reward_components=reward_components,
        public_history_before=turn.public_history_before,
        public_history_after=turn.public_history_after,
        leak_level=turn.leak_result.leak_level,
    )


def _metric_workflow(**overrides):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    params = {
        "success_reward": 1.0,
        "early_success_bonus": 0.0,
        "leak_penalty": -0.05,
        "enable_turn_penalty": True,
        "turn_penalty": -0.01,
        "length_penalty_threshold_chars": 10,
        "length_penalty_per_100_chars": -0.5,
        "pairwise_reward_enabled": True,
        "pairwise_reward_scale": 0.05,
    }
    params.update(overrides)
    for name, value in params.items():
        setattr(workflow, name, value)
    return workflow


def test_rollout_stats_uses_clean_metric_names_and_reward_breakdown(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    workflow = _metric_workflow()
    traces = [
        _trace(1, {"leak": -0.05, "turn_penalty": -0.01}, leaked=True),
        _trace(
            2,
            {
                "success_credit": 1.0,
                "length_penalty": -0.1,
                "pairwise": 0.05,
            },
            correct=True,
        ),
    ]

    workflow._log_rollout_stats(
        total_reward=sum(trace.reward for trace in traces),
        traces=traces,
        termination_reason="success",
        pre_success=False,
        leak_count=1,
    )

    assert captured["reward"] == pytest.approx(0.89)
    assert captured["turns"] == 2
    assert captured["leaks"] == 1
    assert captured["pre_solved"] == 0.0
    assert captured["solved"] == 1.0
    assert captured["solve_turn"] == 2
    assert captured["stop/max_turns"] == 0.0
    assert captured["stop/context_limit"] == 0.0
    assert captured["stop/leak"] == 0.0

    for old_key in (
        "num_turns",
        "samples_per_episode",
        "leak_count",
        "pre_success",
        "term_success",
        "success_round",
        "termination_pre_solved",
        "termination_success",
        "termination_max_turns",
        "termination_context_budget_limit",
    ):
        assert old_key not in captured

    expected_components = {
        "success": 1.0,
        "leak": -0.05,
        "turn_penalty": -0.01,
        "length_penalty": -0.1,
        "pairwise": 0.05,
    }
    total_abs = sum(abs(value) for value in expected_components.values())
    for name, value in expected_components.items():
        assert captured[f"reward_component/{name}"] == pytest.approx(value)
        assert captured[f"reward_share/{name}"] == pytest.approx(abs(value) / total_abs)


def test_rollout_stats_logs_leak_termination(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    workflow = _metric_workflow()

    workflow._log_rollout_stats(
        total_reward=-1.0,
        traces=[_trace(1, {"leak": -1.0}, leaked=True)],
        termination_reason="leak",
        pre_success=False,
        leak_count=1,
    )

    assert captured["solved"] == pytest.approx(0.0)
    assert captured["stop/max_turns"] == pytest.approx(0.0)
    assert captured["stop/context_limit"] == pytest.approx(0.0)
    assert captured["stop/leak"] == pytest.approx(1.0)


def test_rollout_stats_routes_student_generalize_metrics_to_generalize(monkeypatch):
    rollout_metrics = {}
    generalize_metrics = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: rollout_metrics.update(metrics),
    )
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_generalize_scalar",
        lambda **metrics: generalize_metrics.update(metrics),
    )
    workflow = _metric_workflow(
        student_generalize_enabled=True,
        student_generalize_level_rewards={"level1": 0.2, "level2": 0.5},
    )
    traces = [
        _trace(
            1,
            {
                "success_credit": 1.0,
                "student_generalize_level1": 0.2,
            },
            correct=True,
        )
    ]
    generalization = [
        tutor_workflow.StudentGeneralizationResult(
            level="level1",
            attempted=True,
            judge_result=_judge(True),
            reward=0.2,
        ),
        tutor_workflow.StudentGeneralizationResult(
            level="level2",
            skipped=True,
            skip_reason="missing_variant",
        ),
    ]

    workflow._log_rollout_stats(
        total_reward=sum(trace.reward for trace in traces),
        traces=traces,
        termination_reason="success",
        pre_success=False,
        leak_count=0,
        student_generalization_results=generalization,
    )

    assert not any(key.startswith("student_generalize/") for key in rollout_metrics)
    assert rollout_metrics[
        "reward_component/student_generalize_level1"
    ] == pytest.approx(0.2)
    assert rollout_metrics[
        "reward_component/student_generalize_level2"
    ] == pytest.approx(0.0)
    assert generalize_metrics["student_level1_attempted"] == pytest.approx(1.0)
    assert generalize_metrics["student_level1_success"] == pytest.approx(1.0)
    assert generalize_metrics[
        "student_level1_correct_given_attempted"
    ] == pytest.approx(1.0)
    assert generalize_metrics["student_level2_attempted"] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_success"] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_skipped"] == pytest.approx(1.0)
    assert "student_level2_correct_given_attempted" not in generalize_metrics
    assert "teacher_success" not in generalize_metrics


def test_generalize_stats_includes_eval_teacher_success(monkeypatch):
    generalize_metrics = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_generalize_scalar",
        lambda **metrics: generalize_metrics.update(metrics),
    )
    workflow = _metric_workflow(student_generalize_enabled=True)
    ctx_cls = tutor_workflow.workflow_context.WorkflowContext
    tutor_workflow.workflow_context.set(ctx_cls(is_eval=True))
    try:
        workflow._log_generalize_stats(
            solved=True,
            student_generalization_results=[
                tutor_workflow.StudentGeneralizationResult(
                    level="level1",
                    attempted=True,
                    judge_result=_judge(False),
                )
            ],
        )
    finally:
        tutor_workflow.workflow_context.set(ctx_cls())

    assert generalize_metrics["teacher_success"] == pytest.approx(1.0)
    assert generalize_metrics["student_level1_attempted"] == pytest.approx(1.0)
    assert generalize_metrics["student_level1_success"] == pytest.approx(0.0)
    assert generalize_metrics[
        "student_level1_correct_given_attempted"
    ] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_attempted"] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_success"] == pytest.approx(0.0)


def test_generalize_stats_logs_unattempted_levels(monkeypatch):
    generalize_metrics = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_generalize_scalar",
        lambda **metrics: generalize_metrics.update(metrics),
    )
    workflow = _metric_workflow(student_generalize_enabled=True)

    workflow._log_generalize_stats(
        solved=False,
        student_generalization_results=[],
    )

    assert generalize_metrics["student_level1_attempted"] == pytest.approx(0.0)
    assert generalize_metrics["student_level1_success"] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_attempted"] == pytest.approx(0.0)
    assert generalize_metrics["student_level2_success"] == pytest.approx(0.0)
    assert "student_level1_correct_given_attempted" not in generalize_metrics
    assert "student_level2_correct_given_attempted" not in generalize_metrics


def test_reward_component_metrics_include_staged_leak_components():
    workflow = _metric_workflow(
        leak_penalty_mode="staged",
        leak_penalty_final_answer=-1.0,
        leak_penalty_compute=-0.5,
        leak_penalty_formula=-0.1,
        enable_turn_penalty=False,
        length_penalty_threshold_chars=0,
        pairwise_reward_enabled=False,
    )

    metrics = workflow._reward_component_metrics(
        [_trace(1, {"leak_compute": -0.5}, leaked=True, leak_level=2)]
    )

    assert "reward_component/leak" not in metrics
    assert metrics["reward_component/success"] == pytest.approx(0.0)
    assert metrics["reward_component/leak_final_answer"] == pytest.approx(0.0)
    assert metrics["reward_component/leak_compute"] == pytest.approx(-0.5)
    assert metrics["reward_component/leak_formula"] == pytest.approx(0.0)
    assert metrics["reward_share/leak_compute"] == pytest.approx(1.0)


def test_reward_component_share_is_zero_when_enabled_components_are_absent():
    workflow = _metric_workflow(
        enable_turn_penalty=False,
        length_penalty_threshold_chars=0,
        pairwise_reward_enabled=False,
    )

    metrics = workflow._reward_component_metrics([])

    assert metrics == {
        "reward_component/success": 0.0,
        "reward_share/success": 0.0,
        "reward_component/leak": 0.0,
        "reward_share/leak": 0.0,
    }
