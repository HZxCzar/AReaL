from __future__ import annotations

import asyncio
import types

import pytest

from examples.tutor import workflow as tutor_workflow
from examples.tutor.core.callers import TextCallResult
from examples.tutor.core.math import score_math_answer
from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.rewards import EpisodeRewardComputer
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PublicHistoryState,
    StudentTurnState,
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


def test_leak_turn_can_receive_success_credit_with_penalty():
    turns = [_turn(1, leaked=True, correct=True)]
    assignments = asyncio.run(
        _outcome_computer().compute(_episode(turns, termination_reason="success"))
    )

    assert assignments[0].reward_components["leak"] == pytest.approx(-1.0)
    assert assignments[0].reward_components["success_credit"] > 0.0
    assert assignments[0].reward == pytest.approx(0.3)


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
    workflow.enable_leak_check = False

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
    workflow.enable_leak_check = False
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
    monkeypatch.setattr(
        tutor_workflow, "PairwiseTutorEvaluator", FakePairwiseEvaluator
    )

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

    first = asyncio.run(
        workflow._score_answer_async(*args, answer_judge_caller=caller)
    )
    second = asyncio.run(
        workflow._score_answer_async(*args, answer_judge_caller=caller)
    )

    assert first.correct is True
    assert second.correct is True
    assert len(caller.calls) == 1


def test_workflow_leak_does_not_skip_student_or_success(monkeypatch):
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.max_turns = 1
    workflow.success_reward = 1.0
    workflow.leak_penalty = -1.0
    workflow.assign_success_reward = False
    workflow.outcome_prior_turn_weight = 0.1
    workflow.outcome_credit_gamma = 0.9
    workflow.early_success_bonus = 0.0
    workflow.enable_turn_penalty = False
    workflow.turn_penalty = -0.01
    workflow.length_penalty_threshold_chars = 0
    workflow.length_penalty_per_100_chars = 0.0
    workflow.length_penalty_min = 0.0
    workflow.pairwise_reward_enabled = False

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
    correct: bool = False,
) -> TurnTrace:
    turn = _turn(turn_idx, leaked=leaked, correct=correct)
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
        assert captured[f"reward_share/{name}"] == pytest.approx(
            abs(value) / total_abs
        )


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

