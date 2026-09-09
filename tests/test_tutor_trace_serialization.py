import json

import pytest

from examples.tutor.core.history import trace_to_json
from examples.tutor.core.types import (
    PublicHistoryState,
    TeacherPreSolveAttempt,
    TeacherPreSolveResult,
    TutorPrivateFeedback,
    TutorTurnState,
    TurnTrace,
)


class RuntimeResponse:
    def __deepcopy__(self, memo):
        raise AssertionError("Runtime training response must not be copied into JSON")


@pytest.mark.parametrize("with_presolve", [False, True])
def test_trace_json_excludes_runtime_response_without_mutating_training(with_presolve):
    response = RuntimeResponse()
    pre = TeacherPreSolveResult(
        enabled=True, mode="filter_solver", accepted=True,
        attempts=[TeacherPreSolveAttempt(1, "answer", None, True, response=response)],
        raw_output="answer",
    ) if with_presolve else None
    state = TutorTurnState(
        task="task", ground_truth="answer", public_history=PublicHistoryState(),
        previous_tutor_visible_output="", previous_feedback=TutorPrivateFeedback(),
        turn_idx=1, max_turns=10, teacher_pre_solve_result=pre,
    )
    trace = TurnTrace(
        turn_idx=1, tutor_state=state, tutor_raw_output="raw", tutor_visible_output="visible",
        leaked=False, student_output="student", judge_correct=False, judge_feedback="",
        reward=0.0, reward_components={}, public_history_before=[], public_history_after=[],
    )
    payload = json.loads(json.dumps(trace_to_json(trace)))
    assert trace.tutor_state is state
    assert state.teacher_pre_solve_result is pre
    if pre is not None:
        assert pre.attempts[0].response is response
        saved = payload["tutor_state"]["teacher_pre_solve_result"]
        assert saved["attempts"][0]["raw_output"] == "answer"
        assert "response" not in saved["attempts"][0]
