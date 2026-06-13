from __future__ import annotations

import asyncio

from examples.tutor.core.types import JudgeResult
from examples.tutor.scripts.filter_with_llm_judge import (
    classify_pre_solved_row,
    classify_teacher_solved_row,
)


class FakeWorkflow:
    def __init__(self, *, student_answer: str = "student answer"):
        self.student_answer = student_answer
        self.judge_calls: list[tuple[str, str, str, object]] = []

    async def _run_student(self, state):
        del state
        return self.student_answer, None

    async def _score_answer_async(
        self,
        task,
        ground_truth,
        answer,
        *,
        answer_judge_caller,
    ):
        self.judge_calls.append((task, ground_truth, answer, answer_judge_caller))
        correct = bool(getattr(answer_judge_caller, "correct", False))
        raw_result = {
            "extracted_answer": answer,
            "answer_judge": {
                "enabled": True,
                "used": True,
                "correct": correct,
            },
        }
        return JudgeResult(
            raw_output="judge output",
            correct=correct,
            feedback="Correct." if correct else "Incorrect.",
            parse_error=None,
            raw_result=raw_result,
        )


class FakeAnswerJudgeCaller:
    def __init__(self, *, correct: bool):
        self.correct = correct


class FakeTeacherClient:
    def __init__(self, *, answer: str = "teacher answer", error: str | None = None):
        self.answer = answer
        self.error = error
        self.calls = 0

    async def solve(self, *, system_prompt, user_prompt):
        del system_prompt, user_prompt
        self.calls += 1
        if self.error is not None:
            return "", {}, self.error
        return self.answer, {"completion_tokens": 1}, None


def test_pre_solved_row_with_llm_judge_correct_is_dropped():
    workflow = FakeWorkflow(student_answer="equivalent but not exact")
    caller = FakeAnswerJudgeCaller(correct=True)

    result = asyncio.run(
        classify_pre_solved_row(
            workflow=workflow,
            answer_judge_caller=caller,
            row={"id": "row-1", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=1,
            keep_on_error=False,
        )
    )

    assert result.status == "pre_solved"
    assert result.kept is False
    assert result.attempts[0].raw_result["answer_judge"]["used"] is True
    assert workflow.judge_calls[0][3] is caller


def test_teacher_solved_row_with_llm_judge_correct_is_kept():
    workflow = FakeWorkflow()
    caller = FakeAnswerJudgeCaller(correct=True)
    client = FakeTeacherClient(answer="equivalent teacher answer")

    result = asyncio.run(
        classify_teacher_solved_row(
            client=client,
            workflow=workflow,
            answer_judge_caller=caller,
            system_prompt="system",
            user_template="{task}",
            row={"id": "row-2", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=1,
            keep_on_error=False,
        )
    )

    assert result.status == "teacher_solved"
    assert result.kept is True
    assert result.attempts[0].usage == {"completion_tokens": 1}
    assert workflow.judge_calls[0][2] == "equivalent teacher answer"


def test_teacher_unsolved_after_llm_judge_incorrect_is_dropped():
    workflow = FakeWorkflow()
    caller = FakeAnswerJudgeCaller(correct=False)
    client = FakeTeacherClient(answer="wrong answer")

    result = asyncio.run(
        classify_teacher_solved_row(
            client=client,
            workflow=workflow,
            answer_judge_caller=caller,
            system_prompt="system",
            user_template="{task}",
            row={"id": "row-3", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=1,
            keep_on_error=False,
        )
    )

    assert result.status == "teacher_unsolved"
    assert result.kept is False
    assert result.attempts[0].raw_result["answer_judge"]["correct"] is False


def test_teacher_error_respects_keep_on_error():
    workflow = FakeWorkflow()
    caller = FakeAnswerJudgeCaller(correct=True)
    client = FakeTeacherClient(error="connection failed")

    result = asyncio.run(
        classify_teacher_solved_row(
            client=client,
            workflow=workflow,
            answer_judge_caller=caller,
            system_prompt="system",
            user_template="{task}",
            row={"id": "row-4", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=1,
            keep_on_error=True,
        )
    )

    assert result.status == "error"
    assert result.kept is True
    assert result.error == "All teacher attempts failed."
    assert workflow.judge_calls == []
