from __future__ import annotations

import asyncio
from types import SimpleNamespace

from examples.tutor.core.types import JudgeResult
from examples.tutor.scripts.filter_task import (
    build_aux_request_params,
    build_teacher_request_params,
    classify_pre_solved_row,
    classify_teacher_solved_row,
    request_max_completion_tokens,
    request_temperature,
    request_top_p,
    resolve_student_base_url,
    resolve_teacher_base_url,
    strip_dedicated_request_params,
)


class FakeWorkflow:
    def __init__(self, *, student_answer: str = "student answer"):
        self.student_answer = student_answer
        self.student_calls = 0
        self.judge_calls: list[tuple[str, str, str, object]] = []

    async def _run_student(self, state):
        del state
        self.student_calls += 1
        return f"{self.student_answer} {self.student_calls}", None

    async def _score_answer_async(
        self,
        task,
        ground_truth,
        answer,
        *,
        answer_judge_caller,
    ):
        self.judge_calls.append((task, ground_truth, answer, answer_judge_caller))
        correct = bool(answer_judge_caller.next_correct())
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
    def __init__(self, *, correct: bool | list[bool]):
        self.correct = correct if isinstance(correct, list) else [correct]
        self.calls = 0

    def next_correct(self) -> bool:
        index = min(self.calls, len(self.correct) - 1)
        self.calls += 1
        return bool(self.correct[index])


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


def make_request_args(**overrides):
    values = {
        "base_url": "",
        "student_base_url": "",
        "teacher_base_url": "",
        "model": "",
        "student_model": "",
        "teacher_model": "",
        "api_key": "",
        "student_api_key": "",
        "teacher_api_key": "",
        "timeout": 120.0,
        "student_timeout": None,
        "teacher_timeout": None,
        "thinking": "off",
        "request_params": "",
        "request_params_file": None,
        "student_request_params": "",
        "student_request_params_file": None,
        "teacher_request_params": "",
        "teacher_request_params_file": None,
        "max_tokens": 0,
        "student_max_tokens": 0,
        "teacher_max_tokens": 0,
        "temperature": None,
        "student_temperature": None,
        "teacher_temperature": None,
        "top_p": None,
        "student_top_p": None,
        "teacher_top_p": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_request_config():
    return SimpleNamespace(
        enable_thinking=True,
        gconfig=SimpleNamespace(
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
        ),
        auxiliary_model=SimpleNamespace(
            base_url="http://config-student/v1",
            model="config-student",
            api_key="CONFIG_KEY",
            timeout=99,
            enable_thinking=True,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            request_params={
                "seed": 42,
                "extra_headers": {"x-inspire-inference-key": "student"},
                "extra_body": {"min_p": 0},
            },
        ),
    )


def test_request_params_use_role_defaults_and_allow_role_overrides():
    config = make_request_config()
    args = make_request_args(
        request_params='{"seed": 7, "top_p": 0.8}',
        student_request_params='{"temperature": 0.2}',
        teacher_request_params='{"max_tokens": 128}',
        student_max_tokens=256,
    )

    aux_params = build_aux_request_params(args, config)
    teacher_params = build_teacher_request_params(args, config)

    assert aux_params["seed"] == 7
    assert teacher_params["seed"] == 7
    assert request_max_completion_tokens(aux_params) == 256
    assert request_max_completion_tokens(teacher_params) == 128
    assert request_temperature(aux_params) == 0.2
    assert request_temperature(teacher_params) == 0.6
    assert request_top_p(aux_params) == 0.8
    assert request_top_p(teacher_params) == 0.8
    assert aux_params["extra_headers"] == {"x-inspire-inference-key": "student"}
    assert aux_params["extra_body"]["min_p"] == 0
    assert teacher_params["extra_body"]["top_k"] == 20
    assert "extra_headers" not in teacher_params
    assert aux_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert (
        teacher_params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    )
    assert strip_dedicated_request_params(aux_params) == {
        "seed": 7,
        "extra_headers": {"x-inspire-inference-key": "student"},
        "extra_body": {
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_endpoint_resolution_supports_heterogeneous_student_and_teacher():
    config = make_request_config()

    defaults = make_request_args()
    assert resolve_student_base_url(defaults, config) == "http://config-student/v1"
    assert resolve_teacher_base_url(defaults) == "http://127.0.0.1:30008/v1"

    shared = make_request_args(base_url="http://shared/v1")
    assert resolve_student_base_url(shared, config) == "http://shared/v1"
    assert resolve_teacher_base_url(shared) == "http://shared/v1"

    separate = make_request_args(
        base_url="http://shared/v1",
        student_base_url="http://student/v1",
        teacher_base_url="http://teacher/v1",
    )
    assert resolve_student_base_url(separate, config) == "http://student/v1"
    assert resolve_teacher_base_url(separate) == "http://teacher/v1"


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


def test_pre_solved_row_tries_until_any_attempt_is_correct():
    workflow = FakeWorkflow(student_answer="attempt")
    caller = FakeAnswerJudgeCaller(correct=[False, True, False])

    result = asyncio.run(
        classify_pre_solved_row(
            workflow=workflow,
            answer_judge_caller=caller,
            row={"id": "row-1", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=3,
            keep_on_error=False,
        )
    )

    assert result.status == "pre_solved"
    assert result.kept is False
    assert len(result.attempts) == 2
    assert workflow.student_calls == 2


def test_pre_solved_row_keeps_after_all_attempts_are_incorrect():
    workflow = FakeWorkflow(student_answer="attempt")
    caller = FakeAnswerJudgeCaller(correct=[False, False, False])

    result = asyncio.run(
        classify_pre_solved_row(
            workflow=workflow,
            answer_judge_caller=caller,
            row={"id": "row-1", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=3,
            keep_on_error=False,
        )
    )

    assert result.status == "student_unsolved"
    assert result.kept is True
    assert len(result.attempts) == 3


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


def test_teacher_solved_row_tries_until_any_attempt_is_correct():
    workflow = FakeWorkflow()
    caller = FakeAnswerJudgeCaller(correct=[False, True, False])
    client = FakeTeacherClient(answer="teacher answer")

    result = asyncio.run(
        classify_teacher_solved_row(
            client=client,
            workflow=workflow,
            answer_judge_caller=caller,
            system_prompt="system",
            user_template="{task}",
            row={"id": "row-2", "task": "task", "ground_truth": "42"},
            index=0,
            attempts=3,
            keep_on_error=False,
        )
    )

    assert result.status == "teacher_solved"
    assert result.kept is True
    assert len(result.attempts) == 2
    assert client.calls == 2


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
            attempts=3,
            keep_on_error=False,
        )
    )

    assert result.status == "teacher_unsolved"
    assert result.kept is False
    assert len(result.attempts) == 3
    assert client.calls == 3
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
