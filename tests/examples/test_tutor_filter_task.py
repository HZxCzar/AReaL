from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from examples.tutor.core.types import JudgeResult
from examples.tutor.scripts import filter_task
from examples.tutor.scripts.filter_task import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_STUDENT_BASE_URL_ENV,
    DEFAULT_TEACHER_BASE_URL,
    build_answer_judge_caller,
    build_aux_request_params,
    build_teacher_request_params,
    classify_pre_solved_row,
    classify_teacher_solved_row,
    parse_args,
    request_max_completion_tokens,
    request_temperature,
    request_top_p,
    resolve_student_base_url,
    resolve_student_concurrency,
    resolve_teacher_base_url,
    resolve_teacher_concurrency,
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
        "student_concurrency": 0,
        "teacher_concurrency": 0,
        "concurrency": 0,
        "thinking": "off",
        "student_thinking": None,
        "teacher_thinking": None,
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
            max_concurrent_calls=4,
            answer_judge_enabled=True,
            answer_judge_max_tokens=256,
            request_params={
                "seed": 42,
                "extra_headers": {"x-inspire-inference-key": "student"},
                "extra_body": {"min_p": 0},
            },
        ),
        tokenizer_path="/models/qwen3-8b",
        sglang=SimpleNamespace(context_length=40960),
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


def test_role_thinking_overrides_shared_choice_independently():
    """Role-specific thinking flags should override the deprecated shared flag."""
    config = make_request_config()
    args = make_request_args(
        thinking="on",
        student_thinking="off",
        teacher_thinking="unset",
    )

    aux_params = build_aux_request_params(args, config)
    teacher_params = build_teacher_request_params(args, config)

    assert aux_params["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "chat_template_kwargs" not in teacher_params["extra_body"]


def test_qwen3_nonthinking_sampling_params_are_applied_to_both_roles():
    """Official Qwen3 non-thinking sampling values should survive param merging."""
    config = make_request_config()
    role_params = '{"seed":42,"extra_body":{"top_k":20,"min_p":0}}'
    args = make_request_args(
        student_thinking="off",
        teacher_thinking="off",
        student_max_tokens=2048,
        teacher_max_tokens=2048,
        student_temperature=0.7,
        teacher_temperature=0.7,
        student_top_p=0.8,
        teacher_top_p=0.8,
        student_request_params=role_params,
        teacher_request_params=role_params,
    )

    for params in (
        build_aux_request_params(args, config),
        build_teacher_request_params(args, config),
    ):
        assert request_max_completion_tokens(params) == 2048
        assert request_temperature(params) == 0.7
        assert request_top_p(params) == 0.8
        assert params["seed"] == 42
        assert params["extra_body"] == {
            "top_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }


def test_role_concurrency_prefers_role_then_shared_then_config():
    """Each role should resolve concurrency without coupling the two endpoints."""
    config = make_request_config()

    role_args = make_request_args(
        concurrency=3,
        student_concurrency=4,
        teacher_concurrency=12,
    )
    assert resolve_student_concurrency(role_args, config) == 4
    assert resolve_teacher_concurrency(role_args, config) == 12

    shared_args = make_request_args(concurrency=3)
    assert resolve_student_concurrency(shared_args, config) == 3
    assert resolve_teacher_concurrency(shared_args, config) == 3

    config_args = make_request_args()
    assert resolve_student_concurrency(config_args, config) == 4
    assert resolve_teacher_concurrency(config_args, config) == 4


def test_default_filter_config_path_exists():
    """The default filter config should follow config directory renames."""
    assert Path(DEFAULT_CONFIG_PATH).is_file()


def test_zero_argument_cli_uses_aligned_filter_defaults(monkeypatch):
    """The production filter should start with no role-specific CLI arguments."""
    monkeypatch.setattr(sys, "argv", ["filter_task.py"])

    args = parse_args()

    assert args.student_model == "qwen3-1.7b"
    assert args.student_base_url.startswith("https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp.")
    assert args.student_thinking == "off"
    assert args.teacher_base_url == DEFAULT_TEACHER_BASE_URL
    assert args.teacher_model == "qwen3-8b"
    assert args.teacher_thinking == "off"
    assert args.student_max_tokens == args.teacher_max_tokens == 2048
    assert args.student_temperature == args.teacher_temperature == 0.7
    assert args.student_top_p == args.teacher_top_p == 0.8
    assert args.student_concurrency == args.teacher_concurrency == 4
    assert args.student_attempts == args.teacher_attempts == 1
    assert args.splits == ["train", "test"]
    assert args.overwrite is True


def test_endpoint_resolution_supports_cloud_student_and_teacher(monkeypatch):
    config = make_request_config()
    monkeypatch.delenv(DEFAULT_STUDENT_BASE_URL_ENV, raising=False)

    defaults = make_request_args()
    assert resolve_student_base_url(defaults, config).startswith(
        "https://chj8bobdbm9acbj8kkcqgemj8pj8e9gp."
    )
    assert resolve_teacher_base_url(defaults) == DEFAULT_TEACHER_BASE_URL

    env_defaults = make_request_args(student_base_url="")
    monkeypatch.setenv(DEFAULT_STUDENT_BASE_URL_ENV, "http://cloud-student/v1")
    assert resolve_student_base_url(env_defaults, config) == "http://cloud-student/v1"

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


def test_answer_judge_uses_remote_8b_teacher_endpoint(monkeypatch):
    """The weaker 1.7B student must not judge filter correctness."""
    config = make_request_config()
    args = make_request_args(
        teacher_base_url="http://teacher-8b/v1",
        teacher_model="qwen3-8b",
        teacher_api_key="TEACHER_KEY",
        teacher_timeout=88,
        teacher_top_p=0.8,
    )
    captured = {}

    def fake_async_llm_caller(caller_config):
        captured["config"] = caller_config
        return SimpleNamespace(request_config={})

    monkeypatch.setattr(filter_task, "AsyncLLMCaller", fake_async_llm_caller)

    caller = build_answer_judge_caller(
        config,
        args,
        teacher_request_params={
            "max_completion_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.8,
            "extra_body": {
                "top_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
        max_concurrency=4,
    )

    assert caller is not None
    assert captured["config"].base_url == "http://teacher-8b/v1"
    assert captured["config"].model == "qwen3-8b"
    assert captured["config"].api_key == "TEACHER_KEY"
    assert captured["config"].max_tokens == 256
    assert captured["config"].temperature == 0.0
    assert captured["config"].request_params["extra_body"] == {
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


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
