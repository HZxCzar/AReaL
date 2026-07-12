from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from examples.tutor.prompts import POLARIS_FILTER_SOLVER_USER_TEMPLATE
from examples.tutor.scripts.filter_polaris_task import (
    GenerationResult,
    ScoreOutcome,
    build_role_configs,
    classify_polaris_row,
    render_official_prompt,
)


class FakeClient:
    def __init__(self, *results: GenerationResult) -> None:
        self.results = deque(results)
        self.prompts: list[str] = []
        self.seed_offsets: list[int] = []

    async def solve(self, prompt: str, *, seed_offset: int = 0) -> GenerationResult:
        self.prompts.append(prompt)
        self.seed_offsets.append(seed_offset)
        return self.results.popleft()


def generation(answer: str = "", error: str | None = None) -> GenerationResult:
    return GenerationResult(answer=answer, usage={}, error=error)


def scorer(correct_answers: set[str], error_answers: set[str] | None = None):
    async def _score(task: str, ground_truth: str, answer: str) -> ScoreOutcome:
        assert task == "What is 1+1?"
        assert ground_truth == "2"
        if error_answers and answer in error_answers:
            return ScoreOutcome(
                correct=False,
                feedback="scoring failed",
                raw_result={},
                error="scoring failed",
            )
        return ScoreOutcome(
            correct=answer in correct_answers,
            feedback="Correct." if answer in correct_answers else "Incorrect.",
            raw_result={"extracted_answer": answer},
        )

    return _score


def test_render_official_prompt_uses_fixed_polaris_template():
    """The filter must use the exact repository Polaris prompt template."""
    task = "What is 1+1?"

    prompt = render_official_prompt(task)

    assert prompt == POLARIS_FILTER_SOLVER_USER_TEMPLATE.format(task=task)
    assert prompt.endswith(
        "Let's think step by step and output the final answer within \\boxed{}. "
    )


def test_role_configs_use_student_api_and_actor_teacher_decoding():
    """API routing and decoding parameters must match their actual Polaris roles."""
    student_model = SimpleNamespace(
        base_url="https://student.example/v1",
        api_key="student-key",
        model="qwen3-1.7b",
        timeout=120,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        max_concurrent_calls=4,
        request_params={
            "seed": 42,
            "extra_body": {"top_k": 20, "min_p": 0},
        },
    )
    auxiliary = SimpleNamespace(
        base_url="https://teacher.example/v1",
        api_key="teacher-key",
        model="qwen3-8b",
        timeout=120,
        max_concurrent_calls=4,
        request_params={
            "seed": 42,
            "extra_headers": {"x-inspire-inference-key": "teacher-route"},
            "extra_body": {"top_k": 20, "min_p": 0},
        },
    )
    config = SimpleNamespace(
        dataset_type="polaris",
        answer_scorer="polaris",
        student_models=[student_model],
        auxiliary_model=auxiliary,
        enable_thinking=False,
        gconfig=SimpleNamespace(
            max_new_tokens=16384,
            temperature=0.7,
            top_p=1.0,
            top_k=int(1e8),
        ),
    )
    args = SimpleNamespace(
        student_base_url="",
        teacher_base_url="",
        student_api_key="",
        teacher_api_key="",
        student_model="",
        teacher_model="",
        student_max_tokens=0,
        teacher_max_tokens=0,
        student_temperature=None,
        teacher_temperature=None,
        student_top_p=None,
        teacher_top_p=None,
        student_concurrency=0,
        teacher_concurrency=0,
    )

    student, teacher = build_role_configs(config, args)

    assert (student.model, student.max_tokens, student.top_p) == (
        "qwen3-1.7b",
        2048,
        0.8,
    )
    assert (teacher.model, teacher.max_tokens, teacher.top_p) == (
        "qwen3-8b",
        16384,
        1.0,
    )
    assert teacher.request_params["extra_headers"] == {
        "x-inspire-inference-key": "teacher-route"
    }
    assert teacher.request_params["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.asyncio
async def test_student_first_attempt_correct_skips_remaining_calls():
    """Any correct student attempt must drop the row before calling the teacher."""
    student = FakeClient(generation("student-correct"), generation("unused"))
    teacher = FakeClient(generation("teacher-correct"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=0,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer({"student-correct", "teacher-correct"}),
    )

    assert result.status == "student_solved"
    assert not result.kept
    assert len(student.prompts) == 1
    assert not teacher.prompts


@pytest.mark.asyncio
async def test_student_second_attempt_correct_skips_teacher():
    """The second student attempt is sufficient to drop a task."""
    student = FakeClient(generation("student-wrong"), generation("student-correct"))
    teacher = FakeClient(generation("teacher-correct"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=0,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer({"student-correct", "teacher-correct"}),
    )

    assert result.status == "student_solved"
    assert not result.kept
    assert len(result.student_attempts) == 2
    assert not teacher.prompts


@pytest.mark.asyncio
async def test_student_twice_wrong_teacher_correct_keeps_row():
    """A task is kept only after two student failures and a teacher success."""
    student = FakeClient(generation("wrong-1"), generation("wrong-2"))
    teacher = FakeClient(generation("teacher-correct"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=3,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer({"teacher-correct"}),
    )

    assert result.status == "teacher_solved"
    assert result.kept
    assert len(result.student_attempts) == 2
    assert len(result.teacher_attempts) == 1
    assert student.prompts == teacher.prompts * 2
    assert student.seed_offsets == [6, 7]
    assert teacher.seed_offsets == [3]


@pytest.mark.asyncio
async def test_student_twice_wrong_teacher_wrong_drops_row():
    """A teacher-unsolved task must not enter the filtered dataset."""
    student = FakeClient(generation("wrong-1"), generation("wrong-2"))
    teacher = FakeClient(generation("teacher-wrong"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=0,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer(set()),
    )

    assert result.status == "teacher_unsolved"
    assert not result.kept


@pytest.mark.asyncio
async def test_student_error_does_not_misclassify_task_as_unsolved():
    """An incomplete student pass must be dropped as unknown, not sent to teacher."""
    student = FakeClient(generation(error="timeout"), generation("student-wrong"))
    teacher = FakeClient(generation("teacher-correct"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=0,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer({"teacher-correct"}),
    )

    assert result.status == "student_error"
    assert not result.kept
    assert not teacher.prompts


@pytest.mark.asyncio
async def test_teacher_scoring_error_does_not_keep_row():
    """A Polaris scorer error must not be treated as a teacher failure or success."""
    student = FakeClient(generation("wrong-1"), generation("wrong-2"))
    teacher = FakeClient(generation("teacher-answer"))

    result = await classify_polaris_row(
        row={"id": "row-1", "task": "What is 1+1?", "ground_truth": "2"},
        index=0,
        student_client=student,
        teacher_client=teacher,
        score_fn=scorer(set(), {"teacher-answer"}),
    )

    assert result.status == "teacher_error"
    assert not result.kept
