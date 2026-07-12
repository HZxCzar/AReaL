from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import TutorAuxiliaryModelConfig, TutorConfig
from examples.tutor.core import polaris
from examples.tutor.core.polaris import (
    grade_answer_sympy,
    score_polaris_answer,
    score_polaris_answer_async,
)
from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.core.types import PublicHistoryState, StudentTurnState
from examples.tutor.data_formats.polaris import (
    build_polaris_splits_from_rows,
    polaris_item_to_tutor_row,
)
from examples.tutor.prompts import POLARIS_INSTRUCTION
from examples.tutor.workflow import TutorAgentWorkflow


def test_tutor_config_requires_dataset_type():
    with pytest.raises(ValueError, match="dataset_type"):
        TutorConfig()


def test_tutor_config_auto_scorer_follows_dataset_type():
    math_config = TutorConfig(dataset_type="math")
    polaris_config = TutorConfig(
        dataset_type="polaris",
        leak_handling_mode="disabled",
    )

    assert math_config.answer_scorer == "math"
    assert polaris_config.answer_scorer == "polaris"


def test_tutor_config_rejects_mismatched_explicit_scorer():
    with pytest.raises(ValueError, match="match dataset_type"):
        TutorConfig(dataset_type="math", answer_scorer="aime")


def test_workflow_rejects_mismatched_explicit_scorer():
    with pytest.raises(ValueError, match="match dataset_type"):
        TutorAgentWorkflow(dataset_type="math", answer_scorer="aime")


def test_tutor_config_rejects_polaris_leak_checks():
    with pytest.raises(ValueError, match="incompatible with leak checks"):
        TutorConfig(dataset_type="polaris")


def test_tutor_config_rejects_polaris_llm_answer_judge():
    with pytest.raises(ValueError, match="answer_judge_enabled"):
        TutorConfig(
            dataset_type="polaris",
            leak_handling_mode="disabled",
            auxiliary_model=TutorAuxiliaryModelConfig(answer_judge_enabled=True),
        )


def test_polaris_scorer_uses_boxed_mathd_equivalence():
    result = score_polaris_answer(
        "Compute one half.",
        "\\frac{1}{2}",
        "The final answer is \\boxed{1/2}.",
    )

    assert result.correct
    assert result.raw_result["method"] == "polaris_mathd_sympy_rule"
    assert result.raw_result["mathd_correct"]


@pytest.mark.asyncio
async def test_polaris_process_round_trip():
    score_process = polaris.PolarisScoreProcess()
    try:
        result = await score_process.score("Compute 1+1.", "2", "\\boxed{2}")
    finally:
        await asyncio.to_thread(score_process.close)

    assert result.correct


def test_polaris_sympy_fallback_is_thread_safe():
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(grade_answer_sympy, "1/2", "\\frac{1}{2}")

    assert future.result()


def test_polaris_sympy_fallback_uses_bounded_math_verify(monkeypatch):
    parse_timeouts = []
    verify_timeouts = []

    def fake_parse(value, extraction_config, *, parsing_timeout):
        del extraction_config
        parse_timeouts.append(parsing_timeout)
        return [value]

    def fake_verify(gold, target, *, float_rounding, timeout_seconds):
        del gold, target, float_rounding
        verify_timeouts.append(timeout_seconds)
        return True

    monkeypatch.setattr(polaris, "parse", fake_parse)
    monkeypatch.setattr(polaris, "verify", fake_verify)

    assert grade_answer_sympy("1/2", "\\frac{1}{2}")
    assert parse_timeouts == [
        polaris._POLARIS_PARSE_TIMEOUT_SECONDS,
        polaris._POLARIS_PARSE_TIMEOUT_SECONDS,
    ]
    assert verify_timeouts == [polaris._POLARIS_VERIFY_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_polaris_async_scorer_does_not_block_event_loop(monkeypatch):
    expected = score_polaris_answer("Compute one half.", "1/2", "\\boxed{1/2}")
    score_process = polaris.PolarisScoreProcess()

    def slow_score(task, ground_truth, student_answer, cancel_event):
        del task, ground_truth, student_answer, cancel_event
        time.sleep(0.1)
        return expected

    monkeypatch.setattr(score_process, "_score_blocking", slow_score)
    try:
        score_task = asyncio.create_task(
            score_polaris_answer_async(
                "task", "answer", "student", score_process=score_process
            )
        )
        await asyncio.sleep(0.01)

        assert not score_task.done()
        assert await score_task == expected
    finally:
        score_process.close()


@pytest.mark.asyncio
async def test_polaris_async_scorer_timeout_counts_answer_as_incorrect(monkeypatch):
    score_process = polaris.PolarisScoreProcess(timeout_seconds=0.01)

    def stuck_score(task, ground_truth, student_answer, cancel_event):
        del task, ground_truth, student_answer, cancel_event
        time.sleep(0.1)
        raise AssertionError("late scorer result should be ignored")

    monkeypatch.setattr(score_process, "_score_blocking", stuck_score)
    try:
        result = await score_polaris_answer_async(
            "Compute one half.",
            "\\frac{1}{2}",
            "The final answer is \\boxed{1/2}.",
            score_process=score_process,
        )

        assert not result.correct
        assert not result.raw_result["mathd_correct"]
        assert "timed out" in result.raw_result["scoring_error"]
    finally:
        score_process.close()


@pytest.mark.asyncio
async def test_workflow_uses_async_polaris_scorer(monkeypatch):
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.dataset_type = "polaris"
    workflow.answer_judge_enabled = False
    expected = score_polaris_answer("task", "7", "\\boxed{7}")
    received_process = None

    async def fake_async_score(task, ground_truth, student_answer, *, score_process):
        nonlocal received_process
        assert (task, ground_truth, student_answer) == ("task", "7", "\\boxed{7}")
        received_process = score_process
        return expected

    monkeypatch.setattr(polaris, "score_polaris_answer_async", fake_async_score)

    result = await workflow._score_answer_async(
        "task",
        "7",
        "\\boxed{7}",
        answer_judge_caller=None,
    )

    assert result == expected
    assert received_process is None


@pytest.mark.asyncio
async def test_concurrent_episodes_use_separate_polaris_processes(monkeypatch):
    created = []

    class FakeScoreProcess:
        def __init__(self):
            self.closed = False
            created.append(self)

        def close(self):
            time.sleep(0.05)
            self.closed = True

    monkeypatch.setattr(polaris, "PolarisScoreProcess", FakeScoreProcess)
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.dataset_type = "polaris"
    observed = []
    both_started = asyncio.Event()

    async def fake_run_episode(data, **_kwargs):
        observed.append((data["id"], tutor_workflow._POLARIS_SCORE_PROCESS.get()))
        if len(observed) == 2:
            both_started.set()
        await both_started.wait()
        return None

    workflow._run_episode = fake_run_episode
    heartbeat_ran = False

    async def heartbeat():
        nonlocal heartbeat_ran
        await asyncio.sleep(0.01)
        heartbeat_ran = True

    await asyncio.gather(
        workflow.arun_episode(object(), {"id": 1}),
        workflow.arun_episode(object(), {"id": 2}),
        heartbeat(),
    )

    assert observed[0][1] is not observed[1][1]
    assert len(created) == 2
    assert all(process.closed for process in created)
    assert heartbeat_ran


def test_polaris_timeout_terminates_only_its_process():
    class Connection:
        closed = False

        @staticmethod
        def send(_request):
            return None

        @staticmethod
        def poll(_timeout):
            return False

        def close(self):
            self.closed = True

    class Process:
        alive = True
        terminated = False
        pid = 1

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout):
            del timeout

        def kill(self):
            self.alive = False

    score_process = polaris.PolarisScoreProcess(timeout_seconds=0.01)
    connection = Connection()
    process = Process()
    score_process._connection = connection
    score_process._process = process

    with pytest.raises(TimeoutError, match="timed out"):
        score_process._score_blocking("task", "answer", "student", threading.Event())

    assert connection.closed
    assert process.terminated
    assert score_process._connection is None
    assert score_process._process is None

    score_process.timeout_seconds = polaris._POLARIS_SCORE_TIMEOUT_SECONDS
    restarted_result = asyncio.run(
        score_process.score("Compute 1+1.", "2", "\\boxed{2}")
    )
    score_process.close()
    assert restarted_result.correct


def test_polaris_scorer_strips_thinking_before_extracting_answer():
    result = score_polaris_answer(
        "Pick the visible answer.",
        "7",
        "<thinking>Hidden draft says \\boxed{42}.</thinking>\n"
        "Visible final answer: \\boxed{7}.",
    )

    assert result.correct
    assert "Hidden draft" not in result.raw_result["student_answer"]
    assert result.raw_result["extracted_answer"] == "7"


def test_polaris_scorer_requires_visible_boxed_answer():
    result = score_polaris_answer(
        "Compute six plus one.",
        "7",
        "The final answer is 7.",
    )

    assert not result.correct
    assert result.raw_result["format_error"] == "missing_boxed_answer"


def test_strip_reasoning_for_context_handles_think_and_thinking_tags():
    assert strip_reasoning_for_context("<think>hidden</think>visible") == "visible"
    assert (
        strip_reasoning_for_context("<thinking>hidden</thinking>visible") == "visible"
    )


def test_polaris_converter_maps_rows_and_splits():
    rows = [
        polaris_item_to_tutor_row(
            {"problem": f"Problem {idx}", "answer": str(idx), "difficulty": "easy"},
            fallback_id=f"row-{idx}",
        )
        for idx in range(3)
    ]

    train_rows, test_rows = build_polaris_splits_from_rows(rows, test_size=1)

    assert rows[0] == {
        "id": "row-0",
        "task": "Problem 0",
        "ground_truth": "0",
        "metadata": {"source": "polaris", "difficulty": "easy"},
    }
    assert [row["id"] for row in train_rows] == ["row-0", "row-1"]
    assert [row["id"] for row in test_rows] == ["row-2"]


def test_polaris_initial_student_prompt_matches_official_instruction():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.dataset_type = "polaris"

    prompt = workflow._build_student_prompt_from_state(
        StudentTurnState(
            task="Find x.",
            public_history=PublicHistoryState(),
            previous_student_output="",
            latest_tutor_visible_output="",
        )
    )

    assert prompt == f"Find x.\n\n{POLARIS_INSTRUCTION}"


def test_polaris_teacher_pre_solve_prompt_matches_official_instruction():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.dataset_type = "polaris"

    prompt = workflow._build_teacher_pre_solve_prompt(task="Find x.")

    assert prompt == f"Find x.\n\n{POLARIS_INSTRUCTION}"


def test_default_teacher_pre_solve_prompt_keeps_existing_template():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)

    prompt = workflow._build_teacher_pre_solve_prompt(task="Find x.")

    assert prompt == (
        "Task:\nFind x.\n\nSolve the problem. Put your final answer in \\boxed{}.\n"
    )


def test_non_initial_polaris_student_prompt_keeps_tutoring_template():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.dataset_type = "polaris"

    prompt = workflow._build_student_prompt_from_state(
        StudentTurnState(
            task="Find x.",
            public_history=PublicHistoryState(summary="Student round 0", turn_count=1),
            previous_student_output="previous",
            latest_tutor_visible_output="hint",
        )
    )

    assert "Public conversation history:" in prompt
    assert POLARIS_INSTRUCTION not in prompt
