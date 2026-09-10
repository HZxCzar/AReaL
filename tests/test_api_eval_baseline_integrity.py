"""API-eval baselines must not silently drop failed measurements."""

import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.tutor.scripts.evaluate_api_teacher import RecordingTutorWorkflow
from examples.tutor.workflow import TutorAgentWorkflow


@pytest.fixture
def baseline_workflow(monkeypatch):
    workflow = RecordingTutorWorkflow.__new__(RecordingTutorWorkflow)
    workflow.free_chat_no_teaching_baseline = True
    workflow.free_chat_enabled = True
    workflow.student_generalize_replays = 8
    workflow._no_teaching_baselines = {}
    workflow._no_teaching_baseline_lock = asyncio.Lock()
    workflow.answer_judge_used_count = 0
    workflow.answer_judge_failed_count = 0
    workflow.answer_judge_override_correct_count = 0
    workflow._free_chat_student_prompts = lambda mode: ("system", "{{ task }}")

    async def score(self, task, ground_truth, text, **kwargs):
        return SimpleNamespace(correct=text == "42", raw_result={})

    monkeypatch.setattr(TutorAgentWorkflow, "_score_answer_async", score)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    return workflow


DATA = {"id": "baseline-test", "task": "Find the answer", "answer": "42"}


@pytest.mark.asyncio
async def test_baseline_retries_only_failed_replay_and_keeps_eight_samples(
    baseline_workflow,
):
    """A transient 503 must not change the baseline denominator from 8 to 7."""
    counts = Counter()

    async def call_text(messages, *, rid_prefix):
        counts[rid_prefix] += 1
        failed = rid_prefix.endswith("r1") and counts[rid_prefix] == 1
        return SimpleNamespace(
            text="wrong" if rid_prefix.endswith("r1") else "42",
            error="503 unavailable" if failed else None,
        )

    caller = SimpleNamespace(call_text=call_text)
    baseline = await baseline_workflow._no_teaching_baseline(
        DATA, aux_caller=caller, answer_judge_caller=None
    )
    assert baseline == 7 / 8
    assert counts["no-teaching-baseline-r1"] == 2
    assert sum(counts.values()) == 9
    assert (
        await baseline_workflow._no_teaching_baseline(
            DATA, aux_caller=caller, answer_judge_caller=None
        )
        == baseline
    )
    assert sum(counts.values()) == 9  # Only complete baselines are cached.


@pytest.mark.asyncio
async def test_baseline_persistent_failure_raises_without_caching_partial_result(
    baseline_workflow,
):
    """Exhausted retries leave an explicit episode failure, not a 7-sample score."""
    counts = Counter()

    async def call_text(messages, *, rid_prefix):
        counts[rid_prefix] += 1
        return SimpleNamespace(
            text="42", error="503" if rid_prefix.endswith("r1") else None
        )

    with pytest.raises(RuntimeError, match="Incomplete no-teaching baseline"):
        await baseline_workflow._no_teaching_baseline(
            DATA,
            aux_caller=SimpleNamespace(call_text=call_text),
            answer_judge_caller=None,
        )
    assert counts["no-teaching-baseline-r1"] == 3
    assert baseline_workflow._no_teaching_baselines == {}

    # Context is restored even on failure: unrelated calls keep their contract.
    caller = SimpleNamespace(
        call_text=AsyncMock(return_value=SimpleNamespace(text="", error="503"))
    )
    result = await baseline_workflow._call_auxiliary_messages([], aux_caller=caller)
    assert result.error == "503"
    assert caller.call_text.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_baseline_judge_failure_retries_or_invalidates_measurement(
    baseline_workflow, monkeypatch, recover
):
    """Judge errors cannot silently become valid baseline labels either."""
    counts = Counter()

    async def call_text(messages, *, rid_prefix):
        return SimpleNamespace(text=rid_prefix, error=None)

    async def score(self, task, ground_truth, text, **kwargs):
        counts[text] += 1
        failed = text.endswith("r1") and (not recover or counts[text] == 1)
        return SimpleNamespace(
            correct=not failed,
            raw_result={
                "answer_judge": {
                    "enabled": True,
                    "used": not failed,
                    "error": "503" if failed else None,
                }
            },
        )

    monkeypatch.setattr(TutorAgentWorkflow, "_score_answer_async", score)
    operation = baseline_workflow._no_teaching_baseline(
        DATA,
        aux_caller=SimpleNamespace(call_text=call_text),
        answer_judge_caller=object(),
    )
    if recover:
        assert await operation == 1.0
        assert counts["no-teaching-baseline-r1"] == 2
        assert baseline_workflow.answer_judge_used_count == 8
        assert baseline_workflow.answer_judge_failed_count == 0
    else:
        with pytest.raises(RuntimeError, match="answer judge failed after 3"):
            await operation
        assert counts["no-teaching-baseline-r1"] == 3
        assert baseline_workflow._no_teaching_baselines == {}


@pytest.mark.asyncio
async def test_baseline_retry_scope_does_not_leak_to_concurrent_calls(
    baseline_workflow,
):
    """Parallel unrelated eval tasks must not inherit baseline retry policy."""
    caller = SimpleNamespace(
        call_text=AsyncMock(return_value=SimpleNamespace(text="", error="503"))
    )
    baseline, unrelated = await asyncio.gather(
        baseline_workflow._no_teaching_baseline(
            DATA, aux_caller=caller, answer_judge_caller=None
        ),
        baseline_workflow._call_auxiliary_messages([], aux_caller=caller),
        return_exceptions=True,
    )
    assert isinstance(baseline, RuntimeError)
    assert unrelated.error == "503"
    assert caller.call_text.await_count == 8 * 3 + 1
