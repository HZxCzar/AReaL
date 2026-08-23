"""Regression tests for resampling a teacher turn that ran to the generation cap.

A draft that stops on 'length' closed no tags, so it is a format error however it
reads, and under a token-mean loss it carries its whole length into the gradient.
On 20260822_133603 those capped turns took 1.7% of the batch gradient, then 14.1%,
then 47.7% over two steps, and the policy did not recover. Retrying is the only
remedy that keeps the sample out of the batch rather than making it count for less.

What must hold: off is byte-identical to the behaviour from before the switch
existed, a clean draft costs nothing extra, and an exhausted retry falls back to
the old path rather than inventing a new one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from examples.tutor.workflow import TutorAgentWorkflow


def _caller(stop_reasons: list[str]):
    """An actor caller that reports stop_reasons in order, repeating the last."""
    calls: list[str] = []

    async def generate(messages, **kwargs):
        calls.append(kwargs["rid_prefix"])
        reason = stop_reasons[min(len(calls) - 1, len(stop_reasons) - 1)]
        return SimpleNamespace(
            response=SimpleNamespace(stop_reason=reason),
            raw_text=f"draft{len(calls)}",
        )

    return SimpleNamespace(generate=generate), calls


def _run(enabled: bool, attempts: int, stop_reasons: list[str]):
    workflow = object.__new__(TutorAgentWorkflow)
    workflow.length_retry_enabled = enabled
    workflow.length_retry_attempts = attempts
    workflow._build_tutor_messages = lambda state: []
    workflow._clean_teacher_input_token_reserve = lambda m, sel, guide: 0

    caller, calls = _caller(stop_reasons)
    state = SimpleNamespace(
        turn_idx=3, teacher_prompt_selection=None, guidance=None
    )
    response, raw_text = asyncio.run(
        TutorAgentWorkflow._generate_tutor_response(
            workflow, state, actor_caller=caller
        )
    )
    return response, raw_text, calls


def test_disabled_keeps_the_capped_draft_and_the_original_request_id():
    """Off must not perturb anything, including the rid the engine sees."""
    response, raw_text, calls = _run(False, 3, ["length"])

    assert calls == ["tutor-3"]
    assert raw_text == "draft1"
    assert response.stop_reason == "length"


def test_a_clean_draft_costs_no_extra_generation():
    _response, raw_text, calls = _run(True, 3, ["stop"])

    assert calls == ["tutor-3"]
    assert raw_text == "draft1"


def test_a_capped_draft_is_resampled_and_discarded():
    """The retry is what keeps the 4096-token draft out of the training batch."""
    response, raw_text, calls = _run(True, 3, ["length", "stop"])

    assert calls == ["tutor-3", "tutor-3-len1"]
    # The returned draft is the clean one; the capped draft reaches nothing.
    assert raw_text == "draft2"
    assert response.stop_reason == "stop"


def test_exhausted_retries_fall_back_to_the_last_draft():
    """Once spent this changes nothing -- normal format handling takes over."""
    response, raw_text, calls = _run(True, 3, ["length"])

    assert calls == ["tutor-3", "tutor-3-len1", "tutor-3-len2"]
    assert raw_text == "draft3"
    assert response.stop_reason == "length"


def test_attempts_of_one_disables_retrying_even_when_enabled():
    _response, _raw_text, calls = _run(True, 1, ["length"])

    assert calls == ["tutor-3"]
