from types import SimpleNamespace

import pytest

from examples.tutor.evaluate_teacher_api import (
    prepare_provider_request,
)
from examples.tutor.workflow import TutorAgentWorkflow


@pytest.fixture
def workflow():
    # Prompt/parser tests need no model servers, tokenizer, or API clients.
    w = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    w.teacher_response_format = "thinking"
    w.enable_thinking = True
    w.teacher_end_enabled = True
    w.free_chat_enabled = True
    w.teacher_anti_leak_instruction_enabled = True
    return w


@pytest.mark.parametrize(
    "text,expected",
    [
        (" Try factoring. ", ("Try factoring.", False, None)),
        ("先设 x > 0。\n再考虑 x < 2。", ("先设 x > 0。\n再考虑 x < 2。", False, None)),
        ("Try **factoring**.", ("Try **factoring**.", False, None)),
        (" <end> ", ("", True, None)),
    ],
)
def test_thinking_actions(workflow, text, expected):
    assert workflow._parse_tutor_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<reasoning>private</reasoning><output>hint</output>",
        "<output>hint <think>private</think></output>",
        "<output></output>",
        "<output>hint</output><end></end>",
        "<output>truncated",
        "",
        "  \n ",
        "hint <end>",
        "<think>private</think>hint",
        "<output>hint</output>",
        "<end></end>",
    ],
)
def test_malformed_output_never_becomes_student_text(workflow, text):
    visible, ended, error = workflow._parse_tutor_action(text)
    assert visible == "" and not ended and error


@pytest.mark.parametrize("history_tags", ["stripped", "masked", "unmasked"])
def test_thinking_history_preserves_feedback_without_reasoning(workflow, history_tags):
    turns = [
        {"role": "teacher", "content": "hint"},
        {"role": "student", "content": "reply", "env": "preference complaint"},
    ]
    workflow.teacher_history_tags = history_tags
    workflow.free_chat_enabled = False
    workflow._teacher_system_for_state = lambda state, clean=False: "system"
    state = SimpleNamespace(
        guidance=None,
        public_history=SimpleNamespace(turns=turns),
        previous_tutor_raw_outputs=(
            "<reasoning>SECRET</reasoning><output>hint</output>",
        ),
    )
    messages = workflow._build_tutor_messages(state)
    assert messages[1:] == [
        {"role": "assistant", "content": "hint"},
        {"role": "user", "content": "reply\n\npreference complaint"},
    ]
    prompt = workflow._free_chat_open_prompt()
    assert "Reply directly to the student" in prompt
    assert "<output>" not in prompt
    assert "first use this tagged section" not in prompt
    assert "Do not reveal" in prompt


@pytest.mark.parametrize("native_thinking", [False, True])
def test_thinking_format_independent_of_native_switch(workflow, native_thinking):
    """The format contract is independent of backend reasoning settings."""
    workflow.enable_thinking = native_thinking
    assert workflow._parse_tutor_action("hint") == (
        "hint",
        False,
        None,
    )
    workflow.teacher_end_enabled = False
    assert workflow._parse_tutor_action("<end>")[2]
    assert "<end>" not in workflow._teacher_output_format_prompt()


def test_default_format_keeps_original_prompt_and_parser(workflow):
    """Absent new config, the existing non-thinking protocol is unchanged."""
    from examples.tutor.configs import TutorConfig
    from examples.tutor.prompts import (
        FREE_CHAT_TEACHER_OPEN_PROMPT,
        NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
        TEACHER_ANTI_LEAK_INSTRUCTION,
    )

    assert (
        TutorConfig.__dataclass_fields__["teacher_response_format"].default
        == "non_thinking"
    )
    del workflow.teacher_response_format
    workflow.enable_thinking = False
    assert workflow._free_chat_open_prompt() == "\n\n".join(
        [
            FREE_CHAT_TEACHER_OPEN_PROMPT,
            NON_THINKING_TEACHER_OUTPUT_FORMAT_WITH_END_PROMPT,
            TEACHER_ANTI_LEAK_INSTRUCTION,
        ]
    )
    assert workflow._parse_tutor_action(
        "<reasoning>private</reasoning><output>hint</output>"
    ) == ("hint", False, None)
    assert workflow._parse_tutor_action("<reasoning>done</reasoning><end></end>") == (
        "",
        True,
        None,
    )
    assert workflow._parse_tutor_action("<end>")[2]


def test_provider_parameters_preserve_messages_and_explicit_budget():
    messages = [{"role": "user", "content": "original prompt"}]
    original = dict(
        messages=messages, max_completion_tokens=2048, temperature=1.0, top_p=1.0
    )
    gemini = prepare_provider_request(
        original,
        provider="gemini",
        effort="medium",
        sampling="config",
        thinking_reserve=4096,
        output_limit="config",
    )
    assert gemini == dict(
        messages=messages,
        max_tokens=6144,
        temperature=1.0,
        top_p=1.0,
        reasoning_effort="medium",
    )
    luna = prepare_provider_request(
        original,
        provider="openai",
        effort="medium",
        sampling="provider-default",
        thinking_reserve=0,
        output_limit="config",
    )
    assert luna == dict(
        messages=messages, max_completion_tokens=2048, reasoning_effort="medium"
    )
    assert original["max_completion_tokens"] == 2048


@pytest.mark.parametrize("provider", ["generic", "openai", "gemini"])
@pytest.mark.parametrize("budget", [2048, 4096])
def test_default_teacher_requests_omit_all_output_limits(provider, budget):
    """Teaching and pre-solve omit caps without mutating messages or input params."""
    request = dict(
        messages=[{"role": "user", "content": "problem"}],
        max_completion_tokens=budget,
        max_tokens=budget,
        max_output_tokens=budget,
        model="teacher",
    )
    result = prepare_provider_request(
        request,
        provider=provider,
        effort="medium",
        sampling="config",
        thinking_reserve=0,
    )
    assert result == dict(
        messages=request["messages"], model="teacher", reasoning_effort="medium"
    )
    assert request["max_completion_tokens"] == budget
