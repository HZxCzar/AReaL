"""The ped teacher's private reasoning section, and that it stays prompt-only.

Without use_thinking the two comparison arms are asymmetric: the tutor teacher
is told to emit <reasoning>...</reasoning> and only <output> reaches the
student, so it has a private scratchpad, while the ped teacher has none and
must reason in the same text the student reads.

Turning it on must not switch the model into native thinking mode -- the
<think> block has to stay ordinary generated text driven by PedagogicalRL's own
teacher prompt.
"""

from __future__ import annotations

import asyncio
import types

from examples.pedagogical_rl.config import PedagogicalGenerationConfig
from examples.pedagogical_rl.state import ClassroomEpisode, student_visible_text
from examples.pedagogical_rl.workflow import PedagogicalRLWorkflow


def _episode(include_thinking: bool) -> ClassroomEpisode:
    return ClassroomEpisode(
        problem="What is 2+2?",
        answer="4",
        include_thinking=include_thinking,
        forced_student_name=None,
    )


def test_thinking_instruction_is_prompt_driven():
    assert "<think>" in _episode(True).teacher_system_prompt
    assert "<think>" not in _episode(False).teacher_system_prompt


def test_the_reasoning_section_stays_private():
    """A scratchpad the student can read is not a scratchpad."""

    raw = "<think>the answer is 4, do not say it</think>What do you get for 2+2?"
    visible = student_visible_text(raw)
    assert visible == "What do you get for 2+2?"
    assert "4" not in visible

    episode = _episode(True)
    episode.add_teacher(raw)
    episode.add_student("Is it 5?")
    student_view = episode.student_messages()
    assert all("do not say it" not in m["content"] for m in student_view)


def test_teacher_call_keeps_the_non_thinking_chat_template():
    """use_thinking must not enable the model's native thinking mode."""

    seen = {}

    class _Completions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="<think>x</think>hi")
                    )
                ]
            )

    workflow = PedagogicalRLWorkflow.__new__(PedagogicalRLWorkflow)
    workflow.generation = PedagogicalGenerationConfig(use_thinking=True)
    workflow.gconfig = types.SimpleNamespace(
        max_tokens=40960, temperature=0.7, top_p=1.0
    )
    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions())
    )

    out = asyncio.run(workflow._teacher_turn(client, _episode(True)))

    assert out == "<think>x</think>hi"
    assert seen["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
