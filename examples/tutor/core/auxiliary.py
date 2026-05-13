from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from examples.tutor.core.text import strip_reasoning_for_context


@dataclass(slots=True)
class AuxiliaryModelCallResult:
    text: str
    error: str | None = None


async def call_auxiliary_text(
    caller: Any, messages: list[dict[str, str]]
) -> AuxiliaryModelCallResult:
    try:
        raw_output = await caller.call_text(messages)
    except Exception as exc:
        return AuxiliaryModelCallResult(text="", error=str(exc))
    return AuxiliaryModelCallResult(
        text=strip_reasoning_for_context(raw_output),
        error=None,
    )
