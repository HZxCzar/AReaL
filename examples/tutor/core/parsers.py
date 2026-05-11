from __future__ import annotations

import re

from examples.common.parsing import join_errors, parse_json_dict

from .text import strip_reasoning_for_context
from .types import LeakCheckResult


def parse_public_summary(raw_output: str) -> str:
    text = strip_reasoning_for_context(raw_output)
    parsed, _ = parse_json_dict(text)
    if isinstance(parsed, dict):
        parts = []
        for key in [
            "student_progress",
            "visible_tutor_guidance",
            "student_current_misconception",
            "latest_student_state",
        ]:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"{key}: {value.strip()}")
        if parts:
            return "\n".join(parts)
    return text.strip()


def parse_leak_check_result(raw_output: str) -> LeakCheckResult:
    parsed, parse_error = parse_json_dict(raw_output)
    if parsed is None:
        lowered = raw_output.lower()
        leaked = '"leaked": true' in lowered or re.search(r"\byes\b", lowered) is not None
        return LeakCheckResult(
            raw_output=raw_output,
            leaked=leaked,
            feedback="Failed to parse leak-check output.",
            parse_error=parse_error,
            raw_result={},
        )
    leaked = parsed.get("leaked")
    feedback = parsed.get("feedback", "")
    if not isinstance(leaked, bool):
        parse_error = join_errors(parse_error, '"leaked" must be a boolean')
        leaked = False
    if not isinstance(feedback, str):
        parse_error = join_errors(parse_error, '"feedback" must be a string')
        feedback = str(feedback)
    return LeakCheckResult(
        raw_output=raw_output,
        leaked=leaked,
        feedback=feedback or (
            "The teacher revealed the answer directly. The student did not see this turn."
            if leaked
            else "No answer leakage detected."
        ),
        parse_error=parse_error,
        raw_result=parsed,
    )
