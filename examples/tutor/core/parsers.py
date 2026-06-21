from __future__ import annotations

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
        return LeakCheckResult(
            raw_output=raw_output,
            leaked=True,
            feedback="Failed to parse leak-check output.",
            parse_error=parse_error,
            raw_result={},
        )
    leaked = parsed.get("leaked")
    feedback = parsed.get("feedback", "")
    if not isinstance(leaked, bool):
        parse_error = join_errors(parse_error, '"leaked" must be a boolean')
        leaked = True
    if not isinstance(feedback, str):
        parse_error = join_errors(parse_error, '"feedback" must be a string')
        feedback = str(feedback)
    if parse_error:
        leaked = True
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


def _staged_leak_feedback(level: int) -> str:
    if level == 1:
        return "The teacher directly gave the final answer."
    if level == 2:
        return "The teacher did the computation for the student."
    if level == 3:
        return "The teacher gave the needed formula."
    return "No answer leakage detected."


def parse_staged_leak_check_result(raw_output: str) -> LeakCheckResult:
    parsed, parse_error = parse_json_dict(raw_output)
    if parsed is None:
        return LeakCheckResult(
            raw_output=raw_output,
            leaked=True,
            feedback="Failed to parse staged leak-check output.",
            parse_error=parse_error,
            raw_result={},
            leak_level=1,
        )

    level = parsed.get("level")
    feedback = parsed.get("feedback", "")
    if (
        not isinstance(level, int)
        or isinstance(level, bool)
        or level not in {1, 2, 3, 4}
    ):
        parse_error = join_errors(parse_error, '"level" must be an integer from 1 to 4')
        level = 1
    if not isinstance(feedback, str):
        parse_error = join_errors(parse_error, '"feedback" must be a string')
        feedback = str(feedback)
    if parse_error:
        level = 1
        feedback = "Failed to parse staged leak-check output."
    return LeakCheckResult(
        raw_output=raw_output,
        leaked=level != 4,
        feedback=feedback or _staged_leak_feedback(level),
        parse_error=parse_error,
        raw_result=parsed,
        leak_level=level,
    )
