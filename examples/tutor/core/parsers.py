from __future__ import annotations

import re

from examples.common.parsing import join_errors, parse_json_dict

from .text import strip_reasoning_for_context
from .types import LeakCheckResult, ProgressJudgment, ProgressLabel


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


def parse_progress_judgment(raw_output: str) -> ProgressJudgment:
    parsed, parse_error = parse_json_dict(raw_output)
    if parsed is None:
        lowered = raw_output.lower()
        if "improved" in lowered:
            label: ProgressLabel = "improved"
        elif "regressed" in lowered or "worse" in lowered:
            label = "regressed"
        elif "same" in lowered or "unchanged" in lowered:
            label = "same"
        else:
            label = "unknown"
        return ProgressJudgment(
            raw_output=raw_output,
            label=label,
            confidence="low",
            feedback="Failed to parse progress-judge output.",
            parse_error=parse_error,
            raw_result={},
        )
    label_value = parsed.get("label", parsed.get("progress", "unknown"))
    if label_value not in {"improved", "same", "regressed", "unknown"}:
        parse_error = join_errors(
            parse_error,
            '"label" must be one of improved, same, regressed, unknown',
        )
        label_value = "unknown"
    confidence_value = parsed.get("confidence", "low")
    if confidence_value not in {"high", "medium", "low"}:
        confidence_value = "low"
    feedback = parsed.get("feedback", "")
    if not isinstance(feedback, str):
        feedback = str(feedback)
    return ProgressJudgment(
        raw_output=raw_output,
        label=label_value,
        confidence=confidence_value,
        feedback=feedback or "No progress feedback provided.",
        parse_error=parse_error,
        raw_result=parsed,
    )


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
