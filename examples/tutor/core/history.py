from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .text import strip_reasoning_for_context
from .types import LeakCheckResult, TurnTrace


def trace_to_history_record(
    trace: TurnTrace,
    leak_result: LeakCheckResult | None,
    student_error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "round_idx": trace.turn_idx,
        "teacher_raw_output": trace.tutor_raw_output,
        "teacher_action": trace.tutor_visible_output,
        "student_answer": trace.student_output,
        "student_error": student_error,
        "judge_feedback": trace.judge_feedback,
        "judge_correct": trace.judge_correct,
        "progress_label": trace.progress.label,
        "progress_feedback": trace.progress.feedback,
        "reward": trace.reward,
        "reward_components": dict(trace.reward_components),
        "leak_detected": trace.leaked,
        "public_history_before": trace.public_history_before,
        "public_history_after": trace.public_history_after,
    }
    if leak_result is not None:
        record["leak_feedback"] = leak_result.feedback
    return record


def trace_to_json(trace: TurnTrace) -> dict[str, Any]:
    data = asdict(trace)
    data["tutor_state"]["ground_truth"] = trace.tutor_state.ground_truth
    return data


def student_visible_history_summaries(history: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for record in history:
        if record.get("leak_detected"):
            continue
        summary = record.get("public_history_after") or record.get(
            "student_visible_summary"
        )
        if isinstance(summary, str) and summary.strip():
            summaries.append(strip_reasoning_for_context(summary))
    return summaries


def latest_visible_student_answer(history: list[dict[str, Any]]) -> str:
    for record in reversed(history):
        if not record.get("leak_detected") and record.get("student_answer"):
            return strip_reasoning_for_context(str(record["student_answer"]))
    return ""
