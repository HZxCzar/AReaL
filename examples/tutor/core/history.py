from __future__ import annotations

from dataclasses import asdict
from typing import Any

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
        "reward": trace.reward,
        "reward_components": dict(trace.reward_components),
        "leak_detected": trace.leaked,
        "invalid_due_to_leak": trace.invalid_due_to_leak,
        "teacher_format_error": trace.tutor_format_error,
        "public_history_before": trace.public_history_before,
        "public_history_after": trace.public_history_after,
        "previous_teacher_similarity": trace.previous_teacher_similarity,
        "teacher_similarity_error": trace.teacher_similarity_error,
        "teacher_progress_judge": (
            asdict(trace.teacher_progress_judge_result)
            if trace.teacher_progress_judge_result is not None
            else None
        ),
    }
    if trace.leak_level is not None:
        record["leak_level"] = trace.leak_level
    if leak_result is not None:
        record["leak_feedback"] = leak_result.feedback
    return record


def trace_to_json(trace: TurnTrace) -> dict[str, Any]:
    data = asdict(trace)
    data["tutor_state"]["ground_truth"] = trace.tutor_state.ground_truth
    return data
