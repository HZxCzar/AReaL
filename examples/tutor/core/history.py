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
        "leak_masked": trace.leak_masked,
        "teacher_format_error": trace.tutor_format_error,
        "personality_complaint_explained": trace.personality_complaint_explained,
        "personality_gate_terminated": trace.personality_gate_terminated,
        "teacher_exact_repeat": trace.teacher_exact_repeat,
        "public_history_before": trace.public_history_before,
        "public_history_after": trace.public_history_after,
        "student_turn_behavior": (
            asdict(trace.student_turn_behavior)
            if trace.student_turn_behavior is not None
            else None
        ),
        "previous_teacher_similarity": trace.previous_teacher_similarity,
        "teacher_similarity_error": trace.teacher_similarity_error,
        "teacher_progress_judge": (
            asdict(trace.teacher_progress_judge_result)
            if trace.teacher_progress_judge_result is not None
            else None
        ),
        "student_request_judge": (
            asdict(trace.student_request_judge_result)
            if trace.student_request_judge_result is not None
            else None
        ),
    }
    if trace.student_question_generation is not None:
        record["student_question_generation"] = asdict(
            trace.student_question_generation
        )
    if trace.leak_level is not None:
        record["leak_level"] = trace.leak_level
    if leak_result is not None:
        record["leak_feedback"] = leak_result.feedback
    return record


def trace_to_json(trace: TurnTrace) -> dict[str, Any]:
    data = asdict(trace)
    if trace.student_question_generation is None:
        data.pop("student_question_generation", None)
    gate = data.get("personality_gate_result")
    if isinstance(gate, dict):
        for key in (
            "classification_label",
            "classification_logprobs",
            "classification_probabilities",
            "classification_margin",
        ):
            if gate.get(key) is None:
                gate.pop(key, None)
    data["tutor_state"]["ground_truth"] = trace.tutor_state.ground_truth
    return data
