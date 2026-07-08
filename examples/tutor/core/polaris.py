from __future__ import annotations

import json
from typing import Any

from areal.reward import get_math_verify_worker

from .math import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)
from .text import strip_reasoning_for_context
from .types import JudgeResult


def score_polaris_answer(
    task: str, ground_truth: Any, student_answer: str
) -> JudgeResult:
    visible_answer = strip_reasoning_for_context(student_answer)
    extracted_answer = extract_polaris_answer(visible_answer)
    target_answers = extract_polaris_ground_truths(ground_truth)

    if not extracted_answer:
        return _result(
            task=task,
            visible_answer=visible_answer,
            extracted_answer=extracted_answer,
            target_answers=target_answers,
            correct=False,
            format_error="missing_boxed_answer",
            mathd_correct=False,
            sympy_correct=False,
        )
    if not target_answers:
        return _result(
            task=task,
            visible_answer=visible_answer,
            extracted_answer=extracted_answer,
            target_answers=target_answers,
            correct=False,
            format_error="missing_ground_truth",
            mathd_correct=False,
            sympy_correct=False,
        )

    mathd_correct = any(
        grade_answer_mathd(extracted_answer, target) for target in target_answers
    )
    sympy_correct = False
    if not mathd_correct:
        sympy_correct = any(
            grade_answer_sympy(extracted_answer, target) for target in target_answers
        )
    return _result(
        task=task,
        visible_answer=visible_answer,
        extracted_answer=extracted_answer,
        target_answers=target_answers,
        correct=mathd_correct or sympy_correct,
        format_error=None,
        mathd_correct=mathd_correct,
        sympy_correct=sympy_correct,
    )


def extract_polaris_answer(response: str) -> str:
    boxed = last_boxed_only_string(response)
    if boxed is None:
        return ""
    return remove_boxed(boxed)


def extract_polaris_ground_truths(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, list):
        raw_targets = ground_truth
    else:
        raw_targets = [ground_truth]

    targets: list[str] = []
    for raw_target in raw_targets:
        target = str(raw_target)
        boxed = last_boxed_only_string(target)
        if boxed is not None:
            target = remove_boxed(boxed)
        if target.strip():
            targets.append(target)
    return targets


def grade_answer_mathd(model_answer: str, ground_truth: str) -> bool:
    return is_equiv(model_answer, ground_truth)


def grade_answer_sympy(model_answer: str, ground_truth: str) -> bool:
    worker = get_math_verify_worker()
    return bool(worker.verify(f"\\boxed{{{model_answer}}}", str(ground_truth)))


def _result(
    *,
    task: str,
    visible_answer: str,
    extracted_answer: str,
    target_answers: list[str],
    correct: bool,
    format_error: str | None,
    mathd_correct: bool,
    sympy_correct: bool,
) -> JudgeResult:
    raw_result = {
        "method": "polaris_mathd_sympy_rule",
        "task": task,
        "student_answer": visible_answer,
        "extracted_answer": extracted_answer,
        "target_answers": target_answers,
        "normalized_prediction": strip_string(extracted_answer),
        "normalized_targets": [strip_string(target) for target in target_answers],
        "format_error": format_error,
        "mathd_correct": mathd_correct,
        "sympy_correct": sympy_correct,
    }
    return JudgeResult(
        raw_output=json.dumps(
            {
                "correct": correct,
                "feedback": "Correct." if correct else "Incorrect.",
                "scoring": raw_result,
            },
            ensure_ascii=True,
            indent=2,
        ),
        correct=correct,
        feedback="Correct." if correct else "Incorrect.",
        parse_error=None,
        raw_result=raw_result,
    )
