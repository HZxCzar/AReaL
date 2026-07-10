from __future__ import annotations

import asyncio
import atexit
import json
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from .math import (
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
    strip_string,
)
from .text import strip_reasoning_for_context
from .types import JudgeResult

_POLARIS_MATH_VERIFY_TARGETS = (
    ExprExtractionConfig(try_extract_without_anchor=True),
    LatexExtractionConfig(),
)
_POLARIS_PARSE_TIMEOUT_SECONDS = 5
_POLARIS_VERIFY_TIMEOUT_SECONDS = 5
_POLARIS_SCORE_TIMEOUT_SECONDS = 30

_score_executor: ProcessPoolExecutor | None = None
_score_executor_lock = threading.Lock()


def _get_score_executor() -> ProcessPoolExecutor:
    global _score_executor
    if _score_executor is None:
        with _score_executor_lock:
            if _score_executor is None:
                _score_executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                )
    return _score_executor


def _shutdown_score_executor() -> None:
    global _score_executor
    if _score_executor is not None:
        _score_executor.shutdown(wait=False, cancel_futures=True)
        _score_executor = None


atexit.register(_shutdown_score_executor)


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


async def score_polaris_answer_async(
    task: str, ground_truth: Any, student_answer: str
) -> JudgeResult:
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        _get_score_executor(),
        score_polaris_answer,
        task,
        ground_truth,
        student_answer,
    )
    try:
        return await asyncio.wait_for(future, timeout=_POLARIS_SCORE_TIMEOUT_SECONDS)
    except TimeoutError:
        return _score_without_sympy(
            task,
            ground_truth,
            student_answer,
            scoring_error=(
                "Polaris symbolic scoring timed out after "
                f"{_POLARIS_SCORE_TIMEOUT_SECONDS}s."
            ),
        )
    except Exception as exc:
        return _score_without_sympy(
            task,
            ground_truth,
            student_answer,
            scoring_error=f"Polaris symbolic scoring failed: {exc}",
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
    if threading.current_thread() is not threading.main_thread():
        future = _get_score_executor().submit(
            _grade_answer_sympy_bounded,
            model_answer,
            ground_truth,
        )
        try:
            return bool(future.result(timeout=_POLARIS_SCORE_TIMEOUT_SECONDS))
        except Exception:
            return False
    return _grade_answer_sympy_bounded(model_answer, ground_truth)


def _grade_answer_sympy_bounded(model_answer: str, ground_truth: str) -> bool:
    try:
        parsed_target = parse(
            str(ground_truth),
            _POLARIS_MATH_VERIFY_TARGETS,
            parsing_timeout=_POLARIS_PARSE_TIMEOUT_SECONDS,
        )
        parsed_prediction = parse(
            f"\\boxed{{{model_answer}}}",
            _POLARIS_MATH_VERIFY_TARGETS,
            parsing_timeout=_POLARIS_PARSE_TIMEOUT_SECONDS,
        )
        if not parsed_target or not parsed_prediction:
            return False
        return bool(
            verify(
                parsed_target,
                parsed_prediction,
                float_rounding=6,
                timeout_seconds=_POLARIS_VERIFY_TIMEOUT_SECONDS,
            )
        )
    except Exception:
        return False


def _score_without_sympy(
    task: str,
    ground_truth: Any,
    student_answer: str,
    *,
    scoring_error: str,
) -> JudgeResult:
    visible_answer = strip_reasoning_for_context(student_answer)
    extracted_answer = extract_polaris_answer(visible_answer)
    target_answers = extract_polaris_ground_truths(ground_truth)
    mathd_correct = bool(extracted_answer) and any(
        grade_answer_mathd(extracted_answer, target) for target in target_answers
    )
    return _result(
        task=task,
        visible_answer=visible_answer,
        extracted_answer=extracted_answer,
        target_answers=target_answers,
        correct=mathd_correct,
        format_error=None,
        mathd_correct=mathd_correct,
        sympy_correct=False,
        scoring_error=scoring_error,
    )


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
    scoring_error: str | None = None,
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
    if scoring_error is not None:
        raw_result["scoring_error"] = scoring_error
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
        parse_error=scoring_error,
        raw_result=raw_result,
    )
