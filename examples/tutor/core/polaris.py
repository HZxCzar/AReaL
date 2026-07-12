from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
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


def _polaris_score_worker(connection: Any) -> None:
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                connection.send((True, score_polaris_answer(*request)))
            except BaseException as exc:
                connection.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class PolarisScoreProcess:
    """One sequential, restartable Polaris judge process for one rollout."""

    def __init__(self, timeout_seconds: float = _POLARIS_SCORE_TIMEOUT_SECONDS):
        self.timeout_seconds = float(timeout_seconds)
        self._context = multiprocessing.get_context("spawn")
        self._connection: Any | None = None
        self._process: multiprocessing.Process | None = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()

    def _detach_current(self) -> tuple[Any | None, multiprocessing.Process | None]:
        with self._state_lock:
            connection, self._connection = self._connection, None
            process, self._process = self._process, None
        return connection, process

    @staticmethod
    def _stop_resources(
        connection: Any | None,
        process: multiprocessing.Process | None,
        *,
        wait: bool,
    ) -> None:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is None:
            return
        try:
            if process.pid is not None and process.is_alive():
                process.terminate()
            if wait:
                process.join(timeout=1)
                if process.pid is not None and process.is_alive():
                    process.kill()
                    process.join(timeout=1)
        except (AssertionError, OSError, ValueError):
            pass

    def _clear_if_current(
        self, connection: Any, process: multiprocessing.Process
    ) -> None:
        with self._state_lock:
            if self._connection is connection:
                self._connection = None
            if self._process is process:
                self._process = None

    def _ensure_started(
        self, cancel_event: threading.Event
    ) -> tuple[Any, multiprocessing.Process]:
        with self._state_lock:
            existing_connection = self._connection
            existing_process = self._process
        if (
            existing_connection is not None
            and existing_process is not None
            and existing_process.is_alive()
        ):
            return existing_connection, existing_process
        self._terminate()
        parent_connection, child_connection = self._context.Pipe()
        process = self._context.Process(
            target=_polaris_score_worker,
            args=(child_connection,),
            daemon=True,
        )
        with self._state_lock:
            self._connection = parent_connection
            self._process = process
        try:
            process.start()
        except Exception:
            child_connection.close()
            self._clear_if_current(parent_connection, process)
            self._stop_resources(parent_connection, process, wait=True)
            raise
        child_connection.close()
        if cancel_event.is_set():
            self._clear_if_current(parent_connection, process)
            self._stop_resources(parent_connection, process, wait=True)
            raise TimeoutError(
                f"Polaris scoring timed out after {self.timeout_seconds}s."
            )
        return parent_connection, process

    def _terminate(self) -> None:
        connection, process = self._detach_current()
        self._stop_resources(connection, process, wait=True)

    def _abort(self, cancel_event: threading.Event) -> None:
        cancel_event.set()
        connection, process = self._detach_current()
        self._stop_resources(connection, process, wait=False)

    def _score_blocking(
        self,
        task: str,
        ground_truth: Any,
        student_answer: str,
        cancel_event: threading.Event,
    ) -> JudgeResult:
        with self._lock:
            connection, process = self._ensure_started(cancel_event)
            try:
                connection.send((task, ground_truth, student_answer))
                if not connection.poll(self.timeout_seconds):
                    self._clear_if_current(connection, process)
                    self._stop_resources(connection, process, wait=True)
                    raise TimeoutError(
                        f"Polaris scoring timed out after {self.timeout_seconds}s."
                    )
                succeeded, result = connection.recv()
            except TimeoutError:
                raise
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._clear_if_current(connection, process)
                self._stop_resources(connection, process, wait=True)
                if cancel_event.is_set():
                    raise TimeoutError(
                        f"Polaris scoring timed out after {self.timeout_seconds}s."
                    ) from exc
                raise RuntimeError(
                    "Polaris judge process stopped unexpectedly."
                ) from exc
            if not succeeded:
                raise RuntimeError(f"Polaris judge process failed: {result}")
            return result

    async def score(
        self, task: str, ground_truth: Any, student_answer: str
    ) -> JudgeResult:
        cancel_event = threading.Event()
        score_task = asyncio.create_task(
            asyncio.to_thread(
                self._score_blocking,
                task,
                ground_truth,
                student_answer,
                cancel_event,
            )
        )

        def consume_result(task: asyncio.Task) -> None:
            try:
                task.result()
            except BaseException:
                pass

        try:
            return await asyncio.wait_for(
                asyncio.shield(score_task), timeout=self.timeout_seconds
            )
        except asyncio.CancelledError:
            self._abort(cancel_event)
            score_task.add_done_callback(consume_result)
            raise
        except TimeoutError:
            self._abort(cancel_event)
            score_task.add_done_callback(consume_result)
            raise TimeoutError(
                f"Polaris scoring timed out after {self.timeout_seconds}s."
            ) from None

    def close(self) -> None:
        if not self._lock.acquire(timeout=1):
            self._abort(threading.Event())
            return
        try:
            if self._connection is not None and self._process is not None:
                try:
                    if self._process.is_alive():
                        self._connection.send(None)
                        self._process.join(timeout=1)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._terminate()
        finally:
            self._lock.release()


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
    task: str,
    ground_truth: Any,
    student_answer: str,
    *,
    score_process: PolarisScoreProcess | None = None,
) -> JudgeResult:
    owns_process = score_process is None
    score_process = score_process or PolarisScoreProcess()
    try:
        return await score_process.score(task, ground_truth, student_answer)
    except TimeoutError:
        visible_answer = strip_reasoning_for_context(student_answer)
        return _result(
            task=task,
            visible_answer=visible_answer,
            extracted_answer=extract_polaris_answer(visible_answer),
            target_answers=extract_polaris_ground_truths(ground_truth),
            correct=False,
            format_error=None,
            mathd_correct=False,
            sympy_correct=False,
            scoring_error=(
                "Polaris scoring timed out after "
                f"{score_process.timeout_seconds}s; counted as incorrect."
            ),
        )
    except Exception as exc:
        return _score_without_sympy(
            task,
            ground_truth,
            student_answer,
            scoring_error=f"Polaris symbolic scoring failed: {exc}",
        )
    finally:
        if owns_process:
            await asyncio.to_thread(score_process.close)


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


def _sympy_grade_worker(connection: Any, model_answer: str, ground_truth: str) -> None:
    try:
        connection.send(_grade_answer_sympy_bounded(model_answer, ground_truth))
    finally:
        connection.close()


def grade_answer_sympy(model_answer: str, ground_truth: str) -> bool:
    if threading.current_thread() is not threading.main_thread():
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_sympy_grade_worker,
            args=(child_connection, model_answer, ground_truth),
            daemon=True,
        )
        process.start()
        child_connection.close()
        try:
            if not parent_connection.poll(_POLARIS_SCORE_TIMEOUT_SECONDS):
                return False
            return bool(parent_connection.recv())
        except (EOFError, OSError):
            return False
        finally:
            parent_connection.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
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
