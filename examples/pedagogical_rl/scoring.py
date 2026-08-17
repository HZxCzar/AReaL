from __future__ import annotations

from dataclasses import dataclass

from examples.common.parsing import parse_json_dict
from examples.pedagogical_rl.api import PedagogicalAPIClient
from examples.tutor.core.math import score_math_answer
from examples.tutor.prompts import (
    ANSWER_JUDGE_USER_TEMPLATE,
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    render_prompt,
)


def extract_boxed_answer(solution: str) -> str | None:
    """Replicate PedagogicalRL's last-balanced-``\\boxed{}`` extraction."""

    solution = solution or ""
    last_boxed_start = solution.rfind("\\boxed{")
    if last_boxed_start == -1:
        return None
    start_index = last_boxed_start + len("\\boxed{")
    depth = 1
    for index in range(start_index, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[start_index:index]
    return None


def native_answer_correct(solution: str, ground_truth: str) -> bool:
    """Replicate PedagogicalRL's ``Answer`` reward model exactly."""

    extracted = extract_boxed_answer(solution)
    return str(ground_truth).strip().lower() == str(extracted).strip().lower()


@dataclass(slots=True)
class UnifiedAnswerResult:
    correct: bool
    extracted_answer: str
    exact_correct: bool
    judge_used: bool
    judge_error: str | None = None
    judge_output: str = ""


async def score_unified_answer(
    *,
    task: str,
    ground_truth: str,
    student_answer: str,
    judge: PedagogicalAPIClient,
    judge_max_tokens: int = 256,
) -> UnifiedAnswerResult:
    """Use AReaL's math exact scorer, then its exact answer-judge prompt."""

    exact = score_math_answer(task, ground_truth, student_answer)
    extracted = str(exact.raw_result.get("extracted_answer", ""))
    if exact.correct:
        return UnifiedAnswerResult(
            correct=True,
            extracted_answer=extracted,
            exact_correct=True,
            judge_used=False,
        )

    user_prompt = render_prompt(
        ANSWER_JUDGE_USER_TEMPLATE,
        task=task,
        ground_truth=ground_truth,
        extracted_answer=extracted,
    )
    try:
        outputs = await judge.generate(
            [
                {"role": "system", "content": DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
            max_tokens=judge_max_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        raw_output = outputs[0]
        parsed, parse_error = parse_json_dict(raw_output)
        if parse_error or parsed is None or not isinstance(parsed.get("correct"), bool):
            return UnifiedAnswerResult(
                correct=False,
                extracted_answer=extracted,
                exact_correct=False,
                judge_used=True,
                judge_error=parse_error or '"correct" must be a boolean',
                judge_output=raw_output,
            )
        return UnifiedAnswerResult(
            correct=bool(parsed["correct"]),
            extracted_answer=extracted,
            exact_correct=False,
            judge_used=True,
            judge_output=raw_output,
        )
    except Exception as exc:
        return UnifiedAnswerResult(
            correct=False,
            extracted_answer=extracted,
            exact_correct=False,
            judge_used=True,
            judge_error=f"{type(exc).__name__}: {exc}",
        )
