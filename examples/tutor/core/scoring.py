from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .types import JudgeResult

AnswerScorerName = Literal["aime", "math", "polaris"]
AnswerScorer = Callable[[str, str, str], JudgeResult]


def get_answer_scorer(name: str) -> AnswerScorer:
    normalized = name.strip().lower()
    if normalized == "aime":
        from .aime import score_aime_answer

        return score_aime_answer
    if normalized == "math":
        from .math import score_math_answer

        return score_math_answer
    if normalized == "polaris":
        from .polaris import score_polaris_answer

        return score_polaris_answer
    raise ValueError("answer_scorer must be one of: aime, math, polaris")
