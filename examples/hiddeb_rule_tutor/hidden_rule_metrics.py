from __future__ import annotations

import re
from typing import Any


FORBIDDEN_TUTOR_PATTERNS = [
    "final_rule",
    "the rule is",
    "a string is valid if",
    "you are correct",
    "you're correct",
    "correctly identified",
    "you identified",
    "your rule is",
    "valid password",
]


def _words(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2}


def tutor_leakage_penalty(reply: str, batch: list[dict[str, object]] | None = None) -> float:
    lowered = reply.lower()
    penalty = 1.0 if any(token in lowered for token in FORBIDDEN_TUTOR_PATTERNS) else 0.0
    if batch is not None and not any(str(item["x"]) in reply for item in batch):
        penalty = max(penalty, 0.5)
    return penalty


def tutor_turn_quality(reply: str, batch: list[dict[str, object]] | None = None) -> float:
    lowered = reply.lower()
    words = _words(reply)
    evidence = 0.0
    if batch:
        shown = sum(1 for item in batch if str(item["x"]) in reply)
        evidence = shown / max(1, len(batch))
    elif "valid" in lowered and "invalid" in lowered:
        evidence = 0.5

    contrast_terms = {
        "contrast",
        "compare",
        "similar",
        "different",
        "distinguish",
        "alternative",
        "hypothesis",
    }
    feature_terms = {
        "length",
        "vowel",
        "consonant",
        "repeat",
        "repeated",
        "double",
        "first",
        "last",
        "order",
        "position",
        "third",
        "letter",
    }
    test_terms = {"test", "request", "example", "batch", "change"}
    uncertainty_terms = {"uncertain", "maybe", "candidate", "could", "might", "rule out"}

    contrast = 1.0 if words & contrast_terms else 0.0
    feature = min(1.0, len(words & feature_terms) / 2.0)
    next_test = 1.0 if words & test_terms else 0.0
    uncertainty = 1.0 if any(term in lowered for term in uncertainty_terms) else 0.0
    length = min(1.0, len(reply.split()) / 90.0)

    return (
        0.35 * evidence
        + 0.20 * contrast
        + 0.20 * feature
        + 0.15 * next_test
        + 0.05 * uncertainty
        + 0.05 * length
    )


def transcript_safety_and_quality(transcript: list[dict[str, str]]) -> tuple[float, float]:
    if not transcript:
        return 0.0, 0.0
    qualities = [tutor_turn_quality(turn.get("tutor", "")) for turn in transcript]
    leaks = [tutor_leakage_penalty(turn.get("tutor", "")) for turn in transcript]
    return sum(qualities) / len(qualities), sum(leaks) / len(leaks)


def hidden_rule_tutoring_score(
    *,
    heldout_accuracy: float,
    rule_matched: bool,
    examples_used: int,
    max_examples: int,
    turns_taken: int,
    max_turns: int,
    tutor_quality: float,
    leakage_penalty: float,
) -> float:
    example_fraction = examples_used / max(1, max_examples)
    turn_fraction = turns_taken / max(1, max_turns)
    return (
        float(heldout_accuracy)
        + (0.15 if rule_matched else 0.0)
        + 0.20 * float(tutor_quality)
        - 0.08 * example_fraction
        - 0.04 * turn_fraction
        - 0.45 * float(leakage_penalty)
    )


def summarize_episode(
    result: Any,
    *,
    rounds: int,
) -> dict[str, float]:
    tutor_quality, leakage = transcript_safety_and_quality(result.transcript)
    score = hidden_rule_tutoring_score(
        heldout_accuracy=float(result.heldout_accuracy),
        rule_matched=bool(result.rule_matched),
        examples_used=int(result.examples_used),
        max_examples=max(1, rounds * 4),
        turns_taken=int(result.turns_taken),
        max_turns=max(1, rounds),
        tutor_quality=tutor_quality,
        leakage_penalty=leakage,
    )
    return {
        "tutoring_score": score,
        "tutor_quality": tutor_quality,
        "leakage_penalty": leakage,
    }
