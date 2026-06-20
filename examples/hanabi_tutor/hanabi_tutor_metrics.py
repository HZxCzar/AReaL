from __future__ import annotations

import re
from typing import Any


def _words(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1}


def hanabi_tutor_quality(advice: str) -> float:
    lowered = advice.lower()
    words = _words(advice)
    public_state_terms = {
        "stack",
        "firework",
        "discard",
        "token",
        "fuse",
        "deck",
        "hint",
        "information",
        "known",
        "public",
    }
    action_terms = {"play", "discard", "hint", "save", "tempo"}
    risk_terms = {"risk", "safe", "unsafe", "misplay", "critical", "needed", "loss"}
    planning_terms = {"because", "if", "consider", "alternative", "priority", "therefore"}

    public_state = min(1.0, len(words & public_state_terms) / 2.0)
    action_focus = min(1.0, len(words & action_terms) / 2.0)
    risk = min(1.0, len(words & risk_terms) / 2.0)
    planning = min(1.0, sum(1 for term in planning_terms if term in lowered) / 2.0)
    concise = 1.0 if 25 <= len(advice.split()) <= 140 else 0.4
    return 0.30 * public_state + 0.25 * action_focus + 0.20 * risk + 0.20 * planning + 0.05 * concise


def hanabi_direct_action_penalty(advice: str) -> float:
    lowered = advice.lower()
    direct_patterns = [
        r"\b(play|discard)\s+[rygbw][1-5]\b",
        r"\bplay\s+(red|yellow|green|blue|white)\s*[1-5]\b",
        r"\bdiscard\s+(red|yellow|green|blue|white)\s*[1-5]\b",
        r"\banswer\s*:\s*(play|discard|hint)\b",
        r"<answer>.*?(play|discard|hint).*?</answer>",
    ]
    return 1.0 if any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in direct_patterns) else 0.0


def hanabi_action_quality(event: str, reward: float, parsed_action: str) -> float:
    lowered = event.lower()
    action = parsed_action.lower().strip()
    if not action or "invalid" in lowered or "unidentifiable" in lowered or "incomplete" in lowered:
        return -1.0
    if "successfully played" in lowered:
        return 1.0
    if "misplayed" in lowered or "fuse token lost" in lowered:
        return -1.0
    if action.startswith("hint ") and "hinted" in lowered:
        return 0.45
    if action.startswith("discard ") and "discarded" in lowered:
        return 0.20 + min(0.25, max(0.0, reward))
    return max(-0.5, min(0.5, reward))


def hanabi_tutoring_score_from_stats(
    *,
    score: float,
    target_score: float,
    fuse_tokens: int,
    max_fuse_tokens: int,
    invalid_actions: int,
    turns: int,
    avg_action_quality: float,
    avg_tutor_quality: float,
    avg_leakage: float,
    avg_direct_action: float,
) -> float:
    score_part = score / max(1.0, target_score)
    fuse_part = fuse_tokens / max(1, max_fuse_tokens)
    invalid_rate = invalid_actions / max(1, turns)
    return (
        score_part
        + 0.20 * fuse_part
        + 0.30 * avg_action_quality
        + 0.20 * avg_tutor_quality
        - 0.30 * invalid_rate
        - 0.45 * avg_leakage
        - 0.25 * avg_direct_action
    )


def hanabi_tutoring_score(env: Any, traces: list[dict[str, Any]] | None = None) -> float:
    stats = env.get_stats()
    trajectory = getattr(env, "trajectory", [])
    turns = int(stats.get("turns", len(trajectory) or 1))
    invalid = sum(
        1
        for item in trajectory
        if any(token in item.lower() for token in ("invalid", "unidentifiable", "incomplete"))
    )
    avg_action_quality = 0.0
    avg_tutor_quality = 0.0
    avg_leakage = 0.0
    avg_direct_action = 0.0
    if traces:
        avg_action_quality = sum(float(row.get("hanabi_action_quality", 0.0)) for row in traces) / len(traces)
        avg_tutor_quality = sum(float(row.get("teacher_quality", 0.0)) for row in traces) / len(traces)
        avg_leakage = sum(float(row.get("teacher_leakage", 0.0)) for row in traces) / len(traces)
        avg_direct_action = sum(float(row.get("teacher_direct_action", 0.0)) for row in traces) / len(traces)
    return hanabi_tutoring_score_from_stats(
        score=float(stats.get("score", 0.0)),
        target_score=float(getattr(env, "target_score", 25)),
        fuse_tokens=int(stats.get("fuse_tokens", 0)),
        max_fuse_tokens=int(getattr(env, "max_fuse_tokens", 3)),
        invalid_actions=invalid,
        turns=turns,
        avg_action_quality=avg_action_quality,
        avg_tutor_quality=avg_tutor_quality,
        avg_leakage=avg_leakage,
        avg_direct_action=avg_direct_action,
    )
