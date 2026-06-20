from __future__ import annotations

import re
from typing import Any


def _words(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9']+", text.lower()) if len(tok) > 1}


def werewolf_tutor_quality(advice: str) -> float:
    lowered = advice.lower()
    words = _words(advice)
    public_terms = {
        "public",
        "discussion",
        "vote",
        "claim",
        "statement",
        "pattern",
        "alive",
        "night",
        "phase",
    }
    reasoning_terms = {"because", "if", "evidence", "consistent", "inconsistent", "risk", "uncertain"}
    social_terms = {
        "pressure",
        "defend",
        "accuse",
        "question",
        "credibility",
        "coalition",
        "suspicion",
        "agenda",
    }
    action_class_terms = {"vote", "question", "claim", "protect", "inspect", "poison", "kill", "speak"}

    public = min(1.0, len(words & public_terms) / 2.0)
    reasoning = min(1.0, sum(1 for term in reasoning_terms if term in lowered) / 2.0)
    social = min(1.0, len(words & social_terms) / 2.0)
    action_class = min(1.0, len(words & action_class_terms) / 2.0)
    concise = 1.0 if 25 <= len(advice.split()) <= 150 else 0.4
    return 0.30 * public + 0.25 * reasoning + 0.20 * social + 0.20 * action_class + 0.05 * concise


def werewolf_direct_action_penalty(advice: str) -> float:
    lowered = advice.lower()
    direct_patterns = [
        r"\b(vote|kill|poison|save|check|shoot)\s+player\s*\d+\b",
        r"\b(player\s*\d+)\s+(?:is|seems|must be)\s+(?:a\s+)?(?:werewolf|villager|witch|foreseer|hunter)\b",
        r"\b(?:the\s+)?(?:werewolf|wolf|villager|witch|foreseer|hunter)\s+is\s+player\s*\d+\b",
        r"\banswer\s*:\s*(vote|kill|poison|save|check|shoot)\b",
        r"<answer>.*?(vote|kill|poison|save|check|shoot).*?</answer>",
    ]
    return 1.0 if any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in direct_patterns) else 0.0


def werewolf_action_quality(
    *,
    parsed_action: str,
    step_reward: Any,
    role: str | None = None,
    phase: str | None = None,
) -> float:
    action = parsed_action.lower().strip()
    if not action:
        return -1.0
    if action == "skip":
        return -0.35 if phase in {"day", "hunter"} else -0.10
    if action.startswith("say "):
        words = len(action.split())
        return 0.35 if words >= 12 else 0.05

    if isinstance(step_reward, (list, tuple)):
        villager_reward = float(step_reward[0]) if len(step_reward) > 0 else 0.0
        werewolf_reward = float(step_reward[1]) if len(step_reward) > 1 else 0.0
        reward = werewolf_reward if role == "werewolf" else villager_reward
    else:
        reward = float(step_reward or 0.0)
    if reward > 0:
        return min(1.0, reward)
    if reward < 0:
        return max(-1.0, reward)
    if action.startswith(("vote ", "kill ", "poison ", "save ", "check ", "shoot ")):
        return 0.05
    return -0.25


def werewolf_tutoring_score_from_stats(
    *,
    stats: dict[str, Any],
    turns: int,
    invalid_actions: int,
    avg_action_quality: float,
    avg_tutor_quality: float,
    avg_leakage: float,
    avg_direct_action: float,
) -> float:
    vill_win = float(stats.get("vill_wins", 0))
    were_win = float(stats.get("were_wins", 0))
    correct = (
        float(stats.get("villager_correct_votes", 0))
        + float(stats.get("witch_correct_heals", 0))
        + float(stats.get("witch_correct_poisons", 0))
        + float(stats.get("hunter_correct_shots", 0))
        + 0.5 * float(stats.get("werewolf_correct_kills", 0))
    )
    wrong = float(stats.get("villager_wrong_votes", 0))
    invalid_rate = invalid_actions / max(1, turns)
    return (
        0.60 * vill_win
        - 0.60 * were_win
        + 0.12 * correct
        - 0.10 * wrong
        + 0.30 * avg_action_quality
        + 0.20 * avg_tutor_quality
        - 0.30 * invalid_rate
        - 0.45 * avg_leakage
        - 0.25 * avg_direct_action
    )


def werewolf_tutoring_score(env: Any, trace: list[dict[str, Any]] | None = None) -> float:
    stats = env.get_stats()
    trajectory = getattr(env, "trajectory", [])
    turns = len(trace or []) or len(trajectory) or 1
    invalid = sum(1 for item in trajectory if "invalid" in item.lower() or "skipping this turn" in item.lower())
    avg_action_quality = 0.0
    avg_tutor_quality = 0.0
    avg_leakage = 0.0
    avg_direct_action = 0.0
    if trace:
        avg_action_quality = sum(float(row.get("werewolf_action_quality", 0.0)) for row in trace) / len(trace)
        avg_tutor_quality = sum(float(row.get("teacher_quality", 0.0)) for row in trace) / len(trace)
        avg_leakage = sum(float(row.get("teacher_leakage", 0.0)) for row in trace) / len(trace)
        avg_direct_action = sum(float(row.get("teacher_direct_action", 0.0)) for row in trace) / len(trace)
    return werewolf_tutoring_score_from_stats(
        stats=stats,
        turns=turns,
        invalid_actions=invalid,
        avg_action_quality=avg_action_quality,
        avg_tutor_quality=avg_tutor_quality,
        avg_leakage=avg_leakage,
        avg_direct_action=avg_direct_action,
    )
