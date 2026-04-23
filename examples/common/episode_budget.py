from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _estimate_text_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    return (
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


@dataclass(slots=True)
class EpisodeBudgetSnapshot:
    turn_prompt_tokens: int
    turn_completion_tokens: int
    turn_total_tokens: int
    stop_reason: str | None
    stop_feedback: str | None


class EpisodeTokenBudget:
    def __init__(
        self,
        *,
        max_episode_total_tokens: int | None = None,
    ):
        self.max_episode_total_tokens = max_episode_total_tokens

    def observe_turn(
        self,
        *,
        response: Any,
        prompt_text: str,
        completion_text: str,
    ) -> EpisodeBudgetSnapshot:
        prompt_tokens, completion_tokens = _extract_usage_tokens(response)
        if prompt_tokens is None:
            prompt_tokens = _estimate_text_tokens(prompt_text)
        if completion_tokens is None:
            completion_tokens = _estimate_text_tokens(completion_text)

        turn_total_tokens = prompt_tokens + completion_tokens

        stop_reason = None
        stop_feedback = None
        if (
            self.max_episode_total_tokens is not None
            and turn_total_tokens >= self.max_episode_total_tokens
        ):
            stop_reason = "episode_total_token_budget"
            stop_feedback = (
                "Episode terminated because serialized trajectory length exceeded budget: "
                f"{turn_total_tokens} >= {self.max_episode_total_tokens}."
            )

        return EpisodeBudgetSnapshot(
            turn_prompt_tokens=prompt_tokens,
            turn_completion_tokens=completion_tokens,
            turn_total_tokens=turn_total_tokens,
            stop_reason=stop_reason,
            stop_feedback=stop_feedback,
        )
