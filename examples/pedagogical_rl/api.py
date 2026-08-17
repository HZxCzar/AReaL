from __future__ import annotations

import asyncio
from typing import Any, Protocol

from examples.pedagogical_rl.config import PedagogicalAPIModelConfig

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - lightweight CPU test environments
    AsyncOpenAI = None


class ChatCompletionClient(Protocol):
    chat: Any


class PedagogicalAPIClient:
    """Small async client supporting the native multi-choice student call."""

    def __init__(
        self,
        config: PedagogicalAPIModelConfig,
        *,
        client: ChatCompletionClient | None = None,
    ) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        if client is not None:
            self.client = client
        else:
            if AsyncOpenAI is None:
                raise RuntimeError("openai is required for PedagogicalRL API calls")
            if not config.base_url:
                raise ValueError("PedagogicalRL API base_url is required")
            if not config.api_key:
                raise ValueError("PedagogicalRL API api_key is required")
            self.client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        if n < 1:
            raise ValueError("n must be positive")
        # config.timeout is sized for one completion; the eight-choice student
        # call generates eight and otherwise times out on every problem.
        request_timeout = float(self.config.timeout) * max(1, int(n))
        async with self._semaphore:
            response = await self.client.chat.completions.create(
                timeout=request_timeout,
                model=self.config.model,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=self.config.seed,
                extra_headers=self.config.extra_headers or None,
                extra_body={
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                    "chat_template_kwargs": self.config.chat_template_kwargs,
                },
            )
        choices = sorted(response.choices, key=lambda choice: choice.index)
        if len(choices) != n:
            raise RuntimeError(
                f"API model {self.config.model!r} returned {len(choices)} choices, "
                f"expected {n}"
            )
        return [choice.message.content or "" for choice in choices]
