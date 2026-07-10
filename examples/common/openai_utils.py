from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from examples.common.chat_budget import ChatContextBudget

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    AsyncOpenAI = None


@dataclass(slots=True)
class AuxModelConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: int = 120
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_concurrency: int = 8
    request_params: dict[str, Any] = field(default_factory=dict)
    tokenizer_path: str | None = None
    context_length: int | None = None
    context_window_margin: int = 256


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class LLMCallResult:
    text: str
    token_logprobs: tuple[TokenLogprob, ...] = ()


def resolve_request_config(config: AuxModelConfig) -> dict[str, Any]:
    resolved = dict(config.request_params)
    extra_body = resolved.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError(f"Expected extra_body to be a dict, got: {type(extra_body)!r}")
    if config.temperature is not None:
        resolved["temperature"] = config.temperature
    if config.top_p is not None:
        resolved["top_p"] = config.top_p
    if config.max_tokens is not None:
        resolved["max_tokens"] = config.max_tokens
    return resolved


class AsyncLLMCaller:
    def __init__(self, config: AuxModelConfig):
        self.config = config
        self.request_config = resolve_request_config(config)
        self.context_budget = ChatContextBudget(
            tokenizer_path=config.tokenizer_path,
            context_length=config.context_length,
            safety_margin=config.context_window_margin,
        )
        self._semaphore = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._client = None
        if AsyncOpenAI is not None:
            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "EMPTY",
                timeout=config.timeout,
                max_retries=0,
            )

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        request_overrides: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        if self._client is None:
            raise RuntimeError("openai package is required for auxiliary model calls")
        resolved_request_config = {
            **self.request_config,
            **(request_overrides or {}),
        }
        request_kwargs = {
            key: value
            for key, value in resolved_request_config.items()
            if key != "extra_body" and value is not None
        }
        if "max_tokens" in request_kwargs:
            request_kwargs["max_completion_tokens"] = int(
                request_kwargs.pop("max_tokens")
            )
        requested_max_completion_tokens = request_kwargs.get("max_completion_tokens")
        if requested_max_completion_tokens is not None:
            (
                safe_max_completion_tokens,
                prompt_tokens,
            ) = self.context_budget.clamp_max_completion_tokens(
                messages, int(requested_max_completion_tokens)
            )
            if safe_max_completion_tokens <= 0:
                raise RuntimeError(
                    "No completion budget remaining after accounting for prompt length: "
                    f"prompt_tokens={prompt_tokens}, "
                    f"context_length={self.context_budget.context_length}, "
                    f"safety_margin={self.context_budget.safety_margin}."
                )
            request_kwargs["max_completion_tokens"] = safe_max_completion_tokens
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                extra_body=resolved_request_config.get("extra_body") or None,
                **request_kwargs,
            )
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        choice_logprobs = getattr(choice, "logprobs", None)
        logprob_content = getattr(choice_logprobs, "content", None) or []
        token_logprobs = tuple(
            TokenLogprob(
                token=str(item.token),
                logprob=float(item.logprob),
                bytes=(
                    tuple(int(value) for value in item.bytes)
                    if getattr(item, "bytes", None) is not None
                    else None
                ),
            )
            for item in logprob_content
        )
        return LLMCallResult(text=content, token_logprobs=token_logprobs)

    async def call_text(self, messages: list[dict[str, str]]) -> str:
        return (await self.call(messages)).text


def make_teacher_client(extra_kwargs: dict[str, Any]) -> AsyncOpenAI:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required for teacher model calls")
    return AsyncOpenAI(
        base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
        api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
        http_client=extra_kwargs.get("http_client"),
        max_retries=0,
    )
