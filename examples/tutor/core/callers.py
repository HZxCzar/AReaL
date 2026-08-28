from __future__ import annotations

import asyncio
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from examples.common.chat_budget import ChatContextBudget
from examples.common.openai_utils import AsyncLLMCaller, TokenLogprob
from examples.tutor.core.generation_budget import (
    ContextBudgetLimitExceeded,
    ensure_response_within_train_sample_budget,
    prepare_train_sample_generation_config,
    raise_if_over_budget,
    with_max_new_tokens,
)
from examples.tutor.core.text import strip_reasoning_for_context

from areal.api import ModelRequest, ModelResponse
from areal.api.cli_args import GenerationHyperparameters


@dataclass(slots=True)
class TextCallResult:
    text: str
    raw_text: str = ""
    error: str | None = None
    token_logprobs: tuple[TokenLogprob, ...] = ()


@dataclass(slots=True)
class CandidateLogprobResult:
    raw_text: str = ""
    candidate_logprobs: dict[int, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class ActorCallResult:
    response: ModelResponse
    raw_text: str
    visible_text: str


@dataclass(slots=True)
class EngineChatResult:
    response: ModelResponse
    raw_text: str
    visible_text: str


def apply_chat_template(
    tokenizer: Any | None,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool = True,
) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking,
                )
            )
        except TypeError:
            return list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            )
    text = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages
    )
    return encode_text(tokenizer, text)


def decode_output(
    tokenizer: Any | None,
    response: ModelResponse,
) -> str:
    tokenizer = response.tokenizer or tokenizer
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(
                response.output_tokens, skip_special_tokens=False
            ).replace("<|im_end|>", "")
        except TypeError:
            return tokenizer.decode(response.output_tokens).replace("<|im_end|>", "")
    return "".join(chr(max(0, int(token))) for token in response.output_tokens)


def encode_text(tokenizer: Any | None, text: str) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    return [ord(ch) for ch in text]


class ApiAuxiliaryCaller:
    def __init__(
        self,
        caller: AsyncLLMCaller,
        *,
        request_overrides: dict[str, Any] | None = None,
    ):
        self.caller = caller
        self.request_overrides = dict(request_overrides or {})

    @property
    def request_config(self) -> dict[str, Any]:
        return {**self.caller.request_config, **self.request_overrides}

    async def call_text(
        self,
        messages: list[dict[str, str]],
        *,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        del rid_prefix
        try:
            result = await self.caller.call(
                messages,
                request_overrides=self.request_overrides,
            )
        except Exception as exc:
            return TextCallResult(text="", raw_text="", error=str(exc))
        return TextCallResult(
            text=strip_reasoning_for_context(result.text),
            raw_text=result.text,
            error=None,
            token_logprobs=result.token_logprobs,
        )

    async def call_candidate_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        candidates: dict[int, str],
        rid_prefix: str = "auxiliary-candidates",
    ) -> CandidateLogprobResult:
        """Read one constrained next-token distribution from an SGLang API."""
        del rid_prefix
        try:
            if not candidates:
                raise ValueError("candidate logprob request needs at least one token.")
            if len(candidates) > 20:
                raise ValueError(
                    "OpenAI-compatible top_logprobs supports at most 20 candidates, "
                    f"got {len(candidates)}."
                )
            spellings = list(candidates.values())
            if len(set(spellings)) != len(spellings):
                raise ValueError(
                    f"candidate token spellings must be unique, got {spellings}."
                )

            # SGLang applies this grammar before computing top_logprobs, so the
            # response contains the complete candidate set rather than an
            # arbitrary top-k slice of the full vocabulary.
            regex = "(?:" + "|".join(re.escape(token) for token in spellings) + ")"
            extra_body = dict(self.request_config.get("extra_body") or {})
            extra_body.pop("ebnf", None)
            extra_body.pop("json_schema", None)
            extra_body.update({"regex": regex, "top_k": -1, "min_p": 0.0})
            result = await self.caller.call(
                messages,
                request_overrides={
                    **self.request_overrides,
                    "max_tokens": None,
                    "max_completion_tokens": 1,
                    "n": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "response_format": None,
                    "logprobs": True,
                    "top_logprobs": len(candidates),
                    "extra_body": extra_body,
                },
            )
            if not result.token_logprobs:
                raise RuntimeError(
                    "candidate logprob response contained no generated-token logprobs."
                )
            first = result.token_logprobs[0]
            returned: dict[str, float] = {first.token: float(first.logprob)}
            for item in first.top_logprobs:
                returned[item.token] = max(
                    returned.get(item.token, float("-inf")), float(item.logprob)
                )
            missing = [token for token in spellings if token not in returned]
            if missing:
                raise RuntimeError(
                    f"candidate logprob response omitted constrained tokens: {missing}."
                )
            logprobs = {
                int(token_id): float(returned[token])
                for token_id, token in candidates.items()
            }
            if not all(math.isfinite(value) for value in logprobs.values()):
                raise RuntimeError(
                    f"candidate logprob response contained non-finite values: {logprobs}."
                )
            return CandidateLogprobResult(
                raw_text=result.text,
                candidate_logprobs=logprobs,
            )
        except Exception as exc:
            return CandidateLogprobResult(error=str(exc))

    async def call_text_many(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        rid_prefix: str = "auxiliary",
        timeout: float | None = None,
    ) -> list[TextCallResult]:
        """Draw ``n`` samples from a single request, as PedagogicalRL does."""

        del rid_prefix
        try:
            results = await self.caller.call_many(
                messages,
                n=n,
                request_overrides=self.request_overrides,
                timeout=timeout,
            )
        except Exception as exc:
            return [TextCallResult(text="", raw_text="", error=str(exc))] * n
        return [
            TextCallResult(
                text=strip_reasoning_for_context(result.text),
                raw_text=result.text,
                error=None,
                token_logprobs=result.token_logprobs,
            )
            for result in results
        ]


class AReaLEngineChatCaller:
    def __init__(
        self,
        *,
        engine: Any,
        tokenizer: Any | None,
        enable_thinking: bool,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        gconfig: Any | None,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
        metadata: dict[str, Any],
        rid_prefix: str,
    ) -> EngineChatResult:
        input_ids = apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        budget = prepare_train_sample_generation_config(
            input_ids=input_ids,
            gconfig=gconfig,
            max_completion_tokens=max_completion_tokens,
            max_train_sample_tokens=max_train_sample_tokens,
        )
        raise_if_over_budget(budget)
        req = ModelRequest(
            rid=f"{rid_prefix}-{uuid.uuid4().hex}",
            input_ids=input_ids,
            gconfig=budget.gconfig,
            metadata=dict(metadata),
            tokenizer=self.tokenizer,
        )
        response = await self.engine.agenerate(req)
        ensure_response_within_train_sample_budget(
            input_len=response.input_len,
            output_len=response.output_len,
            max_train_sample_tokens=max_train_sample_tokens,
        )
        raw_text = decode_output(self.tokenizer, response)
        return EngineChatResult(
            response=response,
            raw_text=raw_text,
            visible_text=strip_reasoning_for_context(raw_text),
        )


class AReaLEngineAuxiliaryCaller:
    def __init__(
        self,
        *,
        chat_caller: AReaLEngineChatCaller,
        base_gconfig: Any | None,
        max_completion_tokens: int,
        temperature: float,
        top_p: float | None,
        max_concurrency: int,
        context_length: int | None,
        context_window_margin: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.chat_caller = chat_caller
        self.base_gconfig = base_gconfig
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.temperature = float(temperature)
        self.top_p = 1.0 if top_p is None else float(top_p)
        self.context_length = context_length
        self.context_window_margin = max(0, int(context_window_margin))
        self._semaphore = semaphore or asyncio.Semaphore(max(1, int(max_concurrency)))

    async def call_text(
        self,
        messages: list[dict[str, str]],
        *,
        rid_prefix: str = "auxiliary",
    ) -> TextCallResult:
        try:
            async with self._semaphore:
                result = await self.chat_caller.generate(
                    messages,
                    gconfig=self._generation_config(),
                    metadata={"disable_lora": True},
                    max_completion_tokens=self.max_completion_tokens,
                    max_train_sample_tokens=None,
                    rid_prefix=rid_prefix,
                )
        except Exception as exc:
            return TextCallResult(text="", raw_text="", error=str(exc))
        return TextCallResult(
            text=result.visible_text,
            raw_text=result.raw_text,
            error=None,
        )

    async def call_candidate_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        candidates: dict[int, str],
        rid_prefix: str = "auxiliary-candidates",
    ) -> CandidateLogprobResult:
        """Read exact requested token logprobs from the SGLang engine path."""
        try:
            if not candidates:
                raise ValueError("candidate logprob request needs at least one token.")
            if len(set(candidates)) != len(candidates):
                raise ValueError("candidate token ids must be unique.")
            async with self._semaphore:
                result = await self.chat_caller.generate(
                    messages,
                    gconfig=self._candidate_logprob_generation_config(),
                    metadata={
                        "disable_lora": True,
                        "token_ids_logprob": list(candidates),
                    },
                    max_completion_tokens=1,
                    max_train_sample_tokens=None,
                    rid_prefix=rid_prefix,
                )
            positions = getattr(result.response, "output_token_ids_logprobs", None)
            if not positions or not positions[0]:
                raise RuntimeError(
                    "candidate logprob response contained no token_ids_logprob data."
                )
            returned = {
                int(token_id): float(logprob) for logprob, token_id in positions[0]
            }
            missing = [token_id for token_id in candidates if token_id not in returned]
            if missing:
                raise RuntimeError(
                    f"candidate logprob response omitted token ids: {missing}."
                )
            logprobs = {
                int(token_id): float(returned[token_id]) for token_id in candidates
            }
            if not all(math.isfinite(value) for value in logprobs.values()):
                raise RuntimeError(
                    f"candidate logprob response contained non-finite values: {logprobs}."
                )
            return CandidateLogprobResult(
                raw_text=result.raw_text,
                candidate_logprobs=logprobs,
            )
        except Exception as exc:
            return CandidateLogprobResult(error=str(exc))

    async def call_text_many(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        rid_prefix: str = "auxiliary",
        timeout: float | None = None,
    ) -> list[TextCallResult]:
        """Sample ``n`` times. The engine backend seeds per request, so unlike
        the API caller these are separate calls rather than one ``n``-choice
        request."""

        del timeout
        return list(
            await asyncio.gather(
                *(
                    self.call_text(messages, rid_prefix=f"{rid_prefix}-{index}")
                    for index in range(n)
                )
            )
        )

    def _generation_config(self) -> Any:
        base_gconfig = self.base_gconfig or GenerationHyperparameters()
        max_tokens = getattr(base_gconfig, "max_tokens", None)
        if self.context_length is not None:
            max_tokens = max(1, int(self.context_length) - self.context_window_margin)
        kwargs = {
            "n_samples": 1,
            "max_new_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if hasattr(base_gconfig, "new"):
            return base_gconfig.new(**kwargs)
        values = dict(getattr(base_gconfig, "__dict__", {}))
        values.update(kwargs)
        return GenerationHyperparameters(**values)

    def _candidate_logprob_generation_config(self) -> Any:
        base_gconfig = self._generation_config()
        kwargs = {
            "n_samples": 1,
            "max_new_tokens": 1,
            "greedy": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
        }
        if hasattr(base_gconfig, "new"):
            return base_gconfig.new(**kwargs)
        values = dict(getattr(base_gconfig, "__dict__", {}))
        values.update(kwargs)
        return GenerationHyperparameters(**values)


class AReaLEngineActorCaller:
    def __init__(
        self,
        *,
        chat_caller: AReaLEngineChatCaller,
        gconfig: Any | None,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
    ) -> None:
        self.chat_caller = chat_caller
        self.gconfig = gconfig
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.max_train_sample_tokens = max_train_sample_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        lora_version: int | None,
        rid_prefix: str,
        max_completion_tokens: int | None = None,
        input_token_reserve: int = 0,
    ) -> ActorCallResult:
        metadata: dict[str, Any] = {}
        if lora_version is not None:
            metadata["lora_version"] = int(lora_version)
        completion_budget = (
            self.max_completion_tokens
            if max_completion_tokens is None
            else max(1, int(max_completion_tokens))
        )
        train_sample_budget = self.max_train_sample_tokens
        if train_sample_budget is not None:
            train_sample_budget -= max(0, int(input_token_reserve))
        # The budget resolver prefers gconfig.max_new_tokens whenever a gconfig is
        # given, so a per-call budget larger than the shared one would be dropped
        # silently. Widen the gconfig for this call when that happens. Only when it
        # is LARGER: a caller asking for fewer tokens keeps the shared cap, which is
        # what every caller but the pre-solve does today.
        gconfig = self.gconfig
        if gconfig is not None:
            shared = getattr(gconfig, "max_new_tokens", None)
            if shared is not None and completion_budget > int(shared):
                gconfig = with_max_new_tokens(gconfig, completion_budget)
        result = await self.chat_caller.generate(
            messages,
            gconfig=gconfig,
            max_completion_tokens=completion_budget,
            max_train_sample_tokens=train_sample_budget,
            metadata=metadata,
            rid_prefix=rid_prefix,
        )
        return ActorCallResult(
            response=result.response,
            raw_text=result.raw_text,
            visible_text=result.visible_text,
        )


class ExternalActorCaller:
    def __init__(
        self,
        *,
        client: Any,
        tokenizer: Any | None,
        context_budget: ChatContextBudget,
        temperature: float,
        top_p: float,
        enable_thinking: bool,
        max_completion_tokens: int,
        max_train_sample_tokens: int | None,
    ) -> None:
        self.client = client
        self.tokenizer = tokenizer
        self.context_budget = context_budget
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.enable_thinking = enable_thinking
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.max_train_sample_tokens = max_train_sample_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        lora_version: int | None,
        rid_prefix: str,
        max_completion_tokens: int | None = None,
        input_token_reserve: int = 0,
    ) -> ActorCallResult:
        del lora_version, rid_prefix
        completion_budget = (
            self.max_completion_tokens
            if max_completion_tokens is None
            else max(1, int(max_completion_tokens))
        )
        train_sample_budget = self.max_train_sample_tokens
        if train_sample_budget is not None:
            train_sample_budget -= max(0, int(input_token_reserve))
        input_ids = apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        budget = prepare_train_sample_generation_config(
            input_ids=input_ids,
            gconfig=None,
            max_completion_tokens=completion_budget,
            max_train_sample_tokens=train_sample_budget,
        )
        raise_if_over_budget(budget)
        safe_max_completion_tokens, _ = self.context_budget.clamp_max_completion_tokens(
            messages,
            int(budget.max_new_tokens),
        )
        if safe_max_completion_tokens <= 0:
            raise ContextBudgetLimitExceeded(
                "tutor prompt exceeded external client context budget"
            )
        response_obj = await self.client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=max(1, safe_max_completion_tokens),
        )
        raw_text = response_obj.choices[0].message.content or ""
        output_tokens = encode_text(self.tokenizer, raw_text)
        ensure_response_within_train_sample_budget(
            input_len=len(input_ids),
            output_len=len(output_tokens),
            max_train_sample_tokens=train_sample_budget,
        )
        response = ModelResponse(
            input_tokens=list(input_ids),
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            tokenizer=self.tokenizer,
        )
        return ActorCallResult(
            response=response,
            raw_text=raw_text,
            visible_text=strip_reasoning_for_context(raw_text),
        )
