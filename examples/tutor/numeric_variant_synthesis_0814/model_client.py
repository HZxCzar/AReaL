"""Bounded OpenAI-compatible clients for the shared tutor deployments."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from examples.tutor.core.text import strip_reasoning_for_context


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump(exclude_none=True))
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _safe_error(error: BaseException) -> str:
    text = str(error)
    secret = os.getenv("INF_API_KEY", "")
    return text.replace(secret, "[REDACTED]") if secret else text


class SerialTutorClients:
    """A bounded request pool shared by teacher and student."""

    def __init__(
        self,
        call_log: Path,
        min_interval_seconds: float = 2.0,
        max_concurrent_calls: int = 1,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be nonnegative")
        if max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be positive")
        api_key = os.environ["INF_API_KEY"]
        self.call_log = call_log
        self.min_interval_seconds = min_interval_seconds
        self.max_concurrent_calls = max_concurrent_calls
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._start_lock = asyncio.Lock()
        self._last_started = 0.0
        timeout = httpx.Timeout(180.0)
        self._teacher_http = httpx.AsyncClient(trust_env=False, timeout=timeout)
        self._student_http = httpx.AsyncClient(trust_env=False, timeout=timeout)
        self.teacher = AsyncOpenAI(
            base_url=os.environ["TUTOR_QWEN3_8B_BASE_URL"],
            api_key=api_key,
            timeout=180,
            max_retries=0,
            http_client=self._teacher_http,
        )
        self.student = AsyncOpenAI(
            base_url=os.environ["TUTOR_QWEN3_1_7B_BASE_URL"],
            api_key=api_key,
            timeout=180,
            max_retries=0,
            http_client=self._student_http,
        )

    async def close(self) -> None:
        await self.teacher.close()
        await self.student.close()

    async def __aenter__(self) -> "SerialTutorClients":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _append_log(self, payload: dict[str, Any]) -> None:
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        with self.call_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    async def teacher_call(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> str:
        return await self._call(
            student=False,
            messages=messages,
            stage=stage,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    async def student_call(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        seed: int = 42,
    ) -> str:
        return await self._call(
            student=True,
            messages=messages,
            stage=stage,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    async def _call(
        self,
        *,
        student: bool,
        messages: list[dict[str, str]],
        stage: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> str:
        model = "qwen3-1.7b" if student else "qwen3-8b"
        header = (
            "tutor-train-qwen17b-student"
            if student
            else "tutor-train-qwen8b-auxiliary"
        )
        client = self.student if student else self.teacher
        retryable = {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "ReadTimeout",
        }

        async with self._semaphore:
            for attempt in range(1, 4):
                if self.min_interval_seconds > 0:
                    async with self._start_lock:
                        since_last = time.monotonic() - self._last_started
                        wait_for = self.min_interval_seconds - since_last
                        if wait_for > 0:
                            await asyncio.sleep(wait_for)
                        self._last_started = time.monotonic()
                started = time.time()
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        max_completion_tokens=max_tokens,
                        seed=seed,
                        extra_headers={"x-inspire-inference-key": header},
                        extra_body={
                            "top_k": 20,
                            "min_p": 0,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    output = (response.choices[0].message.content or "").strip()
                    visible = strip_reasoning_for_context(output).strip()
                    self._append_log(
                        {
                            "stage": stage,
                            "provider": "student" if student else "teacher",
                            "model": model,
                            "attempt": attempt,
                            "seed": seed,
                            "started_unix": started,
                            "elapsed_seconds": round(time.time() - started, 3),
                            "usage": _usage_dict(response.usage),
                            "ok": True,
                        }
                    )
                    return visible
                except Exception as error:
                    name = error.__class__.__name__
                    self._append_log(
                        {
                            "stage": stage,
                            "provider": "student" if student else "teacher",
                            "model": model,
                            "attempt": attempt,
                            "seed": seed,
                            "started_unix": started,
                            "elapsed_seconds": round(time.time() - started, 3),
                            "ok": False,
                            "error_type": name,
                            "error": _safe_error(error),
                        }
                    )
                    if name not in retryable or attempt == 3:
                        raise RuntimeError(_safe_error(error)) from error
                    await asyncio.sleep(2 ** (attempt - 1))
        raise RuntimeError("unreachable request state")
