from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    max_concurrency: int = 8
    api_params_config_path: str | None = None
    api_params_key: str | None = None


def load_api_params_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected api params config to be a JSON object, got: {type(data)!r}"
        )
    return data


def infer_api_params_key(base_url: str, model: str) -> str | None:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if host in {"127.0.0.1", "localhost"} and port is not None:
        return f"sglang:{port}"
    if "openrouter.ai" in host:
        return f"openrouter:{model}"
    if host in {"api.openai.com", "openai.com"}:
        return f"openai:{model}"
    if "api.together.xyz" in host:
        return f"together:{model}"
    if "api.sambanova.ai" in host:
        return f"sambanova:{model}"
    if "ark.cn-beijing.volces.com" in host:
        return f"doubao:{model}"
    return None


def split_endpoint_config(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(entry)
    extra_body = base.pop("extra_body", {})
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict):
        raise ValueError(f"Expected extra_body to be a dict, got: {type(extra_body)!r}")
    return base, dict(extra_body)


def resolve_endpoint_request_config(config: AuxModelConfig) -> dict[str, Any]:
    api_params_config = load_api_params_config(config.api_params_config_path)
    resolved: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    resolved_extra_body: dict[str, Any] = {}
    applied_keys: list[str] = []

    default_entry = api_params_config.get("default")
    if isinstance(default_entry, dict):
        default_base, default_extra = split_endpoint_config(default_entry)
        resolved.update(default_base)
        resolved_extra_body.update(default_extra)
        applied_keys.append("default")

    endpoint_key = config.api_params_key or infer_api_params_key(config.base_url, config.model)
    endpoint_entry = api_params_config.get(endpoint_key) if endpoint_key else None
    if isinstance(endpoint_entry, dict):
        endpoint_base, endpoint_extra = split_endpoint_config(endpoint_entry)
        resolved.update(endpoint_base)
        resolved_extra_body.update(endpoint_extra)
        applied_keys.append(endpoint_key)

    resolved["extra_body"] = resolved_extra_body
    resolved["resolved_api_params_key"] = endpoint_key
    resolved["applied_api_params_keys"] = applied_keys
    return resolved


class AsyncLLMCaller:
    def __init__(self, config: AuxModelConfig):
        self.config = config
        self.request_config = resolve_endpoint_request_config(config)
        self._semaphore = asyncio.Semaphore(max(1, int(config.max_concurrency)))
        self._client = None
        if AsyncOpenAI is not None:
            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "EMPTY",
                timeout=config.timeout,
                max_retries=0,
            )

    async def call_text(self, messages: list[dict[str, str]]) -> str:
        if self._client is None:
            raise RuntimeError("openai package is required for auxiliary model calls")
        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_completion_tokens=int(self.request_config["max_tokens"]),
                temperature=float(self.request_config["temperature"]),
                top_p=float(self.request_config["top_p"]),
                extra_body=self.request_config.get("extra_body") or None,
            )
        content = response.choices[0].message.content
        return (content or "").strip()


def make_teacher_client(extra_kwargs: dict[str, Any]) -> AsyncOpenAI:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required for teacher model calls")
    return AsyncOpenAI(
        base_url=extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL"),
        api_key=extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
        http_client=extra_kwargs.get("http_client"),
        max_retries=0,
    )
