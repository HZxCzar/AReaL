from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from datasets import load_from_disk


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class TraceSink:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "trace.log"

    def append(self, role: str, content: str) -> None:
        body = (content or "").rstrip() or "(empty)"
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(f"[{role}]\n{body}\n\n")

    def append_messages(self, role: str, messages: list[dict[str, Any]]) -> None:
        rendered = []
        for message in messages:
            message_role = str(message.get("role", "unknown"))
            content = str(message.get("content", "")).rstrip() or "(empty)"
            rendered.append(f"[{message_role}]\n{content}")
        self.append(role, "\n\n".join(rendered))

    def dump_json(self, name: str, payload: Any) -> None:
        path = self.output_dir / name
        path.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_demo_rows(dataset_path: str, split: str, limit: int) -> list[dict[str, Any]]:
    dataset = load_from_disk(dataset_path)
    rows = dataset[split] if hasattr(dataset, "keys") else dataset
    return [dict(rows[i]) for i in range(min(limit, len(rows)))]


class LoggedTeacherClient:
    def __init__(self, client: Any, sink: TraceSink, model_override: str | None = None):
        self._client = client
        self._sink = sink
        self._model_override = model_override
        self.logged_usage: list[dict[str, Any]] = []
        self.chat = _LoggedTeacherChat(self)


class _LoggedTeacherChat:
    def __init__(self, parent: LoggedTeacherClient):
        self.completions = _LoggedTeacherCompletions(parent)


class _LoggedTeacherCompletions:
    def __init__(self, parent: LoggedTeacherClient):
        self._parent = parent

    async def create(self, **kwargs):
        if self._parent._model_override:
            kwargs["model"] = self._parent._model_override
        self._parent._sink.append_messages(
            "teacher_input", list(kwargs.get("messages") or [])
        )
        response = await self._parent._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        self._parent._sink.append("teacher", content)
        usage = getattr(response, "usage", None)
        usage_payload = None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                usage_payload = usage.model_dump()
            elif isinstance(usage, dict):
                usage_payload = usage
            else:
                usage_payload = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
            self._parent.logged_usage.append(usage_payload)
            self._parent._sink.append(
                "teacher_usage",
                json.dumps(usage_payload, ensure_ascii=False, indent=2),
            )
        return response


@contextmanager
def patch_teacher_factory(module: Any, factory):
    original = module.make_teacher_client
    module.make_teacher_client = factory
    try:
        yield
    finally:
        module.make_teacher_client = original
