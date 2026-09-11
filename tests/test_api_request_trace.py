"""Verify real HTTP-body capture using an offline transport, never paid calls."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from examples.tutor.core.api_request_trace import ACTIVE_TRACE, attach


@pytest.mark.asyncio
async def test_serialized_messages_preserved_without_auth_headers(tmp_path):
    """Capture the same JSON the transport sees, including unmodified message text."""
    received = []

    def respond(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "<end>",
                            "reasoning_content": "Separate reasoning",
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
                "provider_extra": {"thought_signature": "opaque-signature"},
            },
        )

    path = tmp_path / "requests.jsonl"
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        attach(SimpleNamespace(_client=client), "teacher")
        attach(SimpleNamespace(_client=client), "teacher")
        token = ACTIVE_TRACE.set(
            {"path": str(path), "episode_key": "episode:1", "execution_try": 1}
        )
        try:
            payload = {
                "model": "test",
                "reasoning_effort": "medium",
                "messages": [
                    {"role": "system", "content": "Full system prompt\nDo not modify."},
                    {"role": "user", "content": "原始题目\n<end>"},
                ],
            }
            response = await client.post(
                "https://private.test/v1/chat/completions",
                headers={"Authorization": "Bearer private-key"},
                json=payload,
            )
            assert response.json()["choices"][0]["message"]["content"] == "<end>"
        finally:
            ACTIVE_TRACE.reset(token)
    text = path.read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert len(records) == 2
    assert records[0]["payload"] == received[0] == payload
    assert records[0]["request_id"] == records[1]["request_id"]
    assert records[1]["payload"] == response.json()
    assert (
        records[1]["choices"][0]["message"]["reasoning_content"] == "Separate reasoning"
    )
    assert (
        records[1]["payload"]["provider_extra"]["thought_signature"]
        == "opaque-signature"
    )
    assert "private-key" not in text and "private.test" not in text


@pytest.mark.asyncio
async def test_concurrent_episodes_keep_their_own_ids(tmp_path):
    """Shared teacher clients must never mix parallel episodes' context labels."""
    path = tmp_path / "requests.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": []})
        )
    ) as client:
        attach(SimpleNamespace(_client=client), "teacher")

        async def run(i):
            token = ACTIVE_TRACE.set(
                {"path": str(path), "episode_key": str(i), "execution_try": 1}
            )
            try:
                await client.post(
                    "https://example.test",
                    json={"messages": [{"role": "user", "content": str(i)}]},
                )
            finally:
                ACTIVE_TRACE.reset(token)

        await asyncio.gather(*(run(i) for i in range(4)))
        before = path.read_text()
        await client.post("https://example.test", json={"messages": []})
        assert path.read_text() == before
    records = [json.loads(line) for line in before.splitlines()]
    assert len(records) == 8
    for row in records:
        if row["event"] == "request":
            assert row["payload"]["messages"][0]["content"] == row["episode_key"]
