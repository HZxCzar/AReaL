"""Formal full-evaluation entrypoint tests; no live endpoints or paid calls."""

from types import SimpleNamespace

import httpx
import pytest

from examples.tutor.scripts.eval_api_full import evaluation_command, local_preflight


@pytest.mark.parametrize("provider", ["generic", "openai", "gemini"])
@pytest.mark.parametrize("proxy", [False, True])
def test_full_protocol_and_proxy_isolation(tmp_path, provider, proxy):
    """All providers use Luna semantics, with only local roles bypassing proxy."""
    args = SimpleNamespace(
        provider=provider,
        teacher_model="teacher",
        teacher_pricing=[1, 0.1, 1, 2],
        budget_usd=2,
        concurrency=16,
        env_file=tmp_path / "env",
        output_dir=tmp_path / "out",
        keep_env_proxy=proxy,
        resume=False,
        backfill=False,
        dry_run=False,
    )
    env = {
        "STUDENT_BASE_URL": "https://student.test/v1",
        "AUX_BASE_URL": "https://judge.test/v1",
        "HTTPS_PROXY": "http://localhost:7890",
        "ALL_PROXY": "socks5://localhost:7891",
    }
    cmd = evaluation_command(args, tmp_path, env)
    for flag, value in [
        ("--reasoning-effort", "medium"),
        ("--teacher-format", "thinking"),
        ("--teacher-sampling", "provider-default"),
        ("--teacher-output-limit", "provider-default"),
        ("--limit", "0"),
        ("--attempts", "1"),
        ("--teacher-presolve", "on"),
        ("--teacher-budget-usd", "2"),
    ]:
        assert cmd[cmd.index(flag) + 1] == value
    assert ("--keep-env-proxy" in cmd) == proxy
    assert ("HTTPS_PROXY" in env) == proxy
    assert "ALL_PROXY" not in env
    assert "student.test" in env["NO_PROXY"] and "judge.test" in env["NO_PROXY"]
    assert env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] == "0"
    assert "--resume" not in cmd
    args.output_dir.mkdir()
    (args.output_dir / "run_config.json").write_text("{}")
    if proxy:
        env["HTTPS_PROXY"] = "http://localhost:7890"
    assert "--resume" in evaluation_command(args, tmp_path, env)


@pytest.mark.parametrize("status", [200, 401])
def test_local_preflight_failure_precedes_paid_work(monkeypatch, status):
    """Validate role authentication without sending any teacher request."""
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            status, json={"choices": [{"message": {"content": "hi"}}]}
        )

    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
    )
    if status == 401:
        with pytest.raises(RuntimeError, match="no teacher calls"):
            local_preflight("http://student.test", "http://judge.test", "test-key")
        assert len(calls) == 1
    else:
        local_preflight("http://student.test", "http://judge.test", "test-key")
        assert len(calls) == 2
