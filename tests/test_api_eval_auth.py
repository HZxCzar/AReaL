"""Regression coverage for role credentials and local endpoint preflight."""

import sys

import httpx
import pytest

from examples.tutor import evaluate_teacher_api as entrypoint
from examples.tutor.scripts import evaluate_api_teacher as evaluator
from examples.tutor.scripts.eval_luna_full import check_local_endpoints


def test_inf_key_survives_temporary_config_environment(monkeypatch, tmp_path):
    """The real entrypoint must not replace selected role keys with EMPTY."""
    monkeypatch.setenv("INF_API_KEY", "test-role-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test-teacher-secret")
    checked = []

    async def fake_evaluate(args):
        config, students = evaluator.load_experiment_config(args.config, args.overrides)
        assert config.auxiliary_model.api_key == "test-role-secret"
        assert all(s["api_key"] == "test-role-secret" for s in students)
        checked.append(True)

    monkeypatch.setattr(evaluator, "main_async", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--provider",
            "openai",
            "--teacher-model",
            "test-teacher",
            "--teacher-base-url",
            "http://localhost:1/v1",
            "--student-base-url",
            "http://localhost:2/v1",
            "--aux-base-url",
            "http://localhost:3/v1",
            "--student-api-key-env",
            "INF_API_KEY",
            "--aux-api-key-env",
            "INF_API_KEY",
            "--config",
            "examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml",
            "--output-dir",
            str(tmp_path),
        ],
    )
    entrypoint.main()
    assert checked == [True]


@pytest.mark.parametrize("status", [200, 401])
def test_preflight_checks_local_roles_before_teacher(monkeypatch, status):
    """Both roles must return valid chat responses; auth rejection fails fast."""
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            status, json={"choices": [{"message": {"content": "Hi"}}]}
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    if status == 401:
        with pytest.raises(RuntimeError, match="student preflight failed"):
            check_local_endpoints("http://student/v1", "http://judge/v1", "test-key")
        assert len(requests) == 1
    else:
        check_local_endpoints("http://student/v1", "http://judge/v1", "test-key")
        assert len(requests) == 2
