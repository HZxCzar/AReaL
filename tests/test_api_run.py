"""Paper experiment config, privacy, protocol parity and resume regression tests."""

import copy
import json
import os
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.api_run import runner


def test_public_entrypoint_has_no_proxy_activation(tmp_path):
    script = runner.PACKAGE / "run.sh"
    assert "clash" not in script.read_text().lower()
    result = subprocess.run(
        ["bash", str(script), "gemini-3.8-flash", "--dry-run"],
        cwd=tmp_path,
        env={**os.environ, "TUTOR_PYTHON": "/bin/echo"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == (
        "-m examples.tutor.scripts.api_run.runner gemini-3.8-flash --dry-run"
    )


def test_no_proxy_environment_needs_no_local_service(setup_run):
    config, args, env = setup_run
    for key in ("HTTPS_PROXY", "ALL_PROXY"):
        env.pop(key, None)
    runner.prepare(config, args, env)
    assert not any(env.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))


@pytest.fixture
def setup_run(tmp_path):
    config = runner.load_config(runner.PACKAGE / "configs/gemini-3.8-flash.yaml")
    args = SimpleNamespace(
        env_file=tmp_path / "private.env",
        output_dir=tmp_path / "run",
        budget_usd=2,
        concurrency=1,
        dry_run=False,
        backfill=False,
        limit=0,
    )
    env = {
        "GEMINI_BASE_URL": "https://teacher.test/v1",
        "GEMINI_API_KEY": "do-not-publish",
        "STUDENT_BASE_URL": "https://student.test/v1",
        "AUX_BASE_URL": "https://judge.test/v1",
        "INF_API_KEY": "private-local-key",
        "HTTPS_PROXY": "http://localhost:1234",
        "ALL_PROXY": "socks5://localhost:1235",
    }
    return config, args, env


@pytest.mark.parametrize("name", ["gemini-3.8-flash", "gpt-5.6-luna", "gpt-5.6-terra"])
def test_presets_inherit_one_protocol(name):
    """Adding a model config does not fork the paper protocol."""
    config = runner.load_config(runner.PACKAGE / "configs" / f"{name}.yaml")
    assert config["teacher"]["reasoning_effort"] == "medium"
    assert config["teacher"]["sampling"] == "provider-default"
    assert config["teacher"]["format"] == "thinking"
    assert config["teacher"]["output_limit"] == "provider-default"
    assert config["protocol"] == str(runner.PACKAGE / "protocol.yaml")
    assert config["execution"]["budget_usd"] == 2


def test_manifest_omits_secrets_and_endpoint_values(setup_run):
    """Public run metadata contains environment names/hashes, never credentials."""
    config, args, env = setup_run
    command, manifest = runner.prepare(config, args, env)
    serialized = json.dumps(manifest)
    for secret in (
        "do-not-publish",
        "private-local-key",
        "teacher.test",
        "student.test",
        "judge.test",
    ):
        assert secret not in serialized
    assert "do-not-publish" not in command
    assert "private-local-key" not in command
    assert "--keep-env-proxy" in command
    assert "student.test" in env["NO_PROXY"]
    assert "teacher.test" not in env["NO_PROXY"]
    assert "ALL_PROXY" not in env
    assert env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] == "0"


def test_resume_preserves_budget_and_rejects_changed_model(setup_run, tmp_path):
    """Operational overrides may change, but model/protocol changes cannot mix results."""
    config, args, env = setup_run
    _, first = runner.prepare(config, args, env)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(first))
    args.budget_usd = 100
    args.concurrency = 16
    args.output_dir.mkdir()
    (args.output_dir / "run_config.json").write_text("{}")
    command, second = runner.prepare(config, args, env)
    runner.check_manifest(path, second)
    assert "--resume" in command
    assert command[command.index("--teacher-budget-usd") + 1] == "100"
    config["teacher"]["model"] = "different-model"
    _, changed = runner.prepare(config, args, env)
    with pytest.raises(ValueError, match="refuse to mix"):
        runner.check_manifest(path, changed)


def test_explicit_unknown_reserve_preserves_experiment_identity(setup_run):
    """An audited missing-usage reservation does not reset the experiment ledger."""
    config, args, env = setup_run
    _, before = runner.prepare(config, args, env)
    args.unknown_request_reserve_usd = 0.1
    _, after = runner.prepare(config, args, env)
    assert before == after
    assert env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] == "0.1"
    for invalid in [-1, float("nan"), float("inf")]:
        args.unknown_request_reserve_usd = invalid
        with pytest.raises(ValueError, match="reserve"):
            runner.prepare(config, args, env)


def test_invalid_inheritance_and_unknown_options_fail(tmp_path, setup_run):
    """Typos and cyclic inheritance must not silently run another experiment."""
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        runner.load_config(path)
    config = copy.deepcopy(setup_run[0])
    config["teacher"]["temprature"] = 1
    with pytest.raises(ValueError, match="unknown"):
        runner.validate_config(config)


def test_roles_cannot_diverge_from_actual_protocol(setup_run):
    """A changed preflight model is not enough to change the fixed student/judge."""
    config, args, env = setup_run
    config["roles"]["student"]["model"] = "wrong-model"
    with pytest.raises(ValueError, match="actual protocol"):
        runner.prepare(config, args, env)


def test_public_protocol_has_no_private_deployment_literals():
    """Published protocol is standalone and requires no dated config files."""
    text = (runner.PACKAGE / "protocol.yaml").read_text()
    assert "defaults:" not in text
    for private in (
        "/inspire/",
        "sii.edu.cn",
        "saltlab.stanford.edu",
        "x-inspire-inference-key",
    ):
        assert private not in text
    protocol = yaml.safe_load(text)
    assert "TUTOR_DATASET" in protocol["valid_dataset"]["path"]


def test_standalone_protocol_preserves_historical_evaluation(monkeypatch):
    """All non-deployment config fields and expanded students match the paper baseline."""
    from examples.tutor.scripts import evaluate_api_teacher as evaluator

    for key in ("TUTOR_QWEN3_1_7B_BASE_URL", "TUTOR_QWEN3_8B_BASE_URL"):
        monkeypatch.setenv(key, "http://localhost:1/v1")
    monkeypatch.setenv("INF_API_KEY", "EMPTY")
    monkeypatch.delenv("TUTOR_TOKENIZER", raising=False)
    monkeypatch.delenv("TUTOR_DATASET", raising=False)
    old, old_students = evaluator.load_experiment_config(
        str(
            runner.REPO
            / "examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml"
        ),
        ["trial_name=protocol-parity"],
    )
    new, new_students = evaluator.load_experiment_config(
        str(runner.PACKAGE / "protocol.yaml"), ["trial_name=protocol-parity"]
    )
    deployment = {
        "path",
        "tokenizer_path",
        "model_path",
        "fileroot",
        "nfs_record_root",
        "debug_trace_dir",
        "trial_name",
        "extra_headers",
    }

    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items() if k not in deployment}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        return value

    assert normalized(asdict(new)) == normalized(asdict(old))
    assert normalized(new_students) == normalized(old_students)
