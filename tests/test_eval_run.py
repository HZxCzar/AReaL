"""Portable checkpoint evaluation: protocol, privacy, sharding and resume tests."""

import json
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.eval_run import runner
from examples.tutor.scripts.eval_run.summarize import summarize


@pytest.fixture
def setup_eval(tmp_path):
    config = runner.load_config(runner.PACKAGE / "configs/untrained.yaml")
    args = SimpleNamespace(
        output_dir=tmp_path / "run", limit=0, shard_count=4, shard_index=0
    )
    env = {
        "TEACHER_BASE_URL": "http://teacher.test/v1",
        "TEACHER_API_KEY": "teacher-secret",
        "STUDENT_BASE_URL": "http://student.test/v1",
        "STUDENT_API_KEY": "student-secret",
        "AUX_BASE_URL": "http://aux.test/v1",
        "AUX_API_KEY": "aux-secret",
    }
    return config, args, env


@pytest.mark.parametrize("preset", ["untrained", "subgoal-1500", "all-legacy-1500"])
def test_presets_share_one_fixed_protocol(preset):
    """Teacher presets never inherit dated training defaults."""
    config = runner.load_config(runner.PACKAGE / "configs" / f"{preset}.yaml")
    assert config["protocol"] == str(runner.PACKAGE / "protocol.yaml")
    assert config["evaluation"]["attempts"] == 1
    assert config["evaluation"]["id_preferences"] == [
        "none",
        "attempt-diagnosis",
        "subgoal-decomposition",
        "contrastive-comparison",
    ]
    assert config["teacher"]["enable_thinking"] is False


def test_untrained_format_does_not_enable_thinking(setup_eval):
    config, args, env = setup_eval
    command, _, protocol = runner.prepare(config, args, env)
    params = json.loads(command[command.index("--teacher-request-params") + 1])
    assert protocol["teacher_response_format"] == "thinking"
    assert protocol["enable_thinking"] is False
    assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "lora_path" not in params["extra_body"]
    assert protocol["length_retry"] == {"enabled": True, "attempts": 3}


def test_lora_is_teacher_only(setup_eval):
    _, args, env = setup_eval
    config = runner.load_config(runner.PACKAGE / "configs/subgoal-1500.yaml")
    env["TEACHER_ADAPTER"] = "registered-adapter"
    command, manifest, protocol = runner.prepare(config, args, env)
    params = json.loads(command[command.index("--teacher-request-params") + 1])
    assert params["extra_body"]["lora_path"] == "registered-adapter"
    assert (
        "lora_path" not in protocol["auxiliary_model"]["request_params"]["extra_body"]
    )
    assert "registered-adapter" not in json.dumps(manifest)


def test_prepare_has_no_writes_or_secret_arguments(setup_eval):
    config, args, env = setup_eval
    command, manifest, protocol = runner.prepare(config, args, env)
    assert not args.output_dir.exists()
    public = json.dumps(manifest) + yaml.safe_dump(protocol)
    for secret in (
        "teacher-secret",
        "student-secret",
        "aux-secret",
        "teacher.test",
        "student.test",
        "aux.test",
    ):
        assert secret not in public
    for secret in ("teacher-secret", "student-secret", "aux-secret"):
        assert secret not in " ".join(command)
    assert env["DEEPSEEK_API_KEY"] == "teacher-secret"
    assert env["EVAL_RUN_STUDENT_KEY"] == "student-secret"
    assert env["EVAL_RUN_AUX_KEY"] == "aux-secret"


def test_protocol_snapshot_parity_with_existing_paper_protocol():
    """Keep the previously checked standalone paper protocol, except deployment/log labels."""
    before = yaml.safe_load(
        (runner.PACKAGE.parent / "api_run/protocol.yaml").read_text()
    )
    after = yaml.safe_load((runner.PACKAGE / "protocol.yaml").read_text())
    for value in (before, after):
        value.pop("trial_name")
        value.pop("stats_logger")
        for key in ("model", "base_url", "api_key"):
            value["auxiliary_model"].pop(key)
        for key in ("base_url", "api_key"):
            value["student_axes"][0]["template"].pop(key)
    assert after == before


@pytest.mark.parametrize("field,value", [("version", 99), ("unknown", True)])
def test_unknown_schema_is_rejected(setup_eval, field, value):
    config, _, _ = setup_eval
    config[field] = value
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_roles_are_applied_to_actual_protocol(setup_eval):
    config, args, env = setup_eval
    config["roles"]["auxiliary"]["model"] = "another-judge"
    _, _, protocol = runner.prepare(config, args, env)
    assert protocol["auxiliary_model"]["model"] == "another-judge"


def test_resume_allows_only_episode_concurrency_change(setup_eval, tmp_path):
    """Increasing scheduling parallelism preserves protocol and completed work."""
    config, args, env = setup_eval
    config["execution"]["concurrency"] = 2
    old_command, old, old_protocol = runner.prepare(config, args, env)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(old))
    config["execution"]["concurrency"] = 8
    command, new, protocol = runner.prepare(config, args, env)
    runner.check_manifest(path, new)
    assert old_protocol == protocol
    assert old["protocol_sha256"] == new["protocol_sha256"]
    assert old_command[old_command.index("--concurrency") + 1] == "2"
    assert command[command.index("--concurrency") + 1] == "8"
    assert new["experiment"]["execution"]["concurrency"] == 8
    assert old["experiment"]["execution"]["concurrency"] == 2
    for field in (
        "episode_timeout_seconds",
        "episode_error_retries",
        "student_concurrency",
        "auxiliary_concurrency",
    ):
        config["execution"][field] += 1
        _, changed, _ = runner.prepare(config, args, env)
        with pytest.raises(ValueError, match="changed"):
            runner.check_manifest(path, changed)
        config["execution"][field] -= 1


def test_changed_settings_cannot_resume(setup_eval, tmp_path):
    config, args, env = setup_eval
    _, manifest, _ = runner.prepare(config, args, env)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    runner.check_manifest(path, manifest)
    config["teacher"]["enable_thinking"] = True
    _, changed, _ = runner.prepare(config, args, env)
    with pytest.raises(ValueError, match="changed"):
        runner.check_manifest(path, changed)


def test_manifest_ignores_unrelated_tools_but_checks_evaluator(setup_eval, tmp_path):
    """Legacy broad hashes must not block unrelated human-study additions."""
    config, args, env = setup_eval
    _, manifest, _ = runner.prepare(config, args, env)
    legacy = json.loads(json.dumps(manifest))
    legacy["source_sha256"]["examples/tutor/scripts/eval_run/runner.py"] = "old-wrapper"
    legacy["source_sha256"]["examples/tutor/human_study/eval.py"] = "unrelated"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(legacy))
    runner.check_manifest(path, manifest)
    legacy["source_sha256"]["examples/tutor/scripts/evaluate_api_teacher.py"] = (
        "changed"
    )
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="changed"):
        runner.check_manifest(path, manifest)


def test_shards_are_disjoint_and_cover_full_matrix(setup_eval):
    config, args, env = setup_eval
    sets = []
    for index in range(4):
        args.shard_index = index
        _, manifest, _ = runner.prepare(config, args, env)
        assert manifest["shard_index"] == index
        assert env["TUTOR_EVAL_SHARD_INDEX"] == str(index)
        sets.append(set(range(index, 528 * 7, 4)))
    assert len(set.union(*sets)) == 3696
    assert sum(map(len, sets)) == 3696
    assert all(len(values) == 924 for values in sets)


def test_proxy_policy_is_explicit_and_external(setup_eval):
    config, args, env = setup_eval
    env["HTTPS_PROXY"] = "http://proxy.test:8080"
    command, _, _ = runner.prepare(config, args, env)
    assert env["HTTPS_PROXY"] == "http://proxy.test:8080"
    assert "--keep-env-proxy" in command
    config["execution"]["proxy"] = "direct"
    command, _, _ = runner.prepare(config, args, env)
    assert "HTTPS_PROXY" not in env
    assert "--keep-env-proxy" not in command


def test_incomplete_summary_is_not_success(setup_eval):
    config, args, _ = setup_eval
    (args.output_dir / "evaluation").mkdir(parents=True)
    path = args.output_dir / "evaluation/summary.json"
    path.write_text(
        json.dumps(
            {
                "modes": {"presolve_on": {"completed_attempts": 924}},
                "pending_backfill": {"count": 1},
            }
        )
    )
    assert runner.completion_status(args.output_dir, config, args)["complete"] is False


def test_public_files_contain_no_cluster_literals():
    for path in runner.PACKAGE.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
            text = path.read_text()
            assert "/inspire/" not in text
            assert "sii.edu.cn" not in text
    assert "defaults:" not in (runner.PACKAGE / "protocol.yaml").read_text()


def test_merge_uses_latest_records_and_rejects_overlapping_shards(tmp_path):
    config = runner.load_config(runner.PACKAGE / "configs/untrained.yaml")
    config["evaluation"].update(preferences=["none"], expected_questions=2)
    dirs = [tmp_path / str(i) for i in range(2)]
    for i, directory in enumerate(dirs):
        directory.mkdir()
        (directory / "evaluation").mkdir()
        manifest = {
            "experiment": config,
            "shard_count": 2,
            "shard_index": i,
            "limit": 0,
        }
        (directory / "experiment.json").write_text(json.dumps(manifest))
        (directory / "completion.json").write_text('{"complete":true}')
        row = {
            "key": str(i),
            "personality_gate": {"personality": "none"},
            "generalization": {"original": {"replay_count": 8, "score": 1}},
            "no_teaching_baseline": 0.5,
        }
        (directory / "evaluation/results.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(row) + "\n"
        )
    report = summarize(dirs)
    assert report["cells"]["none"]["episodes"] == 2
    assert report["cells"]["none"]["improvement_pp"] == 50
    (dirs[1] / "evaluation/results.jsonl").write_text(
        (dirs[0] / "evaluation/results.jsonl").read_text()
    )
    with pytest.raises(ValueError, match="overlap"):
        summarize(dirs)


def test_first_launch_and_resume_use_real_evaluator_directory_check(setup_eval):
    """Wrapper metadata must not collide with the evaluator's empty-dir check."""
    from pathlib import Path

    from examples.tutor.scripts.evaluate_api_teacher import prepare_output_dir

    config, args, env = setup_eval
    command, manifest, protocol = runner.prepare(config, args, env)
    args.output_dir.mkdir()
    (args.output_dir / "experiment.json").write_text(json.dumps(manifest))
    (args.output_dir / "protocol.yaml").write_text(yaml.safe_dump(protocol))
    (args.output_dir / "console.log").touch()
    (args.output_dir / "invocations.jsonl").touch()
    target = Path(command[command.index("--output-dir") + 1])
    assert target == args.output_dir / "evaluation"
    assert "--resume" not in command
    signature = {"test_protocol": "unchanged"}
    prepare_output_dir(target, signature=signature, resume=False)
    assert (target / "run_config.json").is_file()
    command, _, _ = runner.prepare(config, args, env)
    assert "--resume" in command
    prepare_output_dir(target, signature=signature, resume=True)
    with pytest.raises(ValueError, match="Resume settings differ"):
        prepare_output_dir(target, signature={"test_protocol": "changed"}, resume=True)
