"""Portable checkpoint evaluation: protocol, privacy, sharding and resume tests."""

import json
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.eval_run import runner
from examples.tutor.scripts.eval_run.summarize import (
    gate_counts,
    leak_counts,
    summarize,
)


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


@pytest.mark.parametrize(
    "preset",
    ["untrained", "subgoal-1500", "all-legacy-1500", "all-gt0-1500", "pedrl-1500"],
)
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


@pytest.mark.parametrize("preset", ["pedrl-1500", "all-gt0-1500"])
def test_pedrl_uses_same_protocol_as_trained_teachers(setup_eval, preset):
    """PedRL changes only teacher identity/adapter, not the comparison protocol."""
    _, args, env = setup_eval
    env["TEACHER_ADAPTER"] = "test-adapter"
    ours = runner.load_config(runner.PACKAGE / "configs/all-legacy-1500.yaml")
    pedrl = runner.load_config(runner.PACKAGE / "configs" / f"{preset}.yaml")
    _, _, ours_protocol = runner.prepare(ours, args, env.copy())
    _, _, pedrl_protocol = runner.prepare(pedrl, args, env.copy())
    assert ours_protocol == pedrl_protocol
    assert pedrl["teacher"]["format"] == "non_thinking"
    assert pedrl["teacher"]["enable_thinking"] is False
    assert pedrl["roles"]["auxiliary"]["model"] == "qwen3.8-27b-fp8"


def test_untrained_format_does_not_enable_thinking(setup_eval):
    config, args, env = setup_eval
    command, _, protocol = runner.prepare(config, args, env)
    params = json.loads(command[command.index("--teacher-request-params") + 1])
    assert protocol["teacher_response_format"] == "thinking"
    assert protocol["enable_thinking"] is False
    assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "lora_path" not in params["extra_body"]
    assert protocol["length_retry"] == {"enabled": True, "attempts": 3}


def test_no_presolve_changes_only_teacher_pre_enabled(setup_eval):
    """Ablation retains student baseline, format and the complete paper protocol."""
    config, args, env = setup_eval
    _, before_manifest, before = runner.prepare(config, args, env.copy())
    ablation = runner.load_config(runner.PACKAGE / "configs/untrained-no-presolve.yaml")
    _, after_manifest, after = runner.prepare(ablation, args, env.copy())
    assert before["evaluator"]["teacher_pre_enabled"] is True
    assert after["evaluator"]["teacher_pre_enabled"] is False
    after["evaluator"]["teacher_pre_enabled"] = True
    assert before == after
    assert before_manifest["protocol_sha256"] != after_manifest["protocol_sha256"]
    assert ablation["teacher"]["adapter_env"] is None


@pytest.mark.parametrize("enabled", [True, False])
def test_completion_checks_selected_presolve_mode(setup_eval, enabled):
    """Both ablation arms finish; a summary for the opposite arm is rejected."""
    config, args, _ = setup_eval
    config["teacher"]["presolve"] = enabled
    directory = args.output_dir / "evaluation"
    directory.mkdir(parents=True)
    mode = "presolve_on" if enabled else "presolve_off"
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "modes": {mode: {"completed_attempts": 924}},
                "pending_backfill": {"count": 0},
            }
        )
    )
    assert runner.completion_status(args.output_dir, config, args)["complete"]
    config["teacher"]["presolve"] = not enabled
    assert not runner.completion_status(args.output_dir, config, args)["complete"]


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


def test_archiving_preserves_manifest_and_checks_content(
    setup_eval, tmp_path, monkeypatch
):
    """Moving an old launcher preserves resume identity, not permission to edit it."""
    config, args, env = setup_eval
    _, _, protocol = runner.prepare(config, args, env)
    root = tmp_path / "repo"
    for field in ("prompts_path", "complaints_path"):
        relative = protocol["personality"][field]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((runner.REPO / relative).read_bytes())
    monkeypatch.setattr(runner, "REPO", root)
    original = root / "examples/tutor/scripts/old.py"
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text("# historical launcher\n")
    _, before, _ = runner.prepare(config, args, env)
    archived = root / "legacy/eval/examples/tutor/scripts/old.py"
    archived.parent.mkdir(parents=True)
    original.rename(archived)
    _, after, _ = runner.prepare(config, args, env)
    assert before == after
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(before))
    runner.check_manifest(manifest, after)
    archived.write_text("# changed launcher\n")
    _, changed, _ = runner.prepare(config, args, env)
    with pytest.raises(ValueError, match="changed"):
        runner.check_manifest(manifest, changed)
    original.write_text("# conflicting launcher\n")
    with pytest.raises(ValueError, match="Conflicting"):
        runner.prepare(config, args, env)


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


def test_gate_fail_resume_migration_is_exact_and_preserves_hashes():
    """Only the reviewed gate acceptance change is compatible with old output."""
    path = "examples/tutor/scripts/evaluate_api_teacher.py"
    old_hash = "e27b6afbc99791adcc90d638f83c76fd3aa7c2f6b02b91d13289b4cbdc066ff5"
    new_hash = "97c317910d0e10e0f71d627e2ef8e4023e626816a5b9240e14946e47168c788f"
    old = {"source_sha256": {path: old_hash}}
    new = {"source_sha256": {path: new_hash}}
    assert runner.resume_identity(old) == runner.resume_identity(new)
    assert new["source_sha256"][path] == new_hash
    new["source_sha256"][path] = "unreviewed-edit"
    assert runner.resume_identity(old) != runner.resume_identity(new)


def test_gate_fail_resume_reaches_inner_signature_check(setup_eval):
    """The outer runner must authorize the exact migration in the inner guard."""
    from examples.tutor.scripts.evaluate_api_teacher import prepare_output_dir

    config, args, env = setup_eval
    directory = args.output_dir / "evaluation"
    directory.mkdir(parents=True)
    old = {
        "evaluator_sha256": "b7dca3800f4a98b30f90c46b2ca0f593e233969b1f0ede2c024d7a9d595616f3",
        "teacher_model": "unchanged",
    }
    (directory / "run_config.json").write_text(json.dumps({"signature": old}))
    command, _, _ = runner.prepare(config, args, env)
    assert "--allow-evaluator-code-change-on-resume" in command
    current = dict(old, evaluator_sha256="d983e84980eda38ebeaa748e9d126ab1dcadbbc5d4739736af38c53d2f6af251")
    prepare_output_dir(directory, signature=current, resume=True, allow_evaluator_code_change=True)
    with pytest.raises(ValueError, match="Resume settings differ"):
        prepare_output_dir(directory, signature=dict(current, teacher_model="different"),
                           resume=True, allow_evaluator_code_change=True)


def test_public_files_contain_no_cluster_literals():
    for path in runner.PACKAGE.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
            text = path.read_text()
            assert "/inspire/" not in text
            assert "sii.edu.cn" not in text
    assert "defaults:" not in (runner.PACKAGE / "protocol.yaml").read_text()


@pytest.mark.parametrize(
    "preference", ["attempt-diagnosis", "independent-verification"]
)
def test_gate_counts_excludes_only_actual_first_turn_calls(preference):
    """Subtract sampled first turns, not one turn per episode."""
    rows = [
        {
            "personality_gate": dict(
                sampled_turn_count=3,
                passed_turn_count=2,
                turn1_sampled=True,
                turn1_passed=True,
            )
        },
        {
            "personality_gate": dict(
                sampled_turn_count=2,
                passed_turn_count=1,
                turn1_sampled=False,
                turn1_passed=None,
            )
        },
    ]
    assert gate_counts(rows, preference) == (2, 4)
    assert gate_counts(rows, "subgoal-decomposition") == (3, 5)


@pytest.mark.parametrize(
    "preference", ["attempt-diagnosis", "independent-verification"]
)
def test_gate_counts_first_turn_only_has_no_eligible_calls(preference):
    """A warm-start-only episode contributes neither a success nor a failure."""
    rows = [
        {
            "personality_gate": dict(
                sampled_turn_count=1,
                passed_turn_count=1,
                turn1_sampled=True,
                turn1_passed=True,
            )
        }
    ]
    assert gate_counts(rows, preference) == (0, 0)
    with pytest.raises(ValueError, match="Missing turn-1"):
        gate_counts(
            [{"personality_gate": dict(sampled_turn_count=1, passed_turn_count=1)}],
            preference,
        )


def test_leak_counts_uses_turns_not_episode_incidence():
    """Unequal episode lengths and multiple leaks use pooled turn counts."""
    rows = [dict(leak_count=2, num_turns=3), dict(leak_count=0, num_turns=7)]
    assert leak_counts(rows) == (2, 10)
    assert leak_counts([dict(leak_count=0, num_turns=0)]) == (0, 0)
    with pytest.raises(ValueError, match="Inconsistent leak"):
        leak_counts([dict(leak_count=2, num_turns=1)])


@pytest.mark.parametrize("gate_errors", [0, 1])
def test_merge_uses_latest_records_and_rejects_overlapping_shards(tmp_path, gate_errors):
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
            "personality_gate": {
                "personality": "none",
                "gate_error_count": gate_errors,
                "sampled_turn_count": 1,
                "passed_turn_count": 0,
            },
            "generalization": {"original": {"replay_count": 8, "score": 1}},
            "no_teaching_baseline": 0.5,
            "leak_count": 2 if i == 0 else 0,
            "num_turns": 3 if i == 0 else 7,
        }
        (directory / "evaluation/results.jsonl").write_text(
            json.dumps(row) + "\n" + json.dumps(row) + "\n"
        )
    report = summarize(dirs)
    assert report["cells"]["none"]["episodes"] == 2
    assert report["cells"]["none"]["improvement_pp"] == 50
    assert report["cells"]["none"]["leak_percent"] == 20
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
