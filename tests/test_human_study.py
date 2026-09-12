"""Offline human-study tests: synthetic inputs and fake HTTP, no model calls."""

import json
from pathlib import Path

import pytest

from examples.tutor.human_study import eval as runner
from examples.tutor.human_study import local_8gpu
from examples.tutor.human_study.common import load_dataset, read_json, write_json
from examples.tutor.human_study.export_judge import export
from examples.tutor.human_study.prepare_data import SOURCES, exclusion_reasons, prepare

PACKAGE = Path(runner.__file__).parent


def test_standard_scaffolding_prompt_preserves_two_sentence_requirement():
    study = runner.load_yaml(PACKAGE / "configs/study.yaml")
    case = {
        "problem": "Q",
        "history": [
            {"user": "Teacher", "text": "Explain."},
            {"user": "Student", "text": "S"},
        ],
    }
    assert runner.messages(case, study) == [
        {
            "role": "user",
            "content": "You are an experienced math teacher and you are going to respond to a student in a useful and caring way. The student is trying to solve the following problem.\n"
            "Problem: Q\nConversation:\nTeacher: Explain.\nStudent: S\n"
            "Teacher (maximum two sentences): ",
        }
    ]


def test_local_launch_allows_external_short_outputs():
    local_8gpu.check_ipc_path("/tmp/human-study")
    with pytest.raises(ValueError, match="too long"):
        local_8gpu.check_ipc_path("/tmp/" + "x" * 110)
    assert local_8gpu.gpu_groups("0,1,2,3,4,5,6,7") == ["0,1,2,3", "4,5,6,7"]
    with pytest.raises(ValueError):
        local_8gpu.gpu_groups("0,1,2,3,0,1,2,3")
    command = local_8gpu.server_command(Path("base"), Path("adapter"), 33101, True)
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--lora-paths") + 1] == "step1499=adapter"
    assert "--lora-paths" not in local_8gpu.server_command(
        Path("base"), Path("adapter"), 33100, False
    )


def test_local_cache_paths_are_inside_runtime(tmp_path):
    env = local_8gpu.environment(tmp_path)
    for key in [
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ]:
        assert Path(env[key]).is_relative_to(tmp_path)
    assert Path(env["TMPDIR"]) == tmp_path.parent / "tmp"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def source(tmp_path, empty=False):
    root = tmp_path / "source"
    root.mkdir()
    for split, name in SOURCES.items():
        write_json(
            root / name,
            [
                {
                    "problem": "" if empty else "How much is 2+2?",
                    "reference_solution": "PRIVATE SOLUTION",
                    "dialog_history": [
                        {"user": "Teacher", "text": "Try this."},
                        {"user": "Student", "text": "3?  \n"},
                        {"user": "Teacher", "text": "PRIVATE REFERENCE"},
                    ],
                }
            ],
        )
    return root


def settings(tmp_path, monkeypatch, name="first"):
    study = PACKAGE / "configs/study.yaml"
    model = tmp_path / (name + ".yaml")
    # JSON is also valid YAML; no test-only dependency needed.
    write_json(
        model,
        {
            "version": 1,
            "name": name,
            "model": name,
            "endpoint_env": "TEST_STUDY_ENDPOINT",
            "key_env": "TEST_STUDY_KEY",
            "lora_path": None,
            "identity": {"kind": "models"},
        },
    )
    monkeypatch.setenv("TEST_STUDY_ENDPOINT", "https://example.invalid/v1")
    monkeypatch.setenv("TEST_STUDY_KEY", "SECRET_TEST_TOKEN")
    return study, model


def fake_request(url, key, payload=None):
    if url.endswith("/models"):
        return {"data": [{"id": "first"}, {"id": "second"}]}
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Teacher: Hi\n\n<output>literal</output>  ",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 10},
    }


def test_prepare_full_no_dedup_and_no_reference_leak(tmp_path):
    data = prepare(source(tmp_path))
    assert (
        len(data["cases"]) == 2
    )  # identical question, distinct source contexts retained
    assert "pending" not in data
    for case in data["cases"]:
        assert case["history"][-1]["text"] == "3?  \n"
        assert "PRIVATE" not in json.dumps(case)


def test_empty_problem_is_excluded_without_manual_review(tmp_path):
    root = source(tmp_path, empty=True)
    data = prepare(root)
    assert len(data["excluded"]) == 2 and not data["cases"]
    assert "pending" not in data
    assert all(
        r["reasons"] == ["rule1_invalid_or_empty_problem"] for r in data["excluded"]
    )


def test_old_drafts_and_tampering_block_generation(tmp_path):
    data = prepare(source(tmp_path))
    path = tmp_path / "data.json"
    write_json(path, data)
    assert load_dataset(path) == data
    data["cases"] = []
    write_json(path, data)
    with pytest.raises(ValueError, match="changed"):
        load_dataset(path)
    write_json(path, {"schema_version": 1})
    with pytest.raises(ValueError, match="rule-based"):
        load_dataset(path)


@pytest.mark.parametrize(
    "turns",
    [
        None,
        {},
        [{"user": "Unknown", "text": "Hi"}],
        [{"user": "Teacher", "text": "  \n"}],
        [{"user": "Student", "text": None}],
        [{"user": [], "text": "Hi"}],
        ["not a turn"],
    ],
)
def test_rule2_invalid_fields(turns):
    assert exclusion_reasons({"problem": "Q", "dialog_history": turns}) == [
        "rule2_invalid_dialogue_fields"
    ]


def test_no_rules_three_four_or_turn_order_filters(tmp_path):
    root = source(tmp_path)
    record = {
        "problem": "Q",
        "dialog_history": [
            {"user": "Teacher", "text": "First"},
            {"user": "Student", "text": "Last"},
        ],
    }
    assert exclusion_reasons(record) == []
    write_json(root / SOURCES["standard"], [record])
    data = prepare(root)
    assert len(data["cases"]) == 2
    assert data["cases"][0]["history"] == record["dialog_history"][:-1]


def test_filter_does_not_modify_sources_and_records_overlap(tmp_path):
    root = source(tmp_path)
    write_json(
        root / SOURCES["standard"],
        [{"problem": "", "dialog_history": [{"user": "Student", "text": ""}]}],
    )
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    data = prepare(root)
    assert len(data["excluded"]) == 1
    assert len(data["excluded"][0]["reasons"]) == 2
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_raw_preservation_resume_and_export(tmp_path, monkeypatch):
    data = prepare(source(tmp_path))
    dataset = tmp_path / "data.json"
    write_json(dataset, data)
    study, model = settings(tmp_path, monkeypatch)
    calls = []

    def request(*args):
        calls.append(args)
        return fake_request(*args)

    left = tmp_path / "left"
    assert runner.run(dataset, study, model, left, request=request)
    assert len(calls) == 3
    assert runner.run(dataset, study, model, left, request=request)
    assert len(calls) == 4  # identity GET only, no repeated generation
    result = read_json(left / "records" / data["cases"][0]["id"] / "result.json")
    assert result["raw_response"] == "Teacher: Hi\n\n<output>literal</output>  "
    assert result["flags"]["has_training_tags"]
    assert "PRIVATE" not in json.dumps(result["request"])
    assert "stop" not in result["request"]
    assert "SECRET_TEST_TOKEN" not in "".join(
        p.read_text() for p in left.rglob("*.json")
    )
    _, other = settings(tmp_path, monkeypatch, "second")
    right = tmp_path / "right"
    assert runner.run(dataset, study, other, right, request=fake_request)
    output = tmp_path / "export"
    assert export(left, right, output) == 2
    human = (output / "human/items.jsonl").read_text()
    assert '"A": "Teacher: Hi\\n\\n<output>literal</output>  "' in human
    for forbidden in [
        "first",
        "second",
        "model",
        "source_index",
        "PRIVATE",
        "SECRET_TEST_TOKEN",
    ]:
        assert forbidden not in human
    assert all(
        set(json.loads(line)) == {"id", "problem", "history", "A", "B"}
        for line in human.splitlines()
    )
    with pytest.raises(ValueError, match="exists"):
        export(left, right, output)
    changed = read_json(model)
    changed["name"] = "changed"
    write_json(model, changed)
    with pytest.raises(ValueError, match="changed"):
        runner.run(dataset, study, model, left, request=fake_request)


def test_failed_request_not_silently_retried(tmp_path):
    case = prepare(source(tmp_path))["cases"][0]
    study = {"teacher_instruction": "Tutor", "generation": {"max_tokens": 10}}
    model = {"model": "m", "lora_path": None}

    def fail(*args):
        raise RuntimeError("SECRET_TEST_TOKEN")

    assert (
        runner.generate_case(tmp_path, case, study, model, "url", "key", request=fail)
        == "failed"
    )
    assert (
        runner.generate_case(tmp_path, case, study, model, "url", "key", request=fail)
        == "needs_retry_approval"
    )
    assert (
        runner.generate_case(
            tmp_path, case, study, model, "url", "key", retry=True, request=fake_request
        )
        == "generated"
    )
    assert "SECRET_TEST_TOKEN" not in "".join(
        p.read_text() for p in (tmp_path / "records").rglob("*.json")
    )


def test_response_saved_before_interruption_is_recovered(tmp_path):
    case = prepare(source(tmp_path))["cases"][0]
    study = {"teacher_instruction": "Tutor", "generation": {}}
    model = {"model": "m", "lora_path": None}
    attempt = tmp_path / "records" / case["id"] / "attempt-test"
    write_json(attempt / "request.json", runner.payload_for(case, study, model))
    write_json(attempt / "response.json", fake_request("url", "key", {}))
    assert (
        runner.generate_case(
            tmp_path,
            case,
            study,
            model,
            "url",
            "key",
            request=lambda *a: pytest.fail("no HTTP"),
        )
        == "recovered"
    )


def test_lora_identity_and_payload():
    model = runner.load_yaml(PACKAGE / "configs/models/trained_1500.yaml")
    study = runner.load_yaml(PACKAGE / "configs/study.yaml")
    runner.validate_configs(study, model)
    case = {"problem": "Q", "history": [{"user": "Student", "text": "S"}]}
    assert runner.payload_for(case, study, model)["lora_path"] == "step1499"
    with pytest.raises(ValueError, match="Adapter"):
        runner.check_identity(
            "https://example.invalid/v1",
            "key",
            model,
            request=lambda *a: {
                "model_path": "Qwen3-8B",
                "enable_lora": True,
                "lora_paths": [],
            },
        )


def test_empty_reply_is_retained_not_reasoning_fallback():
    response = {
        "choices": [
            {
                "message": {"content": None, "reasoning_content": "hidden"},
                "finish_reason": "length",
            }
        ]
    }
    result = runner.result_from_response({"id": "x"}, {}, response)
    assert result["raw_response"] == ""
    assert result["flags"]["empty"] and result["flags"]["length_limited"]


def test_export_rejects_mixed_protocol_and_missing_pairs(tmp_path, monkeypatch):
    data = prepare(source(tmp_path))
    dataset = tmp_path / "data.json"
    write_json(dataset, data)
    study, model = settings(tmp_path, monkeypatch)
    left, right = tmp_path / "left", tmp_path / "right"
    runner.run(dataset, study, model, left, request=fake_request)
    runner.run(dataset, study, model, right, request=fake_request)
    manifest = read_json(right / "manifest.json")
    changed = json.loads(json.dumps(manifest))
    changed["study"]["generation"]["temperature"] = 1
    write_json(right / "manifest.json", changed)
    with pytest.raises(ValueError, match="different"):
        export(left, right, tmp_path / "bad-export")
    write_json(right / "manifest.json", manifest)
    record_path = right / "records" / data["cases"][0]["id"] / "result.json"
    record = read_json(record_path)
    record["raw_response"] = "rewritten"
    write_json(record_path, record)
    with pytest.raises(ValueError, match="differ"):
        export(left, right, tmp_path / "bad-export")
    record_path.unlink()
    with pytest.raises(FileNotFoundError):
        export(left, right, tmp_path / "bad-export")
    assert not (tmp_path / "bad-export").exists()


def test_explicit_retry_after_malformed_response(tmp_path):
    case = prepare(source(tmp_path))["cases"][0]
    study = {"teacher_instruction": "Tutor", "generation": {}}
    model = {"model": "m", "lora_path": None}
    assert (
        runner.generate_case(
            tmp_path,
            case,
            study,
            model,
            "url",
            "key",
            request=lambda *a: {"error": "bad"},
        )
        == "failed"
    )
    assert (
        runner.generate_case(
            tmp_path, case, study, model, "url", "key", request=fake_request
        )
        == "needs_retry_approval"
    )
    assert (
        runner.generate_case(
            tmp_path, case, study, model, "url", "key", retry=True, request=fake_request
        )
        == "generated"
    )


def test_config_rejects_generation_model_override():
    study = runner.load_yaml(PACKAGE / "configs/study.yaml")
    model = runner.load_yaml(PACKAGE / "configs/models/trained_1500.yaml")
    study["generation"]["model"] = "wrong"
    with pytest.raises(ValueError, match="Unsupported"):
        runner.validate_configs(study, model)
