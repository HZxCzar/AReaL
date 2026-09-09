import json
from dataclasses import asdict, fields

import pytest

from examples.tutor.scripts.evaluate_api_teacher import EpisodeResult, write_json
from examples.tutor.scripts.merge_tutor_eval_shards import merge_cell


def _shards(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for index in range(2):
        shard = tmp_path / "shards" / str(index)
        trace = shard / "traces" / f"row_{index}.json"
        write_json(trace, {"dataset_index": index})
        alias = shard / "adapter_alias"
        alias.symlink_to(checkpoint, target_is_directory=True)
        signature = {
            "shard": {"count": 2, "index": index},
            "dataset_size": 2,
            "dataset_sha256": "same-full-dataset-hash",
            "attempts": 1,
            "modes": [{"name": "presolve_on", "enabled": True}],
            "teacher": {
                "base_url": f"http://localhost:{index}",
                "request_params": {"extra_body": {"lora_path": str(alias)}},
            },
            "test_semantics": {
                "generalization_levels": [],
                "student_generalize_enabled": False,
                "student_generalize_replays": 0,
            },
        }
        write_json(shard / "run_config.json", {"signature": signature})
        values = {field.name: 0 for field in fields(EpisodeResult)}
        values.update(
            key=f"presolve_on:{index}:1",
            mode="presolve_on",
            presolve_enabled=True,
            dataset_index=index,
            attempt=1,
            item_id=str(index),
            student_name="student",
            student_model="student",
            termination_reason="max_turns",
            error=None,
            generalization={},
            personality_gate={},
            code_stats=None,
            student_prompt_index=None,
            trace_path=str(trace),
            solve_turn=None,
            teacher_pre_accepted=True,
        )
        (shard / "results.jsonl").write_text(
            json.dumps(asdict(EpisodeResult(**values))) + "\n"
        )


def test_merge_preserves_global_rows_and_trajectories(tmp_path):
    """Distinct pair endpoints/aliases retain the same model and global records."""
    _shards(tmp_path)
    report = merge_cell(tmp_path, 2)
    assert report["dataset_rows"] == report["recorded_total_attempts"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert [row["dataset_index"] for row in rows] == [0, 1]
    assert all("/shards/" in row["trace_path"] for row in rows)
    signature = json.loads((tmp_path / "run_config.json").read_text())["signature"]
    assert signature["dataset_sha256"] == "same-full-dataset-hash"
    assert "shard" not in signature


@pytest.mark.parametrize("defect", ["missing", "hash", "trajectory", "overlap"])
def test_merge_rejects_invalid_shards(tmp_path, defect):
    """Bad coverage, mismatched data, and missing traces never produce a summary."""
    _shards(tmp_path)
    shard = tmp_path / "shards" / "1"
    if defect == "missing":
        (shard / "results.jsonl").write_text("")
    elif defect == "hash":
        path = shard / "run_config.json"
        data = json.loads(path.read_text())
        data["signature"]["dataset_sha256"] = "different"
        write_json(path, data)
    elif defect == "trajectory":
        (shard / "traces" / "row_1.json").unlink()
    else:
        path = shard / "results.jsonl"
        data = json.loads(path.read_text())
        data.update(dataset_index=0, key="presolve_on:0:1")
        path.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError):
        merge_cell(tmp_path, 2)
    assert not (tmp_path / "summary.json").exists()
