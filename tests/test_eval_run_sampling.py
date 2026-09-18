"""Protocol teacher sampling reaches requests without changing existing presets."""

import json
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.eval_run import runner


@pytest.mark.parametrize("official", [False, True])
@pytest.mark.parametrize("preset", ["untrained", "all-legacy-1500"])
def test_teacher_protocol_sampling_preserved(tmp_path, official, preset):
    """Both teacher formats keep explicit sampling; old protocol stays unchanged."""
    config = runner.load_config(runner.PACKAGE / "configs" / f"{preset}.yaml")
    if official:
        protocol = yaml.safe_load(open(config["protocol"]))
        protocol["gconfig"].update(temperature=0.7, top_p=0.8)
        protocol["eval_gconfig"]["top_p"] = 0.8
        protocol["teacher_api_request_params"] = {"extra_body": {"top_k": 20, "min_p": 0}}
        path = tmp_path / "official.yaml"
        path.write_text(yaml.safe_dump(protocol))
        config["protocol"] = str(path)
    env = {
        "TEACHER_BASE_URL": "http://teacher.test/v1", "TEACHER_API_KEY": "EMPTY",
        "STUDENT_BASE_URL": "http://student.test/v1", "STUDENT_API_KEY": "EMPTY",
        "AUX_BASE_URL": "http://aux.test/v1", "AUX_API_KEY": "EMPTY",
        "TEACHER_ADAPTER": "test-adapter",
    }
    args = SimpleNamespace(output_dir=tmp_path / "out", limit=0, shard_count=4, shard_index=0)
    command, _, protocol = runner.prepare(config, args, env)
    params = json.loads(command[command.index("--teacher-request-params") + 1])
    extra = params["extra_body"]
    assert params["seed"] == 42
    assert extra["chat_template_kwargs"]["enable_thinking"] is False
    assert protocol["teacher_response_format"] == ("thinking" if preset == "untrained" else "non_thinking")
    assert ("lora_path" in extra) == (preset != "untrained")
    assert "lora_path" not in protocol["teacher_api_request_params"]["extra_body"]
    if official:
        assert extra["top_k"] == 20 and extra["min_p"] == 0
        assert protocol["gconfig"]["temperature"] == 0.7
        assert protocol["eval_gconfig"]["top_p"] == 0.8
    else:
        assert "top_k" not in extra and "min_p" not in extra
        assert protocol["gconfig"]["temperature"] == 1.0
        assert protocol["eval_gconfig"]["top_p"] == 1.0
