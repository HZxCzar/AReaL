"""The diverse-model arm changes student identity without preference conditioning."""

import json

import pytest

from examples.tutor.scripts.student_sim_eval import runner as shared
from examples.tutor.scripts.student_sim_eval.diverse_models import runner
from examples.tutor.scripts.student_sim_eval.prompted_strict import runner as prompted

ENV = dict(
    SIM_QWEN_MODEL="qwen3-1.7b",
    SIM_GEMMA_MODEL="gemma-3-1b-it",
    SIM_LLAMA_MODEL="llama-3.1-8b-instruct",
    SIM_PHI_MODEL="phi-4-mini-instruct",
    SIM_TEACHER_MODEL="teacher",
    SIM_TEACHER_PRICES="[1,0,0,1]",
    SIM_AUX_MODEL="judge",
)
CONFIG = shared.load_config(runner.PACKAGE / "config.yaml")


@pytest.mark.parametrize("row,strategy", shared.cells(CONFIG))
def test_each_cell_routes_its_student_without_gate_or_preference(row, strategy):
    """All 24 cells preserve the shared scientific protocol and short teacher style."""
    slot = next(s for s in CONFIG["students"] if s["name"] == row)
    experiment, protocol = runner.compile_cell(CONFIG, row, strategy, ENV)
    assert len(shared.cells(CONFIG)) == 24
    assert CONFIG["expected_questions"] == 50
    assert CONFIG["sampling"]["seed"] == 42
    assert experiment["roles"]["student"]["model"] == ENV[slot["model_env"]]
    assert experiment["roles"]["student"]["endpoint_env"] == "STUDENT_BASE_URL"
    assert protocol["student_axes"][0]["template"]["model"] == ENV[slot["model_env"]]
    assert protocol["student_axes"][0]["personalities"] == ["none"]
    assert protocol["student_system_prompt"] == ""
    assert protocol["personality"]["gate_sample_rate"] == 0
    assert (
        protocol["teacher_system_prompt"]
        == prompted.TEACHER_STRATEGIES[strategy] + prompted.TEACHER_REMINDER
    )
    assert protocol["max_turns"] == 10
    assert protocol["student_generalize"]["replays"] == 8
    assert protocol["length_retry"]["enabled"]
    assert protocol["leak_handling_mode"] == "reward_only"
    assert protocol["reward"]["leak_penalty"] == 0


def test_sample_selection_reuses_existing_ids_for_all_rows(tmp_path):
    """A pilot is a prefix of the same frozen list, not a separate per-model sample."""
    ids = [f"test-{i}" for i in range(50)]
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(ids))
    env = {"SIM_QUESTION_IDS": str(path)}
    assert shared.question_ids(CONFIG, env, 0) == ids
    assert shared.question_ids(CONFIG, env, 2) == ids[:2]


def test_rectangular_figure_shows_models_and_improvement_only(tmp_path):
    """The reused renderer accepts four-by-six matrices without a gate panel."""
    report = dict(
        method="different-models",
        cells=[
            dict(student=r, strategy=s, improvement_pp=12.5)
            for r, s in shared.cells(CONFIG)
        ],
    )
    path = tmp_path / "heatmap.svg"
    shared.render_svg(path, report)
    content = path.read_text()
    assert content.count("+12.5") == 24
    assert "improvement (pp)" in content
    assert "gate pass" not in content.lower()
    for slot in CONFIG["students"]:
        assert slot["name"] in content
