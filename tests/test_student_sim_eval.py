"""Matrix construction, real prompt integration and strict result validation."""

import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.student_sim_eval import runner


@pytest.fixture
def env():
    """No live endpoints or credentials are needed by these tests."""
    return {
        "SIM_TEACHER_MODEL": "teacher-test",
        "SIM_STUDENT_MODEL": "student-test",
        "SIM_AUX_MODEL": "judge-test",
        "SIM_TEACHER_PRICES": "[1, 0, 0, 1]",
        **{f"SIM_STUDENT_{s}_MODEL": f"student-{s}" for s in "ABC"},
    }


def preset(name):
    return runner.load_config(runner.PACKAGE / "configs" / f"{name}.yaml")


@pytest.mark.parametrize(
    "name,count", [("ours", 49), ("prompt-only", 49), ("different-models", 21)]
)
def test_matrix_has_each_pair_exactly_once(name, count):
    """None is a single row/column, not extra evaluation passes."""
    pairs = runner.cells(preset(name))
    assert len(pairs) == len(set(pairs)) == count


def test_methods_share_protocol_except_simulation(env):
    """Prompt-only cannot accidentally retain gates or change the scoring protocol."""
    row, strategy = "subgoal-decomposition", "contrastive-comparison"
    _, ours = runner.compile_cell(preset("ours"), row, strategy, env)
    _, prompt = runner.compile_cell(preset("prompt-only"), row, strategy, env)
    assert ours["student_axes"][0]["personalities"] == [row]
    assert prompt["student_axes"][0]["personalities"] == ["none"]
    assert prompt["personality"]["gate_sample_rate"] == 0
    assert prompt["student_system_prompt"]
    assert not ours["student_system_prompt"]
    assert runner.STRATEGIES[strategy] in ours["teacher_system_prompt"]
    assert runner.PREFERENCES[row] not in ours["teacher_system_prompt"]
    for value in (ours, prompt):
        value.pop("student_system_prompt")
        value["student_axes"][0].pop("personalities")
        value["personality"].pop("gate_sample_rate")
    assert ours == prompt


def test_none_is_equivalent_across_preference_methods(env):
    """Unconstrained rows differ only by an inactive gate sampling setting."""
    _, ours = runner.compile_cell(preset("ours"), "none", "none", env)
    _, prompt = runner.compile_cell(preset("prompt-only"), "none", "none", env)
    ours["personality"]["gate_sample_rate"] = 0
    assert ours == prompt


def test_different_model_changes_actual_student_not_label_only(env):
    """Role and Hydra template must select the same deployed model."""
    experiment, protocol = runner.compile_cell(
        preset("different-models"), "student-b", "none", env
    )
    assert experiment["roles"]["student"]["model"] == "student-B"
    assert protocol["student_axes"][0]["template"]["model"] == "student-B"
    assert protocol["student_axes"][0]["personalities"] == ["none"]


@pytest.mark.parametrize(
    "method,row",
    [
        ("ours", "contrastive-comparison"),
        ("prompt-only", "contrastive-comparison"),
        ("different-models", "student-a"),
    ],
)
def test_all_methods_record_leaks_without_intervention(env, method, row):
    """All simulation methods audit leaks without an online leak gate or penalty."""
    _, protocol = runner.compile_cell(preset(method), row, "causal-justification", env)
    assert protocol["leak_handling_mode"] == "reward_only"
    assert protocol["reward"]["leak_penalty"] == 0.0
    assert protocol["teacher_anti_leak_instruction_enabled"] is True
    assert protocol["personality"]["explain_ratio"] == 0.0
    assert protocol["personality"]["gate_sample_rate"] == (
        1.0 if method == "ours" else 0.0
    )


@pytest.mark.parametrize("personality", [s for s in runner.STRATEGIES if s != "none"])
def test_ablation_complaints_use_shared_bare_pool(env, personality):
    """The actual complaint selector must not draw preference explanations."""
    import random

    from examples.tutor.workflow import TutorAgentWorkflow

    _, protocol = runner.compile_cell(preset("ours"), personality, "none", env)
    pool = json.loads(
        (runner.REPO / protocol["personality"]["complaints_path"]).read_text()
    )
    state = SimpleNamespace(
        personality_explain_ratio=protocol["personality"]["explain_ratio"],
        personality_complaints_bare=pool["bare"],
        personality_complaints_explain=pool["explain"],
        _personality_rng=lambda **kwargs: random.Random(kwargs["turn_idx"]),
    )
    for turn in range(1, 11):
        complaint, explained = TutorAgentWorkflow._draw_personality_complaint(
            state, personality, turn_idx=turn
        )
        assert complaint in pool["bare"]
        assert explained is False


@pytest.mark.parametrize(
    "method,row",
    [
        ("ours", "subgoal-decomposition"),
        ("prompt-only", "subgoal-decomposition"),
        ("different-models", "student-a"),
    ],
)
def test_generated_protocol_loads_through_actual_evaluator(
    tmp_path, monkeypatch, env, method, row
):
    """Check real Hydra/dataclass validation and student-axis expansion without API calls."""
    from examples.tutor.scripts.evaluate_api_teacher import load_experiment_config

    for name in ("EVAL_RUN_STUDENT_URL", "EVAL_RUN_AUX_URL"):
        monkeypatch.setenv(name, "http://localhost:12345/v1")
    for name in ("EVAL_RUN_STUDENT_KEY", "EVAL_RUN_AUX_KEY"):
        monkeypatch.setenv(name, "EMPTY")
    _, protocol = runner.compile_cell(
        preset(method), row, "contrastive-comparison", env
    )
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    config, students = load_experiment_config(str(path), [])
    assert len(students) == 1
    assert config.teacher_system_prompt == protocol["teacher_system_prompt"]
    assert config.student_system_prompt == protocol["student_system_prompt"]
    assert config.length_retry.enabled
    assert config.student_generalize.replays == 8


@pytest.mark.parametrize(
    "field,value",
    [("strategies", ["none", "none"]), ("method", "other"), ("unexpected", True)],
)
def test_invalid_matrix_is_rejected(tmp_path, field, value):
    """Reject misspellings and duplicate cells before API execution."""
    config = preset("ours")
    config[field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        runner.load_config(path)


def test_cyclic_inheritance_is_rejected(tmp_path):
    """Recursive extends cannot hang the planner."""
    path = tmp_path / "cycle.yaml"
    path.write_text("extends: cycle.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        runner.load_config(path)


def test_free_chat_preference_uses_real_shared_resolver():
    """The adapter augments the actual free-chat resolver, preserving the retest task."""
    from examples.tutor.scripts.student_sim_eval.evaluate import with_preference
    from examples.tutor.workflow import TutorAgentWorkflow

    state = SimpleNamespace(student_system_prompt="Preference sentinel")
    original = TutorAgentWorkflow._free_chat_student_prompts
    base, retest = original(state)
    actual, actual_retest = with_preference(original)(state)
    assert actual == base + "\n\nPreference sentinel"
    assert actual_retest == retest
    state.student_system_prompt = ""
    assert with_preference(original)(state) == (base, retest)


@pytest.mark.parametrize("strategy", ["none", "step-demonstration"])
@pytest.mark.parametrize("anti_leak", [True, False])
def test_teacher_reminder_is_request_only_and_history_is_unchanged(
    env, strategy, anti_leak
):
    """Repeat one tail reminder while preserving raw dialogue and excluding presolve."""
    from examples.tutor.prompts import FREE_CHAT_TEACHER_STUDENT_AWARENESS_CONTEXT
    from examples.tutor.scripts.student_sim_eval.evaluate import with_strategy
    from examples.tutor.workflow import TutorAgentWorkflow

    turns = []
    state = SimpleNamespace(
        task="2+2?",
        ground_truth="GROUND_TRUTH_SENTINEL",
        public_history=SimpleNamespace(turns=turns),
        teacher_pre_solve_result=SimpleNamespace(raw_output="PRESOLVE_SENTINEL"),
    )
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.max_turns = 10
    workflow.teacher_show_ground_truth = False
    workflow.teacher_anti_leak_instruction_enabled = anti_leak
    workflow.teacher_response_format = "thinking"
    workflow.teacher_end_enabled = True
    _, protocol = runner.compile_cell(preset("ours"), "none", strategy, env)
    workflow.free_chat_student_awareness_prompt_enabled = protocol["free_chat"][
        "student_awareness_prompt_enabled"
    ]
    prefix = protocol["teacher_system_prompt"]
    wrapped = with_strategy(prefix)
    first = wrapped(workflow, state)
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[0]["content"] == workflow._free_chat_teacher_system(
        state.task, state.ground_truth
    )
    assert FREE_CHAT_TEACHER_STUDENT_AWARENESS_CONTEXT in first[0]["content"]
    assert turns == []
    expected_reminder = (
        "Teaching the student using the following method.\n\n" + prefix
        if prefix
        else "Teaching the student."
    )
    if anti_leak:
        expected_reminder += "\n\nPlease do not directly reveal the final answer or an equivalent expression."
    assert first[-1]["content"] == expected_reminder
    assert "2+2?" in first[0]["content"]
    assert "2+2?" not in first[1]["content"]
    for reply in ("Student's actual work", "I am still confused."):
        turns.extend(
            [
                {"role": "teacher", "content": "Teacher's unmodified reply"},
                {"role": "student", "content": reply},
            ]
        )
        frozen = copy.deepcopy(turns)
        request = wrapped(workflow, state, clean=True)
        assert request[0] == first[0]
        assert "No matter what" not in str(request)
        assert "ASSIGNED STYLE" not in str(request)
        assert request[1:-1] == workflow._render_conversation(turns, speaker="teacher")
        assert request[-2] == {"role": "user", "content": reply}
        assert request[-1] == first[-1]
        assert turns == frozen
        assert sum(m == first[-1] for m in request) == 1
        assert "PRESOLVE_SENTINEL" not in str(request)
        assert "GROUND_TRUTH_SENTINEL" not in str(request)
        if prefix:
            assert prefix in request[-1]["content"]
            assert all(prefix not in m["content"] for m in request[:-1])
        else:
            assert "ASSIGNED STYLE" not in str(request)
    student_view = workflow._render_conversation(turns, speaker="student")
    assert first[-1]["content"] not in str(student_view)


def record(index=0):
    return dict(
        key=f"student:{index}:0",
        dataset_index=index,
        item_id=str(index),
        attempt=0,
        generalization={"original": {"replay_count": 8, "score": 0.75}},
        no_teaching_baseline=0.25,
    )


def write_records(path, records):
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_summary_uses_latest_record_and_correct_units(tmp_path):
    """A retried episode is counted once; fractional accuracy becomes percentage points."""
    old = record()
    old["error"] = "retryable"
    write_records(tmp_path, [old, record(), record(1)])
    metrics, identities = runner.cell_metrics(tmp_path, 2, 8)
    assert metrics["improvement_pp"] == 50
    assert len(identities) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"error": "timeout"},
        {"no_teaching_baseline": None},
        {"generalization": {"original": {"replay_count": 7, "score": 1}}},
        {"personality_gate": {"gate_error_count": 1}},
    ],
)
def test_summary_rejects_diagnostic_or_incomplete_retest(tmp_path, change):
    """Infrastructure failures must not be silently treated as wrong answers."""
    item = record()
    item.update(copy.deepcopy(change))
    write_records(tmp_path, [item])
    with pytest.raises(ValueError):
        runner.cell_metrics(tmp_path, 1, 8)


def test_summary_checks_cross_cell_question_identity(tmp_path):
    """Equal counts do not excuse comparing different questions."""
    manifest = dict(
        method="ours",
        questions=1,
        cells=[
            dict(row="none", strategy=s, path=s, replays=8)
            for s in ("none", "subgoal-decomposition")
        ],
    )
    for index, cell in enumerate(manifest["cells"]):
        write_records(tmp_path / cell["path"] / "evaluation", [record(index)])
    with pytest.raises(ValueError, match="same questions"):
        runner.summarize(tmp_path, manifest)


def test_summary_writes_heatmap_and_csv(tmp_path):
    """A complete matrix produces portable plotting data and a standalone SVG."""
    manifest = dict(
        method="ours",
        questions=1,
        cells=[dict(row="none", strategy="none", path="none/none", replays=8)],
    )
    write_records(tmp_path / "none/none/evaluation", [record()])
    runner.summarize(tmp_path, manifest)
    assert "+50.0" in (tmp_path / "heatmap.svg").read_text()
    assert "improvement_pp" in (tmp_path / "heatmap.csv").read_text()


def test_resume_allows_budget_only_not_scientific_change():
    """Raising a spending cap cannot authorize changes to a student or teacher."""
    manifest = {
        "cells": [{"experiment": {"teacher": "same", "execution": {"budget_usd": 2}}}]
    }
    changed = copy.deepcopy(manifest)
    changed["cells"][0]["experiment"]["execution"]["budget_usd"] = 10
    assert runner.scientific_identity(manifest) == runner.scientific_identity(changed)
    changed["cells"][0]["experiment"]["teacher"] = "other"
    assert runner.scientific_identity(manifest) != runner.scientific_identity(changed)


def test_teacher_settings_match_existing_luna_protocol(env):
    """Use the same Luna format/provider policies, not a new capped teacher path."""
    experiment, _ = runner.compile_cell(preset("ours"), "none", "none", env)
    existing = runner.api.load_config(runner.api.PACKAGE / "configs/gpt-5.6-luna.yaml")
    for name in (
        "provider",
        "format",
        "reasoning_effort",
        "sampling",
        "output_limit",
        "endpoint_env",
        "key_env",
    ):
        assert experiment["teacher"][name] == existing["teacher"][name]


def test_reviewed_question_list_is_frozen_before_smoke_limit(tmp_path):
    """A smoke test is a prefix of reviewed IDs, not a prefix of the raw dataset."""
    path = tmp_path / "ids.json"
    ids = [f"question-{index}" for index in range(100)]
    path.write_text(json.dumps(ids))
    config = preset("ours")
    assert runner.question_ids(config, {"SIM_QUESTION_IDS": str(path)}, 2) == ids[:2]
    assert runner.question_ids(config, {"SIM_QUESTION_IDS": str(path)}, 0) == ids
    path.write_text(json.dumps(["duplicate"] * 100))
    with pytest.raises(ValueError, match="Duplicate"):
        runner.question_ids(config, {"SIM_QUESTION_IDS": str(path)}, 0)


def test_dataset_selection_uses_ids_not_row_positions():
    """Every strategy receives the same reviewed order through the real Dataset API."""
    from datasets import Dataset

    from examples.tutor.scripts.student_sim_eval.evaluate import select_questions

    dataset = Dataset.from_dict({"id": ["c", "a", "b"], "question": ["C", "A", "B"]})
    selected = select_questions(dataset, ["b", "c"])
    assert selected["question"] == ["B", "C"]
    with pytest.raises(ValueError, match="missing"):
        select_questions(dataset, ["not-present"])


def test_seeded_selection_is_stable_and_not_first_rows():
    """Every method/strategy shares the same seeded sample despite row reordering."""
    population = [f"q-{i}" for i in range(528)]
    selected = runner.sample_ids(population, 100, 42)
    assert selected == runner.sample_ids(list(reversed(population)), 100, 42)
    assert len(selected) == len(set(selected)) == 100
    assert selected != runner.sample_ids(population, 100, 43)
    assert set(selected) != set(population[:100])
    assert set(runner.sample_ids(population, 528, 42)) == set(population)


def test_default_sampling_loads_local_test_split(monkeypatch):
    """The full-run selector uses the test split and applies smoke limits afterward."""
    import datasets

    population = [f"q-{i}" for i in range(528)]
    monkeypatch.setattr(
        datasets, "load_from_disk", lambda path: {"test": {"id": population}}
    )
    expected = runner.sample_ids(population, 100, 42)
    assert runner.question_ids(preset("ours"), {}, 2) == expected[:2]


@pytest.mark.parametrize("strategy", [s for s in runner.STRATEGIES if s != "none"])
def test_every_style_contains_only_a_short_positive_recipe(env, strategy):
    """Model-facing descriptions contain no labels or extra persistence contract."""
    _, protocol = runner.compile_cell(preset("ours"), "none", strategy, env)
    prompt = protocol["teacher_system_prompt"]
    assert prompt == runner.STRATEGIES[strategy]
    assert prompt.startswith("Teaching ")
    assert "No matter what" not in prompt
    assert "ASSIGNED STYLE" not in prompt
    assert len(runner.STRATEGIES[strategy].split()) < 85
    assert "do not" not in prompt.lower()
    assert "complain" not in prompt.lower()
    assert protocol["teacher_adaptive_instruction_enabled"] is False
    assert protocol["free_chat"]["student_awareness_prompt_enabled"] is True


def test_none_remains_unassigned_not_a_seventh_fixed_style(env):
    """None has no anti-switch contract; its teacher may choose its own approach."""
    _, protocol = runner.compile_cell(preset("ours"), "none", "none", env)
    assert protocol["teacher_system_prompt"] == ""


def test_fixed_style_opening_does_not_ask_teacher_to_switch(env):
    """The standard opening must not contradict the experiment's fixed-style assignment."""
    from examples.tutor.prompts import TEACHER_ADAPTIVE_INSTRUCTION
    from examples.tutor.workflow import TutorAgentWorkflow

    _, protocol = runner.compile_cell(
        preset("ours"), "none", "subgoal-decomposition", env
    )
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.teacher_response_format = "thinking"
    workflow.teacher_adaptive_instruction_enabled = protocol[
        "teacher_adaptive_instruction_enabled"
    ]
    assert TEACHER_ADAPTIVE_INSTRUCTION not in workflow._free_chat_open_prompt()
