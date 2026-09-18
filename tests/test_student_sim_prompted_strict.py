"""Strict student prompts are turn-local and cannot contaminate scored tests."""

import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from examples.tutor.scripts.student_sim_eval import runner as shared
from examples.tutor.scripts.student_sim_eval.prompted_strict import runner
from examples.tutor.scripts.student_sim_eval.prompted_strict.prompts import (
    INSTRUCTION,
    turn_instruction,
    with_student_instruction,
)


def compiled(preference="independent-verification"):
    """Compile locally, without deployment access or credentials."""
    config = shared.load_config(runner.PACKAGE / "config.yaml")
    env = dict(
        SIM_TEACHER_MODEL="teacher-test",
        SIM_STUDENT_MODEL="student-test",
        SIM_AUX_MODEL="judge-test",
        SIM_TEACHER_PRICES="[1,0,0,1]",
    )
    return config, runner.compile_cell(config, preference, "step-demonstration", env)


@pytest.mark.parametrize("preference", runner.TEACHER_STRATEGIES)
def test_compiler_uses_exact_gate_criterion_without_external_gate(preference):
    """All six student types share one instruction and preserve scientific settings."""
    config, (experiment, protocol) = compiled(preference)
    assert len(shared.cells(config)) == 36
    assert config["expected_questions"] == 50
    assert config["sampling"]["seed"] == 42
    gates = json.loads(
        (shared.REPO / protocol["personality"]["prompts_path"]).read_text()
    )
    complaints = json.loads(
        (shared.REPO / protocol["personality"]["complaints_path"]).read_text()
    )
    assert all(
        reply not in protocol["student_system_prompt"] for reply in complaints["bare"]
    )
    assert protocol["personality"]["explain_ratio"] == 0
    assert protocol["student_system_prompt"] == (
        INSTRUCTION
        + "\n\nLearning preference:\n"
        + gates["personalities"][preference]["preference"]
    )
    assert protocol["student_axes"][0]["personalities"] == ["none"]
    assert protocol["personality"]["gate_sample_rate"] == 0
    assert protocol["teacher_system_prompt"] == (
        runner.TEACHER_STRATEGIES["step-demonstration"] + runner.TEACHER_REMINDER
    )
    assert experiment["run_name"].startswith("prompted-strict-")
    assert protocol["max_turns"] == 10
    assert protocol["student_generalize"]["replays"] == 8
    assert protocol["length_retry"]["enabled"]
    assert protocol["leak_handling_mode"] == "reward_only"
    assert protocol["reward"]["leak_penalty"] == 0


def test_seeded_complaint_is_single_and_independent_of_call_order():
    """Resume and concurrent ordering preserve each problem/turn draw."""
    pool = [f"Scripted reply {index}." for index in range(6)]
    states = [
        SimpleNamespace(
            task="fixed problem",
            public_history=SimpleNamespace(
                turns=[dict(role="student", content="reply")] * i
            ),
        )
        for i in range(30)
    ]
    forward = [turn_instruction(INSTRUCTION, pool, 42, state) for state in states]
    backward = [
        turn_instruction(INSTRUCTION, pool, 42, state) for state in reversed(states)
    ]
    assert forward == backward[::-1]
    assert len(set(forward)) > 1
    assert all(sum(reply in prompt for reply in pool) == 1 for prompt in forward)


def test_actual_student_builder_keeps_history_system_and_tests_unchanged(monkeypatch):
    """Exercise real dialogue and probe builders across two turns, without APIs."""
    from examples.tutor.workflow import TutorAgentWorkflow

    _, (_, protocol) = compiled()
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.student_system_prompt = protocol["student_system_prompt"]
    from examples.tutor.core.types import PublicHistoryState, StudentTurnState

    state = StudentTurnState(
        task="HIDDEN_PROBLEM",
        previous_student_output="",
        public_history=PublicHistoryState(turns=[]),
        latest_tutor_visible_output="Teacher's first reply",
        student_mode="text",
        student_mask=None,
        student_turn_behavior=None,
    )
    original = TutorAgentWorkflow._build_student_messages
    wrapped = with_student_instruction(original, protocol)
    system_before, test_template_before = workflow._free_chat_student_prompts()
    first = wrapped(workflow, state)
    assert first[0] == dict(role="system", content=system_before)
    assert first[-1][
        "content"
    ] == state.latest_tutor_visible_output + "\n\n" + turn_instruction(
        protocol["student_system_prompt"],
        json.loads(
            (shared.REPO / protocol["personality"]["complaints_path"]).read_text()
        )["bare"],
        protocol["seed"],
        state,
    )
    assert "HIDDEN_PROBLEM" not in str(first)
    assert state.public_history.turns == []

    state.public_history.turns.extend(
        [
            dict(role="teacher", content=state.latest_tutor_visible_output),
            dict(role="student", content="I don't understand."),
        ]
    )
    state.latest_tutor_visible_output = "Teacher's next reply"
    before = copy.deepcopy(state)
    monkeypatch.setattr(TutorAgentWorkflow, "_build_student_messages", wrapped)
    second = workflow._build_student_messages(state)
    assert second[1:-1] == [
        dict(role="user", content="Teacher's first reply"),
        dict(role="assistant", content="I don't understand."),
    ]
    assert INSTRUCTION not in str(second[:-1])
    assert second[-1]["content"].count("Before responding, check whether") == 1
    assert state == before
    assert workflow._free_chat_student_prompts() == (
        system_before,
        test_template_before,
    )
    # Retests use the independent probe path, not the modified dialogue builder.
    probe = workflow._build_student_probe_messages(
        episode_artifact=SimpleNamespace(
            task="HIDDEN_PROBLEM", student_name="test", student_mode="text"
        ),
        anchor=SimpleNamespace(public_history=state.public_history),
        level="original",
        transfer_task="RETEST_PROBLEM",
    )
    assert INSTRUCTION not in str(probe)
    assert "RETEST_PROBLEM" in probe[-1]["content"]


def test_evaluator_installs_only_turn_adapter_and_restores_it(tmp_path, monkeypatch):
    """The real entrypoint leaves the baseline resolver untouched and restores hooks."""
    from examples.tutor.scripts.student_sim_eval import evaluate as base
    from examples.tutor.scripts.student_sim_eval.prompted_strict import evaluate
    from examples.tutor.workflow import TutorAgentWorkflow

    _, (_, protocol) = compiled()
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    (tmp_path / "question_ids.json").write_text('["test-1"]')
    monkeypatch.setattr("sys.argv", ["evaluate", "--config", str(path)])
    resolver = TutorAgentWorkflow._free_chat_student_prompts
    builder = TutorAgentWorkflow._build_student_messages
    teacher = TutorAgentWorkflow._build_tutor_messages
    called = []

    def fake_main():
        assert TutorAgentWorkflow._free_chat_student_prompts is resolver
        assert TutorAgentWorkflow._build_student_messages is not builder
        assert TutorAgentWorkflow._build_tutor_messages is not teacher
        called.append(True)

    monkeypatch.setattr(base.evaluate_teacher_api, "main", fake_main)
    evaluate.main()
    assert called == [True]
    assert TutorAgentWorkflow._free_chat_student_prompts is resolver
    assert TutorAgentWorkflow._build_student_messages is builder
    assert TutorAgentWorkflow._build_tutor_messages is teacher


@pytest.mark.parametrize("field", ["gate", "axis"])
def test_student_adapter_rejects_external_gate(field):
    """A misconfigured setting fails before any student request."""
    _, (_, protocol) = compiled()
    if field == "gate":
        protocol["personality"]["gate_sample_rate"] = 1
    else:
        protocol["student_axes"][0]["personalities"] = ["independent-verification"]
    with pytest.raises(ValueError, match="External preference gating"):
        with_student_instruction(lambda self, state: [], protocol)
