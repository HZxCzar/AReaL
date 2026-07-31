from __future__ import annotations

import asyncio
import json
import random
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    TutorConfig,
    TutorStudentTurnBehaviorConfig,
    TutorTeacherWarmupPromptConfig,
)
from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PromptPoolSelection,
    PublicHistoryState,
    StudentTurnBehavior,
    StudentTurnState,
    TurnArtifact,
    TutorPrivateFeedback,
    TutorTurnState,
)


def _judge(correct: bool) -> JudgeResult:
    return JudgeResult(
        raw_output="",
        correct=correct,
        feedback="ok" if correct else "incorrect",
        parse_error=None,
        raw_result={},
    )


def _leak(leaked: bool = False) -> LeakCheckResult:
    return LeakCheckResult(
        raw_output="",
        leaked=leaked,
        feedback="leaked" if leaked else "ok",
        parse_error=None,
        raw_result={},
    )


class _RecordingTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        rendered += f"|thinking:{enable_thinking}"
        return list(rendered.encode("utf-8"))


def test_load_prompt_pool_validates_json_string_array(tmp_path):
    """Test prompt pools are stripped, non-empty, and unique."""
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps([" first ", "second"]), encoding="utf-8")

    assert tutor_workflow.load_prompt_pool(str(valid_path), role="teacher") == (
        "first",
        "second",
    )
    assert tutor_workflow.load_prompt_pool("", role="teacher") == ()

    invalid_payloads = [[], {}, [""], ["same", "same"], ["valid", 1]]
    for index, payload in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="prompt pool"):
            tutor_workflow.load_prompt_pool(str(path), role="student")


def test_load_prompt_pool_reports_missing_and_malformed_files(tmp_path):
    """Test file errors identify the affected prompt pool."""
    with pytest.raises(ValueError, match="teacher prompt pool file not found"):
        tutor_workflow.load_prompt_pool(str(tmp_path / "missing.json"), role="teacher")

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="student prompt pool must be valid JSON"):
        tutor_workflow.load_prompt_pool(str(malformed_path), role="student")


def test_load_student_turn_behaviors_validates_weighted_pool(tmp_path):
    """Test turn behavior assets preserve names, prompts, and exact probabilities."""
    path = tmp_path / "turn-behaviors.json"
    path.write_text(
        json.dumps(
            [
                {"name": "base", "probability": 0.5, "instruction": ""},
                {
                    "name": "ask_question",
                    "probability": 0.3,
                    "instruction": "Ask one focused question.",
                },
                {
                    "name": "show_work",
                    "probability": 0.2,
                    "instruction": "Show one concrete step.",
                },
            ]
        ),
        encoding="utf-8",
    )

    behaviors = tutor_workflow.load_student_turn_behaviors(str(path))

    assert [behavior.name for behavior in behaviors] == [
        "base",
        "ask_question",
        "show_work",
    ]
    assert [behavior.probability for behavior in behaviors] == [0.5, 0.3, 0.2]
    assert behaviors[0].instruction == ""


@pytest.mark.parametrize(
    "payload",
    [
        [
            {"name": "base", "probability": 0.4, "instruction": ""},
            {"name": "ask", "probability": 0.4, "instruction": "Ask."},
        ],
        [
            {"name": "base", "probability": 0.5, "instruction": ""},
            {"name": "also_base", "probability": 0.5, "instruction": ""},
        ],
        [
            {"name": "base", "probability": 0.5, "instruction": ""},
            {"name": "base", "probability": 0.5, "instruction": "Ask."},
        ],
    ],
)
def test_load_student_turn_behaviors_rejects_invalid_distribution(tmp_path, payload):
    """Test malformed probability, base, and name definitions fail early."""
    path = tmp_path / "invalid-turn-behaviors.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="student turn behavior"):
        tutor_workflow.load_student_turn_behaviors(str(path))


def test_student_turn_behavior_config_requires_path_when_enabled():
    """Test the config switch cannot silently enable an empty behavior pool."""
    with pytest.raises(ValueError, match="student_turn_behavior.path"):
        TutorStudentTurnBehaviorConfig(enabled=True)


def test_student_turn_behavior_separate_calls_are_explicit_and_unique():
    """Test separate calls are opt-in and do not replace inline instructions."""
    assert TutorStudentTurnBehaviorConfig().separate_call_behavior_names == []
    with pytest.raises(ValueError, match="requires student turn behaviors"):
        TutorStudentTurnBehaviorConfig(
            separate_call_behavior_names=["ask_question"]
        )
    with pytest.raises(ValueError, match="must be unique"):
        TutorStudentTurnBehaviorConfig(
            enabled=True,
            path="behaviors.json",
            separate_call_behavior_names=["ask_question", "ask_question"],
        )


def test_student_turn_behavior_sampling_is_per_response_and_eval_disabled(
    monkeypatch,
):
    """Test each response has its own deterministic draw and eval has no draw."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.prompt_pool_seed = 42
    workflow.student_turn_behavior_enabled = True
    workflow.student_turn_behaviors = (
        StudentTurnBehavior(0, "base", "", 0.5),
        StudentTurnBehavior(1, "ask", "Ask one question.", 0.5),
    )
    workflow._student_turn_behavior_fallback_rng = random.Random(3)
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=17),
    )

    first = workflow._select_student_turn_behavior(response_index=0)
    repeated = workflow._select_student_turn_behavior(response_index=0)
    second = workflow._select_student_turn_behavior(response_index=1)

    assert first == repeated
    assert first in workflow.student_turn_behaviors
    assert second in workflow.student_turn_behaviors
    expected_draws = [
        random.Random(f"42:student-turn-behavior:17:{index}").random()
        for index in (0, 1)
    ]
    expected_indices = [int(draw >= 0.5) for draw in expected_draws]
    assert [first.index, second.index] == expected_indices

    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=17),
    )
    assert workflow._select_student_turn_behavior(response_index=0) is None


def test_student_turn_behavior_only_changes_current_user_prompt():
    """Test a non-request behavior still conditions the normal student response."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_system_prompt = "base student prompt"
    workflow.student_turn_behavior_separate_call_behavior_names = frozenset(
        {"ask_question"}
    )
    state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="hint",
        student_turn_behavior=StudentTurnBehavior(
            1,
            "show_work",
            "Show one concrete mathematical step.",
            0.2,
        ),
    )

    conditioned_system = workflow._student_system_prompt_for_state(state)
    conditioned_user = workflow._build_student_prompt_from_state(state)
    state.student_turn_behavior = StudentTurnBehavior(0, "base", "", 0.5)
    clean_system = workflow._student_system_prompt_for_state(state)
    clean_user = workflow._build_student_prompt_from_state(state)

    assert conditioned_system == "base student prompt"
    assert clean_system == "base student prompt"
    assert conditioned_user.endswith("Show one concrete mathematical step.")
    assert "Show one concrete mathematical step." not in clean_user


def test_inline_student_instruction_remains_available_for_any_behavior_name():
    """Test behavior names stay inline unless the config explicitly separates them."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_turn_behavior_separate_call_behavior_names = frozenset()
    instruction = "Ask one short question at the end of this response."
    state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="hint",
        student_turn_behavior=StudentTurnBehavior(
            1,
            "ask_question",
            instruction,
            0.2,
        ),
    )

    assert workflow._build_student_prompt_from_state(state).endswith(instruction)


def test_student_request_behavior_uses_separate_question_call():
    """Test a request is absent from the normal prompt and appended by a second call."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_system_prompt = "base student prompt"
    workflow.student_turn_behavior_separate_call_behavior_names = frozenset(
        {"ask_question"}
    )
    instruction = (
        "Ask your teacher one specific question about a mathematical step you do not "
        "understand."
    )
    behavior = StudentTurnBehavior(1, "ask_question", instruction, 0.2)
    state = StudentTurnState(
        task="Find x.",
        public_history=PublicHistoryState(summary="Tutor round 1: Factor first."),
        previous_student_output="I tried expanding.",
        latest_tutor_visible_output="Look for a common factor.",
        student_prompt_selection=PromptPoolSelection(
            index=0,
            suffix="You prefer short questions.",
        ),
        student_turn_behavior=behavior,
    )
    captured = {}

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            text="Why does factoring help isolate x?",
            raw_text="Why does factoring help isolate x?",
            error=None,
        )

    workflow._call_auxiliary_prompt = call_auxiliary_prompt

    normal_prompt = workflow._build_student_prompt_from_state(state)
    combined, effective_behavior, generation = asyncio.run(
        workflow._maybe_append_student_question(
            state,
            "I expanded both sides but still got the wrong value.",
            should_generate=True,
            aux_caller=object(),
        )
    )

    assert instruction not in normal_prompt
    assert captured["system_prompt"].endswith("You prefer short questions.")
    assert instruction in captured["user_prompt"]
    assert "You are a student learning math" in captured["system_prompt"]
    assert "Your conversation with the teacher" in captured["user_prompt"]
    assert "Your teacher's latest message" in captured["user_prompt"]
    assert "Your latest response" in captured["user_prompt"]
    assert "Look for a common factor." in captured["user_prompt"]
    assert "I expanded both sides but still got the wrong value." in captured[
        "user_prompt"
    ]
    assert combined.endswith("Why does factoring help isolate x?")
    assert effective_behavior is behavior
    assert generation is not None
    assert generation.question == "Why does factoring help isolate x?"
    assert generation.error is None


def test_student_request_behavior_without_following_teacher_skips_question_call():
    """Test a sampled request is dropped when no later teacher reply can answer it."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_turn_behavior_separate_call_behavior_names = frozenset(
        {"ask_question"}
    )
    behavior = StudentTurnBehavior(1, "ask_question", "Ask one question.", 0.2)
    state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="hint",
        student_turn_behavior=behavior,
    )

    async def unexpected_call(**_kwargs):
        raise AssertionError("question model call should be skipped")

    workflow._call_auxiliary_prompt = unexpected_call

    combined, effective_behavior, generation = asyncio.run(
        workflow._maybe_append_student_question(
            state,
            "normal response",
            should_generate=False,
            aux_caller=object(),
        )
    )

    assert combined == "normal response"
    assert effective_behavior is None
    assert generation is None


def test_student_question_call_failure_does_not_target_next_teacher():
    """Test a failed dedicated question call keeps the normal response unmodified."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_turn_behavior_separate_call_behavior_names = frozenset(
        {"ask_question"}
    )
    behavior = StudentTurnBehavior(1, "ask_question", "Ask one question.", 0.2)
    state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="hint",
        student_turn_behavior=behavior,
    )

    async def failed_call(**_kwargs):
        return types.SimpleNamespace(
            text="",
            raw_text="",
            error="student endpoint unavailable",
        )

    workflow._call_auxiliary_prompt = failed_call

    combined, effective_behavior, generation = asyncio.run(
        workflow._maybe_append_student_question(
            state,
            "normal response",
            should_generate=True,
            aux_caller=object(),
        )
    )

    assert combined == "normal response"
    assert effective_behavior is None
    assert generation is not None
    assert generation.error == "student endpoint unavailable"


def test_question_overlay_explicitly_enables_separate_student_call():
    """Test the question experiment opts in without changing the base config."""
    path = Path(
        "examples/tutor/configs/math/0723/2gpu/"
        "qwen8b-train-qwen1.7b-eval3-math-pre-aleak-question.yaml"
    )

    overlay = OmegaConf.load(path)

    assert overlay.prompt_pool.student_turn_behavior.enabled is True
    assert overlay.prompt_pool.student_turn_behavior.separate_call_behavior_names == [
        "ask_question"
    ]
    assert overlay.reward.student_request_judge.behavior_names == ["ask_question"]


def test_teacher_warmup_config_requires_prompt_and_positive_steps():
    """Test enabled warm-up configs reject incomplete schedules."""
    with pytest.raises(ValueError, match="prompt_path"):
        TutorTeacherWarmupPromptConfig(enabled=True, steps=50)
    with pytest.raises(ValueError, match="steps"):
        TutorTeacherWarmupPromptConfig(enabled=True, prompt_path="prompt.txt")

    config = TutorTeacherWarmupPromptConfig(
        enabled=True,
        prompt_path=" prompt.txt ",
        steps=50,
    )

    assert config.prompt_path == "prompt.txt"
    assert config.steps == 50


def test_load_prompt_text_reports_missing_and_empty_files(tmp_path):
    """Test full prompt assets must exist and contain visible instructions."""
    with pytest.raises(ValueError, match="prompt file not found"):
        tutor_workflow.load_prompt_text(
            str(tmp_path / "missing.txt"), role="teacher warm-up"
        )

    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be non-empty"):
        tutor_workflow.load_prompt_text(str(empty_path), role="teacher warm-up")


@pytest.mark.parametrize(
    ("version", "expected"),
    [(0, 1.0), (25, 0.5), (49, 0.02), (50, 0.0), (51, 0.0)],
)
def test_teacher_warmup_probability_linearly_reaches_zero(version, expected):
    """Test warm-up probability follows the pinned policy version."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_warmup_enabled = True
    workflow.teacher_warmup_steps = 50

    assert workflow._teacher_warmup_probability(version) == pytest.approx(expected)

    workflow.teacher_warmup_enabled = False
    assert workflow._teacher_warmup_probability(version) == pytest.approx(0.0)


def test_prompt_pool_sampling_is_task_deterministic_and_eval_disabled(monkeypatch):
    """Test training task IDs deterministically select an episode-local suffix."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.prompt_pool_seed = 42
    workflow._teacher_prompt_pool_fallback_rng = random.Random(1)
    workflow._student_prompt_pool_fallback_rng = random.Random(2)
    pool = ("one", "two", "three")

    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=17),
    )
    first = workflow._sample_prompt_pool(pool, role="teacher")
    second = workflow._sample_prompt_pool(pool, role="teacher")
    student = workflow._sample_prompt_pool(pool, role="student")

    assert first == second
    assert first is not None
    assert first.suffix == pool[first.index]
    assert student is not None
    assert student.suffix == pool[student.index]

    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=17),
    )
    assert workflow._sample_prompt_pool(pool, role="teacher") is None
    assert workflow._sample_prompt_pool(pool, role="student") is None


def test_eval_student_prompt_selection_honors_forced_index(monkeypatch):
    """Test evaluation rows pin a persona independently of rollout task IDs."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ("persona zero", "persona one", "persona two")
    context = types.SimpleNamespace(is_eval=True, task_id=17)
    monkeypatch.setattr(tutor_workflow.workflow_context, "get", lambda: context)
    data = {TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 1}

    assert workflow._select_student_prompt({}) is None
    first = workflow._select_student_prompt(data)
    context.task_id = 99
    second = workflow._select_student_prompt(data)

    assert (
        first
        == second
        == PromptPoolSelection(index=1, suffix="persona one", pool="seen")
    )


def test_eval_student_prompt_selection_uses_heldout_pool(monkeypatch):
    """Test held-out evaluation never falls back to the seen training pool."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ("seen zero",)
    workflow.student_heldout_prompt_pool = ("heldout zero", "heldout one")
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=17),
    )

    selection = workflow._select_student_prompt(
        {
            TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "heldout",
            TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 1,
        }
    )

    assert selection == PromptPoolSelection(
        index=1, suffix="heldout one", pool="heldout"
    )


def test_training_student_prompt_selection_ignores_eval_index(monkeypatch):
    """Test training keeps deterministic task sampling despite reserved input."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ("zero", "one", "two")
    workflow.student_prompt_include_base = False
    workflow.prompt_pool_seed = 42
    workflow._student_prompt_pool_fallback_rng = random.Random(2)
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=17),
    )

    selection = workflow._select_student_prompt(
        {TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 0}
    )
    expected_index = random.Random("42:student:17").randrange(3)

    assert selection == PromptPoolSelection(
        index=expected_index,
        suffix=workflow.student_prompt_pool[expected_index],
        pool="seen",
    )


def test_training_student_prompt_selection_respects_include_base(monkeypatch):
    """Test the base prompt is an equal training option only when enabled."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ("zero", "one", "two")
    workflow.prompt_pool_seed = 42

    class ChooseLast:
        def randrange(self, stop):
            return stop - 1

    workflow._student_prompt_pool_fallback_rng = ChooseLast()
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=None),
    )

    workflow.student_prompt_include_base = True
    assert workflow._select_student_prompt({}) is None

    workflow.student_prompt_include_base = False
    assert workflow._select_student_prompt({}) == PromptPoolSelection(
        index=2,
        suffix="two",
        pool="seen",
    )


@pytest.mark.parametrize("prompt_index", [True, 1.0, "1", -1, 3])
def test_eval_student_prompt_selection_rejects_invalid_index(monkeypatch, prompt_index):
    """Test forced persona indices must be valid integer pool positions."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ("zero", "one", "two")
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=17),
    )

    with pytest.raises(ValueError, match="student prompt index"):
        workflow._select_student_prompt(
            {TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: prompt_index}
        )


def test_eval_student_prompt_selection_requires_loaded_pool(monkeypatch):
    """Test forced evaluation cannot silently fall back to the base prompt."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_prompt_pool = ()
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=17),
    )

    with pytest.raises(ValueError, match="non-empty seen student prompt pool"):
        workflow._select_student_prompt({TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD: 0})


def test_teacher_warmup_selection_overrides_then_hands_off_to_pool(monkeypatch):
    """Test version zero uses the full prompt and the cutoff uses the pool."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.prompt_pool_seed = 42
    workflow.teacher_warmup_enabled = True
    workflow.teacher_warmup_steps = 50
    workflow.teacher_warmup_prompt_path = "warmup.txt"
    workflow._teacher_warmup_fallback_rng = random.Random(1)
    workflow._teacher_prompt_pool_fallback_rng = random.Random(2)
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=7),
    )
    pool = ("pool A", "pool B")

    warmup = workflow._select_teacher_prompt(pool, rollout_version=0)
    after_cutoff = workflow._select_teacher_prompt(pool, rollout_version=50)
    expected_index = random.Random("42:teacher:7").randrange(len(pool) + 1)

    assert warmup is not None
    assert warmup.source == "warmup_full"
    assert warmup.rollout_version == 0
    assert warmup.warmup_probability == pytest.approx(1.0)
    assert warmup.prompt_path == "warmup.txt"
    assert after_cutoff is not None
    if expected_index == len(pool):
        assert after_cutoff.source == "pool_base"
        assert after_cutoff.index == -1
        assert after_cutoff.suffix == ""
    else:
        assert after_cutoff.source == "pool"
        assert after_cutoff.index == expected_index
        assert after_cutoff.suffix == pool[expected_index]
    assert after_cutoff.warmup_probability == pytest.approx(0.0)


def test_teacher_pool_includes_clean_base_as_equal_implicit_item(monkeypatch):
    """Test the clean teacher prompt is a virtual item beside JSON suffixes."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.prompt_pool_seed = 42
    workflow.teacher_warmup_enabled = False
    workflow.teacher_system_prompt = "clean base teacher"
    workflow._teacher_prompt_pool_fallback_rng = random.Random(2)
    pool = ("pool A", "pool B")
    base_task_id = next(
        task_id
        for task_id in range(100)
        if random.Random(f"42:teacher:{task_id}").randrange(len(pool) + 1) == len(pool)
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False, task_id=base_task_id),
    )

    selection = workflow._select_teacher_prompt(pool, rollout_version=50)

    assert selection is not None
    assert selection.source == "pool_base"
    assert selection.index == -1
    assert selection.suffix == ""
    assert workflow._teacher_system_prompt_for_selection(selection) == (
        workflow.teacher_system_prompt
    )


def test_teacher_warmup_selection_is_disabled_during_eval(monkeypatch):
    """Test evaluation always uses the clean base prompt."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_warmup_enabled = True
    workflow.teacher_warmup_steps = 50
    workflow.prompt_pool_seed = 42
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=7),
    )

    assert workflow._select_teacher_prompt(("pool",), rollout_version=0) is None


def test_warmup_full_prompt_replaces_instead_of_appending_clean_base():
    """Test the historical full prompt is the sole rollout system message."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_system_prompt = "clean base"
    workflow.teacher_warmup_prompt = "historical full prompt"
    selection = PromptPoolSelection(
        index=-1,
        suffix="",
        source="warmup_full",
        rollout_version=0,
        warmup_probability=1.0,
    )

    assert (
        workflow._teacher_system_prompt_for_selection(selection)
        == "historical full prompt"
    )
    assert "clean base" not in workflow._teacher_system_prompt_for_selection(selection)


def test_short_warmup_prompt_reserves_clean_training_input_budget():
    """Test a shorter rollout prompt still budgets for the clean train prompt."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.tokenizer = _RecordingTokenizer()
    workflow.enable_thinking = True
    workflow.max_train_sample_tokens = 256
    workflow.teacher_system_prompt = "a substantially longer clean teacher prompt"
    workflow.teacher_warmup_prompt = "short"
    workflow._build_tutor_prompt = lambda _state: "teacher user prompt"
    captured = {}

    class ActorCaller:
        async def generate(self, _messages, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(response=object(), raw_text="hint")

    state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState(),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=1,
        max_turns=1,
        teacher_prompt_selection=PromptPoolSelection(
            index=-1,
            suffix="",
            source="warmup_full",
        ),
    )

    asyncio.run(workflow._generate_tutor_response(state, actor_caller=ActorCaller()))

    assert captured["input_token_reserve"] > 0


def test_rollout_stats_report_warmup_source_and_prompt_token_delta(monkeypatch):
    """Test warm-up observability exposes schedule and selected-source outcomes."""
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    selection = PromptPoolSelection(
        index=-1,
        suffix="",
        source="warmup_full",
        rollout_version=25,
        warmup_probability=0.5,
        prompt_path="warmup.txt",
    )

    workflow._log_rollout_stats(
        total_reward=-1.0,
        traces=[],
        termination_reason="leak",
        pre_success=False,
        leak_count=1,
        teacher_prompt_selection=selection,
        inference_prompt_tokens=1200,
        training_prompt_tokens=200,
    )

    assert captured["prompt_schedule/warmup_probability"] == pytest.approx(0.5)
    assert captured["prompt_schedule/rollout_version"] == pytest.approx(25.0)
    assert captured["prompt_schedule/warmup_selected"] == pytest.approx(1.0)
    assert captured["prompt_schedule/prompt_token_delta"] == pytest.approx(1000.0)
    assert captured["prompt_source/warmup_full/selected"] == pytest.approx(1.0)
    assert captured["prompt_source/warmup_full/leaked"] == pytest.approx(1.0)
    assert captured["prompt_source/warmup_full/reward"] == pytest.approx(-1.0)
    assert captured["prompt_source/pool/selected"] == pytest.approx(0.0)
    assert captured["prompt_source/pool_base/selected"] == pytest.approx(0.0)


def test_rollout_stats_report_only_grouped_student_prompt_metrics(monkeypatch):
    """Test persona metrics include their pool and omit legacy flat aliases."""
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.student_prompt_pool = ("persona zero", "persona one", "persona two")

    workflow._log_rollout_stats(
        total_reward=1.5,
        traces=[],
        termination_reason="pre_solved",
        pre_success=True,
        leak_count=0,
        student_prompt_selection=PromptPoolSelection(
            index=1,
            suffix="persona one",
        ),
    )

    assert captured["student_prompt/seen/0/selected"] == pytest.approx(0.0)
    assert captured["student_prompt/seen/1/selected"] == pytest.approx(1.0)
    assert captured["student_prompt/seen/2/selected"] == pytest.approx(0.0)
    assert captured["student_prompt/seen/1/solved"] == pytest.approx(0.0)
    assert captured["student_prompt/seen/1/pre_solved"] == pytest.approx(1.0)
    assert captured["student_prompt/seen/1/reward"] == pytest.approx(1.5)
    assert captured["student_prompt/seen/1/turns"] == pytest.approx(0.0)
    assert captured["student_prompt/seen/1/call_failed"] == pytest.approx(0.0)
    flat_prefixes = tuple(f"student_prompt/{index}/" for index in range(3))
    assert not any(key.startswith(flat_prefixes) for key in captured)


def test_rollout_stats_report_initial_student_turn_behavior(monkeypatch):
    """Test a pre-solved response still contributes to behavior sampling metrics."""
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.student_turn_behaviors = (
        StudentTurnBehavior(0, "base", "", 0.5),
        StudentTurnBehavior(1, "ask", "Ask.", 0.5),
    )

    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="pre_solved",
        pre_success=True,
        leak_count=0,
        initial_student_turn_behavior=workflow.student_turn_behaviors[1],
    )

    assert captured["student_turn_behavior/total_responses"] == pytest.approx(1.0)
    assert captured["student_turn_behavior/base/selected_count"] == pytest.approx(0.0)
    assert captured["student_turn_behavior/ask/selected_count"] == pytest.approx(1.0)
    assert captured["student_turn_behavior/ask/selected_fraction"] == pytest.approx(1.0)


def test_eval_rollout_stats_record_core_repeat_metrics(monkeypatch):
    """Test evaluation exposes final correctness and core repeat stability."""
    captured = []
    monkeypatch.setattr(
        tutor_workflow, "_safe_scalar", lambda **metrics: captured.append(metrics)
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=7),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.eval_repeat_count = 1
    workflow._eval_repeat_outcomes = {}

    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="pre_solved",
        pre_success=True,
        leak_count=0,
    )

    merged = {key: value for metrics in captured for key, value in metrics.items()}
    assert merged["solved"] == pytest.approx(0.0)
    assert merged["final_correct"] == pytest.approx(1.0)
    assert merged["repeat/final_correct/mean_task_sample_variance"] == pytest.approx(
        0.0
    )
    assert merged["repeat/final_correct/pairwise_success_jaccard"] == pytest.approx(1.0)
    assert not any(key.startswith("repeat/solved/") for key in merged)


def test_forced_persona_eval_does_not_record_repeat_outcomes(monkeypatch):
    """Test persona-only rows do not contaminate comparable repeat metrics."""
    captured = []
    monkeypatch.setattr(
        tutor_workflow, "_safe_scalar", lambda **metrics: captured.append(metrics)
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True, task_id=7),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.eval_repeat_count = 1
    workflow._eval_repeat_outcomes = {}
    selection = tutor_workflow.PromptPoolSelection(
        index=0,
        suffix="persona",
        source="pool",
        pool="seen",
    )

    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="pre_solved",
        pre_success=True,
        leak_count=0,
        student_prompt_selection=selection,
    )

    assert not any(key.startswith("repeat/") for metrics in captured for key in metrics)


def test_eval_student_summary_uses_only_base_prompt_rows(monkeypatch):
    """Test the comparable student score excludes forced persona rollouts."""
    captured = []
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.append(metrics),
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=True),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.student_prompt_pool = ("persona zero", "persona one")

    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        student_name="qwen3-1.7b",
    )
    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        student_name="qwen3-1.7b",
        student_prompt_selection=PromptPoolSelection(
            index=1,
            suffix="persona one",
            pool="seen",
        ),
    )

    base_metrics, persona_metrics = captured
    assert base_metrics["solved"] == pytest.approx(0.0)
    assert base_metrics["student/qwen3-1.7b/solved"] == pytest.approx(0.0)
    assert "solved" not in persona_metrics
    assert "reward" not in persona_metrics
    assert "student/qwen3-1.7b/solved" not in persona_metrics
    assert persona_metrics["student_prompt/seen/1/solved"] == pytest.approx(0.0)


def test_training_persona_keeps_student_summary_metrics(monkeypatch):
    """Test persona sampling does not remove the existing training summary."""
    captured = {}
    monkeypatch.setattr(
        tutor_workflow,
        "_safe_scalar",
        lambda **metrics: captured.update(metrics),
    )
    monkeypatch.setattr(
        tutor_workflow.workflow_context,
        "get",
        lambda: types.SimpleNamespace(is_eval=False),
    )
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_model_runtimes = {}
    workflow.student_prompt_pool = ("persona zero",)

    workflow._log_rollout_stats(
        total_reward=0.0,
        traces=[],
        termination_reason="max_turns",
        pre_success=False,
        leak_count=0,
        student_name="qwen3-1.7b",
        student_prompt_selection=PromptPoolSelection(
            index=0,
            suffix="persona zero",
            pool="seen",
        ),
    )

    assert captured["solved"] == pytest.approx(0.0)
    assert captured["student/qwen3-1.7b/solved"] == pytest.approx(0.0)
    assert captured["student_prompt/seen/0/solved"] == pytest.approx(0.0)


def test_teacher_rollout_uses_suffix_but_training_tensor_uses_clean_prompt():
    """Test teacher strategy conditioning is removed only at the tensor boundary."""
    tokenizer = _RecordingTokenizer()
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_system_prompt = "base teacher"
    workflow.teacher_user_prompt_template = "unused"
    workflow.teacher_pre_enabled = False
    workflow.teacher_show_ground_truth = False
    workflow.enable_thinking = True
    workflow.tokenizer = tokenizer
    workflow._build_tutor_prompt = lambda _state: "teacher user prompt"
    selection = PromptPoolSelection(index=1, suffix="use a Socratic question")
    state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState(),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=1,
        max_turns=2,
        teacher_prompt_selection=selection,
    )

    rollout_messages = workflow._build_tutor_messages(state)
    rollout_input = tokenizer.apply_chat_template(
        rollout_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    response = types.SimpleNamespace(
        input_tokens=rollout_input,
        output_tokens=[901, 902],
        output_logprobs=[-0.1, -0.2],
        output_versions=[4, 4],
        stop_reason="stop",
        input_len=len(rollout_input),
        output_len=2,
        tokenizer=tokenizer,
    )
    artifact = TurnArtifact(
        turn_idx=1,
        tutor_state=state,
        tutor_prompt="teacher user prompt",
        tutor_response=response,
        tutor_raw_output="hint",
        tutor_visible_output="hint",
        leak_result=_leak(),
        public_history_before="",
        public_history_after="",
    )

    clean_input = workflow._clean_tutor_input_tokens(artifact)
    tensor_dict = response_to_tensordict(
        response,
        reward=1.0,
        input_tokens_override=clean_input,
    )

    assert selection.suffix in rollout_messages[0]["content"]
    assert clean_input != rollout_input
    assert tensor_dict["input_ids"].tolist()[0] == clean_input + [901, 902]
    assert tensor_dict["loss_mask"].tolist()[0] == [0] * len(clean_input) + [1, 1]
    assert tensor_dict["logprobs"].tolist()[0][-2:] == pytest.approx([-0.1, -0.2])
    assert tensor_dict["versions"].tolist()[0][-2:] == [4, 4]

    workflow.teacher_warmup_prompt = "historical full prompt"
    state.teacher_prompt_selection = PromptPoolSelection(
        index=-1,
        suffix="",
        source="warmup_full",
        rollout_version=0,
        warmup_probability=1.0,
    )
    warmup_messages = workflow._build_tutor_messages(state)
    response.input_tokens = tokenizer.apply_chat_template(
        warmup_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    response.input_len = len(response.input_tokens)

    warmup_clean_input = workflow._clean_tutor_input_tokens(artifact)

    assert warmup_messages[0]["content"] == "historical full prompt"
    assert warmup_clean_input == clean_input


def test_student_calls_use_selected_behavior_suffix():
    """Test a student selection augments the system prompt for each student call."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_system_prompt = "base student"
    workflow._build_student_prompt_from_state = lambda _state: "student user prompt"
    captured = {}

    async def call_auxiliary_prompt(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(text="attempt", error=None)

    workflow._call_auxiliary_prompt = call_auxiliary_prompt
    selection = PromptPoolSelection(index=0, suffix="verify every step")
    state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="hint",
        student_prompt_selection=selection,
    )

    output, error = asyncio.run(workflow._run_student(state, aux_caller=object()))

    assert output == "attempt"
    assert error is None
    assert captured["system_prompt"] == "base student\n\nverify every step"


def test_concurrent_teacher_calls_keep_episode_local_suffixes():
    """Test concurrent calls never read a mutable workflow-level selection."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_system_prompt = "base teacher"
    workflow._build_tutor_prompt = lambda state: f"turn {state.turn_idx}"
    captured_system_prompts = []

    class ActorCaller:
        async def generate(self, messages, **_kwargs):
            await asyncio.sleep(0)
            captured_system_prompts.append(messages[0]["content"])
            return types.SimpleNamespace(response=object(), raw_text="hint")

    def state(selection):
        return TutorTurnState(
            task="task",
            ground_truth="42",
            public_history=PublicHistoryState(),
            previous_tutor_visible_output="",
            previous_feedback=TutorPrivateFeedback(),
            turn_idx=1,
            max_turns=1,
            teacher_prompt_selection=selection,
        )

    async def run_calls():
        await asyncio.gather(
            workflow._generate_tutor_response(
                state(PromptPoolSelection(index=0, suffix="strategy A")),
                actor_caller=ActorCaller(),
            ),
            workflow._generate_tutor_response(
                state(PromptPoolSelection(index=1, suffix="strategy B")),
                actor_caller=ActorCaller(),
            ),
        )

    asyncio.run(run_calls())

    assert sorted(captured_system_prompts) == [
        "base teacher\n\nstrategy A",
        "base teacher\n\nstrategy B",
    ]


def test_student_generalization_reuses_episode_behavior_suffix():
    """Test transfer probes retain the student behavior sampled for the episode."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.student_generalize_enabled = True
    workflow.student_generalize_mode = "only_success"
    workflow.student_generalize_confidence_enabled = False
    workflow.student_generalize_level_rewards = {"level1": 0.2, "level2": 0.5}
    workflow.student_generalize_bank = {}
    workflow.student_system_prompt = "base student"
    selection = PromptPoolSelection(index=3, suffix="check a small example")
    tutor_state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState(),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=1,
        max_turns=1,
    )
    turn = TurnArtifact(
        turn_idx=1,
        tutor_state=tutor_state,
        tutor_prompt="prompt",
        tutor_response=None,
        tutor_raw_output="hint",
        tutor_visible_output="hint",
        leak_result=_leak(),
        public_history_before="",
        public_history_after="learned context",
        student_state=StudentTurnState(
            task="task",
            public_history=PublicHistoryState(),
            previous_student_output="wrong",
            latest_tutor_visible_output="hint",
            student_prompt_selection=selection,
        ),
        student_output="correct answer",
        judge_result=_judge(True),
    )
    episode = EpisodeArtifact(
        task="task",
        ground_truth="42",
        initial_student_answer="wrong",
        initial_student_error=None,
        initial_judge_result=_judge(False),
        turns=[turn],
        termination_reason="success",
        pre_success=False,
        leak_count=0,
        latest_student_answer="correct answer",
        student_prompt_selection=selection,
    )
    calls = []

    async def call_auxiliary_prompt(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(
            text="transfer answer", error=None, token_logprobs=()
        )

    async def score_answer(*_args, **_kwargs):
        return _judge(False)

    workflow._call_auxiliary_prompt = call_auxiliary_prompt
    workflow._score_answer_async = score_answer

    results = asyncio.run(
        workflow._run_student_generalization(
            {
                "metadata": {
                    "student_generalize": {
                        "level1": {"task": "near task", "ground_truth": "43"},
                        "level2": {"task": "far task", "ground_truth": "44"},
                    }
                }
            },
            episode,
            aux_caller=object(),
            answer_judge_caller=None,
        )
    )

    assert len(results) == 2
    assert [call["system_prompt"] for call in calls] == [
        "base student\n\ncheck a small example",
        "base student\n\ncheck a small example",
    ]


def test_prompt_pool_example_yaml_loads_typed_config(monkeypatch):
    """Test the checked-in prompt-pool experiment resolves both JSON paths."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/"
        "baseline-overfit-1-leakt-batch128-rebn-nomean-5-prompt-pools.yaml"
    )

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    assert isinstance(config, TutorConfig)
    assert config.prompt_pool.teacher_path.endswith("teacher_strategy_suffixes.json")
    assert config.prompt_pool.student_seen_path.endswith(
        "student_behavior_suffixes.json"
    )
    assert tutor_workflow.load_prompt_pool(
        config.prompt_pool.teacher_path, role="teacher"
    )
    assert tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_seen_path, role="student"
    )


@pytest.mark.parametrize(
    ("config_name", "baseline_name"),
    [
        (
            "qwen8b-qwen1.7b-math-student5.yaml",
            "qwen8b-qwen1.7b-math-baseline.yaml",
        ),
        (
            "qwen8b-qwen1.7b-math-student5-pre.yaml",
            "qwen8b-qwen1.7b-math-pre.yaml",
        ),
    ],
)
def test_pass2_student_personality_pool_config_loads_five_personas(
    monkeypatch, config_name, baseline_name
):
    """Test the pass@2 pilot changes only the five-way student persona pool."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path("examples/tutor/configs/math/july/pass@2") / config_name
    baseline_path = path.with_name(baseline_name)

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )
    student_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_seen_path, role="student"
    )
    pilot_source = OmegaConf.load(path)
    baseline_source = OmegaConf.load(baseline_path)
    assert pilot_source.prompt_pool.include_base is True
    pilot_source.pop("trial_name")
    pilot_source.pop("prompt_pool")
    baseline_source.pop("trial_name")

    assert isinstance(config, TutorConfig)
    assert config.prompt_pool.teacher_path == ""
    assert config.prompt_pool.student_seen_path.endswith(
        "student_personality_suffixes_5.json"
    )
    assert config.prompt_pool.test_persona is True
    assert config.prompt_pool.student_eval_paths == {
        "seen": config.prompt_pool.student_seen_path
    }
    assert len(student_prompts) == 5
    assert all(prompt.startswith("[Persona: ") for prompt in student_prompts)
    assert tuple(
        prompt.split("]", 1)[0].removeprefix("[Persona: ") for prompt in student_prompts
    ) == (
        "QuickHelpSeeker",
        "IndependentPerseverer",
        "ReceptiveFollower",
        "SkepticalDefender",
        "AdaptiveExplorer",
    )
    assert all(
        prompt.endswith(
            "Do not mention the persona label or these behavior instructions."
        )
        for prompt in student_prompts
    )
    assert OmegaConf.to_container(
        pilot_source, resolve=False
    ) == OmegaConf.to_container(baseline_source, resolve=False)


def test_pass2_heldout_persona_config_separates_train_and_eval_pools(monkeypatch):
    """Test held-out personas evaluate without entering the training pool."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/pass@2/"
        "qwen8b-qwen1.7b-math-student5-heldout5.yaml"
    )

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )
    seen_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_seen_path, role="seen student"
    )
    heldout_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_heldout_path, role="held-out student"
    )

    assert config.prompt_pool.include_base is True
    assert OmegaConf.load(path).prompt_pool.include_base is True
    assert config.prompt_pool.student_train_path == config.prompt_pool.student_seen_path
    assert config.prompt_pool.student_eval_paths == {
        "seen": config.prompt_pool.student_seen_path,
        "heldout": config.prompt_pool.student_heldout_path,
    }
    assert len(seen_prompts) == len(heldout_prompts) == 5
    assert set(seen_prompts).isdisjoint(heldout_prompts)


def test_pass2_soft_seen_persona_config_preserves_original_pool(monkeypatch):
    """Test the soft pilot uses a new seen pool without rewriting the old one."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    config_dir = Path("examples/tutor/configs/math/july/pass@2")
    soft_path = config_dir / "qwen8b-qwen1.7b-math-student5-soft-heldout5.yaml"
    original_path = config_dir / "qwen8b-qwen1.7b-math-student5-heldout5.yaml"

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(soft_path))
    )
    soft_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_seen_path, role="soft seen student"
    )
    original_source = OmegaConf.load(original_path)

    assert config.prompt_pool.student_seen_path.endswith(
        "student_personality_soft_suffixes_5.json"
    )
    assert original_source.prompt_pool.student_seen_path.endswith(
        "student_personality_suffixes_5.json"
    )
    assert config.prompt_pool.include_base is True
    assert len(soft_prompts) == 5
    assert tuple(
        prompt.split("]", 1)[0].removeprefix("[Persona: ") for prompt in soft_prompts
    ) == (
        "QuickHelpSeeker",
        "IndependentPerseverer",
        "ReceptiveFollower",
        "SkepticalDefender",
        "AdaptiveExplorer",
    )
    rigid_fragments = (
        "Mandatory response rule",
        "entire reply must",
        "Start with exactly",
        "This rule overrides",
    )
    assert all(
        fragment not in prompt
        for prompt in soft_prompts
        for fragment in rigid_fragments
    )


def test_pass2_preference_persona_config_uses_short_non_mandatory_prompts(monkeypatch):
    """Test the preference pilot stays concise and avoids response rules."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/pass@2/"
        "qwen8b-qwen1.7b-math-student5-preference-heldout5.yaml"
    )

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )
    seen_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_seen_path, role="preference seen student"
    )
    heldout_prompts = tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_heldout_path,
        role="preference held-out student",
    )

    assert len(seen_prompts) == len(heldout_prompts) == 5
    assert config.prompt_pool.include_base is True
    preference_words = (
        "comfortable",
        "satisfying",
        "naturally",
        "tend",
        "enjoy",
        "curious",
        "feel",
        "like",
        "think best",
        "dislike",
    )
    forbidden_rules = ("always", "mandatory", "must", "exactly", "every reply")
    for prompt in (*seen_prompts, *heldout_prompts):
        normalized = prompt.lower()
        assert len(prompt.split()) <= 42
        assert any(word in normalized for word in preference_words)
        assert all(rule not in normalized for rule in forbidden_rules)


def test_pass2_persona_v2_configs_use_validated_behavior_pools(monkeypatch):
    """Test persona-v2 configs share concise seen and held-out behavior pools."""
    # Arrange
    monkeypatch.setenv("INF_API_KEY", "test-key")
    config_dir = Path("examples/tutor/configs/math/july/pass@2/persona-v2")
    config_paths = sorted(config_dir.glob("*.yaml"))

    # Act
    configs = [
        OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
        )
        for path in config_paths
    ]
    heldout_config = next(
        config
        for path, config in zip(config_paths, configs, strict=True)
        if path.name == "qwen8b-qwen1.7b-math-student5-heldout5.yaml"
    )
    seen_prompts = tutor_workflow.load_prompt_pool(
        heldout_config.prompt_pool.student_seen_path, role="persona-v2 seen student"
    )
    heldout_prompts = tutor_workflow.load_prompt_pool(
        heldout_config.prompt_pool.student_heldout_path,
        role="persona-v2 held-out student",
    )

    # Assert
    assert len(configs) == 4
    assert all(
        config.prompt_pool.student_seen_path.endswith(
            "student_personality_behavior_v2_suffixes_5.json"
        )
        for config in configs
    )
    assert heldout_config.prompt_pool.student_heldout_path.endswith(
        "student_personality_heldout_behavior_v2_suffixes_5.json"
    )
    assert tuple(
        prompt.split("]", 1)[0].removeprefix("[Persona: ") for prompt in seen_prompts
    ) == (
        "QuickHelpSeeker",
        "IndependentPerseverer",
        "ReceptiveFollower",
        "SkepticalDefender",
        "ThinkAloudCollaborator",
    )
    assert tuple(
        prompt.split("]", 1)[0].removeprefix("[Persona: ") for prompt in heldout_prompts
    ) == (
        "MinimalHintSeeker",
        "CriticalFollower",
        "ConfidenceSharer",
        "ChoiceExplainer",
        "FeedbackRestater",
    )
    forbidden_rules = ("always", "mandatory", "must", "exactly", "every reply")
    for prompt in (*seen_prompts, *heldout_prompts):
        normalized = prompt.lower()
        assert len(prompt.split()) <= 42
        assert any(word in normalized for word in ("prefer", "feel", "like", "tend"))
        assert all(rule not in normalized for rule in forbidden_rules)


def test_teacher_warmup_asset_exactly_matches_20260704_prompt():
    """Test the historical warm-up prompt is copied without rewriting lessons."""
    source_path = Path(
        "examples/tutor/configs/math/0708/"
        "baseline-overfit-8-hard-leakt-prompt-2-pre-batch128-rebn-nomean.yaml"
    )
    asset_path = Path("examples/tutor/prompt_pools/teacher_warmup_20260704.txt")

    source_prompt = str(OmegaConf.load(source_path).teacher_system_prompt).strip()
    asset_prompt = tutor_workflow.load_prompt_text(
        str(asset_path), role="teacher warm-up"
    )

    assert asset_prompt == source_prompt


def test_teacher_pool_example_enables_fifty_step_warmup(monkeypatch):
    """Test the active teacher-pool experiment enables the historical curriculum."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path(
        "examples/tutor/configs/math/july/"
        "baseline-overfit-1-leakt-batch128-rebn-nomean-5-teacher-pools-wp.yaml"
    )

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    assert isinstance(config, TutorConfig)
    assert config.prompt_pool.teacher_warmup.enabled is True
    assert config.prompt_pool.teacher_warmup.steps == 50
    assert config.prompt_pool.teacher_warmup.prompt_path.endswith(
        "teacher_warmup_20260704.txt"
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "baseline-overfit-1-leakt-batch128-rebn-nomean-5-prompt-pools.yaml",
        "baseline-overfit-1-leakt-batch128-rebn-nomean-5-teacher-pools.yaml",
        "baseline-overfit-1-leakt-batch128-rebn-nomean-5-teacher-pools-wp.yaml",
    ],
)
def test_pool_examples_penalize_max_turn_failure(monkeypatch, config_name):
    """Test all pool experiments make max-turn failure as costly as leakage."""
    monkeypatch.setenv("INF_API_KEY", "test-key")
    path = Path("examples/tutor/configs/math/july") / config_name

    config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TutorConfig), OmegaConf.load(path))
    )

    assert isinstance(config, TutorConfig)
    assert config.reward.max_turn_penalty == pytest.approx(-1.0)
