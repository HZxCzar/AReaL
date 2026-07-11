from __future__ import annotations

import asyncio
import json
import random
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import TutorConfig, TutorTeacherWarmupPromptConfig
from examples.tutor.core.pairwise import PairwiseTutorEvaluator
from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    PromptPoolSelection,
    PublicHistoryState,
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


def test_pairwise_reference_teacher_reuses_episode_selection():
    """Test the lagged teacher receives the current episode's strategy suffix."""
    workflow = tutor_workflow.TutorAgentWorkflow.__new__(
        tutor_workflow.TutorAgentWorkflow
    )
    workflow.teacher_system_prompt = "base teacher"
    workflow._build_tutor_prompt = lambda _state: "teacher user prompt"
    workflow.gconfig = None
    workflow.max_completion_tokens = 32
    workflow.max_train_sample_tokens = 128
    captured = {}

    class ChatCaller:
        async def generate(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(raw_text="reference hint")

    state = TutorTurnState(
        task="task",
        ground_truth="42",
        public_history=PublicHistoryState(),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=1,
        max_turns=1,
        teacher_prompt_selection=PromptPoolSelection(
            index=4, suffix="use backward reasoning"
        ),
    )

    output = asyncio.run(
        workflow._generate_reference_tutor_response(
            state,
            chat_caller=ChatCaller(),
            reference_version=3,
        )
    )

    assert output == "reference hint"
    assert captured["messages"][0]["content"] == (
        "base teacher\n\nuse backward reasoning"
    )
    assert captured["kwargs"]["metadata"] == {"lora_version": 3}


def test_pairwise_reference_student_reuses_episode_selection():
    """Test pairwise comparisons do not introduce a second student behavior."""
    selection = PromptPoolSelection(index=2, suffix="work backward")
    current_state = StudentTurnState(
        task="task",
        public_history=PublicHistoryState(),
        previous_student_output="wrong",
        latest_tutor_visible_output="current hint",
        student_prompt_selection=selection,
    )
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
        tutor_raw_output="current hint",
        tutor_visible_output="current hint",
        leak_result=_leak(),
        public_history_before="",
        public_history_after="",
        student_state=current_state,
        student_output="current answer",
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
        latest_student_answer="current answer",
        student_prompt_selection=selection,
    )
    captured_states = []

    async def generate_reference_tutor(_state, _version):
        return "reference hint"

    async def run_student(state):
        captured_states.append(state)
        return "reference answer", None

    async def run_leak_check(_task, _ground_truth, _teacher_action):
        return _leak()

    async def score_answer(_task, _ground_truth, _student_output):
        return _judge(False)

    evaluator = PairwiseTutorEvaluator(
        reward_scale=0.1,
        reward_caller=object(),
        generate_reference_tutor=generate_reference_tutor,
        run_student=run_student,
        run_leak_check=run_leak_check,
        score_answer=score_answer,
    )

    result = asyncio.run(evaluator.evaluate_turn(episode, turn, reference_version=0))

    assert result.outcome == "current"
    assert captured_states[0].student_prompt_selection == selection


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
    assert config.prompt_pool.student_path.endswith("student_behavior_suffixes.json")
    assert tutor_workflow.load_prompt_pool(
        config.prompt_pool.teacher_path, role="teacher"
    )
    assert tutor_workflow.load_prompt_pool(
        config.prompt_pool.student_path, role="student"
    )


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
