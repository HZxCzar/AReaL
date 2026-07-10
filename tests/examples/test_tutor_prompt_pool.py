from __future__ import annotations

import asyncio
import json
import random
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.tutor import workflow as tutor_workflow
from examples.tutor.configs import TutorConfig
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
