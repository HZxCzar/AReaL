from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from examples.pedagogical_rl.config import (
    PedagogicalAPIModelConfig,
    PedagogicalGenerationConfig,
    PedagogicalTeacherPreConfig,
)
from examples.pedagogical_rl.state import ClassroomEpisode, ConversationType
from examples.pedagogical_rl.workflow import PedagogicalRLWorkflow
from examples.tutor.prompts import (
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
)

from areal.api.cli_args import GenerationHyperparameters
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, **_kwargs):
        return list(text)


class _FakeActorClient:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.chat = SimpleNamespace(completions=self)
        self.reward = None
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        message = SimpleNamespace(content=output)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def set_last_reward(self, reward: float):
        self.reward = reward

    def export_interactions(self, style: str):
        assert style == "concat"
        return {"leaf": "teacher-only-interaction"}


class _FakeStudentClient:
    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []
        self.requested_ns: list[int] = []

    async def generate(self, messages, *, n, **_kwargs):
        self.calls.append(messages)
        self.requested_ns.append(n)
        final = bool(
            messages
            and "conversation with the teacher has ended"
            in messages[-1]["content"].lower()
        )
        output = r"Complete work. \boxed{5}" if final else r"I think \boxed{0}"
        return [output] * n


class _FakeJudgeClient:
    def __init__(
        self,
        *,
        leak: bool,
        answer_correct: bool = False,
        native_reject_rules: set[str] | None = None,
    ):
        self.leak = leak
        self.answer_correct = answer_correct
        self.native_reject_rules = native_reject_rules or set()
        self.call_kinds: list[str] = []
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages, *, n, **_kwargs):
        self.calls.append(messages)
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system == RAWBASE_LEAK_CHECK_SYSTEM_PROMPT:
            self.call_kinds.append("rawbase")
            value = "true" if self.leak else "false"
            return [f'{{"leaked": {value}, "feedback": "gate"}}'] * n
        if system == DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT:
            self.call_kinds.append("answer")
            value = "true" if self.answer_correct else "false"
            return [f'{{"correct": {value}}}'] * n

        prompt = messages[0]["content"]
        rule = (
            "does_not_leak_answer"
            if "revealed too much information too early" in prompt
            else "follows_pedagogical_values"
        )
        self.call_kinds.append(f"native:{rule}")
        decision = "REJECT" if rule in self.native_reject_rules else "OK"
        return [f'{{"reasoning": "fine", "decision": "{decision}"}}'] * n


def _problem_for(conversation_type: ConversationType) -> str:
    for index in range(1000):
        problem = f"What is 2+3? Variant {index}"
        episode = ClassroomEpisode(problem=problem, answer="5")
        if episode.conversation_type is conversation_type:
            return problem
    raise AssertionError("failed to find problem with requested conversation type")


def _make_workflow(
    actor,
    student,
    judge,
    *,
    leak_judge_mode: str = "pedagogical_rl",
    teacher_pre: PedagogicalTeacherPreConfig | None = None,
    eval_repeat_count: int = 1,
):
    model = PedagogicalAPIModelConfig(
        base_url="http://unused", model="fake", api_key="x"
    )
    return PedagogicalRLWorkflow(
        gconfig=GenerationHyperparameters(
            n_samples=1,
            max_new_tokens=4096,
            max_tokens=40960,
            temperature=0.7,
            top_p=1.0,
        ),
        tokenizer=_Tokenizer(),
        student_model=model,
        judge_model=model,
        generation=PedagogicalGenerationConfig(
            leak_judge_mode=leak_judge_mode,
        ),
        teacher_pre=teacher_pre or PedagogicalTeacherPreConfig(),
        eval_repeat_count=eval_repeat_count,
        student_client=student,
        judge_client=judge,
        actor_client_factory=lambda **_kwargs: actor,
    )


def _run_episode(
    workflow: PedagogicalRLWorkflow,
    *,
    is_eval: bool,
    task: str,
    task_id: int = 1,
):
    workflow_context.set(
        WorkflowContext(is_eval=is_eval, task_id=task_id, lora_version=0)
    )
    try:
        return asyncio.run(
            workflow.arun_episode(None, {"task": task, "ground_truth": "5"})
        )
    finally:
        workflow_context.set(WorkflowContext())


def test_training_pedagogical_rl_mode_uses_only_native_gates():
    actor = _FakeActorClient(["Good work.<end_of_conversation>"])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=True)
    workflow = _make_workflow(actor, student, judge)
    metric_calls = []
    workflow._safe_stats = metric_calls.append

    result = _run_episode(
        workflow,
        is_eval=False,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert result == {"leaf": "teacher-only-interaction"}
    assert "rawbase" not in judge.call_kinds
    assert judge.call_kinds.count("native:does_not_leak_answer") == 2
    assert judge.call_kinds.count("native:follows_pedagogical_values") == 2
    assert actor.reward == pytest.approx(1.1)
    assert metric_calls[0]["leak_gate/pedagogical_rl"] == 1.0
    assert metric_calls[0]["native_leak/rejected"] == 0.0


def test_training_native_rejection_hard_gates_final_solution():
    actor = _FakeActorClient(["Stop.<end_of_conversation>"])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(
        leak=False,
        native_reject_rules={"does_not_leak_answer"},
    )
    workflow = _make_workflow(actor, student, judge)
    metric_calls = []
    workflow._safe_stats = metric_calls.append

    _run_episode(
        workflow,
        is_eval=False,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert actor.reward == -1.0
    assert judge.call_kinds.count("native:does_not_leak_answer") == 2
    assert "native:follows_pedagogical_values" not in judge.call_kinds
    assert student.requested_ns == []
    assert metric_calls[0]["end_rm_reward"] == -1.0


def test_training_turn_mode_replaces_native_leak_gate_and_terminates():
    actor = _FakeActorClient(["The final answer is 5."])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=True)
    workflow = _make_workflow(
        actor,
        student,
        judge,
        leak_judge_mode="turn",
    )
    metric_calls = []
    workflow._safe_stats = metric_calls.append

    _run_episode(
        workflow,
        is_eval=False,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert actor.reward == -1.0
    assert judge.call_kinds.count("rawbase") == 1
    assert "native:does_not_leak_answer" not in judge.call_kinds
    assert judge.call_kinds.count("native:follows_pedagogical_values") == 2
    assert student.calls == []
    assert metric_calls[0]["turn_leak/any"] == 1.0
    assert metric_calls[0]["leak_gate/turn"] == 1.0


def test_eval_turn_leak_is_observed_without_terminating_shared_rollout():
    actor = _FakeActorClient(
        ["The answer may be 5; what do you think?", "Finish.<end_of_conversation>"]
    )
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=True)
    workflow = _make_workflow(actor, student, judge)
    metric_calls = []
    workflow._safe_stats = metric_calls.append

    result = _run_episode(
        workflow,
        is_eval=True,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert result == {"leaf": "teacher-only-interaction"}
    assert len(actor.calls) == 2
    assert judge.call_kinds.count("rawbase") == 2
    assert judge.call_kinds.count("native:does_not_leak_answer") == 2
    assert judge.call_kinds.count("native:follows_pedagogical_values") == 2
    assert student.requested_ns == [1, 8]
    assert actor.reward == 1.0
    assert metric_calls[0]["turn_leak/any"] == 1.0
    assert metric_calls[0]["stop/leak"] == 0.0
    assert metric_calls[0]["final_correct"] == 1.0
    assert metric_calls[0]["accuracy/raw"] == 1.0
    assert metric_calls[0]["accuracy/turn_leak_gate"] == 0.0
    assert metric_calls[0]["accuracy/pedagogical_rl_leak_gate"] == 1.0


def test_eval_native_rejection_is_diagnostic_and_final_still_runs():
    actor = _FakeActorClient(["Stop.<end_of_conversation>"])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(
        leak=False,
        native_reject_rules={
            "does_not_leak_answer",
            "follows_pedagogical_values",
        },
    )
    workflow = _make_workflow(actor, student, judge)
    metric_calls = []
    workflow._safe_stats = metric_calls.append

    _run_episode(
        workflow,
        is_eval=True,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert judge.call_kinds.count("native:does_not_leak_answer") == 2
    assert judge.call_kinds.count("native:follows_pedagogical_values") == 2
    assert student.requested_ns == [8]
    assert actor.reward == 1.0
    assert metric_calls[0]["native_leak/rejected"] == 1.0
    assert metric_calls[0]["native_pedagogy/rejected"] == 1.0
    assert metric_calls[0]["final_correct"] == 1.0
    assert metric_calls[0]["accuracy/turn_leak_gate"] == 1.0
    assert metric_calls[0]["accuracy/pedagogical_rl_leak_gate"] == 0.0


def test_eval_preserves_native_guided_and_attempted_state_machine():
    guided_actor = _FakeActorClient(["Stop.<end_of_conversation>"])
    guided_student = _FakeStudentClient()
    guided_judge = _FakeJudgeClient(leak=False)
    guided = _make_workflow(guided_actor, guided_student, guided_judge)

    _run_episode(
        guided,
        is_eval=True,
        task=_problem_for(ConversationType.GUIDED),
    )

    attempted_actor = _FakeActorClient(["Stop.<end_of_conversation>"])
    attempted_student = _FakeStudentClient()
    attempted_judge = _FakeJudgeClient(leak=False)
    attempted = _make_workflow(attempted_actor, attempted_student, attempted_judge)

    _run_episode(
        attempted,
        is_eval=True,
        task=_problem_for(ConversationType.ATTEMPTED),
    )

    assert guided_student.requested_ns == [8]
    assert attempted_student.requested_ns == [1, 8]
    assert "answer" not in guided_judge.call_kinds
    assert "answer" not in attempted_judge.call_kinds


def test_teacher_pre_verify_retries_and_uses_first_correct_private_draft():
    actor = _FakeActorClient(
        ["wrong draft", r"correct draft \boxed{5}", "Stop.<end_of_conversation>"]
    )
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=False, answer_correct=False)
    workflow = _make_workflow(
        actor,
        student,
        judge,
        teacher_pre=PedagogicalTeacherPreConfig(
            enabled=True,
            verify=True,
            attempts=3,
            max_tokens=128,
        ),
    )

    _run_episode(
        workflow,
        is_eval=False,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert [call.get("store") for call in actor.calls[:2]] == [False, False]
    assert "store" not in actor.calls[2]
    assert actor.calls[0]["max_completion_tokens"] == 128
    assert judge.call_kinds.count("answer") == 1
    teacher_system = actor.calls[2]["messages"][0]["content"]
    assert r"correct draft \boxed{5}" in teacher_system
    assert "wrong draft" not in teacher_system


def test_teacher_pre_draft_is_hidden_from_student_and_both_leak_judges():
    draft = r"private solution draft with \boxed{5}"
    actor = _FakeActorClient([draft, "Stop.<end_of_conversation>"])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=False)
    workflow = _make_workflow(
        actor,
        student,
        judge,
        teacher_pre=PedagogicalTeacherPreConfig(enabled=True, verify=False),
    )

    _run_episode(
        workflow,
        is_eval=True,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert len(actor.calls) == 2
    assert actor.calls[0]["store"] is False
    assert "answer" not in judge.call_kinds
    assert draft in actor.calls[1]["messages"][0]["content"]
    assert all(draft not in str(messages) for messages in student.calls)
    assert all(draft not in str(messages) for messages in judge.calls)


def test_teacher_pre_verified_failure_returns_none_for_atomic_group_rejection():
    actor = _FakeActorClient(["wrong 1", "wrong 2", "wrong 3"])
    student = _FakeStudentClient()
    judge = _FakeJudgeClient(leak=False, answer_correct=False)
    workflow = _make_workflow(
        actor,
        student,
        judge,
        teacher_pre=PedagogicalTeacherPreConfig(
            enabled=True,
            verify=True,
            attempts=3,
        ),
    )

    result = _run_episode(
        workflow,
        is_eval=False,
        task=_problem_for(ConversationType.GUIDED),
    )

    assert result is None
    assert workflow.require_complete_group is True
    assert actor.reward is None
    assert all(call["store"] is False for call in actor.calls)
    assert judge.call_kinds.count("answer") == 3


def test_eval_repeat_metrics_match_tutor_aggregation():
    workflow = _make_workflow(
        _FakeActorClient([]),
        _FakeStudentClient(),
        _FakeJudgeClient(leak=False),
        eval_repeat_count=3,
    )
    metric_calls = []
    workflow._safe_stats = metric_calls.append
    workflow_context.set(WorkflowContext(is_eval=True, task_id=7, lora_version=0))
    try:
        for value in (1.0, 0.0, 1.0):
            workflow._record_eval_repeat_metrics(value)
    finally:
        workflow_context.set(WorkflowContext())

    assert set(metric_calls[0]) == {"repeat/final_correct/mean_task_sample_variance"}
    assert metric_calls[0][
        "repeat/final_correct/mean_task_sample_variance"
    ] == pytest.approx(1.0 / 3.0)
    jaccards = [
        call["repeat/final_correct/pairwise_success_jaccard"]
        for call in metric_calls[1:]
    ]
    assert jaccards == [0.0, 1.0, 0.0]
