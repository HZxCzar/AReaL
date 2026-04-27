from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _install_areal_stubs_if_needed() -> None:
    try:
        importlib.import_module("areal.utils.hf_utils")
        return
    except Exception:
        for name in list(sys.modules):
            if name == "areal" or name.startswith("areal."):
                del sys.modules[name]

    areal = ModuleType("areal")
    workflow_context = SimpleNamespace(
        stat_scope=lambda: "examples",
        get=lambda: SimpleNamespace(task_id=None, is_eval=False),
    )
    api = ModuleType("areal.api")
    api.RolloutWorkflow = object
    experimental = ModuleType("areal.experimental")
    openai_mod = ModuleType("areal.experimental.openai")
    openai_mod.ArealOpenAI = object
    utils = ModuleType("areal.utils")
    logger = SimpleNamespace(
        debug=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    utils.logging = SimpleNamespace(getLogger=lambda *args, **kwargs: logger)
    utils.stats_tracker = SimpleNamespace(
        get=lambda _scope: SimpleNamespace(scalar=lambda **kwargs: None)
    )
    hf_utils = ModuleType("areal.utils.hf_utils")
    hf_utils.load_hf_tokenizer = lambda path: path

    areal.workflow_context = workflow_context
    sys.modules.update(
        {
            "areal": areal,
            "areal.api": api,
            "areal.experimental": experimental,
            "areal.experimental.openai": openai_mod,
            "areal.utils": utils,
            "areal.utils.hf_utils": hf_utils,
        }
    )


_install_areal_stubs_if_needed()

tutor_workflow = importlib.import_module("examples.tutor.workflow")
GeneratedProblemResult = tutor_workflow.GeneratedProblemResult
LeakCheckResult = tutor_workflow.LeakCheckResult
TutorAgentWorkflow = tutor_workflow.TutorAgentWorkflow


@pytest.fixture(autouse=True)
def _patch_render_prompt(monkeypatch):
    def _render_prompt(template: str, **context: Any) -> str:
        if "New related task:" in template:
            visible_history = context["visible_history"]
            history_text = (
                "\n".join(visible_history)
                if visible_history
                else "No visible teacher turns before transfer."
            )
            return (
                f"Original task:\n{context['original_task']}\n\n"
                f"Initial student answer:\n{context['initial_student_answer']}\n\n"
                f"Visible tutoring history:\n{history_text}\n\n"
                f"New related task:\n{context['transfer_task']}\n\n"
                "You are now solving the new related task."
            )
        if "Current teacher feedback:" in template:
            visible_history = context["visible_history"]
            history_text = (
                "\n".join(visible_history)
                if visible_history
                else "No previous visible turns."
            )
            return (
                f"Task:\n{context['task']}\n\n"
                f"Visible student history:\n{history_text}\n\n"
                f"Current teacher feedback:\n{context['teacher_feedback']}"
            )
        if "Turn 0:" in template:
            return (
                f"Task:\n{context['task']}\n\n"
                f"Turn 0:\n- Student initial answer: "
                f"{context['initial_student_answer']}"
            )
        if "Turn {{ previous_round_idx }} update:" in template:
            return "followup"
        return ""

    monkeypatch.setattr(tutor_workflow, "render_prompt", _render_prompt)


@dataclass
class _TeacherMessage:
    content: str

    def model_dump(self, exclude_none: bool = True) -> dict[str, str]:
        del exclude_none
        return {"role": "assistant", "content": self.content}


class _TeacherCompletions:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.outputs.pop(0)
        message = _TeacherMessage(content)
        return SimpleNamespace(
            id=f"completion-{len(self.calls)}",
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _TeacherClient:
    def __init__(self, outputs: list[str]):
        self.completions = _TeacherCompletions(outputs)
        self.chat = SimpleNamespace(completions=self.completions)


class _ScriptedTutorWorkflow(TutorAgentWorkflow):
    def __init__(
        self,
        *,
        student_outputs: list[str],
        transfer_generation: GeneratedProblemResult | None = None,
    ):
        super().__init__(
            term_success_reward=1.0,
            transfer_bonus_reward=0.5,
            max_turns=2,
            debug_trace_dir="",
        )
        self.student_outputs = list(student_outputs)
        self.student_prompts: list[str] = []
        self.transfer_generation = transfer_generation or GeneratedProblemResult(
            raw_output="",
            task="",
            ground_truth="",
            similarity_notes="",
            parse_error=None,
            raw_result={},
        )

    async def _call_student_prompt(self, prompt: str) -> str:
        self.student_prompts.append(prompt)
        return self.student_outputs.pop(0)

    async def _run_leak_check(
        self, task: str, ground_truth: str, teacher_action: str
    ) -> LeakCheckResult:
        del task, ground_truth, teacher_action
        return LeakCheckResult("", False, "ok", None, {})

    async def _run_transfer_generation(
        self, task: str, ground_truth: str
    ) -> GeneratedProblemResult:
        del task, ground_truth
        return self.transfer_generation


@pytest.mark.asyncio
async def test_run_episode_when_pre_solved_returns_none_and_logs_pre_success(
    monkeypatch,
):
    """Pre-solved samples are logged but not exported for GRPO update."""
    stats: list[dict[str, Any]] = []
    monkeypatch.setattr(tutor_workflow, "_safe_scalar", lambda **kwargs: stats.append(kwargs))
    workflow = _ScriptedTutorWorkflow(student_outputs=["The answer is 7."])
    teacher = _TeacherClient(outputs=["unused"])

    result = await workflow._run_episode(
        {"task": "original task", "ground_truth": "7"},
        external_client=teacher,
    )

    assert result is None
    assert teacher.completions.calls == []
    assert workflow.last_history == []
    assert stats[-1]["pre_success"] == 1.0
    assert stats[-1]["term_success"] == 0.0
    assert stats[-1]["transfer_success"] == 0.0


@pytest.mark.asyncio
async def test_run_episode_when_tutor_solves_and_transfer_passes_logs_term_metrics(
    monkeypatch,
):
    """Post-teaching success receives term reward and transfer bonus."""
    stats: list[dict[str, Any]] = []
    monkeypatch.setattr(tutor_workflow, "_safe_scalar", lambda **kwargs: stats.append(kwargs))
    workflow = _ScriptedTutorWorkflow(
        student_outputs=[
            "The answer is 1.",
            "The answer is 7.",
            "The answer is 9.",
        ],
        transfer_generation=GeneratedProblemResult(
            raw_output="{}",
            task="new related task",
            ground_truth="9",
            similarity_notes="",
            parse_error=None,
            raw_result={},
        ),
    )
    teacher = _TeacherClient(outputs=["Helpful hint"])

    result = await workflow._run_episode(
        {"task": "original task", "ground_truth": "7"},
        external_client=teacher,
    )

    assert result == (1.5, "completion-1")
    assert stats[-1]["pre_success"] == 0.0
    assert stats[-1]["term_success"] == 1.0
    assert stats[-1]["transfer_success"] == 1.0
    assert stats[-1]["term_reward_sum"] == 1.0
    assert stats[-1]["transfer_bonus_sum"] == 0.5
    transfer_prompt = workflow.student_prompts[-1]
    assert "Original task:\noriginal task" in transfer_prompt
    assert "Initial student answer:\nThe answer is 1." in transfer_prompt
    assert "Teacher guidance: Helpful hint." in transfer_prompt
    assert "Student reply: The answer is 7." in transfer_prompt
    assert "New related task:\nnew related task" in transfer_prompt


def test_build_transfer_student_prompt_excludes_leaked_turns():
    """Transfer-visible history follows the same leak filtering as normal student history."""
    workflow = TutorAgentWorkflow(debug_trace_dir="")
    prompt = workflow._build_transfer_student_prompt(
        original_task="original",
        initial_student_answer="initial",
        history=[
            {
                "round_idx": 1,
                "teacher_action": "leaked answer",
                "student_answer": "",
                "judge_feedback": "",
                "leak_detected": True,
            },
            {
                "round_idx": 2,
                "teacher_action": "visible hint",
                "student_answer": "revised answer",
                "judge_feedback": "Correct.",
                "leak_detected": False,
            },
        ],
        transfer_task="transfer",
    )

    assert "leaked answer" not in prompt
    assert "visible hint" in prompt
    assert "revised answer" in prompt
    assert "transfer" in prompt
