"""Regression tests for code-student post-dialogue re-tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from examples.tutor.core.callers import TextCallResult
from examples.tutor.core.code_exec import CodeSession
from examples.tutor.core.types import JudgeResult, PublicHistoryState
from examples.tutor.workflow import (
    StudentGeneralizationAnchor,
    StudentGeneralizationCase,
    TutorAgentWorkflow,
)


def test_code_retest_uses_fresh_execution_but_keeps_transcript():
    """Old source remains visible, while old runtime names are unavailable."""

    async def run_scenario() -> tuple[list, list[list[dict[str, str]]], CodeSession]:
        workflow = object.__new__(TutorAgentWorkflow)
        workflow.student_generalize_enabled = True
        workflow.student_generalize_retest_original = True
        workflow.student_generalize_replays = 2
        workflow.student_generalize_confidence_enabled = False
        workflow.student_generalize_turn_credit = False
        workflow.free_chat_enabled = True
        workflow.free_chat_transfer_prompts = False
        workflow.student_mask_active = False
        workflow.student_model_runtimes = {}

        visible_old_program = "old_answer = 42"
        anchor = StudentGeneralizationAnchor(
            public_history=PublicHistoryState(
                summary="",
                turn_count=2,
                turns=[
                    {"role": "teacher", "content": "Work out the answer."},
                    {
                        "role": "student",
                        "content": (
                            "```python\n"
                            f"{visible_old_program}\n"
                            "```\n[result]\n(no output)"
                        ),
                    },
                ],
            ),
            previous_student_output="(no output)",
            teacher_feedback="",
            reward_turn_idx=1,
        )
        episode = SimpleNamespace(
            task="What is six times seven?",
            student_mode="code",
            student_name="code-original",
            student_prompt_selection=None,
            turns=[],
        )

        workflow._student_generalization_anchor = (
            lambda _episode, allow_unsuccessful=False: anchor
        )
        workflow._preleak_retest_active = lambda: False
        workflow._probe_levels = lambda: ("original",)
        workflow._student_generalization_cases = lambda _data: {
            "original": StudentGeneralizationCase(
                level="original",
                task=episode.task,
                ground_truth="42",
                reward=1.0,
            )
        }

        seen_messages: list[list[dict[str, str]]] = []

        async def fake_student_call(messages, *, aux_caller, rid_prefix):
            del aux_caller
            seen_messages.append(messages)
            program = (
                "old_answer" if "-r0-" in rid_prefix else "old_answer = 42\nold_answer"
            )
            raw = f"```python\n{program}\n```"
            return TextCallResult(text=raw, raw_text=raw)

        async def fake_score(
            task,
            ground_truth,
            answer,
            *,
            answer_judge_caller,
            code_output=False,
        ):
            del task, ground_truth, answer_judge_caller
            assert code_output is True
            correct = answer.strip() == "42"
            return JudgeResult(
                raw_output="",
                correct=correct,
                feedback="Correct." if correct else "Incorrect.",
                parse_error=None,
                raw_result={},
            )

        workflow._call_auxiliary_messages = fake_student_call
        workflow._score_answer_async = fake_score

        tutoring_session = CodeSession()
        assert (await tutoring_session.run(visible_old_program)).ok
        results = await workflow._run_student_generalization(
            {"task": episode.task, "ground_truth": "42"},
            episode,
            aux_caller=None,
            answer_judge_caller=None,
            code_session=tutoring_session,
        )
        return results, seen_messages, tutoring_session

    results, seen_messages, tutoring_session = asyncio.run(run_scenario())

    assert results[0].replay_count == 2
    assert results[0].replay_correct == 1
    assert len(seen_messages) == 2
    assert all(
        "old_answer = 42" in "\n".join(message["content"] for message in messages)
        for messages in seen_messages
    )
    assert tutoring_session.turns_kept == 1
