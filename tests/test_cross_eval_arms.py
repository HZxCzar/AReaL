"""Each arm's adapter into the shared cross-eval instrument.

test_cross_eval.py covers the instrument. This covers the two thin layers that
feed it -- that each arm hands it the right transcript, the right protocol
label, and specs built from its own config -- because that is where the two
arms can silently stop being comparable.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from examples.pedagogical_rl import cross_eval as ce
from examples.pedagogical_rl.config import (
    PedagogicalAPIModelConfig,
    PedagogicalGenerationConfig,
)
from examples.pedagogical_rl.cross_eval_config import CrossEvalConfig
from examples.pedagogical_rl.workflow import PedagogicalRLWorkflow
from examples.tutor.workflow import TutorAgentWorkflow


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text or "") // 4 + 1))


class FakeApiClient:
    """Stands in for PedagogicalAPIClient."""

    def __init__(self, text: str = "\\boxed{7}") -> None:
        self.text = text
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    async def generate(self, messages, *, n, max_tokens, temperature, top_p):
        self.calls.append((messages, n))
        return [self.text] * n


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()
        self.index = 0


class FakeActorClient:
    def __init__(self, content: str = "keep going") -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []
        create = self._create
        self.chat = type(
            "Chat", (), {"completions": type("C", (), {"create": create})()}
        )()

    async def _create(self, *, messages, **kwargs):
        self.calls.append(messages)
        return type("R", (), {"choices": [FakeChoice(self.content)]})()


def _ped_workflow(cross_eval_config: CrossEvalConfig) -> PedagogicalRLWorkflow:
    gconfig = type("G", (), {"max_tokens": 4096, "temperature": 1.0, "top_p": 1.0})()
    return PedagogicalRLWorkflow(
        gconfig=gconfig,
        tokenizer=FakeTokenizer(),
        student_model=PedagogicalAPIModelConfig(base_url="x", model="s", api_key="k"),
        judge_model=PedagogicalAPIModelConfig(base_url="x", model="j", api_key="k"),
        generation=PedagogicalGenerationConfig(),
        cross_eval=cross_eval_config,
        student_client=FakeApiClient(),
        judge_client=FakeApiClient('{"leaked": false, "feedback": "clean"}'),
    )


def _turn_artifact(visible: str, student: str) -> SimpleNamespace:
    """Only the two fields _cross_eval_transcript reads.

    A real TurnArtifact carries the whole turn -- prompts, response, leak
    result, judge -- and none of it reaches the transcript, so building one here
    would couple this test to fields it is not about.
    """

    return SimpleNamespace(tutor_visible_output=visible, student_output=student)


# --------------------------------------------------------------------------
# Tutor arm
# --------------------------------------------------------------------------


def test_tutor_transcript_is_the_visible_dialogue():
    transcript = TutorAgentWorkflow._cross_eval_transcript(
        [_turn_artifact("first hint", "ok"), _turn_artifact("second hint", "")]
    )
    assert transcript == [
        {"role": "teacher", "content": "first hint"},
        {"role": "student", "content": "ok"},
        {"role": "teacher", "content": "second hint"},
    ]


def test_tutor_transcript_keeps_a_malformed_turn_as_the_empty_message():
    """A malformed teacher turn hands the student nothing. The transcript has to
    say so, or the scorers see a turn the student never got."""

    transcript = TutorAgentWorkflow._cross_eval_transcript(
        [_turn_artifact("", "what?")]
    )
    assert transcript == [
        {"role": "teacher", "content": ""},
        {"role": "student", "content": "what?"},
    ]


def test_tutor_refuses_a_budget_that_disagrees_with_its_own_rollout():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.free_chat_budget = 5
    with pytest.raises(ValueError, match="must equal free_chat.budget"):
        workflow._init_cross_eval({"enabled": True, "free_chat": {"budget": 3}})
    # The matching one is accepted.
    workflow._init_cross_eval({"enabled": True, "free_chat": {"budget": 5}})
    assert workflow.cross_eval_enabled


def test_tutor_refuses_history_tags_that_disagree_with_its_own_rollout():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.free_chat_budget = 5
    workflow.teacher_history_tags = "masked"
    with pytest.raises(ValueError, match="teacher_history_tags"):
        workflow._init_cross_eval(
            {"enabled": True, "free_chat": {"teacher_history_tags": "stripped"}}
        )
    workflow._init_cross_eval(
        {"enabled": True, "free_chat": {"teacher_history_tags": "masked"}}
    )
    assert workflow.cross_eval_enabled


def test_tutor_refuses_the_cross_without_free_chat():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = False
    workflow.free_chat_budget = 0
    with pytest.raises(ValueError, match="requires free_chat.enabled"):
        workflow._init_cross_eval({"enabled": True})


def test_tutor_specs_follow_the_arms_own_settings():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.free_chat_budget = 5
    workflow.enable_thinking = False
    workflow.teacher_show_ground_truth = False
    workflow.free_chat_student_has_not_seen_problem = True
    workflow.teacher_history_tags = "masked"
    workflow.student_generalize_replays = 4
    workflow._init_cross_eval(
        {
            "enabled": True,
            "free_chat": {"budget": 5},
            "classroom": {"max_teacher_turns": 10},
            "retest": {"replays": 0},
            "interview": {"attempts": 8},
            "leak_judges": {"enabled": True, "native_attempts": 2},
        }
    )
    specs = workflow._cross_eval_specs()
    assert specs["free_chat"].budget == 5
    # Also follows the arm: the teacher must see its own history here exactly as
    # it does in the rollout being compared.
    assert specs["free_chat"].teacher_history_tags == "masked"
    # Follows the arm, not the cross-eval block: the teacher must open the same
    # way it does in the rollout being compared.
    assert specs["free_chat"].student_has_not_seen_problem is True
    # replays 0 falls back to what the arm is actually rewarded on.
    assert specs["retest"].replays == 4
    assert specs["interview"].attempts == 8
    assert specs["leak_judges"].native_attempts == 2


def test_tutor_leak_judges_can_be_switched_off_entirely():
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.free_chat_enabled = True
    workflow.free_chat_budget = 5
    workflow.enable_thinking = False
    workflow.teacher_show_ground_truth = False
    workflow.free_chat_student_has_not_seen_problem = False
    workflow.teacher_history_tags = "masked"
    workflow.student_generalize_replays = 4
    workflow._init_cross_eval({"enabled": True, "leak_judges": {"enabled": False}})
    assert workflow._cross_eval_specs()["leak_judges"] is None


def test_sample_rate_picks_the_same_problems_on_both_arms():
    """The two arms walk the eval split in different code and a different order.
    Selection is a hash of the problem text so the crossed subset is identical
    anyway -- if it were a counter or a seed, the two columns would be measured
    on different problems."""

    tutor = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    tutor.cross_eval_enabled = True
    tutor.cross_eval_sample_rate = 0.5
    ped = _ped_workflow(CrossEvalConfig(enabled=True, sample_rate=0.5))

    problems = [f"problem number {index}" for index in range(200)]
    tutor_pick = [tutor._cross_eval_selected(p) for p in problems]
    ped_pick = [ped._cross_eval_selected(p) for p in problems]
    assert tutor_pick == ped_pick
    # And it actually samples rather than taking everything or nothing.
    assert 60 < sum(tutor_pick) < 140


def test_sample_rate_one_takes_every_problem():
    tutor = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    tutor.cross_eval_enabled = True
    tutor.cross_eval_sample_rate = 1.0
    assert all(tutor._cross_eval_selected(f"p{i}") for i in range(50))


# --------------------------------------------------------------------------
# Ped arm
# --------------------------------------------------------------------------


def test_ped_arm_runs_the_full_cross_from_its_own_transcript():
    workflow = _ped_workflow(
        CrossEvalConfig(
            enabled=True,
            free_chat={"budget": 2},
            classroom={"max_teacher_turns": 2},
            retest={"replays": 2},
            interview={"attempts": 2},
            leak_judges={"native_attempts": 1},
        )
    )
    episode = ce.ClassroomEpisode(problem="2+2?", answer="7")
    episode.add_teacher("try adding")
    episode.add_student("ok")
    actor = FakeActorClient()
    metrics, details = asyncio.run(
        workflow._run_cross_eval(actor_client=actor, episode=episode)
    )
    for protocol in ("free_chat", "classroom"):
        for scorer in ("retest", "interview"):
            assert f"xeval/{protocol}/{scorer}/success" in metrics
    # Its own dialogue was handed in; only the tutor protocol was generated.
    assert metrics["xeval/classroom/turns"] == 1.0
    assert metrics["xeval/free_chat/turns"] == 2.0
    assert set(details) == {"free_chat", "classroom"}


def test_ped_arm_is_off_unless_enabled():
    workflow = _ped_workflow(CrossEvalConfig(enabled=False))
    assert workflow._cross_eval_selected("anything") is False


def test_ped_arm_does_not_hand_its_teacher_the_ground_truth_in_free_chat():
    """The tutor arm runs free chat with teacher_show_ground_truth off. Turning
    it on here would make the free_chat column two different setups."""

    workflow = _ped_workflow(
        CrossEvalConfig(
            enabled=True,
            free_chat={"budget": 1},
            classroom={"max_teacher_turns": 1},
            retest={"replays": 1},
            interview={"attempts": 1},
            leak_judges={"enabled": False},
        )
    )
    episode = ce.ClassroomEpisode(problem="2+2?", answer="SECRET_ANSWER")
    episode.add_teacher("try adding")
    actor = FakeActorClient()
    asyncio.run(workflow._run_cross_eval(actor_client=actor, episode=episode))
    free_chat_system = actor.calls[-1][0]["content"]
    assert "SECRET_ANSWER" not in free_chat_system


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
