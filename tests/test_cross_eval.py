"""The shared cross-eval instrument, exercised with fake models.

Everything here is protocol shape and bookkeeping: which messages each protocol
builds, which scorer runs on which transcript, and what happens when a call
fails. No endpoint is touched, so this runs on the CPU host and is the check to
run after any edit to examples/pedagogical_rl/cross_eval.py.
"""

from __future__ import annotations

import asyncio

import pytest

from examples.pedagogical_rl import cross_eval as ce
from examples.pedagogical_rl.state import ConversationType


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text or "") // 4 + 1))


class Recorder:
    """Fake teacher/student/judge that records every call it receives."""

    def __init__(
        self,
        *,
        teacher_text: str = "<reasoning>r</reasoning><output>lesson</output>",
        student_text: str = "the answer is \\boxed{7}",
        judge_text: str = '{"leaked": false, "feedback": "clean"}',
    ) -> None:
        self.teacher_text = teacher_text
        self.student_text = student_text
        self.judge_text = judge_text
        self.teacher_calls: list[list[dict[str, str]]] = []
        self.student_calls: list[tuple[list[dict[str, str]], int, str]] = []
        self.judge_calls: list[list[dict[str, str]]] = []

    async def teacher(self, messages, *, rid_prefix=""):
        self.teacher_calls.append(messages)
        return self.teacher_text

    async def student(self, messages, *, n=1, max_tokens=None, rid_prefix="", timeout=None):
        self.student_calls.append((messages, n, rid_prefix))
        return [self.student_text] * n

    async def judge(self, messages, *, rid_prefix=""):
        self.judge_calls.append(messages)
        return self.judge_text


async def always_correct(*, task, ground_truth, answer):
    return True


async def never_correct(*, task, ground_truth, answer):
    return False


def test_free_chat_dialogue_runs_exactly_the_budget():
    rec = Recorder()
    result = asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=3),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    assert result.teacher_turns == 3
    assert [turn["role"] for turn in result.transcript] == [
        "teacher",
        "student",
        "teacher",
        "student",
        "teacher",
        "student",
    ]
    assert result.format_errors == 0
    # Only the tagged <output> half reaches the student.
    assert result.transcript[0]["content"] == "lesson"


def test_free_chat_student_is_never_shown_the_task():
    """The whole point of the tutor protocol: the student is not told what this
    is about. If the task leaks into the student's prompt the re-test stops
    measuring teaching."""

    rec = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="UNIQUE_TASK_TOKEN",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=2),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    for messages, _n, _rid in rec.student_calls:
        assert "UNIQUE_TASK_TOKEN" not in "".join(m["content"] for m in messages)
    # The teacher does get it.
    assert "UNIQUE_TASK_TOKEN" in rec.teacher_calls[0][0]["content"]


def test_masked_history_hands_the_teacher_its_own_tags_back():
    """Under 'stripped' the teacher reads its own untagged replies and imitates
    them -- 6.3% malformed at depth 1 rising to 41.7% at depth 5. math/0810 runs
    'masked' for that reason, and the crossed dialogue has to do the same or the
    free_chat column compares two different settings."""

    rec = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=2, teacher_history_tags="masked"),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    own_turn = next(
        m for m in rec.teacher_calls[-1] if m["role"] == "assistant"
    )
    assert "<reasoning>" in own_turn["content"]
    assert "<output>" in own_turn["content"]
    assert "lesson" in own_turn["content"]

    rec_stripped = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=2, teacher_history_tags="stripped"),
            teacher_call=rec_stripped.teacher,
            student_call=rec_stripped.student,
        )
    )
    own_turn = next(
        m for m in rec_stripped.teacher_calls[-1] if m["role"] == "assistant"
    )
    assert own_turn["content"] == "lesson"


def test_the_student_never_sees_the_teachers_tag_skeleton():
    """Masking is the teacher's own view. The student is never shown the tag
    protocol, so its side of the same dialogue stays plain text."""

    rec = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=2, teacher_history_tags="masked"),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    for messages, _n, _rid in rec.student_calls:
        assert "<reasoning>" not in "".join(m["content"] for m in messages)


def test_free_chat_counts_a_malformed_turn_and_keeps_going():
    rec = Recorder(teacher_text="no tags at all")
    result = asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=2),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    assert result.format_errors == 2
    assert result.teacher_turns == 2
    assert result.transcript[0]["content"] == ""


def test_free_chat_preamble_carries_the_draft_when_there_is_one():
    rec = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=1, teacher_draft="the answer is 4"),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    roles = [m["role"] for m in rec.teacher_calls[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert rec.teacher_calls[0][2]["content"] == "the answer is 4"

    rec_nodraft = Recorder()
    asyncio.run(
        ce.run_free_chat_dialogue(
            task="2+2?",
            ground_truth="4",
            spec=ce.FreeChatSpec(budget=1),
            teacher_call=rec_nodraft.teacher,
            student_call=rec_nodraft.student,
        )
    )
    assert [m["role"] for m in rec_nodraft.teacher_calls[0]] == ["system", "user"]


def test_classroom_dialogue_follows_their_state_machine():
    rec = Recorder(teacher_text="think about factoring")
    result, episode = asyncio.run(
        ce.run_classroom_dialogue(
            problem="2+2?",
            answer="4",
            spec=ce.ClassroomSpec(
                max_teacher_turns=3, forced_type=ConversationType.GUIDED
            ),
            tokenizer=FakeTokenizer(),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    assert episode.conversation_type is ConversationType.GUIDED
    assert result.initial_attempt == ""
    assert result.teacher_turns == 3
    assert result.transcript[0]["role"] == "teacher"


def test_classroom_attempted_opens_with_a_student_attempt():
    rec = Recorder()
    result, episode = asyncio.run(
        ce.run_classroom_dialogue(
            problem="2+2?",
            answer="4",
            spec=ce.ClassroomSpec(
                max_teacher_turns=2, forced_type=ConversationType.ATTEMPTED
            ),
            tokenizer=FakeTokenizer(),
            teacher_call=rec.teacher,
            student_call=rec.student,
        )
    )
    assert result.initial_attempt
    assert result.transcript[0]["role"] == "student"
    assert episode.teacher_turns == 2


def test_retest_shows_the_task_only_in_the_final_turn():
    rec = Recorder()
    transcript = [
        {"role": "teacher", "content": "start with the square"},
        {"role": "student", "content": "ok"},
    ]
    result = asyncio.run(
        ce.score_retest(
            transcript=transcript,
            task="UNIQUE_TASK_TOKEN",
            ground_truth="7",
            spec=ce.RetestSpec(replays=4),
            student_call=rec.student,
            answer_judge=always_correct,
        )
    )
    assert result.complete and result.score == 1.0 and result.attempts == 4
    # Four independent requests, not one n-choice call: this is the tutor arm's
    # own reward, and it has to stay the same quantity.
    assert len(rec.student_calls) == 4
    assert all(n == 1 for _messages, n, _rid in rec.student_calls)
    messages = rec.student_calls[0][0]
    assert "UNIQUE_TASK_TOKEN" not in messages[0]["content"]
    assert "UNIQUE_TASK_TOKEN" in messages[-1]["content"]
    # The teacher speaks as user, the student's own turns as assistant.
    assert messages[1] == {"role": "user", "content": "start with the square"}
    assert messages[2] == {"role": "assistant", "content": "ok"}


def test_retest_scores_the_fraction_the_judge_accepts():
    rec = Recorder()
    calls = {"n": 0}

    async def judge_alternating(*, task, ground_truth, answer):
        calls["n"] += 1
        return calls["n"] % 2 == 0

    result = asyncio.run(
        ce.score_retest(
            transcript=[{"role": "teacher", "content": "hint"}],
            task="t",
            ground_truth="7",
            spec=ce.RetestSpec(replays=4),
            student_call=rec.student,
            answer_judge=judge_alternating,
        )
    )
    assert result.score == 0.5
    assert result.any_correct == 1.0


def test_interview_uses_one_n_choice_request_and_their_prompt():
    rec = Recorder()
    result = asyncio.run(
        ce.score_interview(
            transcript=[
                {"role": "teacher", "content": "consider the discriminant"},
                {"role": "student", "content": "got it"},
            ],
            task="UNIQUE_TASK_TOKEN",
            ground_truth="7",
            spec=ce.InterviewSpec(attempts=8),
            student_call=rec.student,
        )
    )
    assert result.complete and result.attempts == 8
    assert result.score == 1.0  # \boxed{7} == "7"
    assert len(rec.student_calls) == 1
    assert rec.student_calls[0][1] == 8
    # Their student prompt DOES carry the problem. That asymmetry against the
    # re-test is a real difference between the measurements, kept on purpose.
    assert "UNIQUE_TASK_TOKEN" in rec.student_calls[0][0][0]["content"]


def test_interview_exact_match_is_stricter_than_the_math_scorer():
    """Their scorer is a lowercased string compare on the last boxed span, so an
    equivalent form scores wrong. Both numbers are reported for that reason."""

    rec = Recorder(student_text="so x = \\boxed{\\frac{1}{2}}")
    result = asyncio.run(
        ce.score_interview(
            transcript=[{"role": "teacher", "content": "hint"}],
            task="t",
            ground_truth="1/2",
            spec=ce.InterviewSpec(attempts=2),
            student_call=rec.student,
        )
    )
    assert result.score == 0.0
    assert result.score_math == 1.0


def test_an_incomplete_cell_scores_zero_and_is_flagged():
    async def dead_student(messages, *, n=1, max_tokens=None, rid_prefix="", timeout=None):
        raise RuntimeError("endpoint down")

    result = asyncio.run(
        ce.score_interview(
            transcript=[{"role": "teacher", "content": "hint"}],
            task="t",
            ground_truth="7",
            spec=ce.InterviewSpec(attempts=8),
            student_call=dead_student,
        )
    )
    assert not result.complete and result.score == 0.0
    metrics = ce.cell_metrics(
        protocol="free_chat", scorer="interview", result=result
    )
    assert metrics["xeval/free_chat/interview/incomplete"] == 1.0
    assert metrics["xeval/free_chat/interview/success"] == 0.0


def test_cross_eval_fills_all_four_cells_and_reuses_the_own_transcript():
    rec = Recorder()
    own = [
        {"role": "teacher", "content": "lesson"},
        {"role": "student", "content": "ok"},
    ]
    metrics, details = asyncio.run(
        ce.run_cross_eval(
            task="2+2?",
            ground_truth="7",
            own_protocol="free_chat",
            own_transcript=own,
            teacher_call=rec.teacher,
            student_call=rec.student,
            judge_call=rec.judge,
            answer_judge=always_correct,
            tokenizer=FakeTokenizer(),
            free_chat=ce.FreeChatSpec(budget=2),
            classroom=ce.ClassroomSpec(
                max_teacher_turns=2, forced_type=ConversationType.GUIDED
            ),
            retest=ce.RetestSpec(replays=2),
            interview=ce.InterviewSpec(attempts=2),
            leak_judges=ce.LeakJudgeSpec(native_attempts=1),
        )
    )
    for protocol in ("free_chat", "classroom"):
        for scorer in ("retest", "interview"):
            assert f"xeval/{protocol}/{scorer}/success" in metrics
        assert f"xeval/{protocol}/leak/turn" in metrics
        assert f"xeval/{protocol}/leak/native" in metrics
    # The free-chat transcript was handed in, so only the classroom protocol
    # generated teacher turns. Re-running our own would cost a whole dialogue
    # and measure a second sample of the same thing.
    assert metrics["xeval/free_chat/turns"] == 1.0
    assert metrics["xeval/classroom/turns"] == 2.0
    assert len(details["free_chat"]["transcript"]) == 2


def test_cross_eval_from_the_ped_arm_is_the_mirror_image():
    rec = Recorder()
    own = [
        {"role": "student", "content": "I tried x=3"},
        {"role": "teacher", "content": "close, check the sign"},
    ]
    metrics, _details = asyncio.run(
        ce.run_cross_eval(
            task="2+2?",
            ground_truth="7",
            own_protocol="classroom",
            own_transcript=own,
            own_initial_attempt="I tried x=3",
            teacher_call=rec.teacher,
            student_call=rec.student,
            judge_call=rec.judge,
            answer_judge=always_correct,
            tokenizer=FakeTokenizer(),
            free_chat=ce.FreeChatSpec(budget=2),
            classroom=ce.ClassroomSpec(max_teacher_turns=2),
            retest=ce.RetestSpec(replays=2),
            interview=ce.InterviewSpec(attempts=2),
        )
    )
    assert metrics["xeval/classroom/turns"] == 1.0
    assert metrics["xeval/free_chat/turns"] == 2.0
    for protocol in ("free_chat", "classroom"):
        for scorer in ("retest", "interview"):
            assert f"xeval/{protocol}/{scorer}/success" in metrics


def test_cross_eval_skips_scoring_an_empty_dialogue():
    """A dead teacher leaves no transcript. Scoring it would hand the student a
    bare problem and file the result as teaching."""

    async def dead_teacher(messages, *, rid_prefix=""):
        raise RuntimeError("actor down")

    rec = Recorder()
    metrics, _details = asyncio.run(
        ce.run_cross_eval(
            task="2+2?",
            ground_truth="7",
            own_protocol="free_chat",
            own_transcript=[{"role": "teacher", "content": "lesson"}],
            teacher_call=dead_teacher,
            student_call=rec.student,
            judge_call=rec.judge,
            answer_judge=always_correct,
            tokenizer=FakeTokenizer(),
            free_chat=ce.FreeChatSpec(budget=2),
            classroom=ce.ClassroomSpec(max_teacher_turns=2),
            retest=ce.RetestSpec(replays=1),
            interview=ce.InterviewSpec(attempts=1),
        )
    )
    assert metrics["xeval/classroom/retest/incomplete"] == 1.0
    assert metrics["xeval/classroom/interview/incomplete"] == 1.0
    assert "xeval/classroom/retest/success" not in metrics
    assert metrics["xeval/free_chat/retest/success"] == 1.0


def test_success_and_leak_is_only_logged_on_a_successful_rollout():
    rec = Recorder(judge_text='{"leaked": true, "feedback": "gave it away"}')
    own = [{"role": "teacher", "content": "the answer is 7"}]
    metrics, _ = asyncio.run(
        ce.run_cross_eval(
            task="2+2?",
            ground_truth="7",
            own_protocol="free_chat",
            own_transcript=own,
            teacher_call=rec.teacher,
            student_call=rec.student,
            judge_call=rec.judge,
            answer_judge=always_correct,
            tokenizer=FakeTokenizer(),
            free_chat=ce.FreeChatSpec(budget=1),
            classroom=ce.ClassroomSpec(max_teacher_turns=1),
            retest=ce.RetestSpec(replays=1),
            interview=ce.InterviewSpec(attempts=1),
            leak_judges=ce.LeakJudgeSpec(native_attempts=1),
            run_other_protocol=False,
        )
    )
    assert metrics["xeval/free_chat/leak/turn"] == 1.0
    assert metrics["xeval/free_chat/retest/success_and_leak/turn"] == 1.0

    failed = asyncio.run(
        ce.run_cross_eval(
            task="2+2?",
            ground_truth="7",
            own_protocol="free_chat",
            own_transcript=own,
            teacher_call=rec.teacher,
            student_call=rec.student,
            judge_call=rec.judge,
            answer_judge=never_correct,
            tokenizer=FakeTokenizer(),
            free_chat=ce.FreeChatSpec(budget=1),
            classroom=ce.ClassroomSpec(max_teacher_turns=1),
            retest=ce.RetestSpec(replays=1),
            interview=ce.InterviewSpec(attempts=1),
            leak_judges=ce.LeakJudgeSpec(native_attempts=1),
            run_other_protocol=False,
        )
    )[0]
    assert "xeval/free_chat/retest/success_and_leak/turn" not in failed


def test_both_arms_emit_the_same_metric_names():
    """The comparison is read off one set of series. If the two arms ever emit
    different names for the same cell, the table silently stops being paired."""

    rec = Recorder()
    common = dict(
        task="2+2?",
        ground_truth="7",
        teacher_call=rec.teacher,
        student_call=rec.student,
        judge_call=rec.judge,
        answer_judge=always_correct,
        tokenizer=FakeTokenizer(),
        free_chat=ce.FreeChatSpec(budget=1),
        classroom=ce.ClassroomSpec(
            max_teacher_turns=1, forced_type=ConversationType.GUIDED
        ),
        retest=ce.RetestSpec(replays=1),
        interview=ce.InterviewSpec(attempts=1),
    )
    ours, _ = asyncio.run(
        ce.run_cross_eval(
            own_protocol="free_chat",
            own_transcript=[{"role": "teacher", "content": "lesson"}],
            **common,
        )
    )
    theirs, _ = asyncio.run(
        ce.run_cross_eval(
            own_protocol="classroom",
            own_transcript=[{"role": "teacher", "content": "lesson"}],
            **common,
        )
    )
    assert set(ours) == set(theirs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
