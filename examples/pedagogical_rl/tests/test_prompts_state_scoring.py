from __future__ import annotations

from examples.pedagogical_rl.prompts import SOURCE_PROMPT_SHA256, prompt_hashes
from examples.pedagogical_rl.scoring import extract_boxed_answer, native_answer_correct
from examples.pedagogical_rl.state import (
    ClassroomEpisode,
    ConversationType,
    student_visible_text,
)


class _Tokenizer:
    def encode(self, text: str, **_kwargs):
        return list(text)


def test_embedded_prompts_match_pedagogicalrl_source_hashes():
    """The copied templates must remain byte-for-byte identical to the source."""

    assert prompt_hashes() == SOURCE_PROMPT_SHA256


def test_student_visible_text_matches_balanced_only_hiding_rule():
    """Balanced thinking is hidden, while an unclosed block remains visible."""

    assert student_visible_text("a<think>secret</think>b<end_of_conversation>") == "ab"
    assert student_visible_text("a<think>unclosed secret") == "a<think>unclosed secret"


def test_classroom_perspectives_mask_only_student_view():
    """Teacher keeps private thinking; student/native judges see the public text."""

    episode = ClassroomEpisode(
        problem="2+3?",
        answer="5",
        forced_type=ConversationType.ATTEMPTED,
        forced_student_name="Alex",
    )
    episode.add_initial_attempt("I think 4")
    episode.add_teacher("<think>The answer is 5</think>Try adding once more.")

    assert episode.teacher_messages()[-1]["content"].startswith("<think>")
    assert episode.student_messages()[-1]["content"] == "Try adding once more."
    assert episode.hidden_conversation()[-1]["content"] == "Try adding once more."


def test_dialogue_stops_after_tenth_teacher_turn():
    """Ten teacher messages, not ten combined messages, define the turn limit."""

    episode = ClassroomEpisode(
        problem="p",
        answer="a",
        forced_type=ConversationType.GUIDED,
        forced_student_name="Alex",
    )
    for index in range(10):
        episode.add_teacher(f"teacher {index}")
        if index < 9:
            episode.add_student(f"student {index}")

    assert episode.should_stop_dialogue(
        tokenizer=_Tokenizer(),
        max_teacher_turns=10,
        max_tokens_in_conversation=10000,
    )
    assert episode.termination_reason == "max_turns"


def test_native_answer_reward_uses_last_balanced_box_exactly():
    """The native Answer reward is strict string equality after box extraction."""

    assert (
        extract_boxed_answer(r"first \boxed{1}, last \boxed{\frac{1}{2}}")
        == r"\frac{1}{2}"
    )
    assert native_answer_correct(r"work \boxed{5}", "5")
    assert not native_answer_correct(r"work \boxed{5.0}", "5")
    assert not native_answer_correct("5", "5")
