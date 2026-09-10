from examples.pedagogical_rl.config import PedagogicalGenerationConfig
from examples.pedagogical_rl.state import ClassroomEpisode


def test_masked_history_changes_only_teacher_input():
    episode = ClassroomEpisode(
        problem="1+1?", answer="2", teacher_output_format="unified_xml"
    )
    raw = "<reasoning>private\ncalculation</reasoning><output>Try adding.</output>"
    episode.add_teacher(raw)
    episode.add_student("My reasoning is private too.")
    original = episode.teacher_messages()
    student = episode.student_messages()
    episode.mask_teacher_history_reasoning = True
    masked = episode.teacher_messages()
    assert masked[0] == original[0]
    assert masked[2] == original[2]
    assert masked[1]["content"] == (
        "<reasoning>\n(your earlier private reasoning, omitted from this transcript)"
        "\n</reasoning><output>Try adding.</output>"
    )
    assert episode.conversation[0]["content"] == raw
    assert episode.student_messages() == student


def test_history_mask_default_preserves_previous_behavior():
    assert PedagogicalGenerationConfig().mask_teacher_history_reasoning is False
    episode = ClassroomEpisode(problem="1+1?", answer="2", teacher_output_format="unified_xml")
    raw = "<reasoning>original</reasoning><output>Hint.</output>"
    episode.add_teacher(raw)
    assert episode.teacher_messages()[1]["content"] == raw
