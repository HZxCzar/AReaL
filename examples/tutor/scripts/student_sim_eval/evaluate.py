"""Process-local free-chat prompt adapter; all evaluation stays in the shared API engine."""

import argparse
import json
from pathlib import Path

import yaml

from examples.tutor import evaluate_teacher_api
from examples.tutor.scripts import evaluate_api_teacher
from examples.tutor.workflow import TutorAgentWorkflow


def with_preference(original):
    """Use one prompt in dialogue, no-teaching baseline, and independent retests."""

    def prompts(self, *args, **kwargs):
        system, retest = original(self, *args, **kwargs)
        suffix = self.student_system_prompt.strip()
        return (f"{system}\n\n{suffix}" if suffix else system), retest

    return prompts


def with_strategy(description):
    """Build a fresh request; the final teaching reminder is never stored in history."""

    def messages(self, tutor_state, **kwargs):
        conversation = self._render_conversation(
            tutor_state.public_history.turns, speaker="teacher"
        )
        reminder = "Teaching the student."
        if description.strip():
            reminder = (
                "Teaching the student using the following method.\n\n"
                + description.strip()
            )
        if getattr(self, "teacher_anti_leak_instruction_enabled", False):
            reminder += "\n\nPlease do not directly reveal the final answer or an equivalent expression."
        return [
            {
                "role": "system",
                "content": self._free_chat_teacher_system(
                    tutor_state.task, tutor_state.ground_truth
                ),
            },
            *conversation,
            {"role": "user", "content": reminder},
        ]

    return messages


def select_questions(dataset, selected):
    """Select by stable item identity in frozen order, before tutoring starts."""
    if "id" not in dataset.column_names:
        raise ValueError("Curated evaluation requires a stable dataset id column")
    indices = {}
    for index, item in enumerate(dataset["id"]):
        key = str(item)
        if key in indices:
            raise ValueError(f"Ambiguous duplicate dataset ID: {key}")
        indices[key] = index
    missing = set(selected) - indices.keys()
    if missing:
        raise ValueError(
            f"Reviewed question IDs missing from dataset: {sorted(missing)}"
        )
    return dataset.select([indices[item] for item in selected])


def main(*, student_message_adapter=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    protocol = yaml.safe_load(Path(args.config).read_text())
    selected = json.loads(Path(args.config).with_name("question_ids.json").read_text())
    original = TutorAgentWorkflow._free_chat_student_prompts
    original_student_messages = TutorAgentWorkflow._build_student_messages
    original_teacher = TutorAgentWorkflow._build_tutor_messages
    original_dataset = evaluate_api_teacher.prepare_test_dataset

    def prepare_dataset(*positional, **kwargs):
        # Load the complete eligible split, not its first --limit rows, then
        # select the reviewed IDs. Our generated protocol has one student axis.
        kwargs["limit"] = 0
        return select_questions(original_dataset(*positional, **kwargs), selected)

    try:
        evaluate_api_teacher.prepare_test_dataset = prepare_dataset
        if student_message_adapter is None:
            TutorAgentWorkflow._free_chat_student_prompts = with_preference(original)
        else:
            TutorAgentWorkflow._build_student_messages = student_message_adapter(
                original_student_messages, protocol
            )
        TutorAgentWorkflow._build_tutor_messages = with_strategy(
            protocol["teacher_system_prompt"]
        )
        evaluate_teacher_api.main()
    finally:
        TutorAgentWorkflow._free_chat_student_prompts = original
        TutorAgentWorkflow._build_student_messages = original_student_messages
        TutorAgentWorkflow._build_tutor_messages = original_teacher
        evaluate_api_teacher.prepare_test_dataset = original_dataset


if __name__ == "__main__":
    main()
