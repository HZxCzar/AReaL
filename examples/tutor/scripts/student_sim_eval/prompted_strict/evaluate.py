"""Keep the shared teacher/prepare/test flow; wrap tutoring student requests only."""

from examples.tutor.scripts.student_sim_eval import evaluate as shared

from .prompts import with_student_instruction


def main():
    shared.main(student_message_adapter=with_student_instruction)


if __name__ == "__main__":
    main()
