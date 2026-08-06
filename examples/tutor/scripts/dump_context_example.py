from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from examples.tutor.core.types import (
    JudgeResult,
    PublicHistoryState,
    StudentTurnState,
    TeacherPreSolveResult,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import (
    ANSWER_JUDGE_USER_TEMPLATE,
    DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT,
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    DEFAULT_TEACHER_SYSTEM_PROMPT,
    INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    render_prompt,
)
from examples.tutor.workflow import (
    ORIGINAL_RETEST_LEVEL,
    StudentGeneralizationAnchor,
    TutorAgentWorkflow,
)


TASK = "If x + 7 = 19, find x."
GROUND_TRUTH = "12"
V1_TASK = "If y + 9 = 24, find y."
V2_TASK = "Solve 3z + 6 = 21 for z."


def judge(correct: bool, extracted: str, feedback: str) -> JudgeResult:
    return JudgeResult(
        raw_output=json.dumps({"correct": correct}),
        correct=correct,
        feedback=feedback,
        parse_error=None,
        raw_result={"extracted_answer": extracted},
    )


def make_workflow() -> TutorAgentWorkflow:
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.max_turns = 10
    workflow.enable_thinking = False
    workflow.teacher_anti_leak_instruction_enabled = True
    workflow.teacher_adaptive_instruction_enabled = False
    workflow.teacher_system_prompt = workflow._resolve_teacher_system_prompt(
        DEFAULT_TEACHER_SYSTEM_PROMPT
    )
    workflow.student_system_prompt = DEFAULT_STUDENT_SYSTEM_PROMPT
    workflow.teacher_show_ground_truth = False
    workflow.teacher_pre_enabled = True
    workflow.leak_penalty_mode = "rawbase"
    workflow.answer_judge_system_prompt = DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT
    return workflow


def answer_judge_messages(extracted_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_prompt(
                ANSWER_JUDGE_USER_TEMPLATE,
                task=TASK,
                ground_truth=GROUND_TRUTH,
                extracted_answer=extracted_answer,
            ),
        },
    ]


def rawbase_messages(teacher_output: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RAWBASE_LEAK_CHECK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_prompt(
                RAWBASE_LEAK_CHECK_USER_TEMPLATE,
                ground_truth=GROUND_TRUTH,
                teacher_action=teacher_output,
            ),
        },
    ]


async def build_dump() -> dict[str, object]:
    workflow = make_workflow()
    private_solution = TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=True,
        raw_output=(
            "Subtract 7 from both sides: x + 7 - 7 = 19 - 7, so x = 12. "
            "Therefore \\boxed{12}."
        ),
        error=None,
        verification_enabled=True,
    )

    initial_output = "I think x = \\boxed{10}."
    initial_judge = judge(False, "10", "The extracted answer is not equivalent.")
    initial_state = StudentTurnState(
        task=TASK,
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output=INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
    )
    initial_student_messages = workflow._build_student_messages(initial_state)

    initial_turns = workflow._initial_conversation(initial_output)
    initial_turns[-1]["env"] = workflow._teacher_env_feedback(initial_judge, 0)
    history0 = PublicHistoryState(
        summary=workflow._build_initial_public_summary(initial_output),
        turn_count=0,
        turns=initial_turns,
    )

    teacher1_visible = (
        "What operation undoes adding 7? Apply that operation to both sides, "
        "and show the equation before simplifying."
    )
    teacher1_raw = (
        "<reasoning>\nThe student guessed. Prompt the inverse operation without "
        "performing the arithmetic for them.\n</reasoning>\n<output>\n"
        f"{teacher1_visible}\n</output>"
    )
    teacher_state1 = TutorTurnState(
        task=TASK,
        ground_truth=GROUND_TRUTH,
        public_history=history0,
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(
            kind="student_judged",
            student_output=initial_output,
            judge_correct=False,
            judge_feedback=initial_judge.feedback,
        ),
        turn_idx=1,
        max_turns=workflow.max_turns,
        teacher_pre_solve_result=private_solution,
        student_reply_before_teacher=initial_output,
    )
    teacher1_messages = workflow._build_tutor_messages(teacher_state1)
    student_state1 = StudentTurnState(
        task=TASK,
        public_history=history0,
        previous_student_output=initial_output,
        latest_tutor_visible_output=teacher1_visible,
    )
    student1_messages = workflow._build_student_messages(student_state1)
    student1_output = "Subtract 7 from both sides, so x = 13. \\boxed{13}"
    student1_judge = judge(False, "13", "The arithmetic is still incorrect.")
    history1 = await workflow._run_public_summary_update(
        old_public_history=history0,
        previous_student_answer=initial_output,
        tutor_visible_output=teacher1_visible,
        current_student_answer=student1_output,
        env_feedback=workflow._teacher_env_feedback(student1_judge, 1),
    )

    teacher2_visible = (
        "Keep the subtraction symbolic first: x + 7 - 7 = 19 - 7. "
        "Now simplify each side carefully and state x."
    )
    teacher2_raw = (
        "<reasoning>\nThe student chose the right inverse operation but made an "
        "arithmetic error. Isolate that exact step.\n</reasoning>\n<output>\n"
        f"{teacher2_visible}\n</output>"
    )
    teacher_state2 = TutorTurnState(
        task=TASK,
        ground_truth=GROUND_TRUTH,
        public_history=history1,
        previous_tutor_visible_output=teacher1_visible,
        previous_feedback=TutorPrivateFeedback(
            kind="student_judged",
            student_output=student1_output,
            judge_correct=False,
            judge_feedback=student1_judge.feedback,
        ),
        turn_idx=2,
        max_turns=workflow.max_turns,
        teacher_pre_solve_result=private_solution,
        student_reply_before_teacher=student1_output,
    )
    teacher2_messages = workflow._build_tutor_messages(teacher_state2)
    student_state2 = StudentTurnState(
        task=TASK,
        public_history=history1,
        previous_student_output=student1_output,
        latest_tutor_visible_output=teacher2_visible,
    )
    student2_messages = workflow._build_student_messages(student_state2)
    student2_output = "x + 7 - 7 = 19 - 7, so x = 12. \\boxed{12}"
    student2_judge = judge(True, "12", "Exact scorer accepted the answer.")
    history2 = await workflow._run_public_summary_update(
        old_public_history=history1,
        previous_student_answer=student1_output,
        tutor_visible_output=teacher2_visible,
        current_student_answer=student2_output,
        env_feedback=workflow._teacher_env_feedback(student2_judge, 2),
    )

    episode = SimpleNamespace(task=TASK, student_prompt_selection=None)
    anchor = StudentGeneralizationAnchor(
        public_history=history2,
        previous_student_output=student2_output,
        teacher_feedback=teacher2_visible,
        reward_turn_idx=2,
    )
    original_messages = workflow._build_student_probe_messages(
        episode_artifact=episode,
        anchor=anchor,
        level=ORIGINAL_RETEST_LEVEL,
        transfer_task=TASK,
    )
    v1_messages = workflow._build_student_probe_messages(
        episode_artifact=episode,
        anchor=anchor,
        level="level1",
        transfer_task=V1_TASK,
    )
    v2_messages = workflow._build_student_probe_messages(
        episode_artifact=episode,
        anchor=anchor,
        level="level2",
        transfer_task=V2_TASK,
    )

    training1 = workflow._clean_tutor_messages(
        SimpleNamespace(tutor_state=teacher_state1, tutor_messages=teacher1_messages)
    )
    training2 = workflow._clean_tutor_messages(
        SimpleNamespace(tutor_state=teacher_state2, tutor_messages=teacher2_messages)
    )

    return {
        "config_semantics": {
            "teacher": "qwen3-8b actor, enable_thinking=false",
            "student": "qwen3-1.7b, enable_thinking=false",
            "max_turns": workflow.max_turns,
            "teacher_pre_solve": "enabled; private draft is in every teacher system message",
            "teacher_show_ground_truth": False,
            "answer_judge": "local exact scorer first; Qwen answer judge only if exact scorer is false",
            "leak_checker": "rawbase",
            "reward_only": "leak checks run post-hoc; no leak feedback enters either dialogue",
            "terminate": "leak check runs before the student call; a leaked teacher turn ends the episode and is not shown to the student",
            "original_retest": "runs after every termination from the last student-visible chat; reward is always 0",
            "transfer_mode": "V1/V2 still obey student_generalize.mode (only_success or always)",
            "probe_branching": "original, V1 and V2 are fresh independent branches of the same completed chat",
        },
        "initial_student": {
            "messages": initial_student_messages,
            "model_output_example": initial_output,
        },
        "round_1": {
            "teacher_messages": teacher1_messages,
            "teacher_raw_output_example": teacher1_raw,
            "student_visible_teacher_output": teacher1_visible,
            "student_messages": student1_messages,
            "student_output_example": student1_output,
            "answer_judge_messages_if_exact_false": answer_judge_messages("13"),
            "answer_judge_output_example": {"correct": False},
            "teacher_env_on_next_round": workflow._teacher_env_feedback(
                student1_judge, 1
            ),
            "judge_feedback_text_enters_dialogue": False,
            "rawbase_leak_judge_messages": rawbase_messages(teacher1_visible),
            "leak_judge_feedback_enters_dialogue": False,
        },
        "round_2": {
            "teacher_messages": teacher2_messages,
            "teacher_raw_output_example": teacher2_raw,
            "student_visible_teacher_output": teacher2_visible,
            "student_messages": student2_messages,
            "student_output_example": student2_output,
            "answer_judge_api_call": "none: local exact scorer accepted \\boxed{12}",
            "rawbase_leak_judge_messages": rawbase_messages(teacher2_visible),
        },
        "post_tutoring_independent_student_branches": {
            "original_retest_messages": original_messages,
            "variant1_messages": v1_messages,
            "variant2_messages": v2_messages,
        },
        "teacher_training_samples": [
            {
                "turn": 1,
                "input_messages_loss_mask": 0,
                "input_messages": training1,
                "target_loss_mask": 1,
                "target": teacher1_raw,
            },
            {
                "turn": 2,
                "input_messages_loss_mask": 0,
                "input_messages": training2,
                "target_loss_mask": 1,
                "target": teacher2_raw,
            },
        ],
        "not_teacher_training_targets": [
            "student outputs",
            "answer-judge outputs",
            "leak-judge outputs",
            "environment feedback",
            "original/V1/V2 probe outputs",
        ],
        "training_note": (
            "response_to_tensordict concatenates teacher input_tokens + teacher "
            "output_tokens and sets loss_mask=[0]*input_len+[1]*output_len. "
            "With no teacher prompt-pool selection, training input equals rollout "
            "input. If a teacher prompt-pool suffix was sampled, only that suffix "
            "is removed from the training system message."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(build_dump()), ensure_ascii=False, indent=2))
