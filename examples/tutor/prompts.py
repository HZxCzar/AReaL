from __future__ import annotations

from functools import cache
from typing import Any

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a careful math tutor. Your goal is to improve the student and help "
    "the student answer correctly. The student's answer may be truncated due to length; if "
    "so, ask them to continue or be brief. The judge extracts the final answer "
    "from the last \\boxed{...} and checks with the answer key after normalization."
)

DEFAULT_WORLD_MODEL_SYSTEM_PROMPT = (
    "You are a student response predictor. Given the context available to the "
    "teacher and the teacher's latest visible reply, predict the same student's "
    "next visible reply. Output only the predicted student reply."
)

WORLD_MODEL_USER_TEMPLATE = """\
Teacher system prompt:
{{ teacher_system_prompt }}

Teacher context:
{{ teacher_user_prompt }}

Teacher's latest visible reply:
{{ teacher_visible_output }}

Predict the same student's next visible reply. Output only that reply.
"""

NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT = """\
For each reply, use exactly these two tagged sections:
<reasoning>
your private thinking about what the student needs next
</reasoning>
<output>
your message shown to the student
</output>
The student will see only the text inside <output>...</output>. Put all private
thinking inside <reasoning>...</reasoning>. Do not use JSON or Markdown code fences."""

TEACHER_ANTI_LEAK_INSTRUCTION = (
    "Do not reveal the problem's answer or an equivalent expression to the student."
)

TEACHER_ADAPTIVE_INSTRUCTION = (
    "Pay close attention to the student's responses and adapt your teaching strategy "
    "accordingly. Respond flexibly to the student's current behavior instead of "
    "following or repeating a fixed approach."
)

DEFAULT_STUDENT_SYSTEM_PROMPT = (
    "You are a real student solving the task. Use the visible tutoring history "
    "and the teacher's latest feedback naturally. Continue from your previous "
    "visible work when that is the clearest next step. Make a revised answer "
    "attempt when you can, or briefly say what is confusing and ask one short "
    "question when you are stuck. When you make an answer attempt, put your "
    "final answer in \\boxed{}."
)

STUDENT_QUESTION_SYSTEM_PROMPT = (
    "You are a student learning math with help from a teacher."
)

POLARIS_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}. "
)

FILTER_SOLVER_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem independently. "
    "Show your reasoning if useful. Put the final answer in the last "
    "\\boxed{...}. Do not use any answer key or hidden solution."
)

FILTER_SOLVER_USER_TEMPLATE = """\
Task:
{task}

Solve the problem. Put your final answer in \\boxed{{}}.
"""

POLARIS_FILTER_SOLVER_USER_TEMPLATE = """\
{task}

""" + POLARIS_INSTRUCTION.replace("{}", "{{}}")

DEFAULT_LEAK_CHECK_SYSTEM_PROMPT = (
    "You are a strict answer leakage detector. Decide whether the teacher's "
    "latest visible message directly reveals the ground-truth answer, the final "
    "arithmetic that determines it, or an equivalent final computation. Return "
    "valid JSON only with keys leaked (boolean) and feedback (string)."
)

DEFAULT_STAGED_LEAK_CHECK_SYSTEM_PROMPT = (
    "You are a strict answer leakage severity judge. Classify the teacher's "
    "latest visible message into exactly one leakage level. Level 1 means the "
    "teacher directly gives a final answer equivalent to the ground truth; an "
    "incorrect final answer is not Level 1. Level 2 means the teacher does the "
    "computation for the student. Level 3 means the teacher gives the needed "
    "formula but does not compute it. Level 4 means none of those leakage "
    "cases occurred. If multiple levels apply, choose the most severe level, "
    "i.e. the lowest numbered level. Return valid JSON only with keys level "
    "(integer 1-4) and feedback (string)."
)

RAWBASE_LEAK_CHECK_SYSTEM_PROMPT = (
    "Mechanically judge answer containment. Find every value in the teacher "
    "message that is either a number or an expression made only from numbers "
    "and arithmetic operators. Convert LaTeX arithmetic notation, compute "
    "those values exactly, normalize them with the ground truth, and set "
    "leaked=true when any value is equal to the ground truth. Return valid "
    "JSON only with keys leaked (boolean) and feedback (string)."
)

FEEDBACK_LEAK_CHECK_SYSTEM_PROMPT_SUFFIX = (
    "Feedback mode is enabled. Use the existing feedback string as private "
    "tutor-facing guidance: if leakage occurred, briefly explain the leakage "
    "type and how the tutor should revise without quoting the ground-truth "
    "answer or adding a final-answer-equivalent expression. Do not add new "
    "JSON keys."
)

DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are a strict math answer equivalence judge. Compare only the extracted "
    "student answer with the ground-truth answer for the given task. Mark correct "
    "only if they are mathematically equivalent final answers. Return valid JSON "
    "only with key correct (boolean)."
)

DEFAULT_TEACHER_PROGRESS_JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether one reply from a math teacher actually improved a
particular student's state in a tutoring conversation.

Use the task, private answer key, private reference solution, and full conversation
history to compare the student's reply immediately before the target teacher reply
with the student's reply immediately after it. Judge the observed change in the
student, not how helpful the teacher reply sounds in isolation. A mathematically
correct explanation is not progress if the student still shows the same error or no
greater ability to solve the problem afterward. Give credit only for genuine new
progress in the later student reply that is attributable to the target teacher
reply and advances the student toward solving the original task. Completing an
unrelated activity introduced by the teacher is not progress. Completing a
teacher-introduced intermediate step counts only when that step is relevant to the
original solution and resolves or reduces a real obstacle in the student's work.
Base the score on correct understanding or work demonstrated in the later student
reply, not on claims such as "I understand," increased length, a different
presentation of the same result, or superficial activity.

Check the student's mathematical claims against the private answer key and reference
solution rather than trusting either the teacher's or the student's claim that
something is correct. A later reply that merely continues, reformats, or repeats the
same incorrect work is score 0. Score 1 requires a specific correct and relevant
result in the later reply that was not already demonstrated in the earlier reply.
In the reason, identify that concrete before-to-after change. If there is no such
change to identify, use score 0.

Use score 2, rather than score 1, when the later student reply newly reaches the
correct final answer or an essentially complete correct solution matching the answer
key and reference solution. If the later reply keeps the same incorrect final
conclusion as the earlier reply, use score 0 unless it also demonstrates a new
correct intermediate result that materially reduces the remaining work.

Give one integer score:
0 = no genuine progress: the same material error remains with no new correct and
relevant intermediate result, or the student's work gets worse
1 = partial progress: the student demonstrates at least one new correct and relevant
intermediate result or fixes a material misconception, but important errors remain
2 = substantial progress: the student resolves the main obstacle and reaches a
correct or essentially complete solution to the original task

Return valid JSON only with keys score and reason:
{"score": 0, "reason": "brief comparison of the student's state before and after the target teacher reply"}
"""

DEFAULT_STUDENT_REQUEST_JUDGE_SYSTEM_PROMPT = """\
The student has asked the teacher a question. Judge whether the target teacher
reply appropriately answers that question.

Give one integer score:
1 = the teacher appropriately answered the student's question
-1 = the teacher did not appropriately answer the student's question

Return valid JSON only. Keep the reason brief and use plain text without
backslashes:
{"score": 1, "reason": "brief explanation"}
"""

NO_VISIBLE_TUTORING_HISTORY = "No visible tutoring history yet."
NO_PREVIOUS_VISIBLE_TUTORING_HISTORY = "No previous visible tutoring history."
EMPTY_PLACEHOLDER = "(empty)"
NONE_PLACEHOLDER = "(none)"
NONE_YET_PLACEHOLDER = "(none yet)"
INITIAL_TEACHER_FEEDBACK_PLACEHOLDER = "(none, produce the first answer attempt)"
LEAK_CHECK_DISABLED_FEEDBACK = "Leak check disabled."
LEAK_CHECK_PENDING_FEEDBACK = "Leak check pending."
LEAK_CHECK_NO_DETAIL_FEEDBACK = "The leak checker did not provide a detailed reason."
LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE = "Leak check failed: {error}"
RAWBASE_LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE = "Rawbase leak check failed: {error}"
PRIVATE_LEAK_LEVEL_SUFFIX_TEMPLATE = " (leak level {leak_level})"
PRIVATE_LEAK_FEEDBACK_TEMPLATE = (
    "Turn {turn_idx}: tutor output was rejected for answer leakage{level}. "
    "Leak feedback: {feedback}. "
    "This turn and the student's response to it are invalid and were "
    "not added to the student-visible history. Continue from the last "
    "valid public state without using the invalid student response."
)
PUBLIC_HISTORY_ENTRY_TEMPLATE = "{speaker} round {round_idx}:\n{visible_text}"

TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE = """\
Private teacher solution draft hidden from the student:
The following text is the teacher's private solution generated before
tutoring. Treat it as a hidden answer-draft and reasoning reference for the tutor only.
Use it only to keep your teaching path consistent and to check which parts of the student's work are actually valid.

<teacher_private_solution_draft mode="filter_solver">
{{ raw_output }}
</teacher_private_solution_draft>"""

TEACHER_STATE_USER_TEMPLATE = """\
Task:
{{ task }}

{% if show_ground_truth %}
Private ground truth key:
{{ ground_truth }}

{% endif %}
Public conversation history shown to both tutor and student:
{{ public_history }}

Previous tutor output:
{{ previous_tutor_output }}

Private feedback for the tutor:
{% if feedback_kind == "leak" %}
- Latest private event: a previous tutor turn was invalidated for answer leakage.
{% if leak_feedback %}
- Leak feedback: {{ leak_feedback }}
{% endif %}
{% elif feedback_kind == "student_judged" %}
- Latest student output: {{ student_output }}
- Answer correctness: {{ 'correct' if judge_correct else 'incorrect' }}
- Judge feedback: {{ judge_feedback }}
{% else %}
- No previous private feedback.
{% endif %}
{% if leak_history %}

Private invalid-turn timeline for this episode:
{{ leak_history }}
{% endif %}

Round: {{ current_round }}/{{ max_turns }}
Remaining rounds: {{ remaining_rounds }}

Goal: help the student answer correctly.
"""

STUDENT_STATE_USER_TEMPLATE = """\
Task:
{{ task }}

Public conversation history:
{{ public_history }}

Your previous answer:
{{ previous_student_output }}

Latest teacher feedback:
{{ teacher_feedback }}

Reply as the student. Continue from your previous work when useful. If you can
continue, make your next answer attempt. If you are stuck, briefly say what is
confusing and ask one short question.
"""

STUDENT_TRANSFER_USER_TEMPLATE = """\
Original task:
{{ original_task }}

Previous conversation history for the original task:
{{ public_history }}

Your previous answer on the original task:
{{ previous_student_output }}

Latest teacher feedback on the original task:
{{ teacher_feedback }}

Now solve a new related transfer task. Use the previous conversation only as a
source of reusable methods, checks, and concepts. Do not continue the original
task, and do not reuse the original numerical answer unless you independently
derive it for the new task.

New transfer task:
{{ transfer_task }}

Reply as the student with your answer attempt for the new transfer task. If you
are stuck, briefly say what is confusing and ask one short question.
"""

LEAK_CHECK_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Teacher Message:
{{ teacher_action or '(empty)' }}

Return JSON only with this schema:
{
  "leaked": false,
  "feedback": "short explanation"
}
"""

STAGED_LEAK_CHECK_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Teacher Message:
{{ teacher_action or '(empty)' }}

Choose exactly one level:
- Level 1: The teacher directly gives a final answer equivalent to the ground truth.
- Level 2: The teacher does the computation for the student.
- Level 3: The teacher gives the needed formula but does not compute it.
- Level 4: None of the above occurred.
If multiple levels apply, choose the most severe level, i.e. the lowest numbered level.

Return JSON only with this schema:
{
  "level": 4,
  "feedback": "short explanation"
}
"""

RAWBASE_LEAK_CHECK_USER_TEMPLATE = """\
Ground Truth:
{{ ground_truth }}

Teacher Message:
{{ teacher_action or '(empty)' }}

After extracting and computing numeric content from the teacher message, does
any extracted value equal the ground truth?

Return JSON only with this schema:
{
  "leaked": <boolean>,
  "feedback": "<short explanation>"
}
"""

ANSWER_JUDGE_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Extracted Student Answer:
{{ extracted_answer or '(empty)' }}

Decide whether the extracted student answer is mathematically equivalent to the
ground truth as a final answer to the task. Ignore superficial notation
differences, such as including the function name on the left side of an equation,
when the right-hand side is equivalent. Do not use any hidden student reasoning.

Return JSON only with this schema:
{
  "correct": true or false
}
"""

TEACHER_PROGRESS_JUDGE_USER_TEMPLATE = """\
Task:
{{ task }}

Private answer key:
{{ ground_truth }}

Private reference solution:
{{ reference_solution or '(not available)' }}

Conversation history before the target interaction:
{{ public_history or '(none)' }}

Student reply immediately before the target teacher reply:
{{ student_reply_before_teacher or '(empty)' }}

Target teacher reply whose effect is being evaluated:
{{ target_teacher_reply or '(empty)' }}

Student reply immediately after the target teacher reply:
{{ student_reply_after_teacher or '(empty)' }}
"""

STUDENT_REQUEST_JUDGE_USER_TEMPLATE = """\
Original math task:
{{ task }}

Conversation history before the target interaction:
{{ public_history or '(none)' }}

Actual student reply immediately before the target teacher reply:
{{ student_reply_before_teacher or '(empty)' }}

Target teacher reply:
{{ target_teacher_reply or '(empty)' }}
"""

STUDENT_QUESTION_USER_TEMPLATE = """\
Your math problem:
{{ task }}

Your conversation with the teacher:
{{ public_history or '(none)' }}

Your teacher's latest message:
{{ teacher_feedback or '(none)' }}

Your latest response:
{{ student_answer or '(empty)' }}

Based on your conversation with the teacher and your latest response:
{{ question_instruction }}

Output only your question.
"""


@cache
def _get_template_env():
    from jinja2 import Environment, StrictUndefined

    return Environment(
        autoescape=False,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )


def render_prompt(template: str, **context: Any) -> str:
    return _get_template_env().from_string(template).render(**context).strip()
