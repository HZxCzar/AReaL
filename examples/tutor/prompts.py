from __future__ import annotations

from functools import cache
from typing import Any

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a careful math tutor. Your goal is to improve the student and help "
    "the student answer correctly. The student's answer may be truncated due to length; if "
    "so, ask them to continue or be brief. The judge extracts the final answer "
    "from the last \\boxed{...} and checks with the answer key after normalization."
)

NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT = """\
For each reply, write a JSON object with two fields:
{
  "reasoning": "your private thinking about what the student needs next",
  "output": "your message shown to the student"
}
Use "reasoning" to think through your tutoring strategy. The student will not see
that field. Put the student-facing guidance in "output"."""

DEFAULT_STUDENT_SYSTEM_PROMPT = (
    "You are a real student solving the task. Use the visible tutoring history "
    "and the teacher's latest feedback naturally. Continue from your previous "
    "visible work when that is the clearest next step. Make a revised answer "
    "attempt when you can, or briefly say what is confusing and ask one short "
    "question when you are stuck. When you make an answer attempt, put your "
    "final answer in \\boxed{}."
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

PAIRWISE_STUDENT_COMPARISON_SYSTEM_PROMPT = (
    "You are a strict math tutoring evaluator. Compare two anonymized student "
    "replies after different private tutor hints. Judge only the student replies "
    "and the visible pre-turn state; do not infer from tutor wording. Return valid "
    "JSON only."
)

NO_VISIBLE_TUTORING_HISTORY = "No visible tutoring history yet."
NO_PREVIOUS_VISIBLE_TUTORING_HISTORY = "No previous visible tutoring history."
EMPTY_PLACEHOLDER = "(empty)"
NONE_PLACEHOLDER = "(none)"
NONE_YET_PLACEHOLDER = "(none yet)"
INITIAL_TEACHER_FEEDBACK_PLACEHOLDER = "(none, produce the first answer attempt)"
LEAK_CHECK_DISABLED_FEEDBACK = "Leak check disabled."
LEAK_CHECK_PENDING_FEEDBACK = "Leak check pending."
LEAK_CHECK_NO_DETAIL_FEEDBACK = (
    "The leak checker did not provide a detailed reason."
)
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

PAIRWISE_STUDENT_COMPARISON_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Visible public history before this turn:
{{ public_history }}

Student's previous answer before this turn:
{{ previous_student_output }}

Student Reply A:
{{ student_reply_a }}

Student Reply B:
{{ student_reply_b }}

Decide which student reply shows better mathematical progress toward the ground truth.
Prefer the reply that is exact-correct, fixes a previous error, advances a valid
intermediate step, or asks a more useful clarifying question. Treat empty,
off-topic, repeated, or regressed work as worse. If both replies are equivalent
or impossible to distinguish, choose "tie".

Return JSON only:
{
  "winner": "A",
  "confidence": 0.0,
  "feedback": "short explanation"
}
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
