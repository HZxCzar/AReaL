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
    "Decide whether the teacher message reveals the correct ground-truth "
    "answer. "
    "Return valid JSON only with keys leaked (boolean) and feedback (string)."
)

DEFAULT_ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are a strict math answer equivalence judge. Compare only the extracted "
    "student answer with the ground-truth answer for the given task. Mark correct "
    "only if they are mathematically equivalent final answers. Return valid JSON "
    "only with key correct (boolean)."
)

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You summarize only public tutoring history. You may use only the visible "
    "student and tutor messages provided by the user. Do not infer from hidden "
    "solutions, judge feedback, leak feedback, or any private answer key. Return "
    "valid JSON only."
)

PAIRWISE_STUDENT_COMPARISON_SYSTEM_PROMPT = (
    "You are a strict math tutoring evaluator. Compare two anonymized student "
    "replies after different private tutor hints. Judge only the student replies "
    "and the visible pre-turn state; do not infer from tutor wording. Return valid "
    "JSON only."
)

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
- Previous tutor output was flagged for leaking answer information.
{% if show_ground_truth %}
- Leak feedback: {{ leak_feedback }}
{% endif %}
{% elif feedback_kind == "student_judged" %}
- Latest student output: {{ student_output }}
- Answer correctness: {{ 'correct' if judge_correct else 'incorrect' }}
- Judge feedback: {{ judge_feedback }}
{% else %}
- No previous private feedback.
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

SUMMARY_USER_TEMPLATE = """\
Existing public history summary:
{{ old_public_summary }}

Previous visible student answer:
{{ previous_student_answer }}

Visible tutor guidance:
{{ tutor_output }}

Current visible student answer:
{{ current_student_answer }}

Create an updated public state summary for future tutor and student turns.
Use only the visible text above. Do not include ground truth, judge correctness,
leak feedback, or hidden solution details.

Return JSON only:
{
  "student_progress": "what the student has visibly tried or established",
  "visible_tutor_guidance": "important guidance that was actually shown",
  "student_current_misconception": "likely issue inferred only from visible text",
  "latest_student_state": "concise description of the latest visible student state"
}
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

Does the teacher message reveal the correct ground-truth answer?

Return JSON only with this schema:
{
  "leaked": false,
  "feedback": ""
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
  "correct": false
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
