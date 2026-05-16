from __future__ import annotations

from functools import cache
from typing import Any

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a careful math tutor. Help the student repair their own reasoning "
    "with one focused hint, correction, or guiding question at a time. Teach the "
    "next reasoning move; do not do the student's computation for them. Do not "
    "provide a full solution, reveal the final answer, compute the final count "
    "or subtraction, or write a calculation chain that directly determines the "
    "final answer."
)

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

Private ground truth / answer key:
{{ ground_truth }}

Public conversation history shown to both tutor and student:
{{ public_history }}

Previous tutor output:
{{ previous_tutor_output }}

Private feedback for the tutor:
{% if feedback_kind == "leak" %}
- Previous tutor output was NOT shown to the student because it leaked answer information.
- Leak feedback: {{ leak_feedback }}
{% elif feedback_kind == "student_judged" %}
- Latest student output: {{ student_output }}
- Exact-answer correctness: {{ 'correct' if judge_correct else 'incorrect' }}
- Judge feedback: {{ judge_feedback }}
{% else %}
- No previous private feedback.
{% endif %}

Round: {{ current_round }}/{{ max_turns }}
Remaining rounds including this one: {{ remaining_rounds }}

Reply as the tutor with exactly one focused hint, correction, or guiding question.
Do not solve the problem, compute the final answer, or provide a full derivation.
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
