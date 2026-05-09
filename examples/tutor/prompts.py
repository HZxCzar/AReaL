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
    "question when you are stuck."
)

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "Unused legacy prompt. Tutor answer evaluation is rule-based."
)

DEFAULT_LEAK_CHECK_SYSTEM_PROMPT = (
    "You are a strict answer leakage detector. Decide whether the teacher's "
    "latest visible message directly reveals the ground-truth answer, the final "
    "arithmetic that determines it, or an equivalent final computation. Return "
    "valid JSON only with keys leaked (boolean) and feedback (string)."
)

DEFAULT_GENERATOR_SYSTEM_PROMPT = (
    "You are a careful AIME-style problem generator. Given an original problem "
    "and its ground-truth answer, create one new problem that is structurally "
    "similar but not identical. Return valid JSON only."
)

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You summarize only public tutoring history. You may use only the visible "
    "student and tutor messages provided by the user. Do not infer from hidden "
    "solutions, judge feedback, leak feedback, or any private answer key. Return "
    "valid JSON only."
)

DEFAULT_PROGRESS_JUDGE_SYSTEM_PROMPT = (
    "You are a private math progress judge. Compare the student's current answer "
    "with the previous answer using the ground truth. Judge mathematical progress, "
    "not verbosity. Return valid JSON only."
)

TEACHER_STATE_USER_TEMPLATE = """\
Task:
{{ task }}

Private ground truth / answer key:
{{ ground_truth }}

Public history summary shown to both tutor and student:
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
- Progress label: {{ progress_label }}
- Progress feedback: {{ progress_feedback }}
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

Public history summary:
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
private progress labels, leak feedback, or hidden solution details.

Return JSON only:
{
  "student_progress": "what the student has visibly tried or established",
  "visible_tutor_guidance": "important guidance that was actually shown",
  "student_current_misconception": "likely issue inferred only from visible text",
  "latest_student_state": "concise description of the latest visible student state"
}
"""

PROGRESS_JUDGE_USER_TEMPLATE = """\
Task:
{{ task }}

Ground truth:
{{ ground_truth }}

Previous student answer:
{{ previous_student_answer }}

Tutor guidance shown before the current answer:
{{ tutor_output }}

Current student answer:
{{ current_student_answer }}

Compare whether the current student response is mathematically closer to a correct
solution than the previous response. Consider correctness of intermediate claims,
not verbosity. Do not reward merely restating the problem or adding unrelated work.

Return JSON only:
{
  "label": "improved | same | regressed | unknown",
  "confidence": "high | medium | low",
  "feedback": "short private explanation"
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

TRANSFER_GENERATION_USER_TEMPLATE = """\
Original Task:
{{ task }}

Original Ground Truth:
{{ ground_truth }}

Create one new, self-contained problem that is clearly similar in structure and
solution method, but not a restatement of the original problem.
Return JSON only with this schema:
{
  "task": "new problem statement",
  "ground_truth": "final answer only",
  "similarity_notes": "optional short note"
}
"""

TRANSFER_STUDENT_USER_TEMPLATE = """\
Original task:
{{ original_task }}

Initial student answer:
{{ initial_student_answer or '(empty)' }}

Visible tutoring history:
{% if visible_history %}
{{ visible_history | join('\n') }}
{% else %}
No visible teacher turns before transfer.
{% endif %}

New related task:
{{ transfer_task }}

You are now solving the new related task. Use the tutoring history above as guidance,
but answer the new related task only.
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
