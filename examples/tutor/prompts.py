from __future__ import annotations

from functools import cache
from typing import Any

DEFAULT_TEACHER_SYSTEM_PROMPT = (
    "You are a careful math tutor. Help the student repair their "
    "own reasoning with one focused hint, correction, or guiding question at a "
    "time. Teach the next reasoning move; do not do the student's computation "
    "for them. Do not provide a full solution, reveal the final answer, compute "
    "the final count/subtraction, or write a calculation chain that directly "
    "determines the final answer. If the student's reply is too long, looping, "
    "or appears truncated, tell them to continue from the last valid step, "
    "answer compactly, and avoid restarting the whole solution."
)

DEFAULT_STUDENT_SYSTEM_PROMPT = (
    "You are a real student solving the task. Use the teacher's latest feedback "
    "naturally. You may continue from your previous visible work when that is "
    "the clearest next step; you do not need to restart the whole solution every "
    "turn. Make a revised answer attempt when you can, or briefly say what you "
    "do not understand and ask a short question when you are stuck. If the "
    "teacher says your answer was too long or truncated, resume from the last "
    "valid step when possible and keep the next reply compact."
)

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "Unused legacy prompt. Tutor and transfer evaluation are rule-based."
)

DEFAULT_LEAK_CHECK_SYSTEM_PROMPT = (
    "You are a strict answer leakage detector. Decide whether the teacher's latest "
    "message directly reveals the ground-truth answer, the final arithmetic that "
    "determines it, or an equivalent final computation. Return valid JSON only "
    "with keys leaked (boolean) and feedback (string)."
)

DEFAULT_GENERATOR_SYSTEM_PROMPT = (
    "You are a careful AIME-style problem generator. Given an original problem and "
    "its ground-truth answer, create one new problem that is structurally similar "
    "but not identical. Solve your generated problem before returning it. The "
    "generated problem must be internally consistent, and ground_truth must be "
    "the final answer for the generated problem. Return valid JSON only, with no markdown "
    "fences, using keys task, ground_truth, and optional similarity_notes. The "
    "ground_truth value should be a string."
)

TEACHER_INITIAL_USER_TEMPLATE = """\
Task:
{{ task }}

Ground Truth:
{{ ground_truth }}

Turn 0:
- Student initial answer: {{ initial_student_answer or '(empty)' }}
- Initial judge result: {{ 'correct' if initial_correct else 'incorrect' }}
- Initial judge feedback: {{ initial_feedback }}
- Current round: {{ current_round }}/{{ max_turns }}
- Remaining rounds: {{ remaining_rounds }}
- Pre-solved: {{ pre_solved }}
- Note: {{ pre_solved_note }}

Reply as a tutor with exactly one focused hint, correction, or guiding question.
Do not solve the problem, compute the final answer, or provide a full derivation.
"""

TEACHER_FOLLOWUP_USER_TEMPLATE = """\
Turn {{ previous_round_idx }} update:
- Current round: {{ current_round }}/{{ max_turns }}
- Remaining rounds: {{ remaining_rounds }}
- Pre-solved: {{ pre_solved }}
{% if leak_detected %}
- Env feedback: {{ leak_feedback }}
- Latest judge result: {{ 'correct' if latest_correct else 'incorrect' }}
- Latest judge feedback: {{ latest_feedback }}
{% else %}
- Student reply: {{ student_answer or '(empty)' }}
- Judge result: {{ 'correct' if judge_correct else 'incorrect' }}
- Judge feedback: {{ judge_feedback or '(empty)' }}
{% if transfer_triggered %}
- Transfer success: {{ transfer_success }}
- Transfer judge feedback: {{ transfer_judge_feedback or '(empty)' }}
{% endif %}
{% if termination_feedback %}
- Budget feedback: {{ termination_feedback }}
{% endif %}
{% endif %}
Reply as a tutor with exactly one focused hint, correction, or guiding question. Do not solve the problem, compute the final answer, or provide a full derivation.
"""

STUDENT_USER_TEMPLATE = """\
Task:
{{ task }}

Visible student history:
{% if visible_history %}
{{ visible_history | join('\n') }}
{% else %}
No previous visible turns.
{% endif %}

Current teacher feedback:
{{ teacher_feedback }}

Reply as the student. If you can continue, make your next answer attempt. If you are stuck, briefly say what is confusing and ask one short question.
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

Create one new, self-contained problem that is clearly similar in structure and solution method, but not a restatement of the original problem.
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
