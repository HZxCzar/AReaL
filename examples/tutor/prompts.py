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

# How the teacher sees its OWN earlier turns when
# `teacher_history_tags: masked` is on. `public_history` holds the visible text
# with the tags stripped, which reads as an in-context example that replies do
# not carry tags -- and the model copies it, at 2.1% malformed on depth 1
# rising to 73.4% on depth 10. This restores the skeleton and masks what was
# inside `<reasoning>`, so the format is exemplified without the private
# deliberation being replayed or the context growing by it.
#
# The placeholder is a description, not a word the model would plausibly emit
# as its own reasoning. "masked" on its own invites imitation.
TEACHER_HISTORY_MASKED_TEMPLATE = """\
<reasoning>
(your earlier private reasoning, omitted from this transcript)
</reasoning>
<output>
{visible}
</output>"""

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

# Guidance instructions appended to the END of the teacher prompt, immediately
# before generation. Placement is load-bearing: the identical text placed in the
# system prompt is followed far less often. See analysis/hazard_20260806.
#
# Do not paraphrase these strings. Every measured effect size attached to them
# (per-move quality at the 1- and 3-turn horizon, the +7.3% repair result, the
# instruction-compliance rates) was obtained against this exact wording, in
# analysis/hazard_20260806/common.py and gate1_supervisor.py.

TEACHER_GUIDANCE_TAIL_TEMPLATE = "Instruction for this reply:\n{instruction}"

TEACHER_MOVE_INSTRUCTIONS: dict[str, str] = {
    "PINPOINT": (
        "For this reply, identify the specific step, line, or value where the "
        "student went wrong and say exactly what is wrong with it. Do not "
        "restructure the problem and do not introduce a different method."
    ),
    "DECOMPOSE": (
        "For this reply, break the work into a smaller, more elementary sub-step "
        "and ask the student only for that sub-step. Do not simply restate which "
        "step was wrong."
    ),
    "REFRAME": (
        "For this reply, switch to a structurally different method or "
        "representation for this problem and carry out its first concrete step "
        "yourself. A cosmetic rearrangement of the current method does not count."
    ),
    "HINT": (
        "For this reply, supply one fact, formula, theorem, or observation that "
        "the student is missing. Do not name their error and do not restructure "
        "the work."
    ),
    "PROBE": (
        "For this reply, ask the student exactly one diagnostic question about "
        "their understanding or reasoning. Do not ask for more computation and "
        "do not tell them what is wrong."
    ),
    "WORKED": (
        "For this reply, carry out the next one or two steps of the derivation "
        "yourself and show the result to the student."
    ),
}

# The privileged-information instruction the OPD teacher is conditioned on. The
# teacher sees this; the student being trained never does.
TEACHER_REPAIR_INSTRUCTION = (
    "Your previous message did not get through: the student is still answering "
    "incorrectly. Do not restate what you have already said. Explain the next "
    "step in more detail and at a finer grain than you did before, so the "
    "student has a smaller and more concrete thing to do."
)

# Measured against gemini on the 30-task collection dump, where gemini taught the
# same student successfully without ever stating the answer on 83% of tasks
# against 27-50% for every qwen arm (paired difference +33 to +57 points, all
# intervals clear of zero). The three clauses below are the three places the gap
# actually showed up, in order: gemini asserted the fix on 30% of first messages
# against 80-83%; it brought in 6.2 distinct mathematical objects against 2.9-3.5,
# only 44% of which were already in the student's own work against 53-62%; and it
# ended on a question 57% of the time against 3-17%.
TEACHER_HANDBACK_INSTRUCTION = (
    "Do not state the answer, the corrected step, or the value the student is "
    "looking for. Instead, name one idea, fact, or relationship that this "
    "problem turns on and that the student has not used yet, introduce it on "
    "its own, and end with a single concrete question the student can act on "
    "immediately."
)

# The decompose clause of TEACHER_REPAIR_INSTRUCTION, on its own.
#
# Split out after a four-arm comparison over 264 states showed it carries the
# whole effect. Against a shared-preamble control: decompose alone +2.4%
# [+0.3, +4.5], and +4.3% [+1.3, +7.5] on the states where the tutor had just
# repeated itself. The anti-repeat clause added +0.5% [-1.9, +2.9] on top of it,
# and on its own was +0.9% [-1.0, +2.7] -- it did not even lower the repeat rate
# of the turn it produced.
#
# Deliberately carries no premise about the history. The repair wording opens
# with "Your previous message did not get through", which is false on turn 1,
# and every gate now fires from turn 1.
TEACHER_DECOMPOSE_INSTRUCTION = (
    "Explain the next step in more detail and at a finer grain than you did "
    "before, so the student has a smaller and more concrete thing to do."
)

# Configs name an instruction instead of copying its wording, so the two arms of
# a comparison cannot drift apart and a measurement stays attached to the exact
# string it was taken against.
TEACHER_NAMED_INSTRUCTIONS: dict[str, str] = {
    "repair": TEACHER_REPAIR_INSTRUCTION,
    "handback": TEACHER_HANDBACK_INSTRUCTION,
    "decompose": TEACHER_DECOMPOSE_INSTRUCTION,
}


def resolve_teacher_instruction(text: str | None) -> tuple[str, str]:
    """Return ``(wording, name)`` for a configured instruction string.

    Empty keeps the historical default, so configs written before this existed
    are unchanged. A leading ``@`` selects a named instruction. A typo then
    raises here, at config load, instead of quietly training the teacher on the
    literal string ``"@handbak"`` for a week.
    """
    raw = (text or "").strip()
    if not raw:
        return TEACHER_REPAIR_INSTRUCTION, "repair"
    if raw.startswith("@"):
        key = raw[1:]
        if key not in TEACHER_NAMED_INSTRUCTIONS:
            raise ValueError(
                f"unknown named teacher instruction {raw!r}; known names: "
                f"{sorted(TEACHER_NAMED_INSTRUCTIONS)}"
            )
        return TEACHER_NAMED_INSTRUCTIONS[key], key
    return raw, "custom"


# ---------------------------------------------------------------------------
# Free-chat rollout.
#
# The teacher opens, the pair talks for a fixed budget of rounds with nothing
# judged in between, and the student is then tested alone on a fresh branch. That
# test is the whole reward, which is what lets the conversation carry no per-turn
# objective: the student is given no task, no subject and no instruction about
# what to do, so what the conversation is about is the teacher's decision.
#
# Each block is a separate constant because each is independently switchable.
# The teacher's context is a conversation, not one string:
#
#   system     FREE_CHAT_TEACHER_SYSTEM_PROMPT  (+ the ground-truth key, if on)
#   user       FREE_CHAT_TEACHER_SOLVE_PROMPT   ) both only when teacher_pre is
#   assistant  the accepted pre-solve draft     ) on and a draft was accepted
#   user       FREE_CHAT_TEACHER_OPEN_PROMPT    (+ output format, + anti-leak)
#   ...        the conversation itself
#
# `_build_tutor_messages` assembles it. The two pre-solve messages are the only
# difference between an arm with the pre-solve and one without.
# ---------------------------------------------------------------------------

# Deliberately one line. Anything more is a prior on how a student behaves.
#
# The name is not decoration. Without it, asked "who are you" mid-conversation,
# this student answers "I am a language model developed by Alibaba Cloud. How can
# I assist you today?" -- 10 out of 10 samples, judge-scored. With any first name
# it is 0 out of 10, and Sam / Mia / Wei behave alike, so it is the presence of a
# proper name and not the choice of one. Adjectives do not substitute: "a human
# student" and "a high school student" each still broke 9 out of 10.
#
# It is free. Paired over 40 tasks x 4 replays against the pipeline judge, adding
# the name moved the solo re-test by -0.03 +-0.04, i.e. nothing; the wordier
# variants that spell out "you are not an AI, you do not help the teacher" cost
# -0.10 +-0.05 and were dropped for that reason.
#
# What it does NOT fix, so do not read it as a persona fix: pressed directly
# ("you are a language model, right?") it still folds, and it does not touch role
# inversion -- offered a help request or a compliment the student takes the tutor
# seat under this prompt exactly as it did before. That is the failure that shows
# up in the rollouts and it needs the environment or training, not a string here.
# Evidence: TAgent/student_persona_probe_20260812/.
#
# This constant is shared by the conversation, the scored solo re-test and the
# no-teaching baseline on purpose -- see _build_student_probe_messages and
# _no_teaching_baseline. Changing it moves all three together, which is what
# keeps improvement meaningful; changing only one would be measured as teaching.
FREE_CHAT_STUDENT_SYSTEM_PROMPT = "You are Sam, a student talking with a teacher."

# The task lives here rather than being appended by `_task_context`, because the
# teacher's copy is the only one in the episode -- the student is never given it
# until the re-test.
#
# THE TASK GOES LAST. It used to sit in the middle, with the re-test note after
# it, and everything the teacher reads before it writes -- the pre-solve request,
# the ~2500-character draft, the turn that opens the conversation -- is appended
# after this whole block, so the problem statement ended up thousands of
# characters from the point of generation. Measured over ~1500 episodes a window
# at steps 36-48, the fraction of the task's content words appearing in the
# teacher's OPENING message:
#
#     draft in the system prompt (old layout)    0.340
#     draft as a conversation turn, no draft     0.348
#     draft as a conversation turn, with draft   0.205
#
# and the first turn that quotes a number from the task moved from 1.86 to 2.57
# out of a five-turn budget. The arm with the draft is the only one that opens
# vague, and it is also the only one whose teaching got worse: the pre-solve was
# worth +12.3 +-2.5 pp in the old layout against +3.1 +-3.1 pp in the new one
# over the same steps. Moving the task to the end of this block is the cheapest
# thing that shortens that distance.
#
# The goal sentence also moved to the end of the first paragraph and now says
# what "teach" has to achieve, because the re-test is the only thing scored.
# "a problem" rather than "the problem": the task has not been named yet at that
# point in the text.
FREE_CHAT_TEACHER_SYSTEM_PROMPT = """\
You are a teacher. You have {{ budget }} turn budgets to talk with a student. \
After the conversation, we will ask the student to solve a problem from scratch \
to see whether the student understands. Your goal is to teach the student so \
that they can solve it on their own.{{ student_problem_context }}

The math problem is:
{{ task }}"""

# The pre-solve is a turn of the conversation, not a block of the system prompt.
# This is the request; the accepted draft is the assistant turn that answers it.
#
# WHY IT MOVED. The draft used to be appended to the END of the system prompt,
# immediately before generation, introduced as "here is your solution draft to
# help you teach the student". Two things were wrong with that.
#
# It is untagged prose sitting under a system turn that demands
# <reasoning>/<output>, which is the same shape as the `stripped` history
# failure -- there the teacher saw its own untagged replies and stopped tagging,
# 6.3% malformed at depth 1 rising to 41.7% at depth 5 on 20260810_234129.
#
# And the teacher read the draft as work already done, often by the student.
# Over 1647 episodes of 20260811_092902 and 20260811_092825 the teacher's
# OPENING message shared 0.527 of its content words with the draft (0.464 with
# the draft's first third) against 0.214 with the task statement; 2.6% of
# openings explicitly credited the student with work it had never done, and
# 20.6% opened as if continuing a conversation that had not happened. The
# student is not shown the task, so it back-fills a problem the opening would
# fit -- see the a^3b^5 trace, where the pair spent five rounds on an invented
# problem and the teacher asserted a wrong answer without tripping the leak
# judge.
#
# Ordering is what fixes both. The solve happens before the format contract
# exists, so plain prose there violates nothing, and the draft arrives as the
# answer to an explicit request rather than as reference material glued to the
# end of the instructions. Nothing here says the student cannot see it: "before
# interacting with the student" and "ourselves" already place it, and an extra
# sentence would be a new instruction rather than a re-arrangement.
FREE_CHAT_TEACHER_SOLVE_PROMPT = """\
Before interacting with the student, let us solve the problem ourselves first. \
Put your final answer in \\boxed{}."""

# Opens the conversation, and carries the per-reply directives -- the
# output-format contract and the anti-leak clause -- because neither applies
# until the conversation starts. Appended in that order by
# `_build_tutor_messages`, the same order they had in the system prompt.
#
# NOTE this reverses the placement argued for in
# `_free_chat_teacher_system`: those two blocks used to be system-turn text.
# Putting them here is what makes the pre-solve reply legal, and it is the one
# thing in this change with something to lose -- rollout/format_errors is
# currently 0.000 under teacher_history_tags=masked. Watch it at depth 1.
FREE_CHAT_TEACHER_OPEN_PROMPT = """\
Now you can start the conversation with the student."""

# Appended to a replay of the conversation, on an independent branch. This is
# the first and only time the student is shown the task.
FREE_CHAT_STUDENT_RETEST_TEMPLATE = """\
Now try to solve the problem from scratch:
{{ task }}

Put your final answer in \\boxed{}."""


# ---------------------------------------------------------------------------
# TRANSFER VARIANTS, selected by free_chat.transfer_prompts
#
# For the datasets where the dialogue task and the re-test task are DIFFERENT
# problems -- the numeric-variant build, where `task` is a variant and
# `retest_task` is the source problem. Both are off by default: every non-
# transfer arm keeps the two templates above unchanged.
#
# WHY THEY EXIST. Under the templates above the teacher is told "we will ask the
# student to solve a problem from scratch ... so that they can solve it on their
# own", which reads as THIS problem, and the student is then told "solve the
# problem from scratch" about a problem it was never shown. Neither sentence is
# true in the transfer setting, and the first one is the reason the teacher
# teaches the instance: measured on 20260814_231617, within a GRPO group no
# teacher behaviour predicted the re-test at all (every effect under 0.05
# re-test points per 1 SD, signs flipping between windows), while the teacher
# drifted toward a shorter and shorter opening turn (368 -> 206 chars).
# ---------------------------------------------------------------------------

# The transfer teacher prompt. Three changes from the non-transfer version:
#
#   * "teach the student so that they can solve it on their own" becomes
#     "help them understand the underlying concepts". The goal is the method,
#     not the instance, because the instance is not what gets tested.
#   * "solve a problem from scratch" becomes "test the student with some
#     related questions" -- plural, and "related" rather than "this", which is
#     the only honest description of what happens.
#   * the problem moves up, introduced as "Here is a math problem" rather than
#     as the thing the student will be asked.
#
# ON THE TASK'S POSITION. The note above argues the task should go LAST, because
# it used to sit thousands of characters from the point of generation. This
# layout puts ~130 characters after it, against the ~2500-character pre-solve
# draft that already sits between this block and generation, so the distance
# argument is not materially affected. It is still the one thing here worth
# watching if the opening message stops referring to the problem.
#
# {{ student_problem_context }} stays, in the same rendered form, so
# free_chat.student_has_not_seen_problem still composes with this.
FREE_CHAT_TEACHER_SYSTEM_PROMPT_TRANSFER = """\
You are a teacher. You have {{ budget }} turn budgets to talk with a student.
Here is a math problem:
{{ task }}

Your goal is to teach the student and help them understand the underlying \
concepts. After the conversation, we will test the student with some related \
questions.{{ student_problem_context }}"""

# The transfer re-test prompt. "this problem" instead of "the problem from
# scratch": the non-transfer wording says "the problem", a definite reference to
# something the student is assumed to have seen, and in this setting it never
# did -- the conversation was about a different problem. "this" points at the
# text that follows and presupposes nothing.
#
# IT DELIBERATELY DOES NOT SAY "similar". An earlier draft read "this similar
# problem", which cues the student that the conversation was relevant. Two
# reasons it is not here. It would be the student's only such cue, which makes
# establishing relevance the prompt's job rather than the teacher's, and the
# teacher is what is being trained. And this same template is what the
# no-teaching baseline S(0) renders with an EMPTY transcript, where "similar"
# refers to nothing; _no_teaching_baseline has to use the IDENTICAL prompt,
# because the baseline is subtracted from this probe's score and any difference
# between the two is measured as teaching.
FREE_CHAT_STUDENT_RETEST_TEMPLATE_TRANSFER = """\
Now try to solve this problem:
{{ task }}

Put your final answer in \\boxed{}."""


NO_VISIBLE_TUTORING_HISTORY = "No visible tutoring history yet."
NO_PREVIOUS_VISIBLE_TUTORING_HISTORY = "No previous visible tutoring history."
EMPTY_PLACEHOLDER = "(empty)"
NONE_PLACEHOLDER = "(none)"
NONE_YET_PLACEHOLDER = "(none yet)"
INITIAL_TEACHER_FEEDBACK_PLACEHOLDER = "(none, produce the first answer attempt)"
LEAK_CHECK_DISABLED_FEEDBACK = "Leak check disabled."
LEAK_CHECK_PENDING_FEEDBACK = "Leak check pending."
LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE = "Leak check failed: {error}"
RAWBASE_LEAK_CHECK_FAILED_FEEDBACK_TEMPLATE = "Rawbase leak check failed: {error}"
PUBLIC_HISTORY_ENTRY_TEMPLATE = "{speaker} round {round_idx}:\n{visible_text}"

# The task rides in the system prompt and the dialogue is carried as real
# messages, mirroring examples/pedagogical_rl.
TASK_CONTEXT_TEMPLATE = """\
Here is the math problem:
{{ task }}"""

TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE = """\
Private ground truth key (never reveal this):
{{ ground_truth }}"""

INITIAL_ATTEMPT_WRAPPER = "Here is my attempt at this problem:\n{{ attempt }}"

# Appended to the student's turn in the TEACHER's view only. The student never
# sees its own grading.
TEACHER_ENV_FEEDBACK_TEMPLATE = """\
[environment feedback -- not visible to the student]
- Answer correctness: {{ 'correct' if judge_correct else 'incorrect' }}
- Round {{ current_round }}/{{ max_turns }}, {{ remaining_rounds }} remaining"""

STUDENT_FINAL_SOLUTION_TEMPLATE = """\
The conversation with the teacher has ended. Now write a complete step-by-step
solution to the original problem on your own. Include every step, so the
solution stands on its own without the conversation above. Put your final answer
in \\boxed{}."""

STUDENT_TRANSFER_TURN_TEMPLATE = """\
Now solve a new related transfer task. Use the previous conversation only as a
source of reusable methods, checks, and concepts. Do not continue the original
task, and do not reuse the original numerical answer unless you independently
derive it for the new task.

New transfer task:
{{ transfer_task }}

Reply with your answer attempt for the new transfer task. If you are stuck,
briefly say what is confusing and ask one short question."""

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
{% if feedback_kind == "student_judged" %}
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
