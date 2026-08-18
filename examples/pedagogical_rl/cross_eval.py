"""One instrument for the tutor / PedagogicalRL head-to-head.

Two dialogue protocols and two scorers, crossed. Each arm trains its own way and
then, at evaluation, runs BOTH protocols with its own teacher and scores BOTH
resulting transcripts with BOTH scorers -- four cells per arm, eight across the
pair, all under identically named metrics.

    xeval/<protocol>/<scorer>/success

        protocol  free_chat   the tutor rollout: the teacher opens, the pair
                              talks for a fixed budget, nothing is judged in
                              between and the student is never shown the task
                  classroom   PedagogicalRL's GUIDED/ATTEMPTED state machine,
                              up to max_teacher_turns, student replies in-band

        scorer    retest      the tutor measurement: replay the transcript on a
                              fresh branch, show the task for the first time,
                              `replays` independent solo attempts, LLM answer
                              judge, score = fraction correct
                  interview   PedagogicalRL's measurement: the student re-reads
                              the dialogue in their student-perspective form,
                              `attempts` solutions from ONE n-choice request,
                              scored by exact match on the last boxed answer

WHY IT LIVES HERE. The alternative is each arm implementing the other's
protocol, which is two copies of one measurement and they drift. This follows
the precedent already set by judges.py: the shared instrument sits in the
pedagogical_rl package and the tutor workflow imports it.

WHAT IS DELIBERATELY NOT SHARED. Everything about how each protocol talks to a
model is passed in as a plain async callable, so an arm keeps its own client,
its own concurrency limit and its own retry behaviour. The protocol and the
scoring are what must be identical; the transport must not be.

ONE ASYMMETRY IS PRESERVED ON PURPOSE. Their student's system prompt contains
the problem statement (SIMPLE_STUDENT_PROMPT takes `problem=`); ours does not --
under free chat the student first sees the task in the final turn. That is a
real difference between the two measurements, not an artefact, so `interview`
keeps their prompt and `retest` keeps ours even when both score the same
transcript.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from examples.pedagogical_rl.judges import LEAK_RULE, run_whole_dialogue_judge
from examples.pedagogical_rl.prompts import (
    INITIAL_ATTEMPT_WRAPPER,
    SIMPLE_STUDENT_PROMPT,
    STUDENT_FINAL_PROMPT,
)
from examples.pedagogical_rl.prompts import render as render_native
from examples.pedagogical_rl.scoring import native_answer_correct
from examples.pedagogical_rl.state import (
    ClassroomEpisode,
    ConversationType,
    student_visible_text,
)
from examples.tutor.core.math import score_math_answer
from examples.tutor.core.parsers import parse_leak_check_result
from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.prompts import (
    FREE_CHAT_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_STUDENT_SYSTEM_PROMPT,
    FREE_CHAT_TEACHER_OPEN_PROMPT,
    FREE_CHAT_TEACHER_SOLVE_PROMPT,
    FREE_CHAT_TEACHER_SYSTEM_PROMPT,
    NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT,
    RAWBASE_LEAK_CHECK_SYSTEM_PROMPT,
    RAWBASE_LEAK_CHECK_USER_TEMPLATE,
    TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE,
    TEACHER_HISTORY_MASKED_TEMPLATE,
    render_prompt,
)

Protocol_ = Literal["free_chat", "classroom"]
Scorer = Literal["retest", "interview"]

PROTOCOLS: tuple[Protocol_, ...] = ("free_chat", "classroom")
SCORERS: tuple[Scorer, ...] = ("retest", "interview")

# A transcript is the dialogue and nothing else, in the two roles both
# protocols agree on. Everything downstream -- both scorers, both leak judges --
# is a function of this plus the task, which is what makes the cross legal.
Transcript = list[dict[str, str]]

# messages -> one completion. The arm's trainable teacher.
TeacherCall = Callable[..., Awaitable[str]]
# messages, n -> n completions. The frozen student, shared by both arms.
StudentCall = Callable[..., Awaitable[list[str]]]
# messages -> one completion. The auxiliary model: leak judges and answer judge.
JudgeCall = Callable[..., Awaitable[str]]
# task, ground_truth, answer -> correct. The tutor arm's LLM answer judge.
AnswerJudge = Callable[..., Awaitable[bool]]


@dataclass(slots=True)
class FreeChatSpec:
    """The tutor protocol's parameters. Defaults match math/0810."""

    budget: int = 5
    enable_thinking: bool = False
    show_ground_truth: bool = False
    # Tells the teacher, in the system prompt, that the student has not been
    # shown the problem. Mirrors free_chat.student_has_not_seen_problem; it must
    # follow the arm's own setting or the teacher opens differently here than it
    # does in the rollout being compared.
    student_has_not_seen_problem: bool = False
    # Set from the arm's own teacher_pre. A draft rides in as a user/assistant
    # exchange ahead of the conversation, exactly as the tutor rollout does it.
    teacher_draft: str = ""
    max_student_tokens: int = 2048
    # How the teacher sees ITS OWN earlier turns: 'stripped' hands back the bare
    # visible text, 'masked' restores the tag skeleton with the reasoning replaced
    # by a placeholder, 'unmasked' replays the exact prior reply including its
    # private reasoning. Only the teacher view; the student and the re-test always
    # get the public visible text.
    #
    # THIS IS NOT COSMETIC AND IT MUST FOLLOW THE TUTOR ARM. Under 'stripped' the
    # teacher imitates its own untagged replies and malformed turns measured 6.3%
    # at depth 1 rising to 41.7% at depth 5 on 20260810_234129. math/0810 now runs
    # 'unmasked'. If the ped arm ran our protocol under a different mode than the
    # arm it is being compared against, the free_chat column would be measuring the
    # setting rather than the teacher.
    teacher_history_tags: str = "unmasked"


@dataclass(slots=True)
class ClassroomSpec:
    """PedagogicalRL's protocol parameters. Defaults match their baseline."""

    max_teacher_turns: int = 10
    max_tokens_in_conversation: int = 24576
    max_tokens_per_student_turn: int = 2048
    max_tokens_per_student_attempt: int = 2048
    include_thinking: bool = False
    teacher_draft: str = ""
    # Their type and student name are a hash of the problem, so both arms land
    # on the same one for the same row without having to agree on a seed.
    forced_type: ConversationType | None = None
    forced_student_name: str | None = None


@dataclass(slots=True)
class RetestSpec:
    """The tutor scorer's parameters."""

    replays: int = 4
    max_tokens: int = 2048


@dataclass(slots=True)
class InterviewSpec:
    """PedagogicalRL's scorer parameters."""

    attempts: int = 8
    max_tokens: int = 2048
    # Empty selects their unnamed variant. The free-chat dialogue never used a
    # name, so scoring one with a named persona would introduce a stranger.
    student_name: str = ""
    timeout: float | None = None


@dataclass(slots=True)
class LeakJudgeSpec:
    """Both leak judges, as metrics only. Neither ends a rollout."""

    turn_enabled: bool = True
    native_enabled: bool = True
    native_attempts: int = 2
    native_max_retries: int = 5
    max_tokens: int = 1024


@dataclass(slots=True)
class DialogueResult:
    transcript: Transcript = field(default_factory=list)
    initial_attempt: str = ""
    teacher_turns: int = 0
    termination_reason: str = ""
    format_errors: int = 0
    error: str | None = None


@dataclass(slots=True)
class ScoreResult:
    """One cell of the cross.

    ``complete`` is false when any model call in the cell failed. An incomplete
    cell scores 0.0 and is reported under `incomplete` rather than being
    averaged in as a wrong answer -- a failed call is missing data, and dropping
    only the failed attempts would bias the mean toward short completions.
    """

    score: float = 0.0
    any_correct: float = 0.0
    complete: bool = False
    attempts: int = 0
    # `interview` only: the same attempts under AReaL's normalizing math scorer,
    # so a ranking that survives only one scorer is visible as such.
    score_math: float | None = None
    solutions: list[str] = field(default_factory=list)


def as_chat(
    transcript: Transcript,
    *,
    speaker: str,
    own_turn_template: str | None = None,
    own_turn_raw_outputs: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    """The dialogue seen from one side: own turns assistant, the other user.

    The same mapping examples/tutor/workflow.py:_render_conversation applies, so
    a transcript scored here is shaped the way the arm that produced it would
    have shaped it -- INCLUDING the three teacher_history_tags modes, because a
    head-to-head that differed on this would be measuring the setting rather
    than the teacher.

    ``own_turn_template`` re-wraps the speaker own turns and takes a single
    ``{visible}`` field. ``own_turn_raw_outputs`` takes precedence when an
    aligned non-empty raw reply exists; that is the teacher-only unmasked
    history, where the exact prior reply including its private reasoning is
    replayed. A malformed turn stores an empty string and falls back to the
    masked skeleton, so the failure mode stripped had -- the teacher imitating
    its own untagged output -- cannot come back through this path. Only the
    teacher passes either argument: the student is never shown the tag protocol.
    """

    rendered = []
    own_turn_idx = 0
    for turn in transcript:
        is_own = turn["role"] == speaker
        content = turn["content"]
        if is_own:
            raw_content = (
                own_turn_raw_outputs[own_turn_idx]
                if own_turn_raw_outputs is not None
                and own_turn_idx < len(own_turn_raw_outputs)
                else ""
            )
            own_turn_idx += 1
            if raw_content:
                content = raw_content
            elif own_turn_template is not None:
                content = own_turn_template.format(visible=content)
        rendered.append(
            {"role": "assistant" if is_own else "user", "content": content}
        )
    return rendered



def _free_chat_teacher_system(spec: FreeChatSpec, task: str, ground_truth: str) -> str:
    system = render_prompt(
        FREE_CHAT_TEACHER_SYSTEM_PROMPT,
        budget=int(spec.budget),
        task=task,
        student_problem_context=(
            "\n\nThe student has not seen the math problem yet."
            if spec.student_has_not_seen_problem
            else ""
        ),
    )
    if ground_truth and spec.show_ground_truth:
        system = f"{system}\n\n" + render_prompt(
            TEACHER_GROUND_TRUTH_CONTEXT_TEMPLATE, ground_truth=ground_truth
        )
    return system


def _free_chat_preamble(spec: FreeChatSpec) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    draft = strip_reasoning_for_context(spec.teacher_draft or "").strip()
    if draft:
        messages.append({"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT})
        messages.append({"role": "assistant", "content": draft})
    parts = [FREE_CHAT_TEACHER_OPEN_PROMPT]
    if not spec.enable_thinking:
        parts.append(NON_THINKING_TEACHER_OUTPUT_FORMAT_PROMPT)
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


def _teacher_visible(raw_output: str, *, enable_thinking: bool) -> tuple[str, bool]:
    """The student-visible half of a teacher turn, and whether it was malformed.

    Mirrors examples/tutor/workflow.py:_parse_tutor_visible_output. A malformed
    turn hands the student an empty message and the dialogue continues, which is
    `format_handling_mode: continue` -- the eval default, because terminating is
    a training policy and must not decide what gets measured.
    """

    from examples.tutor.core.parsers import parse_tagged_teacher_output

    if enable_thinking:
        return strip_reasoning_for_context(raw_output), False
    output, _ = parse_tagged_teacher_output(raw_output)
    if output is None:
        return "", True
    return strip_reasoning_for_context(output), False


async def run_free_chat_dialogue(
    *,
    task: str,
    ground_truth: str,
    spec: FreeChatSpec,
    teacher_call: TeacherCall,
    student_call: StudentCall,
) -> DialogueResult:
    """The tutor protocol, driven by whatever teacher the caller supplies.

    Nothing is judged in between and nothing ends it early: the budget is the
    ending. A leak is recorded afterwards as a metric, never as a termination,
    because terminating on a leak is a training policy and no real conversation
    truncates itself.
    """

    result = DialogueResult()
    system = _free_chat_teacher_system(spec, task, ground_truth)
    preamble = _free_chat_preamble(spec)
    transcript: Transcript = []
    # All three modes, matching the tutor arm. Under masked the teacher gets its
    # tag skeleton back with the reasoning replaced; under unmasked it reads its
    # exact prior replies, reasoning included. Only the teacher view changes --
    # the student and the re-test always replay the public visible text below.
    own_turn_template = (
        TEACHER_HISTORY_MASKED_TEMPLATE
        if spec.teacher_history_tags in {"masked", "unmasked"}
        else None
    )
    unmasked_history = spec.teacher_history_tags == "unmasked"
    # One entry per teacher turn, empty when that turn was malformed, so the
    # fallback to the masked skeleton lines up with the transcript.
    own_raw_outputs: tuple[str, ...] = ()
    for turn_idx in range(1, int(spec.budget) + 1):
        teacher_messages = [
            {"role": "system", "content": system},
            *preamble,
            *as_chat(
                transcript,
                speaker="teacher",
                own_turn_template=own_turn_template,
                own_turn_raw_outputs=(
                    own_raw_outputs if unmasked_history else None
                ),
            ),
        ]
        try:
            raw_output = await teacher_call(
                teacher_messages, rid_prefix=f"xeval-freechat-teacher-t{turn_idx}"
            )
        except Exception as exc:  # noqa: BLE001 - a dead call ends the dialogue
            result.error = f"{type(exc).__name__}: {exc}"
            break
        visible, malformed = _teacher_visible(
            raw_output, enable_thinking=spec.enable_thinking
        )
        if malformed:
            result.format_errors += 1
        transcript.append({"role": "teacher", "content": visible})
        result.teacher_turns += 1
        own_raw_outputs = (
            *own_raw_outputs,
            "" if malformed else raw_output,
        )

        student_messages = [
            {"role": "system", "content": FREE_CHAT_STUDENT_SYSTEM_PROMPT},
            *as_chat(transcript, speaker="student"),
        ]
        try:
            replies = await student_call(
                student_messages,
                n=1,
                max_tokens=spec.max_student_tokens,
                rid_prefix=f"xeval-freechat-student-t{turn_idx}",
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            break
        transcript.append(
            {
                "role": "student",
                "content": strip_reasoning_for_context(replies[0] if replies else ""),
            }
        )
    result.transcript = transcript
    result.termination_reason = "error" if result.error else "budget"
    return result


async def _rawbase_leak_check(
    *, teacher_output: str, ground_truth: str, judge_call: JudgeCall
) -> bool:
    """AReaL's turn-level leak judge -- the one the tutor arm trains against."""

    prompt = render_prompt(
        RAWBASE_LEAK_CHECK_USER_TEMPLATE,
        ground_truth=ground_truth,
        teacher_action=student_visible_text(teacher_output),
    )
    try:
        raw = await judge_call(
            [
                {"role": "system", "content": RAWBASE_LEAK_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            rid_prefix="xeval-turn-leak-judge",
        )
    except Exception:  # noqa: BLE001 - a dead judge must not read as clean
        return True
    return bool(parse_leak_check_result(raw).leaked)


async def run_classroom_dialogue(
    *,
    problem: str,
    answer: str,
    spec: ClassroomSpec,
    tokenizer: Any,
    teacher_call: TeacherCall,
    student_call: StudentCall,
) -> tuple[DialogueResult, ClassroomEpisode]:
    """PedagogicalRL's dialogue, driven by whatever teacher the caller supplies.

    Their loop from workflow.py:_eval_episode with the scoring removed and the
    per-turn leak check moved out: at evaluation the judge is a metric, so it
    does not belong in the loop that decides when to stop.
    """

    episode = ClassroomEpisode(
        problem=problem,
        answer=answer,
        include_thinking=spec.include_thinking,
        forced_type=spec.forced_type,
        forced_student_name=spec.forced_student_name,
    )
    if spec.teacher_draft:
        episode.teacher_draft = spec.teacher_draft
    result = DialogueResult()

    def stop() -> bool:
        return episode.should_stop_dialogue(
            tokenizer=tokenizer,
            max_teacher_turns=spec.max_teacher_turns,
            max_tokens_in_conversation=spec.max_tokens_in_conversation,
        )

    if episode.conversation_type is ConversationType.ATTEMPTED:
        try:
            initial = await student_call(
                episode.initial_student_messages(),
                n=1,
                max_tokens=spec.max_tokens_per_student_attempt,
                rid_prefix="xeval-classroom-initial",
            )
            episode.add_initial_attempt(initial[0] if initial else "")
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"

    while result.error is None and not stop():
        try:
            teacher_output = await teacher_call(
                episode.teacher_messages(),
                rid_prefix=f"xeval-classroom-teacher-t{episode.teacher_turns + 1}",
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            break
        episode.add_teacher(teacher_output)
        if stop():
            break
        try:
            replies = await student_call(
                episode.student_messages(),
                n=1,
                max_tokens=spec.max_tokens_per_student_turn,
                rid_prefix=f"xeval-classroom-student-t{episode.teacher_turns}",
            )
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
            break
        episode.add_student(replies[0] if replies else "")

    result.transcript = [
        {"role": turn["role"], "content": student_visible_text(turn["content"])}
        for turn in episode.conversation
    ]
    result.initial_attempt = episode.initial_attempt or ""
    result.teacher_turns = episode.teacher_turns
    result.termination_reason = episode.termination_reason or (
        "error" if result.error else "max_turns"
    )
    return result, episode


def retest_messages(*, transcript: Transcript, task: str) -> list[dict[str, str]]:
    """The tutor measurement's prompt: the dialogue, then solve it alone.

    Factored out because the SFT arm trains on exactly this with an empty
    transcript. Both callers go through here so the training input and the eval
    input cannot drift apart -- which is the whole point of that arm: it is the
    best case for this setting, a student fine-tuned on the very prompt it is
    then tested under.
    """

    return [
        {"role": "system", "content": FREE_CHAT_STUDENT_SYSTEM_PROMPT},
        *as_chat(transcript, speaker="student"),
        {
            "role": "user",
            "content": render_prompt(FREE_CHAT_STUDENT_RETEST_TEMPLATE, task=task),
        },
    ]


async def score_retest(
    *,
    transcript: Transcript,
    task: str,
    ground_truth: str,
    spec: RetestSpec,
    student_call: StudentCall,
    answer_judge: AnswerJudge,
) -> ScoreResult:
    """The tutor measurement, on any transcript.

    Replay the dialogue on a fresh branch and ask for a solution from scratch.
    The task appears for the first time in the final turn, which is what the
    free-chat student saw; a classroom student was told the problem during the
    dialogue, and repeating it here changes nothing about that.
    """

    replays = max(1, int(spec.replays))
    messages = retest_messages(transcript=transcript, task=task)
    try:
        # Independent requests, not one n-choice call: this is how the tutor arm
        # measures its own reward, and the two have to stay the same quantity.
        replies = await asyncio.gather(
            *(
                student_call(
                    messages,
                    n=1,
                    max_tokens=spec.max_tokens,
                    rid_prefix=f"xeval-retest-r{index}",
                )
                for index in range(replays)
            )
        )
    except Exception:  # noqa: BLE001
        return ScoreResult(complete=False, attempts=replays)
    solutions = [reply[0] if reply else "" for reply in replies]
    if len(solutions) != replays or not all(solutions):
        return ScoreResult(complete=False, attempts=replays, solutions=solutions)
    verdicts = await asyncio.gather(
        *(
            answer_judge(task=task, ground_truth=ground_truth, answer=solution)
            for solution in solutions
        ),
        return_exceptions=True,
    )
    scored = [bool(v) for v in verdicts if not isinstance(v, BaseException)]
    if len(scored) != replays:
        return ScoreResult(complete=False, attempts=replays, solutions=solutions)
    return ScoreResult(
        score=sum(scored) / len(scored),
        any_correct=float(any(scored)),
        complete=True,
        attempts=replays,
        solutions=solutions,
    )


def interview_messages(
    *,
    transcript: Transcript,
    task: str,
    spec: InterviewSpec,
    initial_attempt: str = "",
) -> list[dict[str, str]]:
    """PedagogicalRL's student-perspective replay of any dialogue."""

    student_name = str(spec.student_name or "") or None
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": render_native(
                SIMPLE_STUDENT_PROMPT, student_name=student_name, problem=task
            ),
        }
    ]
    if initial_attempt:
        messages.append(
            {
                "role": "assistant",
                "content": render_native(
                    INITIAL_ATTEMPT_WRAPPER, attempt=initial_attempt
                ),
            }
        )
    for turn in transcript:
        content = student_visible_text(turn["content"]).strip()
        if not content:
            continue
        messages.append(
            {
                "role": "assistant" if turn["role"] == "student" else "user",
                "content": content,
            }
        )
    messages.append(
        {"role": "user", "content": render_native(STUDENT_FINAL_PROMPT)}
    )
    return messages


async def score_interview(
    *,
    transcript: Transcript,
    task: str,
    ground_truth: str,
    spec: InterviewSpec,
    student_call: StudentCall,
    initial_attempt: str = "",
) -> ScoreResult:
    """PedagogicalRL's measurement, on any transcript.

    One n-choice request, not `attempts` separate ones. The student endpoint
    pins a seed, so identical messages sent separately can return the same
    completion every time and collapse the average.
    """

    attempts = max(1, int(spec.attempts))
    messages = interview_messages(
        transcript=transcript, task=task, spec=spec, initial_attempt=initial_attempt
    )
    try:
        solutions = await student_call(
            messages,
            n=attempts,
            max_tokens=spec.max_tokens,
            rid_prefix="xeval-interview",
            timeout=spec.timeout,
        )
    except Exception:  # noqa: BLE001
        return ScoreResult(complete=False, attempts=attempts)
    if len(solutions) != attempts:
        return ScoreResult(complete=False, attempts=attempts, solutions=list(solutions))
    native = [float(native_answer_correct(s, ground_truth)) for s in solutions]
    math = [
        float(bool(score_math_answer(task, ground_truth, s).correct)) for s in solutions
    ]
    return ScoreResult(
        score=sum(native) / len(native),
        any_correct=float(any(native)),
        complete=True,
        attempts=attempts,
        score_math=sum(math) / len(math),
        solutions=list(solutions),
    )


async def score_leaks(
    *,
    transcript: Transcript,
    ground_truth: str,
    spec: LeakJudgeSpec,
    judge_call: JudgeCall,
) -> dict[str, float]:
    """Both leak judges on one transcript, as metrics.

    Each arm optimises one of these, so a leak rate read under the arm's own
    judge flatters it. Running both on both arms is what makes the leak column
    of the table mean anything.
    """

    metrics: dict[str, float] = {}
    teacher_turns = [t["content"] for t in transcript if t["role"] == "teacher"]
    if spec.turn_enabled:
        verdicts = await asyncio.gather(
            *(
                _rawbase_leak_check(
                    teacher_output=content,
                    ground_truth=ground_truth,
                    judge_call=judge_call,
                )
                for content in teacher_turns
            )
        )
        metrics["leak/turn"] = float(any(verdicts))
    if spec.native_enabled:

        async def call(prompt: str) -> list[str]:
            return [await judge_call(
                [{"role": "user", "content": prompt}],
                rid_prefix="xeval-native-leak-judge",
            )]

        decisions = await run_whole_dialogue_judge(
            rule=LEAK_RULE,
            conversation=transcript,
            call=call,
            attempts=spec.native_attempts,
            max_retries=spec.native_max_retries,
        )
        metrics["leak/native"] = float(any(d.rejected for d in decisions))
    return metrics


async def score_no_dialogue(
    *,
    task: str,
    ground_truth: str,
    retest: RetestSpec,
    interview: InterviewSpec,
    student_call: StudentCall,
    answer_judge: AnswerJudge,
) -> dict[Scorer, ScoreResult]:
    """Both scorers with no dialogue at all: the student is handed the problem.

    This is the row the SFT arm occupies, and the one the taught arms are read
    against. It is the same two scorers on an empty transcript rather than a
    third measurement, so a distilled student and a taught student are compared
    on one instrument.

    Note both scorers already show the student the problem -- ours in the final
    turn, theirs in the system prompt -- so an empty transcript is a complete
    prompt for each, not a degenerate one.
    """

    retest_result, interview_result = await asyncio.gather(
        score_retest(
            transcript=[],
            task=task,
            ground_truth=ground_truth,
            spec=retest,
            student_call=student_call,
            answer_judge=answer_judge,
        ),
        score_interview(
            transcript=[],
            task=task,
            ground_truth=ground_truth,
            spec=interview,
            student_call=student_call,
        ),
    )
    return {"retest": retest_result, "interview": interview_result}


def no_dialogue_metrics(results: dict[Scorer, ScoreResult]) -> dict[str, float]:
    """The no-dialogue row, under the same names shape as the crossed cells."""

    metrics: dict[str, float] = {}
    for scorer, result in results.items():
        prefix = f"xeval/no_dialogue/{scorer}"
        metrics[f"{prefix}/success"] = float(result.score)
        metrics[f"{prefix}/success_any"] = float(result.any_correct)
        metrics[f"{prefix}/incomplete"] = float(not result.complete)
        if result.score_math is not None:
            metrics[f"{prefix}/success_math"] = float(result.score_math)
    return metrics


def cell_metrics(
    *, protocol: Protocol_, scorer: Scorer, result: ScoreResult
) -> dict[str, float]:
    """One cell's series, under names both arms emit identically."""

    prefix = f"xeval/{protocol}/{scorer}"
    metrics = {
        f"{prefix}/success": float(result.score),
        f"{prefix}/success_any": float(result.any_correct),
        f"{prefix}/incomplete": float(not result.complete),
    }
    if result.score_math is not None:
        metrics[f"{prefix}/success_math"] = float(result.score_math)
    return metrics


async def run_cross_eval(
    *,
    task: str,
    ground_truth: str,
    own_protocol: Protocol_,
    own_transcript: Transcript,
    own_initial_attempt: str = "",
    teacher_call: TeacherCall,
    student_call: StudentCall,
    judge_call: JudgeCall,
    answer_judge: AnswerJudge,
    tokenizer: Any,
    free_chat: FreeChatSpec,
    classroom: ClassroomSpec,
    retest: RetestSpec,
    interview: InterviewSpec,
    leak_judges: LeakJudgeSpec | None = None,
    run_other_protocol: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """The full cross for one problem, from one arm's point of view.

    The caller passes the transcript its own eval rollout already produced --
    re-running it here would measure a second sample of the same thing and cost
    a whole dialogue. The other protocol is run fresh with the same teacher.

    Returns the metric dict and a details dict for the debug trace.
    """

    transcripts: dict[Protocol_, DialogueResult] = {
        own_protocol: DialogueResult(
            transcript=own_transcript,
            initial_attempt=own_initial_attempt,
            teacher_turns=sum(1 for t in own_transcript if t["role"] == "teacher"),
            termination_reason="own",
        )
    }
    other: Protocol_ = "classroom" if own_protocol == "free_chat" else "free_chat"
    if run_other_protocol:
        if other == "classroom":
            dialogue, _episode = await run_classroom_dialogue(
                problem=task,
                answer=ground_truth,
                spec=classroom,
                tokenizer=tokenizer,
                teacher_call=teacher_call,
                student_call=student_call,
            )
        else:
            dialogue = await run_free_chat_dialogue(
                task=task,
                ground_truth=ground_truth,
                spec=free_chat,
                teacher_call=teacher_call,
                student_call=student_call,
            )
        transcripts[other] = dialogue

    metrics: dict[str, float] = {}
    details: dict[str, Any] = {}
    for protocol, dialogue in transcripts.items():
        details[protocol] = {
            "transcript": dialogue.transcript,
            "teacher_turns": dialogue.teacher_turns,
            "termination_reason": dialogue.termination_reason,
            "error": dialogue.error,
        }
        metrics[f"xeval/{protocol}/turns"] = float(dialogue.teacher_turns)
        if dialogue.format_errors:
            metrics[f"xeval/{protocol}/format_errors"] = float(dialogue.format_errors)
        teacher_turns = sum(1 for t in dialogue.transcript if t["role"] == "teacher")
        if dialogue.error is not None or not teacher_turns:
            # No teaching happened, so there is nothing to score. A transcript
            # can be non-empty and still qualify: an ATTEMPTED classroom episode
            # opens with the student's own attempt, and scoring that after the
            # teacher call died would measure the student's unaided ability and
            # file it as teaching. A dead dialogue is missing data, the same as
            # a dead scorer call, and is reported rather than averaged in.
            for scorer in SCORERS:
                metrics[f"xeval/{protocol}/{scorer}/incomplete"] = 1.0
            continue
        retest_result, interview_result = await asyncio.gather(
            score_retest(
                transcript=dialogue.transcript,
                task=task,
                ground_truth=ground_truth,
                spec=retest,
                student_call=student_call,
                answer_judge=answer_judge,
            ),
            score_interview(
                transcript=dialogue.transcript,
                task=task,
                ground_truth=ground_truth,
                spec=interview,
                student_call=student_call,
                initial_attempt=dialogue.initial_attempt,
            ),
        )
        metrics.update(
            cell_metrics(protocol=protocol, scorer="retest", result=retest_result)
        )
        metrics.update(
            cell_metrics(protocol=protocol, scorer="interview", result=interview_result)
        )
        details[protocol]["retest"] = retest_result.solutions
        details[protocol]["interview"] = interview_result.solutions
        if leak_judges is not None:
            leaks = await score_leaks(
                transcript=dialogue.transcript,
                ground_truth=ground_truth,
                spec=leak_judges,
                judge_call=judge_call,
            )
            metrics.update(
                {f"xeval/{protocol}/{key}": value for key, value in leaks.items()}
            )
            # P(leak | success) is read as the ratio of these two means. It is
            # deliberately not logged per rollout: undefined for a failed one,
            # and wrong to average a per-rollout ratio.
            for scorer, result in (
                ("retest", retest_result),
                ("interview", interview_result),
            ):
                if result.any_correct:
                    for judge_name, leaked in leaks.items():
                        metrics[
                            f"xeval/{protocol}/{scorer}/success_and_{judge_name}"
                        ] = float(leaked)
    return metrics, details
