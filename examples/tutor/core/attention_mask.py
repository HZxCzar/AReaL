"""What a student is allowed to see of the dialogue it is having.

Every student in the free-chat rollout is the same frozen model. What makes two
of them behave like different learners is not a persona and not a fine-tune -- it
is which part of the transcript reaches them.

WHY NOT A PERSONA. A self-description is a request, and this student ignores
requests: prompting it to behave like a particular learner is measured to change
nothing. A mask is not a request. The student cannot use information that is not
in its context, so the constraint binds whether the model cooperates or not.

WHY NOT AN SFT PROFILE. A trained profile has to be defended -- why those failure
modes, why that data. A mask installs no behaviour and fits no parameters. It
removes information, and the differential teaching need FOLLOWS from the removal
rather than from a modelling choice: if the teacher's turn 1 is gone by turn 4,
what survives is whatever the student was induced to say, so eliciting beats
explaining. That is derived, not stipulated.

TWO KINDS OF LIMIT, and they behave differently on the live turn.

  A MEMORY limit describes what survives into later turns. `teacher_fade` and
  `student_fade` are memory limits, so they touch HISTORY ONLY. The message being
  answered right now is always fully visible -- a student that cannot read the
  turn it is replying to cannot reply at all, and every arm would collapse
  together.

  An ATTENTION limit describes what is read in the first place. `long_drop` is an
  attention limit, so it also truncates the LIVE turn: someone who stops reading
  long messages stops reading the current one too, not merely the ones being
  recalled. This is the distinction that makes brevity actually pay -- under a
  history-only version a long message still lands in full at the moment it is
  sent, and only fades afterwards.

TRUNCATION HAPPENS ON READ, NOT ON WRITE. The shared public history always holds
what the teacher actually said. These functions return a masked VIEW and never
mutate the input, because the same history object is rendered for the teacher --
whose view must stay complete, or it could see what the student missed and the
inference problem would disappear -- and is read by the reward path.

DETERMINISTIC ON PURPOSE. A probabilistic mask would need an episode-seeded RNG
so the student has forgotten the same things at re-test that it had forgotten
mid-dialogue. Get that wrong and the mask stops being a memory model and becomes
noise, with nothing in the metrics to reveal it. Everything here is a pure
function of the history.

The two fades also correspond to a live disagreement in instructional research --
the Coverage hypothesis (explain as completely as possible) against the
Generation hypothesis (give sparse material and make the learner produce the
rest). `student_fade` is a learner for whom Coverage is right, `teacher_fade` one
for whom Generation is, and `long_drop` is what stops a single "long message that
ends by asking for a restatement" from satisfying both at once.
"""

from __future__ import annotations

from typing import Any

MASK_FULL = "full"
MASK_TEACHER_FADE = "teacher_fade"
MASK_STUDENT_FADE = "student_fade"
MASK_LONG_DROP = "long_drop"
MASK_MODES = (MASK_FULL, MASK_TEACHER_FADE, MASK_STUDENT_FADE, MASK_LONG_DROP)

# Memory limits act on recall, so they leave the live turn alone. Attention
# limits act on reading, so they do not.
HISTORY_ONLY_MODES = (MASK_TEACHER_FADE, MASK_STUDENT_FADE)

# A FADED TURN IS REPLACED, NOT DELETED, and there are three reasons.
#
# Coherence. Deleting a turn leaves the next one referring to nothing: "list the
# arrangements THAT LEAVES" points at a message the student cannot see, and the
# reply is confusion rather than a forgetful student's reply. A placeholder says
# an exchange happened whose content is gone, which is what forgetting actually
# feels like from the inside.
#
# Role alternation. Deleting every turn of one role produces runs of consecutive
# user or assistant messages. Some chat templates merge or reject those, which
# would silently change what the model receives -- and a silent merge is exactly
# the class of bug that makes a mask look ineffective. Replacing keeps the
# transcript strictly alternating.
#
# Precedent. This is what observation masking does in agent context management:
# older observations are swapped for a placeholder rather than dropped.
#
# ONE MARKER, ONE WORD. The role already says whose turn it was -- the student's
# own turns render as assistant and the teacher's as user -- so the marker does
# not need to repeat it, and every word here is paid for on every masked turn of
# every episode.
#
# The meaning is carried by the system prompt, which is where the notation is
# explained once instead of inline hundreds of times. That note is a READING
# CONVENTION, not a persona: it says what a token means rather than asking the
# model to act forgetful. Personas are measured not to change this student's
# behaviour at all, so the mask has to work by absence of information.
PLACEHOLDER_TURN = "(forget)"
# The SAME token where a message was cut off, so the student can tell it stopped
# reading rather than that the teacher stopped writing. Deliberately not a
# distinct "(...)": one marker needs one sentence of explanation instead of two,
# and the student's situation is identical either way -- content it does not have.
TRUNCATION_MARKER = PLACEHOLDER_TURN

DEFAULT_MASK: dict[str, Any] = {
    "mode": MASK_FULL,
    "keep_recent": 0,
    "long_drop_words": 75,
    "placeholder": True,
}


def normalize_mask(raw: Any) -> dict[str, Any]:
    """Config -> a validated plain dict. Unknown modes raise at startup.

    Anything empty yields the full-visibility default, so a config that says
    nothing about masking behaves exactly as it did before this module existed.
    """
    if raw is None:
        return dict(DEFAULT_MASK)
    if isinstance(raw, str):
        raw = {"mode": raw}
    if not isinstance(raw, dict):
        # omegaconf structured configs arrive as objects, not dicts.
        raw = {
            key: getattr(raw, key)
            for key in DEFAULT_MASK
            if getattr(raw, key, None) is not None
        }
    mask = dict(DEFAULT_MASK)
    mask.update({key: value for key, value in raw.items() if value is not None})
    mode = str(mask["mode"]).strip()
    if mode not in MASK_MODES:
        raise ValueError(
            f"student mask mode must be one of {MASK_MODES}, got {mode!r}."
        )
    mask["mode"] = mode
    mask["keep_recent"] = int(mask["keep_recent"])
    if mask["keep_recent"] < 0:
        raise ValueError(
            f"student mask keep_recent must be >= 0, got {mask['keep_recent']}."
        )
    mask["long_drop_words"] = int(mask["long_drop_words"])
    if mode == MASK_LONG_DROP and mask["long_drop_words"] < 1:
        raise ValueError(
            "long_drop needs long_drop_words >= 1; a limit of 0 deletes every "
            "teacher message outright, which is teacher_fade with extra steps."
        )
    mask["placeholder"] = bool(mask["placeholder"])
    return mask


def is_identity(mask: dict[str, Any] | None) -> bool:
    """True when this mask cannot change anything, so callers can skip it."""
    return mask is None or str(mask.get("mode", MASK_FULL)) == MASK_FULL


def truncate_words(text: str, limit: int, *, marker: bool = True) -> str:
    """Cut after `limit` words -- the point where the student stops reading.

    Words rather than sentences because the quantity a teacher controls is
    length, and a word budget is the cleanest scalar to put pressure on. A
    sentence rule would let a teacher pack an unbounded amount into three very
    long sentences and pay nothing.

    `marker` appends an ellipsis so the student can tell it stopped reading
    rather than that the teacher stopped writing -- a message that simply ends
    mid-clause reads as a malformed teacher, which is a different failure and
    one the format metrics already track separately.
    """
    words = (text or "").split()
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    return f"{cut} {TRUNCATION_MARKER}" if marker else cut


def apply_student_mask(
    turns: list[dict[str, str]], mask: dict[str, Any] | None
) -> list[dict[str, str]]:
    """The student's view of the dialogue HISTORY.

    ``turns`` is oldest-first, each entry at least {"role", "content"} with role
    in {"teacher", "student"}. Returns a NEW list and never mutates the input.
    """
    if is_identity(mask) or not turns:
        return list(turns)
    mode = mask["mode"]

    if mode in HISTORY_ONLY_MODES:
        target = "teacher" if mode == MASK_TEACHER_FADE else "student"
        keep_recent = int(mask["keep_recent"])
        positions = [
            index for index, turn in enumerate(turns) if turn.get("role") == target
        ]
        # keep_recent counts back from the most recent turn of that role, so 1 is
        # "remembers only the last thing" and 0 is "remembers none of them".
        # Anything at or above the count is a no-op.
        survivors = (
            set(positions[len(positions) - keep_recent :]) if keep_recent else set()
        )
        placeholder = bool(mask["placeholder"])
        masked: list[dict[str, str]] = []
        for index, turn in enumerate(turns):
            if turn.get("role") != target or index in survivors:
                masked.append(turn)
            elif placeholder:
                # Same role, so alternation survives; content gone, so the
                # information does too.
                masked.append({**turn, "content": PLACEHOLDER_TURN})
            # else: dropped outright, the pre-placeholder behaviour
        return masked

    if mode == MASK_LONG_DROP:
        limit = int(mask["long_drop_words"])
        placeholder = bool(mask["placeholder"])
        return [
            turn
            if turn.get("role") != "teacher"
            else {
                **turn,
                "content": truncate_words(
                    turn.get("content", ""), limit, marker=placeholder
                ),
            }
            for turn in turns
        ]

    raise ValueError(f"unhandled student mask mode {mode!r}")


def mask_current_turn(text: str, mask: dict[str, Any] | None) -> str:
    """The teacher message the student is answering RIGHT NOW.

    Only `long_drop` touches this, and that asymmetry is the whole point of
    separating memory limits from attention limits. Returning `text` unchanged
    for the fades is not an oversight -- it is what keeps the dialogue coherent
    for a student whose limitation is recall rather than reading.
    """
    if is_identity(mask):
        return text
    if mask["mode"] == MASK_LONG_DROP:
        return truncate_words(
            text, int(mask["long_drop_words"]), marker=bool(mask["placeholder"])
        )
    return text


def mask_label(mask: dict[str, Any] | None) -> str:
    """Short stable string for logs and metric names."""
    if is_identity(mask):
        return MASK_FULL
    mode = mask["mode"]
    core = (
        f"{mode}:k{mask['keep_recent']}"
        if mode in HISTORY_ONLY_MODES
        else f"{mode}:w{mask['long_drop_words']}"
    )
    # Recorded in the label because it changes what the student reads, so two runs
    # differing only in this are not comparable.
    return core if mask["placeholder"] else f"{core}:hard"
