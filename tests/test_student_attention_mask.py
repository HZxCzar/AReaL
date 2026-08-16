#!/usr/bin/env python3
"""Tests for the student attention masks.

The failure this guards against is not a crash. It is a mask that silently does
not apply: the run then looks exactly like the unmasked control, and the
conclusion reads as "different learners need the same teaching" when the truth is
that the plumbing broke.

So most assertions are that a mask really removes what it claims to, plus the one
asymmetry the design turns on -- memory limits leave the live turn whole,
attention limits do not.

    python tests/test_student_attention_mask.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.tutor.core.attention_mask import (  # noqa: E402
    DEFAULT_MASK,
    PLACEHOLDER_TURN,
    TRUNCATION_MARKER,
    apply_student_mask,
    is_identity,
    mask_current_turn,
    mask_label,
    normalize_mask,
    truncate_words,
)

HISTORY = [
    {"role": "teacher", "content": "T1 alpha beta gamma delta epsilon zeta"},
    {"role": "student", "content": "S1 my working"},
    {"role": "teacher", "content": "T2 brief"},
    {"role": "student", "content": "S2 more working"},
    {"role": "teacher", "content": "T3 eta theta iota kappa"},
    {"role": "student", "content": "S3 last working"},
]
LONG_WORDS = " ".join(f"w{index}" for index in range(30))


def roles(turns):
    return [turn["role"] for turn in turns]


def check(name, condition):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}")
    return bool(condition)


def main() -> int:
    ok = True
    drop10 = normalize_mask({"mode": "long_drop", "long_drop_words": 10})

    print("normalize_mask")
    ok &= check("empty -> full", normalize_mask(None)["mode"] == "full")
    ok &= check("bare string accepted",
                normalize_mask("student_fade")["mode"] == "student_fade")
    ok &= check("defaults filled",
                normalize_mask("student_fade")["long_drop_words"] == 75)
    ok &= check("placeholder defaults on", normalize_mask("student_fade")["placeholder"] is True)
    for bad, label in (
        ({"mode": "nonsense"}, "unknown mode raises"),
        ({"mode": "long_drop", "long_drop_words": 0}, "long_drop word limit 0 raises"),
        ({"mode": "student_fade", "keep_recent": -1}, "negative keep_recent raises"),
    ):
        try:
            normalize_mask(bad)
            ok &= check(label, False)
        except ValueError:
            ok &= check(label, True)

    print("\nidentity")
    ok &= check("full is identity", is_identity(normalize_mask("full")))
    ok &= check("None is identity", is_identity(None))
    ok &= check("student_fade is not", not is_identity(normalize_mask("student_fade")))
    ok &= check("identity returns the same content",
                apply_student_mask(HISTORY, normalize_mask("full")) == HISTORY)

    def teacher_texts(turns):
        return [t["content"] for t in turns if t["role"] == "teacher"]

    def student_texts(turns):
        return [t["content"] for t in turns if t["role"] == "student"]

    print("\nteacher_fade -- keeps only what the student said itself")
    faded = apply_student_mask(HISTORY, normalize_mask("teacher_fade"))
    ok &= check("every turn is still there, so alternation survives",
                roles(faded) == roles(HISTORY))
    ok &= check("all teacher CONTENT is gone",
                teacher_texts(faded) == [PLACEHOLDER_TURN] * 3)
    ok &= check("the student's own words are untouched",
                student_texts(faded) == student_texts(HISTORY))
    kept = apply_student_mask(
        HISTORY, normalize_mask({"mode": "teacher_fade", "keep_recent": 1})
    )
    ok &= check("keep_recent=1 leaves exactly one real teacher message",
                sum(1 for t in teacher_texts(kept)
                    if t != PLACEHOLDER_TURN) == 1)
    ok &= check("and it is the MOST RECENT one",
                teacher_texts(kept)[-1].startswith("T3"))
    ok &= check("keep_recent beyond the count is a no-op",
                apply_student_mask(HISTORY, normalize_mask(
                    {"mode": "teacher_fade", "keep_recent": 99})) == HISTORY)
    hard = apply_student_mask(
        HISTORY, normalize_mask({"mode": "teacher_fade", "placeholder": False}))
    ok &= check("placeholder=false deletes outright instead",
                "teacher" not in roles(hard) and len(hard) == 3)

    print("\nstudent_fade -- keeps only what the student was told")
    faded = apply_student_mask(HISTORY, normalize_mask("student_fade"))
    ok &= check("alternation survives", roles(faded) == roles(HISTORY))
    ok &= check("all student CONTENT is gone",
                student_texts(faded) == [PLACEHOLDER_TURN] * 3)
    ok &= check("teacher content is untouched",
                teacher_texts(faded) == teacher_texts(HISTORY))
    ok &= check("no two adjacent turns share a role",
                all(faded[i]["role"] != faded[i + 1]["role"]
                    for i in range(len(faded) - 1)))

    print("\nlong_drop -- only short teacher turns arrive intact")
    history = [
        {"role": "teacher", "content": LONG_WORDS},
        {"role": "student", "content": "my working"},
        {"role": "teacher", "content": "brief note"},
    ]
    dropped = apply_student_mask(history, drop10)
    ok &= check("every turn is still present", len(dropped) == len(history))
    ok &= check("long teacher turn cut to the limit (plus the marker)",
                len(dropped[0]["content"].split()) == 11)
    ok &= check("the cut keeps the FRONT -- it stopped reading",
                dropped[0]["content"].split()[0] == "w0")
    ok &= check("truncation ends in the marker",
                dropped[0]["content"].endswith(TRUNCATION_MARKER))
    ok &= check("short teacher turn untouched", dropped[2]["content"] == "brief note")
    ok &= check("student turns are never truncated",
                dropped[1]["content"] == "my working")
    ok &= check("truncate_words is a no-op under the limit",
                truncate_words("a b c", 10) == "a b c")

    print("\nlive turn: a memory limit leaves it whole, an attention limit does not")
    for mode in ("full", "teacher_fade", "student_fade"):
        ok &= check(f"{mode} leaves the live turn whole",
                    mask_current_turn(LONG_WORDS, normalize_mask(mode)) == LONG_WORDS)
    ok &= check("None leaves the live turn whole",
                mask_current_turn(LONG_WORDS, None) == LONG_WORDS)
    ok &= check("long_drop TRUNCATES the live turn",
                len(mask_current_turn(LONG_WORDS, drop10).split()) == 11)

    print("\nno mutation of the shared history")
    before = [dict(turn) for turn in HISTORY]
    for mode in ("teacher_fade", "student_fade", "long_drop"):
        apply_student_mask(HISTORY, normalize_mask(mode))
    ok &= check("input list unchanged", HISTORY == before)
    ok &= check("teacher still sees the full first turn",
                HISTORY[0]["content"] == before[0]["content"])

    print("\nedge cases")
    ok &= check("empty history is safe",
                apply_student_mask([], normalize_mask("teacher_fade")) == [])
    ok &= check("a lone teacher turn under teacher_fade becomes a placeholder",
                apply_student_mask([HISTORY[0]], normalize_mask("teacher_fade"))[0]["content"]
                == PLACEHOLDER_TURN)
    ok &= check("empty content survives long_drop",
                apply_student_mask([{"role": "teacher", "content": ""}],
                                   drop10)[0]["content"] == "")
    ok &= check("extra keys on a turn are preserved",
                apply_student_mask([{"role": "teacher", "content": LONG_WORDS,
                                     "env": "note"}], drop10)[0].get("env") == "note")

    print("\nlabels")
    ok &= check("full label", mask_label(normalize_mask("full")) == "full")
    ok &= check("fade label carries keep_recent",
                mask_label(normalize_mask({"mode": "student_fade", "keep_recent": 2}))
                == "student_fade:k2")
    ok &= check("long_drop label carries the word limit",
                mask_label(normalize_mask("long_drop")) == "long_drop:w75")
    ok &= check("hard deletion is marked in the label",
                mask_label(normalize_mask({"mode": "student_fade", "placeholder": False}))
                == "student_fade:k0:hard")

    print("\nDEFAULT_MASK is not mutated by callers")
    snapshot = dict(DEFAULT_MASK)
    mask = normalize_mask(None)
    mask["mode"] = "student_fade"
    ok &= check("module default intact", DEFAULT_MASK == snapshot)

    # The masks are only worth anything if a GRPO group faces ONE learner. A
    # per-rollout draw puts mask identity straight into the advantage --
    # group_baseline='episode' subtracts the group mean -- and with three masks
    # that is variance the teacher can neither see nor control, since it speaks
    # first. Measured at 29-33% of within-group spread with only two students.
    print("\nthe student draw is group-scoped")
    from examples.tutor.workflow import TutorAgentWorkflow  # noqa: E402

    flow = TutorAgentWorkflow(dataset_type="math", answer_scorer="math")

    def draw(problem, version=0, role="student"):
        return flow._group_rng(role, problem, version).random()

    ok &= check("eight rollouts of one problem draw alike",
                len({draw("train-1") for _ in range(8)}) == 1)
    ok &= check("different problems can draw differently",
                len({draw(f"train-{i}") for i in range(8)}) > 1)
    ok &= check("a new weight version redraws",
                draw("train-1", 0) != draw("train-1", 1))
    ok &= check("roles are independent streams",
                draw("train-1", 0, "student") != draw("train-1", 0, "teacher"))
    ok &= check("a None version is one stable bucket",
                flow._group_rng("student", "train-1", None).random()
                == flow._group_rng("student", "train-1", None).random())

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
