#!/usr/bin/env python3
"""Print the exact message lists each party sees under each student mask.

Uses the real prompt constants and the real mask functions, so what this prints
is what the rollout actually sends. A three-round episode, then the re-test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.tutor.core.attention_mask import (  # noqa: E402
    apply_student_mask,
    mask_current_turn,
    normalize_mask,
)
from examples.tutor.prompts import (  # noqa: E402
    FREE_CHAT_STUDENT_MASK_NOTE,
    FREE_CHAT_STUDENT_RETEST_TEMPLATE,
    FREE_CHAT_STUDENT_SYSTEM_PROMPT,
    FREE_CHAT_TEACHER_OPEN_PROMPT,
    FREE_CHAT_TEACHER_SYSTEM_PROMPT,
    TEACHER_HISTORY_MASKED_TEMPLATE,
    render_prompt,
)

TASK = (
    "Place each of the digits 6, 7, 8 and 9 in exactly one square to make the "
    "smallest possible product."
)
T1 = (
    "Look at the shape of the problem first. You are building two two-digit numbers "
    "and multiplying them, so each digit lands in either a tens place or a ones "
    "place. A digit in a tens place is worth ten times what it is worth in a ones "
    "place, which means the tens places dominate the size of the product. To make "
    "the product small you therefore want the two smallest digits in the tens "
    "places. That gives you a much smaller set of candidates to compare than the "
    "twenty-four arrangements you started with."
)
S1 = "So I should put 6 and 7 in the tens places?"
T2 = "Yes. Now list the arrangements that leaves and compare them."
S2 = "68 times 79, and 69 times 78."
T3 = "Compute both products and say which one is smaller."
S3 = "68 times 79 is 5372, and 69 times 78 is 5382."

HISTORY = [
    {"role": "teacher", "content": T1},
    {"role": "student", "content": S1},
    {"role": "teacher", "content": T2},
    {"role": "student", "content": S2},
]
FULL_EPISODE = HISTORY + [
    {"role": "teacher", "content": T3},
    {"role": "student", "content": S3},
]


def show(messages, width=104):
    for message in messages:
        body = " ".join(message["content"].split())
        count = len(message["content"].split())
        head = f"  [{message['role']:9s} {count:3d}w] "
        room = width - len(head)
        if len(body) <= room:
            print(head + body)
        else:
            print(head + body[:room])
            pad = " " * len(head)
            rest = body[room:]
            while rest:
                print(pad + rest[:room])
                rest = rest[room:]


MASKED_SYSTEM = "\n\n".join(
    (FREE_CHAT_STUDENT_SYSTEM_PROMPT, FREE_CHAT_STUDENT_MASK_NOTE)
)


def to_student_messages(turns, live=None):
    # Every student in a masked run gets the note, control included, so the
    # control differs by its mask alone.
    messages = [{"role": "system", "content": MASKED_SYSTEM}]
    for turn in turns:
        messages.append(
            {
                "role": "assistant" if turn["role"] == "student" else "user",
                "content": turn["content"],
            }
        )
    if live is not None:
        messages.append({"role": "user", "content": live})
    return messages


def main() -> int:
    banner = "=" * 104
    print(banner)
    print("THE TEACHER AT TURN 3 -- identical under every mask; it never sees the masking")
    print(banner)
    teacher = [
        {
            "role": "system",
            "content": render_prompt(
                FREE_CHAT_TEACHER_SYSTEM_PROMPT,
                budget=5,
                task=TASK,
                # Added on this branch by the student-unaware ablation; empty is
                # the default (the student IS told a problem is coming).
                student_problem_context="",
            ),
        },
        {"role": "user", "content": FREE_CHAT_TEACHER_OPEN_PROMPT},
    ]
    for turn in HISTORY:
        teacher.append(
            {
                "role": "assistant" if turn["role"] == "teacher" else "user",
                "content": (
                    TEACHER_HISTORY_MASKED_TEMPLATE.format(visible=turn["content"])
                    if turn["role"] == "teacher"
                    else turn["content"]
                ),
            }
        )
    show(teacher)

    for mode in ("full", "student_fade", "teacher_fade", "long_drop"):
        mask = normalize_mask(mode)
        print()
        print(banner)
        print(f"STUDENT AT TURN 3   mask = {mode}")
        print(banner)
        show(
            to_student_messages(
                apply_student_mask(HISTORY, mask), mask_current_turn(T3, mask)
            )
        )

    for mode in ("full", "student_fade", "teacher_fade", "long_drop"):
        mask = normalize_mask(mode)
        print()
        print(banner)
        print(f"RE-TEST (reward is scored on this)   mask = {mode}")
        print(banner)
        show(
            to_student_messages(
                apply_student_mask(FULL_EPISODE, mask),
                render_prompt(FREE_CHAT_STUDENT_RETEST_TEMPLATE, task=TASK),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
