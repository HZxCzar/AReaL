#!/usr/bin/env python3
"""Reconstruct what a student ACTUALLY saw, from a live run's debug traces.

The traces record the shared public history and the student's replies, not the
masked view -- the mask is applied on read, so nothing writes it down. It is a
deterministic pure function of the history though, so replaying it against the
recorded transcript reproduces the student's context exactly.

    python dump_live_masked_views.py --trial 20260816_070410_0810fc-masked-students
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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
    render_prompt,
)

MASKS = {
    "student-fade": {"mode": "student_fade", "keep_recent": 0},
    "teacher-fade": {"mode": "teacher_fade", "keep_recent": 0},
    "long-drop": {"mode": "long_drop", "long_drop_words": 75},
    "full-control": {"mode": "full"},
}
ROOT = (
    "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/"
    "tutor/debug_traces/tutor-math-baseline"
)


def show(messages, width=100, clip=None):
    for message in messages:
        body = " ".join((message["content"] or "").split())
        count = len((message["content"] or "").split())
        if clip and len(body) > clip:
            body = body[:clip] + " ..."
        head = f"    [{message['role']:9s} {count:4d}w] "
        room = width - len(head)
        print(head + body[:room])
        rest = body[room:]
        while rest:
            print(" " * len(head) + rest[:room])
            rest = rest[room:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--scan", type=int, default=900)
    parser.add_argument("--turn", type=int, default=3, help="1-indexed turn to show")
    parser.add_argument("--clip", type=int, default=420)
    args = parser.parse_args()

    files = sorted(
        glob.glob(os.path.join(ROOT, args.trial, "train", "*.json")),
        key=os.path.getmtime,
    )[-args.scan :]
    decoder = json.JSONDecoder()
    picked: dict[str, dict] = {}
    for path in reversed(files):
        if len(picked) == len(MASKS):
            break
        try:
            # First document only: these files carry overwrite residue.
            obj, _ = decoder.raw_decode(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        student = obj.get("student") or {}
        name = str(student.get("name") if isinstance(student, dict) else student or "")
        if name in MASKS and name not in picked and len(obj.get("turns") or []) >= args.turn:
            picked[name] = obj

    system = "\n\n".join((FREE_CHAT_STUDENT_SYSTEM_PROMPT, FREE_CHAT_STUDENT_MASK_NOTE))
    for name, mask_cfg in MASKS.items():
        obj = picked.get(name)
        if obj is None:
            print(f"\n(no episode found for {name})")
            continue
        mask = normalize_mask(mask_cfg)
        turns = obj["turns"]
        target = turns[args.turn - 1]
        history = list(target.get("public_history_before") or [])
        live = target.get("tutor_visible_output") or ""

        print("\n" + "=" * 100)
        print(f"{name}   task_id={obj.get('task_id')}   turns={len(turns)}   "
              f"reward={obj.get('total_reward')}")
        print("=" * 100)
        print(f"  --- WHAT THE TEACHER SEES at turn {args.turn} (always unmasked) ---")
        show([{"role": t["role"], "content": t["content"]} for t in history]
             + [{"role": "teacher", "content": live}], clip=args.clip)
        print(f"  --- WHAT THE STUDENT SEES at turn {args.turn} ---")
        visible = apply_student_mask(history, mask)
        messages = [{"role": "system", "content": system}]
        messages += [
            {"role": "assistant" if t["role"] == "student" else "user",
             "content": t["content"]}
            for t in visible
        ]
        messages.append({"role": "user", "content": mask_current_turn(live, mask)})
        show(messages, clip=args.clip)
        print("  --- ITS ACTUAL REPLY ---")
        show([{"role": "student", "content": target.get("student_output") or ""}],
             clip=args.clip)

        final = turns[-1]
        full_history = list(final.get("public_history_after") or [])
        print("  --- THE RE-TEST IT WAS SCORED ON ---")
        retest = [{"role": "system", "content": system}]
        retest += [
            {"role": "assistant" if t["role"] == "student" else "user",
             "content": t["content"]}
            for t in apply_student_mask(full_history, mask)
        ]
        retest.append({
            "role": "user",
            "content": render_prompt(
                FREE_CHAT_STUDENT_RETEST_TEMPLATE, task=obj.get("task", "")
            ),
        })
        show(retest, clip=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
