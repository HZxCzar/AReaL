"""The code student's answer must survive the production scorer.

Replays the real stdout captured by the live run through the real
score_math_answer, before and after code_answer_for_judge.
"""

import glob
import json
import os
import sys

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor.core.math import extract_math_answer, score_math_answer  # noqa: E402
from examples.tutor.workflow import code_answer_for_judge  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {got!r} want {want!r}"))
    if not ok:
        FAILS.append(name)


print("== the bug, stated as a test ==")
check("bare stdout extracts to nothing", extract_math_answer("100"), "")
check("boxed stdout extracts to the answer",
      extract_math_answer(code_answer_for_judge("100")), "100")

print("\n== the last non-empty line is the answer ==")
check("multi-line log -> last line",
      extract_math_answer(code_answer_for_judge("checking 1\nchecking 2\n42")), "42")
check("trailing blank lines ignored",
      extract_math_answer(code_answer_for_judge("7\n\n\n")), "7")
check("empty output stays empty", code_answer_for_judge(""), "")
check("whitespace-only stays empty", code_answer_for_judge("   \n  \n"), "")

print("\n== the full output is preserved for diagnosis ==")
boxed = code_answer_for_judge("checking 1\nchecking 2\n42")
check("log still present", "checking 1" in boxed, True)
check("answer appended", boxed.endswith("\\boxed{42}"), True)

print("\n== braces in the output survive extraction ==")
check("a printed set",
      extract_math_answer(code_answer_for_judge("{0, 1, 4, 9}")), "{0, 1, 4, 9}")

print("\n== REPLAY: the real outputs the live run judged as empty ==")
T = ("/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor"
     "/debug_traces/tutor-math-baseline/20260814_231716_0810fc-two-student-noeval/train")
dec = json.JSONDecoder()
cases = []
for f in sorted(glob.glob(os.path.join(T, "*.json"))):
    try:
        o, _ = dec.raw_decode(open(f, encoding="utf-8").read())
    except Exception:
        continue
    if (o.get("student") or {}).get("mode") != "code":
        continue
    for r in o.get("student_generalization") or []:
        if r.get("level") == "original" and r.get("attempted"):
            out = str(r.get("student_output") or "")
            if out.strip():
                cases.append((o.get("ground_truth", ""), out))

print(f"  {len(cases)} real code-student outputs replayed")
before = sum(1 for gt, out in cases if score_math_answer("", gt, out).correct)
after = sum(1 for gt, out in cases
            if score_math_answer("", gt, code_answer_for_judge(out)).correct)
n = max(len(cases), 1)
print(f"  exact-scorer correct BEFORE the fix   {before:4d}  ({before/n:.1%})")
print(f"  exact-scorer correct AFTER  the fix   {after:4d}  ({after/n:.1%})")
print(f"  recovered                             {after - before:+4d}  ({(after-before)/n:+.1%})")
print("  (exact scorer only -- the LLM judge runs on top and is more lenient,")
print("   so the shipped gain should be at least this)")
check("the fix recovers real score", after > before, True)
check("before was near zero, as the run reported", before / n < 0.02, True)

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
