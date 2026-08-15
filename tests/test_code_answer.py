"""Scoring a code student: the judge sees the program output, not an extraction.

A text student marks its answer with \\boxed{}. A code student's answer is what it
printed. Running the boxed extractor over program output returns an empty string,
so the judge gets asked whether nothing equals the ground truth -- which is what
made the first two-student run score 0.3% when a crude match said 10.9%.
"""

import asyncio
import glob
import json
import os
import sys

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor.core.math import extract_math_answer  # noqa: E402
from examples.tutor.workflow import TutorAgentWorkflow  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {got!r} want {want!r}"))
    if not ok:
        FAILS.append(name)


class Judge:
    """Records what the judge was shown and answers as told."""

    def __init__(self, verdict=True):
        self.verdict = verdict
        self.prompts = []

    async def __call__(self, *, system_prompt, user_prompt, aux_caller, rid_prefix):
        self.prompts.append(user_prompt)
        return type("R", (), {
            "error": None,
            "text": json.dumps({"correct": self.verdict}),
            "raw_text": json.dumps({"correct": self.verdict}),
        })()


def make_workflow(judge=None):
    wf = object.__new__(TutorAgentWorkflow)
    from examples.tutor.core.math import score_math_answer
    for key, value in {
        "dataset_type": "math",
        "answer_scorer": score_math_answer,
        "answer_judge_enabled": True,
        "answer_judge_system_prompt": "judge",
        "_answer_judge_cache": {},
    }.items():
        setattr(wf, key, value)
    if judge is not None:
        wf._call_auxiliary_prompt = judge
    return wf


async def score(wf, gt, out):
    return await wf._score_code_output("task", gt, out, answer_judge_caller=object())


async def main():
    print("== the bug, stated as a test ==")
    check("bare stdout extracts to nothing", extract_math_answer("100"), "")

    print("\n== an exact match short-circuits, costing no judge call ==")
    j = Judge(verdict=False)
    wf = make_workflow(j)
    r = await score(wf, "100", "100")
    check("correct", r.correct, True)
    check("judge was never called", len(j.prompts), 0)

    print("\n== empty output is wrong, and costs no judge call ==")
    j = Judge(verdict=True)          # judge would say correct; must not be asked
    wf = make_workflow(j)
    r = await score(wf, "100", "")
    check("incorrect", r.correct, False)
    check("judge was never called", len(j.prompts), 0)
    r = await score(wf, "100", "   \n \n ")
    check("whitespace-only incorrect", r.correct, False)
    check("still never called", len(j.prompts), 0)

    print("\n== otherwise the judge is shown the output VERBATIM ==")
    j = Judge(verdict=True)
    wf = make_workflow(j)
    log = "No solution for m = -10\nNo solution for m = -9\n100"
    r = await score(wf, "100", log)
    check("one judge call", len(j.prompts), 1)
    shown = j.prompts[0]
    check("the whole log reached the judge", "No solution for m = -10" in shown, True)
    check("including the final line", "100" in shown, True)
    check("ground truth alongside it", "100" in shown, True)
    check("judge verdict is honoured", r.correct, True)

    print("\n== a runaway log is capped so it cannot crowd out the ground truth ==")
    j = Judge(verdict=False)
    wf = make_workflow(j)
    await score(wf, "42", "x\n" * 5000)
    check("truncated", "[truncated]" in j.prompts[0], True)
    check("prompt stays bounded", len(j.prompts[0]) < 4000, True)

    print("\n== REPLAY: real outputs from the live run, exact path only ==")
    T = ("/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output"
         "/tutor/debug_traces/tutor-math-baseline"
         "/20260814_231716_0810fc-two-student-noeval/train")
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

    wf = make_workflow(Judge(verdict=False))   # judge always says no
    exact = 0
    for gt, out in cases:
        if (await score(wf, gt, out)).correct:
            exact += 1
    n = max(len(cases), 1)
    print(f"  {len(cases)} real outputs")
    print(f"  correct on the exact path alone   {exact:4d}  ({exact/n:.1%})")
    print(f"  the live run scored               {0.003:.1%}  (judged on empty strings)")
    check("the exact path alone already beats the broken run", exact / n > 0.05, True)

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


sys.exit(asyncio.run(main()))
