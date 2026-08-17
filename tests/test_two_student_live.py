"""Live check of the production code-student path against the real endpoint.

Drives the actual `_run_student` used in the episode loop, so what is exercised is
the shipped code rather than a probe reimplementation. No GPUs: the teacher's turn
is supplied as a fixed string, which is all the student ever sees anyway.
"""

import asyncio
import os
import sys

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor.core.callers import ApiAuxiliaryCaller  # noqa: E402
from examples.tutor.core.code_exec import CodeSession  # noqa: E402
from examples.tutor.core.types import PublicHistoryState, StudentTurnState  # noqa: E402
from examples.tutor.workflow import TutorAgentWorkflow  # noqa: E402
from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_workflow():
    wf = object.__new__(TutorAgentWorkflow)
    for key, value in {
        "free_chat_enabled": True,
        "free_chat_budget": 5,
        "max_turns": 5,
        "enable_thinking": False,
        "teacher_history_tags": "stripped",
        "dataset_type": "math",
        "student_turn_behavior_enabled": False,
        "student_prompt_pool": (),
        "student_prompt_include_base": False,
    }.items():
        setattr(wf, key, value)
    return wf


def student_caller():
    cfg = AuxModelConfig(
        base_url=os.environ["TUTOR_QWEN3_1_7B_BASE_URL"].strip().strip('"'),
        model="qwen3-1.7b",
        api_key=os.environ["INF_API_KEY"].strip().strip('"'),
        timeout=180,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        max_concurrency=4,
        request_params={
            "extra_headers": {"x-inspire-inference-key": "tutor-train-qwen17b-student"},
            "extra_body": {"top_k": 20, "min_p": 0,
                           "chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    return ApiAuxiliaryCaller(AsyncLLMCaller(cfg))


TEACHER_1 = (
    "Let us work with binary place values. Take the number 1100 in base two and "
    "work out what it is in base ten. Set up the place values yourself."
)
TEACHER_2 = (
    "Good. Now keep the value you just computed and add the base-ten value of "
    "101 to it, reusing what you already defined."
)


def state(mode, history, latest):
    return StudentTurnState(
        task="Add 101_2 + 11_2 + 1100_2 + 11101_2.",
        public_history=PublicHistoryState(
            summary="", turn_count=len(history), turns=list(history)
        ),
        previous_student_output="",
        latest_tutor_visible_output=latest,
        student_mode=mode,
    )


async def main():
    wf = make_workflow()
    caller = student_caller()

    print("== a code student's turn, through the production _run_student ==")
    session = CodeSession()
    history = []
    turn1, err = await wf._run_student(
        state("code", history, TEACHER_1), aux_caller=caller, code_session=session
    )
    check("no error", err is None, str(err or ""))
    check("turn carries a program", "```python" in turn1)
    check("turn carries a result", "[result]" in turn1)
    print("\n--- stored turn 1 ---")
    print("    " + turn1[:600].replace("\n", "\n    "))

    print("\n== turn 2 sees turn 1's namespace (the persistence that matters) ==")
    history.append({"role": "teacher", "content": TEACHER_1})
    history.append({"role": "student", "content": turn1})
    turn2, err2 = await wf._run_student(
        state("code", history, TEACHER_2), aux_caller=caller, code_session=session
    )
    check("no error", err2 is None, str(err2 or ""))
    check("second turn carries a program", "```python" in turn2)
    print("\n--- stored turn 2 ---")
    print("    " + turn2[:600].replace("\n", "\n    "))
    print(f"\n  session kept {session.turns_kept} cells; stats {session.stats()}")
    check("at least one cell was accepted", session.turns_kept >= 1)

    print("\n== the same state as a TEXT student stays prose ==")
    t_turn, t_err = await wf._run_student(
        state("text", [], TEACHER_1), aux_caller=caller, code_session=None
    )
    check("no error", t_err is None, str(t_err or ""))
    check("text student produced no [result] block", "[result]" not in t_turn)
    print("    " + t_turn[:260].replace("\n", "\n    "))

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


sys.exit(asyncio.run(main()))
