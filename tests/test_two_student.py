"""Two-student infrastructure: a text student and a CodeAct student on one model.

Uses the same object.__new__ + setattr idiom as tests/test_tutor_free_chat.py so
no endpoints or GPUs are needed.
"""

import sys
from types import SimpleNamespace

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor import prompts  # noqa: E402
from examples.tutor.configs import (  # noqa: E402
    STUDENT_MODE_CODE, STUDENT_MODE_TEXT, STUDENT_MODES, TutorStudentModelConfig,
)
from examples.tutor.core.types import PublicHistoryState, StudentTurnState  # noqa: E402
from examples.tutor.workflow import (  # noqa: E402
    ORIGINAL_RETEST_LEVEL, TutorAgentWorkflow, extract_program,
)

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {got!r} want {want!r}"))
    if not ok:
        FAILS.append(name)


def make_workflow(**attrs):
    wf = object.__new__(TutorAgentWorkflow)
    defaults = {
        "free_chat_enabled": True,
        "free_chat_budget": 5,
        "max_turns": 5,
        "enable_thinking": False,
        "teacher_history_tags": "stripped",
        "dataset_type": "math",
        "student_turn_behavior_enabled": False,
        "student_prompt_pool": (),
        "student_prompt_include_base": False,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(wf, key, value)
    return wf


def state(mode):
    return StudentTurnState(
        task="Add 101_2 + 11_2.",
        public_history=PublicHistoryState(),
        previous_student_output="",
        latest_tutor_visible_output="Let us look at binary place values.",
        student_mode=mode,
    )


print("== config: mode is a validated first-class field ==")
check("modes are text/code", sorted(STUDENT_MODES), ["code", "text"])
check("default is text",
      TutorStudentModelConfig(name="s", base_url="http://x/v1", model="m").mode, "text")
check("code accepted",
      TutorStudentModelConfig(name="s", base_url="http://x/v1", model="m", mode="CODE").mode,
      "code")
try:
    TutorStudentModelConfig(name="s", base_url="http://x/v1", model="m", mode="pseudocode")
    check("bad mode rejected", "accepted", "ValueError")
except ValueError:
    check("bad mode rejected", True, True)

print("\n== extract_program: the block is what matters, prose is not executed ==")
fence = "Sure, here goes!\n```python\nx = 6 * 7\nx\n```\nHope that helps."
check("prose wrapping a block -> the block", extract_program(fence), "x = 6 * 7\nx\n")
check("bare code with no fence", extract_program("y = 2\ny").strip(), "y = 2\ny")
check("prose only -> None", extract_program("The output will be 7, which means..."), None)
check("empty -> None", extract_program(""), None)
check("unterminated fence still parses",
      extract_program("```python\nz = 1\nz").strip(), "z = 1\nz")
prose_then_bad = "Let me think.\n```python\nThis is not python at all\n```"
check("non-python inside a fence -> None", extract_program(prose_then_bad), None)

print("\n== the two students get different prompts in conversation ==")
wf = make_workflow()
text_msgs = wf._build_student_messages(state(STUDENT_MODE_TEXT))
code_msgs = wf._build_student_messages(state(STUDENT_MODE_CODE))
check("text student system prompt",
      text_msgs[0]["content"], prompts.FREE_CHAT_STUDENT_SYSTEM_PROMPT)
check("code student system prompt",
      code_msgs[0]["content"], prompts.FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT)
check("prompts actually differ",
      text_msgs[0]["content"] != code_msgs[0]["content"], True)
check("code student is told it can only write Python",
      "only act by writing" in code_msgs[0]["content"], True)
check("code student is told state persists",
      "keeps its state" in code_msgs[0]["content"], True)
check("neither student is given the task",
      any("101_2" in m["content"] for m in code_msgs), False)

print("\n== the two students get different re-tests ==")


def probe(mode):
    artifact = SimpleNamespace(
        task="Add 101_2 + 11_2.", student_mode=mode, student_prompt_selection=None,
    )
    anchor = SimpleNamespace(
        public_history=SimpleNamespace(turns=[{"role": "teacher", "content": "hi"}]),
    )
    return wf._build_student_probe_messages(
        episode_artifact=artifact, anchor=anchor,
        level=ORIGINAL_RETEST_LEVEL, transfer_task="",
    )


text_probe, code_probe = probe(STUDENT_MODE_TEXT), probe(STUDENT_MODE_CODE)
check("text re-test asks for a boxed answer",
      "boxed" in text_probe[-1]["content"], True)
check("code re-test asks for a program",
      "Python program" in code_probe[-1]["content"], True)
check("code re-test does NOT demand a print (the session echoes)",
      "print" in code_probe[-1]["content"].lower(), False)
check("both re-tests finally show the task",
      all("101_2" in m[-1]["content"] for m in (text_probe, code_probe)), True)
check("code re-test carries the code system prompt",
      code_probe[0]["content"], prompts.FREE_CHAT_CODE_STUDENT_SYSTEM_PROMPT)

print("\n== teacher's view of a code turn shows program AND result ==")
view = prompts.CODE_STUDENT_TEACHER_VIEW_TEMPLATE
check("template has a program slot", "{{ program }}" in view, True)
check("template has a result slot", "{{ result }}" in view, True)

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}"))
sys.exit(1 if FAILS else 0)
