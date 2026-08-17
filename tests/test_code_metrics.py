"""The code channel's health must reach the metrics, and only for a code student.

These four counters are the arm's validity check: a rise in the code student's
score is ambiguous without them, because it could be better teaching or it could
be the teacher handing over an answer the student merely prints. So the emission
itself is worth a test.
"""

import sys

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor.core.code_exec import CodeSession  # noqa: E402
from examples.tutor.workflow import TutorAgentWorkflow  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {got!r} want {want!r}"))
    if not ok:
        FAILS.append(name)


captured = {}


def make_workflow():
    wf = object.__new__(TutorAgentWorkflow)
    for key, value in {
        "free_chat_enabled": True,
        "student_generalize_enabled": False,
        "eval_average_rollouts": 1,
        "debug_trace_dir": "",
    }.items():
        setattr(wf, key, value)
    return wf


class Trace:
    """Minimal stand-in for TurnTrace: only the fields the metrics path reads."""

    def __init__(self, idx):
        self.turn_idx = idx
        self.judge_correct = False
        self.invalid_due_to_leak = False
        self.tutor_format_error = None
        self.tutor_visible_output = f"teacher turn {idx}"
        self.student_output = ""
        self.student_error = None
        self.leaked = False
        self.leak_level = 0
        self.judge_feedback = ""
        self.reward = 0.0
        self.reward_components = {}
        self.previous_teacher_similarity = None
        self.teacher_similarity_error = None


def run(code_stats, n_turns=5):
    wf = make_workflow()
    captured.clear()
    # _safe_scalar is what forwards to stats_tracker; intercept it instead of
    # standing up a tracker.
    import examples.tutor.workflow as mod
    real = mod._safe_scalar
    mod._safe_scalar = lambda **m: captured.update(m)
    try:
        wf._log_rollout_stats(
            total_reward=0.0,
            traces=[Trace(i + 1) for i in range(n_turns)],
            termination_reason="max_turns",
            pre_success=False,
            leak_count=0,
            code_stats=code_stats,
        )
    finally:
        mod._safe_scalar = real
    return {k.split("/", 1)[-1] if k.startswith("rollout/") else k: v
            for k, v in captured.items()}


print("== a text student emits nothing, so the series is never padded with zeros ==")
m = run(None)
check("no code/* keys for a text student",
      sorted(k for k in m if k.startswith("code/")), [])

print("\n== a healthy code student ==")
m = run({"crashes": 0, "silent_cells": 0, "constant_prints": 0, "no_program": 0})
check("crashes emitted", m.get("code/crashes"), 0.0)
check("silent emitted", m.get("code/silent_cells"), 0.0)
check("constant prints emitted", m.get("code/constant_prints"), 0.0)
check("no_program emitted", m.get("code/no_program"), 0.0)
check("all 5 turns productive", m.get("code/productive_per_turn"), 1.0)

print("\n== the failure that would quietly invalidate the arm ==")
m = run({"crashes": 0, "silent_cells": 0, "constant_prints": 3, "no_program": 0})
check("constant_prints surfaces", m.get("code/constant_prints"), 3.0)
check("as a per-turn rate", m.get("code/constant_prints_per_turn"), 0.6)
check("productive is unaffected (they did run)",
      m.get("code/productive_per_turn"), 1.0)

print("\n== a broken channel ==")
m = run({"crashes": 2, "silent_cells": 1, "constant_prints": 0, "no_program": 1})
check("crash rate", m.get("code/crashes_per_turn"), 0.4)
check("silent rate", m.get("code/silent_cells_per_turn"), 0.2)
check("no_program rate", m.get("code/no_program_per_turn"), 0.2)
check("productive drops to 1 of 5", m.get("code/productive_per_turn"), 0.2)

print("\n== productive never goes negative even if every turn failed ==")
m = run({"crashes": 5, "silent_cells": 3, "constant_prints": 0, "no_program": 2})
check("clamped at zero", m.get("code/productive_per_turn"), 0.0)

print("\n== a real session's tally flows straight through ==")
sess = CodeSession()
sess.crashes, sess.silent_cells, sess.constant_prints, sess.no_program = 1, 0, 2, 0
m = run(sess.stats())
check("session.stats() shape matches what metrics expect",
      sorted(k[5:] for k in m if k.startswith("code/") and not k.endswith("_per_turn")
             and k != "code/productive_per_turn"),
      sorted(["crashes", "constant_prints", "no_program", "silent_cells"]))

print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
