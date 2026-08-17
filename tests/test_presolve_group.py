"""One pre-solve per group per step, and one student per group.

Both of these used to be drawn per rollout, which put them straight into the GRPO
advantage: `actor.group_baseline='episode'` subtracts a group's mean return, so
anything that varies inside a group and that the teacher cannot control is noise
the policy is asked to explain. Measured on 20260815_065439, the student draw
alone was 29-33% of the within-group spread.

Same object.__new__ + setattr idiom as tests/test_two_student.py: no endpoints,
no GPUs, no config composition.
"""

import asyncio
import sys
from types import SimpleNamespace

W = "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student"
sys.path.insert(0, W)

from examples.tutor.configs import TutorTeacherPreConfig  # noqa: E402
from examples.tutor.core.types import TeacherPreSolveResult  # noqa: E402
from examples.tutor.workflow import (  # noqa: E402
    TEACHER_PRE_CACHE_MAX_ENTRIES,
    TutorAgentWorkflow,
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
        "prompt_pool_seed": 17,
        "teacher_pre_enabled": True,
        "teacher_pre_on_reject": "skip",
        "student_model_runtimes": {},
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(wf, key, value)
    wf._teacher_pre_solve_cache = {}
    wf._teacher_pre_solve_lock = asyncio.Lock()
    return wf


def runtimes(*specs):
    """name -> runtime, in the order student_models declares them."""
    return {
        name: SimpleNamespace(
            name=name, model=name, weight=weight, mode=mode,
            caller=f"caller:{name}", confidence_caller=None,
        )
        for name, weight, mode in specs
    }


TWO = ("qwen3-1.7b", 0.5, "text"), ("qwen3-1.7b-code", 0.5, "code")


# ---------------------------------------------------------------------------
print("== teacher_pre.on_reject is validated at config load ==")

check("default is the historical skip", TutorTeacherPreConfig().on_reject, "skip")
check("continue accepted", TutorTeacherPreConfig(on_reject="CONTINUE").on_reject, "continue")
try:
    TutorTeacherPreConfig(on_reject="retry")
    check("a bad mode is rejected", "accepted", "ValueError")
except ValueError:
    check("a bad mode is rejected", True, True)


# ---------------------------------------------------------------------------
print("\n== the pre-solve runs once per (problem, weight version) ==")


def presolve_probe(wf):
    """Replace the real pre-solve with a counter. Sleeps so all eight rollouts of
    a group are genuinely in flight at once -- a cache that only works when the
    callers happen to be serialised would pass without this."""
    calls = []

    async def fake(task, ground_truth, *, actor_caller, answer_judge_caller, lora_version):
        calls.append((task, lora_version))
        await asyncio.sleep(0.01)
        return TeacherPreSolveResult(
            enabled=True, mode="filter_solver", accepted=True, attempts=[],
            raw_output=f"draft for {task} v{lora_version}", error=None,
            verification_enabled=True,
        )

    wf._run_teacher_pre_solve = fake
    return calls


async def gather_group(wf, *, task="P1", version=3, n=8, group_key="id-1"):
    return await asyncio.gather(*[
        wf._teacher_pre_solve_for_group(
            task, "gt", actor_caller=None, answer_judge_caller=None,
            lora_version=version, group_key=group_key,
        )
        for _ in range(n)
    ])


wf = make_workflow()
calls = presolve_probe(wf)
results = asyncio.run(gather_group(wf))
check("eight concurrent rollouts -> one generation", len(calls), 1)
check("every rollout gets the same draft object", len({id(r) for r, _ in results}), 1)
check("exactly one rollout is the cache miss", sum(1 for _, hit in results if not hit), 1)
check("the other seven are hits", sum(1 for _, hit in results if hit), 7)
check("cache_hit rate is (n-1)/n", sum(hit for _, hit in results) / len(results), 0.875)

# A weight update has to invalidate it: the draft is a function of the weights.
asyncio.run(gather_group(wf, version=4))
check("the next weight version regenerates", len(calls), 2)
check("and it was asked at the new version", calls[-1][1], 4)

# The same version again is still cached, so a group split across two waves of
# the same step does not pay twice.
asyncio.run(gather_group(wf, version=3, n=4))
check("re-asking an old version stays cached", len(calls), 2)

# Different problems never share.
asyncio.run(gather_group(wf, task="P2", version=3, group_key="id-2"))
check("a different problem generates its own", len(calls), 3)


# ---------------------------------------------------------------------------
print("\n== a transient failure does not poison the group for the whole step ==")

wf = make_workflow()
attempts = {"n": 0}


async def flaky(task, ground_truth, *, actor_caller, answer_judge_caller, lora_version):
    attempts["n"] += 1
    if attempts["n"] == 1:
        raise RuntimeError("endpoint hiccup")
    return TeacherPreSolveResult(
        enabled=True, mode="filter_solver", accepted=True, attempts=[],
        raw_output="second try", error=None, verification_enabled=True,
    )


wf._run_teacher_pre_solve = flaky


async def one():
    try:
        await wf._teacher_pre_solve_for_group(
            "P", "gt", actor_caller=None, answer_judge_caller=None,
            lora_version=1, group_key="k",
        )
        return "no-raise"
    except RuntimeError:
        return "raised"


check("the first caller sees the error", asyncio.run(one()), "raised")
check("the failed entry was dropped", len(wf._teacher_pre_solve_cache), 0)
result, hit = asyncio.run(
    wf._teacher_pre_solve_for_group(
        "P", "gt", actor_caller=None, answer_judge_caller=None,
        lora_version=1, group_key="k",
    )
)
check("a sibling retries and succeeds", result.raw_output, "second try")
check("and it is a miss, not a cached exception", hit, False)


# ---------------------------------------------------------------------------
print("\n== the cache is bounded, and deeper than one step ==")

wf = make_workflow()
presolve_probe(wf)


async def many():
    for i in range(TEACHER_PRE_CACHE_MAX_ENTRIES + 40):
        await wf._teacher_pre_solve_for_group(
            f"P{i}", "gt", actor_caller=None, answer_judge_caller=None,
            lora_version=1, group_key=f"k{i}",
        )


asyncio.run(many())
check("bounded at the cap", len(wf._teacher_pre_solve_cache), TEACHER_PRE_CACHE_MAX_ENTRIES)
check("the cap clears one step's 16 problems by a wide margin",
      TEACHER_PRE_CACHE_MAX_ENTRIES >= 16 * 4, True)
check("the newest group survived", ("k295", 1) in wf._teacher_pre_solve_cache, True)


# ---------------------------------------------------------------------------
print("\n== the student is drawn once per group, not once per rollout ==")

wf = make_workflow(student_model_runtimes=runtimes(*TWO))


def draw(group_key, version, n=8):
    return {
        wf._select_student({"id": group_key}, aux_caller=None,
                           group_key=group_key, rollout_version=version).name
        for _ in range(n)
    }


check("all eight rollouts of a group get one student", len(draw("id-1", 3)), 1)
check("still one at another problem", len(draw("id-2", 3)), 1)

# Over problems the mix has to stay at the configured weights, and over versions a
# problem must not be pinned to one student for every epoch of the run.
names = [
    wf._select_student({"id": f"id-{i}"}, aux_caller=None,
                       group_key=f"id-{i}", rollout_version=3).name
    for i in range(400)
]
share = names.count("qwen3-1.7b-code") / len(names)
check("the 50/50 weights still hold across problems", 0.42 < share < 0.58, True)

per_version = {
    wf._select_student({"id": "id-1"}, aux_caller=None,
                       group_key="id-1", rollout_version=v).name
    for v in range(40)
}
check("one problem is not pinned to one student for the whole run",
      len(per_version), 2)

# A single-student arm cannot be affected by any of this.
solo = make_workflow(student_model_runtimes=runtimes(("qwen3-1.7b", 1.0, "text")))
check("a one-student arm draws the same student either way",
      {solo._select_student({"id": "x"}, aux_caller=None, group_key="x",
                            rollout_version=v).name for v in range(20)},
      {"qwen3-1.7b"})


print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    raise SystemExit(1)
print("all pre-solve group checks passed")
