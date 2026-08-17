#!/usr/bin/env python3
"""Pre-solve on-policy distillation: the draft leaves the prompt and enters the loss.

Two keys turn this on -- `teacher_pre.visibility: opd_only` and
`opd.context: presolve` -- and the whole claim rests on two prompts:

  * the ROLLOUT prompt must be byte-identical to the no-pre-solve arm's, or the
    policy being trained is not the deployed teacher;
  * the OPD TEACHER prompt must differ from it by exactly the two pre-solve
    messages, or the reverse KL is not attributable to the draft.

Both are checked here by construction rather than by comparing against literal
text, so a later prompt edit moves both sides and these still mean something.

The third thing checked is that the switch OFF changes nothing. Every default is
asserted against the historical value and the preamble is exercised with the
switch absent, because two arms are training against this code right now.

ONE DEFAULT DELIBERATELY IS NOT HISTORICAL. `teacher_pre.share_per_group` went
False -> True on 20260815 as a bug fix, not as a switch: sharing is a property of
how a GRPO group is built, and an arm that did not opt in silently handed its
group `gconfig.n_samples` different teacher prompts. Section [9] asserts the new
default and covers the in-prompt arms that were affected.

Run from the repo root with the venv and .env active and PYTHONPATH set to the
repo root; run_official.sh does this.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from dataclasses import asdict
from types import SimpleNamespace

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import (
    TutorConfig,
    TutorOpdConfig,
    TutorTeacherPreConfig,
)
from examples.tutor.core.types import (
    PublicHistoryState,
    TeacherPreSolveResult,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import FREE_CHAT_TEACHER_SOLVE_PROMPT
from areal.infra.workflow_context import _current_context
from examples.tutor.configs import TutorEvaluatorConfig
from examples.tutor.train import _build_eval_workflow_kwargs
from examples.tutor.workflow import TutorAgentWorkflow

FAILURES: list[str] = []
BASE = "examples/tutor/configs/math/0810"
TASK = "Let f(x) = x^2 - 4x + 7. What is the minimum value of f?"
GROUND_TRUTH = "3"
DRAFT = "Complete the square: f(x) = (x-2)^2 + 3, so the minimum is \\boxed{3}."
HISTORY = [
    {"role": "teacher", "content": "What shape does a quadratic have?"},
    {"role": "student", "content": "A parabola, I think."},
]


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def make_workflow(**attrs: object) -> TutorAgentWorkflow:
    """Only the attributes the methods under test read. See test_tutor_free_chat."""
    workflow = object.__new__(TutorAgentWorkflow)
    defaults = {
        "free_chat_enabled": True,
        "free_chat_budget": 5,
        "max_turns": 5,
        "free_chat_student_has_not_seen_problem": False,
        "enable_thinking": False,
        "teacher_show_ground_truth": False,
        "teacher_anti_leak_instruction_enabled": True,
        "teacher_adaptive_instruction_enabled": False,
        "teacher_history_tags": "masked",
        "dataset_type": "math",
        # The switch, off.
        "teacher_pre_enabled": True,
        "teacher_pre_visibility": "rollout",
        "teacher_pre_on_reject": "skip",
        "teacher_pre_share_per_group": False,
        "opd_context": "instruction",
        "opd_skip_guided_rows": True,
        "opd_skip_leaked_rows": True,
        "opd_min_prior_failed_turns": 0,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(workflow, key, value)
    return workflow


def pre_solve(*, accepted: bool = True, raw_output: str = DRAFT):
    return TeacherPreSolveResult(
        enabled=True,
        mode="filter_solver",
        accepted=accepted,
        attempts=[],
        raw_output=raw_output,
        error=None,
        verification_enabled=True,
    )


def tutor_state(*, turn_idx: int = 2, pre=None) -> TutorTurnState:
    return TutorTurnState(
        task=TASK,
        ground_truth=GROUND_TRUTH,
        public_history=PublicHistoryState(turns=list(HISTORY)),
        previous_tutor_visible_output="",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=turn_idx,
        max_turns=5,
        teacher_pre_solve_result=pre,
    )


def artifact(state: TutorTurnState) -> SimpleNamespace:
    return SimpleNamespace(
        turn_idx=state.turn_idx,
        tutor_state=state,
        leak_result=SimpleNamespace(leaked=False),
        invalid_due_to_leak=False,
        tutor_response=SimpleNamespace(output_tokens=[1, 2, 3]),
        opd_skip_reason="",
        opd_prompt_tokens=None,
    )


def load(allocation: str, name: str) -> TutorConfig:
    config, _ = load_expr_config(
        ["--config", f"{BASE}/{allocation}/{name}.yaml"], TutorConfig
    )
    return config



def sharing_workflow(**attrs):
    workflow = make_workflow(**attrs)
    workflow._teacher_pre_solve_shared = {}
    workflow._teacher_pre_solve_shared_lock = asyncio.Lock()
    workflow.calls = 0

    async def fake_run(task, ground_truth, **kwargs):
        workflow.calls += 1
        # Read the counter BEFORE yielding: all eight coroutines interleave at
        # the await, so reading it after would hand every one of them the final
        # count and make distinct drafts look identical.
        nth = workflow.calls
        await asyncio.sleep(0)
        return pre_solve(raw_output=f"draft {nth}")

    workflow._run_teacher_pre_solve = fake_run
    return workflow


def gather_group(workflow, *, version, n=8, problem="p1"):
    """The group's drafts. Unwraps the (result, cache_hit) pair the workflow returns."""
    async def go():
        return await asyncio.gather(*[
            workflow._teacher_pre_solve_for_group(
                TASK,
                GROUND_TRUTH,
                actor_caller=None,
                answer_judge_caller=None,
                lora_version=version,
                group_key=problem,
            )
            for _ in range(n)
        ])

    return [result for result, _cache_hit in asyncio.run(go())]



class _FakeContext:
    """The minimum a rollout context needs for the eval-only code paths."""

    def __init__(self, *, is_eval: bool = False):
        self.is_eval = is_eval
        self.task_id = 0
        self.lora_version = None


def in_context(is_eval: bool):
    return _current_context.set(_FakeContext(is_eval=is_eval))


def main() -> int:
    print("\n[1] the switch is off by default and the defaults are the old values")
    pre_cfg = TutorTeacherPreConfig()
    opd_cfg = TutorOpdConfig()
    check("teacher_pre.visibility defaults to rollout", pre_cfg.visibility == "rollout")
    check("teacher_pre.on_reject defaults to skip", pre_cfg.on_reject == "skip")
    check("opd.context defaults to instruction", opd_cfg.context == "instruction")

    for bad, field in (("both", "visibility"), ("maybe", "on_reject")):
        try:
            TutorTeacherPreConfig(**{field: bad})
        except ValueError:
            check(f"teacher_pre.{field} rejects {bad!r}", True)
        else:
            check(f"teacher_pre.{field} rejects {bad!r}", False, "no raise")
    try:
        TutorOpdConfig(context="draft")
    except ValueError:
        check("opd.context rejects an unknown value", True)
    else:
        check("opd.context rejects an unknown value", False, "no raise")

    print("\n[2] switch off: the draft is still in the rollout prompt")
    off = make_workflow()
    state = tutor_state(pre=pre_solve())
    off_preamble = off._free_chat_preamble(state)
    check(
        "preamble is request + draft + open prompt",
        len(off_preamble) == 3
        and off_preamble[0]["content"] == FREE_CHAT_TEACHER_SOLVE_PROMPT
        and off_preamble[1]["content"] == DRAFT,
        f"{[m['role'] for m in off_preamble]}",
    )
    check(
        "a rejected draft is absent, as before",
        len(off._free_chat_preamble(tutor_state(pre=pre_solve(accepted=False)))) == 1,
    )
    check(
        "teacher_pre_enabled=False is unchanged",
        len(make_workflow(teacher_pre_enabled=False)._free_chat_preamble(state)) == 1,
    )
    check(
        "no_presolve is not a skip reason under context=instruction",
        off._opd_skip_reason(artifact(tutor_state(pre=pre_solve(accepted=False))))
        != "no_presolve",
    )

    print("\n[3] switch on: the rollout prompt equals the no-pre-solve arm's")
    on = make_workflow(teacher_pre_visibility="opd_only", opd_context="presolve")
    nopre = make_workflow(teacher_pre_enabled=False)
    rollout = on._build_tutor_messages(state, clean=True, include_guidance=False)
    control = nopre._build_tutor_messages(state, clean=True, include_guidance=False)
    check(
        "rollout prompt is byte-identical to teacher_pre.enabled=false",
        rollout == control,
        f"{len(rollout)} vs {len(control)} messages",
    )
    check("the draft appears nowhere in it", not any(DRAFT in m["content"] for m in rollout))

    print("\n[4] switch on: the OPD prompt differs by exactly the two messages")
    teacher = on._build_tutor_messages(
        state, clean=True, include_guidance=False, presolve_visible=True
    )
    check("two messages longer", len(teacher) == len(rollout) + 2, f"{len(teacher)}")
    check("system turn untouched", teacher[0] == rollout[0])
    check(
        "the request is inserted at index 1",
        teacher[1] == {"role": "user", "content": FREE_CHAT_TEACHER_SOLVE_PROMPT},
    )
    check(
        "the draft is the assistant turn that answers it",
        teacher[2] == {"role": "assistant", "content": DRAFT},
    )
    check(
        "everything after it is the rollout prompt, unchanged",
        teacher[3:] == rollout[1:],
    )
    check(
        "so the open prompt still starts the conversation",
        teacher[3]["role"] == "user" and teacher[3] == rollout[1],
    )

    print("\n[5] presolve_visible overrides the config in both directions")
    check(
        "None follows visibility=opd_only and hides it",
        len(on._free_chat_preamble(state, presolve_visible=None)) == 1,
    )
    check(
        "True forces it in",
        len(on._free_chat_preamble(state, presolve_visible=True)) == 3,
    )
    check(
        "False forces it out even under visibility=rollout",
        len(off._free_chat_preamble(state, presolve_visible=False)) == 1,
    )

    print("\n[6] a turn with no accepted draft is skipped, not supervised for free")
    check(
        "rejected draft -> no_presolve",
        on._opd_skip_reason(artifact(tutor_state(pre=pre_solve(accepted=False))))
        == "no_presolve",
    )
    check(
        "missing result -> no_presolve",
        on._opd_skip_reason(artifact(tutor_state(pre=None))) == "no_presolve",
    )
    check(
        "empty draft text -> no_presolve",
        on._opd_skip_reason(artifact(tutor_state(pre=pre_solve(raw_output="  "))))
        == "no_presolve",
    )
    check(
        "an accepted draft is supervised",
        on._opd_skip_reason(artifact(tutor_state(pre=pre_solve()))) == "",
    )

    print("\n[7] the arm loads, and half-configured versions of it do not")
    try:
        config = load("4gpu", "presolve-opd")
    except Exception as exc:  # noqa: BLE001
        check("4gpu/presolve-opd loads", False, str(exc)[:200])
        return 1
    check("4gpu/presolve-opd loads", True)
    check("visibility is opd_only", config.teacher_pre.visibility == "opd_only")
    check("opd context is presolve", config.opd.context == "presolve")
    check("teacher_pre is on", config.teacher_pre.enabled)
    check("opd is on", config.opd.enabled)
    check(
        "on_reject keeps the episode set equal to -nopre's",
        config.teacher_pre.on_reject == "continue",
    )
    check(
        "supervision starts at turn 1",
        config.opd.min_prior_failed_turns == 0,
    )
    check(
        "free_chat is on, which the presolve context requires",
        config.free_chat.enabled,
    )

    for label, mutate, needle in (
        ("visibility without the opd context", lambda c: setattr(c.opd, "context", "instruction"), "opd_only"),
        ("the opd context without visibility", lambda c: setattr(c.teacher_pre, "visibility", "rollout"), "presolve"),
        ("the opd context without a pre-solve", lambda c: setattr(c.teacher_pre, "enabled", False), "teacher_pre.enabled"),
        ("the opd context without free chat", lambda c: setattr(c.free_chat, "enabled", False), "free_chat.enabled"),
    ):
        broken = load("4gpu", "presolve-opd")
        mutate(broken)
        try:
            broken.__post_init__()
        except ValueError as exc:
            check(f"rejects {label}", needle in str(exc), str(exc)[:160])
        else:
            check(f"rejects {label}", False, "no raise")

    print("\n[8] every pre-existing 0810 arm still loads unchanged")
    for allocation, arms in (
        ("2gpu", ("base", "leak-reward", "leak-terminate")),
        ("4gpu", ("base", "leak-reward", "leak-terminate", "turn-credit")),
    ):
        for name in arms:
            try:
                other = load(allocation, name)
            except Exception as exc:  # noqa: BLE001
                check(f"{allocation}/{name} loads", False, str(exc)[:160])
                continue
            check(
                f"{allocation}/{name} keeps the old pre-solve behaviour",
                other.teacher_pre.visibility == "rollout"
                and other.teacher_pre.on_reject == "skip"
                and other.opd.context == "instruction",
            )

    print("\n[9] share_per_group: one draft per problem per weight version")
    off_share = sharing_workflow()
    drafts = gather_group(off_share, version=3)
    check(
        "off: one generation per rollout, as before",
        off_share.calls == 8 and len({d.raw_output for d in drafts}) == 8,
        f"calls={off_share.calls}",
    )

    on_share = sharing_workflow(teacher_pre_share_per_group=True)
    drafts = gather_group(on_share, version=3)
    check(
        "on: 8 concurrent rollouts share one generation",
        on_share.calls == 1 and len({d.raw_output for d in drafts}) == 1,
        f"calls={on_share.calls}",
    )
    gather_group(on_share, version=3)
    check("a second wave at the same version reuses it", on_share.calls == 1)
    gather_group(on_share, version=4)
    check("a weight update regenerates", on_share.calls == 2, f"calls={on_share.calls}")
    gather_group(on_share, version=4, problem="p2")
    check("a different problem gets its own", on_share.calls == 3)
    gather_group(on_share, version=9)
    check(
        "versions older than the previous one are dropped",
        all(v >= 8 for _, v in on_share._teacher_pre_solve_shared),
        f"{sorted(on_share._teacher_pre_solve_shared)}",
    )

    no_version = sharing_workflow(teacher_pre_share_per_group=True)
    gather_group(no_version, version=None)
    check(
        "no version to key on falls back to per-rollout",
        no_version.calls == 8,
        f"calls={no_version.calls}",
    )
    check(
        "the arm turns sharing on",
        load("4gpu", "presolve-opd").teacher_pre.share_per_group,
    )
    # Was False until 20260815, which is the bug this asserts against: sharing is
    # a property of how GRPO groups are built, not a per-arm choice, and an arm
    # that forgot to opt in silently gave its group n_samples different prompts.
    check(
        "share_per_group defaults ON",
        TutorTeacherPreConfig().share_per_group is True,
    )
    check(
        "the workflow's own default matches the config's",
        inspect.signature(TutorAgentWorkflow.__init__)
        .parameters["teacher_pre_share_per_group"]
        .default
        is True,
    )
    # The regression. Every one of these puts the draft IN the prompt
    # (visibility 'rollout', the default) and none of them mentions
    # share_per_group, which is exactly the shape that was broken: measured on
    # 20260814_223024, 12 of 19 problems had all 8 rollouts on 8 distinct drafts.
    for allocation, name in (
        ("4gpu", "leak-local-stable"),
        ("4gpu", "leak-local"),
        ("4gpu", "leak-local-stable-noeval"),
        ("2gpu", "base"),
    ):
        try:
            config = load(allocation, name)
        except Exception as exc:  # noqa: BLE001
            check(f"{allocation}/{name} loads", False, str(exc)[:160])
            continue
        if not config.teacher_pre.enabled:
            continue
        check(
            f"{allocation}/{name} shares its draft without opting in",
            config.teacher_pre.share_per_group is True
            and config.teacher_pre.visibility == "rollout",
        )

    print("\n[10] the stable/leak-local settings, and the student-unaware pair")
    stable = load("4gpu", "leak-local-stable")
    arm = load("4gpu", "presolve-opd")
    unaware = load("4gpu", "presolve-opd-student-unaware")

    # Inherited from leak-local-stable, not restated in the arm. If any of these
    # drifts, the arm has stopped being a single-factor delta against it.
    for name, cfg in (("presolve-opd", arm), ("student-unaware", unaware)):
        check(
            f"{name} keeps the leak-local reward",
            cfg.reward.turn_local_components == ["leak"],
            str(cfg.reward.turn_local_components),
        )
        check(
            f"{name} keeps the stable settings",
            cfg.format_handling_mode == "terminate"
            and cfg.gconfig.max_new_tokens == 2048
            and cfg.actor.episode_loss_weighting is True,
            f"{cfg.format_handling_mode} {cfg.gconfig.max_new_tokens} "
            f"{cfg.actor.episode_loss_weighting}",
        )
        check(
            f"{name} matches leak-local-stable on all four",
            (
                cfg.reward.turn_local_components,
                cfg.format_handling_mode,
                cfg.gconfig.max_new_tokens,
                cfg.actor.episode_loss_weighting,
            )
            == (
                stable.reward.turn_local_components,
                stable.format_handling_mode,
                stable.gconfig.max_new_tokens,
                stable.actor.episode_loss_weighting,
            ),
        )

    check(
        "turn_local_components requires rebn, which is what runs",
        arm.actor.advantage_estimator == "rebn",
        arm.actor.advantage_estimator,
    )
    check(
        "and is incompatible with reward_norm, which is off",
        arm.actor.reward_norm is None,
        str(arm.actor.reward_norm),
    )
    check("presolve-opd does not tell the teacher", not arm.free_chat.student_has_not_seen_problem)
    check("the unaware arm does", unaware.free_chat.student_has_not_seen_problem)
    check(
        "both still distil the pre-solve",
        arm.opd.context == "presolve" and unaware.opd.context == "presolve",
    )

    def flat(value, prefix=""):
        out = {}
        for key, item in value.items():
            if isinstance(item, dict):
                out.update(flat(item, prefix + key + "."))
            else:
                out[prefix + key] = item
        return out

    def normalize(value, trial):
        return str(value).replace(trial, "<TRIAL>")

    left = flat(asdict(arm))
    right = flat(asdict(unaware))
    differing = sorted(
        key
        for key in left
        if normalize(left[key], arm.trial_name)
        != normalize(right[key], unaware.trial_name)
    )
    check(
        "the pair differs in exactly one key",
        differing == ["free_chat.student_has_not_seen_problem"],
        str(differing),
    )

    print("\n[11] the pre-solve does not run at eval under opd_only")
    on = make_workflow(teacher_pre_visibility="opd_only", opd_context="presolve")
    off = make_workflow()
    on.teacher_progress_judge_enabled = False
    off.teacher_progress_judge_enabled = False

    token = in_context(True)
    try:
        check("opd_only + eval: skipped", on._presolve_unused_at_eval() is True)
        check(
            "visibility=rollout + eval: still runs, so old arms are unchanged",
            off._presolve_unused_at_eval() is False,
        )
        on.teacher_progress_judge_enabled = True
        check(
            "the progress judge reads the draft, so it holds the skip off",
            on._presolve_unused_at_eval() is False,
        )
        on.teacher_progress_judge_enabled = False
    finally:
        _current_context.reset(token)

    token = in_context(False)
    try:
        check("opd_only + train: runs", on._presolve_unused_at_eval() is False)
    finally:
        _current_context.reset(token)

    check(
        "outside a rollout context: runs",
        on._presolve_unused_at_eval() is False,
    )
    # evaluator.teacher_pre_verify is inherited as false from the 0810 base and is
    # now irrelevant either way: the pre-solve does not run at eval at all, so
    # there is nothing left for a verification override to govern. Assert the
    # behaviour rather than the value.
    arm_cfg = load("4gpu", "presolve-opd")
    from_arm = make_workflow(
        teacher_pre_visibility=arm_cfg.teacher_pre.visibility,
        opd_context=arm_cfg.opd.context,
    )
    from_arm.teacher_progress_judge_enabled = arm_cfg.reward.teacher_progress_judge.enabled
    token = in_context(True)
    try:
        check(
            "the arm as configured skips its eval pre-solve",
            from_arm._presolve_unused_at_eval() is True,
        )
    finally:
        _current_context.reset(token)

    print("\n[12] the 2gpu arm changes the allocation and nothing else")
    four = load("4gpu", "presolve-opd")
    two, _ = load_expr_config(
        ["--config", f"{BASE}/2gpu/presolve-opd.yaml"], TutorConfig
    )
    check("2gpu trial name cannot collide with 4gpu", two.trial_name.startswith("0810fc2-"))
    check(
        "one GPU for generation, one for the actor",
        two.cluster.n_gpus_per_node == 2
        and two.rollout.backend == "sglang:d1p1t1"
        and two.actor.backend == "fsdp:d1p1t1",
        f"{two.cluster.n_gpus_per_node} {two.rollout.backend} {two.actor.backend}",
    )
    check("ref follows the actor backend", two.ref.backend == two.actor.backend)
    # No assertion on max_concurrent_rollouts. It is a tunable inside rollout.*,
    # which the allocation-only invariant below already permits to differ, and
    # pinning the number here would fail on the next time it is tuned rather than
    # on anything being wrong.

    def flat2(value, prefix=""):
        out = {}
        for key, item in value.items():
            if isinstance(item, dict):
                out.update(flat2(item, prefix + key + "."))
            else:
                out[prefix + key] = item
        return out

    left = flat2(asdict(four))
    right = flat2(asdict(two))
    differing = sorted(
        key
        for key in left
        if str(left[key]).replace(four.trial_name, "<T>")
        != str(right[key]).replace(two.trial_name, "<T>")
    )
    allocation_only = {
        "trial_name",
        "cluster",
        "rollout",
        "actor",
        "ref",
        "stats_logger",
    }
    smuggled = [key for key in differing if key.split(".")[0] not in allocation_only]
    check(
        "no experiment setting differs between the allocations",
        not smuggled,
        str(smuggled),
    )
    check(
        "the distillation itself is identical",
        (
            two.opd.enabled,
            two.opd.context,
            two.opd.loss_weight,
            two.opd.min_prior_failed_turns,
            two.teacher_pre.visibility,
            two.teacher_pre.on_reject,
            two.teacher_pre.share_per_group,
        )
        == (
            four.opd.enabled,
            four.opd.context,
            four.opd.loss_weight,
            four.opd.min_prior_failed_turns,
            four.teacher_pre.visibility,
            four.teacher_pre.on_reject,
            four.teacher_pre.share_per_group,
        ),
    )
    check(
        "and so are the leak-local and stable settings",
        (
            two.reward.turn_local_components,
            two.format_handling_mode,
            two.gconfig.max_new_tokens,
            two.actor.episode_loss_weighting,
        )
        == (
            four.reward.turn_local_components,
            four.format_handling_mode,
            four.gconfig.max_new_tokens,
            four.actor.episode_loss_weighting,
        ),
    )

    print("\n[13] eval never terminates on format, and the draft has 2048 tokens")
    check(
        "format_terminate defaults to None, so nothing outside 0810 changes",
        TutorEvaluatorConfig().format_terminate is None,
    )

    def eval_modes(cfg):
        train_kwargs = {
            "format_handling_mode": cfg.format_handling_mode,
            "leak_handling_mode": cfg.leak_handling_mode,
        }
        built = _build_eval_workflow_kwargs(train_kwargs, cfg)
        return built["format_handling_mode"], built["leak_handling_mode"]

    for rel in ("presolve-opd", "presolve-opd-student-unaware", "leak-local-stable"):
        cfg = load("4gpu", rel)
        fmt, leak = eval_modes(cfg)
        check(
            f"{rel}: trains on terminate, evaluates on continue",
            cfg.format_handling_mode == "terminate" and fmt == "continue",
            f"train={cfg.format_handling_mode} eval={fmt}",
        )
        check(
            f"{rel}: leaks were already exempt, still are",
            leak == "reward_only",
            leak,
        )

    # An arm that trains on continue is untouched, and the flag -- not the
    # override block -- is what does the work: with None, eval keeps terminate.
    stable = load("4gpu", "leak-local-stable")
    plain = load("4gpu", "leak-local")
    check(
        "an arm already on continue is unchanged",
        eval_modes(plain)[0] == "continue" and plain.format_handling_mode == "continue",
    )
    stable.evaluator.format_terminate = None
    check(
        "with format_terminate=None eval follows training, as before",
        eval_modes(stable)[0] == "terminate",
        eval_modes(stable)[0],
    )
    stable.evaluator.format_terminate = True
    check(
        "with True eval terminates deliberately",
        eval_modes(stable)[0] == "terminate",
    )

    for rel in ("leak-local-stable", "leak-local-student-unaware-stable", "presolve-opd"):
        cfg = load("4gpu", rel)
        check(
            f"{rel} generates 2048 output tokens",
            cfg.gconfig.max_new_tokens == 2048,
            str(cfg.gconfig.max_new_tokens),
        )
    arm = load("4gpu", "presolve-opd")
    budget = (
        arm.teacher_pre.max_tokens
        if arm.teacher_pre.max_tokens > 0
        else arm.gconfig.max_new_tokens
    )
    check(
        "so the pre-solve draft gets 2048 without its own override",
        budget == 2048 and arm.teacher_pre.max_tokens == 0,
        f"budget={budget} teacher_pre.max_tokens={arm.teacher_pre.max_tokens}",
    )
    two, _ = load_expr_config(
        ["--config", f"{BASE}/2gpu/presolve-opd.yaml"], TutorConfig
    )
    check(
        "the 2gpu arm inherits both",
        two.gconfig.max_new_tokens == 2048
        and eval_modes(two)[0] == "continue",
    )

    print("\n[14] a format-terminated episode is trained, not discarded")
    # `inspect` is imported at module scope. Re-importing it here would make it a
    # local for the WHOLE function, so every earlier use in this same function
    # would raise UnboundLocalError.
    from examples.tutor.workflow import TutorAgentWorkflow as _WF

    body = inspect.getsource(inspect.getmodule(_WF))
    check(
        "the exclusion is gone: no bare `return None` on a format termination",
        "format_terminate_scored" not in body,
        "the opt-in flag is still referenced",
    )
    check(
        "and the reasoning is recorded where the branch used to be",
        "A format-terminated episode is TRAINED, not discarded" in body,
    )
    # The flag must not survive anywhere: dead config is worse than none.
    from dataclasses import fields as dc_fields

    check(
        "format_terminate_scored is not a config field any more",
        not any(f.name == "format_terminate_scored" for f in dc_fields(TutorConfig)),
    )
    check(
        "nor a workflow kwarg",
        "format_terminate_scored"
        not in inspect.signature(_WF.__init__).parameters,
    )

    # The arms that train under terminate are the ones this fixes.
    for rel in ("presolve-opd", "leak-local-stable", "leak-local-student-unaware-stable"):
        cfg = load("4gpu", rel)
        check(
            f"{rel} trains under terminate and is now scored",
            cfg.format_handling_mode == "terminate",
            cfg.format_handling_mode,
        )
        check(
            f"{rel} charges enough that terminating is not the cheap exit",
            cfg.reward.format_error_penalty <= -0.35,
            str(cfg.reward.format_error_penalty),
        )
        check(
            f"{rel} still evaluates on the whole dialogue",
            eval_modes(cfg)[0] == "continue",
        )

    fix = load("4gpu", "presolve-opd-fix")
    check(
        "presolve-opd-fix remains the continue control",
        fix.format_handling_mode == "continue",
    )
    check(
        "so presolve-opd vs presolve-opd-fix is now terminate vs continue only",
        load("4gpu", "presolve-opd").format_handling_mode == "terminate"
        and fix.format_handling_mode == "continue",
    )
    check(
        "and the redundant -scored arm is gone",
        not os.path.exists(f"{BASE}/4gpu/presolve-opd-scored.yaml"),
    )

    print("\n[15] the frozen-checkpoint OPD teacher")
    import json as _json

    # TutorOpdConfig is already imported at module scope; re-importing it here
    # would make it a local for the whole of main() and break section 1.
    check(
        "teacher_source defaults to policy, so every existing arm is unchanged",
        TutorOpdConfig().teacher_source == "policy",
    )
    for name in ("presolve-opd", "presolve-opd-fix", "presolve-opd-noeval"):
        check(
            f"{name} still distils from the live policy",
            load("4gpu", name).opd.teacher_source == "policy",
        )

    frozen = load("4gpu", "presolve-opd-frozen")
    check("presolve-opd-frozen loads", True)
    check("it distils from a checkpoint", frozen.opd.teacher_source == "checkpoint")
    check(
        "the ref engine is frozen",
        frozen.ref is not None and frozen.ref.optimizer is None,
    )
    # rl_trainer._validate_cfg refuses any per-engine offload unless the global
    # enable_offload is on, and it raises after the workers have spawned. This is
    # the pairing that crashed the first launch of this arm.
    check(
        "ref.offload and enable_offload agree",
        (not frozen.ref.offload) or frozen.enable_offload,
        f"ref.offload={frozen.ref.offload} enable_offload={frozen.enable_offload}",
    )
    check(
        "kl_ctl stays 0, so ref is not double-claimed",
        frozen.actor.kl_ctl == 0.0,
        str(frozen.actor.kl_ctl),
    )

    adapter = frozen.ref.init_lora_path
    check("the teacher adapter directory exists", os.path.isdir(adapter), adapter)
    meta_path = os.path.join(adapter, "adapter_config.json")
    check("it has an adapter_config.json", os.path.exists(meta_path))
    if os.path.exists(meta_path):
        meta = _json.load(open(meta_path))
        check(
            "rank and alpha match the ref config",
            meta["r"] == frozen.ref.lora_rank
            and meta["lora_alpha"] == frozen.ref.lora_alpha,
            f"{meta['r']}/{meta['lora_alpha']} vs {frozen.ref.lora_rank}/{frozen.ref.lora_alpha}",
        )
        check(
            "target modules match, or _validate_init_lora_config rejects it",
            sorted(meta["target_modules"]) == sorted(frozen.ref.target_modules),
        )
        check(
            "the adapter's base is the model the actor loads",
            meta["base_model_name_or_path"] == frozen.actor.path,
        )
    check(
        "the policy itself starts fresh, not from the teacher",
        not getattr(frozen.actor, "init_lora_path", ""),
        "actor.init_lora_path is set: the policy would start as the teacher",
    )

    # every half-configuration must be rejected rather than silently loading a
    # bare base model as the teacher
    for label, mutate, needle in (
        ("ref.use_lora off", lambda c: setattr(c.ref, "use_lora", False), "ref.use_lora"),
        ("no init_lora_path", lambda c: setattr(c.ref, "init_lora_path", ""), "init_lora_path"),
        ("kl_ctl claiming ref", lambda c: setattr(c.actor, "kl_ctl", 0.1), "kl_ctl"),
        ("no ref block at all", lambda c: setattr(c, "ref", None), "`ref` block"),
    ):
        broken = load("4gpu", "presolve-opd-frozen")
        mutate(broken)
        try:
            broken.__post_init__()
        except ValueError as exc:
            check(f"rejects {label}", needle in str(exc), str(exc)[:120])
        else:
            check(f"rejects {label}", False, "no raise")

    # the trainer must route the OPD pass to the frozen engine, not the actor
    # (`inspect` comes from module scope -- see the note in section [14])
    from areal.trainer import rl_trainer as _rlt

    trainer_src = inspect.getsource(_rlt)
    check(
        "the ref engine is built for a checkpoint teacher too",
        "_opd_frozen_teacher" in trainer_src
        and "config.actor.kl_ctl > 0 or self._opd_frozen_teacher" in trainer_src,
    )
    check(
        "the OPD pass uses that engine instead of the actor",
        "opd_engine = self.ref if self._opd_frozen_teacher else self.actor" in trainer_src
        and "_attach_opd_teacher_logps(opd_engine" in trainer_src,
    )
    check(
        "and the unused ref_logp pass is skipped while kl_ctl is 0",
        "if self.ref is not None and self.config.actor.kl_ctl > 0:" in trainer_src,
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
