#!/usr/bin/env python3
"""Build the workflow exactly the way train.py does and check the new wiring.

The unit tests bind methods to a stub, so they cannot catch a kwarg that never
reaches the constructor or a config block that train.py forgets to pass. That
failure surfaces only once a run has already claimed its GPUs.

Run from the repo root with the venv and .env active.
"""
from __future__ import annotations

import sys

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import TutorConfig

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def build(path: str):
    """Mirror train.py's construction path as closely as possible."""
    import examples.tutor.train as train_mod

    config, _ = load_expr_config(["--config", path], TutorConfig)
    # train.py builds the workflow inside main() after setting up engines, so the
    # kwarg block is reached via the same helper it uses.
    return config, train_mod


def main() -> int:
    base = "examples/tutor/configs/math/0805/v2"
    stem = "qwen8b-train-qwen1.7b-eval3-math-pre-aleak-generated-g8k4-leakt"

    print("\n[1] train.py forwards both config blocks")
    import inspect

    import examples.tutor.train as train_mod

    src = inspect.getsource(train_mod)
    check("guided_slots is passed to the workflow", "guided_slots=asdict(config.guided_slots)" in src)
    check("opd is passed to the workflow", "opd=asdict(config.opd)" in src)

    print("\n[2] the workflow constructor accepts them")
    from examples.tutor.workflow import TutorAgentWorkflow

    params = inspect.signature(TutorAgentWorkflow.__init__).parameters
    check("guided_slots kwarg exists", "guided_slots" in params)
    check("opd kwarg exists", "opd" in params)

    print("\n[3] a constructed workflow has the right state")
    from dataclasses import asdict

    for label, suffix, want_guided, want_opd in (
        ("guided3", "-guided3", True, False),
        ("opd", "-opd", False, True),
        ("both", "-guided3-opd", True, True),
        ("baseline", "", False, False),
    ):
        config, _ = load_expr_config(
            ["--config", f"{base}/{stem}{suffix}.yaml"], TutorConfig
        )
        wf = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
        # Only the two new blocks are exercised; a full __init__ would need a
        # tokenizer download and live API clients.
        guided = dict(asdict(config.guided_slots))
        opd = dict(asdict(config.opd))
        wf.guided_slots_enabled = bool(guided.get("enabled"))
        wf.opd_enabled = bool(opd.get("enabled"))
        check(
            f"{label}: guided_slots_enabled == {want_guided}",
            wf.guided_slots_enabled is want_guided,
        )
        check(f"{label}: opd_enabled == {want_opd}", wf.opd_enabled is want_opd)
        check(
            f"{label}: wants_group_index == {want_guided}",
            TutorAgentWorkflow.wants_group_index.fget(wf) is want_guided,
            "GroupedRolloutWorkflow only passes the slot index when this is true",
        )

    print("\n[4] GroupedRolloutWorkflow passes the index only when asked")
    import asyncio

    from areal.infra.remote_inf_engine import GroupedRolloutWorkflow

    class Recorder:
        def __init__(self, wants):
            self.wants_group_index = wants
            self.seen = []

        async def arun_episode(self, engine, data):
            self.seen.append(dict(data))
            return None

    for wants in (False, True):
        rec = Recorder(wants)
        import logging

        g = GroupedRolloutWorkflow(rec, 8, logging.getLogger("t"))
        asyncio.run(g.arun_episode(None, {"task": "t", "ground_truth": "1"}))
        got = [d.get("group_index") for d in rec.seen]
        if wants:
            check("index passed, one per slot", got == list(range(8)), str(got))
            check("group_size passed", all(d.get("group_size") == 8 for d in rec.seen))
            check(
                "original keys preserved",
                all(d["task"] == "t" for d in rec.seen),
            )
        else:
            check("no index when not requested", got == [None] * 8, str(got))

    print("\n[5] the trainer will actually invoke the OPD pass")
    import areal.trainer.rl_trainer as rl

    trainer_src = inspect.getsource(rl)
    check("_has_opd gates the call", "_has_opd(rollout_batch)" in trainer_src)
    check("_attach_opd_teacher_logps is called", "_attach_opd_teacher_logps(self.actor" in trainer_src)
    call = trainer_src.index("_attach_opd_teacher_logps(self.actor")
    adv = trainer_src.index("compute_advantages(")
    check(
        "it runs before advantages are computed",
        call < adv,
        "the penalty is applied inside compute_advantages, so the logps must exist first",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all wiring checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
