#!/usr/bin/env python3
"""Load the three new configs and check the values that reach the workflow.

A config typo does not fail until a run starts and burns a node allocation, so
this resolves each one through the same loader train.py uses and asserts on the
resulting TutorConfig.
"""
from __future__ import annotations

import sys
from dataclasses import asdict

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import TutorConfig

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def load(path: str) -> TutorConfig:
    config, _ = load_expr_config(["--config", path], TutorConfig)
    return config


def main() -> int:
    base = "examples/tutor/configs/math/0805/v2"
    stem = "qwen8b-train-qwen1.7b-eval3-math-pre-aleak-generated-g8k4-leakt"

    print("\n[guided3] across-task: reserved move slots")
    cfg = load(f"{base}/{stem}-guided3.yaml")
    g = cfg.guided_slots
    check("enabled", g.enabled is True)
    check("3 of 8 slots guided", (g.slots, cfg.gconfig.n_samples) == (3, 8),
          f"got slots={g.slots}, n_samples={cfg.gconfig.n_samples}")
    check("free rollouts remain the majority", g.slots < cfg.gconfig.n_samples)
    check("moves as configured", g.moves == ("DECOMPOSE", "REFRAME", "PROBE"), str(g.moves))
    check("WORKED excluded (leaks 58% when forced)", "WORKED" not in g.moves)
    check("first turn only", g.turns == (1,), str(g.turns))
    check("decoupled loss on (required for the rewritten prompt)",
          cfg.actor.use_decoupled_loss is True)
    check("importance weight capped", cfg.actor.behave_imp_weight_cap == 5.0,
          str(cfg.actor.behave_imp_weight_cap))
    check("group baseline still episode-level", cfg.actor.group_baseline == "episode")
    check("opd off in this arm", cfg.opd.enabled is False)
    check("embedding diversity reward stays off",
          cfg.reward.teacher_diversity.enabled is False)
    check("position-swap context reward stays off",
          cfg.reward.teacher_context.enabled is False)

    print("\n[opd] in-task: distillation from the instructed teacher")
    cfg = load(f"{base}/{stem}-opd.yaml")
    o = cfg.opd
    check("enabled", o.enabled is True)
    check("coef matches the reference default", o.loss_weight == 1.0, str(o.loss_weight))
    check("clip disabled, matching the reference", o.reward_clip == 0.0, str(o.reward_clip))
    check("supervises turn 3 onward", o.min_prior_failed_turns == 2,
          str(o.min_prior_failed_turns))
    check("uses the validated repair wording",
          o.resolved_instruction.startswith("Your previous message did not get through"),
          o.resolved_instruction[:60])
    check("guided slots off in this arm", cfg.guided_slots.enabled is False)

    print("\n[guided3-opd] both")
    cfg = load(f"{base}/{stem}-guided3-opd.yaml")
    check("guided slots on", cfg.guided_slots.enabled is True)
    check("opd on", cfg.opd.enabled is True)
    check("guided rows excluded from OPD", cfg.opd.skip_guided_rows is True,
          "otherwise two prompt perturbations stack on one row")
    check("moves inherited", cfg.guided_slots.moves == ("DECOMPOSE", "REFRAME", "PROBE"),
          str(cfg.guided_slots.moves))

    print("\n[kwargs] the dicts train.py hands the workflow")
    for name, block in (("guided_slots", cfg.guided_slots), ("opd", cfg.opd)):
        d = asdict(block)
        check(f"{name} is asdict-able", isinstance(d, dict) and bool(d))
    check("guided_slots kwarg carries moves", "moves" in asdict(cfg.guided_slots))
    check("opd kwarg carries loss_weight", "loss_weight" in asdict(cfg.opd))

    print("\n[validation] bad configs must be rejected")
    for label, mutate in (
        ("slots >= n_samples", lambda c: setattr(c.guided_slots, "slots", 8)),
        ("unknown move", lambda c: setattr(c.guided_slots, "moves", ("NOPE",))),
        ("negative opd weight", lambda c: setattr(c.opd, "loss_weight", -1.0)),
    ):
        c = load(f"{base}/{stem}-guided3-opd.yaml")
        mutate(c)
        try:
            if label == "unknown move":
                type(c.guided_slots).__post_init__(c.guided_slots)
            elif label == "negative opd weight":
                type(c.opd).__post_init__(c.opd)
            else:
                TutorConfig.__post_init__(c)
        except ValueError as exc:
            check(f"rejected: {label}", True)
            del exc
        else:
            check(f"rejected: {label}", False, "no ValueError raised")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all config checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
