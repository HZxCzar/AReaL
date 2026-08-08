#!/usr/bin/env python3
"""Check the three 0808 arms differ in exactly one thing each.

A control arm that quietly differs from the treatment in a second way is worse
than no control, so this asserts the shared settings really are shared and that
the one intended difference is the only one.

Run from the repo root with the venv and .env active.
"""
from __future__ import annotations

import inspect
import sys

import torch  # noqa: F401  (imported so a broken install fails here, not mid-run)

from areal.api.cli_args import load_expr_config
from examples.tutor.configs import TutorConfig

FAILURES: list[str] = []
BASE = "examples/tutor/configs/math/0808"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def load(name: str) -> TutorConfig:
    config, _ = load_expr_config(["--config", f"{BASE}/{name}.yaml"], TutorConfig)
    return config


def main() -> int:
    print("\n[1] all three inherit the 0805/v2 baseline across directories")
    arms = {name: load(name) for name in ("baseline", "prompt", "opd")}
    for name, cfg in arms.items():
        check(f"{name}: inherited n_samples=8", cfg.gconfig.n_samples == 8,
              str(cfg.gconfig.n_samples))
        check(f"{name}: inherited max_turns=10", cfg.max_turns == 10, str(cfg.max_turns))
        check(f"{name}: inherited group_baseline", cfg.actor.group_baseline == "episode")

    print("\n[2] everything that must be shared is shared")
    shared = {
        "gconfig.n_samples": lambda c: c.gconfig.n_samples,
        "max_turns": lambda c: c.max_turns,
        "train_dataset.batch_size": lambda c: c.train_dataset.batch_size,
        "actor.lr": lambda c: c.actor.optimizer.lr,
        "actor.eps_clip": lambda c: c.actor.eps_clip,
        "actor.group_baseline": lambda c: c.actor.group_baseline,
        "actor.use_decoupled_loss": lambda c: c.actor.use_decoupled_loss,
        "actor.behave_imp_weight_cap": lambda c: c.actor.behave_imp_weight_cap,
        "student_generalize.replays": lambda c: c.student_generalize.replays,
        "leak_handling_mode": lambda c: c.leak_handling_mode,
        "reward.leak_penalty": lambda c: c.reward.leak_penalty,
        "teacher_system_prompt": lambda c: c.teacher_system_prompt,
        "evaluator.student_model_names": lambda c: tuple(
            c.evaluator.student_model_names or ()
        ),
    }
    for label, get in shared.items():
        values = {name: get(cfg) for name, cfg in arms.items()}
        check(
            f"identical across arms: {label}",
            len(set(map(repr, values.values()))) == 1,
            repr(values),
        )

    print("\n[3] each arm differs in exactly its one intended way")
    check("baseline: no prompt instruction", arms["baseline"].prompt_instruction.enabled is False)
    check("baseline: no opd", arms["baseline"].opd.enabled is False)
    check("baseline: no guided slots", arms["baseline"].guided_slots.enabled is False)

    check("prompt: instruction on", arms["prompt"].prompt_instruction.enabled is True)
    check("prompt: opd off", arms["prompt"].opd.enabled is False)
    check("prompt: guided slots off", arms["prompt"].guided_slots.enabled is False)

    check("opd: on", arms["opd"].opd.enabled is True)
    check("opd: prompt instruction off", arms["opd"].prompt_instruction.enabled is False)
    check("opd: guided slots off", arms["opd"].guided_slots.enabled is False)

    print("\n[4] the two treatment arms are told the same thing at the same time")
    p, o = arms["prompt"].prompt_instruction, arms["opd"].opd
    check(
        "same instruction text",
        p.resolved_instruction == o.resolved_instruction,
        f"{p.resolved_instruction[:40]!r} vs {o.resolved_instruction[:40]!r}",
    )
    check(
        "same turn gate",
        p.min_prior_failed_turns == o.min_prior_failed_turns,
        f"{p.min_prior_failed_turns} vs {o.min_prior_failed_turns}",
    )
    check(
        "the validated wording, not a paraphrase",
        p.resolved_instruction.startswith("Your previous message did not get through"),
        p.resolved_instruction[:60],
    )

    print("\n[5] the arms cannot be accidentally combined")
    cfg = load("opd")
    cfg.prompt_instruction.enabled = True
    try:
        TutorConfig.__post_init__(cfg)
    except ValueError as exc:
        check("prompt + opd together is refused", True)
        check("the message says why", "attributable" in str(exc), str(exc)[:100])
    else:
        check("prompt + opd together is refused", False, "no error raised")

    print("\n[6] the prompt arm keeps its instruction; the guided one does not")
    from examples.tutor.core.types import TeacherGuidance

    check(
        "prompt guidance is not stripped from the training prompt",
        TeacherGuidance("prompt", "repair", "x").strip_from_training is False,
        "otherwise the arm trains on a prompt it was not generated from",
    )
    check(
        "move guidance is stripped",
        TeacherGuidance("move", "DECOMPOSE", "x").strip_from_training is True,
    )

    from examples.tutor.workflow import TutorAgentWorkflow

    src = inspect.getsource(TutorAgentWorkflow._select_guidance)
    check(
        "the prompt arm is not disabled at eval",
        src.index('kind="prompt"') < src.index("is_eval"),
        "it is a deployment choice being measured, so it must survive evaluation",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all three arms check out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
