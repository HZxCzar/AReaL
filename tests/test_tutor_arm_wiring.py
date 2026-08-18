#!/usr/bin/env python3
"""Every current arm loads, and every kwarg train.py builds reaches the workflow.

This is the check that replaces a GPU smoke. The unit tests bind methods to stubs,
so they cannot catch a kwarg that train.py renamed, dropped, or never forwarded --
a failure that otherwise surfaces only after a run has claimed its GPUs, or worse,
does not surface at all because the feature silently stays off.

It descends from tests/test_tutor_guided_opd_wiring.py, which did the same job
against math/0805 and went with the answer-attempt config trees. Two things are
stronger here: the kwarg set is compared against the constructor signature rather
than a hand-listed pair of blocks, and every arm in the tree is loaded rather than
four chosen ones.

Run from the repo root with the venv and .env active.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys

from areal.api.cli_args import load_expr_config

from examples.tutor.configs import TutorConfig
from examples.tutor.workflow import TutorAgentWorkflow

FAILURES: list[str] = []
ARM_DIR = pathlib.Path("examples/tutor/configs/math/0810")


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def _train_kwarg_names() -> set[str]:
    """The keyword names in train.py's `workflow_kwargs = dict(...)` call."""
    tree = ast.parse(pathlib.Path("examples/tutor/train.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "workflow_kwargs" not in targets:
            continue
        if isinstance(node.value, ast.Call):
            return {kw.arg for kw in node.value.keywords if kw.arg}
    raise AssertionError("could not find workflow_kwargs = dict(...) in train.py")


def main() -> int:
    print("\n[1] every kwarg train.py passes is accepted by the constructor")
    passed = _train_kwarg_names()
    params = inspect.signature(TutorAgentWorkflow.__init__).parameters
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    check("train.py builds a non-trivial kwarg set", len(passed) > 40, f"{len(passed)}")
    check(
        "the constructor does not swallow unknown kwargs",
        not accepts_var_kw,
        "a **kwargs would make this whole check vacuous",
    )
    unknown = sorted(passed - set(params))
    check("no kwarg is dropped on the floor", not unknown, f"unknown: {unknown}")

    print("\n[2] the features this branch unified are all forwarded")
    for name in (
        "free_chat",
        "student_models",
        "student_generalize_turn_credit",
        "teacher_pre_share_per_group",
        "teacher_pre_visibility",
        "teacher_history_tags",
        "turn_local_reward_components",
        "opd",
        "cross_eval",
    ):
        check(f"train.py forwards {name}", name in passed)

    print("\n[3] every arm in math/0810 loads")
    arms = sorted(p for p in ARM_DIR.rglob("*.yaml"))
    check("there are arms to load", len(arms) > 10, f"{len(arms)}")
    loaded = {}
    for path in arms:
        try:
            config, _ = load_expr_config(["--config", str(path)], TutorConfig)
            loaded[path] = config
        except Exception as exc:  # noqa: BLE001 - the point is to report, not raise
            check(f"loads: {path.relative_to(ARM_DIR)}", False, f"{type(exc).__name__}: {exc}")
    check(f"all {len(arms)} arms loaded", len(loaded) == len(arms))

    print("\n[4] the couplings the workflow refuses at startup hold in the tree")
    for path, config in loaded.items():
        rel = path.relative_to(ARM_DIR)
        if config.student_generalize.turn_credit:
            check(
                f"{rel}: turn_credit implies a no-teaching baseline",
                config.free_chat.no_teaching_baseline,
                "the workflow raises at construction otherwise",
            )
        if config.opd.enabled and config.opd.context == "presolve":
            check(
                f"{rel}: presolve OPD implies visibility opd_only",
                config.teacher_pre.visibility == "opd_only",
                f"got {config.teacher_pre.visibility!r}",
            )
        if config.free_chat.enabled:
            check(
                f"{rel}: free chat implies max_turn_penalty 0",
                config.reward.max_turn_penalty == 0.0,
                f"got {config.reward.max_turn_penalty}",
            )
        check(
            f"{rel}: teacher history is never stripped",
            config.teacher_history_tags in {"masked", "unmasked"},
            f"got {config.teacher_history_tags!r}",
        )

    print("\n[5] the (behavior, information) axes reach every student entry")
    for path, config in loaded.items():
        rel = path.relative_to(ARM_DIR)
        for student in config.student_models:
            check(
                f"{rel}/{student.name}: mode is text or code",
                str(student.mode) in {"text", "code"},
                f"got {student.mode!r}",
            )
            check(
                f"{rel}/{student.name}: mask carries a mode",
                hasattr(student.mask, "mode"),
                "the information axis is missing from the entry",
            )
    names = [
        (path.relative_to(ARM_DIR), [s.name for s in c.student_models])
        for path, c in loaded.items()
    ]
    check(
        "no arm names the same student twice",
        all(len(n) == len(set(n)) for _rel, n in names),
        str([(str(r), n) for r, n in names if len(n) != len(set(n))]),
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES[:6])}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
