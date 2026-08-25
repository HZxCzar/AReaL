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
ARM_DIR = pathlib.Path("examples/tutor/configs/math")


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
        "student_sampling",
        "student_generalize_turn_credit",
        "teacher_pre_share_per_group",
        "teacher_pre_visibility",
        "teacher_history_tags",
        "teacher_private_visibility",
        "student_type_probe",
        "turn_local_reward_components",
        "opd",
        "cross_eval",
        "personality",
    ):
        check(f"train.py forwards {name}", name in passed)

    print("\n[3] every arm under configs/math loads")
    # alloc.yaml is an OVERLAY, not an arm: it carries only the GPU keys and has no
    # defaults list, so loading it alone is a MissingMandatoryValue by design.
    OVERLAYS = {"alloc.yaml"}
    arms = sorted(p for p in ARM_DIR.rglob("*.yaml") if p.name not in OVERLAYS)
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
        # The personality gate is defined over prose manner, and it withholds the
        # student's engagement rather than changing what the student saw. Pairing it
        # with a code student or a mask would mean something else, so configs.py
        # refuses both -- and this is the tree-wide version of that refusal.
        personalities = sorted(
            {
                student.personality
                for student in config.student_models
                if student.personality and student.personality != "none"
            }
        )
        if personalities:
            offenders = [
                student.name
                for student in config.student_models
                if student.personality
                and student.personality != "none"
                and (
                    str(student.mode) != "text"
                    or str(getattr(student.mask, "mode", "")) != "full"
                )
            ]
            check(
                f"{rel}: personality implies text and an unmasked student",
                not offenders,
                f"offenders: {offenders}",
            )
            check(
                f"{rel}: personality names a prompt file",
                bool(config.personality.prompts_path),
                "the gate has no preference prompt to ask",
            )
            check(
                f"{rel}: personality names a complaint file",
                bool(config.personality.complaints_path),
                "a closed gate has nothing to put in the student's slot",
            )
            # Every category is a learner type from prior work, and the citation has
            # to reach the compiled file or the provenance is lost by the time it is
            # written up.
            from examples.tutor.workflow import load_personality_prompts

            prompts = load_personality_prompts(config.personality.prompts_path)
            missing = [p for p in personalities if p not in prompts]
            check(
                f"{rel}: every personality has a prompt",
                not missing,
                f"missing: {missing}",
            )
            unsourced = [
                name
                for name in personalities
                if name in prompts and not prompts[name].get("source")
            ]
            check(
                f"{rel}: every personality cites its source",
                not unsourced,
                f"unsourced: {unsourced}",
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
    print("[6] an allocation file changes allocation-specific resources only")
    from omegaconf import OmegaConf

    ALLOWED = {
        ("cluster", "n_gpus_per_node"),
        ("rollout", "backend"),
        ("rollout", "max_concurrent_rollouts"),
        ("actor", "backend"),
        ("sglang", "mem_fraction_static"),
        # ref.backend is ${actor.backend} in the base, so it follows the allocation
        # by interpolation rather than by being set a second time.
        ("ref", "backend"),
        ("trial_name",),
        ("debug_trace_dir",),
        ("stats_logger", "wandb", "group"),
        ("stats_logger", "wandb", "name"),
    }

    def flat(config):
        return OmegaConf.to_container(OmegaConf.structured(config), resolve=False)

    def differences(left, right):
        out = set()

        def walk(a, b, path=()):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in set(a) | set(b):
                    walk(a.get(key), b.get(key), path + (key,))
            elif a != b:
                out.add(path)

        walk(left, right)
        return out

    # Every <n>gpu/<arm>.yaml is paired with base/<arm>.yaml -- the arm says what it
    # does, the allocation says how many GPUs it gets, and neither may leak into the
    # other. base/default.yaml is the counterpart of <n>gpu/base.yaml.
    pairs = 0
    for arm_path in sorted(loaded):
        parts = arm_path.parts
        if len(parts) < 3 or not parts[-2].endswith("gpu"):
            continue
        stem = arm_path.stem
        base_name = "default" if stem == "base" else stem
        base_path = arm_path.parent.parent / "base" / f"{base_name}.yaml"
        if base_path not in loaded:
            continue
        pairs += 1
        unexpected = sorted(
            path
            for path in differences(flat(loaded[base_path]), flat(loaded[arm_path]))
            - ALLOWED
            if path and path[-1] != "trial_name"
        )
        rel = arm_path.relative_to(ARM_DIR)
        check(
            f"{rel} differs from its base only in allocation and name",
            not unexpected,
            f"also differs at: {unexpected}",
        )
    check("allocation/base pairs were found at all", pairs >= 6, f"{pairs} pairs")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES[:6])}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
