"""Keep the ped-arm overrides callable by the AReaL base classes.

examples/pedagogical_rl branched from an older AReaL than feat/diversity. When
the two were merged, PPOActor._compute_advantages had grown a
world_model_rl_reweight_config parameter that the override did not accept, and
training died with a TypeError 70 minutes in -- after the whole step-0
evaluation had already run.

These tests fail at import time instead.
"""

from __future__ import annotations

import inspect

import pytest

from examples.pedagogical_rl.algorithm import (
    PedagogicalFSDPPPOActor,
    PedagogicalPPOActor,
)

OVERRIDES = [
    (PedagogicalPPOActor, "_compute_advantages"),
    (PedagogicalFSDPPPOActor, "ppo_update"),
]


@pytest.mark.parametrize(
    "cls,method", OVERRIDES, ids=lambda v: getattr(v, "__name__", v)
)
def test_override_accepts_everything_the_base_can_be_called_with(cls, method):
    override = getattr(cls, method)
    base_impl = next(
        getattr(base, method)
        for base in cls.__mro__[1:]
        if getattr(base, method, None) is not None
    )

    sub = inspect.signature(override).parameters
    sup = inspect.signature(base_impl).parameters
    if any(p.kind == p.VAR_KEYWORD for p in sub.values()):
        return

    missing = [
        name
        for name, p in sup.items()
        if name not in sub and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    assert not missing, (
        f"{cls.__name__}.{method} cannot accept {missing}, which the base class "
        "signature allows. AReaL grew a parameter this override predates; add "
        "it and reject it explicitly if the baseline does not implement it."
    )


def test_world_model_arguments_are_refused_not_ignored():
    """Silently dropping them would train something other than the baseline."""

    actor = PedagogicalPPOActor.__new__(PedagogicalPPOActor)
    with pytest.raises(ValueError, match="world-model"):
        actor._compute_advantages({}, world_model_rl_reweight_config={"enabled": True})

    engine = PedagogicalFSDPPPOActor.__new__(PedagogicalFSDPPPOActor)
    with pytest.raises(ValueError, match="world-model"):
        engine.ppo_update([], world_model_batch=[{}])
