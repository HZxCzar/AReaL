from __future__ import annotations

from types import SimpleNamespace

import torch

from examples.pedagogical_rl.algorithm import (
    PedagogicalFSDPPPOActor,
    PedagogicalPPOActor,
)

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization


def test_group_advantage_is_broadcast_only_to_teacher_loss_tokens():
    """Each group-normalized scalar advantage covers the whole teacher mask."""

    actor = object.__new__(PedagogicalPPOActor)
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
    actor.reward_norm = Normalization(
        NormConfig(
            mean_level="group",
            std_level="group",
            std_unbiased=True,
            eps=1.0e-4,
            group_size=2,
        )
    )
    actor.config = SimpleNamespace(
        use_decoupled_loss=False,
        recompute_logprob=True,
    )
    data = {
        "rewards": torch.tensor([1.0, 3.0, 2.0, 2.0]),
        "loss_mask": torch.tensor(
            [[0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0]]
        ),
        "attention_mask": torch.ones(4, 5),
        "logprobs": torch.zeros(4, 5),
        "prox_logp": torch.zeros(4, 5),
    }

    result = actor._compute_advantages(data)

    expected = torch.tensor(
        [
            [0.0, -0.7070568, -0.7070568, 0.0, 0.0],
            [0.0, 0.7070568, 0.7070568, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        result["advantages"] * (1 - result["loss_mask"]),
        torch.zeros_like(result["advantages"]),
        rtol=0.0,
        atol=0.0,
    )


def test_mu_two_runs_two_full_updates_and_two_scheduler_steps():
    """μ=2 reuses the same full rollout batch instead of splitting minibatches."""

    calls: list[object] = []

    class _Actor:
        def ppo_update(self, data):
            calls.append(data)

    engine = object.__new__(PedagogicalFSDPPPOActor)
    engine.config = SimpleNamespace(num_iterations=2)
    engine.actor = _Actor()
    engine.lr_scheduler_step = lambda: calls.append("scheduler")
    batch = [{"trajectory": 1}]

    PedagogicalFSDPPPOActor.ppo_update(engine, batch)

    # One scheduler call is internal; AReaL's outer trainer performs the second.
    assert calls == [batch, "scheduler", batch]
