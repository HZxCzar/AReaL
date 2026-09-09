import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from examples.tutor.core.tensors import response_to_tensordict

from areal.api.cli_args import NormConfig, PPOActorConfig, PPOConfig, PPOCriticConfig
from areal.utils.data import KLEstimator, Normalization
from areal.utils.functional import ppo_actor_loss_fn


def _infer_token_denominator(_data, loss_mask):
    return loss_mask.bool()


_ACTOR_PATH = Path(__file__).resolve().parents[1] / "areal/trainer/ppo/actor.py"
_ACTOR_SPEC = importlib.util.spec_from_file_location("_test_ppo_actor", _ACTOR_PATH)
assert _ACTOR_SPEC is not None and _ACTOR_SPEC.loader is not None
_actor_module = importlib.util.module_from_spec(_ACTOR_SPEC)
_saved_modules = {
    name: sys.modules.get(name)
    for name in ("areal.trainer", "areal.trainer.ppo", "areal.trainer.ppo.stats")
}
try:
    _trainer_module = types.ModuleType("areal.trainer")
    _ppo_module = types.ModuleType("areal.trainer.ppo")
    _stats_module = types.ModuleType("areal.trainer.ppo.stats")
    _stats_module.infer_token_denominator = _infer_token_denominator
    sys.modules["areal.trainer"] = _trainer_module
    sys.modules["areal.trainer.ppo"] = _ppo_module
    sys.modules["areal.trainer.ppo.stats"] = _stats_module
    sys.modules[_ACTOR_SPEC.name] = _actor_module
    _ACTOR_SPEC.loader.exec_module(_actor_module)
finally:
    for _name, _module in _saved_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module
PPOActor = _actor_module.PPOActor
_compute_batch_centered_penalties = _actor_module._compute_batch_centered_penalties
_compute_rebn_returns = _actor_module._compute_rebn_returns
_compute_teacher_context_advantages = _actor_module._compute_teacher_context_advantages


def _make_actor(config: PPOActorConfig) -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.config = config
    actor.reward_bias = config.reward_bias
    actor.reward_scaling = config.reward_scaling
    actor.reward_clip = config.reward_clip
    actor.kl_ctl = config.kl_ctl
    actor.kl_estimator = KLEstimator(config.kl_estimator)
    actor.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
    actor.reward_norm = (
        Normalization(config.reward_norm) if config.reward_norm else None
    )
    actor.discount = config.discount
    actor.gae_lambda = config.gae_lambda
    actor.mask_no_eos_with_zero = config.mask_no_eos_with_zero
    return actor


def test_rebn_returns_discount_future_turn_rewards():
    rewards = torch.tensor([0.0, 0.0, 1.0])
    trajectory_ids = torch.tensor([7, 7, 7])
    turn_indices = torch.tensor([1, 2, 3])

    returns = _compute_rebn_returns(
        rewards, trajectory_ids, turn_indices, turn_discount=0.9
    )

    torch.testing.assert_close(
        returns, torch.tensor([0.81, 0.9, 1.0]), rtol=1e-6, atol=1e-6
    )


def _presolve_mixed_batch():
    return {
        "input_ids": torch.zeros((6, 6), dtype=torch.long),
        "attention_mask": torch.ones((6, 6), dtype=torch.bool),
        "loss_mask": torch.tensor(
            [[0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0]] + [[0, 1, 1, 1, 1, 0]] * 4
        ),
        "logprobs": torch.zeros((6, 6)),
        "rewards": torch.tensor([0.8, 0.2, 1.0, 0.0, 0.0, 0.0]),
        "trajectory_id": torch.arange(6),
        "turn_idx": torch.ones(6, dtype=torch.long),
        "group_id": torch.zeros(6, dtype=torch.long),
        "presolve_mask": torch.tensor([False, False, True, True, True, True]),
        "presolve_advantage": torch.tensor([0.0, 0.0, 1.0, -1 / 3, -1 / 3, -1 / 3]),
    }


def test_presolve_does_not_change_teaching_baseline_or_normalization():
    config = PPOActorConfig(
        advantage_estimator="rebn",
        kl_ctl=0.0,
        group_baseline="episode",
        adv_norm=NormConfig(mean_level=None, std_level="batch"),
        loss_weighting="turn",
    )
    actor = _make_actor(config)
    mixed = actor._compute_advantages(_presolve_mixed_batch())
    baseline_data = {
        k: v[:2].clone()
        for k, v in _presolve_mixed_batch().items()
        if not k.startswith("presolve_")
    }
    config.loss_weighting = "token"
    baseline = _make_actor(config)._compute_advantages(baseline_data)
    torch.testing.assert_close(mixed["group_baseline"][:2], baseline["group_baseline"])
    torch.testing.assert_close(mixed["turn_advantage"][:2], baseline["turn_advantage"])
    # Every response's advantages SUM to its original scalar advantage.
    torch.testing.assert_close(
        mixed["advantages"].sum(-1)[2:], torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3])
    )
    torch.testing.assert_close(mixed["policy_sample_weight"].sum(-1), torch.ones(6))


def test_presolve_all_wrong_batch_has_zero_advantage():
    data = {k: v[2:].clone() for k, v in _presolve_mixed_batch().items()}
    data["presolve_advantage"].zero_()
    data["rewards"].zero_()
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            kl_ctl=0.0,
            group_baseline="episode",
            adv_norm=NormConfig(mean_level=None, std_level="batch"),
            loss_weighting="turn",
        )
    )
    out = actor._compute_advantages(data)
    assert torch.isfinite(out["advantages"]).all()
    assert out["advantages"].count_nonzero() == 0


def test_mixed_response_mean_loss_and_gradient_survive_microbatch_split():
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            kl_ctl=0.0,
            group_baseline="episode",
            adv_norm=None,
            loss_weighting="turn",
        )
    )
    data = actor._compute_advantages(_presolve_mixed_batch())

    def evaluate(parts):
        logp = torch.zeros((6, 6), requires_grad=True)
        total = _actor_module._policy_loss_weight(data)
        loss = 0
        for indices in parts:
            mb = {k: v[indices] for k, v in data.items()}
            mb["prox_logp"] = mb["logprobs"].clone()
            value = _actor_module.grpo_loss_fn(
                logp[indices],
                torch.zeros_like(logp[indices]),
                mb,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=3.0,
                behave_imp_weight_cap=5.0,
                behave_imp_weight_mode="disabled",
            )
            loss = loss + value * _actor_module._policy_loss_weight(mb) / total
        loss.backward()
        return loss.detach(), logp.grad

    full, full_grad = evaluate([slice(None)])
    split, split_grad = evaluate([slice(0, 1), slice(1, 3), slice(3, 6)])
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(full_grad, split_grad)
    expected = -data["advantages"].sum(-1) / 6
    torch.testing.assert_close(full_grad.sum(-1), expected)


def test_rebn_returns_do_not_cross_trajectories():
    rewards = torch.tensor([0.0, 1.0, 0.0, 2.0])
    trajectory_ids = torch.tensor([1, 1, 2, 2])
    turn_indices = torch.tensor([1, 2, 1, 2])

    returns = _compute_rebn_returns(
        rewards, trajectory_ids, turn_indices, turn_discount=1.0
    )

    torch.testing.assert_close(
        returns, torch.tensor([1.0, 1.0, 2.0, 2.0]), rtol=1e-6, atol=1e-6
    )


def test_batch_centered_penalty_subtracts_mean_without_std_scaling():
    """Tiny above-mean differences remain tiny instead of becoming z-scores."""
    penalties = _compute_batch_centered_penalties(
        scores=torch.tensor([0.990, 0.991, 0.989, 0.0]),
        weights=torch.ones(4),
        score_valid_mask=torch.tensor([True, True, True, False]),
        turn_valid_mask=torch.ones(4, dtype=torch.bool),
    )

    torch.testing.assert_close(
        penalties,
        torch.tensor([0.0, -0.001, 0.0, 0.0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_batch_centered_penalty_ignores_first_turn_and_failed_embedding():
    """Invalid scores never enter the batch mean or receive an auxiliary signal."""
    penalties = _compute_batch_centered_penalties(
        scores=torch.tensor([float("nan"), 0.9, 0.7, float("nan")]),
        weights=torch.full((4,), 0.2),
        score_valid_mask=torch.tensor([False, True, True, False]),
        turn_valid_mask=torch.ones(4, dtype=torch.bool),
    )

    torch.testing.assert_close(
        penalties,
        torch.tensor([0.0, -0.02, 0.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rebn_batch_centered_penalty_stays_on_later_teacher_turn():
    """The local diversity advantage is not propagated into prior ReBN turns."""
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=NormConfig(mean_level="batch", std_level=None),
        )
    )
    data = {
        "input_ids": torch.zeros((3, 3), dtype=torch.long),
        "attention_mask": torch.ones((3, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 3, dtype=torch.long),
        "logprobs": torch.zeros((3, 3)),
        "rewards": torch.tensor([0.0, 0.0, 1.0]),
        "trajectory_id": torch.tensor([7, 7, 7]),
        "turn_idx": torch.tensor([1, 2, 3]),
        "batch_centered_penalty_score": torch.tensor([float("nan"), 0.9, 0.7]),
        "batch_centered_penalty_weight": torch.full((3,), 0.2),
        "batch_centered_penalty_valid": torch.tensor([False, True, True]),
    }

    result = actor._compute_advantages(data)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.02, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        result["batch_centered_penalty_advantage"],
        torch.tensor([0.0, -0.02, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_teacher_context_advantage_uses_per_token_gap_and_batch_centering():
    """Only relative real-minus-moved likelihood receives a local signal."""
    local_advantage, real_mean, moved_mean, information_gain = (
        _compute_teacher_context_advantages(
            real_logps=torch.tensor(
                [[-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]]
            ),
            real_loss_mask=torch.tensor(
                [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool
            ),
            moved_logps=torch.tensor(
                [[-1.0, -1.0, 0.0], [-2.0, -2.0, 0.0], [-4.0, -4.0, 0.0]]
            ),
            moved_loss_mask=torch.tensor(
                [[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool
            ),
            weights=torch.full((3,), 0.2),
            score_clips=torch.full((3,), 5.0),
            score_valid_mask=torch.tensor([False, True, True]),
            turn_valid_mask=torch.ones(3, dtype=torch.bool),
        )
    )

    torch.testing.assert_close(
        real_mean, torch.tensor([-1.0, -1.0, -1.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        moved_mean, torch.tensor([-1.0, -2.0, -4.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        information_gain, torch.tensor([0.0, 1.0, 3.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        local_advantage, torch.tensor([0.0, -0.2, 0.2]), rtol=1e-6, atol=1e-6
    )


def test_rebn_teacher_context_advantage_stays_on_later_teacher_turn():
    """Context information gain is not propagated backward through ReBN."""
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=NormConfig(mean_level="batch", std_level=None),
            recompute_logprob=True,
            use_decoupled_loss=True,
        )
    )
    data = {
        "input_ids": torch.zeros((3, 3), dtype=torch.long),
        "attention_mask": torch.ones((3, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 3, dtype=torch.long),
        "logprobs": torch.zeros((3, 3)),
        "prox_logp": torch.tensor(
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        "rewards": torch.tensor([0.0, 0.0, 1.0]),
        "trajectory_id": torch.tensor([7, 7, 7]),
        "turn_idx": torch.tensor([1, 2, 3]),
        "teacher_context_input_ids": torch.zeros((3, 3), dtype=torch.long),
        "teacher_context_attention_mask": torch.ones((3, 3), dtype=torch.bool),
        "teacher_context_loss_mask": torch.tensor([[0, 1, 0]] * 3, dtype=torch.long),
        "teacher_context_logp": torch.tensor(
            [[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]
        ),
        "teacher_context_reward_weight": torch.full((3,), 0.2),
        "teacher_context_reward_score_clip": torch.full((3,), 5.0),
        "teacher_context_reward_apply_to_advantage": torch.ones(3, dtype=torch.bool),
        "teacher_context_reward_valid": torch.tensor([False, True, True]),
    }

    result = actor._compute_advantages(data)

    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["teacher_context_advantage"],
        torch.tensor([0.0, -0.2, 0.2]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rebn_teacher_context_log_only_does_not_change_advantage():
    """Log-only context scoring keeps the terminal training advantage unchanged."""
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=NormConfig(mean_level="batch", std_level=None),
            recompute_logprob=True,
            use_decoupled_loss=True,
        )
    )
    data = {
        "input_ids": torch.zeros((3, 3), dtype=torch.long),
        "attention_mask": torch.ones((3, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 3, dtype=torch.long),
        "logprobs": torch.zeros((3, 3)),
        "prox_logp": torch.tensor(
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
        ),
        "rewards": torch.tensor([0.0, 0.0, 1.0]),
        "trajectory_id": torch.tensor([7, 7, 7]),
        "turn_idx": torch.tensor([1, 2, 3]),
        "teacher_context_input_ids": torch.zeros((3, 3), dtype=torch.long),
        "teacher_context_attention_mask": torch.ones((3, 3), dtype=torch.bool),
        "teacher_context_loss_mask": torch.tensor([[0, 1, 0]] * 3, dtype=torch.long),
        "teacher_context_logp": torch.tensor(
            [[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]
        ),
        "teacher_context_reward_weight": torch.full((3,), 0.2),
        "teacher_context_reward_score_clip": torch.full((3,), 5.0),
        "teacher_context_reward_apply_to_advantage": torch.zeros(3, dtype=torch.bool),
        "teacher_context_reward_valid": torch.tensor([False, True, True]),
    }

    result = actor._compute_advantages(data)

    torch.testing.assert_close(
        result["advantages"], torch.zeros((3, 3)), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result["teacher_context_advantage"],
        torch.tensor([0.0, -0.2, 0.2]),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["turn_advantage"], torch.zeros(3), rtol=0.0, atol=0.0
    )


def test_rebn_advantage_normalizes_each_turn_equally_not_each_token():
    config = PPOActorConfig(
        advantage_estimator="rebn",
        turn_discount=1.0,
        kl_ctl=0.0,
        adv_norm=NormConfig(mean_level="batch", std_level=None),
    )
    actor = _make_actor(config)
    data = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.long),
        "logprobs": torch.zeros((2, 4)),
        "rewards": torch.tensor([0.0, 10.0]),
        "trajectory_id": torch.tensor([3, 4]),
        "turn_idx": torch.tensor([1, 1]),
    }

    result = actor._compute_advantages(data)

    expected = torch.tensor([[-5.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 0.0]])
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-6, atol=1e-6)


def test_rebn_group_baseline_stores_turn_advantage_without_auxiliary_rewards():
    """Group-baseline metrics receive the final weighted training advantage."""
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="episode",
            group_baseline_leave1out=True,
            episode_loss_weighting=True,
        )
    )
    data = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.long),
        "logprobs": torch.zeros((2, 4)),
        "rewards": torch.tensor([1.0, -1.0]),
        "trajectory_id": torch.tensor([1, 2]),
        "turn_idx": torch.ones(2, dtype=torch.long),
        "group_id": torch.zeros(2, dtype=torch.long),
    }

    result = actor._compute_advantages(data)

    torch.testing.assert_close(
        result["group_baseline"],
        torch.tensor([-1.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result["episode_loss_weight"],
        torch.tensor([1.5, 0.75]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([3.0, -1.5]),
        rtol=0.0,
        atol=0.0,
    )


def test_rebn_gate_credit_masks_improvement_before_turn_baseline():
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="turn",
            group_baseline_leave1out=True,
            group_baseline_local_reward_mode="exclude",
        )
    )
    data = {
        "input_ids": torch.zeros((6, 3), dtype=torch.long),
        "attention_mask": torch.ones((6, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 6, dtype=torch.long),
        "logprobs": torch.zeros((6, 3)),
        "rewards": torch.tensor([0.0, -1.0, 1.0, 0.0, 0.5, 0.0]),
        "local_rewards": torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        "gate_masked_rewards": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.5, 0.0]),
        "gate_credit_mask": torch.tensor([True, False, True, True, True, True]),
        "trajectory_id": torch.tensor([11, 11, 11, 22, 22, 33]),
        "turn_idx": torch.tensor([1, 2, 3, 1, 2, 1]),
        "group_id": torch.zeros(6, dtype=torch.long),
    }

    result = actor._compute_advantages(data)

    # Turn 2 of trajectory 11 fails the gate and gets zero improvement before
    # its same-turn LOO comparison; its -1 local penalty remains local. Turn 3
    # has no peer and therefore gets zero relative signal.
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([0.75, -1.5, 0.0, 0.0, 0.5, -0.75]),
        rtol=0.0,
        atol=0.0,
    )


def test_rebn_gate_credit_episode_baseline_uses_unmasked_improvement():
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="episode",
            group_baseline_leave1out=True,
            group_baseline_local_reward_mode="exclude",
        )
    )
    data = {
        "input_ids": torch.zeros((5, 3), dtype=torch.long),
        "attention_mask": torch.ones((5, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 5, dtype=torch.long),
        "logprobs": torch.zeros((5, 3)),
        "rewards": torch.tensor([0.0, 1.0, 0.0, 0.5, 0.0]),
        "gate_masked_rewards": torch.tensor([0.0, 1.0, 0.0, 0.5, 0.0]),
        "gate_credit_mask": torch.tensor([False, True, True, False, False]),
        "trajectory_id": torch.tensor([11, 11, 22, 22, 33]),
        "turn_idx": torch.tensor([1, 2, 1, 2, 1]),
        "group_id": torch.zeros(5, dtype=torch.long),
    }

    result = actor._compute_advantages(data)

    # True episode improvements are [1.0, 0.5, 0.0], regardless of which turns
    # passed the gate. Their leave-one-out baselines are [0.25, 0.5, 0.75].
    torch.testing.assert_close(
        result["group_baseline"],
        torch.tensor([0.25, 0.25, 0.5, 0.5, 0.75]),
        rtol=0.0,
        atol=0.0,
    )
    # Gate-failed turns still begin with zero improvement credit and therefore
    # become negative only when the unmasked episode baseline is subtracted.
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([-0.25, 0.75, 0.0, -0.5, -0.75]),
        rtol=0.0,
        atol=0.0,
    )


def test_rebn_gate_fail_penalty_keeps_small_post_normalization_scale():
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=NormConfig(mean_level=None, std_level="batch"),
            group_baseline="turn",
            group_baseline_leave1out=True,
        )
    )
    data = {
        "input_ids": torch.zeros((2, 3), dtype=torch.long),
        "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 2, dtype=torch.long),
        "logprobs": torch.zeros((2, 3)),
        "rewards": torch.zeros(2),
        "personality_gate_fail_penalty": torch.tensor([0.0, -0.05]),
        "trajectory_id": torch.tensor([11, 22]),
        "turn_idx": torch.ones(2, dtype=torch.long),
        "group_id": torch.zeros(2, dtype=torch.long),
    }

    result = actor._compute_advantages(data)

    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([0.0, -0.05]),
        rtol=0.0,
        atol=0.0,
    )


def test_rebn_world_model_gate_scales_normalized_outcome_before_token_broadcast():
    """The four NLL/advantage quadrants change only normalized outcome credit."""

    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
        )
    )
    data = {
        "input_ids": torch.zeros((4, 3), dtype=torch.long),
        "attention_mask": torch.ones((4, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]] * 4, dtype=torch.long),
        "logprobs": torch.zeros((4, 3)),
        "rewards": torch.tensor([1.0, 1.0, -1.0, -1.0]),
        "trajectory_id": torch.tensor([1, 2, 3, 4]),
        "turn_idx": torch.ones(4, dtype=torch.long),
        "world_model_rl_nll": torch.tensor([0.1, 0.9, 0.1, 0.9]),
        "world_model_rl_surprise": torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        "world_model_rl_reweight_valid": torch.ones(4, dtype=torch.bool),
    }

    result = actor._compute_advantages(
        data,
        world_model_rl_reweight_config={
            "enabled": True,
            "positive_strength": 0.5,
            "negative_strength": 0.5,
            "min_weight": 0.1,
            "max_weight": 2.0,
        },
    )

    torch.testing.assert_close(
        result["world_model_rl_weight"],
        torch.tensor([0.5, 1.5, 1.5, 0.5]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["advantages"],
        torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ]
        ),
        rtol=0,
        atol=0,
    )


def test_rebn_rejects_reward_norm():
    with pytest.raises(ValueError, match="reward_norm"):
        PPOActorConfig(
            advantage_estimator="rebn",
            reward_norm=NormConfig(mean_level="batch", std_level=None),
        )


def test_rebn_rejects_critic():
    with pytest.raises(ValueError, match="critic"):
        PPOConfig(
            actor=PPOActorConfig(advantage_estimator="rebn"),
            critic=PPOCriticConfig(),
        )


def test_ppo_update_strips_turn_metadata_before_microbatch_split(monkeypatch):
    class FakeStatsTracker:
        def denominator(self, **_kwargs):
            pass

        def stat(self, **_kwargs):
            pass

        def scalar(self, **_kwargs):
            pass

        def scope(self, _name):
            class Scope:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return Scope()

    class FakeEngine:
        def train(self):
            pass

    def fake_split(data, mb_spec):
        assert "trajectory_id" not in data
        assert "turn_idx" not in data
        assert not any(key.startswith("teacher_context_") for key in data)
        assert not any(key.startswith("world_model_rl_") for key in data)
        raise RuntimeError("split called")

    monkeypatch.setattr(_actor_module, "stats_tracker", FakeStatsTracker())
    monkeypatch.setattr(
        _actor_module, "split_padded_tensor_dict_into_mb_list", fake_split
    )

    actor = _make_actor(PPOActorConfig())
    actor.engine = FakeEngine()
    data = {
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "rewards": torch.tensor([1.0]),
        "advantages": torch.tensor([[0.0, 1.0, 0.0]]),
        "kl_rewards": torch.tensor([[0.0, 0.0, 0.0]]),
        "tot_rewards": torch.tensor([[0.0, 1.0, 0.0]]),
        "trajectory_id": torch.tensor([123]),
        "turn_idx": torch.tensor([2]),
        "teacher_context_input_ids": torch.tensor([[1, 2, 3]]),
        "teacher_context_attention_mask": torch.tensor([[True, True, True]]),
        "teacher_context_loss_mask": torch.tensor([[0, 1, 0]]),
        "teacher_context_logp": torch.tensor([[-2.0, 0.0, 0.0]]),
        "teacher_context_reward_weight": torch.tensor([0.1]),
        "teacher_context_reward_score_clip": torch.tensor([5.0]),
        "teacher_context_reward_apply_to_advantage": torch.tensor([True]),
        "teacher_context_reward_valid": torch.tensor([True]),
        "teacher_context_real_avg_logp": torch.tensor([-1.0]),
        "teacher_context_moved_avg_logp": torch.tensor([-2.0]),
        "teacher_context_information_gain": torch.tensor([1.0]),
        "teacher_context_advantage": torch.tensor([0.1]),
        "world_model_rl_nll": torch.tensor([0.2]),
        "world_model_rl_surprise": torch.tensor([-1.0]),
        "world_model_rl_reweight_valid": torch.tensor([True]),
        "world_model_rl_weight": torch.tensor([1.5]),
        "world_model_rl_advantage_before_reweight": torch.tensor([-1.0]),
        "world_model_rl_advantage_after_reweight": torch.tensor([-1.5]),
    }

    with pytest.raises(RuntimeError, match="split called"):
        actor._ppo_update(data)


def test_ppo_actor_loss_can_disable_clip():
    logprobs = torch.log(torch.tensor([[2.0]]))
    proximal_logprobs = torch.zeros((1, 1))
    old_logprobs = torch.zeros((1, 1))
    advantages = torch.ones((1, 1))
    loss_mask = torch.ones((1, 1), dtype=torch.bool)

    clipped_loss, clipped_stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        use_ppo_clip=True,
    )
    unclipped_loss, unclipped_stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        use_ppo_clip=False,
    )

    torch.testing.assert_close(clipped_loss, torch.tensor(-1.2), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(unclipped_loss, torch.tensor(-2.0), rtol=1e-6, atol=1e-6)
    assert clipped_stat["clip_mask"].item()
    assert not unclipped_stat["clip_mask"].item()


def test_tutor_response_tensordict_includes_trajectory_metadata():
    response = types.SimpleNamespace(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[5, 5],
        input_len=2,
        output_len=2,
    )

    tensor_dict = response_to_tensordict(
        response,
        reward=1.0,
        personality_gate_fail_penalty=-0.05,
        gate_masked_reward=0.75,
        gate_credit_mask=False,
        trajectory_id=123,
        turn_idx=2,
    )

    assert tensor_dict["trajectory_id"].tolist() == [123]
    assert tensor_dict["turn_idx"].tolist() == [2]
    torch.testing.assert_close(
        tensor_dict["personality_gate_fail_penalty"], torch.tensor([-0.05])
    )
    assert tensor_dict["gate_masked_rewards"].tolist() == [0.75]
    assert tensor_dict["gate_credit_mask"].tolist() == [False]
    assert "no_eos" not in tensor_dict


def test_tutor_response_tensordict_includes_batch_centered_penalty_metadata():
    """Similarity metadata remains raw until the complete training batch exists."""
    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[5],
    )

    valid = response_to_tensordict(
        response,
        reward=0.0,
        batch_centered_penalty_score=0.9,
        batch_centered_penalty_weight=0.2,
    )
    missing = response_to_tensordict(
        response,
        reward=0.0,
        batch_centered_penalty_score=None,
        batch_centered_penalty_weight=0.2,
    )

    assert valid["batch_centered_penalty_score"].tolist() == pytest.approx([0.9])
    assert valid["batch_centered_penalty_weight"].tolist() == pytest.approx([0.2])
    assert valid["batch_centered_penalty_valid"].tolist() == [True]
    assert torch.isnan(missing["batch_centered_penalty_score"]).all()
    assert missing["batch_centered_penalty_valid"].tolist() == [False]


def test_tutor_response_tensordict_moves_later_output_to_preceding_prompt():
    """The counterfactual keeps T2 tokens but replaces its prompt with T1's."""
    response = types.SimpleNamespace(
        input_tokens=[10, 11, 12],
        output_tokens=[20, 21],
        output_logprobs=[-0.1, -0.2],
        output_versions=[5, 5],
    )

    moved = response_to_tensordict(
        response,
        reward=0.0,
        teacher_context_input_tokens=[1, 2],
        teacher_context_reward_weight=0.3,
        teacher_context_reward_score_clip=5.0,
    )
    first_turn = response_to_tensordict(
        response,
        reward=0.0,
        teacher_context_input_tokens=None,
        teacher_context_reward_weight=0.3,
        teacher_context_reward_score_clip=5.0,
    )

    assert moved["teacher_context_input_ids"].tolist() == [[1, 2, 20, 21]]
    assert moved["teacher_context_loss_mask"].tolist() == [[0, 0, 1, 1]]
    assert moved["teacher_context_reward_valid"].tolist() == [True]
    assert moved["teacher_context_reward_apply_to_advantage"].tolist() == [True]
    assert first_turn["teacher_context_input_ids"].tolist() == [[10, 11, 12, 20, 21]]
    assert first_turn["teacher_context_reward_valid"].tolist() == [False]


@pytest.mark.parametrize(
    ("stop_reason", "enabled", "expected_reward"),
    [
        ("length", False, 1.0),
        ("length", True, 0.0),
        ("stop", True, 1.0),
    ],
)
def test_tutor_response_tensordict_optionally_zeroes_length_stop_reward(
    stop_reason, enabled, expected_reward
):
    response = types.SimpleNamespace(
        input_tokens=[1],
        output_tokens=[2],
        output_logprobs=[-0.1],
        output_versions=[5],
        stop_reason=stop_reason,
    )

    tensor_dict = response_to_tensordict(
        response,
        reward=1.0,
        zero_reward_on_length_stop=enabled,
    )

    assert tensor_dict["rewards"].tolist() == pytest.approx([expected_reward])


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("output_logprobs", [-0.1]),
        ("output_logprobs", [-0.1, -0.2, -0.3]),
        ("output_versions", [5]),
        ("output_versions", [5, 5, 5]),
    ],
)
def test_tutor_response_tensordict_rejects_output_metadata_length_mismatch(
    field_name, field_value
):
    """Every generated token must have one behavior logprob and version."""
    response = types.SimpleNamespace(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        output_logprobs=[-0.1, -0.2],
        output_versions=[5, 5],
        stop_reason="length",
    )
    setattr(response, field_name, field_value)

    with pytest.raises(
        ValueError,
        match=rf"{field_name} length mismatch.*output_tokens=2.*{field_name}=",
    ):
        response_to_tensordict(response, reward=1.0)
