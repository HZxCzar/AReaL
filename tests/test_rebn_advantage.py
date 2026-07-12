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
_compute_rebn_returns = _actor_module._compute_rebn_returns


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
        response, reward=1.0, trajectory_id=123, turn_idx=2
    )

    assert tensor_dict["trajectory_id"].tolist() == [123]
    assert tensor_dict["turn_idx"].tolist() == [2]
    assert "no_eos" not in tensor_dict


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
