"""Turn-local reward components must not be accumulated backward by ReBN.

ReBN sums each turn's reward into every earlier turn's return. A penalty raised
on the last turn -- a leak, under leak_handling_mode='terminate' -- therefore
landed undiscounted on the returns of the good turns that preceded it. These
tests pin the fix: components named in reward.turn_local_components stay on the
turn that produced them, and the totals reported everywhere else are unchanged.
"""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from examples.tutor.core.rewards import EpisodeRewardComputer
from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    TurnArtifact,
)

from areal.api.cli_args import PPOActorConfig
from tests.test_rebn_advantage import _make_actor


def _leak(leaked: bool) -> LeakCheckResult:
    return LeakCheckResult(
        raw_output="", leaked=leaked, feedback="", parse_error=None, raw_result={}
    )


def _turn(turn_idx: int, *, leaked: bool) -> TurnArtifact:
    return TurnArtifact(
        turn_idx=turn_idx,
        tutor_state=SimpleNamespace(max_turns=5),
        tutor_messages=[],
        tutor_response=None,
        tutor_raw_output="",
        tutor_visible_output="",
        leak_result=_leak(leaked),
        public_history_before=[],
        public_history_after=[],
    )


def _episode(turns) -> EpisodeArtifact:
    return EpisodeArtifact(
        task="",
        ground_truth="",
        initial_student_answer="",
        initial_student_error=None,
        initial_judge_result=JudgeResult(
            raw_output="", correct=False, feedback="", parse_error=None, raw_result={}
        ),
        turns=list(turns),
        termination_reason="leak",
        pre_success=False,
        leak_count=1,
        latest_student_answer="",
    )


def _assignments(turn_local):
    computer = EpisodeRewardComputer(
        success_reward=0.0,
        leak_penalty=-1.0,
        leak_penalty_mode="rawbase",
        turn_local_components=turn_local,
    )
    episode = _episode([_turn(1, leaked=False), _turn(2, leaked=True)])
    return asyncio.run(computer.compute(episode))


def test_leak_penalty_is_reported_as_turn_local_when_configured():
    assignments = _assignments(("leak",))
    assert [a.local_reward for a in assignments] == [0.0, -1.0]
    # The scalar total is untouched, so total_reward and the
    # reward_component/* metrics keep their previous meaning.
    assert [a.reward for a in assignments] == [0.0, -1.0]


def test_local_reward_defaults_to_zero_so_behaviour_is_unchanged():
    assignments = _assignments(())
    assert [a.local_reward for a in assignments] == [0.0, 0.0]
    assert [a.reward for a in assignments] == [0.0, -1.0]


def _response(n_out: int = 1):
    return SimpleNamespace(
        input_tokens=[1],
        output_tokens=list(range(2, 2 + n_out)),
        output_logprobs=[0.0] * n_out,
        output_versions=[0] * n_out,
        stop_reason="stop",
    )


def test_tensordict_only_carries_the_column_when_asked():
    assert "local_rewards" not in response_to_tensordict(_response(), reward=-1.0)
    row = response_to_tensordict(_response(), reward=-1.0, local_reward=-1.0)
    torch.testing.assert_close(row["local_rewards"], torch.tensor([-1.0]))
    torch.testing.assert_close(row["rewards"], torch.tensor([-1.0]))


def _data(**extra):
    base = {
        "input_ids": torch.zeros((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.long),
        "logprobs": torch.zeros((2, 4)),
        # Turn 2 earned +0.5 of teaching and lost 1.0 to a leak.
        "rewards": torch.tensor([0.0, -0.5]),
        "trajectory_id": torch.tensor([7, 7]),
        "turn_idx": torch.tensor([1, 2]),
    }
    base.update(extra)
    return base


def _actor():
    return _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
        )
    )


def test_without_the_column_the_leak_still_reaches_the_earlier_turn():
    result = _actor()._compute_advantages(_data())
    # Turn 1 taught cleanly and is charged -0.5 anyway.
    torch.testing.assert_close(
        result["turn_advantage"], torch.tensor([-0.5, -0.5]), rtol=0.0, atol=0.0
    )


def test_turn_local_penalty_stays_on_its_own_turn():
    result = _actor()._compute_advantages(
        _data(local_rewards=torch.tensor([0.0, -1.0]))
    )
    # Turn 1 keeps the +0.5 outcome; turn 2 still eats the whole -1.0.
    torch.testing.assert_close(
        result["turn_advantage"], torch.tensor([0.5, -0.5]), rtol=0.0, atol=0.0
    )


def _group_data(**extra):
    """Two two-turn episodes in one rollout group.

    Episode 1 (rows 0-1) teaches cleanly for +0.5. Episode 2 (rows 2-3) teaches
    the same +0.5 but leaks on its last turn for -1.0.
    """
    base = {
        "input_ids": torch.zeros((4, 4), dtype=torch.long),
        "attention_mask": torch.ones((4, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0, 0]] * 4, dtype=torch.long),
        "logprobs": torch.zeros((4, 4)),
        "rewards": torch.tensor([0.0, 0.5, 0.0, -0.5]),
        "trajectory_id": torch.tensor([1, 1, 2, 2]),
        "turn_idx": torch.tensor([1, 2, 1, 2]),
        "group_id": torch.zeros(4, dtype=torch.long),
    }
    base.update(extra)
    return base


def _group_actor():
    return _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="episode",
            group_baseline_leave1out=True,
        )
    )


def test_group_baseline_is_unchanged_by_the_split():
    plain = _group_actor()._compute_advantages(_group_data())
    split = _group_actor()._compute_advantages(
        _group_data(local_rewards=torch.tensor([0.0, 0.0, 0.0, -1.0]))
    )
    torch.testing.assert_close(
        plain["group_baseline"],
        torch.tensor([-0.5, -0.5, 0.5, 0.5]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        split["group_baseline"], plain["group_baseline"], rtol=0.0, atol=0.0
    )


def test_only_the_leaking_turn_keeps_the_penalty():
    plain = _group_actor()._compute_advantages(_group_data())
    split = _group_actor()._compute_advantages(
        _group_data(local_rewards=torch.tensor([0.0, 0.0, 0.0, -1.0]))
    )
    torch.testing.assert_close(
        plain["turn_advantage"],
        torch.tensor([1.0, 1.0, -1.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        split["turn_advantage"],
        torch.tensor([1.0, 1.0, 0.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert split["turn_advantage"][3] == plain["turn_advantage"][3]


def test_local_rewards_require_rebn():
    actor = _make_actor(PPOActorConfig(advantage_estimator="gae", kl_ctl=0.0))
    with pytest.raises(ValueError, match="rebn"):
        actor._compute_advantages(_data(local_rewards=torch.tensor([0.0, -1.0])))
