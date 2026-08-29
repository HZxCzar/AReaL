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
from tests.test_rebn_advantage import _make_actor

from areal.api.cli_args import NormConfig, PPOActorConfig


def _leak(leaked: bool) -> LeakCheckResult:
    return LeakCheckResult(
        raw_output="", leaked=leaked, feedback="", parse_error=None, raw_result={}
    )


def _turn(
    turn_idx: int,
    *,
    leaked: bool,
    format_error: bool = False,
    gate_failed: bool = False,
    gate_terminated: bool = False,
) -> TurnArtifact:
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
        tutor_format_error=format_error,
        personality_gated=gate_failed,
        personality_gate_terminated=gate_terminated,
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


def test_format_penalty_is_reported_as_turn_local_when_configured():
    computer = EpisodeRewardComputer(
        success_reward=0.0,
        leak_penalty=-1.0,
        leak_penalty_mode="rawbase",
        format_error_penalty=-0.5,
        turn_local_components=("format_error",),
    )
    episode = _episode([_turn(1, leaked=False, format_error=True)])
    assignments = asyncio.run(computer.compute(episode))

    assert [a.local_reward for a in assignments] == [-0.5]
    assert [a.reward for a in assignments] == [-0.5]


def test_turn_local_components_can_choose_different_placements():
    computer = EpisodeRewardComputer(
        success_reward=0.0,
        leak_penalty=-1.0,
        leak_penalty_mode="rawbase",
        format_error_penalty=-0.5,
        personality_gate_fail_penalty=-0.25,
        turn_local_components=("leak", "format_error", "personality_gate_fail"),
        turn_local_component_placements={
            "leak": "post_std",
            "format_error": "pre_std",
            "personality_gate_fail": "group_norm",
        },
    )
    assignments = asyncio.run(
        computer.compute(
            _episode([_turn(1, leaked=True, format_error=True, gate_failed=True)])
        )
    )

    assert assignments[0].local_reward_by_placement == {
        "group_norm": -0.25,
        "pre_std": -0.5,
        "post_std": -1.0,
    }


def test_personality_gate_terminate_penalty_is_local_and_only_on_terminal_turn():
    computer = EpisodeRewardComputer(
        success_reward=0.0,
        leak_penalty=-1.0,
        leak_penalty_mode="rawbase",
        personality_gate_terminate_penalty=-0.5,
        turn_local_components=("personality_gate_terminate",),
    )
    episode = _episode(
        [
            _turn(1, leaked=False),
            _turn(2, leaked=False, gate_terminated=True),
        ]
    )
    assignments = asyncio.run(computer.compute(episode))

    assert [a.reward for a in assignments] == [0.0, -0.5]
    assert [a.local_reward for a in assignments] == [0.0, -0.5]
    assert assignments[1].reward_components == {"personality_gate_terminate": -0.5}


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


def test_post_std_local_penalty_keeps_its_configured_advantage_scale():
    actor = _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=NormConfig(mean_level=None, std_level="batch"),
        )
    )
    result = actor._compute_advantages(
        {
            "input_ids": torch.zeros((4, 4), dtype=torch.long),
            "attention_mask": torch.ones((4, 4), dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 0, 0]] * 4, dtype=torch.long),
            "logprobs": torch.zeros((4, 4)),
            "rewards": torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "local_rewards": torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "local_rewards_group_norm": torch.zeros(4),
            "local_rewards_pre_std": torch.zeros(4),
            "local_rewards_post_std": torch.tensor([-1.0, 0.0, 0.0, 0.0]),
            "trajectory_id": torch.arange(4),
            "turn_idx": torch.ones(4, dtype=torch.long),
        }
    )

    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([-1.0, 0.0, 0.0, 0.0]),
        rtol=0.0,
        atol=0.0,
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


def _group_actor(**config_overrides):
    return _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="episode",
            group_baseline_leave1out=True,
            **config_overrides,
        )
    )


def _turn_group_data(**extra):
    """Three equal teaching episodes; episode 3 leaks only at turn 2."""
    base = {
        "input_ids": torch.zeros((6, 4), dtype=torch.long),
        "attention_mask": torch.ones((6, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 0, 0]] * 6, dtype=torch.long),
        "logprobs": torch.zeros((6, 4)),
        "rewards": torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, -0.5]),
        "local_rewards": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        "trajectory_id": torch.tensor([1, 1, 2, 2, 3, 3]),
        "turn_idx": torch.tensor([1, 2, 1, 2, 1, 2]),
        "group_id": torch.zeros(6, dtype=torch.long),
    }
    base.update(extra)
    return base


def _turn_group_actor(**config_overrides):
    return _make_actor(
        PPOActorConfig(
            advantage_estimator="rebn",
            turn_discount=1.0,
            kl_ctl=0.0,
            adv_norm=None,
            group_baseline="turn",
            group_baseline_leave1out=True,
            **config_overrides,
        )
    )


def test_turn_group_baseline_keeps_local_penalty_at_its_depth():
    result = _turn_group_actor()._compute_advantages(_turn_group_data())
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_turn_group_baseline_can_exclude_all_local_rewards():
    actor = _turn_group_actor(group_baseline_local_reward_mode="exclude")
    result = actor._compute_advantages(_turn_group_data())

    torch.testing.assert_close(
        result["group_baseline"], torch.full((6,), 0.5), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_group_baseline_is_unchanged_by_the_split():
    """The episode total is the same either way, so the yardstick must be too.

    This is the regression test for the bug that drove entropy up: the scalar the
    baseline reads was taken from the first turn's return, so a penalty raised on
    the last turn was missing from it. That shortfall sat on every turn as an
    uncancelled negative advantage.
    """
    plain = _group_actor()._compute_advantages(_group_data())
    split = _group_actor()._compute_advantages(
        _group_data(local_rewards=torch.tensor([0.0, 0.0, 0.0, -1.0]))
    )
    # Episode 1 is scored against episode 2's total of 0.5 teaching - 1.0 leak.
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
    # Row 2 is episode 2's clean first turn; row 3 is the turn that leaked.
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
    # The leaker is charged exactly as much as before; only the innocent turn in
    # front of it is relieved.
    assert split["turn_advantage"][3] == plain["turn_advantage"][3]


def test_episode_group_baseline_can_exclude_all_local_rewards():
    actor = _group_actor(group_baseline_local_reward_mode="exclude")
    result = actor._compute_advantages(
        _group_data(local_rewards=torch.tensor([0.0, 0.0, 0.0, -1.0]))
    )

    torch.testing.assert_close(
        result["group_baseline"], torch.full((4,), 0.5), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        result["turn_advantage"],
        torch.tensor([0.0, 0.0, 0.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_group_baseline_local_reward_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="must be 'include', or 'exclude'"):
        PPOActorConfig(
            advantage_estimator="rebn",
            group_baseline_local_reward_mode="unknown",
        )


def test_local_rewards_require_rebn():
    actor = _make_actor(PPOActorConfig(advantage_estimator="gae", kl_ctl=0.0))
    with pytest.raises(ValueError, match="rebn"):
        actor._compute_advantages(_data(local_rewards=torch.tensor([0.0, -1.0])))
