"""Tests for the episode-weighted group baseline and per-episode loss weighting.

The point of these two helpers is that rollout groups are *ragged*: one group holds
G episodes and each episode contributes a different number of turn rows, so nothing
may depend on rows being contiguous or on groups having uniform size.
"""

import torch

from areal.trainer.ppo.actor import (
    _compute_episode_group_baseline,
    _compute_episode_loss_weights,
    _compute_turn_group_baseline,
    _episode_scalars,
)

# Two ragged groups, deliberately interleaved so nothing can rely on contiguity.
#
#   group 0: ep A (2 turns, R=+1.0), ep B (1 turn, R=-1.0), ep C (4 turns, R=-1.0)
#   group 1: ep D (1 turn, R=+1.0), ep E (3 turns, mid-turn reward, R=+1.3)
#
# ep E carries rewards [0, 0.3, 1.0] so its ReBN returns are [1.3, 1.3, 1.0] --
# the case where a flat episode-level broadcast would destroy process detail.
TRAJ = [
    # (trajectory_id, turn_idx, group_id, turn_return, tokens)
    (1001, 1, 0, 1.0, 10),
    (3003, 1, 0, -1.0, 10),
    (5005, 2, 1, 1.3, 10),
    (1001, 2, 0, 1.0, 10),
    (3003, 2, 0, -1.0, 10),
    (4004, 1, 1, 1.0, 10),
    (3003, 3, 0, -1.0, 10),
    (5005, 1, 1, 1.3, 10),
    (2002, 1, 0, -1.0, 10),
    (3003, 4, 0, -1.0, 10),
    (5005, 3, 1, 1.0, 10),
]


def _tensors(rows=TRAJ, valid=None):
    traj = torch.tensor([r[0] for r in rows], dtype=torch.long)
    turn = torch.tensor([r[1] for r in rows], dtype=torch.long)
    group = torch.tensor([r[2] for r in rows], dtype=torch.long)
    ret = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    tok = torch.tensor([r[4] for r in rows], dtype=torch.long)
    mask = torch.ones(len(rows), dtype=torch.bool) if valid is None else valid
    return traj, turn, group, ret, tok, mask


def test_episode_scalar_is_return_at_first_turn():
    traj, turn, _group, ret, _tok, mask = _tensors()
    ep_index, ep_return, n_ep = _episode_scalars(ret, traj, turn, mask)

    assert n_ep == 5
    # Episode E's scalar must be the turn-1 return (1.3), not the last turn's 1.0.
    by_traj = {}
    for row, tid in enumerate(traj.tolist()):
        by_traj[tid] = ep_return[ep_index[row]].item()
    assert by_traj[1001] == 1.0
    assert by_traj[2002] == -1.0
    assert by_traj[3003] == -1.0
    assert by_traj[4004] == 1.0
    assert abs(by_traj[5005] - 1.3) < 1e-6


def test_baseline_is_episode_weighted_not_turn_weighted():
    traj, turn, group, ret, _tok, mask = _tensors()
    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=False
    )

    # group 0 episode returns = [+1, -1, -1] -> episode mean -1/3.
    # The turn-weighted mean would be (1+1-1-1-1-1)/7 = -2/7, which is what we
    # are specifically avoiding: ep C's four rows must not outvote ep A and ep B.
    g0 = baseline[group == 0]
    assert torch.allclose(g0, torch.full_like(g0, -1.0 / 3.0), atol=1e-6)
    assert not torch.allclose(g0, torch.full_like(g0, -2.0 / 7.0), atol=1e-3)

    # group 1 episode returns = [+1.0, +1.3] -> mean 1.15
    g1 = baseline[group == 1]
    assert torch.allclose(g1, torch.full_like(g1, 1.15), atol=1e-6)


def test_leave_one_out_excludes_own_episode():
    traj, turn, group, ret, _tok, mask = _tensors()
    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    got = {tid: baseline[traj == tid][0].item() for tid in {r[0] for r in TRAJ}}

    assert abs(got[1001] - (-1.0)) < 1e-6  # (-1 + -1) / 2
    assert abs(got[2002] - 0.0) < 1e-6  # (+1 + -1) / 2
    assert abs(got[3003] - 0.0) < 1e-6  # (+1 + -1) / 2
    assert abs(got[4004] - 1.3) < 1e-6  # only ep E remains
    assert abs(got[5005] - 1.0) < 1e-6  # only ep D remains


def test_shift_preserves_within_episode_structure():
    """A_t - A_s == G_t - G_s, so mid-turn reward detail survives untouched."""
    traj, turn, group, ret, _tok, mask = _tensors()
    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    advantages = ret - baseline

    ep_e = traj == 5005
    order = torch.argsort(turn[ep_e])
    before = ret[ep_e][order]
    after = advantages[ep_e][order]
    assert torch.allclose(before.diff(), after.diff(), atol=1e-6)
    # and the detail is actually non-trivial
    assert not torch.allclose(before.diff(), torch.zeros_like(before.diff()))


def test_degenerate_group_yields_zero_advantage():
    """All episodes identical -> baseline equals the value -> no gradient, no blowup."""
    rows = [(7001, 1, 0, -1.0, 10), (7002, 1, 0, -1.0, 10), (7003, 1, 0, -1.0, 10)]
    traj, turn, group, ret, _tok, mask = _tensors(rows)
    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    assert torch.allclose(ret - baseline, torch.zeros_like(ret), atol=1e-6)


def test_singleton_group_yields_zero_advantage():
    """One surviving episode has no baseline; leave-one-out must not divide by zero."""
    rows = [(8001, 1, 0, 1.0, 10), (8001, 2, 0, 1.0, 10)]
    traj, turn, group, ret, _tok, mask = _tensors(rows)
    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    adv = ret - baseline
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_invalid_rows_get_zero_baseline_and_are_excluded():
    traj, turn, group, ret, _tok, _mask = _tensors()
    valid = torch.ones(len(TRAJ), dtype=torch.bool)
    valid[traj == 2002] = False  # drop ep B entirely

    baseline = _compute_episode_group_baseline(
        ret, traj, turn, group, valid, leave_one_out=False
    )
    assert baseline[traj == 2002].abs().max().item() == 0.0
    # group 0 now averages only ep A and ep C -> (1 - 1) / 2 = 0
    g0_valid = baseline[(group == 0) & valid]
    assert torch.allclose(g0_valid, torch.zeros_like(g0_valid), atol=1e-6)


def test_loss_weights_equalize_episode_gradient_mass():
    traj, _turn, _group, _ret, tok, mask = _tensors()
    weights = _compute_episode_loss_weights(traj, mask, tok)

    # episode token totals: A=20, B=10, C=40, D=10, E=30 -> mean 22
    expected = {
        1001: 22 / 20,
        2002: 22 / 10,
        3003: 22 / 40,
        4004: 22 / 10,
        5005: 22 / 30,
    }
    for tid, want in expected.items():
        got = weights[traj == tid]
        assert torch.allclose(got, torch.full_like(got, want), atol=1e-6), tid

    # Total gradient mass is unchanged: sum(weight * tokens) == sum(tokens).
    assert abs((weights * tok.float()).sum().item() - tok.sum().item()) < 1e-4

    # Every episode now carries the same mass.
    masses = [
        (weights[traj == tid] * tok[traj == tid].float()).sum().item()
        for tid in expected
    ]
    assert max(masses) - min(masses) < 1e-4


def test_row_order_does_not_matter():
    """The whole point of group_id: correctness must not depend on contiguity."""
    traj, turn, group, ret, tok, mask = _tensors()
    base_a = _compute_episode_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    w_a = _compute_episode_loss_weights(traj, mask, tok)

    perm = torch.randperm(len(TRAJ))
    base_b = _compute_episode_group_baseline(
        ret[perm], traj[perm], turn[perm], group[perm], mask[perm], leave_one_out=True
    )
    w_b = _compute_episode_loss_weights(traj[perm], mask[perm], tok[perm])

    assert torch.allclose(base_a[perm], base_b, atol=1e-6)
    assert torch.allclose(w_a[perm], w_b, atol=1e-6)


TURN_ROWS = [
    # trajectory, turn, group, return, tokens
    (11, 1, 0, 10.0, 1),
    (22, 2, 0, 2.0, 1),
    (44, 1, 1, 100.0, 1),
    (11, 3, 0, 1.0, 1),
    (33, 1, 0, 6.0, 1),
    (55, 2, 1, 20.0, 1),
    (22, 1, 0, 8.0, 1),
    (44, 2, 1, 40.0, 1),
    (11, 2, 0, 4.0, 1),
    (55, 1, 1, 80.0, 1),
]


def test_turn_baseline_compares_only_same_group_and_depth():
    traj, turn, group, ret, _tok, mask = _tensors(TURN_ROWS)
    baseline = _compute_turn_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )
    advantage = ret - baseline

    expected = {
        (11, 1): 3.0,
        (22, 1): 0.0,
        (33, 1): -3.0,
        (11, 2): 2.0,
        (22, 2): -2.0,
        (11, 3): 0.0,  # no zero-imputation: this depth is a singleton
        (44, 1): 20.0,
        (55, 1): -20.0,
        (44, 2): 20.0,
        (55, 2): -20.0,
    }
    for row in range(len(TURN_ROWS)):
        key = (int(traj[row]), int(turn[row]))
        assert advantage[row].item() == expected[key]


def test_turn_baseline_excludes_masked_rows_and_is_permutation_invariant():
    traj, turn, group, ret, _tok, mask = _tensors(TURN_ROWS)
    mask[(traj == 22) & (turn == 2)] = False
    ret[(traj == 22) & (turn == 2)] = 1_000_000.0
    baseline = _compute_turn_group_baseline(
        ret, traj, turn, group, mask, leave_one_out=True
    )

    # The remaining group-0 turn-2 row is a singleton; the masked huge value is
    # neither used as its peer nor assigned a baseline of its own.
    row_a2 = (traj == 11) & (turn == 2)
    assert torch.allclose(baseline[row_a2], ret[row_a2])
    assert baseline[(traj == 22) & (turn == 2)].item() == 0.0

    perm = torch.randperm(len(TURN_ROWS))
    permuted = _compute_turn_group_baseline(
        ret[perm],
        traj[perm],
        turn[perm],
        group[perm],
        mask[perm],
        leave_one_out=True,
    )
    assert torch.allclose(baseline[perm], permuted)
