"""End-to-end plumbing checks for the rollout-group path.

These exercise the real helpers the trainer uses, not reimplementations:
_attach_group_ids -> concat_batch -> microbatch split.
"""

import ast
import dataclasses
import inspect
import textwrap

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec, PPOConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.rl_trainer import _attach_group_ids
from areal.utils.data import concat_batch, split_padded_tensor_dict_into_mb_list


def _episode(traj_id: int, n_turns: int, seqlen: int = 8):
    """One episode's turn rows, shaped like examples/tutor/core/tensors.py emits."""
    return {
        "input_ids": torch.randint(1, 100, (n_turns, seqlen), dtype=torch.long),
        "loss_mask": torch.ones((n_turns, seqlen), dtype=torch.long),
        "attention_mask": torch.ones((n_turns, seqlen), dtype=torch.bool),
        "logprobs": torch.zeros((n_turns, seqlen), dtype=torch.float32),
        "versions": torch.full((n_turns, seqlen), -1, dtype=torch.long),
        "rewards": torch.zeros(n_turns, dtype=torch.float32),
        "trajectory_id": torch.full((n_turns,), traj_id, dtype=torch.long),
        "turn_idx": torch.arange(1, n_turns + 1, dtype=torch.long),
    }


def _group(traj_ids_and_turns, seqlen=8):
    """One dataset row = one rollout group = G episodes concatenated."""
    eps = [_episode(tid, n, seqlen) for tid, n in traj_ids_and_turns]
    return {k: torch.cat([e[k] for e in eps], dim=0) for k in eps[0]}


def test_group_ids_align_with_rows_after_concat():
    # ragged on both axes: different episode counts AND different turn counts
    rollout_batch = [
        _group([(101, 2), (102, 1), (103, 4)]),  # group 0 -> 7 rows
        _group([(201, 1), (202, 3)]),  # group 1 -> 4 rows
        _group([(301, 10)]),  # group 2 -> 10 rows (singleton)
    ]
    _attach_group_ids(rollout_batch)

    for i, traj in enumerate(rollout_batch):
        assert traj["group_id"].shape == traj["rewards"].shape
        assert (traj["group_id"] == i).all()

    batched, meta = concat_batch(rollout_batch)
    assert meta.traj_group_sizes == [7, 4, 10]

    gid = batched["group_id"]
    assert gid.shape[0] == 21
    # group_id must line up row-for-row with trajectory_id after concat
    assert gid[:7].eq(0).all() and gid[7:11].eq(1).all() and gid[11:].eq(2).all()

    # every episode sits entirely inside exactly one group
    for tid in batched["trajectory_id"].unique().tolist():
        rows = batched["trajectory_id"] == tid
        assert gid[rows].unique().numel() == 1, tid


def test_per_row_metadata_must_be_popped_before_microbatching():
    """Why group_id belongs in _ppo_update's pop list.

    The splitter treats 1-D ``[n_rows]`` tensors as non-batch and copies them whole
    into every microbatch, so any per-row metadata left in ``data`` arrives
    misaligned with the rows it is supposed to describe.
    """
    rollout_batch = [_group([(101, 2), (102, 3)]), _group([(201, 2), (202, 1)])]
    _attach_group_ids(rollout_batch)
    batched, _meta = concat_batch(rollout_batch)
    assert batched["group_id"].shape[0] == 8

    mbs = split_padded_tensor_dict_into_mb_list(
        batched, mb_spec=MicroBatchSpec(n_mbs=2)
    )
    chunks = mbs.mbs if hasattr(mbs, "mbs") else mbs
    for mb in chunks:
        # rows really are split ...
        assert mb["input_ids"].shape[0] == 4
        # ... but 1-D metadata is not, so it no longer matches the rows.
        assert mb["group_id"].shape[0] == 8

    # Hence group_id must be dropped alongside the other per-row metadata.
    for key in (
        "trajectory_id",
        "turn_idx",
        "group_id",
        "group_baseline",
        "episode_loss_weight",
    ):
        assert key in _popped_keys(), f"{key} missing from _ppo_update's pop list"


def _popped_keys() -> set[str]:
    """Keys _ppo_update drops from `data` before microbatching, read from source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(PPOActor._ppo_update)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.For) and isinstance(node.iter, ast.List)):
            continue
        if "pop" not in ast.dump(node):
            continue
        keys |= {
            element.value
            for element in node.iter.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    assert keys, "could not locate the pop list in _ppo_update"
    return keys


def test_attach_group_ids_is_idempotent_and_row_exact():
    rollout_batch = [_group([(101, 3)]), _group([(201, 1), (202, 2)])]
    _attach_group_ids(rollout_batch)
    first = [t["group_id"].clone() for t in rollout_batch]
    _attach_group_ids(rollout_batch)
    for a, b in zip(first, rollout_batch, strict=True):
        assert torch.equal(a, b["group_id"])


def test_group_baseline_requires_a_group_larger_than_one():
    """n_samples=1 would make every advantage exactly zero -- reject it loudly."""
    cfg = PPOConfig(experiment_name="t", trial_name="t")
    cfg.actor.advantage_estimator = "rebn"
    cfg.actor.group_baseline = "episode"
    cfg.gconfig = dataclasses.replace(cfg.gconfig, n_samples=1)
    with pytest.raises(ValueError, match="gconfig.n_samples >= 2"):
        cfg.__post_init__()

    cfg.gconfig = dataclasses.replace(cfg.gconfig, n_samples=8)
    cfg.__post_init__()  # must not raise
