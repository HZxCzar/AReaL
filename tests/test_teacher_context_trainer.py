from __future__ import annotations

import pytest
import torch

from areal.trainer.rl_trainer import (
    _attach_teacher_context_logps,
    _has_teacher_context_reward,
)


def _trajectory() -> dict[str, torch.Tensor]:
    return {
        "teacher_context_input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "teacher_context_attention_mask": torch.tensor(
            [[True, True, True], [True, True, False]]
        ),
        "teacher_context_loss_mask": torch.tensor([[0, 1, 1], [0, 1, 0]]),
        "teacher_context_reward_weight": torch.tensor([0.1, 0.1]),
        "teacher_context_reward_score_clip": torch.tensor([5.0, 5.0]),
        "teacher_context_reward_valid": torch.tensor([False, True]),
    }


def test_attach_teacher_context_logps_uses_counterfactual_sequences():
    """The extra actor forward sees only moved input IDs and attention masks."""

    class FakeActor:
        def compute_logp(self, batch):
            assert len(batch) == 1
            assert set(batch[0]) == {"input_ids", "attention_mask"}
            return [torch.full(batch[0]["input_ids"].shape, -2.0)]

    trajectory = _trajectory()

    assert _has_teacher_context_reward([trajectory]) is True
    _attach_teacher_context_logps(FakeActor(), [trajectory])

    torch.testing.assert_close(
        trajectory["teacher_context_logp"],
        torch.full((2, 3), -2.0),
        rtol=0.0,
        atol=0.0,
    )


def test_has_teacher_context_reward_rejects_partial_metadata():
    """A partial rollout cannot silently produce a malformed local reward."""
    trajectory = _trajectory()
    trajectory.pop("teacher_context_loss_mask")

    with pytest.raises(ValueError, match="teacher_context_loss_mask"):
        _has_teacher_context_reward([trajectory])
