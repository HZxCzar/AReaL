import json

import pytest
import torch

from areal.infra.workflow_executor import WorkflowExecutor


@pytest.mark.asyncio
async def test_dump_trajectory_persists_context_diagnostic_join_keys(tmp_path):
    """Readable rollout rows retain trajectory and turn identifiers."""

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):  # noqa: ARG002
            return " ".join(str(token_id) for token_id in token_ids)

    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor._get_dump_dir = lambda is_eval: str(tmp_path)  # noqa: ARG005
    executor._get_tokenizer = lambda: FakeTokenizer()
    trajectory_id = (1 << 62) + 19
    trajectory = {
        "input_ids": torch.tensor([[10, 11, 12], [20, 21, 22]]),
        "rewards": torch.tensor([0.0, 1.0]),
        "loss_mask": torch.tensor([[0, 1, 1], [0, 0, 1]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "versions": torch.tensor([[-1, 4, 4], [-1, -1, 4]]),
        "trajectory_id": torch.tensor([trajectory_id, trajectory_id]),
        "turn_idx": torch.tensor([1, 2]),
    }

    success, reason = await executor._dump_trajectory(
        trajectory, task_id=7, is_eval=False
    )

    assert success is True
    assert reason == ""
    with open(tmp_path / "4" / "7.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert [record["trajectory_id"] for record in records] == [
        trajectory_id,
        trajectory_id,
    ]
    assert [record["turn_idx"] for record in records] == [1, 2]
