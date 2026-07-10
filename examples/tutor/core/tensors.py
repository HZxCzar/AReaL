from __future__ import annotations

from typing import Any

import torch


def response_to_tensordict(
    response: Any,
    *,
    reward: float,
    trajectory_id: int | None = None,
    turn_idx: int | None = None,
    input_tokens_override: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    input_tokens = (
        list(response.input_tokens)
        if input_tokens_override is None
        else list(input_tokens_override)
    )
    input_len = len(input_tokens)
    full_ids = input_tokens + list(response.output_tokens)
    output_logprobs = list(response.output_logprobs)
    if len(output_logprobs) < response.output_len:
        output_logprobs.extend([0.0] * (response.output_len - len(output_logprobs)))
    if len(output_logprobs) > response.output_len:
        output_logprobs = output_logprobs[: response.output_len]
    output_versions = list(response.output_versions)
    if len(output_versions) < response.output_len:
        output_versions.extend([0] * (response.output_len - len(output_versions)))
    if len(output_versions) > response.output_len:
        output_versions = output_versions[: response.output_len]
    trajectory_value = 0 if trajectory_id is None else int(trajectory_id)
    turn_value = 0 if turn_idx is None else int(turn_idx)
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * input_len + output_logprobs,
            dtype=torch.float32,
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * input_len + [1] * response.output_len,
            dtype=torch.long,
        ).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * input_len + output_versions,
            dtype=torch.long,
        ).unsqueeze(0),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([float(reward)], dtype=torch.float32),
        "trajectory_id": torch.tensor([trajectory_value], dtype=torch.long),
        "turn_idx": torch.tensor([turn_value], dtype=torch.long),
    }
