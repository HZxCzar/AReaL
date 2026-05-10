from __future__ import annotations

from typing import Any

import torch


def response_to_tensordict(response: Any, *, reward: float) -> dict[str, torch.Tensor]:
    full_ids = list(response.input_tokens) + list(response.output_tokens)
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
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * response.input_len + output_logprobs,
            dtype=torch.float32,
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * response.input_len + [1] * response.output_len,
            dtype=torch.long,
        ).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * response.input_len + output_versions,
            dtype=torch.long,
        ).unsqueeze(0),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([float(reward)], dtype=torch.float32),
    }
