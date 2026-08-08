from __future__ import annotations

from typing import Any

import torch

from examples.tutor.core.callers import apply_chat_template


def tokenize_teacher_forced_response(
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    *,
    enable_thinking: bool = False,
) -> tuple[list[int], list[int]]:
    """Tokenize a chat response and mask only its assistant target suffix."""

    prompt_ids = apply_chat_template(
        tokenizer,
        messages,
        enable_thinking=enable_thinking,
        add_generation_prompt=True,
    )
    full_ids = apply_chat_template(
        tokenizer,
        [*messages, {"role": "assistant", "content": response}],
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "World Model chat template is not prefix aligned between prompt and target."
        )
    target_len = len(full_ids) - len(prompt_ids)
    if target_len <= 0:
        raise ValueError("World Model response produced no target tokens.")
    return full_ids, [0] * len(prompt_ids) + [1] * target_len


def response_to_tensordict(
    response: Any,
    *,
    reward: float,
    trajectory_id: int | None = None,
    turn_idx: int | None = None,
    input_tokens_override: list[int] | None = None,
    zero_reward_on_length_stop: bool = False,
    batch_centered_penalty_score: float | None = None,
    batch_centered_penalty_weight: float | None = None,
    teacher_context_input_tokens: list[int] | None = None,
    teacher_context_reward_weight: float | None = None,
    teacher_context_reward_score_clip: float | None = None,
    teacher_context_reward_apply_to_advantage: bool | None = None,
    world_model_input_tokens: list[int] | None = None,
    world_model_target_mask: list[int] | None = None,
    world_model_loss_weight: float | None = None,
    world_model_paw_config: dict[str, Any] | None = None,
    opd_input_tokens: list[int] | None = None,
    opd_loss_weight: float | None = None,
    opd_reward_clip: float | None = None,
) -> dict[str, torch.Tensor]:
    input_tokens = (
        list(response.input_tokens)
        if input_tokens_override is None
        else list(input_tokens_override)
    )
    input_len = len(input_tokens)
    output_tokens = list(response.output_tokens)
    output_len = len(output_tokens)
    full_ids = input_tokens + output_tokens
    output_logprobs = list(response.output_logprobs)
    if len(output_logprobs) != output_len:
        raise ValueError(
            "ModelResponse output_logprobs length mismatch: "
            f"output_tokens={output_len}, "
            f"output_logprobs={len(output_logprobs)}, "
            f"stop_reason={getattr(response, 'stop_reason', None)!r}."
        )
    output_versions = list(response.output_versions)
    if len(output_versions) != output_len:
        raise ValueError(
            "ModelResponse output_versions length mismatch: "
            f"output_tokens={output_len}, "
            f"output_versions={len(output_versions)}, "
            f"stop_reason={getattr(response, 'stop_reason', None)!r}."
        )
    trajectory_value = 0 if trajectory_id is None else int(trajectory_id)
    turn_value = 0 if turn_idx is None else int(turn_idx)
    effective_reward = float(reward)
    if (
        zero_reward_on_length_stop
        and getattr(response, "stop_reason", None) == "length"
    ):
        effective_reward = 0.0
    result = {
        "input_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "logprobs": torch.tensor(
            [0.0] * input_len + output_logprobs,
            dtype=torch.float32,
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            [0] * input_len + [1] * output_len,
            dtype=torch.long,
        ).unsqueeze(0),
        "versions": torch.tensor(
            [-1] * input_len + output_versions,
            dtype=torch.long,
        ).unsqueeze(0),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([effective_reward], dtype=torch.float32),
        "trajectory_id": torch.tensor([trajectory_value], dtype=torch.long),
        "turn_idx": torch.tensor([turn_value], dtype=torch.long),
    }
    if batch_centered_penalty_weight is not None:
        score_valid = batch_centered_penalty_score is not None
        result["batch_centered_penalty_score"] = torch.tensor(
            [float(batch_centered_penalty_score) if score_valid else float("nan")],
            dtype=torch.float32,
        )
        result["batch_centered_penalty_weight"] = torch.tensor(
            [float(batch_centered_penalty_weight)],
            dtype=torch.float32,
        )
        result["batch_centered_penalty_valid"] = torch.tensor(
            [score_valid],
            dtype=torch.bool,
        )
    if teacher_context_reward_weight is not None:
        if teacher_context_reward_score_clip is None:
            raise ValueError(
                "teacher_context_reward_score_clip is required when the teacher "
                "context reward is enabled."
            )
        context_valid = teacher_context_input_tokens is not None and output_len > 0
        context_prompt_tokens = (
            list(teacher_context_input_tokens)
            if teacher_context_input_tokens is not None
            else input_tokens
        )
        context_ids = context_prompt_tokens + output_tokens
        result["teacher_context_input_ids"] = torch.tensor(
            context_ids, dtype=torch.long
        ).unsqueeze(0)
        result["teacher_context_attention_mask"] = torch.ones(
            len(context_ids), dtype=torch.bool
        ).unsqueeze(0)
        result["teacher_context_loss_mask"] = torch.tensor(
            [0] * len(context_prompt_tokens) + [1] * output_len,
            dtype=torch.long,
        ).unsqueeze(0)
        result["teacher_context_reward_weight"] = torch.tensor(
            [float(teacher_context_reward_weight)], dtype=torch.float32
        )
        result["teacher_context_reward_score_clip"] = torch.tensor(
            [float(teacher_context_reward_score_clip)], dtype=torch.float32
        )
        result["teacher_context_reward_valid"] = torch.tensor(
            [context_valid], dtype=torch.bool
        )
        result["teacher_context_reward_apply_to_advantage"] = torch.tensor(
            [
                True
                if teacher_context_reward_apply_to_advantage is None
                else bool(teacher_context_reward_apply_to_advantage)
            ],
            dtype=torch.bool,
        )
    if world_model_loss_weight is not None:
        wm_input_tokens = list(world_model_input_tokens or [])
        wm_target_mask = list(world_model_target_mask or [])
        if len(wm_input_tokens) != len(wm_target_mask):
            raise ValueError(
                "World Model input/mask length mismatch: "
                f"input={len(wm_input_tokens)}, mask={len(wm_target_mask)}."
            )
        if wm_input_tokens and not any(wm_target_mask):
            raise ValueError("World Model sample has no target tokens.")
        result["world_model_packed_input_ids"] = torch.tensor(
            wm_input_tokens, dtype=torch.long
        )
        result["world_model_packed_target_mask"] = torch.tensor(
            wm_target_mask, dtype=torch.bool
        )
        result["world_model_seq_lens"] = torch.tensor(
            [len(wm_input_tokens)], dtype=torch.long
        )
        result["world_model_loss_weight"] = float(world_model_loss_weight)
        result["world_model_paw_config"] = dict(world_model_paw_config or {})
    if opd_loss_weight is not None:
        # On-policy distillation: the same output tokens re-prefixed with the
        # instructed teacher prompt. The trainer teacher-forces this sequence to
        # get per-token log-probabilities, then realigns them onto the training
        # layout. A row with no target (weight 0) still carries a well-formed
        # single-token sequence so the padded batch stays rectangular.
        opd_tokens = list(opd_input_tokens or [])
        opd_active = bool(opd_tokens) and output_len > 0
        if not opd_active:
            opd_tokens = [0]
            opd_loss_mask = [0]
        else:
            opd_prompt_len = len(opd_tokens)
            opd_tokens = opd_tokens + output_tokens
            opd_loss_mask = [0] * opd_prompt_len + [1] * output_len
        result["opd_input_ids"] = torch.tensor(
            opd_tokens, dtype=torch.long
        ).unsqueeze(0)
        result["opd_attention_mask"] = torch.ones(
            len(opd_tokens), dtype=torch.bool
        ).unsqueeze(0)
        result["opd_loss_mask"] = torch.tensor(
            opd_loss_mask, dtype=torch.long
        ).unsqueeze(0)
        # Both scalars are broadcast across the full training row rather than
        # stored per row. Per-token columns split with the sequence during
        # micro-batching, and a value that is constant along the row cannot be
        # knocked out of alignment by the -1 roll the loss applies to its masks.
        # A weight of 0 on every position is also how an unselected row is marked.
        result["opd_token_weight"] = torch.full(
            (len(full_ids),),
            float(opd_loss_weight) if opd_active else 0.0,
            dtype=torch.float32,
        ).unsqueeze(0)
        result["opd_reward_clip"] = torch.full(
            (len(full_ids),),
            float(opd_reward_clip if opd_reward_clip is not None else 0.0),
            dtype=torch.float32,
        ).unsqueeze(0)
        result["opd_valid"] = torch.tensor([opd_active], dtype=torch.bool)
    return result
