# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any

import torch
import torch.distributed as dist

from areal.api import TrainEngine
from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra import TrainController
from areal.infra.rpc.serialization import serialize_value
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    batched_call,
    concat_batch,
    concat_padded_tensors,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("PPOActor")


def _unpack_world_model_rows(
    world_model_batch: dict[str, Any], expected_rows: int
) -> list[tuple[torch.Tensor, torch.Tensor, bool, float]]:
    """Restore row-aligned variable-length WM samples from rollout sidecars."""

    input_ids = world_model_batch["world_model_packed_input_ids"]
    target_mask = world_model_batch["world_model_packed_target_mask"]
    seq_lens = world_model_batch["world_model_seq_lens"]
    if input_ids.ndim != 1 or target_mask.ndim != 1 or seq_lens.ndim != 1:
        raise ValueError("World Model sidecar tensors must be packed 1D tensors.")
    if input_ids.numel() != target_mask.numel():
        raise ValueError(
            "World Model packed input/mask length mismatch: "
            f"input={input_ids.numel()}, mask={target_mask.numel()}."
        )
    if seq_lens.numel() != expected_rows:
        raise ValueError(
            "World Model row alignment mismatch: "
            f"expected={expected_rows}, lengths={seq_lens.numel()}."
        )

    # Rollout transport keeps samples packed. One small metadata copy is required to
    # restore sequence boundaries before the GPU training forward.
    lengths = [int(length) for length in seq_lens.detach().cpu().tolist()]
    if any(length < 0 for length in lengths):
        raise ValueError("World Model sequence lengths must be non-negative.")
    if sum(lengths) != input_ids.numel():
        raise ValueError(
            "World Model packed length mismatch: "
            f"sum(seq_lens)={sum(lengths)}, packed={input_ids.numel()}."
        )
    input_rows = torch.split(input_ids, lengths)
    mask_rows = torch.split(target_mask, lengths)
    selected = world_model_batch.get("world_model_selected")
    if selected is None:
        selected_values = [length > 0 for length in lengths]
    else:
        if selected.numel() != expected_rows:
            raise ValueError("World Model selection metadata is not row-aligned.")
        selected_values = [bool(value) for value in selected.cpu().reshape(-1)]
    response_weight = world_model_batch.get("world_model_response_weight")
    if response_weight is None:
        response_weights = [1.0] * expected_rows
    else:
        if response_weight.numel() != expected_rows:
            raise ValueError("World Model response weights are not row-aligned.")
        response_weights = [float(value) for value in response_weight.cpu().reshape(-1)]
    return list(
        zip(
            input_rows,
            mask_rows,
            selected_values,
            response_weights,
            strict=True,
        )
    )


def _prepare_world_model_logp_batch(
    world_model_batch: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[int]] | None:
    """Build one temperature-one forward batch containing every valid WM row."""

    expected_rows = world_model_batch["world_model_seq_lens"].numel()
    rows = _unpack_world_model_rows(world_model_batch, expected_rows)
    valid_indices = [
        index
        for index, (input_ids, target_mask, _, _) in enumerate(rows)
        if input_ids.numel() > 0 and bool(target_mask.any())
    ]
    if not valid_indices:
        return None

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [rows[index][0] for index in valid_indices], batch_first=True
    )
    target_mask = torch.nn.utils.rnn.pad_sequence(
        [rows[index][1].bool() for index in valid_indices], batch_first=True
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones_like(rows[index][0], dtype=torch.bool) for index in valid_indices],
        batch_first=True,
    )
    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_logprob_temperature": torch.ones_like(
                input_ids, dtype=torch.float32
            ),
        },
        target_mask,
        valid_indices,
    )


def _append_world_model_rows(
    policy_batch: dict[str, Any],
    world_model_rows: list[tuple[Any, ...]],
    *,
    policy_temperature: float,
) -> dict[str, Any]:
    """Append independent WM sequences without exposing them to PPO masks."""

    if policy_temperature <= 0:
        raise ValueError("Policy temperature must be positive.")

    policy_input_ids = policy_batch["input_ids"]
    policy_attention_mask = policy_batch["attention_mask"]
    if policy_input_ids.ndim != 2 or policy_attention_mask.ndim != 2:
        raise ValueError("Policy minibatch must use padded 2D input tensors.")
    policy_bs, policy_seqlen = policy_input_ids.shape

    normalized_rows = [
        (
            row[0],
            row[1],
            bool(row[2]) if len(row) > 2 else True,
            float(row[3]) if len(row) > 3 else 1.0,
        )
        for row in world_model_rows
    ]
    valid_rows = [row for row in normalized_rows if row[0].numel() > 0 and row[2]]
    max_world_model_len = max(
        (input_ids.numel() for input_ids, _, _, _ in valid_rows), default=0
    )
    joint_seqlen = max(policy_seqlen, max_world_model_len)
    world_model_bs = len(valid_rows)
    joint_batch: dict[str, Any] = {}

    for key, value in policy_batch.items():
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.shape[0] == policy_bs
            and value.shape[1] == policy_seqlen
        ):
            if value.ndim != 2:
                raise ValueError(
                    f"Unsupported token-aligned tensor rank for {key!r}: {value.ndim}."
                )
            policy_pad = torch.zeros(
                (policy_bs, joint_seqlen - policy_seqlen),
                dtype=value.dtype,
                device=value.device,
            )
            padded_policy = torch.cat((value, policy_pad), dim=1)
            world_model_values = torch.zeros(
                (world_model_bs, joint_seqlen),
                dtype=value.dtype,
                device=value.device,
            )
            if key == "versions":
                world_model_values.fill_(-1)
            joint_batch[key] = torch.cat((padded_policy, world_model_values), dim=0)
        else:
            joint_batch[key] = value

    world_model_input_ids = torch.zeros(
        (world_model_bs, joint_seqlen),
        dtype=policy_input_ids.dtype,
        device=policy_input_ids.device,
    )
    world_model_attention_mask = torch.zeros(
        (world_model_bs, joint_seqlen),
        dtype=policy_attention_mask.dtype,
        device=policy_attention_mask.device,
    )
    world_model_target_mask = torch.zeros(
        (world_model_bs, joint_seqlen),
        dtype=torch.bool,
        device=policy_input_ids.device,
    )
    world_model_response_weight = torch.zeros(
        (world_model_bs, 1), dtype=torch.float32, device=policy_input_ids.device
    )
    for row_idx, (input_ids, target_mask, _, response_weight) in enumerate(valid_rows):
        if input_ids.numel() != target_mask.numel():
            raise ValueError(
                "World Model row input/mask length mismatch: "
                f"input={input_ids.numel()}, mask={target_mask.numel()}."
            )
        row_len = input_ids.numel()
        world_model_input_ids[row_idx, :row_len] = input_ids
        world_model_attention_mask[row_idx, :row_len] = True
        world_model_target_mask[row_idx, :row_len] = target_mask.bool()
        world_model_response_weight[row_idx] = response_weight

    if world_model_bs:
        joint_batch["input_ids"][policy_bs:] = world_model_input_ids
        joint_batch["attention_mask"][policy_bs:] = world_model_attention_mask

    aligned_target_mask = torch.roll(world_model_target_mask, shifts=-1, dims=-1)
    target_counts = aligned_target_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    world_model_token_weight = (
        aligned_target_mask.float() / target_counts * world_model_response_weight
    )
    response_start_mask = aligned_target_mask & (
        aligned_target_mask.long().cumsum(dim=-1) == 1
    )

    zero_policy_mask = torch.zeros(
        (policy_bs, joint_seqlen), dtype=torch.bool, device=policy_input_ids.device
    )
    zero_policy_weight = torch.zeros(
        (policy_bs, joint_seqlen), dtype=torch.float32, device=policy_input_ids.device
    )
    joint_batch["world_model_loss_mask"] = torch.cat(
        (zero_policy_mask, aligned_target_mask), dim=0
    )
    joint_batch["world_model_token_weight"] = torch.cat(
        (zero_policy_weight, world_model_token_weight), dim=0
    )
    joint_batch["world_model_response_start_mask"] = torch.cat(
        (zero_policy_mask, response_start_mask), dim=0
    )
    joint_batch["token_logprob_temperature"] = torch.cat(
        (
            torch.full(
                (policy_bs, joint_seqlen),
                float(policy_temperature),
                dtype=torch.float32,
                device=policy_input_ids.device,
            ),
            torch.ones(
                (world_model_bs, joint_seqlen),
                dtype=torch.float32,
                device=policy_input_ids.device,
            ),
        ),
        dim=0,
    )
    return joint_batch


def _build_world_model_train_batch(
    rows: list[tuple[torch.Tensor, torch.Tensor, bool, float]],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Build a WM-only batch; an empty local shard becomes a zero-loss dummy."""

    batch_device = rows[0][0].device if rows else device or torch.device("cpu")
    if rows:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [row[0] for row in rows], batch_first=True
        )
        target_mask = torch.nn.utils.rnn.pad_sequence(
            [row[1].bool() for row in rows], batch_first=True
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(row[0], dtype=torch.bool) for row in rows],
            batch_first=True,
        )
        response_weight = torch.tensor(
            [row[3] for row in rows],
            dtype=torch.float32,
            device=batch_device,
        ).unsqueeze(-1)
    else:
        input_ids = torch.zeros((1, 1), dtype=torch.long, device=batch_device)
        target_mask = torch.zeros((1, 1), dtype=torch.bool, device=batch_device)
        attention_mask = torch.ones((1, 1), dtype=torch.bool, device=batch_device)
        response_weight = torch.zeros((1, 1), dtype=torch.float32, device=batch_device)

    aligned_target_mask = torch.roll(target_mask, shifts=-1, dims=-1)
    target_counts = aligned_target_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    token_weight = aligned_target_mask.float() / target_counts * response_weight
    response_start_mask = aligned_target_mask & (
        aligned_target_mask.long().cumsum(dim=-1) == 1
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_logprob_temperature": torch.ones_like(input_ids, dtype=torch.float32),
        "world_model_loss_mask": aligned_target_mask,
        "world_model_token_weight": token_weight,
        "world_model_response_start_mask": response_start_mask,
    }


def _joint_loss_weight(input_data: dict[str, Any]) -> torch.Tensor:
    return (
        input_data["loss_mask"].count_nonzero()
        + input_data["world_model_response_start_mask"].count_nonzero()
    )


def _policy_loss_weight(input_data: dict[str, Any]) -> torch.Tensor:
    return input_data["loss_mask"].count_nonzero()


def _world_model_loss_weight(input_data: dict[str, Any]) -> torch.Tensor:
    return input_data["world_model_response_start_mask"].count_nonzero()


def world_model_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    *,
    world_model_loss_weight: float,
    paw_enabled: bool = False,
    cmae_enabled: bool = False,
    confidence_threshold: float = 0.2,
    **_: Any,
) -> torch.Tensor:
    """Response-balanced CE/CMAE objective for the isolated WM adapter."""

    del entropy
    loss_mask = input_data["world_model_loss_mask"].bool()
    token_weight = input_data["world_model_token_weight"].to(logprobs.dtype)
    response_starts = input_data["world_model_response_start_mask"].bool()
    target_probabilities = logprobs.exp()
    if paw_enabled and cmae_enabled:
        confidence_mask = target_probabilities <= float(confidence_threshold)
        token_loss = (1.0 - target_probabilities) * confidence_mask
        clipped_mask = loss_mask & ~confidence_mask
    else:
        confidence_mask = loss_mask
        clipped_mask = torch.zeros_like(loss_mask)
        token_loss = -logprobs

    response_count = response_starts.count_nonzero().to(logprobs.dtype)
    response_loss_sum = (token_loss * token_weight).sum()
    coefficient = torch.as_tensor(
        world_model_loss_weight, dtype=logprobs.dtype, device=logprobs.device
    )
    stats_tracker.denominator(
        world_model_target_tokens=loss_mask,
        world_model_responses=response_starts,
    )
    stats_tracker.stat(
        world_model_token_nll=(-logprobs.detach()).float(),
        world_model_cmae_active=confidence_mask.detach().float(),
        world_model_cmae_clipped=clipped_mask.detach().float(),
        denominator="world_model_target_tokens",
    )
    return torch.where(
        response_count > 0,
        coefficient * response_loss_sum / response_count.clamp_min(1),
        logprobs.sum() * 0.0,
    )


def _global_joint_counts(
    input_data: dict[str, Any],
    *,
    device: torch.device,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Count PPO tokens and WM responses on the collective backend's device."""

    counts = torch.stack(
        (
            input_data["loss_mask"].count_nonzero(),
            input_data["world_model_response_start_mask"].count_nonzero(),
        )
    ).to(device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(counts, group=group)
    return counts


def _merge_policy_world_model_loss(
    policy_loss: torch.Tensor,
    logprobs: torch.Tensor,
    input_data: dict[str, Any],
    *,
    global_policy_tokens: torch.Tensor,
    global_world_model_responses: torch.Tensor,
    world_model_loss_weight: float,
    paw_enabled: bool = False,
    cmae_enabled: bool = False,
    confidence_threshold: float = 0.2,
) -> torch.Tensor:
    """Combine separately normalized PPO and optional CE/CMAE WM objectives."""

    policy_tokens = input_data["loss_mask"].count_nonzero().to(logprobs.dtype)
    response_starts = input_data["world_model_response_start_mask"].bool()
    world_model_responses = response_starts.count_nonzero().to(logprobs.dtype)
    local_weight = policy_tokens + world_model_responses

    world_model_token_weight = input_data["world_model_token_weight"].to(logprobs.dtype)
    world_model_loss_mask = input_data["world_model_loss_mask"].bool()
    target_probabilities = logprobs.exp()
    if paw_enabled and cmae_enabled:
        confidence_mask = target_probabilities <= float(confidence_threshold)
        token_loss = (1.0 - target_probabilities) * confidence_mask
        clipped_mask = world_model_loss_mask & ~confidence_mask
    else:
        confidence_mask = world_model_loss_mask
        clipped_mask = torch.zeros_like(world_model_loss_mask)
        token_loss = -logprobs
    response_loss_sum = (token_loss * world_model_token_weight).sum()

    global_policy_tokens = global_policy_tokens.to(logprobs.dtype)
    global_world_model_responses = global_world_model_responses.to(logprobs.dtype)
    global_weight = global_policy_tokens + global_world_model_responses
    policy_scale = global_weight / global_policy_tokens.clamp_min(1)
    world_model_scale = global_weight / global_world_model_responses.clamp_min(1)
    world_model_active = (global_world_model_responses > 0).to(logprobs.dtype)
    world_model_coefficient = torch.as_tensor(
        world_model_loss_weight, dtype=logprobs.dtype, device=logprobs.device
    )
    numerator = policy_scale * policy_tokens * policy_loss
    numerator = numerator + (
        world_model_coefficient
        * world_model_active
        * world_model_scale
        * response_loss_sum
    )

    stats_tracker.denominator(
        world_model_target_tokens=world_model_loss_mask,
        world_model_responses=response_starts,
    )
    stats_tracker.stat(
        world_model_token_nll=(-logprobs.detach()).float(),
        world_model_cmae_active=confidence_mask.detach().float(),
        world_model_cmae_clipped=clipped_mask.detach().float(),
        denominator="world_model_target_tokens",
    )
    cu_seqlens = input_data["cu_seqlens"]
    sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    sequence_ids = torch.repeat_interleave(
        torch.arange(
            sequence_lengths.numel(), device=logprobs.device, dtype=torch.long
        ),
        sequence_lengths,
    )
    response_losses = torch.zeros(
        sequence_lengths.numel(), dtype=logprobs.dtype, device=logprobs.device
    )
    response_losses.scatter_add_(
        0, sequence_ids, token_loss.detach() * world_model_token_weight
    )
    response_loss_per_token = response_losses[sequence_ids].float()
    stats_tracker.stat(
        world_model_loss=response_loss_per_token,
        weighted_world_model_loss=(
            response_loss_per_token * world_model_coefficient.detach().float()
        ),
        world_model_to_ppo_loss_ratio=(
            response_loss_per_token
            * world_model_coefficient.detach().float()
            / policy_loss.detach().abs().float().clamp_min(1e-6)
        ),
        denominator="world_model_responses",
    )

    return torch.where(
        local_weight > 0,
        numerator / local_weight.clamp_min(1),
        logprobs.sum() * 0.0,
    )


def _compute_rebn_returns(
    rewards: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_indices: torch.Tensor,
    turn_discount: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted future returns over turn-level trajectories."""
    rewards = rewards.float()
    returns = torch.zeros_like(rewards)
    if rewards.numel() == 0:
        return returns
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return returns

    unique_trajectory_ids = torch.unique(trajectory_ids[valid_indices])
    gamma = float(turn_discount)
    for trajectory_id in unique_trajectory_ids.tolist():
        traj_mask = valid_mask & (trajectory_ids == trajectory_id)
        traj_indices = torch.nonzero(traj_mask, as_tuple=False).flatten()
        if traj_indices.numel() == 0:
            continue
        order = torch.argsort(turn_indices[traj_indices])
        ordered_indices = traj_indices[order]
        running_return = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for idx in reversed(ordered_indices.tolist()):
            running_return = rewards[idx] + gamma * running_return
            returns[idx] = running_return
    return returns


def _episode_local_at_first_turn(
    local_rewards: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_indices: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Each episode's total turn-local reward, gathered onto its first turn.

    The group baseline reads one scalar per episode off the return at the
    episode's smallest turn_idx, and that scalar has to be the episode TOTAL or
    the advantages stop centring. Turn-local components deliberately stay on
    their own turn in the returns, so for the baseline they are collected here
    instead. Without this a leak on the last turn appears in a turn return but in
    no baseline, leaving a constant negative advantage on every turn that nothing
    cancels -- which deflates the logits and drives entropy up.
    """
    gathered = torch.zeros_like(local_rewards)
    valid = valid_mask.bool()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return gathered
    for trajectory_id in torch.unique(trajectory_ids[valid_indices]).tolist():
        traj_mask = valid & (trajectory_ids == trajectory_id)
        rows = torch.nonzero(traj_mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        first_row = rows[torch.argmin(turn_indices[rows])]
        gathered[first_row] = local_rewards[rows].sum()
    return gathered


def _episode_scalars(
    turn_returns: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_indices: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce turn rows to one scalar per episode.

    The episode scalar is the ReBN return at the episode's smallest ``turn_idx``,
    i.e. the full (discounted) episode return. Returns ``(episode_index_per_row,
    episode_return, n_episodes)`` where ``episode_index_per_row`` is -1 on invalid
    rows.
    """
    device = turn_returns.device
    n_rows = turn_returns.shape[0]
    episode_index = torch.full((n_rows,), -1, dtype=torch.long, device=device)

    valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return episode_index, turn_returns.new_zeros(0), 0

    unique_ids, inverse = torch.unique(trajectory_ids[valid_rows], return_inverse=True)
    n_episodes = int(unique_ids.numel())
    episode_index[valid_rows] = inverse

    # Pick the row with the smallest turn_idx within each episode.
    big = torch.iinfo(torch.long).max
    first_turn = torch.full((n_episodes,), big, dtype=torch.long, device=device)
    first_turn.scatter_reduce_(
        0, inverse, turn_indices[valid_rows], reduce="amin", include_self=True
    )
    is_first = turn_indices[valid_rows] == first_turn[inverse]

    episode_return = turn_returns.new_zeros(n_episodes)
    episode_return.scatter_(0, inverse[is_first], turn_returns[valid_rows][is_first])
    return episode_index, episode_return, n_episodes


def _compute_episode_group_baseline(
    turn_returns: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_indices: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out: bool,
) -> torch.Tensor:
    """Per-row baseline: the mean episode return of the row's rollout group.

    The mean is taken over *episodes*, not over turn rows, so an episode that ran
    ten turns does not pull the baseline ten times harder than a one-turn episode.
    The caller subtracts this from the turn returns, which preserves within-episode
    return differences exactly.
    """
    device = turn_returns.device
    baseline = torch.zeros_like(turn_returns)

    episode_index, episode_return, n_episodes = _episode_scalars(
        turn_returns, trajectory_ids, turn_indices, valid_mask
    )
    if n_episodes == 0:
        return baseline

    # One group id per episode (constant within an episode by construction).
    valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
    episode_group = torch.zeros(n_episodes, dtype=torch.long, device=device)
    episode_group.scatter_(0, episode_index[valid_rows], group_ids[valid_rows])

    unique_groups, group_inverse = torch.unique(episode_group, return_inverse=True)
    n_groups = int(unique_groups.numel())

    group_sum = turn_returns.new_zeros(n_groups)
    group_sum.index_add_(0, group_inverse, episode_return)
    group_count = turn_returns.new_zeros(n_groups)
    group_count.index_add_(0, group_inverse, torch.ones_like(episode_return))

    if leave_one_out:
        # Exclude each episode from its own baseline; fall back to the plain mean
        # for singleton groups, where leave-one-out is undefined.
        denom = (group_count[group_inverse] - 1.0).clamp_min(1.0)
        episode_baseline = torch.where(
            group_count[group_inverse] > 1.0,
            (group_sum[group_inverse] - episode_return) / denom,
            group_sum[group_inverse] / group_count[group_inverse].clamp_min(1.0),
        )
    else:
        episode_baseline = group_sum[group_inverse] / group_count[
            group_inverse
        ].clamp_min(1.0)

    baseline[valid_rows] = episode_baseline[episode_index[valid_rows]]
    return baseline


def _compute_turn_group_baseline(
    turn_returns: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_indices: torch.Tensor,
    group_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    leave_one_out: bool,
) -> torch.Tensor:
    """Per-row baseline from other episodes at the same group and turn depth.

    Ragged episodes contribute only at turns they actually reached. A singleton
    ``(group_id, turn_idx)`` stratum uses its own return as the baseline, yielding
    zero relative advantage instead of importing a value from another depth.
    """
    baseline = torch.zeros_like(turn_returns)
    valid_rows = torch.nonzero(valid_mask.bool(), as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return baseline

    # ReBN exports exactly one row per episode and turn. Check this explicitly so
    # leave-one-out never accidentally compares a trajectory with a duplicate of
    # itself.
    member_keys = torch.stack(
        (
            group_ids[valid_rows],
            turn_indices[valid_rows],
            trajectory_ids[valid_rows],
        ),
        dim=1,
    )
    if torch.unique(member_keys, dim=0).shape[0] != valid_rows.numel():
        raise ValueError(
            "Turn group baseline requires at most one valid row per "
            "(group_id, turn_idx, trajectory_id)."
        )

    stratum_keys = torch.stack((group_ids[valid_rows], turn_indices[valid_rows]), dim=1)
    _, stratum_inverse = torch.unique(stratum_keys, dim=0, return_inverse=True)
    n_strata = int(stratum_inverse.max().item()) + 1
    valid_returns = turn_returns[valid_rows]

    stratum_sum = turn_returns.new_zeros(n_strata)
    stratum_sum.index_add_(0, stratum_inverse, valid_returns)
    stratum_count = turn_returns.new_zeros(n_strata)
    stratum_count.index_add_(0, stratum_inverse, torch.ones_like(valid_returns))

    if leave_one_out:
        counts = stratum_count[stratum_inverse]
        row_baseline = torch.where(
            counts > 1.0,
            (stratum_sum[stratum_inverse] - valid_returns) / (counts - 1.0),
            valid_returns,
        )
    else:
        row_baseline = stratum_sum[stratum_inverse] / stratum_count[
            stratum_inverse
        ].clamp_min(1.0)

    baseline[valid_rows] = row_baseline
    return baseline


def _realign_opd_teacher_logp(
    teacher_logp: torch.Tensor,
    opd_loss_mask: torch.Tensor,
    rolled_train_mask: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    """Move teacher log-probs from the instructed-teacher layout onto the
    training layout.

    The teacher prompt carries an extra instruction, so it is longer than the
    training prompt and the same output tokens sit at a different offset. Both
    sides cover exactly those output tokens in the same order, so selecting with
    each layout's own mask and copying across is correct without either prompt
    length appearing anywhere.

    Most rows of a real batch are NOT supervised -- a turn is skipped when it is
    too early, guided, or leaked -- and those rows carry a one-token placeholder
    in the teacher layout. Their real output tokens still exist on the training
    side, so the training selection has to be restricted to the supervised rows
    or the two counts differ by roughly an order of magnitude. Row activity is
    derived from ``opd_loss_mask`` itself rather than taken as an argument, so a
    caller cannot get this wrong.

    ``rolled_train_mask`` is already rolled by -1 by the caller;
    ``opd_loss_mask`` is the raw rollout column and is rolled here. The roll is
    what converts "these positions hold output tokens" into "these positions
    predict output tokens", which is the convention `compute_logp` returns
    (labels = roll(input_ids, -1)).
    """
    teacher_logp = teacher_logp.to(template.dtype)
    width = min(teacher_logp.shape[-1], opd_loss_mask.shape[-1])
    opd_loss_mask = opd_loss_mask[..., :width].bool()
    teacher_selection = torch.roll(opd_loss_mask, shifts=-1, dims=-1)
    selected = teacher_logp[..., :width][teacher_selection]

    supervised_rows = opd_loss_mask.any(dim=-1, keepdim=True)
    train_selection = rolled_train_mask.bool() & supervised_rows
    expected = int(train_selection.count_nonzero())
    if selected.numel() != expected:
        raise RuntimeError(
            "OPD token count mismatch between the teacher and training layouts: "
            f"teacher={selected.numel()}, training={expected} over "
            f"{int(supervised_rows.count_nonzero())}/{opd_loss_mask.shape[0]} "
            f"supervised rows (teacher width {teacher_logp.shape[-1]}, mask width "
            f"{opd_loss_mask.shape[-1]}). Both sides must cover the same output "
            "tokens, and the teacher prompt must end where generation began."
        )
    aligned = torch.zeros_like(template)
    aligned[train_selection] = selected
    return aligned


def _compute_opd_advantages(
    behaviour_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    token_weight: torch.Tensor,
    reward_clip: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-policy distillation as a per-token advantage penalty.

    Follows the reference formulation (Thinking Machines, on-policy distillation;
    ``tinker_cookbook/distillation/train_on_policy.py``):

        reverse_kl = log pi_sampled(a_t) - log pi_teacher(a_t)
        advantages = advantages - coef * reverse_kl

    and then the ordinary importance-sampling/PPO loss runs unchanged, so the
    distillation signal inherits the same ratio, clipping and behaviour-importance
    weighting as the task advantage.

    Three details that are easy to get wrong, all matching the reference:

    * The KL is evaluated **only on the sampled token**, not over the full
      vocabulary. Both sides are scored on the tokens the student actually
      produced.
    * The student side is the **sampling-time** log-prob, not a recomputed
      current-policy one. It is data, fixed for the whole update.
    * Discount factor zero: the penalty lands on the token that produced it and is
      not accumulated forward, so it never enters GAE. That is why this is added
      after the task advantage is complete.

    Returning it as an advantage rather than a loss term is also what makes the
    gradient treatment right for free. Nothing here differentiates through the
    sampling distribution -- `behaviour_logp` and `teacher_logp` are both plain
    data at this point -- so the only gradient path is the policy-gradient one the
    PPO surrogate already provides.

    ``reward_clip`` is a local addition, not part of the reference: a single
    outlier token would otherwise pass straight into the advantage. Set it to 0 to
    disable and match the reference exactly.
    """
    active = valid_mask.bool() & (token_weight > 0)
    reverse_kl = behaviour_logp - teacher_logp
    clipped = torch.minimum(torch.maximum(reverse_kl, -reward_clip), reward_clip)
    reverse_kl = torch.where(reward_clip > 0, clipped, reverse_kl)
    reverse_kl = reverse_kl * active
    return -token_weight * reverse_kl, reverse_kl, active


def _compute_episode_loss_weights(
    trajectory_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    """Per-row weight equalizing each episode's gradient mass.

    The PPO loss is a token-level mean, so an episode's gradient weight is
    proportional to its total trainable tokens. Scaling every turn's advantage by
    ``mean_episode_tokens / this_episode_tokens`` makes each episode contribute
    equally while leaving the overall loss scale unchanged. Scaling advantages is
    exactly equivalent to scaling the loss contribution: the PPO clip is applied
    to the ratio, so ``pg_loss`` is positively homogeneous in the advantage.
    """
    weights = torch.ones_like(token_counts, dtype=torch.float32)
    valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return weights

    _unique, inverse = torch.unique(trajectory_ids[valid_rows], return_inverse=True)
    n_episodes = int(_unique.numel())

    episode_tokens = torch.zeros(
        n_episodes, dtype=torch.float32, device=token_counts.device
    )
    episode_tokens.index_add_(0, inverse, token_counts[valid_rows].to(torch.float32))
    episode_tokens = episode_tokens.clamp_min(1.0)
    mean_tokens = episode_tokens.mean()

    weights[valid_rows] = (mean_tokens / episode_tokens)[inverse]
    return weights


def _compute_turn_loss_weights(
    valid_mask: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    """Per-row weight equalizing each turn's gradient mass.

    The same identity the episode version rests on -- the loss is a token mean, so
    scaling a row's advantage scales its loss contribution exactly -- applied one
    level down: every turn contributes the mean turn's worth of gradient however
    long it is.

    THIS IS THE LEVEL THAT BOUNDS ONE DEGENERATE GENERATION. Under 'token' a turn's
    share of the batch is its length, and a teacher that falls into a repetition
    loop runs to gconfig.max_new_tokens before anything stops it. On
    20260822_133603 those capped 4096-token turns took 1.7% of the batch gradient,
    then 14.1%, then 47.7% over two steps, each carrying an advantage near -4 sigma,
    and the policy did not come back. Recomputed at 'turn' the same three batches
    put them at 0.1%, 0.9% and 4.9%.

    The episode level does not reach this. It normalizes by an episode's total
    tokens, and a repetition that terminates its own episode on turn one carries
    4096 of them against a mean episode roughly half that -- so it is scaled by
    about a half, and still held 28.7% of that same batch.
    """
    weights = torch.ones_like(token_counts, dtype=torch.float32)
    valid_rows = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_rows.numel() == 0:
        return weights

    turn_tokens = token_counts[valid_rows].to(torch.float32).clamp_min(1.0)
    weights[valid_rows] = turn_tokens.mean() / turn_tokens
    return weights


def _compute_loss_weights(
    level: str,
    trajectory_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    token_counts: torch.Tensor,
) -> torch.Tensor:
    """Dispatch actor.loss_weighting. 'token' is the identity and never gets here."""
    if level == "episode":
        return _compute_episode_loss_weights(trajectory_ids, valid_mask, token_counts)
    if level == "turn":
        return _compute_turn_loss_weights(valid_mask, token_counts)
    raise ValueError(f"unknown actor.loss_weighting level {level!r}")


def _compute_batch_centered_penalties(
    scores: torch.Tensor,
    weights: torch.Tensor,
    score_valid_mask: torch.Tensor,
    turn_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize scores above the detached mean of valid pairs without std scaling."""

    if not (
        scores.shape == weights.shape == score_valid_mask.shape == turn_valid_mask.shape
    ):
        raise ValueError(
            "Batch-centered penalty tensors must have identical shapes: "
            f"scores={scores.shape}, weights={weights.shape}, "
            f"score_valid_mask={score_valid_mask.shape}, "
            f"turn_valid_mask={turn_valid_mask.shape}."
        )
    detached_scores = scores.detach().float()
    detached_weights = weights.detach().float()
    valid_mask = (
        score_valid_mask.bool()
        & turn_valid_mask.bool()
        & torch.isfinite(detached_scores)
        & (detached_weights > 0.0)
    )
    bounded_scores = detached_scores.clamp(min=-1.0, max=1.0)
    valid_scores = torch.where(
        valid_mask,
        bounded_scores,
        torch.zeros_like(bounded_scores),
    )
    valid_count = valid_mask.sum().clamp_min(1)
    batch_mean = (valid_scores.sum() / valid_count).detach()
    return torch.where(
        valid_mask,
        -detached_weights * torch.relu(bounded_scores - batch_mean),
        torch.zeros_like(detached_scores),
    )


def _compute_teacher_context_advantages(
    real_logps: torch.Tensor,
    real_loss_mask: torch.Tensor,
    moved_logps: torch.Tensor,
    moved_loss_mask: torch.Tensor,
    weights: torch.Tensor,
    score_clips: torch.Tensor,
    score_valid_mask: torch.Tensor,
    turn_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a batch-centered local advantage for context-dependent turns."""

    batch_size = real_logps.shape[0]
    one_dimensional = {
        "weights": weights,
        "score_clips": score_clips,
        "score_valid_mask": score_valid_mask,
        "turn_valid_mask": turn_valid_mask,
    }
    invalid_shapes = {
        name: value.shape
        for name, value in one_dimensional.items()
        if value.shape != (batch_size,)
    }
    if invalid_shapes:
        raise ValueError(
            "Teacher context metadata must have shape [batch_size]; "
            f"batch_size={batch_size}, invalid={invalid_shapes}."
        )
    if real_logps.shape != real_loss_mask.shape:
        raise ValueError(
            "Teacher context real log-probabilities and loss mask must match: "
            f"logps={real_logps.shape}, mask={real_loss_mask.shape}."
        )
    if moved_logps.shape != moved_loss_mask.shape:
        raise ValueError(
            "Teacher context moved log-probabilities and loss mask must match: "
            f"logps={moved_logps.shape}, mask={moved_loss_mask.shape}."
        )
    if moved_logps.shape[0] != batch_size:
        raise ValueError(
            "Teacher context real and moved batches must have the same size: "
            f"real={batch_size}, moved={moved_logps.shape[0]}."
        )

    real_logps = real_logps.detach().float()
    moved_logps = moved_logps.detach().float()
    real_mask = real_loss_mask.detach().bool()
    moved_mask = moved_loss_mask.detach().bool()
    weights = weights.detach().float()
    score_clips = score_clips.detach().float()

    real_counts = real_mask.sum(dim=-1)
    moved_counts = moved_mask.sum(dim=-1)
    real_means = (real_logps * real_mask).sum(dim=-1) / real_counts.clamp_min(1)
    moved_means = (moved_logps * moved_mask).sum(dim=-1) / moved_counts.clamp_min(1)
    information_gain = real_means - moved_means
    valid_mask = (
        score_valid_mask.bool()
        & turn_valid_mask.bool()
        & (real_counts > 0)
        & (moved_counts > 0)
        & torch.isfinite(real_means)
        & torch.isfinite(moved_means)
        & torch.isfinite(information_gain)
        & torch.isfinite(weights)
        & torch.isfinite(score_clips)
        & (weights > 0.0)
        & (score_clips > 0.0)
    )
    clipped_gain = torch.maximum(
        torch.minimum(information_gain, score_clips), -score_clips
    )
    valid_gain = torch.where(valid_mask, clipped_gain, torch.zeros_like(clipped_gain))
    valid_count = valid_mask.sum().clamp_min(1)
    batch_mean = (valid_gain.sum() / valid_count).detach()
    local_advantage = torch.where(
        valid_mask,
        weights * (clipped_gain - batch_mean),
        torch.zeros_like(clipped_gain),
    )
    return local_advantage, real_means, moved_means, information_gain


def _broadcast_turn_values_to_tokens(
    turn_values: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    return turn_values.float().unsqueeze(-1).expand_as(loss_mask) * loss_mask.float()


def _apply_world_model_rl_reweight(
    turn_advantages: torch.Tensor,
    surprise_scores: torch.Tensor,
    score_valid_mask: torch.Tensor,
    turn_valid_mask: torch.Tensor,
    token_counts: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the sign-aware 2x2 World Model gate to normalized turn returns."""

    expected_shape = turn_advantages.shape
    named_tensors = {
        "surprise_scores": surprise_scores,
        "score_valid_mask": score_valid_mask,
        "turn_valid_mask": turn_valid_mask,
        "token_counts": token_counts,
    }
    invalid_shapes = {
        name: value.shape
        for name, value in named_tensors.items()
        if value.shape != expected_shape
    }
    if invalid_shapes:
        raise ValueError(
            "World Model RL reweight tensors must have identical row shapes: "
            f"advantages={expected_shape}, invalid={invalid_shapes}."
        )

    advantages = turn_advantages.float()
    scores = surprise_scores.detach().float().clamp(min=-1.0, max=1.0)
    valid_mask = (
        score_valid_mask.bool()
        & turn_valid_mask.bool()
        & torch.isfinite(scores)
        & torch.isfinite(advantages)
        & (token_counts > 0)
    )
    positive_mask = valid_mask & (advantages > 0.0)
    negative_mask = valid_mask & (advantages < 0.0)
    positive_strength = float(config.get("positive_strength", 1.0))
    negative_strength = float(config.get("negative_strength", 1.0))
    min_weight = float(config.get("min_weight", 0.5))
    max_weight = float(config.get("max_weight", 2.0))

    weights = torch.ones_like(advantages)
    weights = torch.where(
        positive_mask,
        1.0 + positive_strength * scores,
        weights,
    )
    weights = torch.where(
        negative_mask,
        1.0 - negative_strength * scores,
        weights,
    )
    weights = weights.clamp(min=min_weight, max=max_weight)

    # Preserve the effective PPO learning rate separately on positive and negative
    # outcome credit. Token counts match the denominator used after broadcasting.
    token_counts = token_counts.detach().float()
    for sign_mask in (positive_mask, negative_mask):
        weighted_sum = (weights * token_counts * sign_mask).sum()
        denominator = (token_counts * sign_mask).sum()
        if dist.is_initialized():
            dist.all_reduce(weighted_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        mean_weight = weighted_sum / denominator.clamp_min(1.0)
        normalized = weights / mean_weight.clamp_min(1e-6)
        weights = torch.where(sign_mask, normalized, weights)

    weights = weights.clamp(min=min_weight, max=max_weight)
    weights = torch.where(valid_mask, weights, torch.ones_like(weights))
    return advantages * weights, weights


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.behave_imp_weight_cap:
                logger.info(
                    f"  Importance weight cap: {config.behave_imp_weight_cap:.1f} "
                    "(filters out tokens with extreme weights)"
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  advantage_estimator: {config.advantage_estimator}")
        logger.info(f"  use_ppo_clip: {config.use_ppo_clip}")
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info("=" * 70)

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor] | None:
        return batched_call(self._compute_logp, data)

    @trace_perf("ppo_actor.compute_world_model_logp", category="compute")
    @torch.no_grad()
    def compute_world_model_logp(
        self, data: list[dict[str, Any]]
    ) -> list[torch.Tensor] | None:
        self.engine.activate_world_model_adapter()
        return batched_call(self._compute_logp, data)

    def activate_policy_adapter(self) -> None:
        self.engine.activate_policy_adapter()

    def _compute_logp(self, data: dict[str, Any]) -> torch.Tensor | None:
        self.engine.eval()
        return self.engine.forward(
            input_=data,
            aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
        )

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(
        self,
        data: list[dict[str, Any]],
        world_model_rl_reweight_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        compute_fn = functools.partial(
            self._compute_advantages,
            world_model_rl_reweight_config=world_model_rl_reweight_config,
        )
        return batched_call(compute_fn, data)

    def _compute_advantages(
        self,
        data: dict[str, Any],
        world_model_rl_reweight_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        batch_penalty_keys = {
            "batch_centered_penalty_score",
            "batch_centered_penalty_weight",
            "batch_centered_penalty_valid",
        }
        present_batch_penalty_keys = batch_penalty_keys.intersection(data)
        if (
            present_batch_penalty_keys
            and present_batch_penalty_keys != batch_penalty_keys
        ):
            missing = sorted(batch_penalty_keys - present_batch_penalty_keys)
            raise ValueError(
                "Incomplete batch-centered penalty metadata; missing: "
                + ", ".join(missing)
            )
        if present_batch_penalty_keys and self.config.advantage_estimator != "rebn":
            raise ValueError(
                "Batch-centered local penalties require advantage_estimator='rebn'."
            )
        gate_credit_keys = {"gate_masked_rewards", "gate_credit_mask"}
        present_gate_credit_keys = gate_credit_keys.intersection(data)
        if present_gate_credit_keys and present_gate_credit_keys != gate_credit_keys:
            missing = sorted(gate_credit_keys - present_gate_credit_keys)
            raise ValueError(
                "Incomplete gate-pass credit metadata; missing: " + ", ".join(missing)
            )
        if present_gate_credit_keys and self.config.advantage_estimator != "rebn":
            raise ValueError(
                "Gate-pass return credit requires advantage_estimator='rebn'."
            )
        if (
            "personality_gate_fail_penalty" in data
            and self.config.advantage_estimator != "rebn"
        ):
            raise ValueError(
                "The personality-gate fail penalty requires advantage_estimator='rebn'."
            )
        local_reward_placement_keys = {
            "local_rewards_group_norm",
            "local_rewards_pre_std",
            "local_rewards_post_std",
        }
        present_local_reward_placement_keys = local_reward_placement_keys.intersection(
            data
        )
        if (
            present_local_reward_placement_keys
            and present_local_reward_placement_keys != local_reward_placement_keys
        ):
            missing = sorted(
                local_reward_placement_keys - present_local_reward_placement_keys
            )
            raise ValueError(
                "Incomplete turn-local reward placement metadata; missing: "
                + ", ".join(missing)
            )
        if present_local_reward_placement_keys and "local_rewards" not in data:
            raise ValueError(
                "Turn-local reward placement metadata requires the "
                "'local_rewards' total."
            )
        opd_keys = {
            "opd_teacher_logp",
            "opd_loss_mask",
            "opd_token_weight",
            "opd_reward_clip",
        }
        present_opd_keys = opd_keys.intersection(data)
        if present_opd_keys and present_opd_keys != opd_keys:
            raise ValueError(
                "Incomplete on-policy-distillation metadata; missing: "
                + ", ".join(sorted(opd_keys - present_opd_keys))
                + ". opd_teacher_logp is attached by the trainer's OPD forward "
                "pass; the other two come from the rollout."
            )
        teacher_context_keys = {
            "teacher_context_input_ids",
            "teacher_context_attention_mask",
            "teacher_context_loss_mask",
            "teacher_context_logp",
            "teacher_context_reward_weight",
            "teacher_context_reward_score_clip",
            "teacher_context_reward_apply_to_advantage",
            "teacher_context_reward_valid",
        }
        present_teacher_context_keys = teacher_context_keys.intersection(data)
        if (
            present_teacher_context_keys
            and present_teacher_context_keys != teacher_context_keys
        ):
            missing = sorted(teacher_context_keys - present_teacher_context_keys)
            raise ValueError(
                "Incomplete teacher context reward metadata; missing: "
                + ", ".join(missing)
            )
        if present_teacher_context_keys and self.config.advantage_estimator != "rebn":
            raise ValueError(
                "Teacher context local advantages require advantage_estimator='rebn'."
            )
        world_model_rl_reweight_enabled = bool(
            world_model_rl_reweight_config
            and world_model_rl_reweight_config.get("enabled", False)
        )
        world_model_rl_keys = {
            "world_model_rl_nll",
            "world_model_rl_surprise",
            "world_model_rl_reweight_valid",
        }
        present_world_model_rl_keys = world_model_rl_keys.intersection(data)
        if world_model_rl_reweight_enabled:
            if present_world_model_rl_keys != world_model_rl_keys:
                missing = sorted(world_model_rl_keys - present_world_model_rl_keys)
                raise ValueError(
                    "Incomplete World Model RL reweight metadata; missing: "
                    + ", ".join(missing)
                )
            if self.config.advantage_estimator != "rebn":
                raise ValueError(
                    "World Model RL reweighting requires advantage_estimator='rebn'."
                )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)
        # Turn-local reward components. ReBN accumulates turn rewards backward,
        # so without this a penalty raised on the last turn (a leak, say) lands
        # undiscounted in the returns of every earlier turn -- charging good
        # teaching for a mistake it did not make. The workflow reports those
        # components in a separate local_rewards column; we subtract them
        # before accumulation and add them back after, so only the offending
        # turn carries them. Absent column => unchanged behaviour.
        local_reward_score = None
        local_reward_scores_by_placement = None
        post_std_local_reward_advantage = None
        if "local_rewards" in data:
            if self.config.advantage_estimator != "rebn":
                # Only ReBN accumulates rewards across turns, so only ReBN can
                # hold a component back from that accumulation. Fail loudly
                # rather than silently propagating a component the config asked
                # to keep local.
                raise ValueError(
                    "A 'local_rewards' column requires "
                    "actor.advantage_estimator='rebn', got "
                    f"{self.config.advantage_estimator!r}. Clear "
                    "reward.turn_local_components or switch to ReBN."
                )
            local_reward_score = data["local_rewards"].to(reward_score.device)
            # Same scaling and clipping as the total, so the subtraction below
            # stays exact. reward_bias is deliberately not applied twice: it is
            # a shift on the propagating return, not on each component.
            local_reward_score = local_reward_score * self.reward_scaling
            local_reward_score = torch.clip(
                local_reward_score, max=self.reward_clip, min=-self.reward_clip
            )
            if present_local_reward_placement_keys:
                local_reward_scores_by_placement = {}
                for placement in ("group_norm", "pre_std"):
                    placement_score = data[f"local_rewards_{placement}"].to(
                        device=reward_score.device, dtype=reward_score.dtype
                    )
                    placement_score = placement_score * self.reward_scaling
                    local_reward_scores_by_placement[placement] = torch.clip(
                        placement_score,
                        max=self.reward_clip,
                        min=-self.reward_clip,
                    )
                # A post-std component is already expressed in normalized-
                # advantage units, like personality_gate_fail_penalty. It must
                # not inherit reward scaling, clipping, or the batch std.
                post_std_local_reward_advantage = data["local_rewards_post_std"].to(
                    device=reward_score.device, dtype=reward_score.dtype
                )
        gate_masked_reward_score = None
        if present_gate_credit_keys:
            if self.reward_norm is not None:
                raise ValueError(
                    "Gate-pass return credit is incompatible with reward_norm: "
                    "normalizing the total reward would prevent exact component "
                    "subtraction."
                )
            gate_masked_reward_score = data["gate_masked_rewards"].to(
                reward_score.device
            )
            # Match the total reward's linear scaling without applying its bias a
            # second time. The bias remains in the unmasked residual component.
            gate_masked_reward_score = gate_masked_reward_score * self.reward_scaling
            gate_masked_reward_score = torch.clip(
                gate_masked_reward_score,
                max=self.reward_clip,
                min=-self.reward_clip,
            )

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)
        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        if self.config.advantage_estimator == "rebn":
            if "trajectory_id" not in data or "turn_idx" not in data:
                raise ValueError(
                    "advantage_estimator='rebn' requires trajectory_id and turn_idx "
                    "metadata in rollout data."
                )
            kl_only_rewards = kl_rewards.clone()
            kl_only_rewards[batch_indices, seqlens - 1] = 0
            values = torch.zeros_like(kl_only_rewards)
            kl_advantages = self._compute_token_gae(
                rewards=kl_only_rewards,
                values=values,
                loss_mask=loss_mask,
                seq_no_eos_mask=seq_no_eos_mask,
            )
            valid_turn_mask = loss_mask.sum(dim=-1) > 0
            personality_gate_fail_penalty = None
            if "personality_gate_fail_penalty" in data:
                personality_gate_fail_penalty = data[
                    "personality_gate_fail_penalty"
                ].to(device=reward_score.device, dtype=reward_score.dtype)
                personality_gate_fail_penalty = (
                    personality_gate_fail_penalty
                    * valid_turn_mask.to(personality_gate_fail_penalty.dtype)
                )
            batch_centered_penalties = None
            if present_batch_penalty_keys:
                batch_centered_penalties = _compute_batch_centered_penalties(
                    data["batch_centered_penalty_score"].to(reward_score.device),
                    data["batch_centered_penalty_weight"].to(reward_score.device),
                    data["batch_centered_penalty_valid"].to(reward_score.device),
                    valid_turn_mask,
                )
            teacher_context_advantages = None
            if present_teacher_context_keys:
                prox_logp = data.get("prox_logp")
                if prox_logp is None:
                    raise ValueError(
                        "Teacher context reward requires prox_logp from the same "
                        "actor snapshot as the moved-context forward pass."
                    )
                moved_loss_mask = torch.roll(
                    data["teacher_context_loss_mask"].float(), shifts=-1, dims=-1
                )
                (
                    teacher_context_advantages,
                    real_avg_logp,
                    moved_avg_logp,
                    information_gain,
                ) = _compute_teacher_context_advantages(
                    prox_logp,
                    loss_mask,
                    data["teacher_context_logp"],
                    moved_loss_mask,
                    data["teacher_context_reward_weight"].to(reward_score.device),
                    data["teacher_context_reward_score_clip"].to(reward_score.device),
                    data["teacher_context_reward_valid"].to(reward_score.device),
                    valid_turn_mask,
                )
                data["teacher_context_real_avg_logp"] = real_avg_logp
                data["teacher_context_moved_avg_logp"] = moved_avg_logp
                data["teacher_context_information_gain"] = information_gain
            propagating_reward_score = (
                reward_score
                if local_reward_score is None
                else reward_score - local_reward_score
            )
            trajectory_ids = data["trajectory_id"].to(reward_score.device)
            turn_indices = data["turn_idx"].to(reward_score.device)
            if gate_masked_reward_score is None:
                turn_returns = _compute_rebn_returns(
                    propagating_reward_score,
                    trajectory_ids,
                    turn_indices,
                    self.config.turn_discount,
                    valid_mask=valid_turn_mask,
                )
            else:
                residual_returns = _compute_rebn_returns(
                    propagating_reward_score - gate_masked_reward_score,
                    trajectory_ids,
                    turn_indices,
                    self.config.turn_discount,
                    valid_mask=valid_turn_mask,
                )
                masked_returns = _compute_rebn_returns(
                    gate_masked_reward_score,
                    trajectory_ids,
                    turn_indices,
                    self.config.turn_discount,
                    valid_mask=valid_turn_mask,
                )
                credit_mask = data["gate_credit_mask"].to(
                    device=reward_score.device, dtype=masked_returns.dtype
                )
                turn_returns = residual_returns + masked_returns * credit_mask
            baseline_source = turn_returns
            if local_reward_score is not None:
                valid_turn_weights = valid_turn_mask.to(turn_returns.dtype)
                if local_reward_scores_by_placement is not None:
                    group_local_masked = (
                        local_reward_scores_by_placement["group_norm"]
                        * valid_turn_weights
                    )
                    pre_std_local_masked = (
                        local_reward_scores_by_placement["pre_std"] * valid_turn_weights
                    )
                    # Both are turn-local. Only group_norm enters the first
                    # (group-baseline) normalization; both enter the later std.
                    turn_returns = (
                        turn_returns + group_local_masked + pre_std_local_masked
                    )
                    if self.config.group_baseline == "episode":
                        # The episode baseline reads the first turn, so gather all
                        # group_norm components there for the baseline only.
                        baseline_source = (
                            baseline_source
                            + _episode_local_at_first_turn(
                                group_local_masked,
                                data["trajectory_id"].to(reward_score.device),
                                data["turn_idx"].to(reward_score.device),
                                valid_turn_mask,
                            )
                        )
                    else:
                        # A turn baseline compares group_norm components only with
                        # peers at the same depth. pre_std remains outside it.
                        baseline_source = baseline_source + group_local_masked
                else:
                    # Legacy batches carry only the aggregate local_rewards column.
                    # Preserve their actor-wide include/exclude behavior exactly.
                    local_masked = local_reward_score * valid_turn_weights
                    turn_returns = turn_returns + local_masked
                    local_baseline_mode = self.config.group_baseline_local_reward_mode
                    if local_baseline_mode == "include":
                        if self.config.group_baseline == "episode":
                            baseline_source = (
                                baseline_source
                                + _episode_local_at_first_turn(
                                    local_masked,
                                    data["trajectory_id"].to(reward_score.device),
                                    data["turn_idx"].to(reward_score.device),
                                    valid_turn_mask,
                                )
                            )
                        else:
                            baseline_source = turn_returns
            if self.config.group_baseline is not None:
                if "group_id" not in data:
                    raise ValueError(
                        f"actor.group_baseline={self.config.group_baseline!r} "
                        "requires a 'group_id' "
                        "column in rollout data. It is attached by the trainer "
                        "before compute_advantages; check that rollout groups "
                        "survive to that point."
                    )
                baseline_fn = (
                    _compute_episode_group_baseline
                    if self.config.group_baseline == "episode"
                    else _compute_turn_group_baseline
                )
                group_baseline = baseline_fn(
                    baseline_source,
                    data["trajectory_id"].to(reward_score.device),
                    data["turn_idx"].to(reward_score.device),
                    data["group_id"].to(reward_score.device),
                    valid_turn_mask,
                    self.config.group_baseline_leave1out,
                )
                turn_returns = turn_returns - group_baseline
                data["group_baseline"] = group_baseline
            if self.adv_norm is not None and valid_turn_mask.any():
                normalized_turn_returns = torch.zeros_like(turn_returns)
                normalized_turn_returns[valid_turn_mask] = self.adv_norm(
                    turn_returns[valid_turn_mask]
                )
            else:
                normalized_turn_returns = turn_returns
            if world_model_rl_reweight_enabled:
                data["world_model_rl_nll"] = data["world_model_rl_nll"].to(
                    reward_score.device
                )
                data["world_model_rl_surprise"] = data["world_model_rl_surprise"].to(
                    reward_score.device
                )
                data["world_model_rl_reweight_valid"] = data[
                    "world_model_rl_reweight_valid"
                ].to(reward_score.device)
                turn_advantage_before_reweight = normalized_turn_returns.clone()
                (
                    normalized_turn_returns,
                    world_model_rl_weights,
                ) = _apply_world_model_rl_reweight(
                    normalized_turn_returns,
                    data["world_model_rl_surprise"],
                    data["world_model_rl_reweight_valid"],
                    valid_turn_mask,
                    loss_mask.sum(dim=-1),
                    world_model_rl_reweight_config,
                )
                data["world_model_rl_advantage_before_reweight"] = (
                    turn_advantage_before_reweight
                )
                data["world_model_rl_advantage_after_reweight"] = (
                    normalized_turn_returns
                )
                data["world_model_rl_weight"] = world_model_rl_weights
            if self.config.loss_weighting != "token":
                loss_weights = _compute_loss_weights(
                    self.config.loss_weighting,
                    data["trajectory_id"].to(reward_score.device),
                    valid_turn_mask,
                    loss_mask.sum(dim=-1),
                )
                normalized_turn_returns = normalized_turn_returns * loss_weights
                # The column keeps its name across all levels so the series, the
                # drop list below and the tests reading it stay put; it carries
                # whichever level actor.loss_weighting names.
                data["episode_loss_weight"] = loss_weights
            # Per-component post_std local rewards are fixed advantage-unit
            # penalties. Add them at the same final layer as the personality gate
            # penalty so rare leak/format events cannot set the batch std or have
            # their magnitude divided by it.
            if post_std_local_reward_advantage is not None:
                post_std_local_reward_advantage = (
                    post_std_local_reward_advantage
                    * valid_turn_mask.to(post_std_local_reward_advantage.dtype)
                )
                normalized_turn_returns = (
                    normalized_turn_returns + post_std_local_reward_advantage
                )
                data["local_reward_post_std_advantage"] = (
                    post_std_local_reward_advantage
                )
            # This coefficient is already in normalized-advantage units. Adding it
            # here keeps a configured small gate penalty small even when outcome
            # variance is near zero; putting it into rewards before adv_norm would
            # divide away its magnitude. It is one scalar on the rejected turn and
            # is never accumulated backward through the trajectory.
            if personality_gate_fail_penalty is not None:
                normalized_turn_returns = (
                    normalized_turn_returns + personality_gate_fail_penalty
                )
                data["personality_gate_fail_advantage"] = personality_gate_fail_penalty
            if batch_centered_penalties is not None:
                normalized_turn_returns = (
                    normalized_turn_returns + batch_centered_penalties
                )
                data["batch_centered_penalty_advantage"] = batch_centered_penalties
            if teacher_context_advantages is not None:
                applied_teacher_context_advantages = (
                    teacher_context_advantages
                    * data["teacher_context_reward_apply_to_advantage"]
                    .to(reward_score.device)
                    .bool()
                )
                normalized_turn_returns = (
                    normalized_turn_returns + applied_teacher_context_advantages
                )
                data["teacher_context_advantage"] = teacher_context_advantages
            data["turn_advantage"] = normalized_turn_returns
            advantages = kl_advantages + _broadcast_turn_values_to_tokens(
                normalized_turn_returns, loss_mask
            )
            data["returns"] = advantages
        else:
            if "values" not in data:
                values = torch.zeros_like(rewards)
            else:
                values = data["values"]
            advantages = self._compute_token_gae(
                rewards=rewards,
                values=values,
                loss_mask=loss_mask,
                seq_no_eos_mask=seq_no_eos_mask,
            )
            data["returns"] = advantages + values

            # Optionally perform advantage normalization.
            if self.adv_norm is not None:
                advantages = self.adv_norm(advantages, loss_mask)

        # On-policy distillation, added after the task advantage is complete and
        # after normalization -- the reference adds it to already-computed
        # advantages, and keeping it out of adv_norm is what preserves its scale
        # as a KL in nats. It is deliberately not added to `returns`, which stays
        # the task signal.
        opd_teacher_logp = data.get("opd_teacher_logp")
        if opd_teacher_logp is not None:
            # Realigned here rather than in the trainer: compute_logp may hand
            # back RTensor handles, which only become real tensors once the batch
            # reaches this side.
            opd_teacher_logp = _realign_opd_teacher_logp(
                opd_teacher_logp,
                data["opd_loss_mask"],
                loss_mask,
                old_logp,
            )
            opd_advantages, opd_reverse_kl, opd_active = _compute_opd_advantages(
                old_logp,
                opd_teacher_logp,
                data["opd_token_weight"].to(old_logp.dtype),
                data["opd_reward_clip"].to(old_logp.dtype),
                loss_mask,
            )
            advantages = advantages + opd_advantages
            data["opd_reverse_kl"] = opd_reverse_kl
            data["opd_advantage"] = opd_advantages
            data["opd_active"] = opd_active

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        # because we have rolled old_logp by -1
        data["logprobs"] = old_logp

        return data

    def _compute_token_gae(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        loss_mask: torch.Tensor,
        seq_no_eos_mask: torch.Tensor,
    ) -> torch.Tensor:
        bs = rewards.shape[0]
        max_seqlen = rewards.shape[1]
        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask
        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]
            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            # Skip tokens that do not contribute to the loss
            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)
        return torch.stack(advantages_reversed[::-1], dim=1)

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(
        self,
        data: list[dict[str, Any]],
        world_model_batch: list[dict[str, Any]] | None = None,
    ) -> None:
        if world_model_batch is None:
            batched_call(self._ppo_update, data, unpack=False)
            return
        if len(world_model_batch) != len(data):
            raise ValueError(
                "World Model/PPO trajectory count mismatch: "
                f"world_model={len(world_model_batch)}, ppo={len(data)}."
            )
        loss_weights = {
            float(sidecar["world_model_loss_weight"]) for sidecar in world_model_batch
        }
        if len(loss_weights) != 1:
            raise ValueError(
                "World Model loss weight must be identical across a training batch."
            )
        batched_data, _ = concat_batch(data)
        batched_world_model = concat_padded_tensors(world_model_batch)
        batched_world_model["world_model_loss_weight"] = loss_weights.pop()
        self._ppo_update(batched_data, world_model_batch=batched_world_model)

    @trace_perf("ppo_actor.world_model_update", category="compute")
    @stats_tracker.scope_func_wrapper("world_model_actor")
    def world_model_update(
        self,
        world_model_batch: list[dict[str, Any]],
    ) -> bool:
        if not world_model_batch:
            return False
        loss_weights = {
            float(sidecar["world_model_loss_weight"]) for sidecar in world_model_batch
        }
        if len(loss_weights) != 1:
            raise ValueError(
                "World Model loss weight must be identical across a training batch."
            )
        batched_world_model = concat_padded_tensors(world_model_batch)
        rows = _unpack_world_model_rows(
            batched_world_model,
            expected_rows=batched_world_model["world_model_seq_lens"].numel(),
        )
        selected_rows = [
            row for row in rows if row[2] and row[0].numel() > 0 and bool(row[1].any())
        ]

        local_count = torch.tensor([len(selected_rows)], dtype=torch.long)
        if dist.is_initialized():
            gathered_counts = [
                torch.zeros_like(local_count)
                for _ in range(dist.get_world_size(group=self.engine.cpu_group))
            ]
            dist.all_gather(
                gathered_counts,
                local_count,
                group=self.engine.cpu_group,
            )
            data_parallel_ranks = dist.get_process_group_ranks(
                self.engine.data_parallel_group
            )
            counts = [
                int(gathered_counts[global_rank]) for global_rank in data_parallel_ranks
            ]
            rank = self.engine.data_parallel_rank
        else:
            counts = [len(selected_rows)]
            rank = 0
        total_selected = sum(counts)
        if total_selected == 0:
            return False

        update_count = min(self.config.ppo_n_minibatches, total_selected)
        prefix = sum(counts[:rank])
        update_rows: list[list[tuple[torch.Tensor, torch.Tensor, bool, float]]] = [
            [] for _ in range(update_count)
        ]
        for local_index, row in enumerate(selected_rows):
            update_rows[(prefix + local_index) % update_count].append(row)

        paw_config = dict(batched_world_model.get("world_model_paw_config") or {})
        loss_fn = functools.partial(
            world_model_loss_fn,
            world_model_loss_weight=loss_weights.pop(),
            paw_enabled=bool(paw_config.get("enabled", False)),
            cmae_enabled=bool(paw_config.get("cmae_enabled", True)),
            confidence_threshold=float(paw_config.get("confidence_threshold", 0.2)),
        )
        self.engine.activate_world_model_adapter()
        self.engine.train()
        with stats_tracker.scope("update"):
            for rows_for_update in update_rows:
                train_stat = self.engine.train_batch(
                    _build_world_model_train_batch(
                        rows_for_update,
                        device=self.engine.device,
                    ),
                    loss_fn=loss_fn,
                    loss_weight_fn=_world_model_loss_weight,
                )
                stats_tracker.scalar(**train_stat)
        self.engine.lr_scheduler_step()
        return True

    def _ppo_update(
        self,
        data: dict[str, Any],
        world_model_batch: dict[str, Any] | None = None,
    ) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if "batch_centered_penalty_valid" in data:
            result_denominators["batch_centered_penalty_valid"] = data[
                "batch_centered_penalty_valid"
            ].bool()
        if "teacher_context_reward_valid" in data:
            result_denominators["teacher_context_reward_valid"] = data[
                "teacher_context_reward_valid"
            ].bool()
        if "world_model_rl_reweight_valid" in data:
            world_model_rl_valid = data["world_model_rl_reweight_valid"].bool()
            world_model_rl_advantage = data[
                "world_model_rl_advantage_before_reweight"
            ].float()
            world_model_rl_surprise = data["world_model_rl_surprise"].float()
            result_denominators.update(
                world_model_rl_reweight_valid=world_model_rl_valid,
                world_model_rl_positive_predictable=(
                    world_model_rl_valid
                    & (world_model_rl_advantage > 0.0)
                    & (world_model_rl_surprise < 0.0)
                ),
                world_model_rl_positive_surprising=(
                    world_model_rl_valid
                    & (world_model_rl_advantage > 0.0)
                    & (world_model_rl_surprise > 0.0)
                ),
                world_model_rl_negative_predictable=(
                    world_model_rl_valid
                    & (world_model_rl_advantage < 0.0)
                    & (world_model_rl_surprise < 0.0)
                ),
                world_model_rl_negative_surprising=(
                    world_model_rl_valid
                    & (world_model_rl_advantage < 0.0)
                    & (world_model_rl_surprise > 0.0)
                ),
            )
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=infer_token_denominator(data, loss_mask),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )
        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )
        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        if "group_baseline" in data:
            # turn_advantage is exactly 0 for a group whose episodes all scored the
            # same (and for a group reduced to a single surviving episode), so this
            # is the share of rows that contributed no gradient.
            stats_tracker.stat(
                group_baseline=data["group_baseline"].float(),
                zero_advantage_turns=(data["turn_advantage"].abs() < 1e-8).float(),
                denominator="n_seqs",
            )
        if "episode_loss_weight" in data:
            stats_tracker.stat(
                episode_loss_weight=data["episode_loss_weight"].float(),
                denominator="n_seqs",
            )
        if "personality_gate_fail_advantage" in data:
            stats_tracker.stat(
                personality_gate_fail_advantage=data[
                    "personality_gate_fail_advantage"
                ].float(),
                denominator="n_seqs",
            )
        if "local_reward_post_std_advantage" in data:
            stats_tracker.stat(
                local_reward_post_std_advantage=data[
                    "local_reward_post_std_advantage"
                ].float(),
                denominator="n_seqs",
            )
        if "opd_reverse_kl" in data:
            # Averaged over supervised tokens only, so the numbers are not diluted
            # by the turns OPD skipped. opd_reverse_kl is the per-token reverse KL
            # in nats, defined as log pi_sampled - log pi_teacher. It goes
            # NEGATIVE when the instructed teacher assigns the token more mass
            # than the sampler did, i.e. when there is something left to distil,
            # and the advantage is its negation. It should rise toward 0 as the
            # instruction is absorbed; sitting flat and far from 0 means it is not
            # being absorbed and loss_weight is the knob.
            stats_tracker.denominator(opd_tokens=data["opd_active"].bool())
            stats_tracker.stat(
                opd_reverse_kl=data["opd_reverse_kl"].float(),
                opd_advantage=data["opd_advantage"].float(),
                denominator="opd_tokens",
            )
        if "batch_centered_penalty_advantage" in data:
            stats_tracker.stat(
                batch_centered_penalty_score=torch.nan_to_num(
                    data["batch_centered_penalty_score"].float()
                ),
                batch_centered_penalty_advantage=data[
                    "batch_centered_penalty_advantage"
                ].float(),
                denominator="batch_centered_penalty_valid",
            )
        if "teacher_context_advantage" in data:
            stats_tracker.stat(
                teacher_context_real_avg_logp=data[
                    "teacher_context_real_avg_logp"
                ].float(),
                teacher_context_moved_avg_logp=data[
                    "teacher_context_moved_avg_logp"
                ].float(),
                teacher_context_information_gain=data[
                    "teacher_context_information_gain"
                ].float(),
                teacher_context_advantage=data["teacher_context_advantage"].float(),
                denominator="teacher_context_reward_valid",
            )
        if "world_model_rl_weight" in data:
            stats_tracker.stat(
                world_model_rl_nll=data["world_model_rl_nll"].float(),
                world_model_rl_surprise=data["world_model_rl_surprise"].float(),
                world_model_rl_weight=data["world_model_rl_weight"].float(),
                world_model_rl_advantage_before=data[
                    "world_model_rl_advantage_before_reweight"
                ].float(),
                world_model_rl_advantage_after=data[
                    "world_model_rl_advantage_after_reweight"
                ].float(),
                denominator="world_model_rl_reweight_valid",
            )
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behave_imp_weight_cap is not None:
            scalars["behave_imp_weight_cap"] = self.config.behave_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in [
            "rewards",
            "tot_rewards",
            "kl_rewards",
            "gate_masked_rewards",
            "gate_credit_mask",
            "local_rewards",
            "local_rewards_group_norm",
            "local_rewards_pre_std",
            "local_rewards_post_std",
            "local_reward_post_std_advantage",
            "personality_gate_fail_penalty",
            "personality_gate_fail_advantage",
            "trajectory_id",
            "turn_idx",
            "batch_centered_penalty_score",
            "batch_centered_penalty_weight",
            "batch_centered_penalty_valid",
            "batch_centered_penalty_advantage",
            "teacher_context_input_ids",
            "teacher_context_attention_mask",
            "teacher_context_loss_mask",
            "teacher_context_logp",
            "teacher_context_reward_weight",
            "teacher_context_reward_score_clip",
            "teacher_context_reward_apply_to_advantage",
            "teacher_context_reward_valid",
            "teacher_context_real_avg_logp",
            "teacher_context_moved_avg_logp",
            "teacher_context_information_gain",
            "teacher_context_advantage",
            "world_model_rl_nll",
            "world_model_rl_surprise",
            "world_model_rl_reweight_valid",
            "world_model_rl_weight",
            "world_model_rl_advantage_before_reweight",
            "world_model_rl_advantage_after_reweight",
            "turn_advantage",
            "group_id",
            "group_baseline",
            "episode_loss_weight",
            # Every OPD column is consumed while computing advantages; none of it
            # reaches the loss, which sees only the adjusted advantage.
            "opd_teacher_logp",
            "opd_loss_mask",
            "opd_token_weight",
            "opd_reward_clip",
            "opd_valid",
            "opd_reverse_kl",
            "opd_advantage",
            "opd_active",
        ]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        if world_model_batch is not None and getattr(
            self.engine, "enable_tree_training", False
        ):
            raise ValueError(
                "World Model auxiliary training does not support tree training."
            )
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )
        world_model_rows = None
        world_model_paw_config: dict[str, Any] = {}
        if world_model_batch is not None:
            world_model_paw_config = dict(
                world_model_batch.get("world_model_paw_config") or {}
            )
            world_model_rows = _unpack_world_model_rows(
                world_model_batch, expected_rows=data["attention_mask"].shape[0]
            )
            if mb_inputs.forward_indices is None:
                raise RuntimeError(
                    "PPO minibatch split did not return row indices for World Model."
                )
        world_model_row_cursor = 0

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                loss_fn = functools.partial(
                    grpo_loss_fn,
                    eps_clip=self.config.eps_clip,
                    eps_clip_higher=self.config.eps_clip_higher,
                    c_clip=self.config.c_clip,
                    use_ppo_clip=self.config.use_ppo_clip,
                    behave_imp_weight_cap=self.config.behave_imp_weight_cap,
                    m2_threshold=self.m2_threshold,
                    importance_sampling_level=self.config.importance_sampling_level,
                    current_version=current_version,
                    prox_logp_method=self.config.prox_logp_method,
                    use_sapo_loss=self.config.use_sapo_loss,
                    sapo_tau_pos=self.config.sapo_tau_pos,
                    sapo_tau_neg=self.config.sapo_tau_neg,
                    use_decoupled_loss=self.config.use_decoupled_loss,
                    behave_imp_weight_mode=self.config.behave_imp_weight_mode,
                )
                loss_weight_fn = _policy_loss_weight
                train_batch = mb
                if world_model_rows is not None:
                    minibatch_rows = mb["attention_mask"].shape[0]
                    row_indices = mb_inputs.forward_indices[
                        world_model_row_cursor : world_model_row_cursor + minibatch_rows
                    ]
                    world_model_row_cursor += minibatch_rows
                    train_batch = _append_world_model_rows(
                        mb,
                        [world_model_rows[index] for index in row_indices],
                        policy_temperature=self.config.temperature,
                    )
                    counts = _global_joint_counts(
                        train_batch,
                        device=self.engine.device,
                        group=self.engine.data_parallel_group,
                    )
                    loss_fn = functools.partial(
                        joint_grpo_loss_fn,
                        ppo_loss_fn=loss_fn,
                        global_policy_tokens=counts[0],
                        global_world_model_responses=counts[1],
                        world_model_loss_weight=float(
                            world_model_batch["world_model_loss_weight"]
                        ),
                        paw_enabled=bool(world_model_paw_config.get("enabled", False)),
                        cmae_enabled=bool(
                            world_model_paw_config.get("cmae_enabled", True)
                        ),
                        confidence_threshold=float(
                            world_model_paw_config.get("confidence_threshold", 0.2)
                        ),
                    )
                    loss_weight_fn = _joint_loss_weight
                train_stat = self.engine.train_batch(
                    train_batch,
                    loss_fn=loss_fn,
                    loss_weight_fn=loss_weight_fn,
                )
                stats_tracker.scalar(**train_stat)


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def compute_base_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_base_logp", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def compute_world_model_logp(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_world_model_logp",
            *args,
            rpc_meta={"broadcast": True},
            **kwargs,
        )

    def world_model_update(self, *args, **kwargs) -> bool:
        return self._custom_function_call(
            "world_model_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def activate_policy_adapter(self) -> None:
        self._custom_function_call("activate_policy_adapter")

    def compute_advantages(self, *args, **kwargs):
        return self._custom_function_call(
            "compute_advantages", *args, rpc_meta={"broadcast": True}, **kwargs
        )

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call(
            "ppo_update", *args, rpc_meta={"broadcast": True}, **kwargs
        )


class PPOActorControllerV2(GatewayTrainController):
    def compute_logp(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_logp", payload)

    def compute_advantages(self, *args, **kwargs):
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        return self._gateway_post_result("/ppo/actor/compute_advantages", payload)

    def ppo_update(self, *args, **kwargs) -> None:
        payload = {
            "args": serialize_value(list(args)),
            "kwargs": serialize_value(kwargs),
        }
        self._gateway_post("/ppo/actor/update", payload)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behave_imp_weight_cap: float | None,
    use_ppo_clip: bool = True,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    behave_imp_weight_mode: str = "token_mask",
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # Use SAPO or PPO loss
    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            use_ppo_clip=use_ppo_clip,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            behave_imp_weight_cap=behave_imp_weight_cap,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            behave_imp_weight_mode=behave_imp_weight_mode,
        )

    # Joint Distillation KL Loss
    teacher_logp = input_data.get("teacher_logp")
    rkl_stat = None
    if teacher_logp is not None:
        # Coefficients for RL and Knowledge Distillation
        rl_loss_weight = input_data.get("rl_loss_weight", 1.0)
        distill_loss_weight = input_data.get("distill_loss_weight", 0.005)

        teacher_logp = (
            teacher_logp.detach()
        )  # detach to prevent gradient backprop to teacher

        if rl_loss_weight == 0:
            # Pure KD using reverse KL (importance-sampling)
            rkl_reward = teacher_logp - logprobs.detach()
            importance_weight = torch.exp(logprobs - old_logp)

            rkl_weighted_term = importance_weight * rkl_reward * loss_mask

            kd_coef = -1 * distill_loss_weight
            loss = kd_coef * rkl_weighted_term.sum() / loss_mask.sum().clamp(min=1)

            rkl_stat = -1 * rkl_weighted_term
        else:
            # KDRL: Knowledge Distillation + Reinforcement Learning (joint loss)
            rkl_penalty_per_token = (logprobs - teacher_logp) * loss_mask
            rkl_penalty = rkl_penalty_per_token.sum() / loss_mask.sum().clamp(min=1)

            loss = rl_loss_weight * loss + distill_loss_weight * rkl_penalty

            rkl_stat = rkl_penalty_per_token

    # Log training statistics
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    if rkl_stat is not None:
        stats_tracker.stat(
            rkl_loss=rkl_stat,
            denominator="n_valid_tokens",
        )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
        )

    return loss


def joint_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict[str, Any],
    *,
    ppo_loss_fn: Any,
    global_policy_tokens: torch.Tensor,
    global_world_model_responses: torch.Tensor,
    world_model_loss_weight: float,
    paw_enabled: bool = False,
    cmae_enabled: bool = False,
    confidence_threshold: float = 0.2,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine PPO with the configured Student-prediction auxiliary objective."""

    policy_loss = ppo_loss_fn(
        logprobs,
        entropy,
        input_data,
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
    )
    return _merge_policy_world_model_loss(
        policy_loss,
        logprobs,
        input_data,
        global_policy_tokens=global_policy_tokens,
        global_world_model_responses=global_world_model_responses,
        world_model_loss_weight=world_model_loss_weight,
        paw_enabled=paw_enabled,
        cmae_enabled=cmae_enabled,
        confidence_threshold=confidence_threshold,
    )


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )

        if logprobs is not None:
            # Log KL divergence estimators to check for policy drift between the
            # training-time policy (logprobs) and the inference-time policy (old_logp).
            log_ratio = (logprobs.float() - old_logp.float()).detach()

            # Implementation of different estimators for KL divergence.
            # See: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl
            kl_div_estimator_direct = -log_ratio
            kl_div_estimator_taylor = log_ratio**2 / 2.0
            kl_div_estimator_dual = log_ratio.exp() - 1 - log_ratio

            # Register these to TensorBoard
            stats_tracker.stat(
                kl_div_direct=kl_div_estimator_direct,
                kl_div_taylor=kl_div_estimator_taylor,
                kl_div_dual=kl_div_estimator_dual,
                denominator="n_valid_tokens",
            )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_proximal = current_version - 1
        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        proximal_stats = _tensor_scalar_stats(staleness_proximal)
        theta_stats = _tensor_scalar_stats(staleness_theta)

        stats_tracker.scalar(
            sample_staleness_proximal_avg=proximal_stats["avg"],
            sample_staleness_proximal_max=proximal_stats["max"],
            sample_staleness_proximal_min=proximal_stats["min"],
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            v_proximal=v_proximal,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )
