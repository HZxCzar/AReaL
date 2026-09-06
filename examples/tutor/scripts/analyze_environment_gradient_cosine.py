#!/usr/bin/env python3
"""Estimate per-student-environment policy-gradient cosine from saved rollouts.

This is a read-only diagnostic.  It reconstructs reward-v3's gate-masked,
turn-level leave-one-out outcome advantages, differentiates the saved teacher
outputs through a LoRA checkpoint, and compares the resulting gradients.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ENVIRONMENTS = (
    "none",
    "attempt-diagnosis",
    "contrastive-comparison",
    "subgoal-decomposition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise LoRA policy-gradient cosine by environment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-version-min", type=int)
    parser.add_argument("--rollout-version-max", type=int)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Use a prebuilt compatible manifest instead of historical rollouts.",
    )
    parser.add_argument("--groups-per-environment", type=int, default=5)
    parser.add_argument("--debug-trace-every", type=int, default=10)
    parser.add_argument("--max-turns-per-environment", type=int, default=0)
    parser.add_argument("--max-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--max-sequences-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2"
    )
    return parser.parse_args()


def environment_name(student_name: str) -> str:
    for name in ENVIRONMENTS[1:]:
        if student_name.endswith(name):
            return name
    if student_name.endswith("-original"):
        return "none"
    raise ValueError(f"Unknown student environment: {student_name!r}")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def read_rollout_group(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def match_traces(
    trace_dir: Path,
    task_id: int,
    trajectory_ids: set[int],
) -> dict[int, dict[str, Any]]:
    matched: dict[int, dict[str, Any]] = {}
    pattern = str(trace_dir / f"task_{task_id:08d}_*.json")
    for raw_path in glob.glob(pattern):
        trace = load_json(Path(raw_path))
        if trace is None:
            continue
        trajectory_id = int(trace.get("trajectory_id", -1))
        if trajectory_id in trajectory_ids:
            matched[trajectory_id] = trace
    return matched


def improvement(trace: dict[str, Any]) -> float:
    return sum(
        float(turn.get("reward_components", {}).get("student_generalize_original") or 0.0)
        for turn in trace["turns"]
    )


def reconstruct_group(
    rows: list[dict[str, Any]],
    traces: dict[int, dict[str, Any]],
    rollout_version: int,
    task_id: int,
) -> dict[str, Any]:
    rows_by_trajectory: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_trajectory[int(row["trajectory_id"])].append(row)

    episode_improvement = {
        trajectory_id: improvement(traces[trajectory_id])
        for trajectory_id in rows_by_trajectory
    }
    trace_turns = {
        trajectory_id: {
            int(turn["turn_idx"]): turn for turn in traces[trajectory_id]["turns"]
        }
        for trajectory_id in rows_by_trajectory
    }

    records: list[dict[str, Any]] = []
    for trajectory_id, trajectory_rows in rows_by_trajectory.items():
        for row in trajectory_rows:
            turn_idx = int(row["turn_idx"])
            turn = trace_turns[trajectory_id].get(turn_idx)
            if turn is None:
                raise ValueError(
                    f"Missing trace turn {turn_idx} for trajectory {trajectory_id}"
                )
            credit = not bool(turn.get("personality_gated", False)) and not bool(
                turn.get("leak_masked", False)
            )
            records.append(
                {
                    "trajectory_id": trajectory_id,
                    "turn_idx": turn_idx,
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "prompt_len": int(row["prompt_len"]),
                    "seqlen": int(row["seqlen"]),
                    "credit": credit,
                    "raw_return": episode_improvement[trajectory_id]
                    if credit
                    else 0.0,
                }
            )

    # Reward-v3: leave-one-out among trajectories at the same turn depth.
    by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_turn[record["turn_idx"]].append(record)
    all_trajectory_ids = list(rows_by_trajectory)
    for same_turn in by_turn.values():
        if len(same_turn) > 1:
            total = sum(float(record["raw_return"]) for record in same_turn)
            for record in same_turn:
                baseline = (total - float(record["raw_return"])) / (
                    len(same_turn) - 1
                )
                record["advantage"] = float(record["raw_return"]) - baseline
        else:
            record = same_turn[0]
            if record["credit"] and len(all_trajectory_ids) > 1:
                trajectory_id = int(record["trajectory_id"])
                baseline = sum(
                    episode_improvement[other]
                    for other in all_trajectory_ids
                    if other != trajectory_id
                ) / (len(all_trajectory_ids) - 1)
                record["advantage"] = float(record["raw_return"]) - baseline
            else:
                record["advantage"] = 0.0

    student_names = {
        str(trace["student"]["name"]) for trace in traces.values()
    }
    if len(student_names) != 1:
        raise ValueError(
            f"Task {task_id} mixes student environments: {sorted(student_names)}"
        )
    env = environment_name(next(iter(student_names)))
    for record in records:
        record.update(
            rollout_version=rollout_version,
            task_id=task_id,
            environment=env,
        )
    return {
        "environment": env,
        "rollout_version": rollout_version,
        "task_id": task_id,
        "episode_count": len(rows_by_trajectory),
        "records": records,
    }


def discover_groups(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for version in range(args.rollout_version_min, args.rollout_version_max + 1):
        version_dir = args.rollout_root / str(version)
        if not version_dir.is_dir():
            continue
        for path in sorted(version_dir.glob("*.jsonl"), key=lambda item: int(item.stem)):
            task_id = int(path.stem)
            if task_id % args.debug_trace_every:
                continue
            rows = read_rollout_group(path)
            if not rows:
                continue
            trajectory_ids = {int(row["trajectory_id"]) for row in rows}
            traces = match_traces(args.trace_dir, task_id, trajectory_ids)
            if set(traces) != trajectory_ids:
                continue
            group = reconstruct_group(rows, traces, version, task_id)
            groups[group["environment"]].append(group)
    return groups


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    discovered = discover_groups(args)
    counts = {name: len(discovered.get(name, [])) for name in ENVIRONMENTS}
    missing = {
        name: count
        for name, count in counts.items()
        if count < args.groups_per_environment
    }
    if missing:
        raise RuntimeError(
            f"Not enough complete traced groups: requested "
            f"{args.groups_per_environment} each, available={counts}"
        )

    rng = random.Random(args.seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    selected_group_ids: dict[str, list[str]] = {}
    for env_index, env in enumerate(ENVIRONMENTS):
        candidates = list(discovered[env])
        random.Random(args.seed + env_index).shuffle(candidates)
        candidates.sort(key=lambda group: int(group["rollout_version"]))
        chosen = candidates[: args.groups_per_environment]
        selected_group_ids[env] = [
            f"v{group['rollout_version']}/task{group['task_id']}" for group in chosen
        ]
        records = [record for group in chosen for record in group["records"]]
        nonzero = [
            record for record in records if abs(float(record["advantage"])) > 1e-12
        ]
        if args.max_turns_per_environment and len(nonzero) > args.max_turns_per_environment:
            rng.shuffle(nonzero)
            nonzero = nonzero[: args.max_turns_per_environment]
        selected[env] = nonzero

    # The historical per-optimizer-batch std is not stored.  A common positive
    # scalar cannot change cosine, so no estimated std is applied here.
    return {
        "diagnostic": "reward-v3 outcome policy-gradient cosine",
        "checkpoint": str(args.checkpoint.resolve()),
        "rollout_version_min": args.rollout_version_min,
        "rollout_version_max": args.rollout_version_max,
        "groups_per_environment": args.groups_per_environment,
        "available_complete_groups": counts,
        "environments": selected,
        "selected_groups": selected_group_ids,
    }


def batches(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    max_sequences: int,
) -> list[list[dict[str, Any]]]:
    encoded: list[dict[str, Any]] = []
    for example in examples:
        if "input_ids" in example:
            ids = [int(token) for token in example["input_ids"]]
            prompt_ids = ids[: int(example["prompt_len"])]
        else:
            ids = tokenizer.encode(
                example["prompt"] + example["completion"],
                add_special_tokens=False,
            )
            prompt_ids = tokenizer.encode(
                example["prompt"], add_special_tokens=False
            )
        # Generation backends can retain the terminal EOS in input_ids while
        # omitting that final special token when decoding the saved text.  It is
        # unambiguous when the prompt length still matches and total length is
        # short by exactly one token.
        if (
            len(prompt_ids) == int(example["prompt_len"])
            and len(ids) + 1 == int(example["seqlen"])
        ):
            ids.append(tokenizer.eos_token_id)
        if len(ids) != int(example["seqlen"]) or len(prompt_ids) != int(
            example["prompt_len"]
        ):
            raise ValueError(
                "Saved text no longer tokenizes to its recorded lengths: "
                f"v{example['rollout_version']}/task{example['task_id']}/"
                f"trajectory{example['trajectory_id']}/turn{example['turn_idx']} "
                f"got ({len(prompt_ids)}, {len(ids)}), expected "
                f"({example['prompt_len']}, {example['seqlen']})"
            )
        item = dict(example)
        item["input_ids"] = ids
        encoded.append(item)

    encoded.sort(key=lambda item: len(item["input_ids"]))
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = 0
    for item in encoded:
        candidate_max = max(current_max, len(item["input_ids"]))
        candidate_count = len(current) + 1
        if current and (
            candidate_count > max_sequences
            or candidate_max * candidate_count > max_tokens
        ):
            result.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, len(item["input_ids"]))
    if current:
        result.append(current)
    return result


def load_model(checkpoint: Path, device: torch.device, attn_impl: str):
    adapter_config = PeftConfig.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    base_path = adapter_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        device_map={"": device.index},
    )
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(
        base,
        str(checkpoint),
        config=adapter_config,
        is_trainable=True,
        autocast_adapter_dtype=False,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected only LoRA parameters to be trainable")
    return model, tokenizer, trainable


def backward_environment(
    model: Any,
    tokenizer: Any,
    trainable: list[tuple[str, torch.nn.Parameter]],
    examples: list[dict[str, Any]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int, int]:
    model.zero_grad(set_to_none=True)
    output_tokens = 0
    work_batches = batches(
        examples,
        tokenizer,
        args.max_tokens_per_batch,
        args.max_sequences_per_batch,
    )
    for batch_index, batch in enumerate(work_batches, 1):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        advantages = torch.tensor(
            [float(item["advantage"]) for item in batch],
            dtype=torch.float32,
            device=device,
        )
        response_mask = torch.zeros(
            (len(batch), max_len - 1), dtype=torch.bool, device=device
        )
        for index, item in enumerate(batch):
            ids = torch.tensor(item["input_ids"], dtype=torch.long, device=device)
            seq_len = ids.numel()
            prompt_len = int(item["prompt_len"])
            input_ids[index, :seq_len] = ids
            attention_mask[index, :seq_len] = 1
            response_mask[index, prompt_len - 1 : seq_len - 1] = True
            output_tokens += seq_len - prompt_len

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1, :]
            token_nll = F.cross_entropy(
                logits.transpose(1, 2).float(), input_ids[:, 1:], reduction="none"
            )
            # Dividing by a fixed number only controls accumulated gradient scale;
            # cosine is invariant to it.
            loss = (
                token_nll
                * response_mask.to(token_nll.dtype)
                * advantages[:, None]
            ).sum() / 1024.0
        loss.backward()
        if os.environ.get("RANK") == "0":
            print(
                f"[gradient] batch {batch_index}/{len(work_batches)}",
                flush=True,
            )
        del logits, token_nll, loss, input_ids, attention_mask, response_mask

    pieces = []
    for _, parameter in trainable:
        gradient = parameter.grad
        if gradient is None:
            pieces.append(torch.zeros(parameter.numel(), device=device, dtype=torch.float32))
        else:
            pieces.append(gradient.detach().float().reshape(-1))
    return torch.cat(pieces), output_tokens, len(examples)


def write_results(output_dir: Path, manifest: dict[str, Any]) -> None:
    gradients: dict[str, torch.Tensor] = {}
    stats: dict[str, Any] = {}
    for env in ENVIRONMENTS:
        payload = torch.load(output_dir / f"gradient_{env}.pt", map_location="cpu")
        gradient = payload["gradient"].float()
        token_count = int(payload["output_tokens"])
        gradients[env] = gradient / max(token_count, 1)
        stats[env] = {
            "groups": len(manifest["selected_groups"][env]),
            "turns": int(payload["turns"]),
            "output_tokens": token_count,
            "mean_abs_advantage": float(payload["mean_abs_advantage"]),
        }

    def stable_dot(left: torch.Tensor, right: torch.Tensor) -> float:
        total = 0.0
        chunk_size = 1_000_000
        for start in range(0, left.numel(), chunk_size):
            stop = min(start + chunk_size, left.numel())
            total += float(
                torch.dot(left[start:stop].double(), right[start:stop].double())
            )
        return total

    squared_norm = {env: stable_dot(gradient, gradient) for env, gradient in gradients.items()}
    for env in ENVIRONMENTS:
        stats[env]["mean_gradient_norm"] = math.sqrt(squared_norm[env])

    cosine: dict[str, dict[str, float]] = {}
    for left in ENVIRONMENTS:
        cosine[left] = {}
        for right in ENVIRONMENTS:
            denominator = math.sqrt(squared_norm[left] * squared_norm[right])
            cosine[left][right] = stable_dot(
                gradients[left], gradients[right]
            ) / max(denominator, 1e-30)

    result = {
        "definition": (
            "Gradient of -advantage*log_pi over saved teacher response tokens; "
            "reward-v3 gate-masked turn-level LOO outcome advantages only."
        ),
        "interpretation": "negative=conflict, zero=orthogonal/noisy, positive=aligned",
        "manifest": {key: value for key, value in manifest.items() if key != "environments"},
        "environment_stats": stats,
        "cosine": cosine,
    }
    with (output_dir / "result.json").open("w") as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Environment policy-gradient cosine",
        "",
        "| Source gradient | " + " | ".join(ENVIRONMENTS) + " |",
        "|---|" + "---:|" * len(ENVIRONMENTS),
    ]
    for left in ENVIRONMENTS:
        lines.append(
            f"| {left} | "
            + " | ".join(f"{cosine[left][right]:.4f}" for right in ENVIRONMENTS)
            + " |"
        )
    lines.extend(
        [
            "",
            "| Environment | Groups | Turns | Output tokens | Mean |advantage| | Mean gradient norm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for env in ENVIRONMENTS:
        row = stats[env]
        lines.append(
            f"| {env} | {row['groups']} | {row['turns']} | "
            f"{row['output_tokens']} | {row['mean_abs_advantage']:.6f} | "
            f"{row['mean_gradient_norm']:.6g} |"
        )
    lines.extend(
        [
            "",
            "> This is a checkpoint gradient diagnostic on nearby saved rollouts, "
            "not a reconstruction of an already-consumed optimizer step.",
        ]
    )
    (output_dir / "result.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def merge_gradient_shards(
    output_dir: Path,
    shards: int,
) -> None:
    for env in ENVIRONMENTS:
        merged_gradient: torch.Tensor | None = None
        parameter_names: list[str] | None = None
        output_tokens = 0
        turns = 0
        abs_advantage_sum = 0.0
        for shard_index in range(shards):
            path = output_dir / f"gradient_{env}_shard{shard_index}.pt"
            payload = torch.load(path, map_location="cpu")
            shard_names = payload["parameter_names"]
            if parameter_names is None:
                parameter_names = shard_names
            elif parameter_names != shard_names:
                raise RuntimeError(f"LoRA parameter order differs in {path}")
            shard_gradient = payload["gradient"].float()
            if merged_gradient is None:
                merged_gradient = shard_gradient
            else:
                merged_gradient.add_(shard_gradient)
            output_tokens += int(payload["output_tokens"])
            turns += int(payload["turns"])
            abs_advantage_sum += float(payload["abs_advantage_sum"])
        assert merged_gradient is not None and parameter_names is not None
        torch.save(
            {
                "gradient": merged_gradient,
                "parameter_names": parameter_names,
                "output_tokens": output_tokens,
                "turns": turns,
                "mean_abs_advantage": abs_advantage_sum / max(turns, 1),
            },
            output_dir / f"gradient_{env}.pt",
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen3-8B backward pass")
    # Workers do independent model backward passes.  Use Gloo only for tiny
    # barriers and merge the large LoRA gradients through local files; creating
    # four simultaneous 170+ MB NCCL subgroup reductions is needlessly fragile.
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (4, 8):
        raise RuntimeError(f"Use 4 or 8 GPUs; got world_size={world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if rank == 0:
        if args.input_manifest is not None:
            with args.input_manifest.open() as handle:
                manifest = json.load(handle)
            missing = sorted(set(ENVIRONMENTS) - set(manifest.get("environments", {})))
            if missing:
                raise ValueError(f"Input manifest is missing environments: {missing}")
            manifest["checkpoint"] = str(args.checkpoint.resolve())
        else:
            required = {
                "rollout_root": args.rollout_root,
                "trace_dir": args.trace_dir,
                "rollout_version_min": args.rollout_version_min,
                "rollout_version_max": args.rollout_version_max,
            }
            absent = [name for name, value in required.items() if value is None]
            if absent:
                raise ValueError(
                    "Historical-rollout mode requires: " + ", ".join(absent)
                )
            manifest = build_manifest(args)
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle)
        print(
            "[manifest] "
            + ", ".join(
                f"{env}={len(manifest['environments'][env])} nonzero turns"
                for env in ENVIRONMENTS
            ),
            flush=True,
        )
    dist.barrier()
    with manifest_path.open() as handle:
        manifest = json.load(handle)

    env_index = rank % len(ENVIRONMENTS)
    env = ENVIRONMENTS[env_index]
    shards = world_size // len(ENVIRONMENTS)
    shard_index = rank // len(ENVIRONMENTS)
    examples = sorted(
        manifest["environments"][env],
        key=lambda item: int(item["seqlen"]),
        reverse=True,
    )[shard_index::shards]
    print(
        f"[rank {rank}] environment={env} shard={shard_index + 1}/{shards} "
        f"turns={len(examples)}",
        flush=True,
    )

    model, tokenizer, trainable = load_model(
        args.checkpoint, device, args.attn_implementation
    )
    gradient, output_tokens, turns = backward_environment(
        model, tokenizer, trainable, examples, device, args
    )
    payload = {
        "gradient": gradient.cpu(),
        "parameter_names": [name for name, _ in trainable],
        "output_tokens": output_tokens,
        "turns": turns,
        "abs_advantage_sum": sum(
            abs(float(item["advantage"])) for item in examples
        ),
    }
    torch.save(
        payload,
        args.output_dir / f"gradient_{env}_shard{shard_index}.pt",
    )

    del model, tokenizer, trainable, gradient
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        merge_gradient_shards(args.output_dir, shards)
        write_results(args.output_dir, manifest)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
