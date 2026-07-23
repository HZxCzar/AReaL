"""Two-GPU smoke test for isolated LoRA optimization under FSDP2."""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tests.utils import get_model_path

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.engine import FSDPEngine

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def _local_clone(parameter: torch.Tensor) -> torch.Tensor:
    tensor = parameter.data
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().clone()


def _adapter_state(engine: FSDPEngine, adapter_name: str) -> dict[str, torch.Tensor]:
    marker = f".{adapter_name}."
    return {
        name: _local_clone(parameter)
        for name, parameter in engine.model.named_parameters()
        if "lora_" in name and marker in name
    }


def _loss_fn(logprobs, entropy, input_data, **kwargs):
    del entropy, input_data, kwargs
    return -logprobs.mean()


def _loss_weight(input_data):
    return input_data["cu_seqlens"][-1]


def _changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


def run(backend: str, output: str) -> None:
    allocation = ModelAllocation.from_str(backend)
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_fsdp_separate_lora",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.0),
        use_lora=True,
        lora_rank=8,
        lora_alpha=8,
        peft_type="lora",
    )
    engine = FSDPEngine(config)
    engine.create_process_group(parallel_strategy=allocation.parallel)
    engine.initialize(
        None,
        FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=2),
        world_model_separate_lora=True,
    )
    input_ids = torch.randint(10, 100, (2, 8), device=engine.device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
    }

    policy_before = _adapter_state(engine, "default")
    world_before = _adapter_state(engine, "world_model")
    engine.activate_world_model_adapter()
    engine.train_batch(batch, loss_fn=_loss_fn, loss_weight_fn=_loss_weight)
    policy_after_world = _adapter_state(engine, "default")
    world_after_world = _adapter_state(engine, "world_model")

    engine.activate_policy_adapter()
    engine.train_batch(batch, loss_fn=_loss_fn, loss_weight_fn=_loss_weight)
    policy_after_policy = _adapter_state(engine, "default")
    world_after_policy = _adapter_state(engine, "world_model")

    success = (
        bool(policy_before)
        and bool(world_before)
        and not _changed(policy_before, policy_after_world)
        and _changed(world_before, world_after_world)
        and _changed(policy_after_world, policy_after_policy)
        and not _changed(world_after_world, world_after_policy)
    )
    success_tensor = torch.tensor(int(success), device=engine.device)
    dist.all_reduce(
        success_tensor,
        op=dist.ReduceOp.MIN,
        group=engine.data_parallel_group,
    )
    if int(os.environ["RANK"]) == 0:
        with open(output, "w", encoding="utf-8") as file:
            file.write("Passed" if bool(success_tensor) else "Failed")
    dist.barrier(group=engine.cpu_group)
    engine.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.backend, args.output)


if __name__ == "__main__":
    main()
