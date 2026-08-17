from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal import PPOTrainer
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import PPOActorConfig
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.trainer.ppo.actor import PPOActor
from areal.utils.environ import is_single_controller


class PedagogicalDistributedSampler(DistributedSampler):
    """Reproduce PedagogicalRL's stateful seed-42 permutations across epochs."""

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            for _ in range(self.epoch + 1):
                indices = torch.randperm(
                    len(self.dataset), generator=generator
                ).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[
                    :padding_size
                ]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"sampler produced {len(indices)} samples; expected {self.num_samples}"
            )
        return iter(indices)


class PedagogicalPPOActor(PPOActor):
    """GRPO advantage used by PedagogicalRL.

    One group-normalized episode advantage is broadcast to every trainable
    teacher token. Student, judge, and final-solution tokens never enter the
    teacher trajectory and therefore never receive loss.
    """

    def _compute_advantages(
        self,
        data: dict[str, Any],
        world_model_rl_reweight_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # The base class grew this parameter with AReaL's world-model support.
        # PedagogicalRL's advantage is the native flat sequence-level one, so
        # the parameter is refused rather than quietly dropped.
        if world_model_rl_reweight_config:
            raise ValueError(
                "The PedagogicalRL baseline does not implement world-model "
                "advantage reweighting. Disable world_model for this arm."
            )
        reward_score = (data["rewards"] + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, min=-self.reward_clip, max=self.reward_clip
        )
        if self.reward_norm is not None:
            reward_score = self.reward_norm(reward_score)

        loss_mask = torch.roll(data["loss_mask"].float(), shifts=-1, dims=-1)
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            old_logp = data.get("prox_logp")
            if old_logp is None:
                raise ValueError("prox_logp is required when recompute_logprob=True")
            data["logprobs"] = old_logp
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                data["prox_logp"] = old_logp
        old_logp = old_logp * loss_mask

        advantages = reward_score.float().unsqueeze(-1).expand_as(loss_mask)
        advantages = advantages * loss_mask
        token_rewards = torch.zeros_like(advantages)
        attention_lengths = data["attention_mask"].sum(-1).long()
        batch_indices = torch.arange(
            attention_lengths.shape[0], device=attention_lengths.device
        )
        terminal_indices = torch.clamp(attention_lengths - 2, min=0)
        token_rewards[batch_indices, terminal_indices] = reward_score.float()

        data["advantages"] = advantages
        data["returns"] = advantages
        data["kl_rewards"] = torch.zeros_like(advantages)
        data["tot_rewards"] = token_rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp
        return data


class PedagogicalFSDPPPOActor(FSDPPPOActor):
    """FSDP PPO actor that performs PedagogicalRL's μ full-batch updates."""

    def __init__(self, config: PPOActorConfig):
        super().__init__(config)
        self.actor = PedagogicalPPOActor(config, self)

    def ppo_update(
        self,
        data: list[dict[str, Any]],
        world_model_batch: list[dict[str, Any]] | None = None,
    ) -> None:
        if world_model_batch is not None:
            raise ValueError(
                "The PedagogicalRL baseline does not implement world-model "
                "updates. Disable world_model for this arm."
            )
        iterations = int(getattr(self.config, "num_iterations", 1))
        for iteration in range(iterations):
            self.actor.ppo_update(data)
            # The outer AReaL trainer steps once after this method. Step between
            # internal updates so μ updates also correspond to μ scheduler steps.
            if iteration + 1 < iterations:
                self.lr_scheduler_step()


class _EvalRepeatMixin:
    def _evaluate_fn(self, eval_workflow, eval_workflow_kwargs):
        import torch.distributed as dist

        from areal.infra.platforms import current_platform

        if self.actor.is_data_parallel_head():
            count = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.evaluator.average_rollouts,
                        is_eval=True,
                    )
                    count += 1
            self.eval_rollout.wait(count, timeout=None)
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()


class PedagogicalPPOTrainer(_EvalRepeatMixin, PPOTrainer):
    """PPO trainer selecting the example-local FSDP algorithm adapter."""

    def _create_dataloader(
        self,
        dataset,
        dataset_config,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        if dataset_config is not self.config.train_dataset:
            return super()._create_dataloader(
                dataset,
                dataset_config=dataset_config,
                rank=rank,
                world_size=world_size,
            )
        if dataset_config.batch_size % world_size != 0:
            raise ValueError(
                f"batch size({dataset_config.batch_size}) must be divisible by "
                f"world_size({world_size})"
            )
        sampler = PedagogicalDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=dataset_config.shuffle,
            seed=self.config.seed,
            drop_last=True,
        )
        return StatefulDataLoader(
            dataset,
            batch_size=dataset_config.batch_size // world_size,
            sampler=sampler,
            drop_last=dataset_config.drop_last,
            num_workers=dataset_config.num_workers,
            collate_fn=lambda rows: rows,
        )

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> Any:
        if alloc.backend != "fsdp":
            raise ValueError(
                "PedagogicalRL's exact μ adapter currently supports actor.backend=fsdp"
            )
        if is_single_controller():
            actor = PedagogicalFSDPPPOActor.as_controller(actor_config, self.scheduler)
        else:
            actor = PedagogicalFSDPPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
