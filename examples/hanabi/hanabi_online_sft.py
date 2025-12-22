import itertools
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    TrainEngineConfig,
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine, MegatronLMEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.hanabi_online_sft import HanabiOnlineSFTWorkflow

logger = logging.getLogger(__name__)


@dataclass
class HanabiOnlineSFTConfig(BaseExperimentConfig):
    async_training: bool = field(default=True)
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    actor: TrainEngineConfig = field(default_factory=TrainEngineConfig)
    tokenizer_path: str = ""
    score_multiplier: float = 1.0
    misplay_penalty_factor: float = 0.1
    use_question_tokens: bool = False
    env_kwargs: dict | None = None


LOGGING_KEYS = [
    "traj_steps",
    "traj_len",
    "traj_input_len",
    "traj_output_len",
    "total_reward",
    "final_score",
    "info_tokens",
    "fuse_tokens",
    "deck_remaining",
    "format_reward_scale",
    "avgt_qgen",
    "avgt_agent_ans",
    "avgt_teacher_ans",
    "avgt_action_build",
    "avgt_action_gen",
    "avgt_action_decode",
    "avgt_env_step",
    "avgt_summary",
    "avgt_pack",
    "avgt_tokenize",
]


def _append_stop_tokens(tokenizer, stop_token_ids: list[int]) -> None:
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)


def _create_actor(allocation_mode: AllocationMode, actor_config: TrainEngineConfig):
    if allocation_mode.train_backend == "fsdp":
        actor = FSDPLMEngine(config=actor_config)
    elif allocation_mode.train_backend == "megatron":
        actor = MegatronLMEngine(config=actor_config)
    else:
        raise ValueError(
            "Invalid backend: {backend}, expected fsdp or megatron".format(
                backend=allocation_mode.train_backend
            )
        )
    actor.create_process_group(parallel_strategy=allocation_mode.train)
    return actor


def main(args):
    config, _ = load_expr_config(args, HanabiOnlineSFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    rank = int(os.getenv("RANK", "0"))
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    allocation_mode = AllocationMode.from_str(config.allocation_mode)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    actor = _create_actor(allocation_mode, config.actor)
    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=allocation_mode.train.dp_size)

    engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
    actor.initialize(**engine_init_kwargs)

    if config.actor.weight_update_mode == "disk":
        weight_update_meta = WeightUpdateMeta.from_disk(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            file_root=config.cluster.fileroot,
            name="default",
            use_lora=config.actor.use_lora,
            clear_checkpoint_after_load=True,
        )
    elif config.actor.weight_update_mode == "xccl":
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)
    else:
        raise ValueError(
            f"Invalid weight update mode: {config.actor.weight_update_mode}"
        )

    actor.connect_engine(rollout, weight_update_meta)

    _append_stop_tokens(tokenizer, config.gconfig.stop_token_ids)

    workflow = HanabiOnlineSFTWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        env_kwargs=config.env_kwargs,
        score_multiplier=config.score_multiplier,
        misplay_penalty_factor=config.misplay_penalty_factor,
        use_question_tokens=config.use_question_tokens,
    )

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )

    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = itertools.cycle(train_dataloader)
    for global_step in range(start_step, max_steps):
        if (
            config.total_train_steps is not None
            and global_step >= config.total_train_steps
        ):
            break
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = actor.prepare_batch(
                    train_dataloader, workflow=workflow
                )
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        if len(batch) == 0:
            logger.info("Filtered rollout batch skipped due to low score threshold.")
            continue

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("train_step"):
            actor.train_lm(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("lm update")

        if "logging" in batch:
            vals = batch["logging"].float().cpu()
            mask = vals[:, 0] > 0
            if mask.any():
                filtered = vals[mask]
                avg = filtered.mean(0).tolist()
                extra = dict(zip(LOGGING_KEYS, avg))
                stats_tracker.scalar(**extra)

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        stats_logger.commit(
            epoch,
            step,
            global_step,
            stats_tracker.export(reduce_group=actor.data_parallel_group),
        )

    stats_logger.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])