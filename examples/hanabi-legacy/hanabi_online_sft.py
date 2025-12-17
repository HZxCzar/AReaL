import itertools
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    TrainEngineConfig,
    load_expr_config,
)
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.hanabi_online_sft import HanabiOnlineSFTWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding, stats_tracker

from realhf.base import logging

logger = logging.getLogger("Hanabi Online SFT Exp")


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


def main(args):
    config, _ = load_expr_config(args, HanabiOnlineSFTConfig)
    config: HanabiOnlineSFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    logger.warning(
        f"Loading dataset with len {len(train_dataset)} with batch size {config.train_dataset.batch_size} // {world_size}"
    )
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    actor = FSDPLMEngine(config=config.actor)
    actor.initialize(None, ft_spec)

    weight_update_meta = [
        WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot,
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = HanabiOnlineSFTWorkflow(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        env_kwargs=getattr(config, "env_kwargs", None),
        score_multiplier=getattr(config, "score_multiplier", 1.0),
        misplay_penalty_factor=getattr(config, "misplay_penalty_factor", 0.1),
        use_question_tokens=getattr(config, "use_question_tokens", False),
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
                batch = rollout.prepare_batch(train_dataloader, workflow=workflow)
            else:
                batch = rollout.rollout_batch(next(data_generator), workflow=workflow)

        if len(batch) == 0:
            logger.info("Filtered rollout batch skipped due to low score threshold.")
            continue

        batch = batch.to(actor.device)
        dist.barrier(device_ids=[actor.device.index])
        torch.cuda.synchronize()

        with stats_tracker.record_timing("train_step"):
            stats = actor.train_lm(batch)
            actor.step_lr_scheduler()

        if "logging" in batch:
            vals = batch["logging"].float().cpu()
            mask = vals[:, 0] > 0
            if mask.any():
                filtered = vals[mask]
                avg = filtered.mean(0).tolist()
                extra = dict(zip(LOGGING_KEYS, avg))
                stats.update(extra)

        with stats_tracker.record_timing("update_weights"):
            rollout.pause()
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            torch.cuda.synchronize()
            rollout.resume()
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

        stats_logger.commit(epoch, step, global_step, [stats])

    stats_logger.close()
    rollout.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])


"""
vim ../AReaL-Lite/examples/lite/configs/gsm8k_grpo.yaml

sudo srun --mpi=pmi2 --ntasks=1 --gres=gpu:8 \
    --cpus-per-task=10 --mem=1500G --pty --job-name=xmy-teac-werewolf\
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/images/sglang-v0.4.9.post2-cu126-v2.sif

cd /storage/openpsi/users/xmy/inclusionAI/AReaL
pip install -e . --no-deps
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_API_KEY=local-667d8d7f101dad4eb9597d718d0c68f40e3792f9
export WANDB_BASE_URL=http://8.150.1.98:8080

python -m areal.launcher.slurm examples/lite/hanabi_online_sft.py \
    --config examples/lite/configs/hanabi_online_sft.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-hanabi-online-sft-2 \
    trial_name=7b-coef2.0
"""