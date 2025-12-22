import os
import sys

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import SFTConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.dataset import get_custom_dataset
from areal.engine.sft.lm_engine import FSDPLMEngine
from areal.utils.data import pad_sequences_to_tensors
from areal.utils.evaluator import Evaluator
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import seeding, stats_tracker


def main(args):
    config, _ = load_expr_config(args, SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, f"trainer{rank}")

    train_dataset = get_custom_dataset(
        path=config.train_dataset.path,
        rank=rank,
        world_size=world_size,
        split="train",
        type=config.train_dataset.type,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        path=config.valid_dataset.path,
        rank=rank,
        world_size=world_size,
        split="test",
        type=config.valid_dataset.type,
        tokenizer=tokenizer,
    )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )
    engine = FSDPLMEngine(config=config.model)
    engine.initialize(None, ft_spec)

    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        engine,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    global_step = 0
    for epoch in range(total_epochs):
        for step, data in enumerate(train_dataloader):
            if global_step < start_step:
                global_step += 1
                continue
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=len(train_dataloader),
            )
            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("sft"),
            ):
                stats = engine.train_lm(data)
                engine.step_lr_scheduler()
                stats_tracker.scalar(**stats)

            with stats_tracker.record_timing("save"):
                saver.save(engine, epoch, step, global_step, tokenizer=tokenizer)

            with stats_tracker.record_timing("eval"):
                def evaluate_fn():
                    with stats_tracker.scope("sft-eval"):
                        for batch in valid_dataloader:
                            engine.evaluate_lm(batch)

                evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            with stats_tracker.record_timing("checkpoint_for_recover"):
                recover_handler.dump(
                    engine,
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
                stats_tracker.export(reduce_group=engine.parallelism_group),
            )
            global_step += 1

    stats_logger.close()
    engine.destroy()


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

python -m areal.launcher.slurm examples/lite/hanabi_sft.py \
    --config examples/lite/configs/hanabi_sft.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-hanabi-sft-2 \
    trial_name=sft-7b-2-cards

python -m areal.launcher.slurm examples/lite/gsm8k_sft.py \
    --config examples/lite/configs/gsm8k_sft.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-hanabi-sft-1 \
    trial_name=gsm8k_trial
"""