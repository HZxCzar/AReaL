import os
import sys
from dataclasses import dataclass

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.hanabi_online_sft import HanabiOnlineSFTWorkflow


@dataclass
class HanabiOnlineSFTConfig(GRPOConfig):
    score_multiplier: float = 1.0
    misplay_penalty_factor: float = 0.1
    use_question_tokens: bool = False
    env_kwargs: dict | None = None


def _append_stop_tokens(tokenizer, stop_token_ids: list[int]) -> None:
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)


def main(args):
    config, _ = load_expr_config(args, HanabiOnlineSFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    _append_stop_tokens(tokenizer, config.gconfig.stop_token_ids)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        workflow = HanabiOnlineSFTWorkflow(
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            dump_dir=os.path.join(
                StatsLogger.get_log_path(config.stats_logger), "generated"
            ),
            env_kwargs=dict(config.env_kwargs or {}),
            sft_reg=config.actor.sft_reg,
            misplay_penalty_factor=config.misplay_penalty_factor,
            use_question_tokens=config.use_question_tokens,
        )
        trainer.train(workflow=workflow)


if __name__ == "__main__":
    main(sys.argv[1:])

"""
python -m areal.infra.launcher.slurm examples/hanabi/hanabi_online_sft.py \
    --config examples/hanabi/hanabi_online_sft.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-hanabi-online-sft \
    trial_name=online-sft-qwen3
"""
