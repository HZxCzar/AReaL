import os
import sys
from dataclasses import dataclass

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.trainer import WerewolfTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from areal.workflow.werewolf import WerewolfWorkflow


@dataclass
class WerewolfGRPOConfig(GRPOConfig):
    role: str = "villager"
    opp_server_addrs: str = ""
    opp_tokenizer_path: str = ""
    opp_api_key: str = ""
    opp_api_model: str = ""
    teacher_server_addrs: str = ""
    teacher_tokenizer_path: str = ""
    teacher_api_key: str = ""
    teacher_api_model: str = ""
    num_villagers: int = 2
    num_werewolves: int = 3
    num_witches: int = 1
    num_foreseers: int = 1
    num_hunters: int = 1
    env_kwargs: dict | None = None
    max_turns: int = 70
    turn_discount: float = 1.0
    teacher_obs_kwargs: dict | None = None


def _parse_server_addrs(addrs: str) -> list[str] | None:
    if not addrs:
        return None
    return [addr.strip() for addr in addrs.split(",") if addr.strip()]


def _append_stop_tokens(tokenizer, stop_token_ids: list[int]) -> None:
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)


def main(args):
    config, _ = load_expr_config(args, WerewolfGRPOConfig)
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

    opp_tokenizer = load_hf_tokenizer(config.opp_tokenizer_path) if config.opp_tokenizer_path else None
    teacher_tokenizer = (
        load_hf_tokenizer(config.teacher_tokenizer_path) if config.teacher_tokenizer_path else None
    )

    _append_stop_tokens(tokenizer, config.gconfig.stop_token_ids)
    if opp_tokenizer is not None:
        _append_stop_tokens(opp_tokenizer, config.gconfig.stop_token_ids)
    if teacher_tokenizer is not None:
        _append_stop_tokens(teacher_tokenizer, config.gconfig.stop_token_ids)

    opp_rollout = None
    teacher_rollout = None
    try:
        with WerewolfTrainer(
            config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        ) as trainer:
            opp_addrs = _parse_server_addrs(config.opp_server_addrs)
            if config.opp_api_key:
                opp_rollout = None
            elif opp_addrs:
                opp_rollout = RemoteSGLangEngine(config.rollout)
                opp_rollout.initialize(
                    addr=opp_addrs,
                    train_data_parallel_size=trainer.allocation_mode.train.dp_size,
                )

            teacher_addrs = _parse_server_addrs(config.teacher_server_addrs)
            if config.teacher_api_key:
                teacher_rollout = None
            elif teacher_addrs:
                teacher_rollout = RemoteSGLangEngine(config.rollout)
                teacher_rollout.initialize(
                    addr=teacher_addrs,
                    train_data_parallel_size=trainer.allocation_mode.train.dp_size,
                )

            env_kwargs = dict(config.env_kwargs or {})
            env_kwargs.update(
                {
                    "num_villagers": config.num_villagers,
                    "num_werewolves": config.num_werewolves,
                    "num_witches": config.num_witches,
                    "num_foreseers": config.num_foreseers,
                    "num_hunters": config.num_hunters,
                }
            )

            workflow = WerewolfWorkflow(
                gconfig=config.gconfig,
                tokenizer=trainer.tokenizer,
                dump_dir=os.path.join(
                    StatsLogger.get_log_path(config.stats_logger), "generated"
                ),
                role=config.role,
                opp_rollout=opp_rollout,
                opp_tokenizer=opp_tokenizer,
                teacher_rollout=teacher_rollout,
                teacher_tokenizer=teacher_tokenizer,
                opp_api_key=config.opp_api_key,
                opp_api_model=config.opp_api_model,
                teacher_api_key=config.teacher_api_key,
                teacher_api_model=config.teacher_api_model,
                env_kwargs=env_kwargs,
                max_turns=config.max_turns,
                turn_discount=config.turn_discount,
                teacher_obs_kwargs=config.teacher_obs_kwargs,
            )
            trainer.train(workflow)
    finally:
        if teacher_rollout is not None:
            teacher_rollout.destroy()
        if opp_rollout is not None:
            opp_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])

"""
vim ../AReaL-Lite/examples/lite/configs/gsm8k_grpo.yaml

sudo srun --mpi=pmi2 --ntasks=1 --gres=gpu:8 \
    --cpus-per-task=10 --mem=1500G --pty --job-name=xmy-teac-werewolf\
    singularity shell --nv --no-home --writable-tmpfs \
    --bind /storage:/storage /storage/openpsi/images/sglang-v0.4.9.post2-cu126-v2.sif

export UV_DEFAULT_INDEX="https://artifacts.antgroup-inc.cn/simple/"
cd /storage/openpsi/users/xmy/inclusionAI/AReaL-New
pip install uv && uv pip install -e .[all]
export HF_ENDPOINT="https://hf-mirror.com"
export WANDB_API_KEY=local-667d8d7f101dad4eb9597d718d0c68f40e3792f9
export WANDB_BASE_URL=http://8.150.1.98:8080

python -m areal.launcher.slurm examples/werewolf/werewolf_grpo.py \
    --config examples/werewolf/werewolf_grpo.yaml \
    stats_logger.wandb.mode=online \
    experiment_name=xmy-werewolf-grpo \
    trial_name=test-3b
"""