import pathlib
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

from torch.utils.data import Dataset as TorchDataset

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import CodeCoachConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("CodeCoach")


class _RepeatedDataset(TorchDataset):
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self._dataset = dataset
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[self._indices[index]]


def _repeat_to_min_batch(dataset: Any, batch_size: int, split: str) -> Any:
    if batch_size <= 0:
        return dataset
    try:
        dataset_size = len(dataset)
    except (RuntimeError, TypeError):
        return dataset
    if dataset_size == 0 or dataset_size >= batch_size:
        return dataset

    indices = [idx % dataset_size for idx in range(batch_size)]
    logger.info(
        "Repeating CodeCoach %s dataset from %d to %d rows to preserve batch_size=%d",
        split,
        dataset_size,
        len(indices),
        batch_size,
    )
    if hasattr(dataset, "select"):
        return dataset.select(indices)
    return _RepeatedDataset(dataset, indices)


def _load_local_codecoach_dataset(
    *,
    split: str,
    dataset_config: Any,
    tokenizer: Any,
) -> Any:
    local_dataset_config = deepcopy(dataset_config)
    local_dataset_config.scheduling_spec = None
    dataset = get_custom_dataset(
        split=split,
        dataset_config=local_dataset_config,
        tokenizer=tokenizer,
    )
    return _repeat_to_min_batch(dataset, local_dataset_config.batch_size, split)


def main(args):
    config_path = pathlib.Path(args[args.index("--config") + 1])
    trial_name = next(
        line.split(":", 1)[1].strip().strip("'\"")
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("trial_name:")
    )
    args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]
    config, _ = load_expr_config(args, CodeCoachConfig)
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    pairwise = reward.pairwise
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = _load_local_codecoach_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = _load_local_codecoach_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        max_turns=config.max_turns,
        enable_thinking=config.enable_thinking,
        aux_mode=auxiliary_model.mode,
        aux_enable_thinking=auxiliary_model.enable_thinking,
        aux_base_url=auxiliary_model.base_url,
        aux_model=auxiliary_model.model,
        aux_api_key=auxiliary_model.api_key,
        aux_timeout=auxiliary_model.timeout,
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=auxiliary_model.max_concurrent_calls,
        aux_request_params=auxiliary_model.request_params,
        work_dir_root=config.work_dir_root,
        max_train_sample_tokens=config.gconfig.max_tokens,
        token_budget_penalty=reward.token_budget_penalty,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        pairwise_reward_enabled=pairwise.enabled,
        pairwise_reference_lag_steps=pairwise.reference_lag_steps,
        pairwise_reward_scale=pairwise.scale,
        pairwise_compare_all_turns=pairwise.compare_all_turns,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig.new(n_samples=1)
    eval_workflow_kwargs["pairwise_reward_enabled"] = False

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
