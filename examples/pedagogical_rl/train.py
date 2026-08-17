from __future__ import annotations

import pathlib
import random
import sys
from dataclasses import asdict
from datetime import datetime

from examples.pedagogical_rl.algorithm import PedagogicalPPOTrainer
from examples.pedagogical_rl.config import PedagogicalRLConfig

from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def _limit_dataset(dataset, limit: int):
    if limit < 0 or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _prepare_train_dataset(dataset, limit: int, seed: int):
    """Match PedagogicalRL's head selection followed by seeded shuffle."""

    return _limit_dataset(dataset, limit).shuffle(seed=seed)


def _limit_eval_dataset(dataset, limit: int, seed: int):
    """Select evaluation problems exactly as examples/tutor/train.py does.

    Head selection would score this arm on a different subset than the tutor
    arm whenever max_eval_examples is set, silently breaking the comparison.
    Training keeps native head-selection-then-shuffle above.
    """

    if limit < 0 or limit >= len(dataset):
        return dataset
    rng = random.Random(seed)
    return dataset.select(sorted(rng.sample(range(len(dataset)), k=limit)))


def main(args: list[str]) -> None:
    config_path = pathlib.Path(args[args.index("--config") + 1])
    has_trial_name_override = any(arg.startswith("trial_name=") for arg in args)
    if not has_trial_name_override:
        trial_name = next(
            line.split(":", 1)[1].strip().strip("'").strip('"')
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("trial_name:")
        )
        args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]

    config, _ = load_expr_config(args, PedagogicalRLConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = _prepare_train_dataset(
        get_custom_dataset(
            split="train",
            dataset_config=config.train_dataset,
            tokenizer=tokenizer,
        ),
        config.max_train_examples,
        config.seed,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = _limit_eval_dataset(
            get_custom_dataset(
                split="test",
                dataset_config=config.valid_dataset,
                tokenizer=tokenizer,
            ),
            config.max_eval_examples,
            config.seed,
        )

    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "student_model": asdict(config.student_model),
        "judge_model": asdict(config.judge_model),
        "generation": asdict(config.generation),
        "teacher_pre": asdict(config.teacher_pre),
        "debug_trace_dir": config.debug_trace_dir,
        "debug_trace_every_n_rollouts": config.debug_trace_every_n_rollouts,
    }
    eval_workflow_kwargs = dict(workflow_kwargs)
    eval_workflow_kwargs.update(
        {
            # Evaluation only, and deliberately absent from workflow_kwargs so a
            # training rollout cannot run it even if the flag is on.
            "cross_eval": asdict(config.cross_eval),
            "gconfig": config.eval_gconfig,
            "eval_repeat_count": config.evaluator.average_rollouts,
            # Evaluation metrics are emitted by each completed repeat. A failed
            # presolve skips only that repeat; fixed-size atomicity is required
            # only by training-time GRPO normalization.
            "require_complete_group": False,
        }
    )

    with PedagogicalPPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=config.workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=config.eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
