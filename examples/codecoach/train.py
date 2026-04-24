import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import CodeCoachConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, CodeCoachConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        max_turns=config.max_turns,
        student_base_url=config.student_base_url,
        student_model=config.student_model,
        student_api_key=config.student_api_key,
        student_timeout=config.student_timeout,
        student_max_tokens=config.student_max_tokens,
        student_temperature=config.student_temperature,
        student_top_p=config.student_top_p,
        max_concurrent_students=config.max_concurrent_students,
        api_params_config_path=config.api_params_config_path or None,
        api_params_key=config.api_params_key or None,
        work_dir_root=config.work_dir_root,
        max_episode_total_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

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
