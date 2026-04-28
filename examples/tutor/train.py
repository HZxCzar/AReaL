import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import TutorConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, TutorConfig)
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
        enable_thinking=config.enable_thinking,
        aux_base_url=config.aux_base_url,
        aux_model=config.aux_model,
        aux_api_key=config.aux_api_key,
        aux_timeout=config.aux_timeout,
        aux_max_tokens=config.aux_max_tokens,
        aux_temperature=config.aux_temperature,
        aux_top_p=config.aux_top_p,
        max_concurrent_aux_calls=config.max_concurrent_aux_calls,
        api_params_config_path=config.api_params_config_path or None,
        api_params_key=config.api_params_key or None,
        term_success_reward=config.term_success_reward,
        transfer_bonus_reward=config.transfer_bonus_reward,
        counterfactual_student_reward=config.counterfactual_student_reward,
        token_budget_penalty=config.token_budget_penalty,
        teacher_system_prompt=config.teacher_system_prompt,
        student_system_prompt=config.student_system_prompt,
        judge_system_prompt=config.judge_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        generator_system_prompt=config.generator_system_prompt,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
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
