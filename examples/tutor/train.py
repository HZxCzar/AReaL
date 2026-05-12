import pathlib
import sys
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import TutorConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    trial_name = next(line.split(":", 1)[1].strip().strip("'\"") for line in pathlib.Path(args[args.index("--config") + 1]).read_text(encoding="utf-8").splitlines() if line.startswith("trial_name:"))
    args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]
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
        success_reward=config.success_reward,
        leak_penalty=config.leak_penalty,
        outcome_prior_turn_weight=config.outcome_prior_turn_weight,
        outcome_credit_gamma=config.outcome_credit_gamma,
        early_success_bonus=config.early_success_bonus,
        turn_penalty=config.turn_penalty,
        length_penalty_threshold_chars=config.length_penalty_threshold_chars,
        length_penalty_per_100_chars=config.length_penalty_per_100_chars,
        length_penalty_min=config.length_penalty_min,
        teacher_system_prompt=config.teacher_system_prompt,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        summary_system_prompt=config.summary_system_prompt,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
        max_train_sample_tokens=config.gconfig.max_tokens,
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
