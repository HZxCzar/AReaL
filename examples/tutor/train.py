import pathlib
import random
import sys
from copy import deepcopy
from datetime import datetime

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import TutorConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config_path = pathlib.Path(args[args.index("--config") + 1])
    trial_name = next(
        line.split(":", 1)[1].strip().strip("'\"")
        for line in config_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("trial_name:")
    )
    args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]
    config, _ = load_expr_config(args, TutorConfig)
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    pairwise = reward.pairwise
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset_config = config.valid_dataset
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if eval_max_samples <= 0:
            eval_max_samples = None
    if eval_max_samples is not None and valid_dataset_config is not None:
        valid_dataset_config = deepcopy(valid_dataset_config)
        valid_dataset_config.scheduling_spec = None

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_dataset_config,
        tokenizer=tokenizer,
    )
    if eval_max_samples is not None and eval_max_samples < len(valid_dataset):
        rng = random.Random(config.seed)
        eval_indices = sorted(rng.sample(range(len(valid_dataset)), k=eval_max_samples))
        valid_dataset = valid_dataset.select(eval_indices)

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        answer_scorer=config.answer_scorer,
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
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        teacher_system_prompt=config.teacher_system_prompt,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        summary_system_prompt=config.summary_system_prompt,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
        max_train_sample_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        pairwise_reward_enabled=pairwise.enabled,
        pairwise_reference_lag_steps=pairwise.reference_lag_steps,
        pairwise_reward_scale=pairwise.scale,
        pairwise_compare_all_turns=pairwise.compare_all_turns,
        pairwise_judge_both_incorrect=pairwise.judge_both_incorrect,
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
