import pathlib
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import TutorConfig
from core.generalization import (
    load_student_generalize_bank,
    validate_student_generalize_dataset,
)

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def _without_remote_dataset_loading(dataset_config: Any) -> Any:
    local_config = deepcopy(dataset_config)
    local_config.scheduling_spec = None
    return local_config


def _validate_student_generalize_datasets(config: TutorConfig, tokenizer: Any) -> None:
    student_generalize = config.student_generalize
    if not student_generalize.enabled:
        return

    bank = load_student_generalize_bank(student_generalize.path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=_without_remote_dataset_loading(config.train_dataset),
        tokenizer=tokenizer,
    )
    validate_student_generalize_dataset(
        train_dataset,
        split_name="train",
        bank=bank,
    )

    if config.valid_dataset is None:
        return
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=_without_remote_dataset_loading(config.valid_dataset),
        tokenizer=tokenizer,
    )
    validate_student_generalize_dataset(
        valid_dataset,
        split_name="test",
        bank=bank,
    )


def main(args):
    config_path = pathlib.Path(args[args.index("--config") + 1])
    has_trial_name_override = any(arg.startswith("trial_name=") for arg in args)
    if not has_trial_name_override:
        trial_name = next(
            line.split(":", 1)[1].strip().strip("'").strip('"')
            for line in config_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("trial_name:")
        )
        args = [*args, f"trial_name={datetime.now():%Y%m%d_%H%M%S}_{trial_name}"]
    config, _ = load_expr_config(args, TutorConfig)
    auxiliary_model = config.auxiliary_model
    student_generalize = config.student_generalize
    teacher_pre = config.teacher_pre
    reward = config.reward
    pairwise = reward.pairwise
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    _validate_student_generalize_datasets(config, tokenizer)

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
        dataset_type=config.dataset_type,
        answer_scorer=config.answer_scorer,
        max_turns=config.max_turns,
        enable_thinking=config.enable_thinking,
        leak_handling_mode=config.leak_handling_mode,
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
        leak_penalty_mode=reward.leak_penalty_mode,
        leak_penalty_final_answer=reward.leak_penalty_final_answer,
        leak_penalty_compute=reward.leak_penalty_compute,
        leak_penalty_formula=reward.leak_penalty_formula,
        leak_penalty_aggregation=reward.leak_penalty_aggregation,
        leaked_success_reward_scale=reward.leaked_success_reward_scale,
        assign_success_reward=reward.assign_success_reward,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        enable_turn_penalty=reward.enable_turn_penalty,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        teacher_system_prompt=config.teacher_system_prompt,
        teacher_user_prompt_template=config.teacher_user_prompt_template,
        teacher_show_ground_truth=config.teacher_show_ground_truth,
        teacher_pre_enabled=teacher_pre.enabled,
        teacher_pre_mode=teacher_pre.mode,
        teacher_pre_attempts=teacher_pre.attempts,
        teacher_pre_max_tokens=teacher_pre.max_tokens,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        answer_judge_system_prompt=config.answer_judge_system_prompt,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
        max_train_sample_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        student_generalize_enabled=student_generalize.enabled,
        student_generalize_mode=student_generalize.mode,
        student_generalize_path=student_generalize.path,
        student_generalize_level1_reward=student_generalize.level1_reward,
        student_generalize_level2_reward=student_generalize.level2_reward,
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
