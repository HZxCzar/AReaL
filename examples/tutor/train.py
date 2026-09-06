import pathlib
import random
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import (
    TUTOR_EVAL_STUDENT_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    TutorConfig,
    TutorEvaluatorConfig,
)
from core.generalization import (
    has_complete_generalize_case,
    load_student_generalize_bank,
)
from core.math_generalization import (
    MATH_GENERALIZATION_SAMPLE_COUNT,
    default_math_generalization_output_path,
    prepare_math_generalization_sidecar,
)
from core.polaris_generalization import (
    default_polaris_generalization_output_path,
    prepare_polaris_generalization_dataset,
)

from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.infra.data_service import RDataset
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("TutorTrain")


@dataclass(frozen=True, slots=True)
class EvalStudentPrompt:
    pool: str
    index: int
    suffix: str


def _without_remote_dataset_loading(dataset_config: Any) -> Any:
    local_config = deepcopy(dataset_config)
    local_config.scheduling_spec = None
    return local_config


def _filter_student_generalize_dataset(
    dataset: Any,
    *,
    bank: dict[str, Any],
    split_name: str,
    required_levels: tuple[str, ...],
    sample_count: int | None = None,
) -> Any:
    """Keep the rows that can serve every transfer level that is switched on.

    level1 alone keeps the rows that have variant1, level2 alone the rows that
    have variant2, both keeps the intersection. With only the original re-test
    on, required_levels is empty and nothing is filtered -- the re-test is built
    from the row's own task and needs no bank.

    A row without the variant is dropped rather than raising: it is not a broken
    row, it is a row this probe cannot run on. Losing EVERY row is still an
    error, because that is what a bank pointed at the wrong split looks like.
    """
    if not required_levels:
        return dataset
    if isinstance(dataset, RDataset) and not dataset.connected:
        logger.info(
            "student_generalize %s split uses an unconnected RDataset; deferring "
            "missing-variant skips to the tutor workflow.",
            split_name,
        )
        return dataset

    original_count = len(dataset)
    kept_indices = [
        index
        for index in range(original_count)
        if has_complete_generalize_case(
            dataset[index],
            bank,
            sample_count=sample_count,
            required_levels=required_levels,
        )
    ]

    if original_count > 0 and not kept_indices:
        raise ValueError(
            f"student_generalize: no row in the {split_name} split has a complete "
            f"case for every enabled level ({'+'.join(required_levels)}). Switch a "
            "level off, or point student_generalize.path at a bank covering this "
            "split."
        )

    logger.info(
        "student_generalize %s split: kept %d/%d rows carrying %s; dropped %d.",
        split_name,
        len(kept_indices),
        original_count,
        "+".join(required_levels),
        original_count - len(kept_indices),
    )
    if hasattr(dataset, "select"):
        return dataset.select(kept_indices)
    return [dataset[index] for index in kept_indices]


def _prepare_math_generalization_data(config: TutorConfig) -> None:
    generalize = config.student_generalize
    if (
        config.dataset_type != "math"
        or not generalize.enabled
        or generalize.source != "train"
    ):
        return

    output_path = str(
        default_math_generalization_output_path(
            config.train_dataset.path,
            seed=config.seed,
            sample_count=MATH_GENERALIZATION_SAMPLE_COUNT,
        )
    )
    valid_path = config.valid_dataset.path if config.valid_dataset is not None else None
    result = prepare_math_generalization_sidecar(
        train_dataset_path=config.train_dataset.path,
        valid_dataset_path=valid_path,
        output_path=output_path,
        seed=config.seed,
        sample_count=MATH_GENERALIZATION_SAMPLE_COUNT,
    )
    generalize.path = str(result.sidecar_path)


def _prepare_polaris_generalization_data(config: TutorConfig) -> None:
    processing = config.polaris_processing
    if config.dataset_type != "polaris" or not processing.enabled:
        return
    if not config.student_generalize.enabled:
        raise ValueError(
            "polaris_processing.enabled=true requires student_generalize.enabled=true."
        )
    valid_dataset_config = config.valid_dataset
    if (
        valid_dataset_config is not None
        and valid_dataset_config.path != config.train_dataset.path
    ):
        raise ValueError(
            "polaris_processing expects train_dataset.path and valid_dataset.path "
            "to point at the same source Polaris dataset before processing."
        )

    output_path = processing.output_path.strip()
    if not output_path:
        output_path = str(
            default_polaris_generalization_output_path(
                config.train_dataset.path,
                seed=config.seed,
                train_ratio=processing.train_ratio,
                generalize_ratio=processing.generalize_ratio,
            )
        )

    result = prepare_polaris_generalization_dataset(
        source_path=config.train_dataset.path,
        output_path=output_path,
        seed=config.seed,
        train_ratio=processing.train_ratio,
        generalize_ratio=processing.generalize_ratio,
        reuse_generalize_tasks=processing.reuse_generalize_tasks,
        overwrite=processing.overwrite,
        sidecar_filename=processing.sidecar_filename,
    )
    config.train_dataset.path = str(result.dataset_path)
    if valid_dataset_config is not None:
        valid_dataset_config.path = str(result.dataset_path)
    config.student_generalize.path = str(result.sidecar_path)


def _apply_eval_average_rollouts(config: TutorConfig) -> None:
    if config.eval_gconfig is None:
        config.eval_gconfig = config.gconfig.new()
    config.eval_gconfig = config.eval_gconfig.new(
        n_samples=config.evaluator.average_rollouts
    )


def _resolve_eval_student_names(config: TutorConfig) -> list[str]:
    configured_names = [student.name for student in config.student_models]
    requested_names = config.evaluator.student_model_names
    return configured_names if requested_names is None else list(requested_names)


def _expand_eval_dataset_for_students(dataset: Any, student_names: list[str]) -> Any:
    if not student_names:
        return dataset
    if TUTOR_EVAL_STUDENT_FIELD in dataset.column_names:
        raise ValueError(
            f"Validation dataset already contains reserved column "
            f"{TUTOR_EVAL_STUDENT_FIELD!r}."
        )

    from datasets import concatenate_datasets

    expanded = [
        dataset.add_column(TUTOR_EVAL_STUDENT_FIELD, [name] * len(dataset))
        for name in student_names
    ]
    return concatenate_datasets(expanded)


def _expand_eval_dataset_for_student_prompts(
    dataset: Any, student_prompts: tuple[EvalStudentPrompt, ...]
) -> Any:
    if not student_prompts:
        return dataset
    reserved_fields = {
        TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
        TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
    }
    collisions = sorted(reserved_fields.intersection(dataset.column_names))
    if collisions:
        raise ValueError(
            f"Validation dataset already contains reserved columns {collisions}."
        )

    from datasets import concatenate_datasets

    base_prompt_dataset = dataset.add_column(
        TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
        [None] * len(dataset),
    ).add_column(
        TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD,
        [None] * len(dataset),
    )
    persona_datasets = [
        dataset.add_column(
            TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
            [prompt.pool] * len(dataset),
        ).add_column(
            TUTOR_EVAL_STUDENT_PROMPT_INDEX_FIELD, [prompt.index] * len(dataset)
        )
        for prompt in student_prompts
    ]
    return concatenate_datasets([base_prompt_dataset, *persona_datasets])


def _load_eval_student_prompts(config: TutorConfig) -> tuple[EvalStudentPrompt, ...]:
    from examples.tutor.workflow import load_prompt_pool

    prompts = []
    for pool, path in config.prompt_pool.student_eval_paths.items():
        prompts.extend(
            EvalStudentPrompt(pool=pool, index=index, suffix=suffix)
            for index, suffix in enumerate(load_prompt_pool(path, role="student"))
        )
    return tuple(prompts)


def _build_eval_workflow_kwargs(
    workflow_kwargs: dict[str, Any], config: TutorConfig
) -> dict[str, Any]:
    if config.eval_gconfig is None:
        raise ValueError("eval_gconfig must be set before building eval workflow.")
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig.new(n_samples=1)
    eval_workflow_kwargs["eval_repeat_count"] = config.evaluator.average_rollouts
    eval_workflow_kwargs["teacher_prompt_pool_path"] = ""
    eval_paths = config.prompt_pool.student_eval_paths
    eval_workflow_kwargs["student_prompt_pool_path"] = eval_paths.get("seen", "")
    eval_workflow_kwargs["student_heldout_prompt_pool_path"] = eval_paths.get(
        "heldout", ""
    )
    eval_workflow_kwargs["student_turn_behavior_enabled"] = False
    eval_workflow_kwargs["student_turn_behavior_path"] = ""
    eval_workflow_kwargs["student_turn_behavior_separate_call_behavior_names"] = []
    eval_workflow_kwargs["teacher_warmup_enabled"] = False
    eval_workflow_kwargs["teacher_warmup_prompt_path"] = ""
    eval_workflow_kwargs["teacher_warmup_steps"] = 0
    # Evaluate under the deployment condition when asked: no checker verifies
    # the teacher's own solution before it starts teaching, so leaving this on at
    # eval measures a setting that does not exist in use. Training keeps its own
    # value, which is what keeps the training support clean.
    if config.evaluator.teacher_pre_verify is not None:
        eval_workflow_kwargs["teacher_pre_verify"] = bool(
            config.evaluator.teacher_pre_verify
        )
    # And whether it drafts at all. The config decides: an arm whose claim is that
    # the trained teacher no longer needs a draft measures that claim here, and the
    # internal skip in _presolve_unused_at_eval must not be what settles it.
    if config.evaluator.teacher_pre_enabled is not None:
        eval_workflow_kwargs["teacher_pre_enabled"] = bool(
            config.evaluator.teacher_pre_enabled
        )
    # Terminating on a leak is a training policy. Evaluating under it measures a
    # truncation that does not exist at deployment, so unless asked otherwise the
    # eval conversation runs to the budget and the train-consistent number is
    # recovered from the same rollout as a second re-test.
    if config.evaluator.leak_terminate is False:
        training_leak_mode = workflow_kwargs.get("leak_handling_mode")
        # masked_continue is already non-terminating and its hidden-history
        # semantics define the environment, so an eval request to avoid
        # termination must not turn it into reward_only and expose leaked turns.
        eval_workflow_kwargs["leak_handling_mode"] = (
            "reward_only"
            if training_leak_mode == "terminate"
            else training_leak_mode
        )
        eval_workflow_kwargs["eval_preleak_retest"] = (
            training_leak_mode == "terminate"
        )
    # Same argument for a malformed turn: 'terminate' exists to make format drift
    # expensive while learning, and nothing cuts a real conversation short because
    # the teacher mis-tagged a reply. Under 'continue' the student is handed an
    # empty message and the episode runs on, and rollout/format_errors still
    # reports the rate, so this hides nothing -- it only stops a training policy
    # from truncating the measurement. There is no 'disabled' mode to preserve, so
    # unlike the leak override this is a plain assignment.
    if config.evaluator.format_terminate is False:
        eval_workflow_kwargs["format_handling_mode"] = "continue"
    eval_workflow_kwargs["teacher_diversity_reward"] = {"enabled": False}
    eval_workflow_kwargs["teacher_context_reward"] = {"enabled": False}
    eval_workflow_kwargs["teacher_progress_judge"] = {"enabled": False}
    eval_workflow_kwargs["student_request_judge"] = {"enabled": False}
    eval_workflow_kwargs["world_model"] = {"enabled": False}
    if bool((workflow_kwargs.get("student_type_probe") or {}).get("enabled")):
        # This experiment pays the probe only during training. If evaluation is
        # later re-enabled, keep it a plain retest evaluation unless explicitly
        # redesigned.
        eval_workflow_kwargs["student_type_probe"] = {"enabled": False}
    return eval_workflow_kwargs


def _eval_repeat_count_for_item(
    item: dict[str, Any], evaluator: TutorEvaluatorConfig
) -> int:
    is_student_prompt_eval = item.get(TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD) is not None
    if is_student_prompt_eval and evaluator.student_prompt_average_rollouts is not None:
        return evaluator.student_prompt_average_rollouts
    return evaluator.average_rollouts


class _TutorEvalRepeatTrainerMixin:
    def _evaluate_fn(self, eval_workflow, eval_workflow_kwargs):
        import torch.distributed as dist

        from areal.infra.platforms import current_platform

        if self.actor.is_data_parallel_head():
            count = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=_eval_repeat_count_for_item(
                            item, self.config.evaluator
                        ),
                        is_eval=True,
                    )
                    count += 1
            self.eval_rollout.wait(count, timeout=None)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()


def main(args):
    from areal import PPOTrainer
    from areal.utils.environ import is_single_controller

    from examples.tutor.algorithm import TutorFSDPPPOActor

    class TutorPPOTrainer(_TutorEvalRepeatTrainerMixin, PPOTrainer):
        def _create_train_engine(self, actor_config, alloc):
            # Only the fsdp path is overridden: num_iterations is a tutor-local
            # addition and the other backends have no subclass carrying it.
            if (
                alloc.backend != "fsdp"
                or int(getattr(actor_config, "num_iterations", 1)) <= 1
            ):
                return super()._create_train_engine(actor_config, alloc)
            if is_single_controller():
                actor = TutorFSDPPPOActor.as_controller(actor_config, self.scheduler)
            else:
                actor = TutorFSDPPPOActor(config=actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            return actor

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
    _apply_eval_average_rollouts(config)
    auxiliary_model = config.auxiliary_model
    eval_student_prompts = _load_eval_student_prompts(config)
    student_generalize = config.student_generalize
    teacher_pre = config.teacher_pre
    reward = config.reward
    _prepare_polaris_generalization_data(config)
    _prepare_math_generalization_data(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    # Which transfer levels are switched on decides both which rows survive and
    # whether a bank is needed at all. Retest-only leaves this empty, and then
    # nothing below filters or loads a bank.
    generalize_required_levels = (
        student_generalize.transfer_levels() if student_generalize.enabled else ()
    )
    generalize_sample_count = (
        MATH_GENERALIZATION_SAMPLE_COUNT
        if student_generalize.source == "train"
        else None
    )
    student_generalize_bank = (
        load_student_generalize_bank(
            student_generalize.path,
            source=student_generalize.source,
        )
        if generalize_required_levels
        else {}
    )

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    train_dataset = _filter_student_generalize_dataset(
        train_dataset,
        bank=student_generalize_bank,
        split_name="train",
        required_levels=generalize_required_levels,
        sample_count=generalize_sample_count,
    )
    valid_dataset_config = config.valid_dataset
    eval_max_samples = config.evaluator.max_samples
    if eval_max_samples is not None:
        eval_max_samples = int(eval_max_samples)
        if eval_max_samples <= 0:
            eval_max_samples = None
    if (
        eval_max_samples is not None or config.student_models or eval_student_prompts
    ) and valid_dataset_config is not None:
        valid_dataset_config = deepcopy(valid_dataset_config)
        valid_dataset_config.scheduling_spec = None

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=valid_dataset_config,
        tokenizer=tokenizer,
    )
    valid_dataset = _filter_student_generalize_dataset(
        valid_dataset,
        bank=student_generalize_bank,
        split_name="test",
        required_levels=generalize_required_levels,
        sample_count=generalize_sample_count,
    )
    if eval_max_samples is not None and eval_max_samples < len(valid_dataset):
        rng = random.Random(config.seed)
        eval_indices = sorted(rng.sample(range(len(valid_dataset)), k=eval_max_samples))
        valid_dataset = valid_dataset.select(eval_indices)
    valid_dataset = _expand_eval_dataset_for_students(
        valid_dataset,
        _resolve_eval_student_names(config),
    )
    valid_dataset = _expand_eval_dataset_for_student_prompts(
        valid_dataset,
        eval_student_prompts,
    )

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
        student_models=[asdict(student) for student in config.student_models],
        student_sampling=asdict(config.student_sampling),
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        leak_penalty_mode=reward.leak_penalty_mode,
        leak_penalty_final_answer=reward.leak_penalty_final_answer,
        leak_penalty_compute=reward.leak_penalty_compute,
        leak_penalty_formula=reward.leak_penalty_formula,
        leak_penalty_aggregation=reward.leak_penalty_aggregation,
        turn_local_reward_components=tuple(reward.turn_local_components),
        turn_local_reward_component_placements=dict(
            reward.turn_local_component_placements
        ),
        # An omitted per-component placement is deliberately legacy-compatible.
        # It resolves to the same layer selected by the existing actor-wide mode.
        turn_local_reward_default_placement=(
            "group_norm"
            if config.actor.group_baseline_local_reward_mode == "include"
            else "pre_std"
        ),
        format_error_penalty=reward.format_error_penalty,
        teacher_exact_repeat_penalty=reward.teacher_exact_repeat_penalty,
        teacher_exact_repeat_terminate=reward.teacher_exact_repeat_terminate,
        personality_gate_terminate_penalty=(reward.personality_gate_terminate_penalty),
        personality_gate_fail_penalty=reward.personality_gate_fail_penalty,
        leaked_success_reward_scale=reward.leaked_success_reward_scale,
        assign_success_reward=reward.assign_success_reward,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        success_turn_shaping=asdict(reward.success_turn_shaping),
        max_turn_penalty=reward.max_turn_penalty,
        enable_turn_penalty=reward.enable_turn_penalty,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        soft_overlong_penalty=asdict(reward.soft_overlong),
        zero_reward_on_length_stop=reward.zero_reward_on_length_stop,
        teacher_diversity_reward=asdict(reward.teacher_diversity),
        teacher_context_reward=asdict(reward.teacher_context),
        teacher_progress_judge=asdict(reward.teacher_progress_judge),
        student_request_judge=asdict(reward.student_request_judge),
        world_model=asdict(config.world_model),
        guided_slots=asdict(config.guided_slots),
        opd=asdict(config.opd),
        prompt_instruction=asdict(config.prompt_instruction),
        free_chat=asdict(config.free_chat),
        # Reaches eval too: eval_workflow_kwargs is a copy of this dict, and the gate
        # is not training policy the way the two terminates are -- it is what makes a
        # student that student, so switching it off at eval would measure a different
        # learner.
        personality=asdict(config.personality),
        student_type_probe=asdict(config.student_type_probe),
        cross_eval=asdict(config.cross_eval),
        teacher_history_tags=config.teacher_history_tags,
        teacher_private_visibility=config.teacher_private_visibility,
        local_advantage_turn_discount=config.actor.turn_discount,
        teacher_system_prompt=config.teacher_system_prompt,
        teacher_anti_leak_instruction_enabled=(
            config.teacher_anti_leak_instruction_enabled
        ),
        teacher_adaptive_instruction_enabled=(
            config.teacher_adaptive_instruction_enabled
        ),
        teacher_prompt_pool_path=config.prompt_pool.teacher_path,
        teacher_warmup_enabled=config.prompt_pool.teacher_warmup.enabled,
        teacher_warmup_prompt_path=config.prompt_pool.teacher_warmup.prompt_path,
        teacher_warmup_steps=config.prompt_pool.teacher_warmup.steps,
        teacher_user_prompt_template=config.teacher_user_prompt_template,
        teacher_show_ground_truth=config.teacher_show_ground_truth,
        format_handling_mode=config.format_handling_mode,
        teacher_end_enabled=config.teacher_end_enabled,
        length_retry_enabled=config.length_retry.enabled,
        length_retry_attempts=config.length_retry.attempts,
        teacher_pre_enabled=teacher_pre.enabled,
        teacher_pre_mode=teacher_pre.mode,
        teacher_pre_verify=teacher_pre.verify,
        teacher_pre_attempts=teacher_pre.attempts,
        teacher_pre_max_tokens=teacher_pre.max_tokens,
        teacher_pre_visibility=teacher_pre.visibility,
        teacher_pre_on_reject=teacher_pre.on_reject,
        teacher_pre_share_per_group=teacher_pre.share_per_group,
        student_system_prompt=config.student_system_prompt,
        student_prompt_pool_path=config.prompt_pool.student_train_path,
        student_heldout_prompt_pool_path="",
        student_prompt_include_base=config.prompt_pool.include_base,
        student_turn_behavior_enabled=(
            config.prompt_pool.student_turn_behavior.enabled
        ),
        student_turn_behavior_path=config.prompt_pool.student_turn_behavior.path,
        student_turn_behavior_separate_call_behavior_names=(
            config.prompt_pool.student_turn_behavior.separate_call_behavior_names
        ),
        prompt_pool_seed=config.seed,
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
        student_generalize_source=student_generalize.source,
        student_generalize_path=student_generalize.path,
        student_generalize_replays=student_generalize.replays,
        student_generalize_turn_credit=student_generalize.turn_credit,
        student_generalize_gate_pass_credit_only=(
            student_generalize.gate_pass_credit_only
        ),
        student_generalize_turn_credit_replays=(student_generalize.turn_credit_replays),
        student_generalize_retest_original=student_generalize.retest_original,
        student_generalize_level1_enabled=student_generalize.level1_enabled,
        student_generalize_level2_enabled=student_generalize.level2_enabled,
        student_generalize_level1_reward=student_generalize.level1_reward,
        student_generalize_level2_reward=student_generalize.level2_reward,
        student_generalize_retest_reward=student_generalize.retest_reward,
        student_generalize_confidence_enabled=student_generalize.confidence.enabled,
        student_generalize_confidence_reward_scale=(
            student_generalize.confidence.reward_scale
        ),
    )

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    with TutorPPOTrainer(
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
