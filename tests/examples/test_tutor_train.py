from types import SimpleNamespace

import pytest

from examples.tutor.configs import (
    TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD,
    TutorConfig,
    TutorEvaluatorConfig,
    TutorPromptPoolConfig,
    TutorStudentTurnBehaviorConfig,
)
from examples.tutor.train import (
    _apply_eval_average_rollouts,
    _build_eval_workflow_kwargs,
    _eval_repeat_count_for_item,
    _TutorEvalRepeatTrainerMixin,
)


def test_tutor_evaluator_config_defaults_to_three_average_rollouts():
    """Test tutor validation repeats default to three rollout attempts."""
    config = TutorEvaluatorConfig()

    assert config.average_rollouts == 3
    assert config.student_prompt_average_rollouts is None


def test_student_prompt_average_rollouts_rejects_non_positive_values():
    """Test additional student prompt repeats must be positive when configured."""
    with pytest.raises(ValueError, match="student_prompt_average_rollouts"):
        TutorEvaluatorConfig(student_prompt_average_rollouts=0)


def test_eval_repeat_count_distinguishes_base_seen_and_heldout_prompts():
    """Test only additional student prompt rows use their repeat override."""
    evaluator = TutorEvaluatorConfig(
        average_rollouts=3,
        student_prompt_average_rollouts=1,
    )

    assert _eval_repeat_count_for_item({}, evaluator) == 3
    assert (
        _eval_repeat_count_for_item(
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: None}, evaluator
        )
        == 3
    )
    assert (
        _eval_repeat_count_for_item(
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "seen"}, evaluator
        )
        == 1
    )
    assert (
        _eval_repeat_count_for_item(
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "heldout"}, evaluator
        )
        == 1
    )


def test_student_prompt_repeat_count_defaults_to_base_repeat_count():
    """Test unset student prompt repeats preserve the previous behavior."""
    evaluator = TutorEvaluatorConfig(average_rollouts=4)

    assert (
        _eval_repeat_count_for_item(
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "seen"}, evaluator
        )
        == 4
    )


def test_tutor_eval_submits_base_and_student_prompts_with_separate_repeats(
    monkeypatch,
):
    """Test tutor evaluation passes the per-row repeat count to rollout submission."""
    import torch.distributed as dist

    from areal.infra.platforms import current_platform

    submitted_group_sizes = []

    class FakeEvalRollout:
        def submit(self, _item, _workflow, _kwargs, *, group_size, is_eval):
            assert is_eval is True
            submitted_group_sizes.append(group_size)

        def wait(self, count, timeout):
            assert count == 3
            assert timeout is None

    class FakeTrainer(_TutorEvalRepeatTrainerMixin):
        pass

    monkeypatch.setattr(dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(current_platform, "synchronize", lambda: None)
    trainer = FakeTrainer()
    trainer.actor = SimpleNamespace(
        is_data_parallel_head=lambda: True,
        cpu_group=None,
    )
    trainer.config = SimpleNamespace(
        evaluator=TutorEvaluatorConfig(
            average_rollouts=3,
            student_prompt_average_rollouts=1,
        )
    )
    trainer.valid_dataloader = [
        [
            {},
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "seen"},
            {TUTOR_EVAL_STUDENT_PROMPT_GROUP_FIELD: "heldout"},
        ]
    ]
    trainer.eval_rollout = FakeEvalRollout()

    trainer._evaluate_fn("workflow", {"key": "value"})

    assert submitted_group_sizes == [3, 1, 1]


def test_student_personas_are_evaluated_by_default():
    """Test existing configs keep exhaustive persona evaluation by default."""
    config = TutorPromptPoolConfig(student_seen_path="student.json")

    assert config.test_persona is True
    assert config.student_eval_paths == {"seen": "student.json"}


def test_student_persona_evaluation_can_be_disabled():
    """Test disabling persona evaluation leaves only the clean base prompt."""
    config = TutorPromptPoolConfig(
        student_seen_path="student.json",
        student_heldout_path="heldout.json",
        test_persona=False,
    )

    assert config.student_train_path == "student.json"
    assert config.student_eval_paths == {}


def test_student_prompt_training_includes_base_by_default():
    """Test persona training treats the clean base prompt as a default option."""
    config = TutorPromptPoolConfig(student_seen_path="student.json")

    assert config.include_base is True


def test_student_prompt_training_can_exclude_base():
    """Test configs can preserve persona-only training when explicitly requested."""
    config = TutorPromptPoolConfig(
        student_seen_path="student.json",
        include_base=False,
    )

    assert config.include_base is False


def test_split_student_prompt_paths_define_train_and_eval_pools():
    """Test seen personas train while seen and held-out personas both evaluate."""
    config = TutorPromptPoolConfig(
        student_seen_path="seen.json",
        student_heldout_path="heldout.json",
    )

    assert config.student_train_path == "seen.json"
    assert config.student_eval_paths == {
        "seen": "seen.json",
        "heldout": "heldout.json",
    }


def test_heldout_student_prompt_path_requires_seen_pool():
    """Test held-out personas cannot accidentally become the training pool."""
    with pytest.raises(ValueError, match="student_seen_path"):
        TutorPromptPoolConfig(student_heldout_path="heldout.json")


@pytest.mark.parametrize("average_rollouts", [0, -1])
def test_tutor_evaluator_config_rejects_non_positive_average_rollouts(
    average_rollouts,
):
    """Test tutor validation repeat count must be positive."""
    with pytest.raises(ValueError, match="average_rollouts"):
        TutorEvaluatorConfig(average_rollouts=average_rollouts)


def test_apply_eval_average_rollouts_sets_eval_group_size():
    """Test average_rollouts drives PPOTrainer eval group size."""
    config = TutorConfig(
        dataset_type="math",
        evaluator=TutorEvaluatorConfig(average_rollouts=5),
    )
    config.eval_gconfig = config.eval_gconfig.new(n_samples=1)

    _apply_eval_average_rollouts(config)

    assert config.eval_gconfig.n_samples == 5


def test_build_eval_workflow_kwargs_keeps_single_episode_generation():
    """Test repeated validation happens through group size, not workflow sampling."""
    config = TutorConfig(
        dataset_type="math",
        evaluator=TutorEvaluatorConfig(average_rollouts=5),
    )
    _apply_eval_average_rollouts(config)
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "teacher_prompt_pool_path": "teacher.json",
        "student_prompt_pool_path": "student.json",
        "teacher_warmup_enabled": True,
        "teacher_warmup_prompt_path": "warmup.txt",
        "teacher_warmup_steps": 50,
        "teacher_diversity_reward": {"enabled": True, "weight": 2.0},
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert config.eval_gconfig.n_samples == 5
    assert eval_workflow_kwargs["gconfig"].n_samples == 1
    assert eval_workflow_kwargs["eval_repeat_count"] == 5
    assert eval_workflow_kwargs["teacher_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_turn_behavior_enabled"] is False
    assert eval_workflow_kwargs["student_turn_behavior_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_enabled"] is False
    assert eval_workflow_kwargs["teacher_warmup_prompt_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_steps"] == 0
    assert eval_workflow_kwargs["teacher_diversity_reward"] == {"enabled": False}
    assert eval_workflow_kwargs["teacher_progress_judge"] == {"enabled": False}
    assert workflow_kwargs["teacher_prompt_pool_path"] == "teacher.json"
    assert workflow_kwargs["student_prompt_pool_path"] == "student.json"
    assert workflow_kwargs["teacher_warmup_enabled"] is True
    assert workflow_kwargs["teacher_warmup_prompt_path"] == "warmup.txt"
    assert workflow_kwargs["teacher_warmup_steps"] == 50
    assert workflow_kwargs["teacher_diversity_reward"] == {
        "enabled": True,
        "weight": 2.0,
    }


def test_build_eval_workflow_kwargs_disables_turn_behaviors():
    """Test validation stays deterministic when train-only behaviors are enabled."""
    config = TutorConfig(
        dataset_type="math",
        prompt_pool=TutorPromptPoolConfig(
            student_turn_behavior=TutorStudentTurnBehaviorConfig(
                enabled=True,
                path="turn-behaviors.json",
            )
        ),
    )
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "student_turn_behavior_enabled": True,
        "student_turn_behavior_path": "turn-behaviors.json",
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert workflow_kwargs["student_turn_behavior_enabled"] is True
    assert workflow_kwargs["student_turn_behavior_path"] == "turn-behaviors.json"
    assert eval_workflow_kwargs["student_turn_behavior_enabled"] is False
    assert eval_workflow_kwargs["student_turn_behavior_path"] == ""


def test_build_eval_workflow_kwargs_keeps_seen_student_prompt_pool():
    """Test evaluation retains the seen student prompt pool by default."""
    config = TutorConfig(
        dataset_type="math",
        prompt_pool=TutorPromptPoolConfig(
            teacher_path="teacher.json",
            student_seen_path="student.json",
        ),
    )
    _apply_eval_average_rollouts(config)
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "teacher_prompt_pool_path": "teacher.json",
        "student_prompt_pool_path": "student.json",
        "teacher_warmup_enabled": True,
        "teacher_warmup_prompt_path": "warmup.txt",
        "teacher_warmup_steps": 50,
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert eval_workflow_kwargs["teacher_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_prompt_pool_path"] == "student.json"
    assert eval_workflow_kwargs["student_heldout_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_enabled"] is False
    assert eval_workflow_kwargs["teacher_warmup_prompt_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_steps"] == 0


def test_build_eval_workflow_kwargs_loads_seen_and_heldout_student_pools():
    """Test split persona config exposes both pools only to evaluation."""
    config = TutorConfig(
        dataset_type="math",
        prompt_pool=TutorPromptPoolConfig(
            student_seen_path="seen.json",
            student_heldout_path="heldout.json",
        ),
    )
    _apply_eval_average_rollouts(config)
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "teacher_prompt_pool_path": "teacher.json",
        "student_prompt_pool_path": "seen.json",
        "student_heldout_prompt_pool_path": "",
        "teacher_warmup_enabled": False,
        "teacher_warmup_prompt_path": "",
        "teacher_warmup_steps": 0,
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert eval_workflow_kwargs["student_prompt_pool_path"] == "seen.json"
    assert eval_workflow_kwargs["student_heldout_prompt_pool_path"] == "heldout.json"
    assert workflow_kwargs["student_heldout_prompt_pool_path"] == ""


def test_build_eval_workflow_kwargs_disables_student_persona_pools():
    """Test base-only evaluation does not load seen or held-out persona pools."""
    config = TutorConfig(
        dataset_type="math",
        prompt_pool=TutorPromptPoolConfig(
            student_seen_path="seen.json",
            student_heldout_path="heldout.json",
            test_persona=False,
        ),
    )
    _apply_eval_average_rollouts(config)
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "teacher_prompt_pool_path": "teacher.json",
        "student_prompt_pool_path": "seen.json",
        "student_heldout_prompt_pool_path": "",
        "teacher_warmup_enabled": False,
        "teacher_warmup_prompt_path": "",
        "teacher_warmup_steps": 0,
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert eval_workflow_kwargs["student_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_heldout_prompt_pool_path"] == ""
    assert workflow_kwargs["student_prompt_pool_path"] == "seen.json"
