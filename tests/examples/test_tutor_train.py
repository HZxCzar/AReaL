import pytest

from examples.tutor.configs import (
    TutorConfig,
    TutorEvaluatorConfig,
    TutorPromptPoolConfig,
)
from examples.tutor.train import (
    _apply_eval_average_rollouts,
    _build_eval_workflow_kwargs,
)


def test_tutor_evaluator_config_defaults_to_three_average_rollouts():
    """Test tutor validation repeats default to three rollout attempts."""
    config = TutorEvaluatorConfig()

    assert config.average_rollouts == 3


def test_student_prompt_eval_coverage_defaults_disabled():
    """Test existing configs keep clean student prompts during evaluation."""
    config = TutorPromptPoolConfig()

    assert config.eval_all_student_prompts is False


def test_student_prompt_training_includes_base_by_default():
    """Test persona training treats the clean base prompt as a default option."""
    config = TutorPromptPoolConfig(student_path="student.json")

    assert config.include_base is True


def test_student_prompt_training_can_exclude_base():
    """Test configs can preserve persona-only training when explicitly requested."""
    config = TutorPromptPoolConfig(
        student_path="student.json",
        include_base=False,
    )

    assert config.include_base is False


def test_student_prompt_eval_coverage_requires_prompt_pool():
    """Test all-prompt evaluation fails fast without student prompt data."""
    with pytest.raises(ValueError, match="student_path"):
        TutorPromptPoolConfig(eval_all_student_prompts=True)


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
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert config.eval_gconfig.n_samples == 5
    assert eval_workflow_kwargs["gconfig"].n_samples == 1
    assert eval_workflow_kwargs["teacher_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_enabled"] is False
    assert eval_workflow_kwargs["teacher_warmup_prompt_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_steps"] == 0
    assert workflow_kwargs["teacher_prompt_pool_path"] == "teacher.json"
    assert workflow_kwargs["student_prompt_pool_path"] == "student.json"
    assert workflow_kwargs["teacher_warmup_enabled"] is True
    assert workflow_kwargs["teacher_warmup_prompt_path"] == "warmup.txt"
    assert workflow_kwargs["teacher_warmup_steps"] == 50


def test_build_eval_workflow_kwargs_keeps_enabled_student_prompt_pool():
    """Test all-prompt evaluation retains only the student prompt pool."""
    config = TutorConfig(
        dataset_type="math",
        prompt_pool=TutorPromptPoolConfig(
            teacher_path="teacher.json",
            student_path="student.json",
            eval_all_student_prompts=True,
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
