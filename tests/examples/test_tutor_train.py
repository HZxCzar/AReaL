import pytest

from examples.tutor.configs import (
    TutorConfig,
    TutorEvaluatorConfig,
)
from examples.tutor.train import (
    _apply_eval_average_rollouts,
    _build_eval_workflow_kwargs,
)


def test_tutor_evaluator_config_defaults_to_three_average_rollouts():
    """Test tutor validation repeats default to three rollout attempts."""
    config = TutorEvaluatorConfig()

    assert config.average_rollouts == 3


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
        "pairwise_reward_enabled": True,
        "teacher_prompt_pool_path": "teacher.json",
        "student_prompt_pool_path": "student.json",
        "teacher_warmup_enabled": True,
        "teacher_warmup_prompt_path": "warmup.txt",
        "teacher_warmup_steps": 50,
    }

    eval_workflow_kwargs = _build_eval_workflow_kwargs(workflow_kwargs, config)

    assert config.eval_gconfig.n_samples == 5
    assert eval_workflow_kwargs["gconfig"].n_samples == 1
    assert eval_workflow_kwargs["pairwise_reward_enabled"] is False
    assert eval_workflow_kwargs["teacher_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["student_prompt_pool_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_enabled"] is False
    assert eval_workflow_kwargs["teacher_warmup_prompt_path"] == ""
    assert eval_workflow_kwargs["teacher_warmup_steps"] == 0
    assert workflow_kwargs["pairwise_reward_enabled"] is True
    assert workflow_kwargs["teacher_prompt_pool_path"] == "teacher.json"
    assert workflow_kwargs["student_prompt_pool_path"] == "student.json"
    assert workflow_kwargs["teacher_warmup_enabled"] is True
    assert workflow_kwargs["teacher_warmup_prompt_path"] == "warmup.txt"
    assert workflow_kwargs["teacher_warmup_steps"] == 50
