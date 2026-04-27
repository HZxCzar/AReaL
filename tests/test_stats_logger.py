"""Tests for local StatsLogger persistence."""

import json
from unittest.mock import MagicMock, patch


def _make_test_config(fileroot):
    """Create a minimal BaseExperimentConfig for testing StatsLogger."""
    from areal.api.cli_args import BaseExperimentConfig

    config = BaseExperimentConfig(
        experiment_name="test_exp",
        trial_name="trial_0",
        total_train_epochs=1,
    )
    config.stats_logger.experiment_name = "test_exp"
    config.stats_logger.trial_name = "trial_0"
    config.stats_logger.fileroot = str(fileroot)
    return config


def _make_ft_spec():
    """Create a mock FinetuneSpec for testing."""
    from areal.api import FinetuneSpec

    ft_spec = MagicMock(spec=FinetuneSpec)
    ft_spec.total_train_epochs = 1
    ft_spec.steps_per_epoch = 10
    ft_spec.total_train_steps = 10
    return ft_spec


@patch("areal.utils.stats_logger.trackio")
@patch("areal.utils.stats_logger.wandb")
@patch("areal.utils.stats_logger.swanlab")
@patch("areal.utils.stats_logger.dist")
def test_commit_writes_local_jsonl_diagnostics(
    mock_dist, mock_swanlab, mock_wandb, mock_trackio, tmp_path
):
    """commit() should persist full metrics and PPO diagnostics locally."""
    mock_dist.is_initialized.return_value = False

    from areal.utils.stats_logger import StatsLogger

    config = _make_test_config(tmp_path)
    logger = StatsLogger(config, _make_ft_spec())

    data = {
        "ppo_actor/update/entropy/avg": 1.2,
        "ppo_actor/update/clip_ratio/avg": 0.3,
        "ppo_actor/update/behave_imp_weight/min": 0.01,
        "ppo_actor/final_reward/avg": 0.4,
        "unrelated_metric": 9.0,
        "counter__count": 1,
    }
    logger.commit(epoch=0, step=1, global_step=2, data=data)

    with open(logger.metrics_jsonl_path, encoding="utf-8") as f:
        metrics_record = json.loads(f.readline())
    assert metrics_record["epoch"] == 0
    assert metrics_record["epoch_step"] == 1
    assert metrics_record["global_step"] == 2
    assert metrics_record["stats"]["unrelated_metric"] == 9.0
    assert "counter__count" not in metrics_record["stats"]

    with open(logger.ppo_diagnostics_jsonl_path, encoding="utf-8") as f:
        diagnostics_record = json.loads(f.readline())
    assert diagnostics_record["diagnostics"]["ppo_actor/update/clip_ratio/avg"] == 0.3
    assert diagnostics_record["derived"]["entropy_avg"] == 1.2
    assert diagnostics_record["derived"]["behave_imp_weight_min"] == 0.01
    assert "unrelated_metric" not in diagnostics_record["diagnostics"]
