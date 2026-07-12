# SPDX-License-Identifier: Apache-2.0

from areal.api import FinetuneSpec
from areal.api.cli_args import EvaluatorConfig
from areal.utils.evaluator import Evaluator


def _evaluator(*, eval_before_train: bool = True, freq_steps: int = 10) -> Evaluator:
    config = EvaluatorConfig(
        eval_before_train=eval_before_train,
        freq_steps=freq_steps,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=10,
        train_batch_size=1,
    )
    return Evaluator(config, ft_spec)


def test_evaluator_config_enables_pretrain_evaluation_by_default():
    """Test fresh runs evaluate model version 0 by default."""
    assert EvaluatorConfig().eval_before_train is True


def test_evaluate_before_train_fresh_run_executes_without_consuming_schedule():
    """Test version-0 evaluation preserves the regular step schedule."""
    evaluator = _evaluator()
    calls = []

    evaluated = evaluator.evaluate_before_train(
        lambda: calls.append("initial"),
        start_step=0,
    )
    for _ in range(9):
        evaluator.evaluate(lambda: calls.append("scheduled"), 0, 0, 0)

    assert evaluated is True
    assert calls == ["initial"]

    evaluator.evaluate(lambda: calls.append("scheduled"), 0, 0, 0)

    assert calls == ["initial", "scheduled"]


def test_evaluate_before_train_disabled_does_not_execute():
    """Test users can disable version-0 evaluation explicitly."""
    evaluator = _evaluator(eval_before_train=False)
    calls = []

    evaluated = evaluator.evaluate_before_train(
        lambda: calls.append("initial"),
        start_step=0,
    )

    assert evaluated is False
    assert calls == []


def test_evaluate_before_train_resume_does_not_execute():
    """Test checkpoint recovery does not repeat version-0 evaluation."""
    evaluator = _evaluator()
    calls = []

    evaluated = evaluator.evaluate_before_train(
        lambda: calls.append("initial"),
        start_step=5,
    )

    assert evaluated is False
    assert calls == []
