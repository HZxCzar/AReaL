"""CPU DCP recovery checks against uninterrupted LR and optimizer updates."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.utils import logging


def make_engine(schedule="constant", warmup=4):
    from areal.engine.fsdp_engine import FSDPEngine

    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model = nn.Linear(2, 1)
    engine.optimizer = torch.optim.SGD(engine.model.parameters(), lr=0.1, momentum=0.9)

    def factor(step):
        if step < warmup:
            return step / warmup
        if schedule == "linear":
            return max(0.0, (20 - step) / (20 - warmup))
        return 1.0

    engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, factor)
    engine.config = SimpleNamespace(use_lora=False, num_iterations=1)
    engine._separate_lora_enabled = False
    engine.logger = logging.getLogger("SchedulerRecoveryTest")
    return engine


def update(engine):
    lr = engine.lr_scheduler.get_last_lr()
    assert lr == [group["lr"] for group in engine.optimizer.param_groups]
    engine.optimizer.zero_grad()
    engine.model(torch.ones(1, 2)).sum().backward()
    engine.optimizer.step()
    engine.lr_scheduler_step()
    return lr


@pytest.mark.parametrize("schedule", ["constant", "linear"])
@pytest.mark.parametrize("completed", [2, 8])
def test_single_scheduler_dcp_resume_matches_uninterrupted(
    tmp_path, schedule, completed
):
    """Selected earlier snapshots restore LR and momentum, including during warmup."""
    original = make_engine(schedule)
    for _ in range(completed):
        update(original)
    selected = str(tmp_path / "selected")
    original._save_to_dcp(selected, with_optim=True)
    expected = []
    for i in range(6):
        expected.append(update(original))
        if i == 2:
            original._save_to_dcp(str(tmp_path / "newer"), with_optim=True)

    resumed = make_engine(schedule)
    resumed._load_from_dcp(selected, with_optim=True)
    assert resumed.lr_scheduler.last_epoch == completed
    assert [update(resumed) for _ in range(6)] == expected
    assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
    for actual, reference in zip(
        resumed.model.parameters(), original.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("schedule", ["constant", "linear"])
@pytest.mark.parametrize("completed", [2, 8])
def test_legacy_recovery_reconstructs_selected_generation(
    tmp_path, schedule, completed
):
    """Old model/optimizer-only generations resume without repeating warmup."""
    original = make_engine(schedule)
    for _ in range(completed):
        update(original)
    generation = tmp_path / "selected-generation"
    checkpoint = generation / "checkpoints" / "default"
    dcp.save(
        {"dcp": DCPState(original.model, original.optimizer)}, checkpoint_id=checkpoint
    )
    info = generation / "recover_info"
    info.mkdir()
    (info / "step_info.json").write_text(json.dumps({"global_step": completed - 1}))
    resumed = make_engine(schedule)
    resumed._load_from_dcp(str(checkpoint), with_optim=True)
    assert resumed.lr_scheduler.last_epoch == completed
    for _ in range(6):
        assert update(resumed) == update(original)
    for actual, reference in zip(
        resumed.model.parameters(), original.model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_legacy_checkpoint_without_step_info_fails_clearly(tmp_path):
    """Do not silently reset the scheduler for arbitrary old DCP directories."""
    engine = make_engine()
    dcp.save({"dcp": DCPState(engine.model, engine.optimizer)}, checkpoint_id=tmp_path)
    with pytest.raises(ValueError, match="no LR scheduler state or recovery step_info"):
        engine._load_from_dcp(str(tmp_path), with_optim=True)


def test_legacy_inconsistent_lr_fails_instead_of_guessing(tmp_path):
    """A previously restarted warmup cannot be inferred from global step alone."""
    engine = make_engine()
    update(engine)
    checkpoint = tmp_path / "checkpoints" / "default"
    dcp.save(
        {"dcp": DCPState(engine.model, engine.optimizer)}, checkpoint_id=checkpoint
    )
    (tmp_path / "recover_info").mkdir()
    (tmp_path / "recover_info" / "step_info.json").write_text('{"global_step": 9}')
    with pytest.raises(ValueError, match="disagrees with the configured schedule"):
        make_engine()._load_from_dcp(str(checkpoint), with_optim=True)


def test_weights_only_dcp_does_not_restore_scheduler(tmp_path):
    """with_optim=False keeps the receiving scheduler and optimizer unchanged."""
    original = make_engine()
    for _ in range(8):
        update(original)
    original._save_to_dcp(str(tmp_path), with_optim=False)
    resumed = make_engine()
    resumed._load_from_dcp(str(tmp_path), with_optim=False)
    assert resumed.lr_scheduler.last_epoch == 0
    assert resumed.lr_scheduler.get_last_lr() == [0.0]


def test_legacy_multiple_iterations_fails_without_guessing_step_count(tmp_path):
    """Global steps alone cannot identify a custom actor's scheduler frequency."""
    engine = make_engine()
    dcp.save({"dcp": DCPState(engine.model, engine.optimizer)}, checkpoint_id=tmp_path)
    engine.config.num_iterations = 2
    with pytest.raises(ValueError, match="scheduler steps per iteration are ambiguous"):
        engine._load_from_dcp(str(tmp_path), with_optim=True)


def test_new_checkpoint_preserves_multiple_scheduler_steps_per_iteration(tmp_path):
    """New checkpoints restore the saved counter without inferring global steps."""
    original = make_engine()
    original.config.num_iterations = 2
    for _ in range(6):
        update(original)
    original._save_to_dcp(str(tmp_path), with_optim=True)
    resumed = make_engine()
    resumed.config.num_iterations = 2
    resumed._load_from_dcp(str(tmp_path), with_optim=True)
    assert resumed.lr_scheduler.last_epoch == 6
    assert update(resumed) == update(original)
