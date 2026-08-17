from __future__ import annotations

from pathlib import Path

import torch
from datasets import Dataset
from omegaconf import OmegaConf

from examples.pedagogical_rl.algorithm import PedagogicalDistributedSampler
from examples.pedagogical_rl.config import PedagogicalRLConfig
from examples.pedagogical_rl.train import _limit_dataset, _prepare_train_dataset

from areal.api.cli_args import to_structured_cfg


def test_formal_config_loads_locally_before_subset_selection():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "qwen3_8b_qwen3_1_7b_math_pass2_baseline.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.train_dataset.scheduling_spec is None
    assert config.train_dataset.shuffle is True
    assert config.valid_dataset.scheduling_spec is None


def test_formal_config_converts_to_pedagogical_structured_config(monkeypatch):
    monkeypatch.setenv("INF_API_KEY", "test-api-key")
    monkeypatch.setenv("TUTOR_QWEN3_1_7B_BASE_URL", "http://student.test/v1")
    monkeypatch.setenv("TUTOR_QWEN3_8B_BASE_URL", "http://judge.test/v1")
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "qwen3_8b_qwen3_1_7b_math_pass2_baseline.yaml"
    )

    config = OmegaConf.to_object(
        to_structured_cfg(OmegaConf.load(config_path), PedagogicalRLConfig)
    )

    assert isinstance(config, PedagogicalRLConfig)
    assert config.generation.leak_judge_mode == "pedagogical_rl"
    assert config.generation.number_student_attempts == 8
    assert config.teacher_pre.enabled is False
    assert config.dynamic_bs is False
    assert not hasattr(config.evaluator, "student_pre_solve")
    # The endpoints must come from the shared environment, never a hardcoded
    # URL: the tutor arm reads the same two variables, and the comparison is
    # only valid if both arms talk to the same student deployment.
    assert config.student_model.base_url == "http://student.test/v1"
    assert config.judge_model.base_url == "http://judge.test/v1"


def test_limit_dataset_selects_the_requested_prefix():
    dataset = Dataset.from_dict({"id": list(range(10))})

    limited = _limit_dataset(dataset, 4)

    assert limited["id"] == [0, 1, 2, 3]


def test_prepare_train_dataset_selects_head_before_seeded_shuffle():
    dataset = Dataset.from_dict({"id": list(range(10))})

    prepared = _prepare_train_dataset(dataset, 4, seed=42)

    assert sorted(prepared["id"]) == [0, 1, 2, 3]
    assert prepared["id"] == _limit_dataset(dataset, 4).shuffle(seed=42)["id"]


def test_pedagogical_sampler_matches_stateful_seeded_epoch_permutations():
    dataset = Dataset.from_dict({"id": list(range(8))})
    sampler = PedagogicalDistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=42,
        drop_last=True,
    )

    generator = torch.Generator().manual_seed(42)
    expected_epoch_0 = torch.randperm(8, generator=generator).tolist()
    expected_epoch_1 = torch.randperm(8, generator=generator).tolist()

    assert list(sampler) == expected_epoch_0
    sampler.set_epoch(1)
    assert list(sampler) == expected_epoch_1
