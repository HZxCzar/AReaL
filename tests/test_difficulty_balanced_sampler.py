from collections import Counter

import pytest
from torch.utils.data import DistributedSampler

from areal.api.cli_args import DifficultyBalanceConfig, TrainDatasetConfig
from areal.infra.data_service.rdataset import RDataset
from areal.utils.dataloader import (
    DifficultyBalancedDistributedSampler,
    create_dataloader,
)


def make_rows(labels: list[str]) -> list[dict]:
    return [
        {"id": str(i), "metadata": {"difficulty_label": label}}
        for i, label in enumerate(labels)
    ]


def labels_for_indices(dataset: list[dict], indices: list[int]) -> Counter[str]:
    return Counter(dataset[index]["metadata"]["difficulty_label"] for index in indices)


def global_batches(
    dataset: list[dict],
    *,
    world_size: int,
    global_batch_size: int,
    ratios: dict[str, float],
) -> list[list[int]]:
    samplers = [
        DifficultyBalancedDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            global_batch_size=global_batch_size,
            label_field="metadata.difficulty_label",
            ratios=ratios,
            shuffle=False,
            drop_last=True,
        )
        for rank in range(world_size)
    ]
    rank_indices = [list(iter(sampler)) for sampler in samplers]
    local_batch_size = global_batch_size // world_size
    num_batches = len(rank_indices[0]) // local_batch_size
    batches = []
    for batch_idx in range(num_batches):
        batch = []
        for rank in range(world_size):
            start = batch_idx * local_batch_size
            end = start + local_batch_size
            batch.extend(rank_indices[rank][start:end])
        batches.append(batch)
    return batches


def test_difficulty_balanced_sampler_keeps_global_batch_ratio() -> None:
    dataset = make_rows(
        ["easy"] * 8 + ["medium"] * 16 + ["hard"] * 8 + ["noisy"] * 4
    )

    batches = global_batches(
        dataset,
        world_size=2,
        global_batch_size=8,
        ratios={"easy": 2, "medium": 4, "hard": 2, "noisy": 0},
    )

    assert batches
    for batch in batches:
        assert labels_for_indices(dataset, batch) == {
            "easy": 2,
            "medium": 4,
            "hard": 2,
        }


def test_difficulty_balanced_sampler_repeats_small_buckets() -> None:
    dataset = make_rows(["easy"] * 2 + ["medium"] * 3 + ["hard"] * 2)

    batches = global_batches(
        dataset,
        world_size=1,
        global_batch_size=8,
        ratios={"easy": 2, "medium": 4, "hard": 2, "noisy": 0},
    )

    assert len(batches) == 1
    assert labels_for_indices(dataset, batches[0]) == {
        "easy": 2,
        "medium": 4,
        "hard": 2,
    }


def test_difficulty_balanced_sampler_requires_labels() -> None:
    dataset = [{"id": "0", "metadata": {}}]

    with pytest.raises(ValueError, match="requires field 'metadata.difficulty_label'"):
        DifficultyBalancedDistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            global_batch_size=1,
            label_field="metadata.difficulty_label",
            ratios={"easy": 1},
        )


def test_create_dataloader_uses_default_sampler_when_disabled() -> None:
    dataset = make_rows(["easy", "medium"])
    cfg = TrainDatasetConfig(path="dummy", type="rl", batch_size=2)

    dataloader = create_dataloader(dataset, rank=0, world_size=1, dataset_config=cfg)

    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not isinstance(dataloader.sampler, DifficultyBalancedDistributedSampler)


def test_create_dataloader_rejects_rdataset_when_balance_enabled() -> None:
    cfg = TrainDatasetConfig(
        path="dummy",
        type="rl",
        batch_size=2,
        difficulty_balance=DifficultyBalanceConfig(enabled=True),
    )

    with pytest.raises(ValueError, match="train_dataset.scheduling_spec=null"):
        create_dataloader(
            RDataset(path="dummy", type="rl", split="train"),
            rank=0,
            world_size=1,
            dataset_config=cfg,
        )
