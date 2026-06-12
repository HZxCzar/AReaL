# SPDX-License-Identifier: Apache-2.0

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

from datasets import Dataset
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import ValidDatasetConfig, _DatasetConfig


class DifficultyBalancedDistributedSampler:
    """Distributed sampler that keeps each global batch difficulty-balanced."""

    def __init__(
        self,
        dataset: Any,
        num_replicas: int,
        rank: int,
        *,
        global_batch_size: int,
        label_field: str,
        ratios: dict[str, float],
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if global_batch_size <= 0:
            raise ValueError(
                f"global_batch_size must be positive, got {global_batch_size}"
            )
        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"global_batch_size({global_batch_size}) must be divisible by "
                f"num_replicas({num_replicas})"
            )

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.global_batch_size = int(global_batch_size)
        self.label_field = label_field
        self.ratios = {str(k): float(v) for k, v in ratios.items()}
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.samples_per_label = _scale_label_weights(
            self.ratios,
            self.global_batch_size,
        )
        self.label_to_indices = _group_indices_by_label(dataset, label_field)
        self.active_labels = [
            label for label, count in self.samples_per_label.items() if count > 0
        ]
        missing_labels = [
            label for label in self.active_labels if not self.label_to_indices.get(label)
        ]
        if missing_labels:
            available = sorted(self.label_to_indices)
            raise ValueError(
                "difficulty_balance requested labels with no samples: "
                f"{missing_labels}. Available labels: {available}"
            )

        eligible_count = sum(len(self.label_to_indices[label]) for label in self.active_labels)
        if eligible_count <= 0:
            raise ValueError("difficulty_balance found no eligible samples to draw from")
        if self.drop_last:
            self.num_global_batches = max(1, eligible_count // self.global_batch_size)
        else:
            self.num_global_batches = max(1, math.ceil(eligible_count / self.global_batch_size))
        self.total_size = self.num_global_batches * self.global_batch_size
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        pools: dict[str, list[int]] = {}
        cursors: dict[str, int] = {}
        for label in self.active_labels:
            pool = list(self.label_to_indices[label])
            if self.shuffle:
                rng.shuffle(pool)
            pools[label] = pool
            cursors[label] = 0

        global_batches: list[list[int]] = []
        for _ in range(self.num_global_batches):
            batch: list[int] = []
            for label, count in self.samples_per_label.items():
                if count <= 0:
                    continue
                batch.extend(_take_from_pool(pools[label], cursors, label, count, rng, self.shuffle))
            if self.shuffle:
                rng.shuffle(batch)
            global_batches.append(batch)

        if self.shuffle:
            rng.shuffle(global_batches)
        indices = [idx for batch in global_batches for idx in batch]
        if len(indices) != self.total_size:
            raise RuntimeError(
                f"difficulty-balanced sampler built {len(indices)} indices, "
                f"expected {self.total_size}"
            )
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _scale_label_weights(
    ratios: dict[str, float],
    global_batch_size: int,
) -> dict[str, int]:
    if not ratios:
        raise ValueError("difficulty_balance.ratios must not be empty")
    positive_items: list[tuple[str, float]] = []
    result = {str(label): 0 for label in ratios}
    for label, weight in ratios.items():
        weight = float(weight)
        if weight < 0:
            raise ValueError(
                f"difficulty_balance.ratios['{label}'] must be non-negative, got {weight}"
            )
        if weight > 0:
            positive_items.append((str(label), weight))
    total_weight = sum(weight for _, weight in positive_items)
    if total_weight <= 0:
        raise ValueError(
            "difficulty_balance.ratios must contain at least one positive weight"
        )

    fractional: list[tuple[float, str]] = []
    assigned = 0
    for label, weight in positive_items:
        raw = global_batch_size * weight / total_weight
        count = math.floor(raw)
        result[label] = count
        assigned += count
        fractional.append((raw - count, label))

    remainder = global_batch_size - assigned
    fractional.sort(key=lambda item: (-item[0], item[1]))
    for i in range(remainder):
        label = fractional[i % len(fractional)][1]
        result[label] += 1

    if sum(result.values()) != global_batch_size:
        raise RuntimeError("scaled difficulty ratios do not sum to global batch size")
    return result


def _get_nested_value(row: Any, dotted_field: str, index: int) -> Any:
    value = row
    for part in dotted_field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(
                f"Dataset row {index} is missing difficulty field '{dotted_field}'. "
                "Run difficulty labeling first or disable difficulty_balance."
            )
    return value


def _group_indices_by_label(dataset: Any, label_field: str) -> dict[str, list[int]]:
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    missing_examples: list[int] = []
    for index in range(len(dataset)):
        row = dataset[index]
        try:
            label = _get_nested_value(row, label_field, index)
        except KeyError:
            missing_examples.append(index)
            continue
        if label is None or str(label) == "":
            missing_examples.append(index)
            continue
        label_to_indices[str(label)].append(index)

    if missing_examples:
        examples = missing_examples[:10]
        raise ValueError(
            f"difficulty_balance requires field '{label_field}' on every train row; "
            f"missing/empty at indices {examples}"
        )
    return dict(label_to_indices)


def _take_from_pool(
    pool: list[int],
    cursors: dict[str, int],
    label: str,
    count: int,
    rng: random.Random,
    shuffle: bool,
) -> list[int]:
    if not pool:
        raise ValueError(f"difficulty_balance label '{label}' has no samples")
    selected: list[int] = []
    while len(selected) < count:
        cursor = cursors[label]
        remaining = count - len(selected)
        available = len(pool) - cursor
        take = min(remaining, available)
        if take > 0:
            selected.extend(pool[cursor : cursor + take])
            cursors[label] = cursor + take
        if len(selected) < count:
            if shuffle:
                rng.shuffle(pool)
            cursors[label] = 0
    return selected


def create_dataloader(
    dataset,
    rank: int,
    world_size: int,
    dataset_config: _DatasetConfig,
    collate_fn: Callable | None = None,
) -> StatefulDataLoader:
    """Create a stateful dataloader for a dataset with distributed sampler.

    Args:
        dataset: The dataset to create a dataloader for.
        rank: The rank of the process.
        world_size: The world size.
        dataset_config: The dataset config.
        collate_fn: The collate function to use.
    """
    if dataset_config.batch_size % world_size != 0:
        raise ValueError(
            f"batch size({dataset_config.batch_size}) must be divisible by world_size({world_size})!"
        )

    from areal.infra.data_service.rdataset import RDataset, _PrefetchAwareSampler

    drop_sampler_last = True
    if isinstance(dataset_config, ValidDatasetConfig):
        drop_sampler_last = False

    balance_config = getattr(dataset_config, "difficulty_balance", None)
    balance_enabled = bool(getattr(balance_config, "enabled", False))
    if balance_enabled and isinstance(dataset_config, ValidDatasetConfig):
        raise ValueError("difficulty_balance is only supported for train_dataset")

    if balance_enabled:
        if isinstance(dataset, RDataset):
            raise ValueError(
                "difficulty_balance requires a local dataset with readable metadata; "
                "set train_dataset.scheduling_spec=null to disable RDataset/data service."
            )
        sampler = DifficultyBalancedDistributedSampler(
            dataset,
            world_size,
            rank,
            global_batch_size=dataset_config.batch_size,
            label_field=balance_config.label_field,
            ratios=balance_config.ratios,
            shuffle=dataset_config.shuffle,
            drop_last=dataset_config.drop_last,
        )
    else:
        if isinstance(dataset, RDataset) and isinstance(dataset_config, ValidDatasetConfig):
            sampler_cls = _PrefetchAwareEvalSampler
        elif isinstance(dataset, RDataset):
            sampler_cls = _PrefetchAwareSampler
        elif isinstance(dataset_config, ValidDatasetConfig):
            sampler_cls = EvalDistributedSampler
        else:
            sampler_cls = DistributedSampler
        sampler = sampler_cls(
            dataset,
            world_size,
            rank,
            shuffle=dataset_config.shuffle,
            drop_last=drop_sampler_last,
        )

    return StatefulDataLoader(
        dataset,
        batch_size=dataset_config.batch_size // world_size,
        sampler=sampler,
        drop_last=dataset_config.drop_last,
        num_workers=dataset_config.num_workers,
        collate_fn=collate_fn or (lambda x: x),
    )


class EvalDistributedSampler(DistributedSampler):
    r"""A DistributedSampler specifically designed for evaluation (Validation/Testing).

    Unlike the standard :class:`~torch.utils.data.DistributedSampler`, this sampler
    **does not pad** the dataset to make it evenly divisible by the number of replicas.

    In the standard implementation, extra indices are added to ensure every rank has
    the exact same `num_samples`. While useful for training (synchronized batch sizes),
    this causes some validation samples to be evaluated twice, leading to biased metrics
    (e.g., inaccurate accuracy or loss).

    **Key Behaviors:**
    1. **Exact Evaluation:** Ensures every sample in the dataset is evaluated exactly
       once across the entire cluster.
    2. **Uneven Split:** Ranks may receive different amounts of data (difference is at most 1).
       For example, if N=10 and Replicas=3, ranks get [4, 3, 3] samples respectively,
       instead of [4, 4, 4] with padding.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if not drop_last:
            self.total_size = len(dataset)

        if self.rank + (self.num_samples - 1) * self.num_replicas >= self.total_size:
            self.num_samples -= 1


class _PrefetchAwareEvalSampler(EvalDistributedSampler):
    def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset, *args, **kwargs)
        self._rdataset = dataset
        self._trigger_prefetch()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._trigger_prefetch()

    def _trigger_prefetch(self) -> None:
        self._rdataset._start_prefetch(list(super().__iter__()))
