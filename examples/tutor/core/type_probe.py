"""Pure utilities for the teacher's final student-type probe.

The probe is a sidecar: its prompt and sampled answer never enter the teaching
trajectory or the tensors trained by PPO. Behavior and information are read as
separate axes. Non-singleton axes are probed as multiple-choice questions: each
answer option is shown in every letter position, the label-token probabilities
are mapped back to semantic option order, and the rotations are averaged. A
singleton axis is deterministic and requires no model call. The axis
distributions are combined before the correct student's probability is used as
reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

LETTERS = "ABCDEFGH"


@dataclass(frozen=True, slots=True)
class ProbeOption:
    value: str
    description: str


@dataclass(frozen=True, slots=True)
class ProbeAxis:
    name: str
    question: str
    options: tuple[ProbeOption, ...]

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(
                f"probe axis {self.name!r} needs at least one option, "
                f"got {len(self.options)}."
            )
        if len(self.options) > len(LETTERS):
            raise ValueError(
                f"probe axis {self.name!r} has {len(self.options)} options but "
                f"only {len(LETTERS)} answer letters are defined."
            )
        values = [option.value for option in self.options]
        if len(set(values)) != len(values):
            raise ValueError(f"probe axis {self.name!r} repeats an option: {values}.")

    @property
    def size(self) -> int:
        return len(self.options)

    def index_of(self, value: str) -> int:
        for index, option in enumerate(self.options):
            if option.value == value:
                return index
        raise KeyError(
            f"probe axis {self.name!r} has no option {value!r}; "
            f"options are {[option.value for option in self.options]}."
        )


@dataclass(frozen=True, slots=True)
class ProbeReading:
    distribution: tuple[float, ...]
    correct_index: int
    disagreement: float
    calls: int

    @property
    def correct_probability(self) -> float:
        return float(self.distribution[self.correct_index])

    @property
    def is_argmax_correct(self) -> float:
        return argmax_correct(self.distribution, self.correct_index)


@dataclass(frozen=True, slots=True)
class StudentTypeProbeReading:
    behavior: ProbeReading
    information: ProbeReading
    joint_distribution: tuple[float, ...]
    correct_joint_index: int

    @property
    def correct_probability(self) -> float:
        return float(self.joint_distribution[self.correct_joint_index])

    @property
    def is_argmax_correct(self) -> float:
        return argmax_correct(self.joint_distribution, self.correct_joint_index)

    @property
    def calls(self) -> int:
        return self.behavior.calls + self.information.calls


def build_axis(
    name: str,
    question: str,
    values: Sequence[str],
    descriptions: Mapping[str, str],
    *,
    fields: Mapping[str, object] | None = None,
) -> ProbeAxis:
    """Build options from exactly the values present in the student pool."""
    ordered = sorted(dict.fromkeys(str(value) for value in values))
    options: list[ProbeOption] = []
    for value in ordered:
        template = descriptions.get(value)
        if not template:
            raise KeyError(f"probe axis {name!r} has no description for {value!r}.")
        options.append(
            ProbeOption(value=value, description=template.format(**(fields or {})))
        )
    return ProbeAxis(name=name, question=question, options=tuple(options))


def option_rotations(size: int, *, cyclic: bool = True) -> tuple[tuple[int, ...], ...]:
    """All cyclic orders, so every semantic option occupies every slot once."""
    if size <= 0:
        raise ValueError("an option set cannot be empty.")
    if not cyclic:
        return (tuple(range(size)),)
    return tuple(
        tuple((slot + shift) % size for slot in range(size)) for shift in range(size)
    )


def render_options(axis: ProbeAxis, order: Sequence[int]) -> str:
    return "\n".join(
        f"{LETTERS[slot]}) {axis.options[option_index].description}"
        for slot, option_index in enumerate(order)
    )


def restricted_probabilities(logprobs: Sequence[float]) -> list[float]:
    """Softmax over only the requested label-token ids."""
    values = [float(value) for value in logprobs]
    if not values:
        return []
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [1.0 / len(values)] * len(values)
    top = max(finite)
    weights = [
        math.exp(value - top) if math.isfinite(value) else 0.0 for value in values
    ]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [weight / total for weight in weights]


def group_probabilities(
    probabilities: Sequence[float], groups: Sequence[Sequence[int]]
) -> list[float]:
    """Combine the bare and leading-space token spellings of each letter."""
    return [sum(float(probabilities[index]) for index in group) for group in groups]


def unpermute(
    slot_probabilities: Sequence[float], order: Sequence[int], size: int
) -> list[float]:
    if len(slot_probabilities) != len(order):
        raise ValueError(
            f"got {len(slot_probabilities)} slot probabilities for {len(order)} slots."
        )
    result = [0.0] * size
    for slot, option_index in enumerate(order):
        result[option_index] = float(slot_probabilities[slot])
    return result


def average_distributions(distributions: Sequence[Sequence[float]]) -> list[float]:
    if not distributions:
        return []
    size = len(distributions[0])
    if size == 0 or any(len(item) != size for item in distributions):
        raise ValueError("cannot average empty distributions or different sizes.")
    mean = [
        sum(float(item[index]) for item in distributions) / len(distributions)
        for index in range(size)
    ]
    mass = sum(mean)
    if mass <= 0.0:
        return [1.0 / size] * size
    return [value / mass for value in mean]


def joint_distribution(distributions: Sequence[Sequence[float]]) -> list[float]:
    """Return the normalized row-major outer product of axis distributions."""
    if not distributions:
        return []
    joint = [1.0]
    for distribution in distributions:
        values = [float(value) for value in distribution]
        if not values:
            raise ValueError("cannot combine an empty axis distribution.")
        joint = [prefix * value for prefix in joint for value in values]
    mass = sum(joint)
    if mass <= 0.0:
        return [1.0 / len(joint)] * len(joint)
    return [value / mass for value in joint]


def joint_index(indices: Sequence[int], sizes: Sequence[int]) -> int:
    """Map per-axis indices to the row-major joint-distribution index."""
    if len(indices) != len(sizes):
        raise ValueError("joint indices and axis sizes must have equal length.")
    result = 0
    for index, size in zip(indices, sizes, strict=True):
        if size <= 0 or index < 0 or index >= size:
            raise ValueError(f"axis index {index} is outside size {size}.")
        result = result * size + index
    return result


def rotation_disagreement(distributions: Sequence[Sequence[float]]) -> float:
    """Mean total-variation distance from the rotation-averaged answer."""
    if len(distributions) < 2:
        return 0.0
    mean = average_distributions(distributions)
    return sum(
        0.5
        * sum(
            abs(float(value) - mean[index]) for index, value in enumerate(distribution)
        )
        for distribution in distributions
    ) / len(distributions)


def argmax_correct(distribution: Sequence[float], correct_index: int) -> float:
    """Return one only when the correct option is the strict maximum."""
    if not distribution:
        return 0.0
    target = float(distribution[correct_index])
    return float(
        all(
            target > float(value)
            for index, value in enumerate(distribution)
            if index != correct_index
        )
    )
