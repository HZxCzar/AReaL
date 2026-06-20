from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


VOWELS = set("aeiou")
ALPHABET = string.ascii_lowercase


class Rule(Protocol):
    rule_id: str
    description: str

    def __call__(self, text: str) -> bool:
        ...


@dataclass(frozen=True)
class LambdaRule:
    rule_id: str
    description: str
    predicate: Callable[[str], bool]

    def __call__(self, text: str) -> bool:
        return self.predicate(text)


def count_vowels(text: str) -> int:
    return sum(ch in VOWELS for ch in text)


def has_double_letter(text: str) -> bool:
    return any(a == b for a, b in zip(text, text[1:]))


def default_rules() -> list[Rule]:
    return [
        LambdaRule(
            "even_vowels",
            "A string is valid if it contains an even number of vowels.",
            lambda s: count_vowels(s) % 2 == 0,
        ),
        LambdaRule(
            "starts_and_ends_same",
            "A string is valid if its first and last letters are the same.",
            lambda s: len(s) > 0 and s[0] == s[-1],
        ),
        LambdaRule(
            "contains_double_letter",
            "A string is valid if it contains the same letter twice in a row.",
            has_double_letter,
        ),
        LambdaRule(
            "length_multiple_of_three",
            "A string is valid if its length is a multiple of three.",
            lambda s: len(s) % 3 == 0,
        ),
        LambdaRule(
            "contains_z_or_q",
            "A string is valid if it contains z or q.",
            lambda s: "z" in s or "q" in s,
        ),
        LambdaRule(
            "more_consonants_than_vowels",
            "A string is valid if it has more consonants than vowels.",
            lambda s: (len(s) - count_vowels(s)) > count_vowels(s),
        ),
        LambdaRule(
            "alphabetically_non_decreasing",
            "A string is valid if its letters are in non-decreasing alphabetical order.",
            lambda s: all(a <= b for a, b in zip(s, s[1:])),
        ),
        LambdaRule(
            "third_letter_is_vowel",
            "A string is valid if its third letter is a vowel.",
            lambda s: len(s) >= 3 and s[2] in VOWELS,
        ),
    ]


def random_string(rng: random.Random, min_len: int = 3, max_len: int = 10) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def sample_examples(
    rule: Rule,
    rng: random.Random,
    n: int,
    balanced: bool = True,
    max_attempts: int = 50_000,
) -> list[dict[str, object]]:
    if not balanced:
        examples = []
        for _ in range(n):
            text = random_string(rng)
            examples.append({"x": text, "y": bool(rule(text))})
        return examples

    target_pos = n // 2
    target_neg = n - target_pos
    positives: list[dict[str, object]] = []
    negatives: list[dict[str, object]] = []
    seen: set[str] = set()

    attempts = 0
    while (len(positives) < target_pos or len(negatives) < target_neg) and attempts < max_attempts:
        attempts += 1
        text = random_string(rng)
        if text in seen:
            continue
        seen.add(text)
        label = bool(rule(text))
        record = {"x": text, "y": label}
        if label and len(positives) < target_pos:
            positives.append(record)
        elif not label and len(negatives) < target_neg:
            negatives.append(record)

    examples = positives + negatives
    rng.shuffle(examples)
    if len(examples) < n:
        raise RuntimeError(f"Could only sample {len(examples)} examples for rule {rule.rule_id}.")
    return examples


def get_rule(rule_id: str) -> Rule:
    for rule in default_rules():
        if rule.rule_id == rule_id:
            return rule
    known = ", ".join(rule.rule_id for rule in default_rules())
    raise KeyError(f"Unknown rule_id={rule_id!r}. Known rules: {known}")


def iter_rule_cycle(rng: random.Random) -> Iterable[Rule]:
    rules = default_rules()
    while True:
        shuffled = rules[:]
        rng.shuffle(shuffled)
        yield from shuffled
