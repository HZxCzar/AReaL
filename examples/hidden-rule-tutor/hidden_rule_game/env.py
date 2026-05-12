from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .rules import Rule, iter_rule_cycle, sample_examples


class TeacherAgent(Protocol):
    def answer(
        self,
        rule: Rule,
        student_message: str,
        transcript: list[dict[str, str]],
        labeled_examples: list[dict[str, object]],
    ) -> str:
        ...


@dataclass
class EpisodeResult:
    rule_id: str
    true_rule: str
    guessed_rule: str
    heldout_accuracy: float
    reward: float
    turns_taken: int
    examples_used: int
    rule_matched: bool
    transcript: list[dict[str, str]]


class Tutor:
    def __init__(self, rule: Rule, examples: list[dict[str, object]], teacher: TeacherAgent | None = None) -> None:
        self.rule = rule
        self.examples = examples
        self.cursor = 0
        self.teacher = teacher

    def answer(self, student_message: str, transcript: list[dict[str, str]]) -> str:
        lowered = student_message.lower()
        if "more" in lowered or "example" in lowered or self.cursor == 0:
            batch = self.examples[self.cursor : self.cursor + 4]
            self.cursor += len(batch)
            if self.teacher is not None and batch:
                return self.teacher.answer(self.rule, student_message, transcript, batch)
            lines = ["Here are labeled examples:"]
            for item in batch:
                label = "valid" if item["y"] else "invalid"
                lines.append(f"{item['x']} -> {label}")
            return "\n".join(lines)
        if "test:" in lowered:
            text = student_message.split("test:", 1)[1].strip().split()[0]
            return f"{text} -> {'valid' if self.rule(text) else 'invalid'}"
        return "Ask for more examples, propose a rule, or send `test: <string>`."


class HiddenRuleEnv:
    def __init__(
        self,
        seed: int = 7,
        examples_per_episode: int = 48,
        teacher: TeacherAgent | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.rule_iter = iter_rule_cycle(self.rng)
        self.examples_per_episode = examples_per_episode
        self.teacher = teacher

    def run_episode(self, student: object, rounds: int = 8, early_stop: bool = True) -> EpisodeResult:
        rule = next(self.rule_iter)
        examples = sample_examples(rule, self.rng, self.examples_per_episode)
        tutor = Tutor(rule, examples[: rounds * 4], teacher=self.teacher)
        heldout = examples[rounds * 4 :]
        transcript: list[dict[str, str]] = []

        if hasattr(student, "reset"):
            student.reset()

        observation = "A new hidden string rule has been selected. Ask the tutor for evidence."
        for _ in range(rounds):
            message = student.act(observation, transcript)
            reply = tutor.answer(message, transcript)
            transcript.append({"student": message, "tutor": reply})
            observation = reply
            if hasattr(student, "observe"):
                student.observe(message, reply)
            if early_stop and _is_final_rule_message(message):
                break

        guessed_rule = student.guess_rule(transcript)
        predictions = [bool(student.predict_label(item["x"], transcript, guessed_rule)) for item in heldout]
        labels = [bool(item["y"]) for item in heldout]
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / max(1, len(labels))
        reward = accuracy
        rule_matched = _rough_rule_match(guessed_rule, rule.description)
        if rule_matched:
            reward += 0.25

        return EpisodeResult(
            rule_id=rule.rule_id,
            true_rule=rule.description,
            guessed_rule=guessed_rule,
            heldout_accuracy=accuracy,
            reward=reward,
            turns_taken=len(transcript),
            examples_used=tutor.cursor,
            rule_matched=rule_matched,
            transcript=transcript,
        )


def _rough_rule_match(guess: str, truth: str) -> bool:
    guess_words = {w.strip(".,:;()").lower() for w in guess.split() if len(w) > 3}
    truth_words = {w.strip(".,:;()").lower() for w in truth.split() if len(w) > 3}
    return len(guess_words & truth_words) >= 2


def _is_final_rule_message(message: str) -> bool:
    lowered = message.lower()
    if "final_rule:" in lowered or "final rule:" in lowered:
        return True
    if "i think the rule" in lowered or "the rule is" in lowered:
        return True
    return "a string is valid if" in lowered and "?" not in lowered
