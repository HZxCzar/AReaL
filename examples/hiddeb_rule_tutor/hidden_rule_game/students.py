from __future__ import annotations

import re
from dataclasses import dataclass, field

from .inference_engine import GenerationConfig, VLLMEngine
from .rules import count_vowels, has_double_letter


EXAMPLE_RE = re.compile(r"^([a-z]+)\s*->\s*(valid|invalid)$", re.IGNORECASE)


def extract_examples(transcript: list[dict[str, str]]) -> list[tuple[str, bool]]:
    examples: list[tuple[str, bool]] = []
    for turn in transcript:
        for line in turn["tutor"].splitlines():
            match = EXAMPLE_RE.match(line.strip())
            if match:
                examples.append((match.group(1).lower(), match.group(2).lower() == "valid"))
    return examples


@dataclass
class MemoryStudent:
    examples: list[tuple[str, bool]] = field(default_factory=list)

    def reset(self) -> None:
        self.examples.clear()

    def act(self, observation: str, transcript: list[dict[str, str]]) -> str:
        self.examples = extract_examples(transcript)
        if len(self.examples) < 20:
            return "Please provide more labeled examples."
        return f"FINAL_RULE: {self.guess_rule(transcript)}"

    def observe(self, message: str, reply: str) -> None:
        self.examples = extract_examples([{"student": message, "tutor": reply}]) + self.examples

    def guess_rule(self, transcript: list[dict[str, str]]) -> str:
        examples = extract_examples(transcript) or self.examples
        candidates = [
            (
                "A string is valid if it contains an even number of vowels.",
                lambda s: count_vowels(s) % 2 == 0,
            ),
            (
                "A string is valid if its first and last letters are the same.",
                lambda s: len(s) > 0 and s[0] == s[-1],
            ),
            (
                "A string is valid if it contains the same letter twice in a row.",
                has_double_letter,
            ),
            (
                "A string is valid if its length is a multiple of three.",
                lambda s: len(s) % 3 == 0,
            ),
            ("A string is valid if it contains z or q.", lambda s: "z" in s or "q" in s),
            (
                "A string is valid if it has more consonants than vowels.",
                lambda s: (len(s) - count_vowels(s)) > count_vowels(s),
            ),
            (
                "A string is valid if its letters are in non-decreasing alphabetical order.",
                lambda s: all(a <= b for a, b in zip(s, s[1:])),
            ),
            (
                "A string is valid if its third letter is a vowel.",
                lambda s: len(s) >= 3 and s[2] in set("aeiou"),
            ),
        ]
        if not examples:
            return "A string is valid if it matches a simple hidden pattern."
        scored = []
        for description, predicate in candidates:
            correct = sum(bool(predicate(text)) == label for text, label in examples)
            scored.append((correct, description, predicate))
        scored.sort(reverse=True, key=lambda item: item[0])
        return scored[0][1]

    def predict_label(
        self,
        text: str,
        transcript: list[dict[str, str]],
        guessed_rule: str | None = None,
    ) -> bool:
        rule = guessed_rule or self.guess_rule(transcript)
        lowered = rule.lower()
        if "even number of vowels" in lowered:
            return count_vowels(text) % 2 == 0
        if "first and last" in lowered:
            return len(text) > 0 and text[0] == text[-1]
        if "same letter twice" in lowered or "double" in lowered:
            return has_double_letter(text)
        if "multiple of three" in lowered:
            return len(text) % 3 == 0
        if "z or q" in lowered:
            return "z" in text or "q" in text
        if "more consonants" in lowered:
            return (len(text) - count_vowels(text)) > count_vowels(text)
        if "non-decreasing" in lowered or "alphabetical order" in lowered:
            return all(a <= b for a, b in zip(text, text[1:]))
        if "third letter" in lowered:
            return len(text) >= 3 and text[2] in set("aeiou")
        return False


class LLMStudent:
    def __init__(self, engine: VLLMEngine) -> None:
        self.engine = engine

    def reset(self) -> None:
        pass

    def act(self, observation: str, transcript: list[dict[str, str]]) -> str:
        prompt = _format_dialogue(
            transcript,
            suffix=(
                "You are a student trying to infer a hidden Boolean rule over strings. "
                "Ask for more examples when uncertain. When confident, reply with exactly one "
                "line starting with `FINAL_RULE:` followed by your hypothesized rule. "
                "Do not ask for the answer.\n"
                f"Latest tutor message:\n{observation}\nStudent:"
            ),
        )
        return self.engine.generate([prompt], GenerationConfig(max_tokens=80, temperature=0.4))[0]

    def observe(self, message: str, reply: str) -> None:
        pass

    def guess_rule(self, transcript: list[dict[str, str]]) -> str:
        prompt = _format_dialogue(
            transcript,
            suffix="Infer the hidden rule. Reply with one sentence starting with `A string is valid if`.",
        )
        return self.engine.generate([prompt], GenerationConfig(max_tokens=96, temperature=0.1))[0]

    def predict_label(
        self,
        text: str,
        transcript: list[dict[str, str]],
        guessed_rule: str | None = None,
    ) -> bool:
        rule = guessed_rule or self.guess_rule(transcript)
        prompt = (
            f"Rule hypothesis: {rule}\n"
            f"String: {text}\n"
            "Is the string valid under the hypothesis? Reply only `valid` or `invalid`."
        )
        answer = self.engine.generate([prompt], GenerationConfig(max_tokens=8, temperature=0.0))[0]
        return "valid" in answer.lower() and "invalid" not in answer.lower()


def _format_dialogue(transcript: list[dict[str, str]], suffix: str) -> str:
    lines: list[str] = []
    for turn in transcript:
        lines.append(f"Student: {turn['student']}")
        lines.append(f"Tutor: {turn['tutor']}")
    lines.append(suffix)
    return "\n".join(lines)
