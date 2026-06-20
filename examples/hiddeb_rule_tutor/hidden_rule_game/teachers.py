from __future__ import annotations

from .inference_engine import GenerationConfig, VLLMEngine
from .rules import Rule


class LLMTeacher:
    def __init__(self, engine: VLLMEngine) -> None:
        self.engine = engine

    def answer(
        self,
        rule: Rule,
        student_message: str,
        transcript: list[dict[str, str]],
        labeled_examples: list[dict[str, object]],
    ) -> str:
        examples_text = "\n".join(
            f"{item['x']} -> {'valid' if item['y'] else 'invalid'}" for item in labeled_examples
        )
        dialogue = "\n".join(
            f"Student: {turn['student']}\nTutor: {turn['tutor']}" for turn in transcript[-6:]
        )
        prompt = (
            "You are a tutor in a hidden-rule induction game.\n"
            "The student must infer the rule, but you must not directly reveal the rule.\n"
            "You may provide labeled examples and short guidance. Keep the reply concise.\n\n"
            f"Private hidden rule, not to reveal verbatim: {rule.description}\n\n"
            f"Recent dialogue:\n{dialogue}\n\n"
            f"Student message: {student_message}\n\n"
            "Fresh labeled examples you may show exactly:\n"
            f"{examples_text}\n\n"
            "Tutor reply:"
        )
        reply = self.engine.generate([prompt], GenerationConfig(max_tokens=160, temperature=0.3))[0]
        if not _contains_any_example(reply, labeled_examples):
            reply = "Here are labeled examples:\n" + examples_text + "\nLook for a concise pattern."
        return reply


def _contains_any_example(reply: str, examples: list[dict[str, object]]) -> bool:
    return any(str(item["x"]) in reply for item in examples)
