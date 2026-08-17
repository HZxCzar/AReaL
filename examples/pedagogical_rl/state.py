from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from examples.pedagogical_rl.prompts import (
    INITIAL_ATTEMPT_WRAPPER,
    SIMPLE_STUDENT_PROMPT,
    STUDENT_FINAL_PROMPT,
    STUDENT_INITIAL_ATTEMPT_PROMPT,
    TEACHER_PROMPT,
    render,
)
from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.prompts import TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE


class ConversationType(str, Enum):
    GUIDED = "GUIDED"
    ATTEMPTED = "ATTEMPTED"


STUDENT_NAMES: tuple[str | None, ...] = (
    "Alex",
    "Jamie",
    "Taylor",
    "Jordan",
    "Sam",
    "Casey",
    "Morgan",
    "Riley",
    None,
)


def student_visible_text(content: str) -> str:
    """Apply PedagogicalRL's exact teacher-to-student visibility rule."""

    return re.sub(r"<think>.*?</think>", "", content or "", flags=re.S).replace(
        "<end_of_conversation>", ""
    )


@dataclass(slots=True)
class NativeJudgeDecision:
    rule: str
    reasoning: str
    decision: str

    @property
    def rejected(self) -> bool:
        return self.decision == "REJECT"


@dataclass(slots=True)
class ClassroomEpisode:
    problem: str
    answer: str
    include_thinking: bool = False
    forced_type: ConversationType | None = None
    forced_student_name: str | None = None
    conversation: list[dict[str, str]] = field(default_factory=list)
    native_judges: list[NativeJudgeDecision] = field(default_factory=list)
    final_solutions: list[str] = field(default_factory=list)
    initial_attempt: str | None = None
    teacher_draft: str | None = None
    leak_failed: bool = False
    leak_checks: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    conversation_type: ConversationType = field(init=False)
    student_name: str | None = field(init=False)
    teacher_system_prompt: str = field(init=False)
    student_system_prompt: str = field(init=False)
    student_initial_prompt: str = field(init=False)
    student_final_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        problem_hash = hash(self.problem)
        self.conversation_type = self.forced_type or (
            ConversationType.ATTEMPTED if problem_hash % 2 else ConversationType.GUIDED
        )
        self.student_name = (
            self.forced_student_name
            if self.forced_student_name is not None
            else STUDENT_NAMES[problem_hash % len(STUDENT_NAMES)]
        )
        self.teacher_system_prompt = render(
            TEACHER_PROMPT,
            student_name=self.student_name,
            problem=self.problem,
            include_thinking=self.include_thinking,
        )
        self.student_system_prompt = render(
            SIMPLE_STUDENT_PROMPT,
            student_name=self.student_name,
            problem=self.problem,
        )
        self.student_initial_prompt = render(
            STUDENT_INITIAL_ATTEMPT_PROMPT, problem=self.problem
        )
        self.student_final_prompt = render(STUDENT_FINAL_PROMPT)

    @property
    def teacher_turns(self) -> int:
        return sum(message["role"] == "teacher" for message in self.conversation)

    @property
    def failed_native_judges(self) -> bool:
        return any(decision.rejected for decision in self.native_judges)

    @property
    def turn_leak_observed(self) -> bool:
        return any(bool(check.get("leaked")) for check in self.leak_checks)

    def add_initial_attempt(self, attempt: str) -> None:
        self.initial_attempt = attempt
        self.conversation.append(
            {
                "role": "student",
                "content": render(INITIAL_ATTEMPT_WRAPPER, attempt=attempt),
            }
        )

    def add_teacher(self, content: str) -> None:
        self.conversation.append({"role": "teacher", "content": content})

    def add_student(self, content: str) -> None:
        self.conversation.append({"role": "student", "content": content})

    def teacher_messages(self) -> list[dict[str, str]]:
        teacher_prompt = self.teacher_system_prompt
        if self.teacher_draft is not None:
            private_draft = strip_reasoning_for_context(self.teacher_draft).strip()
            if private_draft:
                context = render(
                    TEACHER_PRE_SOLVE_FILTER_CONTEXT_TEMPLATE,
                    raw_output=private_draft,
                )
                teacher_prompt = f"{teacher_prompt.rstrip()}\n\n{context}\n"
        messages = [{"role": "system", "content": teacher_prompt}]
        messages.extend(
            {
                "role": "assistant" if message["role"] == "teacher" else "user",
                "content": message["content"],
            }
            for message in self.conversation
        )
        return messages

    def student_messages(self, *, final: bool = False) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.student_system_prompt}]
        messages.extend(
            {
                "role": "assistant" if message["role"] == "student" else "user",
                "content": student_visible_text(message["content"]),
            }
            for message in self.conversation
        )
        if final:
            messages.append({"role": "user", "content": self.student_final_prompt})
        return messages

    def initial_student_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.student_initial_prompt}]

    def hidden_conversation(self) -> list[dict[str, str]]:
        return [
            {
                "role": message["role"],
                "content": student_visible_text(message["content"]),
            }
            for message in self.conversation
        ]

    def content_token_count(self, tokenizer: Any) -> int:
        return sum(
            len(tokenizer.encode(message["content"])) for message in self.conversation
        )

    def should_stop_dialogue(
        self,
        *,
        tokenizer: Any,
        max_teacher_turns: int,
        max_tokens_in_conversation: int,
    ) -> bool:
        if self.leak_failed:
            return True
        if self.teacher_turns >= max_teacher_turns:
            self.termination_reason = self.termination_reason or "max_turns"
            return True
        if (
            self.conversation
            and self.conversation[-1]["role"] == "teacher"
            and ("<end_of_conversation>" in self.conversation[-1]["content"])
        ):
            self.termination_reason = "end_of_conversation"
            return True
        if self.content_token_count(tokenizer) > max_tokens_in_conversation:
            self.termination_reason = self.termination_reason or "max_tokens"
            return True
        return False

    def to_trace(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "answer": self.answer,
            "conversation_type": self.conversation_type.value,
            "student_name": self.student_name,
            "conversation": self.conversation,
            "initial_attempt": self.initial_attempt,
            "teacher_draft": self.teacher_draft,
            "leak_failed": self.leak_failed,
            "turn_leak_observed": self.turn_leak_observed,
            "leak_checks": self.leak_checks,
            "termination_reason": self.termination_reason,
            "native_judges": [
                {
                    "rule": decision.rule,
                    "reasoning": decision.reasoning,
                    "decision": decision.decision,
                }
                for decision in self.native_judges
            ],
            "final_solutions": self.final_solutions,
        }
