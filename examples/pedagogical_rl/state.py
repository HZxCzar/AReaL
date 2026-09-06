from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from examples.pedagogical_rl.prompts import (
    INITIAL_ATTEMPT_WRAPPER,
    SIMPLE_STUDENT_PROMPT,
    STUDENT_ATTEMPT_PROMPT,
    STUDENT_FINAL_PROMPT,
    STUDENT_INITIAL_ATTEMPT_PROMPT,
    render,
    render_teacher_prompt,
)
from examples.tutor.core.parsers import parse_tagged_teacher_action
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


def student_visible_text(content: str, *, output_format: str = "native") -> str:
    """Return the public part of a teacher action under either interface."""

    if output_format == "unified_xml":
        visible, ended, error = parse_tagged_teacher_action(
            content,
            allow_end=True,
            require_nonempty_output=True,
        )
        if error or ended:
            return ""
        return visible or ""

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
    teacher_output_format: str = "native"
    forced_type: ConversationType | None = None
    forced_student_name: str | None = None
    conversation: list[dict[str, Any]] = field(default_factory=list)
    native_judges: list[NativeJudgeDecision] = field(default_factory=list)
    final_solutions: list[str] = field(default_factory=list)
    evaluation_initial_solutions: list[str] = field(default_factory=list)
    initial_attempt: str | None = None
    teacher_draft: str | None = None
    leak_failed: bool = False
    leak_checks: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str | None = None
    format_errors: list[str] = field(default_factory=list)
    teacher_ended: bool = False
    preference: str = "none"
    preference_gate_checks: list[dict[str, Any]] = field(default_factory=list)
    conversation_type: ConversationType = field(init=False)
    student_name: str | None = field(init=False)
    teacher_system_prompt: str = field(init=False)
    student_system_prompt: str = field(init=False)
    student_initial_prompt: str = field(init=False)
    student_attempt_prompt: str = field(init=False)
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
        self.teacher_system_prompt = render_teacher_prompt(
            student_name=self.student_name,
            problem=self.problem,
            include_thinking=self.include_thinking,
            output_format=self.teacher_output_format,
        )
        self.student_system_prompt = render(
            SIMPLE_STUDENT_PROMPT,
            student_name=self.student_name,
            problem=self.problem,
        )
        self.student_initial_prompt = render(
            STUDENT_INITIAL_ATTEMPT_PROMPT, problem=self.problem
        )
        self.student_attempt_prompt = render(STUDENT_ATTEMPT_PROMPT, problem=self.problem)
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

    @property
    def format_failed(self) -> bool:
        return bool(self.format_errors)

    def add_teacher(self, content: str, *, student_visible: bool = True) -> None:
        self.conversation.append(
            {
                "role": "teacher",
                "content": content,
                "student_visible": student_visible,
            }
        )
        if self.teacher_output_format == "unified_xml":
            _visible, ended, error = parse_tagged_teacher_action(
                content,
                allow_end=True,
                require_nonempty_output=True,
            )
            if error:
                self.format_errors.append(error)
                self.termination_reason = "format_error"
            elif ended:
                # The shared action contract makes <end></end> control-only; it
                # is not an empty teacher message in the student's transcript.
                self.conversation[-1]["student_visible"] = False
                self.teacher_ended = True
                self.termination_reason = "end_of_conversation"
        elif "<end_of_conversation>" in content:
            self.teacher_ended = True
            self.termination_reason = "end_of_conversation"

    def add_student(self, content: str, *, student_visible: bool = True) -> None:
        self.conversation.append(
            {
                "role": "student",
                "content": content,
                "student_visible": student_visible,
            }
        )

    def hide_latest_teacher_from_student(self) -> None:
        if not self.conversation or self.conversation[-1]["role"] != "teacher":
            raise RuntimeError("latest classroom message is not a teacher turn")
        self.conversation[-1]["student_visible"] = False

    def latest_real_student_message(self) -> str | None:
        for message in reversed(self.conversation):
            if message["role"] == "student" and message.get(
                "student_visible", True
            ):
                return str(message["content"])
        return None

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
                "content": (
                    student_visible_text(
                        message["content"],
                        output_format=self.teacher_output_format,
                    )
                    if message["role"] == "teacher"
                    else message["content"]
                ),
            }
            for message in self.conversation
            if message.get("student_visible", True)
        )
        if final:
            messages.append({"role": "user", "content": self.student_final_prompt})
        return messages

    def initial_student_messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.student_initial_prompt}]

    def no_tutor_attempt_messages(self) -> list[dict[str, str]]:
        """Upstream evaluation baseline, separate from the classroom history."""

        return [{"role": "user", "content": self.student_attempt_prompt}]

    def hidden_conversation(self) -> list[dict[str, str]]:
        return [
            {
                "role": message["role"],
                "content": (
                    student_visible_text(
                        message["content"],
                        output_format=self.teacher_output_format,
                    )
                    if message["role"] == "teacher"
                    else message["content"]
                ),
            }
            for message in self.conversation
            if message.get("student_visible", True)
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
        if self.leak_failed or self.format_failed:
            return True
        if self.teacher_turns >= max_teacher_turns:
            self.termination_reason = self.termination_reason or "max_turns"
            return True
        if self.teacher_ended:
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
            "evaluation_initial_solutions": self.evaluation_initial_solutions,
            "teacher_draft": self.teacher_draft,
            "leak_failed": self.leak_failed,
            "turn_leak_observed": self.turn_leak_observed,
            "leak_checks": self.leak_checks,
            "termination_reason": self.termination_reason,
            "format_errors": self.format_errors,
            "teacher_ended": self.teacher_ended,
            "preference": self.preference,
            "preference_gate_checks": self.preference_gate_checks,
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
