from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProgressLabel = Literal["improved", "same", "regressed", "unknown"]
ConfidenceLabel = Literal["high", "medium", "low"]
FeedbackKind = Literal["none", "student_judged", "leak"]


@dataclass(slots=True)
class JudgeResult:
    raw_output: str
    correct: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class LeakCheckResult:
    raw_output: str
    leaked: bool
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class GeneratedProblemResult:
    raw_output: str
    task: str
    ground_truth: str
    similarity_notes: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class PublicHistoryState:
    summary: str = ""
    turn_count: int = 0


@dataclass(slots=True)
class TutorPrivateFeedback:
    kind: FeedbackKind = "none"
    student_output: str = ""
    judge_correct: bool = False
    judge_feedback: str = ""
    progress_label: ProgressLabel = "unknown"
    progress_feedback: str = ""
    leak_feedback: str = ""


@dataclass(slots=True)
class TutorTurnState:
    task: str
    ground_truth: str
    public_history: PublicHistoryState
    previous_tutor_visible_output: str
    previous_feedback: TutorPrivateFeedback
    turn_idx: int
    max_turns: int


@dataclass(slots=True)
class StudentTurnState:
    task: str
    public_history: PublicHistoryState
    previous_student_output: str
    latest_tutor_visible_output: str


@dataclass(slots=True)
class ProgressJudgment:
    raw_output: str
    label: ProgressLabel
    confidence: ConfidenceLabel
    feedback: str
    parse_error: str | None = None
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnArtifact:
    turn_idx: int
    tutor_state: TutorTurnState
    tutor_prompt: str
    tutor_response: Any
    tutor_raw_output: str
    tutor_visible_output: str
    leak_result: LeakCheckResult
    public_history_before: str
    public_history_after: str
    student_state: StudentTurnState | None = None
    student_prompt: str = ""
    student_output: str = ""
    student_error: str | None = None
    judge_result: JudgeResult | None = None


@dataclass(slots=True)
class EpisodeArtifact:
    task: str
    ground_truth: str
    initial_student_answer: str
    initial_student_error: str | None
    initial_judge_result: JudgeResult
    turns: list[TurnArtifact]
    termination_reason: str
    pre_success: bool
    leak_count: int
    latest_student_answer: str


@dataclass(slots=True)
class RewardAssignment:
    progress: ProgressJudgment
    reward: float
    reward_components: dict[str, float]


@dataclass(slots=True)
class TurnTrace:
    turn_idx: int
    tutor_state: TutorTurnState
    tutor_raw_output: str
    tutor_visible_output: str
    leaked: bool
    student_output: str
    judge_correct: bool
    judge_feedback: str
    progress: ProgressJudgment
    reward: float
    reward_components: dict[str, float]
    public_history_before: str
    public_history_after: str
