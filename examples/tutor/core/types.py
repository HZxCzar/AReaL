from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FeedbackKind = Literal["none", "student_judged"]
LeakHandlingMode = Literal["disabled", "reward_only", "terminate"]
StudentGeneralizeMode = Literal["only_success", "always"]
PromptSelectionSource = Literal[
    "pool",
    "pool_base",
    "warmup_full",
]


@dataclass(frozen=True, slots=True)
class PromptPoolSelection:
    index: int
    suffix: str
    source: PromptSelectionSource = "pool"
    rollout_version: int | None = None
    warmup_probability: float = 0.0
    prompt_path: str = ""
    pool: str = ""


@dataclass(frozen=True, slots=True)
class StudentTurnBehavior:
    index: int
    name: str
    instruction: str
    probability: float


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
    leak_level: int | None = None


@dataclass(slots=True)
class TeacherProgressJudgeResult:
    raw_output: str
    score: int | None
    reason: str
    parse_error: str | None
    local_advantage: float = 0.0
    reward_shaping: float = 0.0


@dataclass(slots=True)
class StudentRequestJudgeResult:
    raw_output: str
    score: int | None
    reason: str
    parse_error: str | None
    reward: float = 0.0


@dataclass(slots=True)
class StudentQuestionGenerationResult:
    prompt: str
    raw_output: str
    question: str
    error: str | None


@dataclass(slots=True)
class PublicHistoryState:
    summary: str = ""
    turn_count: int = 0
    # The dialogue as real messages: [{"role": "teacher"|"student", "content": ...}].
    # `summary` is kept only for debug traces and logging; prompts are built from
    # `turns` so the model sees an ordinary multi-turn chat.
    turns: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class TutorPrivateFeedback:
    kind: FeedbackKind = "none"
    student_output: str = ""
    judge_correct: bool = False
    judge_feedback: str = ""


@dataclass(slots=True)
class TeacherPreSolveAttempt:
    attempt: int
    raw_output: str
    error: str | None
    accepted: bool
    judge_result: JudgeResult | None = None


@dataclass(slots=True)
class TeacherPreSolveResult:
    enabled: bool
    mode: str
    accepted: bool
    attempts: list[TeacherPreSolveAttempt] = field(default_factory=list)
    raw_output: str = ""
    error: str | None = None
    verification_enabled: bool = True


@dataclass(slots=True, frozen=True)
class TeacherGuidance:
    """An instruction appended to the end of the teacher prompt for one turn.

    ``kind`` is "move" for a prescribed teaching move (guided slots) and "repair"
    for the on-policy-distillation teacher's privileged instruction. Guidance is
    always stripped before the turn becomes a training sample.
    """

    kind: str
    name: str
    instruction: str
    slot: int | None = None

    @property
    def strip_from_training(self) -> bool:
        """Whether the instruction must be removed from the prompt this turn is
        trained on.

        "move" guidance is an exploration device: the policy must not learn to
        depend on being told, so the instruction is stripped and the turn becomes
        off-policy. "prompt" guidance is the opposite -- it is a deployment
        choice being measured, so it stays in the prompt at training and at
        evaluation, and nothing about the turn is off-policy.
        """
        return self.kind != "prompt"


@dataclass(slots=True)
class TutorTurnState:
    task: str
    ground_truth: str
    public_history: PublicHistoryState
    previous_tutor_visible_output: str
    previous_feedback: TutorPrivateFeedback
    turn_idx: int
    max_turns: int
    teacher_pre_solve_result: TeacherPreSolveResult | None = None
    teacher_prompt_selection: PromptPoolSelection | None = None
    student_reply_before_teacher: str = ""
    preceding_student_turn_behavior: StudentTurnBehavior | None = None
    guidance: TeacherGuidance | None = None


@dataclass(slots=True)
class StudentTurnState:
    task: str
    public_history: PublicHistoryState
    previous_student_output: str
    latest_tutor_visible_output: str
    student_prompt_selection: PromptPoolSelection | None = None
    student_turn_behavior: StudentTurnBehavior | None = None


@dataclass(slots=True)
class TurnArtifact:
    turn_idx: int
    tutor_state: TutorTurnState
    # The exact messages sent to the teacher this turn. Training tokens are
    # re-rendered from these, so they must be what generation actually saw.
    tutor_messages: list[dict[str, str]]
    tutor_response: Any
    tutor_raw_output: str
    tutor_visible_output: str
    leak_result: LeakCheckResult
    public_history_before: list[dict[str, str]]
    public_history_after: list[dict[str, str]]
    tutor_format_error: str | None = None
    student_state: StudentTurnState | None = None
    student_prompt: str = ""
    student_output: str = ""
    student_error: str | None = None
    judge_result: JudgeResult | None = None
    invalid_due_to_leak: bool = False
    previous_teacher_similarity: float | None = None
    teacher_similarity_error: str | None = None
    teacher_progress_judge_result: TeacherProgressJudgeResult | None = None
    student_request_judge_result: StudentRequestJudgeResult | None = None
    student_question_generation: StudentQuestionGenerationResult | None = None
    # Set when this turn was selected for on-policy distillation. Holds the
    # instructed-teacher prompt tokens; the output tokens are appended by
    # ``response_to_tensordict``.
    opd_prompt_tokens: list[int] | None = None
    opd_skip_reason: str = ""


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
    teacher_pre_solve_result: TeacherPreSolveResult | None = None
    student_name: str = ""
    student_model: str = ""
    teacher_prompt_selection: PromptPoolSelection | None = None
    student_prompt_selection: PromptPoolSelection | None = None
    initial_student_turn_behavior: StudentTurnBehavior | None = None
    initial_student_question_generation: StudentQuestionGenerationResult | None = None


@dataclass(slots=True)
class RewardAssignment:
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
    reward: float
    reward_components: dict[str, float]
    public_history_before: list[dict[str, str]]
    public_history_after: list[dict[str, str]]
    student_turn_behavior: StudentTurnBehavior | None = None
    leak_level: int | None = None
    invalid_due_to_leak: bool = False
    tutor_format_error: str | None = None
    previous_teacher_similarity: float | None = None
    teacher_similarity_error: str | None = None
    teacher_progress_judge_result: TeacherProgressJudgeResult | None = None
    student_request_judge_result: StudentRequestJudgeResult | None = None
    student_question_generation: StudentQuestionGenerationResult | None = None
