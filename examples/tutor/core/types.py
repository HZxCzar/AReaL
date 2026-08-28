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
class PersonalityGateResult:
    """One personality gate check on one teacher message.

    `passed` is what routes the turn: True calls the student, False replaces its
    reply with a complaint. `reason` keeps the binary judge's analysis and is empty
    for the logits classifier. `error` is set only when every retry came back
    unclean, in which case `passed` is False -- the conservative default, so a
    broken check never lets through a message that may violate the preference.
    """

    raw_output: str
    passed: bool
    reason: str
    error: str | None = None
    attempts: int = 1
    # False when gate_sample_rate did not select this turn. A turn that was never
    # checked is not evidence of compliance, so it must not land in the numerator or
    # the denominator of the compliance rate.
    sampled: bool = True
    # Set only by the optional classification gate. Binary-gate traces retain their
    # historical shape because trace_to_json removes these keys when they are None.
    classification_label: str | None = None
    classification_logprobs: dict[str, float] | None = None
    classification_probabilities: dict[str, float] | None = None
    classification_margin: float | None = None


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
    # Exact prior teacher replies, parallel to the teacher turns in
    # ``public_history``. Keeping this outside PublicHistoryState prevents the
    # student's conversation and re-test views from ever receiving private
    # reasoning. A tuple snapshots the history stored on each training row.
    previous_tutor_raw_outputs: tuple[str, ...] = ()
    # Privileged facts about the student sampled for this episode. This belongs
    # to the teacher state rather than PublicHistoryState so it can never enter
    # the student's masked or unmasked view. Empty is the default-off control.
    teacher_private_student_profile: str = ""


@dataclass(slots=True)
class StudentTurnState:
    task: str
    public_history: PublicHistoryState
    previous_student_output: str
    latest_tutor_visible_output: str
    student_prompt_selection: PromptPoolSelection | None = None
    student_turn_behavior: StudentTurnBehavior | None = None
    # Which part of the history this student may see, carried from the student
    # selected for the episode. None means this whole student-visible history, so
    # a rollout that configures no masks behaves exactly as before. A personality
    # gate may already have removed teacher-only failed exchanges from this state;
    # the mask can restrict it further. The teacher and training artifact retain
    # their separate complete public history.
    student_mask: dict[str, Any] | None = None
    # The selected student's action space, 'text' or 'code'. Carried on the state
    # rather than read off the workflow because one workflow instance serves every
    # concurrent episode, so anything per-episode has to travel with the episode.
    student_mode: str = "text"
    # What this student demands of the teacher's manner, or "" for the open gate.
    # Here for the same reason as the two above: one workflow, many concurrent
    # episodes, so per-episode state travels with the episode rather than sitting on
    # the workflow where a neighbouring episode would read it.
    student_personality: str = ""


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
    # None when this student has no personality, or when the turn ended on a format
    # error or a leak before the gate could run.
    personality_gate_result: PersonalityGateResult | None = None
    # True when the gate closed and the student never answered: `student_output` holds
    # the injected complaint instead of a student reply. Such a turn is NOT a student
    # attempt -- it contributes to neither solved nor final_correct -- but it does
    # consume one of the turn budget, which is the whole cost of failing the gate.
    personality_gated: bool = False
    # True only when this gated turn injected the remedy-naming complaint rather
    # than a bare "I don't understand" response. This records the sampled branch
    # directly; no downstream logic has to infer it from complaint text.
    personality_complaint_explained: bool = False
    # The gate-failed teacher turn that ended training after an earlier explained
    # complaint. Like a leak/format termination, this turn is trained but never
    # enters either dialogue transcript.
    personality_gate_terminated: bool = False
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
    # 'text' or 'code'. The re-test and the no-teaching baseline both branch on
    # this, and both are built from the artifact rather than from workflow state.
    student_mode: str = "text"
    teacher_prompt_selection: PromptPoolSelection | None = None
    student_prompt_selection: PromptPoolSelection | None = None
    initial_student_turn_behavior: StudentTurnBehavior | None = None
    initial_student_question_generation: StudentQuestionGenerationResult | None = None
    # Both transcripts and both scorers' solutions from the tutor/PedagogicalRL
    # cross, for the debug trace. Evaluation only, and None everywhere else.
    cross_eval_details: dict[str, Any] | None = None


@dataclass(slots=True)
class RewardAssignment:
    reward: float
    reward_components: dict[str, float]
    # The part of reward that belongs to the turn that produced it and must
    # not be accumulated backward onto earlier turns by ReBN. Populated from
    # EpisodeRewardComputer(turn_local_components=...); 0.0 keeps the old
    # behaviour, where every component propagates.
    local_reward: float = 0.0


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
    # None when this student has no personality, or when the turn ended on a format
    # error or a leak before the gate could run.
    personality_gate_result: PersonalityGateResult | None = None
    # True when the gate closed and the student never answered: `student_output` holds
    # the injected complaint instead of a student reply. Such a turn is NOT a student
    # attempt -- it contributes to neither solved nor final_correct -- but it does
    # consume one of the turn budget, which is the whole cost of failing the gate.
    personality_gated: bool = False
    personality_complaint_explained: bool = False
    personality_gate_terminated: bool = False
