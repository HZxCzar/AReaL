"""Common environment contract for tutor-training tasks.

Task adapters implement native transitions. This base class owns the invariant
tutoring-round flow: compress context, obtain legal student actions, advance
untutored participants, and emit one trainable tutor record.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol


ContextTarget = Literal["tutor", "student"]


@dataclass(frozen=True, slots=True)
class Participant:
    participant_id: str
    policy_id: str
    role: str = ""
    team: str = ""
    tutor_eligible: bool = True


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    episode_id: str
    task_data: Mapping[str, Any]
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class CompressedContext:
    public_history: str
    participant_memory: str
    task_summary: str
    legal_actions: tuple[str, ...] = ()
    truncation_metadata: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TutorObservation:
    context: CompressedContext
    active_participants: tuple[Participant, ...]


@dataclass(frozen=True, slots=True)
class TutorAction:
    text: str
    model_response: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StudentAction:
    participant: Participant
    action: str
    raw_responses: tuple[str, ...]
    retry_count: int
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class AutomaticEvent:
    participant: Participant | None
    action: str
    event: str
    reward_components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TutorTurnRecord:
    observation: TutorObservation
    tutor_action: TutorAction
    student_actions: tuple[StudentAction, ...]
    environment_event: str
    reward: float
    reward_components: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TutorStepResult:
    trainable_turn: TutorTurnRecord
    next_observation: TutorObservation | None
    automatic_events: tuple[AutomaticEvent, ...]
    done: bool


@dataclass(frozen=True, slots=True)
class _TaskTransition:
    """Task-adapter result before automatic turns are processed."""

    event: str
    reward_components: Mapping[str, float]
    done: bool


@dataclass(frozen=True, slots=True)
class _AutomaticAdvance:
    """Task-adapter result after advancing untutored participants."""

    events: tuple[AutomaticEvent, ...]
    reward_components: Mapping[str, float]
    done: bool


class StudentPool(Protocol):
    """Provider of student-policy completions keyed by participant metadata."""

    async def act(self, participant: Participant, prompt: str) -> str:
        """Return one raw action completion for ``participant``."""


class TutorEnvironment(ABC):
    """Base environment whose ``step`` executes one complete tutoring round.

    Concrete adapters retain task state, prompts, parsing, rewards, and native
    transitions. They must advance untutored participants before returning
    control, so each step maps to one tutor-training sample.
    """

    def __init__(
        self,
        student_pool: StudentPool,
        *,
        max_student_action_attempts: int = 3,
        max_auto_steps: int = 64,
    ) -> None:
        if max_student_action_attempts < 1:
            raise ValueError("max_student_action_attempts must be at least one")
        if max_auto_steps < 1:
            raise ValueError("max_auto_steps must be at least one")
        self.student_pool = student_pool
        self.max_student_action_attempts = max_student_action_attempts
        self.max_auto_steps = max_auto_steps
        self._observation: TutorObservation | None = None
        self._trajectory: list[TutorTurnRecord] = []

    @abstractmethod
    async def reset(self, episode: EpisodeInput) -> TutorObservation:
        """Initialize native task state and return the first tutor observation."""

    @abstractmethod
    async def compress_context(self, target: ContextTarget) -> CompressedContext:
        """Compress current native state for the named model context."""

    @abstractmethod
    def build_student_prompt(
        self,
        participant: Participant,
        context: CompressedContext,
        tutor_action: TutorAction,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        """Build a task-specific prompt for one student action."""

    @abstractmethod
    def parse_student_action(
        self,
        participant: Participant,
        raw_action: str,
        context: CompressedContext,
    ) -> str | None:
        """Return a legal normalized action, or ``None`` when invalid."""

    @abstractmethod
    def fallback_student_action(
        self,
        participant: Participant,
        context: CompressedContext,
    ) -> str:
        """Return a deterministic legal action after failed repairs."""

    @abstractmethod
    async def apply_tutored_actions(
        self,
        tutor_action: TutorAction,
        student_actions: tuple[StudentAction, ...],
    ) -> _TaskTransition:
        """Apply the tutored participant actions to the native task."""

    @abstractmethod
    async def advance_untutored_participants(
        self, max_steps: int
    ) -> _AutomaticAdvance:
        """Advance non-tutored turns until the next tutor turn or termination."""

    @abstractmethod
    async def next_tutor_observation(self) -> TutorObservation:
        """Return the observation for the next tutor-eligible participant(s)."""

    async def step(self, tutor_action: TutorAction) -> TutorStepResult:
        """Execute one tutoring round and emit exactly one trainable record."""

        observation = self._require_observation()
        student_context = await self.compress_context("student")
        student_actions: list[StudentAction] = []
        for participant in observation.active_participants:
            student_actions.append(
                await self.generate_legal_student_action(
                    participant,
                    student_context,
                    tutor_action,
                )
            )
        tutored_actions = tuple(student_actions)
        transition = await self.apply_tutored_actions(tutor_action, tutored_actions)
        automatic_advance = _AutomaticAdvance((), {}, transition.done)
        if not transition.done:
            automatic_advance = await self.advance_untutored_participants(
                self.max_auto_steps
            )

        reward_components = self._merge_reward_components(
            transition.reward_components,
            automatic_advance.reward_components,
        )
        turn_record = TutorTurnRecord(
            observation=observation,
            tutor_action=tutor_action,
            student_actions=tutored_actions,
            environment_event=transition.event,
            reward=sum(reward_components.values()),
            reward_components=reward_components,
        )
        done = automatic_advance.done
        next_observation = None if done else await self.next_tutor_observation()
        self._trajectory.append(turn_record)
        self._observation = next_observation
        return TutorStepResult(
            trainable_turn=turn_record,
            next_observation=next_observation,
            automatic_events=automatic_advance.events,
            done=done,
        )

    @property
    def trajectory(self) -> tuple[TutorTurnRecord, ...]:
        """One trainable tutor record per completed tutoring round."""

        return tuple(self._trajectory)

    def _start_episode(self, observation: TutorObservation) -> TutorObservation:
        """Store the initial observation and clear records from the previous episode.

        Concrete ``reset`` implementations should return this method's result.
        """

        if not observation.active_participants:
            raise ValueError("initial tutor observation must include an active participant")
        self._trajectory.clear()
        self._observation = observation
        return observation

    async def generate_legal_student_action(
        self,
        participant: Participant,
        context: CompressedContext,
        tutor_action: TutorAction,
    ) -> StudentAction:
        """Generate, validate, repair, and finally fall back to a legal action."""

        raw_responses: list[str] = []
        retry_feedback: str | None = None
        for attempt in range(self.max_student_action_attempts):
            prompt = self.build_student_prompt(
                participant,
                context,
                tutor_action,
                retry_feedback=retry_feedback,
            )
            raw_action = await self.student_pool.act(participant, prompt)
            raw_responses.append(raw_action)
            action = self.parse_student_action(participant, raw_action, context)
            if action is not None:
                return StudentAction(
                    participant=participant,
                    action=action,
                    raw_responses=tuple(raw_responses),
                    retry_count=attempt,
                    used_fallback=False,
                )
            retry_feedback = self._retry_feedback(context)

        return StudentAction(
            participant=participant,
            action=self.fallback_student_action(participant, context),
            raw_responses=tuple(raw_responses),
            retry_count=self.max_student_action_attempts - 1,
            used_fallback=True,
        )

    def _require_observation(self) -> TutorObservation:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if not self._observation.active_participants:
            raise RuntimeError("tutor observation must include an active participant")
        return self._observation

    @staticmethod
    def _retry_feedback(context: CompressedContext) -> str:
        if context.legal_actions:
            return "Use exactly one legal action: " + ", ".join(context.legal_actions)
        return "Return a valid action using the required task format."

    @staticmethod
    def _merge_reward_components(
        *component_maps: Mapping[str, float],
    ) -> dict[str, float]:
        merged: dict[str, float] = {}
        for component_map in component_maps:
            for name, value in component_map.items():
                merged[name] = merged.get(name, 0.0) + float(value)
        return merged
