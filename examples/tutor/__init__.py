"""Shared tutor-environment interfaces for AReaL training examples."""

from .environment import (
    AutomaticEvent,
    CompressedContext,
    EpisodeInput,
    Participant,
    StudentAction,
    StudentPool,
    TutorAction,
    TutorEnvironment,
    TutorObservation,
    TutorStepResult,
    TutorTurnRecord,
)

__all__ = [
    "AutomaticEvent",
    "CompressedContext",
    "EpisodeInput",
    "Participant",
    "StudentAction",
    "StudentPool",
    "TutorAction",
    "TutorEnvironment",
    "TutorObservation",
    "TutorStepResult",
    "TutorTurnRecord",
]
