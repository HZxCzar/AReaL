"""Hanabi adapter for the shared tutor-environment contract."""

from __future__ import annotations

import re
from typing import Any

from examples.hanabi_tutor.hanabi_env import HanabiEnv
from examples.hanabi_tutor.hanabi_tutor_metrics import (
    hanabi_action_quality,
    hanabi_direct_action_penalty,
    hanabi_tutor_quality,
    hanabi_tutoring_score,
)
from examples.tutor.environment import (
    CompressedContext,
    ContextTarget,
    EpisodeInput,
    Participant,
    StudentAction,
    StudentPool,
    TutorAction,
    TutorEnvironment,
    TutorObservation,
    _AutomaticAdvance,
    _TaskTransition,
)


class HanabiTutorEnvironment(TutorEnvironment):
    """One step tutors the current Hanabi player and applies one legal action."""

    def __init__(
        self,
        student_pool: StudentPool,
        *,
        env_kwargs: dict[str, Any] | None = None,
        max_episode_turns: int = 25,
        turn_discount: float = 0.98,
        consistency_coef: float = 0.25,
        reasoning_coef: float = 0.20,
        tutor_quality_coef: float = 0.60,
        action_quality_coef: float = 0.80,
        leakage_coef: float = 1.20,
        direct_action_penalty: float = 0.50,
        task_reward_coef: float = 2.0,
        invalid_action_penalty: float = 0.30,
        max_auto_steps: int = 64,
    ) -> None:
        super().__init__(student_pool, max_auto_steps=max_auto_steps)
        self.env_kwargs = dict(env_kwargs or {})
        self.max_episode_turns = int(max_episode_turns)
        self.turn_discount = float(turn_discount)
        self.consistency_coef = float(consistency_coef)
        self.reasoning_coef = float(reasoning_coef)
        self.tutor_quality_coef = float(tutor_quality_coef)
        self.action_quality_coef = float(action_quality_coef)
        self.leakage_coef = float(leakage_coef)
        self.direct_action_penalty = float(direct_action_penalty)
        self.task_reward_coef = float(task_reward_coef)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.env: HanabiEnv | None = None
        self.obs = ""
        self.guide = ""
        self.info: dict[str, Any] = {}
        self.turn_metrics: list[dict[str, float]] = []

    async def reset(self, episode: EpisodeInput) -> TutorObservation:
        kwargs = self.env_kwargs | dict(episode.task_data.get("env_kwargs", {}))
        self.env = HanabiEnv(**kwargs)
        self.obs, self.guide, self.info = await self.env.sreset(seed=episode.seed)
        self.turn_metrics.clear()
        return self._start_episode(await self.next_tutor_observation())

    async def compress_context(self, target: ContextTarget) -> CompressedContext:
        del target
        env = self._require_env()
        recent_events = "\n".join(env.get_trajectory()[-8:])
        return CompressedContext(
            public_history=recent_events,
            participant_memory="",
            task_summary=f"{self.obs}\n\nAction guide:\n{self.guide}",
            legal_actions=("play <card>", "discard <card>", "hint <player> <color|rank> <value>"),
            truncation_metadata={"trajectory_events": min(8, len(env.get_trajectory()))},
        )

    def build_student_prompt(
        self,
        participant: Participant,
        context: CompressedContext,
        tutor_action: TutorAction,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        del participant, context
        if retry_feedback:
            return f"{retry_feedback}\n\n{self.guide}"
        return (
            f"{self.obs}\n\nTeacher guidance:\n{tutor_action.text}\n\n{self.guide}\n\n"
            "Choose exactly one legal Hanabi action. Use <answer>...</answer>."
        )

    def parse_student_action(
        self, participant: Participant, raw_action: str, context: CompressedContext
    ) -> str | None:
        del participant, context
        action = self._require_env()._parse_action(_strip_answer(raw_action))
        if action.startswith(("play ", "discard ", "hint ")):
            return action
        return None

    def fallback_student_action(
        self, participant: Participant, context: CompressedContext
    ) -> str:
        del participant, context
        env = self._require_env()
        return "discard 1" if env.hands.get(env.agent_player) else "hint player2 color red"

    async def apply_tutored_actions(
        self, tutor_action: TutorAction, student_actions: tuple[StudentAction, ...]
    ) -> _TaskTransition:
        if len(student_actions) != 1:
            raise ValueError("Hanabi expects exactly one active student")
        env = self._require_env()
        action = student_actions[0].action
        self.obs, self.guide, step_reward, done, _, self.info = await env.step(("", [action]))
        event = str(self.info.get("event", ""))
        quality = hanabi_action_quality(event, float(step_reward), action)
        tutor_quality = hanabi_tutor_quality(tutor_action.text)
        leakage = _hanabi_leakage(tutor_action.text)
        direct_action = hanabi_direct_action_penalty(tutor_action.text)
        consistency = _consistency(tutor_action.text, action)
        invalid = float(quality < 0.0)
        components = {
            "teacher_consistency": self.consistency_coef * consistency,
            "teacher_reasoning": self.reasoning_coef * _reasoning(tutor_action.text),
            "teacher_quality": self.tutor_quality_coef * tutor_quality,
            "action_quality": self.action_quality_coef * quality,
            "teacher_leakage": -self.leakage_coef * leakage,
            "student_invalid_action": -self.invalid_action_penalty * invalid,
            "teacher_direct_action": -self.direct_action_penalty * direct_action,
        }
        self.turn_metrics.append({
            "hanabi_action_quality": quality,
            "teacher_quality": tutor_quality,
            "teacher_leakage": leakage,
            "teacher_direct_action": direct_action,
        })
        reached_limit = len(self.turn_metrics) >= self.max_episode_turns
        done = bool(done or reached_limit)
        if done:
            components["final_task_reward"] = self.task_reward_coef * hanabi_tutoring_score(env, self.turn_metrics)
        return _TaskTransition(event, components, done)

    async def advance_untutored_participants(self, max_steps: int) -> _AutomaticAdvance:
        del max_steps
        return _AutomaticAdvance((), {}, False)

    async def next_tutor_observation(self) -> TutorObservation:
        env = self._require_env()
        return TutorObservation(
            context=await self.compress_context("tutor"),
            active_participants=(Participant(env.agent_player, "default", team="hanabi"),),
        )

    def _require_env(self) -> HanabiEnv:
        if self.env is None:
            raise RuntimeError("reset() must be called before using the Hanabi environment")
        return self.env


def _strip_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else (text or "").strip()


def _consistency(advice: str, action: str) -> float:
    action_tokens = set(re.findall(r"[a-z0-9]+", action.lower()))
    if not action_tokens:
        return 0.0
    advice_tokens = set(re.findall(r"[a-z0-9]+", advice.lower()))
    return len(action_tokens & advice_tokens) / len(action_tokens)


def _reasoning(advice: str) -> float:
    markers = ("because", "if", "risk", "public", "alternative", "consider")
    return min(1.0, sum(marker in advice.lower() for marker in markers) / 3.0)


def _hanabi_leakage(advice: str) -> float:
    patterns = (
        r"\byour\s+(?:card|hand).*\b(?:red|blue|green|yellow|white|[rbgyw][1-5])\b",
        r"\bposition\s*\d+\s+is\s+(?:red|blue|green|yellow|white|[rbgyw][1-5])\b",
    )
    return float(any(re.search(pattern, advice.lower()) for pattern in patterns))
