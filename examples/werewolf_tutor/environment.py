"""Side-aware Werewolf adapter for the shared tutor-environment contract."""

from __future__ import annotations

import re
from typing import Any, Iterable

from examples.tutor.environment import (
    AutomaticEvent, CompressedContext, ContextTarget, EpisodeInput, Participant,
    StudentAction, StudentPool, TutorAction, TutorEnvironment, TutorObservation,
    _AutomaticAdvance, _TaskTransition,
)
from examples.werewolf_tutor.werewolf_env import WerewolfEnv
from examples.werewolf_tutor.werewolf_tutor_metrics import (
    werewolf_action_quality, werewolf_direct_action_penalty, werewolf_tutor_quality,
    werewolf_tutoring_score,
)


SIDES = frozenset({"villager", "werewolf"})


class WerewolfTutorEnvironment(TutorEnvironment):
    """Tutor selected sides and automatically simulate frozen-side game turns."""

    def __init__(
        self,
        student_pool: StudentPool,
        *,
        tutored_sides: Iterable[str] = ("villager",),
        frozen_sides: Iterable[str] = ("werewolf",),
        env_kwargs: dict[str, Any] | None = None,
        max_episode_turns: int = 30,
        use_privileged_teacher_observation: bool = False,
        consistency_coef: float = 0.15,
        reasoning_coef: float = 0.20,
        tutor_quality_coef: float = 0.60,
        action_quality_coef: float = 0.80,
        leakage_coef: float = 1.20,
        direct_action_penalty: float = 0.70,
        task_reward_coef: float = 2.0,
        invalid_action_penalty: float = 0.30,
        max_auto_steps: int = 64,
    ) -> None:
        super().__init__(student_pool, max_auto_steps=max_auto_steps)
        self.tutored_sides = _validate_sides(tutored_sides, "tutored_sides")
        self.frozen_sides = _validate_sides(frozen_sides, "frozen_sides", allow_empty=True)
        if self.tutored_sides & self.frozen_sides:
            raise ValueError("tutored_sides and frozen_sides must not overlap")
        if self.tutored_sides | self.frozen_sides != SIDES:
            raise ValueError("tutored_sides and frozen_sides must cover villager and werewolf")
        self.env_kwargs = dict(env_kwargs or {})
        self.max_episode_turns = int(max_episode_turns)
        self.use_privileged_teacher_observation = bool(use_privileged_teacher_observation)
        self.consistency_coef = float(consistency_coef)
        self.reasoning_coef = float(reasoning_coef)
        self.tutor_quality_coef = float(tutor_quality_coef)
        self.action_quality_coef = float(action_quality_coef)
        self.leakage_coef = float(leakage_coef)
        self.direct_action_penalty = float(direct_action_penalty)
        self.task_reward_coef = float(task_reward_coef)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.env: WerewolfEnv | None = None
        self.obs = ""
        self.guide = ""
        self.info: dict[str, Any] = {}
        self.public_events: list[str] = []
        self.turn_metrics: list[dict[str, float]] = []
        self.initial_automatic_events: tuple[AutomaticEvent, ...] = ()

    async def reset(self, episode: EpisodeInput) -> TutorObservation:
        kwargs = self.env_kwargs | dict(episode.task_data.get("env_kwargs", {}))
        self.env = WerewolfEnv(**kwargs)
        self.obs, self.guide, self.info = await self.env.sreset(seed=episode.seed)
        self.public_events.clear()
        self.turn_metrics.clear()
        self.initial_automatic_events = ()
        advance = await self._advance_until_tutored(self.max_auto_steps)
        self.initial_automatic_events = advance.events
        if advance.done:
            raise RuntimeError("episode ended before a tutor-eligible participant acted")
        return self._start_episode(await self.next_tutor_observation())

    async def compress_context(self, target: ContextTarget) -> CompressedContext:
        env = self._require_env()
        side = _side_for_role(str(env.agent_role))
        teacher_view = self.info.get("teacher_observation", self.obs)
        state = teacher_view if target == "tutor" and self.use_privileged_teacher_observation else self.obs
        return CompressedContext(
            public_history="\n".join(self.public_events[-8:]),
            participant_memory="",
            task_summary=f"Active side: {side}\n{state}\n\nAction guide:\n{self.guide}",
            legal_actions=tuple(env._get_valid_actions()),
            truncation_metadata={"public_events": min(8, len(self.public_events))},
        )

    def build_student_prompt(self, participant, context, tutor_action, *, retry_feedback=None) -> str:
        del participant, context
        guidance = tutor_action.text or "No tutor guidance is available for this turn."
        correction = f"\n\n{retry_feedback}" if retry_feedback else ""
        return (
            f"{self.obs}\n\nTeacher guidance:\n{guidance}\n\n{self.guide}\n\n"
            "Choose one legal action. Use <answer>...</answer>." + correction
        )

    def parse_student_action(self, participant, raw_action, context) -> str | None:
        del participant, context
        env = self._require_env()
        action = _strip_answer(raw_action).lower()
        if action in env._get_valid_actions() or action.startswith("say "):
            return f"<answer>{action}</answer>"
        if env.phase == "discussion" and action:
            return f"<answer>say {action}</answer>"
        return None

    def fallback_student_action(self, participant, context) -> str:
        del participant, context
        valid = [action for action in self._require_env()._get_valid_actions() if action != "skip"]
        return f"<answer>{valid[0] if valid else 'skip'}</answer>"

    async def apply_tutored_actions(self, tutor_action, student_actions):
        if len(student_actions) != 1:
            raise ValueError("Werewolf requires one active participant per turn")
        return await self._apply_action(tutor_action, student_actions[0], tutored=True)

    async def advance_untutored_participants(self, max_steps: int) -> _AutomaticAdvance:
        return await self._advance_until_tutored(max_steps)

    async def next_tutor_observation(self) -> TutorObservation:
        env = self._require_env()
        participant = Participant(
            participant_id=str(env.agent_player), policy_id=_side_for_role(str(env.agent_role)),
            role=str(env.agent_role), team=_side_for_role(str(env.agent_role)),
        )
        return TutorObservation(await self.compress_context("tutor"), (participant,))

    async def _advance_until_tutored(self, max_steps: int) -> _AutomaticAdvance:
        events: list[AutomaticEvent] = []
        for _ in range(max_steps):
            env = self._require_env()
            if _side_for_role(str(env.agent_role)) in self.tutored_sides:
                return _AutomaticAdvance(tuple(events), {}, False)
            participant = Participant(str(env.agent_player), _side_for_role(str(env.agent_role)), str(env.agent_role), _side_for_role(str(env.agent_role)), False)
            context = await self.compress_context("student")
            action = await self.generate_legal_student_action(participant, context, TutorAction(""))
            transition = await self._apply_action(TutorAction(""), action, tutored=False)
            events.append(AutomaticEvent(participant, action.action, transition.event, transition.reward_components))
            if transition.done:
                return _AutomaticAdvance(tuple(events), {}, True)
        raise RuntimeError(f"exceeded max_auto_steps={max_steps} while advancing frozen sides")

    async def _apply_action(self, tutor_action: TutorAction, action: StudentAction, *, tutored: bool) -> _TaskTransition:
        env = self._require_env()
        role, phase = str(env.agent_role), str(env.phase)
        self.obs, self.guide, step_reward, done, _, self.info = await env.step(("", [action.action]))
        event = str(self.info.get("event", ""))
        self.public_events.append(event)
        parsed = _strip_answer(action.action)
        quality = werewolf_action_quality(parsed_action=parsed, step_reward=step_reward, role=role, phase=phase)
        metrics = {
            "werewolf_action_quality": quality,
            "teacher_quality": werewolf_tutor_quality(tutor_action.text) if tutored else 0.0,
            "teacher_leakage": _leakage(tutor_action.text) if tutored else 0.0,
            "teacher_direct_action": werewolf_direct_action_penalty(tutor_action.text) if tutored else 0.0,
        }
        self.turn_metrics.append(metrics)
        components: dict[str, float] = {}
        if tutored:
            components = {
                "teacher_consistency": self.consistency_coef * _consistency(tutor_action.text, parsed),
                "teacher_reasoning": self.reasoning_coef * _reasoning(tutor_action.text),
                "teacher_quality": self.tutor_quality_coef * metrics["teacher_quality"],
                "action_quality": self.action_quality_coef * quality,
                "teacher_leakage": -self.leakage_coef * metrics["teacher_leakage"],
                "teacher_direct_action": -self.direct_action_penalty * metrics["teacher_direct_action"],
                "student_invalid_action": -self.invalid_action_penalty * float(action.used_fallback),
            }
        done = bool(done or len(self.turn_metrics) >= self.max_episode_turns)
        if done and tutored:
            components["final_task_reward"] = self.task_reward_coef * werewolf_tutoring_score(env, self.turn_metrics)
        return _TaskTransition(event, components, done)

    def _require_env(self) -> WerewolfEnv:
        if self.env is None:
            raise RuntimeError("reset() must be called before using the Werewolf environment")
        return self.env


def _validate_sides(sides: Iterable[str], field: str, *, allow_empty: bool = False) -> frozenset[str]:
    result = frozenset(str(side).lower() for side in sides)
    invalid = result - SIDES
    if invalid or (not result and not allow_empty):
        raise ValueError(f"{field} must be a non-empty subset of {sorted(SIDES)}")
    return result


def _side_for_role(role: str) -> str:
    return "werewolf" if role == "werewolf" else "villager"


def _strip_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else (text or "").strip()


def _consistency(advice: str, action: str) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", action.lower()))
    return len(tokens & set(re.findall(r"[a-z0-9]+", advice.lower()))) / max(1, len(tokens))


def _reasoning(advice: str) -> float:
    return min(1.0, sum(word in advice.lower() for word in ("because", "evidence", "risk", "uncertain")) / 2.0)


def _leakage(advice: str) -> float:
    return float(bool(re.search(r"player\s*\d+\s+(?:is|must be).*(?:werewolf|villager)", advice.lower())))
