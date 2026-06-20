"""Hidden-rule adapter for the shared tutor-environment contract."""

from __future__ import annotations

from typing import Any

from examples.hiddeb_rule_tutor.hidden_rule_game.env import _is_final_rule_message, _rough_rule_match
from examples.hiddeb_rule_tutor.hidden_rule_game.rules import iter_rule_cycle, sample_examples
from examples.hiddeb_rule_tutor.hidden_rule_metrics import tutor_leakage_penalty, tutor_turn_quality
from examples.tutor.environment import (
    CompressedContext, ContextTarget, EpisodeInput, Participant, StudentAction,
    StudentPool, TutorAction, TutorEnvironment, TutorObservation,
    _AutomaticAdvance, _TaskTransition,
)


class HiddebRuleTutorEnvironment(TutorEnvironment):
    """One step teaches one rule-induction turn and records one tutor sample."""

    def __init__(
        self,
        student_pool: StudentPool,
        *,
        rounds: int = 8,
        examples_per_episode: int = 48,
        tutor_quality_coef: float = 0.80,
        leakage_coef: float = 1.20,
        consistency_coef: float = 0.25,
        task_reward_coef: float = 2.0,
    ) -> None:
        super().__init__(student_pool)
        self.rounds = int(rounds)
        self.examples_per_episode = int(examples_per_episode)
        self.tutor_quality_coef = float(tutor_quality_coef)
        self.leakage_coef = float(leakage_coef)
        self.consistency_coef = float(consistency_coef)
        self.task_reward_coef = float(task_reward_coef)
        self.participant = Participant("student", "default")
        self.rule: Any | None = None
        self.examples: list[dict[str, object]] = []
        self.transcript: list[dict[str, str]] = []
        self.turn_idx = 0

    async def reset(self, episode: EpisodeInput) -> TutorObservation:
        import random

        rng = random.Random(episode.seed)
        self.rule = next(iter_rule_cycle(rng))
        self.examples = sample_examples(self.rule, rng, self.examples_per_episode)
        self.transcript.clear()
        self.turn_idx = 0
        return self._start_episode(await self.next_tutor_observation())

    async def compress_context(self, target: ContextTarget) -> CompressedContext:
        del target
        history = "\n".join(
            f"Student: {turn['student']}\nTutor: {turn['tutor']}"
            for turn in self.transcript[-6:]
        )
        return CompressedContext(
            public_history=history,
            participant_memory="",
            task_summary="Infer the hidden string rule from labeled examples without revealing it directly.",
            legal_actions=("ask for examples", "test: <lowercase-string>", "final_rule: <hypothesis>"),
            truncation_metadata={"transcript_turns": min(6, len(self.transcript))},
        )

    def build_student_prompt(self, participant, context, tutor_action, *, retry_feedback=None) -> str:
        del participant, context
        return (
            f"Tutor reply:\n{tutor_action.text}\n\n"
            f"{retry_feedback or 'Ask for evidence, test a string, or state a final_rule hypothesis.'}"
        )

    def parse_student_action(self, participant, raw_action, context) -> str | None:
        del participant, context
        return raw_action.strip() or None

    def fallback_student_action(self, participant, context) -> str:
        del participant, context
        return "ask for more examples"

    async def apply_tutored_actions(self, tutor_action, student_actions):
        if len(student_actions) != 1:
            raise ValueError("hidden-rule tutoring requires exactly one student")
        student_message = student_actions[0].action
        batch = self.examples[self.turn_idx * 4 : (self.turn_idx + 1) * 4]
        quality = tutor_turn_quality(tutor_action.text, batch)
        leakage = tutor_leakage_penalty(tutor_action.text, batch)
        consistency = _consistency(tutor_action.text, student_message)
        self.transcript.append({"student": student_message, "tutor": tutor_action.text})
        self.turn_idx += 1
        done = self.turn_idx >= self.rounds or _is_final_rule_message(student_message)
        components = {
            "teacher_consistency": self.consistency_coef * consistency,
            "teacher_quality": self.tutor_quality_coef * quality,
            "teacher_leakage": -self.leakage_coef * leakage,
        }
        if done and self.rule is not None:
            guess = student_message.split(":", 1)[-1]
            components["final_task_reward"] = self.task_reward_coef * float(
                _rough_rule_match(guess, self.rule.description)
            )
        return _TaskTransition("student_rule_turn", components, done)

    async def advance_untutored_participants(self, max_steps: int) -> _AutomaticAdvance:
        del max_steps
        return _AutomaticAdvance((), {}, False)

    async def next_tutor_observation(self) -> TutorObservation:
        return TutorObservation(
            context=await self.compress_context("tutor"),
            active_participants=(self.participant,),
        )


def _consistency(advice: str, student_message: str) -> float:
    advice_tokens = {token for token in advice.lower().split() if len(token) > 2}
    student_tokens = {token for token in student_message.lower().split() if len(token) > 2}
    return len(advice_tokens & student_tokens) / max(1, len(student_tokens))
