from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from examples.common.parsing import join_errors, parse_json_dict
from examples.tutor.core.auxiliary import call_auxiliary_text
from examples.tutor.core.text import strip_reasoning_for_context
from examples.tutor.core.types import (
    EpisodeArtifact,
    JudgeResult,
    LeakCheckResult,
    StudentTurnState,
    TutorTurnState,
    TurnArtifact,
)
from examples.tutor.prompts import (
    PAIRWISE_STUDENT_COMPARISON_SYSTEM_PROMPT,
    PAIRWISE_STUDENT_COMPARISON_USER_TEMPLATE,
    render_prompt,
)

PairwiseOutcome = Literal["current", "reference", "tie", "skipped", "failure"]

TutorGenerator = Callable[[TutorTurnState, int], Awaitable[tuple[Any, str]]]
StudentRunner = Callable[[StudentTurnState], Awaitable[tuple[str, str | None]]]
LeakChecker = Callable[[str, str, str], Awaitable[LeakCheckResult]]
AnswerScorer = Callable[[str, str, str], JudgeResult]


@dataclass(slots=True)
class PairwiseJudgeResult:
    raw_output: str
    winner: Literal["A", "B", "tie", "invalid"]
    confidence: float
    feedback: str
    parse_error: str | None
    raw_result: dict[str, Any]


@dataclass(slots=True)
class ReferenceTurnArtifact:
    turn_idx: int
    reference_version: int
    tutor_raw_output: str
    tutor_visible_output: str
    leak_result: LeakCheckResult
    student_state: StudentTurnState | None = None
    student_output: str = ""
    student_error: str | None = None
    judge_result: JudgeResult | None = None


@dataclass(slots=True)
class PairwiseTurnResult:
    turn_idx: int
    reference_version: int
    outcome: PairwiseOutcome
    reward: float
    compared: bool
    reason: str
    reference: ReferenceTurnArtifact | None = None
    judge_result: PairwiseJudgeResult | None = None
    current_label: str = ""
    reference_label: str = ""


def parse_pairwise_judge_result(raw_output: str) -> PairwiseJudgeResult:
    parsed, parse_error = parse_json_dict(raw_output)
    if not isinstance(parsed, dict):
        return PairwiseJudgeResult(
            raw_output=raw_output,
            winner="invalid",
            confidence=0.0,
            feedback="Failed to parse pairwise judge output.",
            parse_error=parse_error or "Expected JSON object.",
            raw_result={},
        )

    winner_raw = parsed.get("winner", "")
    winner: Literal["A", "B", "tie", "invalid"]
    if isinstance(winner_raw, str):
        normalized = winner_raw.strip().lower()
        if normalized == "a":
            winner = "A"
        elif normalized == "b":
            winner = "B"
        elif normalized in {"tie", "equal", "unclear", "none"}:
            winner = "tie"
        else:
            winner = "invalid"
            parse_error = join_errors(parse_error, '"winner" must be "A", "B", or "tie"')
    else:
        winner = "invalid"
        parse_error = join_errors(parse_error, '"winner" must be a string')

    confidence_raw = parsed.get("confidence", 0.0)
    confidence = (
        float(confidence_raw) if isinstance(confidence_raw, int | float) else 0.0
    )
    confidence = max(0.0, min(1.0, confidence))
    feedback_raw = parsed.get("feedback", "")
    feedback = feedback_raw if isinstance(feedback_raw, str) else str(feedback_raw)
    return PairwiseJudgeResult(
        raw_output=raw_output,
        winner=winner,
        confidence=confidence,
        feedback=feedback,
        parse_error=parse_error,
        raw_result=parsed,
    )


class PairwiseTutorEvaluator:
    def __init__(
        self,
        *,
        reward_scale: float,
        reward_caller: Any,
        generate_reference_tutor: TutorGenerator,
        run_student: StudentRunner,
        run_leak_check: LeakChecker,
        score_answer: AnswerScorer,
        compare_all_turns: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        self.reward_scale = float(reward_scale)
        self.reward_caller = reward_caller
        self.generate_reference_tutor = generate_reference_tutor
        self.run_student = run_student
        self.run_leak_check = run_leak_check
        self.score_answer = score_answer
        self.compare_all_turns = compare_all_turns
        self.rng = rng or random.Random()

    async def evaluate(
        self, episode: EpisodeArtifact, *, reference_version: int
    ) -> list[PairwiseTurnResult]:
        turns = list(episode.turns)
        if not self.compare_all_turns:
            comparable_turns = [
                turn
                for turn in turns
                if not turn.leak_result.leaked and turn.student_state
            ]
            turns = comparable_turns[-1:] if comparable_turns else []
        return [
            await self.evaluate_turn(
                episode, turn, reference_version=reference_version
            )
            for turn in turns
        ]

    async def evaluate_turn(
        self,
        episode: EpisodeArtifact,
        current_turn: TurnArtifact,
        *,
        reference_version: int,
    ) -> PairwiseTurnResult:
        if current_turn.leak_result.leaked:
            return self._skipped_result(
                current_turn,
                reference_version,
                reason="current_leaked",
            )

        if current_turn.student_state is None or current_turn.judge_result is None:
            return self._failure_result(
                current_turn, reference_version, reason="missing_current_student_state"
            )
        if current_turn.student_error:
            return self._failure_result(
                current_turn, reference_version, reason="current_student_failed"
            )
        if not current_turn.student_output.strip():
            return self._failure_result(
                current_turn, reference_version, reason="empty_current_student_output"
            )

        try:
            _, reference_raw_output = await self.generate_reference_tutor(
                current_turn.tutor_state, reference_version
            )
        except Exception as exc:
            setattr(exc, "_fatal_rollout_error", True)
            raise

        reference_visible_output = strip_reasoning_for_context(reference_raw_output)
        reference_leak = await self.run_leak_check(
            episode.task, episode.ground_truth, reference_visible_output
        )
        reference = ReferenceTurnArtifact(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            tutor_raw_output=reference_raw_output,
            tutor_visible_output=reference_visible_output,
            leak_result=reference_leak,
        )
        if reference_leak.leaked:
            return self._winner_result(
                current_turn,
                reference_version,
                outcome="current",
                reason="reference_leaked",
                reference=reference,
            )

        reference_student_state = StudentTurnState(
            task=current_turn.student_state.task,
            public_history=current_turn.student_state.public_history,
            previous_student_output=current_turn.student_state.previous_student_output,
            latest_tutor_visible_output=reference_visible_output,
        )
        reference_student_output, reference_student_error = await self.run_student(
            reference_student_state
        )
        reference.student_state = reference_student_state
        reference.student_output = strip_reasoning_for_context(reference_student_output)
        reference.student_error = reference_student_error
        if reference_student_error:
            return self._failure_result(
                current_turn,
                reference_version,
                reason="reference_student_failed",
                reference=reference,
            )
        if not reference.student_output.strip():
            return self._failure_result(
                current_turn,
                reference_version,
                reason="empty_reference_student_output",
                reference=reference,
            )

        reference.judge_result = self.score_answer(
            episode.task, episode.ground_truth, reference.student_output
        )

        current_correct = bool(current_turn.judge_result.correct)
        reference_correct = bool(reference.judge_result.correct)
        if current_correct != reference_correct:
            return self._winner_result(
                current_turn,
                reference_version,
                outcome="current" if current_correct else "reference",
                reason="exact_correctness",
                reference=reference,
            )

        return await self._compare_student_outputs(
            episode,
            current_turn,
            reference_version=reference_version,
            reference=reference,
        )

    async def _compare_student_outputs(
        self,
        episode: EpisodeArtifact,
        current_turn: TurnArtifact,
        *,
        reference_version: int,
        reference: ReferenceTurnArtifact,
    ) -> PairwiseTurnResult:
        current_is_a = self.rng.choice([True, False])
        if current_is_a:
            reply_a = current_turn.student_output
            reply_b = reference.student_output
            current_label = "A"
            reference_label = "B"
        else:
            reply_a = reference.student_output
            reply_b = current_turn.student_output
            current_label = "B"
            reference_label = "A"

        prompt = render_prompt(
            PAIRWISE_STUDENT_COMPARISON_USER_TEMPLATE,
            task=episode.task,
            ground_truth=episode.ground_truth,
            public_history=current_turn.student_state.public_history.summary
            if current_turn.student_state is not None
            else "",
            previous_student_output=current_turn.student_state.previous_student_output
            if current_turn.student_state is not None
            else "",
            student_reply_a=reply_a,
            student_reply_b=reply_b,
        )
        judge_call = await call_auxiliary_text(
            self.reward_caller,
            [
                {
                    "role": "system",
                    "content": PAIRWISE_STUDENT_COMPARISON_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
        )
        if judge_call.error:
            return self._failure_result(
                current_turn,
                reference_version,
                reason=f"pairwise_judge_failed: {judge_call.error}",
                reference=reference,
                current_label=current_label,
                reference_label=reference_label,
            )
        judge_result = parse_pairwise_judge_result(judge_call.text)
        if judge_result.winner == "invalid":
            return self._failure_result(
                current_turn,
                reference_version,
                reason="pairwise_judge_parse_failed",
                reference=reference,
                judge_result=judge_result,
                current_label=current_label,
                reference_label=reference_label,
            )
        if judge_result.winner == "tie":
            outcome: PairwiseOutcome = "tie"
        elif judge_result.winner == current_label:
            outcome = "current"
        else:
            outcome = "reference"
        return self._winner_result(
            current_turn,
            reference_version,
            outcome=outcome,
            reason="pairwise_judge",
            reference=reference,
            judge_result=judge_result,
            current_label=current_label,
            reference_label=reference_label,
        )

    def _winner_result(
        self,
        current_turn: TurnArtifact,
        reference_version: int,
        *,
        outcome: Literal["current", "reference", "tie"],
        reason: str,
        reference: ReferenceTurnArtifact | None,
        judge_result: PairwiseJudgeResult | None = None,
        current_label: str = "",
        reference_label: str = "",
    ) -> PairwiseTurnResult:
        if outcome == "current":
            reward = self.reward_scale
        elif outcome == "reference":
            reward = -self.reward_scale
        else:
            reward = 0.0
        return PairwiseTurnResult(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            outcome=outcome,
            reward=reward,
            compared=True,
            reason=reason,
            reference=reference,
            judge_result=judge_result,
            current_label=current_label,
            reference_label=reference_label,
        )

    def _failure_result(
        self,
        current_turn: TurnArtifact,
        reference_version: int,
        *,
        reason: str,
        reference: ReferenceTurnArtifact | None = None,
        judge_result: PairwiseJudgeResult | None = None,
        current_label: str = "",
        reference_label: str = "",
    ) -> PairwiseTurnResult:
        return PairwiseTurnResult(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            outcome="failure",
            reward=0.0,
            compared=False,
            reason=reason,
            reference=reference,
            judge_result=judge_result,
            current_label=current_label,
            reference_label=reference_label,
        )

    def _skipped_result(
        self,
        current_turn: TurnArtifact,
        reference_version: int,
        *,
        reason: str,
    ) -> PairwiseTurnResult:
        return PairwiseTurnResult(
            turn_idx=current_turn.turn_idx,
            reference_version=reference_version,
            outcome="skipped",
            reward=0.0,
            compared=False,
            reason=reason,
        )
