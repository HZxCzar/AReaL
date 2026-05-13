from __future__ import annotations

from .types import (
    EpisodeArtifact,
    RewardAssignment,
    TurnArtifact,
    TurnTrace,
)


class EpisodeRewardComputer:
    def __init__(
        self,
        *,
        success_reward: float,
        leak_penalty: float,
        outcome_prior_turn_weight: float = 0.1,
        outcome_credit_gamma: float = 0.9,
        early_success_bonus: float = 0.0,
        turn_penalty: float = 0.0,
        length_penalty_threshold_chars: int = 0,
        length_penalty_per_100_chars: float = 0.0,
        length_penalty_min: float = 0.0,
    ) -> None:
        self.success_reward = success_reward
        self.leak_penalty = leak_penalty
        self.outcome_prior_turn_weight = outcome_prior_turn_weight
        self.outcome_credit_gamma = outcome_credit_gamma
        self.early_success_bonus = early_success_bonus
        self.turn_penalty = turn_penalty
        self.length_penalty_threshold_chars = length_penalty_threshold_chars
        self.length_penalty_per_100_chars = length_penalty_per_100_chars
        self.length_penalty_min = length_penalty_min

    async def compute(
        self,
        episode: EpisodeArtifact,
        *,
        pairwise_rewards: dict[int, float] | None = None,
    ) -> list[RewardAssignment]:
        pairwise_rewards = pairwise_rewards or {}
        success_artifact = self._success_artifact(episode)
        success_credits = self._success_credits(episode.turns, success_artifact)
        assignments: list[RewardAssignment] = []
        for artifact in episode.turns:
            components: dict[str, float] = {}
            if artifact.leak_result.leaked:
                components["leak"] = self.leak_penalty
            success_credit = success_credits.get(artifact.turn_idx, 0.0)
            if success_credit:
                components["success_credit"] = success_credit
            if self.turn_penalty:
                components["turn_penalty"] = self.turn_penalty
            length_penalty = self._length_penalty(artifact.tutor_visible_output)
            if length_penalty:
                components["length_penalty"] = length_penalty
            pairwise_reward = pairwise_rewards.get(artifact.turn_idx, 0.0)
            if pairwise_reward:
                components["pairwise"] = float(pairwise_reward)

            reward = float(sum(components.values()))
            assignments.append(
                RewardAssignment(
                    reward=reward,
                    reward_components=components,
                )
            )
        return assignments

    def _success_artifact(
        self, episode: EpisodeArtifact
    ) -> TurnArtifact | None:
        if episode.termination_reason != "success":
            return None
        for artifact in episode.turns:
            if artifact.leak_result.leaked:
                continue
            if artifact.judge_result is not None and artifact.judge_result.correct:
                return artifact
        return None

    def _success_credits(
        self, turns: list[TurnArtifact], success_artifact: TurnArtifact | None
    ) -> dict[int, float]:
        if success_artifact is None:
            return {}
        max_turns = max(1, int(success_artifact.tutor_state.max_turns))
        success_turn = int(success_artifact.turn_idx)
        if max_turns <= 1:
            early_fraction = 0.0
        else:
            early_fraction = max(0.0, (max_turns - success_turn) / (max_turns - 1))
        budget = self.success_reward + self.early_success_bonus * early_fraction

        weights: dict[int, float] = {}
        for artifact in turns:
            if artifact.leak_result.leaked:
                continue
            turn_idx = int(artifact.turn_idx)
            if turn_idx > success_turn:
                continue
            distance = success_turn - turn_idx
            if distance == 0:
                weight = 1.0
            else:
                weight = self.outcome_prior_turn_weight * (
                    self.outcome_credit_gamma ** distance
                )
            if weight > 0:
                weights[turn_idx] = weight

        total_weight = sum(weights.values())
        if total_weight <= 0:
            return {}
        return {
            turn_idx: float(budget * weight / total_weight)
            for turn_idx, weight in weights.items()
        }

    def _length_penalty(self, tutor_visible_output: str) -> float:
        threshold = int(self.length_penalty_threshold_chars)
        per_100_chars = float(self.length_penalty_per_100_chars)
        if threshold <= 0 or per_100_chars == 0.0:
            return 0.0
        excess_chars = max(0, len(tutor_visible_output or "") - threshold)
        if excess_chars <= 0:
            return 0.0
        penalty = per_100_chars * (excess_chars / 100.0)
        if per_100_chars < 0:
            return max(min(0.0, float(self.length_penalty_min)), penalty)
        return min(max(0.0, float(self.length_penalty_min)), penalty)


def artifact_to_trace(
    artifact: TurnArtifact, assignment: RewardAssignment
) -> TurnTrace:
    judge_correct = (
        bool(artifact.judge_result.correct)
        if artifact.judge_result is not None
        else False
    )
    judge_feedback = (
        artifact.judge_result.feedback if artifact.judge_result is not None else ""
    )
    return TurnTrace(
        turn_idx=artifact.turn_idx,
        tutor_state=artifact.tutor_state,
        tutor_raw_output=artifact.tutor_raw_output,
        tutor_visible_output=artifact.tutor_visible_output,
        leaked=artifact.leak_result.leaked,
        student_output=artifact.student_output,
        judge_correct=judge_correct,
        judge_feedback=judge_feedback,
        reward=assignment.reward,
        reward_components=assignment.reward_components,
        public_history_before=artifact.public_history_before,
        public_history_after=artifact.public_history_after,
    )
