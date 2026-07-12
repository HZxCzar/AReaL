from __future__ import annotations

from .types import (
    EpisodeArtifact,
    LeakCheckResult,
    RewardAssignment,
    TurnArtifact,
    TurnTrace,
)


class EpisodeRewardComputer:
    def __init__(
        self,
        *,
        success_reward: float,
        leak_penalty: float | None,
        leak_penalty_mode: str = "binary",
        leak_penalty_final_answer: float | None = None,
        leak_penalty_compute: float | None = None,
        leak_penalty_formula: float | None = None,
        leak_penalty_aggregation: str = "turn",
        format_error_penalty: float = 0.0,
        leaked_success_reward_scale: float = 1.0,
        assign_success_reward: bool = False,
        outcome_prior_turn_weight: float = 0.1,
        outcome_credit_gamma: float = 0.9,
        early_success_bonus: float = 0.0,
        max_turn_penalty: float = 0.0,
        enable_turn_penalty: bool = False,
        turn_penalty: float = 0.0,
        length_penalty_threshold_chars: int = 0,
        length_penalty_per_100_chars: float = 0.0,
        length_penalty_min: float = 0.0,
    ) -> None:
        if leak_penalty_mode not in {"binary", "staged", "rawbase"}:
            raise ValueError(
                "leak_penalty_mode must be 'binary', 'staged', or 'rawbase'."
            )
        if leak_penalty_mode in {"binary", "rawbase"} and leak_penalty is None:
            raise ValueError(
                "leak_penalty must be set in binary/rawbase leak penalty mode."
            )
        if leak_penalty_aggregation not in {"turn", "episode"}:
            raise ValueError("leak_penalty_aggregation must be 'turn' or 'episode'.")
        if format_error_penalty > 0.0:
            raise ValueError("format_error_penalty must be <= 0.")
        if leaked_success_reward_scale < 0.0:
            raise ValueError("leaked_success_reward_scale must be >= 0.")
        staged_penalties = {
            1: ("leak_final_answer", leak_penalty_final_answer),
            2: ("leak_compute", leak_penalty_compute),
            3: ("leak_formula", leak_penalty_formula),
        }
        if leak_penalty_mode == "staged":
            missing = [
                name for name, value in staged_penalties.values() if value is None
            ]
            if missing:
                raise ValueError(
                    "staged leak penalty mode requires explicit values for "
                    f"{', '.join(missing)}."
                )

        self.success_reward = success_reward
        self.leak_penalty_mode = leak_penalty_mode
        self.leak_penalty = float(leak_penalty) if leak_penalty is not None else 0.0
        self.staged_leak_penalties = {
            level: (name, float(value) if value is not None else 0.0)
            for level, (name, value) in staged_penalties.items()
        }
        self.assign_success_reward = assign_success_reward
        self.leak_penalty_aggregation = leak_penalty_aggregation
        self.format_error_penalty = float(format_error_penalty)
        self.leaked_success_reward_scale = float(leaked_success_reward_scale)
        self.outcome_prior_turn_weight = outcome_prior_turn_weight
        self.outcome_credit_gamma = outcome_credit_gamma
        self.early_success_bonus = early_success_bonus
        self.max_turn_penalty = float(max_turn_penalty)
        self.enable_turn_penalty = enable_turn_penalty
        self.turn_penalty = turn_penalty
        self.length_penalty_threshold_chars = length_penalty_threshold_chars
        self.length_penalty_per_100_chars = length_penalty_per_100_chars
        self.length_penalty_min = length_penalty_min

    async def compute(self, episode: EpisodeArtifact) -> list[RewardAssignment]:
        success_artifact = self._success_artifact(episode)
        episode_leaked = any(artifact.leak_result.leaked for artifact in episode.turns)
        success_scale = self.leaked_success_reward_scale if episode_leaked else 1.0
        success_credits = self._success_credits(
            episode.turns,
            success_artifact,
            success_reward_scale=success_scale,
        )
        episode_leak_component = (
            self._episode_leak_component(episode.turns)
            if self.leak_penalty_aggregation == "episode"
            else None
        )
        max_turn_artifact = (
            episode.turns[-1]
            if episode.termination_reason == "max_turns" and episode.turns
            else None
        )
        assignments: list[RewardAssignment] = []
        for artifact in episode.turns:
            components: dict[str, float] = {}
            if episode_leak_component is None:
                leak_component = self._leak_component(artifact.leak_result)
            elif artifact.turn_idx == episode_leak_component[0]:
                leak_component = episode_leak_component[1]
            else:
                leak_component = None
            if leak_component is not None:
                name, value = leak_component
                components[name] = value
            if artifact.tutor_format_error and self.format_error_penalty:
                components["format_error"] = self.format_error_penalty
            success_credit = success_credits.get(artifact.turn_idx, 0.0)
            if success_credit:
                components["success_credit"] = success_credit
            if artifact is max_turn_artifact and self.max_turn_penalty:
                components["max_turn_penalty"] = self.max_turn_penalty
            if self.enable_turn_penalty and self.turn_penalty:
                components["turn_penalty"] = self.turn_penalty
            length_penalty = self._length_penalty(artifact.tutor_visible_output)
            if length_penalty:
                components["length_penalty"] = length_penalty
            reward = float(sum(components.values()))
            assignments.append(
                RewardAssignment(
                    reward=reward,
                    reward_components=components,
                )
            )
        return assignments

    def _leak_component(self, result: LeakCheckResult) -> tuple[str, float] | None:
        if self.leak_penalty_mode in {"binary", "rawbase"}:
            if result.leaked:
                return "leak", self.leak_penalty
            return None

        level = result.leak_level
        if level is None:
            level = 1 if result.leaked else 4
        if level == 4:
            return None
        name, value = self.staged_leak_penalties.get(
            int(level), self.staged_leak_penalties[1]
        )
        return name, value

    def _episode_leak_component(
        self, turns: list[TurnArtifact]
    ) -> tuple[int, tuple[str, float]] | None:
        best_turn_idx: int | None = None
        best_level: int | None = None
        best_component: tuple[str, float] | None = None
        for artifact in turns:
            component = self._leak_component(artifact.leak_result)
            if component is None:
                continue
            level = self._normalized_leak_level(artifact.leak_result)
            if best_component is None or level < int(best_level):
                best_turn_idx = int(artifact.turn_idx)
                best_level = level
                best_component = component
        if best_turn_idx is None or best_component is None:
            return None
        return best_turn_idx, best_component

    def _normalized_leak_level(self, result: LeakCheckResult) -> int:
        if self.leak_penalty_mode != "staged":
            return 1
        level = result.leak_level
        if level is None:
            return 1 if result.leaked else 4
        level = int(level)
        return level if level in {1, 2, 3, 4} else 1

    def _success_artifact(self, episode: EpisodeArtifact) -> TurnArtifact | None:
        if episode.termination_reason != "success":
            return None
        for artifact in episode.turns:
            if artifact.invalid_due_to_leak:
                continue
            if artifact.judge_result is not None and artifact.judge_result.correct:
                return artifact
        return None

    def _success_credits(
        self,
        turns: list[TurnArtifact],
        success_artifact: TurnArtifact | None,
        *,
        success_reward_scale: float,
    ) -> dict[int, float]:
        if success_artifact is None:
            return {}
        max_turns = max(1, int(success_artifact.tutor_state.max_turns))
        success_turn = int(success_artifact.turn_idx)
        if max_turns <= 1:
            early_fraction = 0.0
        else:
            early_fraction = max(0.0, (max_turns - success_turn) / (max_turns - 1))
        budget = (
            self.success_reward + self.early_success_bonus * early_fraction
        ) * success_reward_scale
        if not self.assign_success_reward:
            return {success_turn: float(budget)}

        weights: dict[int, float] = {}
        for artifact in turns:
            if artifact.invalid_due_to_leak:
                continue
            turn_idx = int(artifact.turn_idx)
            if turn_idx > success_turn:
                continue
            distance = success_turn - turn_idx
            if distance == 0:
                weight = 1.0
            else:
                weight = self.outcome_prior_turn_weight * (
                    self.outcome_credit_gamma**distance
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
        leak_level=artifact.leak_result.leak_level,
        invalid_due_to_leak=artifact.invalid_due_to_leak,
        tutor_format_error=artifact.tutor_format_error,
    )
