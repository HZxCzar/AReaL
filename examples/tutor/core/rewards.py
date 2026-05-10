from __future__ import annotations

from collections.abc import Awaitable, Callable

from .types import (
    EpisodeArtifact,
    ProgressJudgment,
    RewardAssignment,
    TurnArtifact,
    TurnTrace,
)

ProgressJudgeFn = Callable[
    [str, str, str, str, str],
    Awaitable[ProgressJudgment],
]


class EpisodeRewardComputer:
    def __init__(
        self,
        *,
        success_reward: float,
        leak_penalty: float,
        progress_rewards: dict[str, float],
        progress_judge: ProgressJudgeFn,
    ) -> None:
        self.success_reward = success_reward
        self.leak_penalty = leak_penalty
        self.progress_rewards = progress_rewards
        self.progress_judge = progress_judge

    async def compute(self, episode: EpisodeArtifact) -> list[RewardAssignment]:
        final_success = episode.termination_reason == "success"
        assignments: list[RewardAssignment] = []
        for artifact in episode.turns:
            if artifact.leak_result.leaked:
                progress = ProgressJudgment(
                    raw_output="",
                    label="unknown",
                    confidence="low",
                    feedback=(
                        "Skipped because the tutor message leaked private answer "
                        "information."
                    ),
                )
                reward = self.leak_penalty
                assignments.append(
                    RewardAssignment(
                        progress=progress,
                        reward=reward,
                        reward_components={"leak": reward},
                    )
                )
                continue

            if artifact.judge_result is not None and artifact.judge_result.correct:
                progress = ProgressJudgment(
                    raw_output="",
                    label="improved",
                    confidence="high",
                    feedback="The student reached the correct final answer.",
                )
                reward = self.success_reward
                assignments.append(
                    RewardAssignment(
                        progress=progress,
                        reward=reward,
                        reward_components={"success": reward},
                    )
                )
                continue

            if artifact.student_state is None:
                progress = ProgressJudgment(
                    raw_output="",
                    label="unknown",
                    confidence="low",
                    feedback="Skipped because no student response artifact was recorded.",
                )
            else:
                progress = await self.progress_judge(
                    episode.task,
                    episode.ground_truth,
                    artifact.student_state.previous_student_output,
                    artifact.student_output,
                    artifact.tutor_visible_output,
                )
            reward = self.progress_rewards.get(progress.label, 0.0)
            if progress.label == "improved" and not final_success:
                reward = 0.0
            assignments.append(
                RewardAssignment(
                    progress=progress,
                    reward=reward,
                    reward_components={f"progress_{progress.label}": reward},
                )
            )
        return assignments


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
        progress=assignment.progress,
        reward=assignment.reward,
        reward_components=assignment.reward_components,
        public_history_before=artifact.public_history_before,
        public_history_after=artifact.public_history_after,
    )
