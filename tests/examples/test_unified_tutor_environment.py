import asyncio

from examples.tutor.environment import (
    AutomaticEvent,
    CompressedContext,
    EpisodeInput,
    Participant,
    TutorAction,
    TutorEnvironment,
    TutorObservation,
    _AutomaticAdvance,
    _TaskTransition,
)


class _FakeStudentPool:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    async def act(self, _participant: Participant, prompt: str) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


class _FakeEnvironment(TutorEnvironment):
    def __init__(self, student_pool: _FakeStudentPool) -> None:
        super().__init__(student_pool, max_student_action_attempts=2)
        self.participant = Participant("villager-1", "student-a", role="villager")
        self.compression_targets: list[str] = []
        self.applied_actions: list[str] = []

    async def reset(self, _episode: EpisodeInput) -> TutorObservation:
        return self._start_episode(TutorObservation(
            context=CompressedContext("public", "memory", "summary", ("vote p2",)),
            active_participants=(self.participant,),
        ))

    async def compress_context(self, target: str) -> CompressedContext:
        self.compression_targets.append(target)
        return CompressedContext("public", "memory", "summary", ("vote p2",))

    def build_student_prompt(
        self,
        _participant: Participant,
        _context: CompressedContext,
        _tutor_action: TutorAction,
        *,
        retry_feedback: str | None = None,
    ) -> str:
        return "student prompt" if retry_feedback is None else retry_feedback

    def parse_student_action(
        self,
        _participant: Participant,
        raw_action: str,
        _context: CompressedContext,
    ) -> str | None:
        return raw_action if raw_action == "vote p2" else None

    def fallback_student_action(
        self, _participant: Participant, _context: CompressedContext
    ) -> str:
        return "vote p2"

    async def apply_tutored_actions(self, _tutor_action, student_actions):
        self.applied_actions.extend(action.action for action in student_actions)
        return _TaskTransition("villager voted", {"action_quality": 0.5}, False)

    async def advance_untutored_participants(self, max_steps: int) -> _AutomaticAdvance:
        assert max_steps == 64
        return _AutomaticAdvance(
            (AutomaticEvent(None, "kill villager-2", "werewolf turn"),),
            {"terminal_credit": 0.25},
            False,
        )

    async def next_tutor_observation(self) -> TutorObservation:
        return TutorObservation(
            context=CompressedContext("next", "memory", "summary"),
            active_participants=(self.participant,),
        )


def test_step_repairs_student_action_and_records_only_tutored_turn() -> None:
    async def run() -> None:
        student_pool = _FakeStudentPool(["invalid", "vote p2"])
        environment = _FakeEnvironment(student_pool)
        await environment.reset(EpisodeInput("episode-1", {}))

        result = await environment.step(TutorAction("Consider the public vote history."))

        assert environment.compression_targets == ["student"]
        assert environment.applied_actions == ["vote p2"]
        assert result.trainable_turn.student_actions[0].retry_count == 1
        assert not result.trainable_turn.student_actions[0].used_fallback
        assert result.trainable_turn.reward == 0.75
        assert len(result.automatic_events) == 1
        assert environment.trajectory == (result.trainable_turn,)
        assert not result.done
        assert result.next_observation is not None
        assert "vote p2" in student_pool.prompts[1]

    asyncio.run(run())


def test_step_uses_deterministic_fallback_after_exhausting_retries() -> None:
    async def run() -> None:
        environment = _FakeEnvironment(_FakeStudentPool(["invalid", "still invalid"]))
        await environment.reset(EpisodeInput("episode-1", {}))

        result = await environment.step(TutorAction("Tutor advice."))

        action = result.trainable_turn.student_actions[0]
        assert action.action == "vote p2"
        assert action.used_fallback
        assert action.retry_count == 1
        assert len(action.raw_responses) == 2

    asyncio.run(run())
