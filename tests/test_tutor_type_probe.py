#!/usr/bin/env python3
"""Final joint student-type probe: scoring, reward, and engine passthrough."""

from __future__ import annotations

import asyncio

from examples.tutor.configs import TutorStudentTypeProbeConfig
from examples.tutor.core.type_probe import (
    ProbeReading,
    StudentTypeProbeReading,
    average_distributions,
    group_probabilities,
    joint_distribution,
    joint_index,
    option_rotations,
    restricted_probabilities,
    rotation_disagreement,
    unpermute,
)
from examples.tutor.core.types import RewardAssignment
from examples.tutor.workflow import TutorAgentWorkflow

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest, get_versioned_lora_name
from areal.engine.sglang_remote import SGLangBackend
from areal.infra.utils.http import HTTPRequestError

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


class _Artifact:
    def __init__(self, *, completed: bool) -> None:
        self.student_state = object() if completed else None


class _Payer:
    type_probe_reward_scale = 1.0


class _DisabledProbe:
    type_probe_enabled = False


class _ExplodingEngine:
    def __getattribute__(self, name: str):
        raise AssertionError(f"disabled probe touched engine attribute {name!r}")


class _LoraEngine:
    def __init__(self, *addresses: str) -> None:
        self.addresses = list(addresses or ("127.0.0.1:1",))


class _LoraCaller:
    def __init__(
        self,
        *,
        outcomes: list[int | str] | None = None,
        engine: object | None = None,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.outcomes = list(outcomes or ["success"])
        self.engine = engine
        self.entered = entered
        self.release = release
        self.calls: list[dict[str, object]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        if isinstance(outcome, int):
            raise HTTPRequestError(
                f"test HTTP {outcome}",
                attempts=1,
                status=outcome,
                last_exception=None,
            )
        if outcome == "timeout":
            raise HTTPRequestError(
                "test timeout",
                attempts=1,
                status=None,
                last_exception=TimeoutError(),
            )
        return object()


class _LoraProbe:
    type_probe_temperature = 1.0
    type_probe_max_new_tokens = 1
    gconfig = GenerationHyperparameters(max_new_tokens=1, lora_name="real-adapter")
    _type_probe_gconfig = TutorAgentWorkflow._type_probe_gconfig
    _verify_type_probe_lora_honored = (
        TutorAgentWorkflow._verify_type_probe_lora_honored
    )
    _shared_type_probe_lora_check_state = (
        TutorAgentWorkflow._shared_type_probe_lora_check_state
    )


def _new_lora_probe() -> _LoraProbe:
    probe = _LoraProbe()
    probe._type_probe_lora_honored = None
    probe._type_probe_lora_check_lock = asyncio.Lock()
    return probe


async def _ensure_lora(
    probe: _LoraProbe,
    caller: _LoraCaller,
    lora_version: int | None = 7,
) -> bool:
    return await TutorAgentWorkflow._ensure_type_probe_lora_honored(
        probe,
        chat_caller=caller,
        lora_version=lora_version,
    )


async def check_lora_liveness_cache(
    outcome: int | str,
) -> tuple[bool, bool, int]:
    engine = _LoraEngine()
    first_caller = _LoraCaller(outcomes=[outcome], engine=engine)
    first = await _ensure_lora(_new_lora_probe(), first_caller)
    second_caller = _LoraCaller(outcomes=[outcome], engine=engine)
    second = await _ensure_lora(_new_lora_probe(), second_caller)
    return first, second, len(first_caller.calls) + len(second_caller.calls)


async def check_lora_engine_scoping() -> tuple[bool, bool, int]:
    first_caller = _LoraCaller(outcomes=[400], engine=_LoraEngine("server-a:1"))
    second_caller = _LoraCaller(outcomes=[400], engine=_LoraEngine("server-a:1"))
    first = await _ensure_lora(_new_lora_probe(), first_caller)
    second = await _ensure_lora(_new_lora_probe(), second_caller)
    return first, second, len(first_caller.calls) + len(second_caller.calls)


async def check_lora_version_none_does_not_poison_cache() -> tuple[bool, bool, int]:
    caller = _LoraCaller(outcomes=[400], engine=_LoraEngine())
    skipped = await _ensure_lora(_new_lora_probe(), caller, lora_version=None)
    verified = await _ensure_lora(_new_lora_probe(), caller, lora_version=7)
    return skipped, verified, len(caller.calls)


async def check_lora_transient_failure_is_not_cached() -> tuple[bool, bool, int]:
    caller = _LoraCaller(outcomes=[500, 400], engine=_LoraEngine())
    transient_raised = False
    try:
        await _ensure_lora(_new_lora_probe(), caller)
    except HTTPRequestError as exc:
        transient_raised = exc.status == 500
    verified = await _ensure_lora(_new_lora_probe(), caller)
    return transient_raised, verified, len(caller.calls)


async def check_lora_waiter_cancellation() -> tuple[bool, bool, int]:
    entered = asyncio.Event()
    release = asyncio.Event()
    engine = _LoraEngine()
    first_caller = _LoraCaller(
        outcomes=[400],
        engine=engine,
        entered=entered,
        release=release,
    )
    first_waiter = asyncio.create_task(
        _ensure_lora(_new_lora_probe(), first_caller)
    )
    await entered.wait()
    first_waiter.cancel()
    cancelled = False
    try:
        await first_waiter
    except asyncio.CancelledError:
        cancelled = True

    second_caller = _LoraCaller(outcomes=[400], engine=engine)
    second_waiter = asyncio.create_task(
        _ensure_lora(_new_lora_probe(), second_caller)
    )
    release.set()
    verified = await second_waiter
    return cancelled, verified, len(first_caller.calls) + len(second_caller.calls)


def main() -> int:
    print("\n[1] cyclic rotations remove answer-letter position")
    behavior_orders = option_rotations(2, cyclic=True)
    check(
        "two behavior options produce two rotations",
        behavior_orders == ((0, 1), (1, 0)),
        str(behavior_orders),
    )
    orders = option_rotations(4, cyclic=True)
    check("four options produce four rotations", len(orders) == 4, str(orders))
    positions = {
        option: sorted(order.index(option) for order in orders) for option in range(4)
    }
    check(
        "every semantic option occupies every slot once",
        all(slots == [0, 1, 2, 3] for slots in positions.values()),
        str(positions),
    )
    information_distribution = [0.1, 0.2, 0.6, 0.1]
    recovered = []
    for order in orders:
        slot_distribution = [information_distribution[option] for option in order]
        recovered.append(unpermute(slot_distribution, order, 4))
    averaged = average_distributions(recovered)
    check(
        "unpermuting and averaging recovers semantic probabilities",
        all(
            close(left, right)
            for left, right in zip(
                averaged, information_distribution, strict=True
            )
        ),
        str(averaged),
    )
    check(
        "identical semantic reads have zero rotation disagreement",
        close(rotation_disagreement(recovered), 0.0),
    )

    print("\n[2] reward is the correct cell of the eight-type distribution")
    behavior_reading = ProbeReading(
        distribution=(0.3, 0.7),
        correct_index=1,
        disagreement=0.0,
        calls=2,
    )
    information_reading = ProbeReading(
        distribution=tuple(information_distribution),
        correct_index=2,
        disagreement=0.0,
        calls=4,
    )
    combined = joint_distribution(
        (behavior_reading.distribution, information_reading.distribution)
    )
    correct_joint_index = joint_index((1, 2), (2, 4))
    reading = StudentTypeProbeReading(
        behavior=behavior_reading,
        information=information_reading,
        joint_distribution=tuple(combined),
        correct_joint_index=correct_joint_index,
    )
    check("outer product contains eight student types", len(combined) == 8)
    check("joint distribution sums to one", close(sum(combined), 1.0))
    check("correct text/code-information cell is row-major", correct_joint_index == 6)
    check(
        "correct joint probability is the product of both axes",
        close(reading.correct_probability, 0.7 * 0.6),
        str(reading.correct_probability),
    )
    check("joint argmax requires the correct pair", reading.is_argmax_correct == 1.0)
    artifacts = [_Artifact(completed=True), _Artifact(completed=False)]
    assignments = [
        RewardAssignment(reward=0.25, reward_components={}, local_reward=0.0),
        RewardAssignment(reward=-0.1, reward_components={}, local_reward=0.0),
    ]
    reward = TutorAgentWorkflow._apply_student_type_probe_reward(
        _Payer(), artifacts, assignments, reading
    )
    check("correct joint probability is the reward", close(reward, 0.42), str(reward))
    check(
        "reward attaches to the final completed teaching round",
        close(assignments[0].reward, 0.67) and close(assignments[1].reward, -0.1),
        str([assignment.reward for assignment in assignments]),
    )
    check(
        "the probe reward is terminal, not turn-local",
        all(close(assignment.local_reward, 0.0) for assignment in assignments),
    )

    print("\n[3] restricted label probabilities retain all options")
    probabilities = restricted_probabilities([0.0, -1.0, float("-inf")])
    check("restricted probabilities sum to one", close(sum(probabilities), 1.0))
    check("an unreachable option stays present at zero", probabilities[2] == 0.0)
    check(
        "bare and leading-space spellings are combined",
        group_probabilities([0.2, 0.3, 0.5], [(0, 1), (2,)]) == [0.5, 0.5],
    )

    print("\n[4] the SGLang request and response carry the restricted read")
    backend = SGLangBackend()
    request = backend.build_generation_request(
        ModelRequest(
            rid="probe-test",
            input_ids=[1, 2],
            gconfig=GenerationHyperparameters(max_new_tokens=1),
            metadata={"token_ids_logprob": [32, 33]},
        ),
        False,
        0,
    )
    check(
        "requested ids reach /generate",
        request.payload.get("token_ids_logprob") == [32, 33],
        str(request.payload),
    )
    plain = backend.build_generation_request(
        ModelRequest(
            rid="plain",
            input_ids=[1],
            gconfig=GenerationHyperparameters(max_new_tokens=1),
            metadata={},
        ),
        False,
        0,
    )
    check(
        "ordinary generation payload is unchanged",
        "token_ids_logprob" not in plain.payload,
    )
    lora_request = backend.build_generation_request(
        ModelRequest(
            rid="lora-probe",
            input_ids=[1],
            gconfig=GenerationHyperparameters(
                max_new_tokens=1, lora_name="probe-adapter"
            ),
            metadata={},
        ),
        True,
        7,
    )
    check(
        "the probe request pins the versioned LoRA path",
        lora_request.payload.get("lora_path")
        == get_versioned_lora_name("probe-adapter", 7),
        str(lora_request.payload),
    )
    parsed = backend.parse_generation_response(
        {
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 32, "A"]],
                "output_token_ids_logprobs": [[[-0.1, 32, "A"], [-2.3, 33, "B"]]],
            }
        }
    )
    check(
        "restricted logprobs parse back with token ids",
        parsed.output_token_ids_logprobs == [[(-0.1, 32), (-2.3, 33)]],
        str(parsed.output_token_ids_logprobs),
    )

    print("\n[5] disabled remains a complete no-op")
    check("config defaults disabled", not TutorStudentTypeProbeConfig().enabled)
    skipped = asyncio.run(
        TutorAgentWorkflow._run_final_student_type_probe(
            _DisabledProbe(),
            turn_artifacts=[],
            selected_student=None,
            engine=_ExplodingEngine(),
            lora_version=7,
            trajectory_id=1,
        )
    )
    check("disabled probe returns without touching the engine", skipped is None)

    print("\n[6] LoRA liveness fails closed and is cached per engine")
    refusing = _LoraCaller(outcomes=[400])
    verified = asyncio.run(
        TutorAgentWorkflow._verify_type_probe_lora_honored(
            _LoraProbe(),
            chat_caller=refusing,
            lora_version=7,
        )
    )
    check("rejecting a bogus adapter verifies LoRA routing", verified)
    check(
        "the liveness request uses a bogus adapter and pinned version",
        len(refusing.calls) == 1
        and refusing.calls[0]["gconfig"].lora_name.startswith(
            "tutor-type-probe-no-such-adapter-"
        )
        and refusing.calls[0]["messages"]
        == [{"role": "user", "content": "Reply with A."}]
        and refusing.calls[0]["metadata"]
        == {"lora_version": 7, "_request_max_attempts": 1},
        str(refusing.calls),
    )
    first, second, calls = asyncio.run(check_lora_liveness_cache("success"))
    check(
        "accepting a bogus adapter disables probe reward",
        first is False and second is False,
    )
    check(
        "the liveness result is checked once across workflow instances",
        calls == 1,
        str(calls),
    )
    first, second, calls = asyncio.run(check_lora_liveness_cache(400))
    check(
        "rejecting a bogus adapter is also cached across workflow instances",
        first is True and second is True and calls == 1,
        str((first, second, calls)),
    )

    first, second, calls = asyncio.run(check_lora_engine_scoping())
    check(
        "separate inference engines verify independently",
        first is True and second is True and calls == 2,
        str((first, second, calls)),
    )

    skipped, verified, calls = asyncio.run(
        check_lora_version_none_does_not_poison_cache()
    )
    check(
        "an unavailable LoRA version neither probes nor poisons the cache",
        skipped is True and verified is True and calls == 1,
        str((skipped, verified, calls)),
    )

    transient_raised, verified, calls = asyncio.run(
        check_lora_transient_failure_is_not_cached()
    )
    check(
        "transient HTTP failures propagate and a later episode retries",
        transient_raised and verified is True and calls == 2,
        str((transient_raised, verified, calls)),
    )

    cancelled, verified, calls = asyncio.run(check_lora_waiter_cancellation())
    check(
        "cancelling one waiter does not cancel the shared verification task",
        cancelled and verified is True and calls == 1,
        str((cancelled, verified, calls)),
    )

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("\nall joint student-type probe checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
