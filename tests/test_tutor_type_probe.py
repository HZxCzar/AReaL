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


class _LoraCaller:
    def __init__(self, *, refuses_unknown_adapter: bool) -> None:
        self.refuses_unknown_adapter = refuses_unknown_adapter
        self.calls: list[dict[str, object]] = []

    async def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.refuses_unknown_adapter:
            raise RuntimeError("unknown adapter")
        return object()


class _LoraProbe:
    type_probe_temperature = 1.0
    type_probe_max_new_tokens = 1
    gconfig = GenerationHyperparameters(max_new_tokens=1, lora_name="real-adapter")
    _type_probe_gconfig = TutorAgentWorkflow._type_probe_gconfig
    _verify_type_probe_lora_honored = (
        TutorAgentWorkflow._verify_type_probe_lora_honored
    )


async def check_lora_liveness_cache() -> tuple[bool, bool, int]:
    probe = _LoraProbe()
    probe._type_probe_lora_honored = None
    probe._type_probe_lora_check_lock = asyncio.Lock()
    caller = _LoraCaller(refuses_unknown_adapter=False)
    first = await TutorAgentWorkflow._ensure_type_probe_lora_honored(
        probe,
        messages=[{"role": "user", "content": "probe"}],
        chat_caller=caller,
        lora_version=7,
    )
    second = await TutorAgentWorkflow._ensure_type_probe_lora_honored(
        probe,
        messages=[{"role": "user", "content": "probe"}],
        chat_caller=caller,
        lora_version=7,
    )
    return first, second, len(caller.calls)


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

    print("\n[6] LoRA liveness fails closed and is cached")
    refusing = _LoraCaller(refuses_unknown_adapter=True)
    verified = asyncio.run(
        TutorAgentWorkflow._verify_type_probe_lora_honored(
            _LoraProbe(),
            messages=[{"role": "user", "content": "probe"}],
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
        and refusing.calls[0]["metadata"] == {"lora_version": 7},
        str(refusing.calls),
    )
    first, second, calls = asyncio.run(check_lora_liveness_cache())
    check(
        "accepting a bogus adapter disables probe reward",
        first is False and second is False,
    )
    check("the liveness result is checked only once", calls == 1, str(calls))

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("\nall joint student-type probe checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
