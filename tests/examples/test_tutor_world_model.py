from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.tutor.configs import TutorPaWConfig, TutorWorldModelConfig
from examples.tutor.core.tensors import (
    response_to_tensordict,
    tokenize_teacher_forced_response,
)
from examples.tutor.core.types import (
    LeakCheckResult,
    PublicHistoryState,
    TurnArtifact,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.workflow import TutorAgentWorkflow

from areal.trainer.ppo.actor import (
    _append_world_model_rows,
    _global_joint_counts,
    _joint_loss_weight,
    _merge_policy_world_model_loss,
    _unpack_world_model_rows,
)
from areal.trainer.rl_trainer import (
    _collect_world_model_turn_nll,
    _pop_world_model_sidecars,
    _prepare_paw_world_model_sidecars,
)
from areal.utils.data import concat_padded_tensors, split_and_unpad_tensor
from areal.utils.functional import (
    gather_logprobs_entropy,
    resolve_logprob_temperature,
)


class _PrefixTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        del tokenize, enable_thinking
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def _response(input_tokens: list[int], output_tokens: list[int]):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[3] * len(output_tokens),
        stop_reason="stop",
    )


def _sidecar(
    input_ids: list[int],
    target_mask: list[int],
    seq_lens: list[int],
    weight=0.1,
    paw: dict | None = None,
):
    result = {
        "world_model_packed_input_ids": torch.tensor(input_ids, dtype=torch.long),
        "world_model_packed_target_mask": torch.tensor(target_mask, dtype=torch.bool),
        "world_model_seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "world_model_loss_weight": weight,
    }
    if paw is not None:
        settings = {
            "enabled": True,
            "entropy_filter_enabled": True,
            "entropy_keep_ratio": 0.75,
            "cmae_enabled": True,
            "confidence_threshold": 0.2,
            "reward_adaptive_enabled": False,
            "max_episode_return": 1.0,
            **paw,
        }
        result["world_model_paw_config"] = settings
    return result


def _policy_row(
    *, trajectory_id: int, reward: float, action_logprob: float
) -> dict[str, torch.Tensor]:
    return {
        "trajectory_id": torch.tensor([trajectory_id]),
        "rewards": torch.tensor([reward]),
        "logprobs": torch.tensor([[0.0, action_logprob]]),
        "loss_mask": torch.tensor([[0, 1]]),
    }


def test_world_model_config_disabled_by_default_and_validates_enabled_fields():
    """World Model must be opt-in and reject unusable enabled settings."""

    assert TutorWorldModelConfig().enabled is False
    with pytest.raises(ValueError, match="loss_weight"):
        TutorWorldModelConfig(enabled=True, loss_weight=0.0)
    with pytest.raises(ValueError, match="system_prompt"):
        TutorWorldModelConfig(enabled=True, system_prompt="")
    with pytest.raises(ValueError, match="at least 1"):
        TutorWorldModelConfig(per_turn_nll_log_every_n_steps=0)


def test_paw_config_is_opt_in_and_validates_ranges():
    """PaW must not alter legacy WM unless explicitly enabled."""

    assert TutorWorldModelConfig().paw.enabled is False
    with pytest.raises(ValueError, match="world_model.enabled"):
        TutorWorldModelConfig(enabled=False, paw=TutorPaWConfig(enabled=True))
    with pytest.raises(ValueError, match="entropy_keep_ratio"):
        TutorPaWConfig(entropy_keep_ratio=0.0)
    with pytest.raises(ValueError, match="confidence_threshold"):
        TutorPaWConfig(confidence_threshold=1.0)


def test_teacher_forced_tokenization_masks_each_student_target_token():
    """Prompt tokens stay masked while the full Student suffix is supervised."""

    tokenizer = _PrefixTokenizer()
    messages = [
        {"role": "system", "content": "predict"},
        {"role": "user", "content": "context"},
    ]
    input_ids, target_mask = tokenize_teacher_forced_response(
        tokenizer, messages, "student reply"
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert input_ids[: len(prompt_ids)] == prompt_ids
    assert target_mask[: len(prompt_ids)] == [0] * len(prompt_ids)
    assert all(target_mask[len(prompt_ids) :])
    assert sum(target_mask) == len(input_ids) - len(prompt_ids)


def test_world_model_example_uses_teacher_context_and_real_student_target():
    """The WM prompt stops after Teacher output; only the real Student reply is target."""

    tokenizer = _PrefixTokenizer()
    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    workflow.teacher_system_prompt = "teach safely"
    workflow.teacher_warmup_prompt = "unused"
    workflow.world_model_system_prompt = "predict student"
    workflow.tokenizer = tokenizer
    workflow.enable_thinking = False
    workflow.max_train_sample_tokens = None
    tutor_state = TutorTurnState(
        task="solve x",
        ground_truth="x=2",
        public_history=PublicHistoryState(summary="student tried x=3"),
        previous_tutor_visible_output="check substitution",
        previous_feedback=TutorPrivateFeedback(),
        turn_idx=2,
        max_turns=10,
    )
    artifact = TurnArtifact(
        turn_idx=2,
        tutor_state=tutor_state,
        tutor_prompt="Problem and prior dialogue",
        tutor_response=SimpleNamespace(tokenizer=tokenizer),
        tutor_raw_output="Try isolating x.",
        tutor_visible_output="Try isolating x.",
        leak_result=LeakCheckResult("", False, "", None, {}),
        public_history_before="student tried x=3",
        public_history_after="student now tries x=2",
        student_output="I subtract 1 and get x=2.",
    )

    example = workflow._build_world_model_example(artifact)
    decoded = "".join(chr(token) for token in example.input_ids)
    target_start = example.target_mask.index(1)
    target_text = "".join(chr(token) for token in example.input_ids[target_start:])

    assert example.valid
    assert "Problem and prior dialogue" in decoded[:target_start]
    assert "Try isolating x." in decoded[:target_start]
    assert "I subtract 1 and get x=2." not in decoded[:target_start]
    assert target_text == "I subtract 1 and get x=2.</assistant>"


@pytest.mark.asyncio
async def test_world_model_examples_offload_tokenization_without_disabled_overhead(
    monkeypatch,
):
    """Enabled WM tokenizes in a worker thread; disabled WM returns immediately."""

    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    artifact = object()
    calls = []

    def fake_build(artifacts):
        calls.append(("build", artifacts))
        return ["built"]

    async def fake_to_thread(function, *args):
        calls.append(("thread", function, args))
        return function(*args)

    workflow._build_world_model_examples = fake_build
    monkeypatch.setattr("examples.tutor.workflow.asyncio.to_thread", fake_to_thread)

    workflow.world_model_enabled = True
    enabled_result = await workflow._build_world_model_examples_async([artifact])
    workflow.world_model_enabled = False
    disabled_result = await workflow._build_world_model_examples_async([artifact])

    assert enabled_result == ["built"]
    assert disabled_result == [None]
    assert [call[0] for call in calls] == ["thread", "build"]


def test_world_model_example_classifies_terminated_leak_before_empty_student():
    """A terminated leaked turn has no Student reply but must count as a leak skip."""

    workflow = TutorAgentWorkflow.__new__(TutorAgentWorkflow)
    artifact = SimpleNamespace(
        student_error=None,
        student_output="",
        invalid_due_to_leak=False,
        leak_result=SimpleNamespace(leaked=True),
    )

    example = workflow._build_world_model_example(artifact)

    assert example.skip_reason == "leak"


def test_response_tensor_sidecar_keeps_zero_length_row_alignment():
    """Skipped WM samples keep one zero length so later Teacher rows stay aligned."""

    valid = response_to_tensordict(
        _response([1, 2], [3]),
        reward=1.0,
        world_model_input_tokens=[10, 11, 12],
        world_model_target_mask=[0, 1, 1],
        world_model_loss_weight=0.1,
    )
    skipped = response_to_tensordict(
        _response([4], [5]),
        reward=0.0,
        world_model_loss_weight=0.1,
    )
    episode = concat_padded_tensors([valid, skipped])

    torch.testing.assert_close(
        episode["world_model_seq_lens"],
        torch.tensor([3, 0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        episode["world_model_packed_input_ids"],
        torch.tensor([10, 11, 12]),
        rtol=0,
        atol=0,
    )


def test_response_tensor_has_no_world_model_fields_when_disabled():
    """The default rollout payload stays PPO-only when WM is disabled."""

    result = response_to_tensordict(_response([1, 2], [3]), reward=1.0)

    assert not any(key.startswith("world_model_") for key in result)


def test_response_tensor_carries_paw_switches_only_inside_world_model_sidecar():
    """Rollout transport preserves PaW settings without changing PPO tensors."""

    result = response_to_tensordict(
        _response([1], [2]),
        reward=0.0,
        trajectory_id=7,
        world_model_input_tokens=[10, 11],
        world_model_target_mask=[0, 1],
        world_model_loss_weight=1.0,
        world_model_paw_config={
            "enabled": True,
            "reward_adaptive_enabled": False,
        },
    )

    assert result["world_model_paw_config"] == {
        "enabled": True,
        "reward_adaptive_enabled": False,
    }
    torch.testing.assert_close(
        result["trajectory_id"], torch.tensor([7]), rtol=0, atol=0
    )


def test_packed_sidecar_survives_trajectory_unpadding():
    """WM sequences longer than PPO rows must not be trimmed by async transport."""

    trajectory = response_to_tensordict(
        _response([1], [2]),
        reward=1.0,
        world_model_input_tokens=list(range(9)),
        world_model_target_mask=[0] * 5 + [1] * 4,
        world_model_loss_weight=0.1,
    )
    restored = split_and_unpad_tensor(trajectory, n_trajs=1, traj_group_sizes=[1])[0]

    assert restored["attention_mask"].shape[-1] == 2
    assert restored["world_model_packed_input_ids"].numel() == 9


def test_world_model_sidecars_are_removed_before_ppo_preprocessing():
    """Ref, critic, prox-logp, and advantage paths receive only PPO fields."""

    trajectories = [
        {"input_ids": torch.tensor([[1]]), **_sidecar([4, 5], [0, 1], [2])},
        {"input_ids": torch.tensor([[2]]), **_sidecar([], [], [0])},
    ]
    sidecars = _pop_world_model_sidecars(trajectories)

    assert sidecars is not None
    assert len(sidecars) == 2
    assert all(
        not any(key.startswith("world_model_") for key in trajectory)
        for trajectory in trajectories
    )
    assert sidecars[0]["world_model_loss_weight"] == 0.1


def test_world_model_sidecar_requires_complete_batch_metadata():
    """Partially enabled batches fail before distributed collectives can diverge."""

    with pytest.raises(ValueError, match="every trajectory"):
        _pop_world_model_sidecars(
            [
                {"input_ids": torch.tensor([[1]]), **_sidecar([], [], [0])},
                {"input_ids": torch.tensor([[2]])},
            ]
        )


def test_world_model_turn_nll_logs_standard_ce_and_episode_outcome():
    """Per-turn diagnostics align next-token NLL with rollout and PaW metadata."""

    class FakeActor:
        def compute_logp(self, batches):
            # Mimic a two-way DP controller, which requires a divisible group count.
            assert len(batches) == 2
            batch = batches[0]
            torch.testing.assert_close(
                batch["token_logprob_temperature"],
                torch.ones((2, 4)),
                rtol=0,
                atol=0,
            )
            assert batches[1]["attention_mask"].all()
            torch.testing.assert_close(
                batches[1]["token_logprob_temperature"],
                torch.ones((1, 4)),
                rtol=0,
                atol=0,
            )
            return [
                torch.tensor([[-2.0] * 4, [-4.0] * 4]),
                torch.zeros((1, 4)),
            ]

    rollout_batch = [
        {
            "trajectory_id": torch.tensor([99, 99]),
            "turn_idx": torch.tensor([1, 2]),
            "rewards": torch.tensor([-0.2, -0.8]),
        },
        {
            "trajectory_id": torch.tensor([100]),
            "turn_idx": torch.tensor([1]),
            "rewards": torch.tensor([0.0]),
        },
    ]
    sidecars = [
        {
            **_sidecar(
                [10, 11, 12, 13, 20, 21, 22],
                [0, 0, 1, 1, 0, 1, 1],
                [4, 3],
            ),
            "world_model_selected": torch.tensor([True, False]),
            "world_model_response_weight": torch.tensor([0.5, 0.0]),
        },
        _sidecar([], [], [0]),
    ]

    records = _collect_world_model_turn_nll(FakeActor(), rollout_batch, sidecars)

    assert records == [
        {
            "trajectory_id": 99,
            "turn_idx": 1,
            "turn_reward": pytest.approx(-0.2),
            "episode_return": pytest.approx(-1.0),
            "paw_selected": True,
            "world_model_response_weight": pytest.approx(0.5),
            "world_model_nll": pytest.approx(2.0),
            "target_tokens": 2,
        },
        {
            "trajectory_id": 99,
            "turn_idx": 2,
            "turn_reward": pytest.approx(-0.8),
            "episode_return": pytest.approx(-1.0),
            "paw_selected": False,
            "world_model_response_weight": pytest.approx(0.0),
            "world_model_nll": pytest.approx(4.0),
            "target_tokens": 2,
        },
    ]


def test_world_model_turn_nll_preserves_empty_leading_dp_group():
    """An empty first group keeps its DP slot and produces no diagnostic row."""

    class FakeActor:
        def compute_logp(self, batches):
            assert len(batches) == 2
            assert batches[0]["attention_mask"].all()
            return [torch.zeros((1, 3)), torch.full((1, 3), -3.0)]

    rollout_batch = [
        {
            "trajectory_id": torch.tensor([100]),
            "turn_idx": torch.tensor([1]),
            "rewards": torch.tensor([0.0]),
        },
        {
            "trajectory_id": torch.tensor([101]),
            "turn_idx": torch.tensor([1]),
            "rewards": torch.tensor([1.0]),
        },
    ]
    sidecars = [
        _sidecar([], [], [0]),
        _sidecar([10, 11, 12], [0, 1, 1], [3]),
    ]

    records = _collect_world_model_turn_nll(FakeActor(), rollout_batch, sidecars)

    assert len(records) == 1
    assert records[0]["trajectory_id"] == 101
    assert records[0]["world_model_nll"] == pytest.approx(3.0)


def test_world_model_turn_nll_skips_forward_when_all_groups_are_empty():
    """An update without any real WM targets does not invoke the actor."""

    class FakeActor:
        def compute_logp(self, batches):
            raise AssertionError(f"unexpected forward: {batches}")

    rollout_batch = [
        {
            "trajectory_id": torch.tensor([100 + index]),
            "turn_idx": torch.tensor([1]),
            "rewards": torch.tensor([0.0]),
        }
        for index in range(2)
    ]
    sidecars = [_sidecar([], [], [0]), _sidecar([], [], [0])]

    records = _collect_world_model_turn_nll(FakeActor(), rollout_batch, sidecars)

    assert records == []


def test_world_model_rows_use_disjoint_shifted_loss_masks():
    """WM target tokens receive CE while the appended rows receive no PPO loss."""

    policy_batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.zeros((1, 4)),
        "advantages": torch.ones((1, 4)),
        "versions": torch.tensor([[-1, 3, 3, -1]]),
    }
    joint = _append_world_model_rows(
        policy_batch,
        [
            (
                torch.tensor([10, 11, 12, 13]),
                torch.tensor([0, 0, 1, 1], dtype=torch.bool),
            )
        ],
        policy_temperature=0.7,
    )

    torch.testing.assert_close(joint["loss_mask"][1], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(
        joint["world_model_loss_mask"][1],
        torch.tensor([False, True, True, False]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        joint["world_model_token_weight"][1],
        torch.tensor([0.0, 0.5, 0.5, 0.0]),
        rtol=0,
        atol=0,
    )
    assert joint["world_model_response_start_mask"][1].nonzero().tolist() == [[1]]
    torch.testing.assert_close(
        joint["token_logprob_temperature"],
        torch.tensor([[0.7, 0.7, 0.7, 0.7], [1.0, 1.0, 1.0, 1.0]]),
        rtol=0,
        atol=0,
    )


def test_mixed_temperature_uses_actor_policy_and_standard_world_model_ce():
    """One logits batch may use actor temperature for PPO and T=1 for WM tokens."""

    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.5, 1.5, -0.5], [-1.0, 0.0, 3.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 2])
    temperatures = resolve_logprob_temperature(
        {"token_logprob_temperature": torch.tensor([[0.7, 0.0, 1.0]])},
        default_temperature=0.7,
    )

    logprobs, entropy = gather_logprobs_entropy(
        logits, labels, temperature=temperatures, chunk_size=2
    )
    expected_distribution = torch.log_softmax(
        logits / torch.tensor([0.7, 0.7, 1.0]).unsqueeze(-1), dim=-1
    )
    expected_logprobs = expected_distribution.gather(-1, labels.unsqueeze(-1)).squeeze(
        -1
    )
    expected_entropy = -(expected_distribution.exp() * expected_distribution).sum(-1)

    torch.testing.assert_close(logprobs, expected_logprobs, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(entropy, expected_entropy, rtol=1e-6, atol=1e-6)


def test_unpack_world_model_rows_preserves_skipped_turns():
    """Packed restoration returns an empty row for each skipped Teacher turn."""

    rows = _unpack_world_model_rows(
        _sidecar([1, 2, 3], [0, 1, 1], [0, 2, 0, 1]), expected_rows=4
    )

    assert [input_ids.numel() for input_ids, _, _, _ in rows] == [0, 2, 0, 1]
    assert rows[1][0].tolist() == [1, 2]
    assert rows[3][0].tolist() == [3]


def test_paw_entropy_filter_selects_update_global_top_fraction(monkeypatch):
    """PaW selects high action entropy across turns inside multi-turn episodes."""

    monkeypatch.setattr(
        "areal.trainer.rl_trainer.stats_tracker.scalar", lambda **_: None
    )
    logprobs = [
        float(torch.log(torch.tensor(probability)))
        for probability in (0.5, 0.9, 0.2, 0.99)
    ]
    turns = [
        _policy_row(trajectory_id=index, reward=0.0, action_logprob=logprob)
        for index, logprob in enumerate(logprobs)
    ]
    turns[1]["trajectory_id"] = turns[0]["trajectory_id"].clone()
    turns[3]["trajectory_id"] = turns[2]["trajectory_id"].clone()
    rollout = [
        concat_padded_tensors(turns[:2]),
        concat_padded_tensors(turns[2:]),
    ]
    sidecar_turns = [
        _sidecar(
            [10, 11],
            [0, 1],
            [2],
            paw={"entropy_keep_ratio": 0.5},
        )
        for _ in turns
    ]
    sidecars = [
        concat_padded_tensors(sidecar_turns[:2]),
        concat_padded_tensors(sidecar_turns[2:]),
    ]

    _prepare_paw_world_model_sidecars(rollout, sidecars)

    assert torch.cat([row["world_model_selected"] for row in sidecars]).tolist() == [
        True,
        False,
        True,
        False,
    ]
    torch.testing.assert_close(
        torch.cat([row["world_model_response_weight"] for row in sidecars]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        rtol=0,
        atol=0,
    )


def test_paw_response_weights_balance_rollout_groups(monkeypatch):
    """Each rollout group contributes one WM mean regardless of its turn count."""

    monkeypatch.setattr(
        "areal.trainer.rl_trainer.stats_tracker.scalar", lambda **_: None
    )
    rollout = [
        concat_padded_tensors(
            [
                _policy_row(trajectory_id=1, reward=0.0, action_logprob=-0.5),
                _policy_row(trajectory_id=1, reward=1.0, action_logprob=-0.5),
            ]
        ),
        _policy_row(trajectory_id=2, reward=-1.0, action_logprob=-0.5),
    ]
    fixed_turn = _sidecar(
        [10, 11],
        [0, 1],
        [2],
        paw={
            "entropy_filter_enabled": False,
            "reward_adaptive_enabled": False,
        },
    )
    adaptive_turn = _sidecar(
        [10, 11],
        [0, 1],
        [2],
        paw={
            "entropy_filter_enabled": False,
            "reward_adaptive_enabled": True,
        },
    )
    fixed_sidecars = [
        concat_padded_tensors([fixed_turn, fixed_turn]),
        fixed_turn,
    ]
    adaptive_sidecars = [
        concat_padded_tensors([adaptive_turn, adaptive_turn]),
        adaptive_turn,
    ]

    _prepare_paw_world_model_sidecars(rollout, fixed_sidecars)
    _prepare_paw_world_model_sidecars(rollout, adaptive_sidecars)

    torch.testing.assert_close(
        torch.cat([row["world_model_response_weight"] for row in fixed_sidecars]),
        torch.tensor([0.75, 0.75, 1.5]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.cat([row["world_model_response_weight"] for row in adaptive_sidecars]),
        torch.tensor([0.0, 0.0, 3.0]),
        rtol=0,
        atol=0,
    )


def test_paw_reward_adaptive_uses_mean_episode_return_within_rollout_group(
    monkeypatch,
):
    """Grouped rollouts average episode returns before computing their WM weight."""

    monkeypatch.setattr(
        "areal.trainer.rl_trainer.stats_tracker.scalar", lambda **_: None
    )
    rollout = [
        concat_padded_tensors(
            [
                _policy_row(trajectory_id=1, reward=0.0, action_logprob=-0.5),
                _policy_row(trajectory_id=1, reward=1.0, action_logprob=-0.5),
                _policy_row(trajectory_id=2, reward=-1.0, action_logprob=-0.5),
            ]
        ),
        _policy_row(trajectory_id=3, reward=1.0, action_logprob=-0.5),
    ]
    turn = _sidecar(
        [10, 11],
        [0, 1],
        [2],
        paw={
            "entropy_filter_enabled": False,
            "reward_adaptive_enabled": True,
        },
    )
    sidecars = [concat_padded_tensors([turn, turn, turn]), turn]

    _prepare_paw_world_model_sidecars(rollout, sidecars)

    torch.testing.assert_close(
        torch.cat([row["world_model_response_weight"] for row in sidecars]),
        torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_paw_group_without_selected_transition_contributes_zero_wm_term(monkeypatch):
    """An empty selected set omits that group instead of inflating other groups."""

    monkeypatch.setattr(
        "areal.trainer.rl_trainer.stats_tracker.scalar", lambda **_: None
    )
    rollout = [
        concat_padded_tensors(
            [
                _policy_row(trajectory_id=1, reward=0.0, action_logprob=-0.5),
                _policy_row(trajectory_id=1, reward=0.0, action_logprob=-0.5),
            ]
        ),
        _policy_row(trajectory_id=2, reward=0.0, action_logprob=-0.5),
    ]
    valid_turn = _sidecar(
        [10, 11],
        [0, 1],
        [2],
        paw={"entropy_filter_enabled": False},
    )
    skipped_turn = _sidecar(
        [],
        [],
        [0],
        paw={"entropy_filter_enabled": False},
    )
    sidecars = [
        concat_padded_tensors([valid_turn, valid_turn]),
        skipped_turn,
    ]

    _prepare_paw_world_model_sidecars(rollout, sidecars)

    torch.testing.assert_close(
        torch.cat([row["world_model_response_weight"] for row in sidecars]),
        torch.tensor([0.5, 0.5, 0.0]),
        rtol=0,
        atol=0,
    )


def test_paw_selected_row_applies_group_response_weight():
    """PaW response scaling changes WM only and excluded rows add no forward row."""

    policy_batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1]], dtype=torch.float32),
        "logprobs": torch.zeros((1, 2)),
        "advantages": torch.ones((1, 2)),
        "versions": torch.tensor([[-1, 3]]),
    }

    joint = _append_world_model_rows(
        policy_batch,
        [
            (
                torch.tensor([10, 11, 12]),
                torch.tensor([0, 1, 1], dtype=torch.bool),
                True,
                2.0,
            ),
            (
                torch.tensor([20, 21]),
                torch.tensor([0, 1], dtype=torch.bool),
                False,
                1.0,
            ),
        ],
        policy_temperature=0.7,
    )

    assert joint["input_ids"].shape[0] == 2
    torch.testing.assert_close(
        joint["world_model_token_weight"][1],
        torch.tensor([1.0, 1.0, 0.0]),
        rtol=0,
        atol=0,
    )


def test_paw_group_weights_produce_mean_of_group_means(monkeypatch):
    """Final WM loss gives equal mass to groups with unequal selected turn counts."""

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker.denominator", lambda **_: None
    )
    monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker.stat", lambda **_: None)
    # Group A has two selected transitions with loss 0.2; group B has one with
    # loss 0.8. Their rollout-group-balanced weights are 0.75, 0.75, and 1.5.
    logprobs = torch.tensor([0.0, -0.2, -0.2, -0.8], requires_grad=True)
    input_data = {
        "loss_mask": torch.tensor([True, False, False, False]),
        "world_model_loss_mask": torch.tensor([False, True, True, True]),
        "world_model_token_weight": torch.tensor([0.0, 0.75, 0.75, 1.5]),
        "world_model_response_start_mask": torch.tensor([False, True, True, True]),
        "cu_seqlens": torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
    }

    loss = _merge_policy_world_model_loss(
        torch.tensor(0.0),
        logprobs,
        input_data,
        global_policy_tokens=torch.tensor(1.0),
        global_world_model_responses=torch.tensor(3.0),
        world_model_loss_weight=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.5), rtol=1e-6, atol=1e-6)


def test_joint_loss_equals_ppo_plus_response_balanced_world_model(monkeypatch):
    """Different response lengths remain equally weighted and every token gets gradient."""

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker.denominator", lambda **_: None
    )
    monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker.stat", lambda **_: None)
    logprobs = torch.tensor([0.0, 0.0, -2.0, -4.0, -6.0], requires_grad=True)
    input_data = {
        "loss_mask": torch.tensor([True, True, False, False, False]),
        "world_model_loss_mask": torch.tensor([False, False, True, True, True]),
        "world_model_token_weight": torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0]),
        "world_model_response_start_mask": torch.tensor(
            [False, False, True, False, True]
        ),
        "cu_seqlens": torch.tensor([0, 2, 4, 5], dtype=torch.int32),
    }
    loss = _merge_policy_world_model_loss(
        torch.tensor(2.0),
        logprobs,
        input_data,
        global_policy_tokens=torch.tensor(2.0),
        global_world_model_responses=torch.tensor(2.0),
        world_model_loss_weight=0.1,
    )

    torch.testing.assert_close(loss, torch.tensor(2.45), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert torch.count_nonzero(logprobs.grad[2:]) == 3
    torch.testing.assert_close(
        logprobs.grad[2:],
        torch.tensor([-0.025, -0.025, -0.05]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_paw_cmae_clips_confident_tokens_and_keeps_full_response_denominator(
    monkeypatch,
):
    """CMAE uses 1-p, clips only p above rho, and still divides by all targets."""

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker.denominator", lambda **_: None
    )
    monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker.stat", lambda **_: None)
    logprobs = torch.log(torch.tensor([1.0, 0.2, 0.3], requires_grad=True))
    logprobs.retain_grad()
    input_data = {
        "loss_mask": torch.tensor([True, False, False]),
        "world_model_loss_mask": torch.tensor([False, True, True]),
        "world_model_token_weight": torch.tensor([0.0, 0.5, 0.5]),
        "world_model_response_start_mask": torch.tensor([False, True, False]),
        "cu_seqlens": torch.tensor([0, 1, 3], dtype=torch.int32),
    }

    loss = _merge_policy_world_model_loss(
        torch.tensor(0.0),
        logprobs,
        input_data,
        global_policy_tokens=torch.tensor(1.0),
        global_world_model_responses=torch.tensor(1.0),
        world_model_loss_weight=1.0,
        paw_enabled=True,
        cmae_enabled=True,
        confidence_threshold=0.2,
    )

    torch.testing.assert_close(loss, torch.tensor(0.4), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert logprobs.grad[1] != 0
    assert logprobs.grad[2] == 0


def test_joint_loss_weight_counts_responses_not_world_model_tokens():
    """A long Student reply contributes one response unit to engine normalization."""

    input_data = {
        "loss_mask": torch.tensor([True, True, False, False, False]),
        "world_model_response_start_mask": torch.tensor(
            [False, False, True, False, False]
        ),
    }

    torch.testing.assert_close(
        _joint_loss_weight(input_data), torch.tensor(3), rtol=0, atol=0
    )


def test_joint_counts_move_to_collective_device_before_all_reduce(monkeypatch):
    """NCCL counts must leave rollout CPU memory before the collective call."""

    observed = {}

    def fake_all_reduce(tensor, group):
        observed["device"] = tensor.device
        observed["group"] = group

    monkeypatch.setattr("areal.trainer.ppo.actor.dist.is_initialized", lambda: True)
    monkeypatch.setattr("areal.trainer.ppo.actor.dist.all_reduce", fake_all_reduce)
    group = object()
    counts = _global_joint_counts(
        {
            "loss_mask": torch.tensor([True, True]),
            "world_model_response_start_mask": torch.tensor([False, True]),
        },
        device=torch.device("meta"),
        group=group,
    )

    assert counts.shape == (2,)
    assert counts.dtype == torch.float32
    assert observed == {"device": torch.device("meta"), "group": group}


def test_joint_loss_stays_exact_after_engine_microbatch_weighting(monkeypatch):
    """Splitting PPO and WM rows must not change PPO + lambda * response CE."""

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker.denominator", lambda **_: None
    )
    monkeypatch.setattr("areal.trainer.ppo.actor.stats_tracker.stat", lambda **_: None)
    policy_mb = {
        "loss_mask": torch.tensor([True, True]),
        "world_model_loss_mask": torch.tensor([False, False]),
        "world_model_token_weight": torch.tensor([0.0, 0.0]),
        "world_model_response_start_mask": torch.tensor([False, False]),
        "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
    }
    world_model_mb = {
        "loss_mask": torch.tensor([False, False, False]),
        "world_model_loss_mask": torch.tensor([True, True, True]),
        "world_model_token_weight": torch.tensor([0.5, 0.5, 1.0]),
        "world_model_response_start_mask": torch.tensor([True, False, True]),
        "cu_seqlens": torch.tensor([0, 2, 3], dtype=torch.int32),
    }
    global_policy_tokens = torch.tensor(2.0)
    global_responses = torch.tensor(2.0)
    policy_loss = _merge_policy_world_model_loss(
        torch.tensor(2.0),
        torch.zeros(2),
        policy_mb,
        global_policy_tokens=global_policy_tokens,
        global_world_model_responses=global_responses,
        world_model_loss_weight=0.1,
    )
    world_model_loss = _merge_policy_world_model_loss(
        torch.tensor(0.0),
        torch.tensor([-2.0, -4.0, -6.0]),
        world_model_mb,
        global_policy_tokens=global_policy_tokens,
        global_world_model_responses=global_responses,
        world_model_loss_weight=0.1,
    )
    global_weight = global_policy_tokens + global_responses
    engine_scaled_loss = (
        _joint_loss_weight(policy_mb) / global_weight * policy_loss
        + _joint_loss_weight(world_model_mb) / global_weight * world_model_loss
    )

    torch.testing.assert_close(
        engine_scaled_loss, torch.tensor(2.45), rtol=1e-6, atol=1e-6
    )
