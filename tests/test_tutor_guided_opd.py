#!/usr/bin/env python3
"""Checks for the two failure modes that produce no error, only bad training.

1. The guidance instruction must not survive into the prompt a turn is trained
   on. If it does, the policy learns to obey an instruction it never sees at
   eval, and nothing anywhere raises.
2. The OPD teacher's log-probabilities are computed under a longer prompt, so
   they sit at a different offset. If the realignment is off by even one
   position, every distillation target is attached to the wrong token and the
   loss still runs.

Run from the repo root:  python3 tests/test_tutor_guided_opd.py
"""
from __future__ import annotations

import inspect
import sys

import torch

from examples.tutor.core.tensors import response_to_tensordict
from examples.tutor.core.types import (
    PublicHistoryState,
    TeacherGuidance,
    TurnArtifact,
    TutorPrivateFeedback,
    TutorTurnState,
)
from examples.tutor.prompts import (
    TEACHER_MOVE_INSTRUCTIONS,
    TEACHER_REPAIR_INSTRUCTION,
)
from examples.tutor.workflow import TutorAgentWorkflow

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


class FakeResponse:
    """Minimal stand-in for ModelResponse."""

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = list(input_tokens)
        self.output_tokens = list(output_tokens)
        self.output_logprobs = [-0.5] * len(output_tokens)
        self.output_versions = [0] * len(output_tokens)
        self.stop_reason = "stop"
        self.input_len = len(self.input_tokens)
        self.tokenizer = None


def make_state(turn_idx: int, guidance: TeacherGuidance | None) -> TutorTurnState:
    history = PublicHistoryState()
    history.turns = [
        {"role": "teacher", "content": "Try isolating x first."},
        {"role": "student", "content": "x = 7", "env": "Round 1 of 10. Incorrect."},
    ]
    return TutorTurnState(
        task="Solve 2x + 3 = 17.",
        ground_truth="7",
        public_history=history,
        previous_tutor_visible_output="Try isolating x first.",
        previous_feedback=TutorPrivateFeedback(
            kind="student_judged",
            student_output="x = 7",
            judge_correct=False,
            judge_feedback="Incorrect.",
        ),
        turn_idx=turn_idx,
        max_turns=10,
        guidance=guidance,
    )


def make_workflow_stub(**attrs):
    """A bare object carrying the workflow methods under test.

    TutorAgentWorkflow.__init__ wants a full training config; the prompt-shaping
    and OPD-selection methods only touch a handful of attributes, so binding them
    to a stub keeps this runnable without one.
    """

    class Stub:
        pass

    stub = Stub()
    defaults = {
        "teacher_system_prompt": "You are a tutor.",
        "teacher_show_ground_truth": False,
        "teacher_anti_leak_instruction_enabled": False,
        "teacher_adaptive_instruction_enabled": False,
        "teacher_pre_enabled": False,
        "enable_thinking": False,
        "max_train_sample_tokens": None,
        "guided_slots_enabled": True,
        "guided_slots_count": 3,
        "guided_slots_moves": ("DECOMPOSE", "REFRAME", "PROBE"),
        "guided_slots_turns": frozenset({1}),
        "guided_slots_rotate_by_task": False,
        "opd_enabled": True,
        "opd_instruction": TEACHER_REPAIR_INSTRUCTION,
        "opd_min_prior_failed_turns": 2,
        "opd_max_turns_per_episode": 0,
        "opd_skip_guided_rows": True,
        "opd_skip_leaked_rows": True,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(stub, key, value)
    method_names = (
        "_build_tutor_messages",
        "_clean_tutor_messages",
        "_teacher_system_for_state",
        "_render_conversation",
        "_task_context",
        "_append_teacher_pre_solve_context",
        "_append_guidance_tail",
        "_resolve_teacher_system_prompt",
        "_teacher_system_prompt_for_selection",
        "_select_guidance",
        "_is_guided_slot",
        "_opd_skip_reason",
        "_append_prompt_pool_suffix",
    )
    for name in method_names:
        method = getattr(TutorAgentWorkflow, name, None)
        if method is None:
            raise AssertionError(f"TutorAgentWorkflow has no method {name}")
        # Static methods are already plain functions; only real instance methods
        # need `self` bound.
        params = list(inspect.signature(method).parameters)
        setattr(
            stub,
            name,
            method.__get__(stub, Stub) if params and params[0] == "self" else method,
        )
    return stub


def test_guidance_stripped() -> None:
    print("\n[1] guidance reaches generation but never training")
    stub = make_workflow_stub()
    move = "DECOMPOSE"
    guidance = TeacherGuidance(
        kind="move",
        name=move,
        instruction=TEACHER_MOVE_INSTRUCTIONS[move],
        slot=0,
    )
    state = make_state(2, guidance)
    rollout = stub._build_tutor_messages(state)
    marker = TEACHER_MOVE_INSTRUCTIONS[move][:40]

    check(
        "instruction present in the rollout prompt",
        any(marker in m["content"] for m in rollout),
    )
    check(
        "instruction is in the LAST message, not the system prompt",
        marker in rollout[-1]["content"] and marker not in rollout[0]["content"],
        "placement is load-bearing; a system-prompt instruction is followed far less",
    )
    check("last message is a user turn", rollout[-1]["role"] == "user")

    artifact = TurnArtifact(
        turn_idx=2,
        tutor_state=state,
        tutor_messages=list(rollout),
        tutor_response=FakeResponse(range(10), range(100, 105)),
        tutor_raw_output="",
        tutor_visible_output="",
        leak_result=type("L", (), {"leaked": False, "leak_level": None})(),
        public_history_before=[],
        public_history_after=[],
    )
    clean = stub._clean_tutor_messages(artifact)
    check(
        "instruction absent from the training prompt",
        not any(marker in m["content"] for m in clean),
        "this is the failure that trains instruction-following silently",
    )

    unguided = make_state(2, None)
    rebuilt = stub._build_tutor_messages(unguided, clean=True, include_guidance=False)
    legacy = [
        {"role": "system", "content": stub._teacher_system_for_state(unguided, clean=True)},
        *stub._build_tutor_messages(unguided)[1:],
    ]
    check(
        "with no guidance the rebuild is byte-identical to the old path",
        rebuilt == legacy,
        "otherwise this change alters every existing run",
    )


def test_slot_assignment() -> None:
    print("\n[2] slot assignment")
    stub = make_workflow_stub()
    picked = [
        stub._select_guidance(group_index=i, turn_idx=1, task="t") for i in range(8)
    ]
    guided = [g for g in picked if g is not None]
    check("exactly `slots` rollouts are guided", len(guided) == 3, f"got {len(guided)}")
    check(
        "each guided slot gets a distinct move",
        len({g.name for g in guided}) == 3,
        f"got {[g.name for g in guided]}",
    )
    check(
        "free slots get nothing",
        all(g is None for g in picked[3:]),
    )
    check(
        "no guidance on turns outside `turns`",
        stub._select_guidance(group_index=0, turn_idx=2, task="t") is None,
    )
    check(
        "no guidance when the group index is absent",
        stub._select_guidance(group_index=None, turn_idx=1, task="t") is None,
    )

    rotating = make_workflow_stub(guided_slots_rotate_by_task=True)
    covered = set()
    for task in (f"task-{i}" for i in range(40)):
        for i in range(3):
            g = rotating._select_guidance(group_index=i, turn_idx=1, task=task)
            if g:
                covered.add(g.name)
    check(
        "rotation still only uses configured moves",
        covered <= set(rotating.guided_slots_moves),
        f"got {covered}",
    )


def test_opd_eligibility() -> None:
    print("\n[3] OPD turn eligibility")
    stub = make_workflow_stub()

    def artifact_for(turn_idx, guidance=None, leaked=False, n_out=5):
        return TurnArtifact(
            turn_idx=turn_idx,
            tutor_state=make_state(turn_idx, guidance),
            tutor_messages=[],
            tutor_response=FakeResponse(range(10), range(100, 100 + n_out)),
            tutor_raw_output="",
            tutor_visible_output="",
            leak_result=type("L", (), {"leaked": leaked, "leak_level": None})(),
            public_history_before=[],
            public_history_after=[],
        )

    check("turn 1 too early", stub._opd_skip_reason(artifact_for(1)) == "too_early")
    check("turn 2 too early", stub._opd_skip_reason(artifact_for(2)) == "too_early")
    check("turn 3 eligible", stub._opd_skip_reason(artifact_for(3)) == "")
    check(
        "guided turn skipped",
        stub._opd_skip_reason(
            artifact_for(3, guidance=TeacherGuidance("move", "PROBE", "x", 0))
        )
        == "guided",
    )
    check("leaked turn skipped", stub._opd_skip_reason(artifact_for(3, leaked=True)) == "leak")
    check("empty output skipped", stub._opd_skip_reason(artifact_for(3, n_out=0)) == "empty")


def test_opd_realignment() -> None:
    """The teacher prompt is longer, so its log-probs sit at a different offset.

    An off-by-one here attaches every distillation target to the wrong token and
    nothing raises.
    """
    print("\n[4] OPD log-prob realignment across differing prompt lengths")
    from areal.trainer.ppo.actor import _realign_opd_teacher_logp

    output_tokens = [500, 501, 502, 503]
    clean_prompt = list(range(1, 8))  # 7 tokens
    opd_prompt = list(range(1, 20))  # 19 tokens: prompt + instruction

    selected = response_to_tensordict(
        FakeResponse(clean_prompt, output_tokens),
        reward=1.0,
        input_tokens_override=clean_prompt,
        opd_input_tokens=opd_prompt,
        opd_loss_weight=1.0,
        opd_reward_clip=0.0,
    )
    skipped = response_to_tensordict(
        FakeResponse(clean_prompt, output_tokens),
        reward=0.0,
        input_tokens_override=clean_prompt,
        opd_input_tokens=None,
        opd_loss_weight=1.0,
        opd_reward_clip=0.0,
    )
    check(
        "selected and skipped rows carry the same keys",
        set(selected) == set(skipped),
        "concat_padded_tensors requires identical key sets",
    )
    check("skipped row has zero weight", float(skipped["opd_token_weight"].max()) == 0.0)
    check("selected row has nonzero weight", float(selected["opd_token_weight"].max()) > 0.0)

    from areal.utils.data import concat_padded_tensors

    traj = concat_padded_tensors([selected, skipped])
    check(
        "input_ids not lengthened by the longer OPD sequence",
        traj["input_ids"].shape[1] == len(clean_prompt) + len(output_tokens),
        f"got {traj['input_ids'].shape[1]}",
    )

    # Teacher log-probs in the OPD layout: index i predicts token i+1, so the
    # output tokens are predicted from positions [P-1, P+L-2]. Plant a signature
    # there and require it to land on the matching training positions.
    signature = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    opd_len = traj["opd_loss_mask"].shape[1]
    fake = torch.zeros((2, opd_len), dtype=torch.float32)
    start = len(opd_prompt) - 1
    fake[0, start : start + len(output_tokens)] = signature
    fake[1, :] = 99.0  # the skipped row selects nothing and must stay at zero

    rolled_train_mask = torch.roll(traj["loss_mask"], shifts=-1, dims=-1)
    # The skipped row is inert via its zero weight, so it must be excluded from
    # the training selection exactly as the caller does.
    rolled_train_mask = rolled_train_mask * (traj["opd_token_weight"] > 0)
    aligned = _realign_opd_teacher_logp(
        fake,
        traj["opd_loss_mask"],
        rolled_train_mask,
        torch.zeros_like(traj["logprobs"]),
    )
    check("aligned tensor matches the training layout", aligned.shape == traj["logprobs"].shape)
    train_start = len(clean_prompt) - 1
    got = aligned[0, train_start : train_start + len(output_tokens)]
    check(
        "signature lands on the positions that predict the output tokens",
        torch.allclose(got, signature),
        f"expected {signature.tolist()}, got {got.tolist()}",
    )
    check(
        "nothing written outside those positions",
        float(aligned[0].abs().sum()) == float(signature.abs().sum()),
        f"total {float(aligned[0].abs().sum())} vs {float(signature.abs().sum())}",
    )
    check("skipped row left at zero", float(aligned[1].abs().sum()) == 0.0)

    # A count mismatch means the two layouts do not cover the same tokens; that
    # must be an error rather than a silent partial copy.
    try:
        _realign_opd_teacher_logp(
            fake,
            traj["opd_loss_mask"],
            torch.ones_like(rolled_train_mask),
            torch.zeros_like(traj["logprobs"]),
        )
    except RuntimeError:
        check("a token-count mismatch raises", True)
    else:
        check("a token-count mismatch raises", False, "silently copied a partial set")


def test_trainer_defers_tensor_ops() -> None:
    """compute_logp can return RTensor handles rather than local tensors.

    The data lives on the worker that produced it and is materialized only once
    the batch reaches the actor, so any tensor method called on it in the trainer
    raises -- which is exactly how the first real run died. The trainer must store
    what compute_logp returned and nothing more; `_attach_teacher_context_logps`
    already works this way.
    """
    print("\n[4b] the trainer does not touch the returned log-probs")
    import inspect

    from areal.trainer import rl_trainer

    src = inspect.getsource(rl_trainer._attach_opd_teacher_logps)
    body = src[src.index('"""', src.index('"""') + 3) + 3 :]
    for forbidden in (".to(", "torch.roll", "torch.zeros_like", "[teacher_sel", ".bool()"):
        check(
            f"no {forbidden} on the returned handle",
            forbidden not in body,
            "RTensor supports none of these until it reaches the actor",
        )
    check(
        "stores the handle verbatim",
        'trajectory["opd_teacher_logp"] = teacher_logp' in body,
    )
    check(
        "keeps opd_loss_mask for the actor to realign with",
        'trajectory.pop("opd_loss_mask"' not in body,
    )


def test_opd_advantage() -> None:
    """OPD enters as a per-token advantage penalty, per the reference.

        reverse_kl = log pi_sampled(a_t) - log pi_teacher(a_t)
        advantages = advantages - coef * reverse_kl

    with the ordinary importance-sampling loss run unchanged afterwards. Both
    sides are scored only on the sampled token, and the student side is the
    sampling-time log-prob, so nothing here differentiates through the sampling
    distribution -- it is all data by this point.
    """
    print("\n[5] OPD advantage penalty")
    from areal.trainer.ppo.actor import _compute_opd_advantages

    batch, length = 2, 6
    mask = torch.zeros((batch, length), dtype=torch.bool)
    mask[:, 2:5] = True
    weight = torch.full((batch, length), 1.0)
    no_clip = torch.zeros((batch, length))

    behaviour = torch.full((batch, length), -2.0)

    # Teacher assigns more mass than the sampler did -> reverse KL positive ->
    # advantage positive -> PPO raises the token.
    adv, kl, active = _compute_opd_advantages(
        behaviour, torch.full((batch, length), -0.5), weight, no_clip, mask
    )
    # Sign convention, stated because it is easy to invert: reverse_kl is
    # log pi_sampled - log pi_teacher, so it goes NEGATIVE when the teacher
    # assigns the token more mass than the sampler did. The advantage is its
    # negation, hence positive, and PPO raises the token.
    check("reverse KL negative when the teacher prefers the token",
          bool((kl[mask] < 0).all()), f"{kl[mask][:3].tolist()}")
    check("advantage raises a teacher-preferred token",
          bool((adv[mask] > 0).all()), f"{adv[mask][:3].tolist()}")
    check("magnitude equals the KL in nats",
          torch.allclose(adv[mask], torch.full((int(mask.sum()),), 1.5)),
          f"{adv[mask][:3].tolist()}")

    adv, kl, _ = _compute_opd_advantages(
        behaviour, torch.full((batch, length), -4.0), weight, no_clip, mask
    )
    check("advantage lowers a teacher-disfavoured token",
          bool((adv[mask] < 0).all()), f"{adv[mask][:3].tolist()}")

    # Self-limiting: identical distributions leave the advantage untouched.
    adv, kl, _ = _compute_opd_advantages(
        behaviour, behaviour.clone(), weight, no_clip, mask
    )
    check("no advantage once the policy matches the teacher",
          float(adv.abs().max()) == 0.0, f"max {float(adv.abs().max())}")

    # Unselected rows and unsupervised positions contribute nothing.
    half = torch.zeros((batch, length))
    half[0] = 1.0
    adv, _, active = _compute_opd_advantages(
        behaviour, torch.full((batch, length), 5.0), half, no_clip, mask
    )
    check("zero weight is exactly inert", float(adv[1].abs().max()) == 0.0)
    check("prompt positions untouched", float(adv[0, :2].abs().max()) == 0.0)
    check("active mask is mask AND weight", bool((active == (mask & (half > 0))).all()))

    # coef scales linearly, as a coefficient on a KL in nats.
    adv_a, _, _ = _compute_opd_advantages(
        behaviour, torch.full((batch, length), -1.0), weight, no_clip, mask
    )
    adv_b, _, _ = _compute_opd_advantages(
        behaviour, torch.full((batch, length), -1.0), weight * 3.0, no_clip, mask
    )
    check("coef scales the penalty linearly",
          torch.allclose(adv_b, adv_a * 3.0))

    # The clip is a local addition; 0 must reproduce the reference exactly.
    unclipped, _, _ = _compute_opd_advantages(
        behaviour, torch.full((batch, length), -50.0), weight, no_clip, mask
    )
    clipped, _, _ = _compute_opd_advantages(
        behaviour,
        torch.full((batch, length), -50.0),
        weight,
        torch.full((batch, length), 2.0),
        mask,
    )
    check("clip=0 leaves the KL untouched",
          torch.allclose(unclipped[mask], torch.full((int(mask.sum()),), -48.0)),
          f"{unclipped[mask][:2].tolist()}")
    check("clip bounds the penalty when set",
          torch.allclose(clipped[mask], torch.full((int(mask.sum()),), -2.0)),
          f"{clipped[mask][:2].tolist()}")


def test_opd_absent_from_loss() -> None:
    """The loss must know nothing about OPD -- it sees only the adjusted
    advantage, which is what lets the distillation signal inherit the PPO ratio,
    clipping and behaviour-importance weighting for free."""
    print("\n[6] OPD does not leak into the loss")
    import inspect

    from areal.trainer.ppo import actor as actor_mod

    source = inspect.getsource(actor_mod.grpo_loss_fn)
    check("grpo_loss_fn contains no OPD term", "opd" not in source.lower())

    advantage_src = inspect.getsource(actor_mod.PPOActor._compute_advantages)
    check("_compute_advantages applies it",
          "_compute_opd_advantages" in advantage_src)
    check("added after the task advantage, so it never enters GAE",
          advantage_src.index("_compute_opd_advantages")
          > advantage_src.index("_compute_token_gae"))
    check("not folded into returns",
          advantage_src.index("_compute_opd_advantages")
          > advantage_src.rindex('data["returns"]'))


def main() -> int:
    test_guidance_stripped()
    test_slot_assignment()
    test_opd_eligibility()
    test_opd_realignment()
    test_trainer_defers_tensor_ops()
    test_opd_advantage()
    test_opd_absent_from_loss()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
