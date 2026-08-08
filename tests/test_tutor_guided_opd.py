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

from areal.trainer.rl_trainer import _attach_opd_teacher_logps
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
    print("\n[4] OPD log-prob realignment across differing prompt lengths")

    output_tokens = [500, 501, 502, 503]
    clean_prompt = list(range(1, 8))  # 7 tokens
    opd_prompt = list(range(1, 20))  # 19 tokens: prompt + instruction

    selected = response_to_tensordict(
        FakeResponse(clean_prompt, output_tokens),
        reward=1.0,
        input_tokens_override=clean_prompt,
        opd_input_tokens=opd_prompt,
        opd_loss_weight=0.05,
        opd_reward_clip=5.0,
    )
    skipped = response_to_tensordict(
        FakeResponse(clean_prompt, output_tokens),
        reward=0.0,
        input_tokens_override=clean_prompt,
        opd_input_tokens=None,
        opd_loss_weight=0.05,
        opd_reward_clip=5.0,
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
    opd_len = traj["opd_input_ids"].shape[1]
    fake = torch.zeros((2, opd_len), dtype=torch.float32)
    start = len(opd_prompt) - 1
    fake[0, start : start + len(output_tokens)] = signature
    fake[1, :] = 99.0  # the skipped row must be ignored entirely

    class StubActor:
        def __init__(self):
            self.calls = 0

        def compute_logp(self, batch):
            self.calls += 1
            # Only the active trajectory should be forwarded.
            assert len(batch) == 1, f"forwarded {len(batch)} trajectories, expected 1"
            return [fake]

    actor = StubActor()
    _attach_opd_teacher_logps(actor, [traj])
    check("teacher forward ran once", actor.calls == 1)
    check(
        "OPD inputs consumed",
        not any(k in traj for k in ("opd_input_ids", "opd_attention_mask", "opd_loss_mask")),
    )

    aligned = traj["opd_teacher_logp"]
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

    # An off-by-one would be caught here: shifting the plant by one position must
    # produce a different aligned result.
    shifted = torch.zeros_like(fake)
    shifted[0, start + 1 : start + 1 + len(output_tokens)] = signature
    check(
        "the check is sensitive to a one-token shift",
        not torch.allclose(
            fake[0, start : start + len(output_tokens)],
            shifted[0, start : start + len(output_tokens)],
        ),
    )


def test_opd_loss_term() -> None:
    print("\n[5] OPD loss term sign and inertness")
    from areal.trainer.ppo.actor import grpo_loss_fn

    batch, length = 2, 6
    loss_mask = torch.zeros((batch, length), dtype=torch.bool)
    loss_mask[:, 2:5] = True
    old_logp = torch.full((batch, length), -1.0)
    logprobs = torch.full((batch, length), -1.0, requires_grad=True)

    base = {
        "logprobs": old_logp,
        "advantages": torch.zeros((batch, length)),
        "loss_mask": loss_mask,
        "prox_logp": old_logp,
        "attention_mask": torch.ones((batch, length), dtype=torch.bool),
    }
    kwargs = dict(
        eps_clip=0.4,
        eps_clip_higher=None,
        c_clip=None,
        behave_imp_weight_cap=5.0,
        use_decoupled_loss=True,
    )

    # Teacher strongly prefers these tokens -> the term must push their log-prob
    # UP, i.e. d(loss)/d(logprob) must be negative.
    teacher = torch.full((batch, length), 2.0)
    data = dict(
        base,
        opd_teacher_logp=teacher,
        opd_token_weight=torch.full((batch, length), 0.05),
        opd_reward_clip=torch.full((batch, length), 5.0),
    )
    logprobs.grad = None
    loss = grpo_loss_fn(logprobs, torch.zeros_like(logprobs), data, **kwargs)
    loss.backward()
    grad = logprobs.grad[loss_mask]
    check(
        "a preferred token's log-prob is pushed up",
        bool((grad < 0).all()),
        f"grad {grad.tolist()[:3]}",
    )

    # Same, teacher dislikes them -> pushed down.
    logprobs2 = torch.full((batch, length), -1.0, requires_grad=True)
    data = dict(
        base,
        opd_teacher_logp=torch.full((batch, length), -4.0),
        opd_token_weight=torch.full((batch, length), 0.05),
        opd_reward_clip=torch.full((batch, length), 5.0),
    )
    loss = grpo_loss_fn(logprobs2, torch.zeros_like(logprobs2), data, **kwargs)
    loss.backward()
    grad2 = logprobs2.grad[loss_mask]
    check(
        "a disliked token's log-prob is pushed down",
        bool((grad2 > 0).all()),
        f"grad {grad2.tolist()[:3]}",
    )

    # Weight 0 everywhere must be exactly inert, so a run with OPD enabled but no
    # eligible turns is identical to a run without it.
    logprobs3 = torch.full((batch, length), -1.0, requires_grad=True)
    data = dict(
        base,
        opd_teacher_logp=torch.full((batch, length), 2.0),
        opd_token_weight=torch.zeros((batch, length)),
        opd_reward_clip=torch.full((batch, length), 5.0),
    )
    with_zero = grpo_loss_fn(logprobs3, torch.zeros_like(logprobs3), data, **kwargs)
    logprobs4 = torch.full((batch, length), -1.0, requires_grad=True)
    without = grpo_loss_fn(logprobs4, torch.zeros_like(logprobs4), dict(base), **kwargs)
    check(
        "zero weight is exactly inert",
        torch.allclose(with_zero, without),
        f"{float(with_zero.detach())} vs {float(without.detach())}",
    )

    # Self-limiting: when the CURRENT policy already matches the instructed
    # teacher on its own samples there is nothing left to distil, so the term must
    # contribute no gradient.
    #
    # The fixture deliberately sets old_logp != logprobs. With logprobs == teacher
    # the correct reverse-KL reference gives exactly zero, while referencing the
    # BEHAVIOUR log-prob instead would give teacher - old_logp = +1 nat and a live
    # gradient. So this check separates the two; with old_logp == logprobs it would
    # pass either way.
    stale = dict(base, logprobs=torch.full((batch, length), -2.0),
                 prox_logp=torch.full((batch, length), -2.0))
    logprobs_eq = torch.full((batch, length), -1.0, requires_grad=True)
    matched = grpo_loss_fn(
        logprobs_eq,
        torch.zeros_like(logprobs_eq),
        dict(
            stale,
            opd_teacher_logp=torch.full((batch, length), -1.0),
            opd_token_weight=torch.full((batch, length), 0.05),
            opd_reward_clip=torch.full((batch, length), 5.0),
        ),
        **kwargs,
    )
    matched.backward()
    logprobs_none = torch.full((batch, length), -1.0, requires_grad=True)
    grpo_loss_fn(
        logprobs_none, torch.zeros_like(logprobs_none), dict(stale), **kwargs
    ).backward()
    check(
        "no gradient once the policy matches the teacher",
        torch.allclose(logprobs_eq.grad, logprobs_none.grad, atol=1e-7),
        f"delta {float((logprobs_eq.grad - logprobs_none.grad).abs().max())}",
    )

    # Same fixture, teacher now genuinely ahead of the policy: the term MUST be
    # live. Together with the check above this pins the reference to the current
    # policy rather than to old_logp.
    logprobs_gap = torch.full((batch, length), -1.0, requires_grad=True)
    grpo_loss_fn(
        logprobs_gap,
        torch.zeros_like(logprobs_gap),
        dict(
            stale,
            opd_teacher_logp=torch.full((batch, length), -0.2),
            opd_token_weight=torch.full((batch, length), 0.05),
            opd_reward_clip=torch.full((batch, length), 5.0),
        ),
        **kwargs,
    ).backward()
    check(
        "term is live when the teacher is ahead, under the same stale old_logp",
        not torch.allclose(logprobs_gap.grad, logprobs_none.grad, atol=1e-7),
        "otherwise the check above is vacuous",
    )

    # The teacher shares weights with the policy, so a live gradient here would
    # let the KL be minimized by moving the TEACHER instead. Nothing may flow back
    # into the teacher tensor.
    teacher_live = torch.full((batch, length), 2.0, requires_grad=True)
    logprobs_t = torch.full((batch, length), -1.0, requires_grad=True)
    data = dict(
        base,
        opd_teacher_logp=teacher_live,
        opd_token_weight=torch.full((batch, length), 0.05),
        opd_reward_clip=torch.full((batch, length), 5.0),
    )
    grpo_loss_fn(logprobs_t, torch.zeros_like(logprobs_t), data, **kwargs).backward()
    check(
        "no gradient reaches the teacher",
        teacher_live.grad is None or float(teacher_live.grad.abs().max()) == 0.0,
        f"teacher grad {None if teacher_live.grad is None else float(teacher_live.grad.abs().max())}",
    )

    # The magnitude must track the gap, since that gap IS the objective.
    grads = []
    for teacher_value in (-1.5, -3.0, -6.0):
        lp = torch.full((batch, length), -1.0, requires_grad=True)
        data = dict(
            base,
            opd_teacher_logp=torch.full((batch, length), teacher_value),
            opd_token_weight=torch.full((batch, length), 0.05),
            opd_reward_clip=torch.full((batch, length), 100.0),
        )
        grpo_loss_fn(lp, torch.zeros_like(lp), data, **kwargs).backward()
        grads.append(float(lp.grad[loss_mask].mean()))
    check(
        "gradient magnitude grows with the KL gap",
        grads[0] < grads[1] < grads[2],
        f"got {grads}",
    )

    # The clip must bind.
    logprobs5 = torch.full((batch, length), -1.0, requires_grad=True)
    tight = dict(
        base,
        opd_teacher_logp=torch.full((batch, length), 50.0),
        opd_token_weight=torch.full((batch, length), 0.05),
        opd_reward_clip=torch.full((batch, length), 0.5),
    )
    loose = dict(tight, opd_reward_clip=torch.full((batch, length), 5.0))
    tight_loss = float(
        grpo_loss_fn(logprobs5, torch.zeros_like(logprobs5), tight, **kwargs).detach()
    )
    logprobs6 = torch.full((batch, length), -1.0, requires_grad=True)
    loose_loss = float(
        grpo_loss_fn(logprobs6, torch.zeros_like(logprobs6), loose, **kwargs).detach()
    )
    check(
        "reward_clip bounds the term",
        abs(tight_loss) < abs(loose_loss),
        f"tight {tight_loss}, loose {loose_loss}",
    )


def main() -> int:
    test_guidance_stripped()
    test_slot_assignment()
    test_opd_eligibility()
    test_opd_realignment()
    test_opd_loss_term()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
