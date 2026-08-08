#!/usr/bin/env python3
"""Exercise OPD on a batch shaped like a real one.

The earlier realignment test hand-built a two-row batch and passed the function a
training mask it had already restricted to supervised rows -- which is not what
the caller passes. It therefore encoded the intended behaviour rather than the
actual call, and missed that most rows of a real batch are not supervised at all.
The counts then differed by an order of magnitude on the first update.

This builds the batch the way the pipeline does -- per-turn tensordicts, padded
into episodes, episodes padded into a group -- with the realistic mix of
supervised and skipped turns, and calls the functions with exactly the arguments
`_compute_advantages` uses.

Run from the repo root:  python3 tests/test_tutor_opd_batch.py
"""
from __future__ import annotations

import sys

import torch

from areal.trainer.ppo.actor import (
    _compute_opd_advantages,
    _realign_opd_teacher_logp,
)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


class FakeResponse:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = list(input_tokens)
        self.output_tokens = list(output_tokens)
        self.output_logprobs = [-1.5] * len(output_tokens)
        self.output_versions = [0] * len(output_tokens)
        self.stop_reason = "stop"
        self.input_len = len(self.input_tokens)
        self.tokenizer = None


def build_batch(episodes):
    """episodes: list of list of (prompt_len, output_len, supervised)."""
    from examples.tutor.core.tensors import response_to_tensordict
    from areal.utils.data import concat_padded_tensors

    rows = []
    per_episode = []
    for turns in episodes:
        turn_dicts = []
        for prompt_len, output_len, supervised in turns:
            prompt = list(range(1, prompt_len + 1))
            output = list(range(9000, 9000 + output_len))
            # The instructed teacher prompt is the training prompt plus the
            # instruction, so it is strictly longer and by a varying amount.
            opd_prompt = list(range(1, prompt_len + 37)) if supervised else None
            turn_dicts.append(
                response_to_tensordict(
                    FakeResponse(prompt, output),
                    reward=1.0 if supervised else 0.0,
                    input_tokens_override=prompt,
                    trajectory_id=len(per_episode) + 1,
                    turn_idx=len(turn_dicts) + 1,
                    opd_input_tokens=opd_prompt,
                    opd_loss_weight=1.0,
                    opd_reward_clip=0.0,
                )
            )
            rows.append((prompt_len, output_len, supervised))
        per_episode.append(concat_padded_tensors(turn_dicts))
    return concat_padded_tensors(per_episode), rows


def main() -> int:
    # 8 episodes, uneven turn counts, and the realistic pattern: OPD only kicks in
    # from turn 3, so most rows are skipped. Prompt and output lengths vary so no
    # accidental alignment can hide an offset error.
    episodes = [
        [(40, 20, False), (55, 31, False), (61, 27, True), (77, 19, True)],
        [(38, 25, False)],
        [(44, 17, False), (66, 40, False)],
        [(51, 33, False), (58, 22, False), (70, 28, True)],
        [(47, 12, False)],
        [(42, 19, False), (63, 35, False), (72, 24, True), (81, 30, True), (90, 16, True)],
        [(39, 21, False), (60, 26, False)],
        [(45, 18, False)],
    ]
    batch, rows = build_batch(episodes)
    n_rows = batch["input_ids"].shape[0]
    n_supervised = sum(1 for _, _, s in rows if s)
    print(f"\n[1] batch shape   rows={n_rows}  supervised={n_supervised}")
    check("row count matches", n_rows == len(rows), f"{n_rows} vs {len(rows)}")
    check(
        "most rows are NOT supervised, as in a real run",
        n_supervised < n_rows / 2,
        f"{n_supervised}/{n_rows}",
    )
    check(
        "the teacher layout is wider than the training layout",
        batch["opd_loss_mask"].shape[1] > batch["input_ids"].shape[1],
        f"{batch['opd_loss_mask'].shape[1]} vs {batch['input_ids'].shape[1]}",
    )

    print("\n[2] realignment, called exactly as _compute_advantages calls it")
    # Mirror the caller: loss_mask is rolled once, up front, and passed straight
    # through with no row filtering applied by the caller.
    rolled_loss_mask = torch.roll(batch["loss_mask"].float(), shifts=-1, dims=-1)
    old_logp = torch.roll(batch["logprobs"], shifts=-1, dims=-1) * rolled_loss_mask

    # Plant a distinct value per supervised row at the positions that predict its
    # output tokens, so a misplacement shows up as a wrong value, not just a count.
    opd_mask = batch["opd_loss_mask"].bool()
    teacher = torch.zeros_like(batch["opd_loss_mask"], dtype=torch.float32)
    rolled_opd = torch.roll(opd_mask, shifts=-1, dims=-1)
    signature = {}
    for row in range(n_rows):
        if not bool(opd_mask[row].any()):
            teacher[row] = 777.0  # must never be read
            continue
        # Kept well away from old_logp (-1.5): a value that happens to equal it
        # gives a genuinely zero KL, which is correct behaviour but would look
        # like the penalty failing to fire.
        value = -0.01 * (row + 1) - 0.05
        teacher[row][rolled_opd[row]] = value
        signature[row] = value

    try:
        aligned = _realign_opd_teacher_logp(
            teacher, batch["opd_loss_mask"], rolled_loss_mask, old_logp
        )
    except RuntimeError as exc:
        check("realignment succeeds on a realistic batch", False, str(exc))
        print()
        print("1 FAILED")
        return 1
    check("realignment succeeds on a realistic batch", True)
    check("aligned matches the training layout", aligned.shape == old_logp.shape)

    ok_values, ok_zero, ok_placement = True, True, True
    for row in range(n_rows):
        row_mask = rolled_loss_mask[row].bool()
        if row in signature:
            got = aligned[row][row_mask]
            if not torch.allclose(got, torch.full_like(got, signature[row])):
                ok_values = False
            # nothing outside the output positions
            if float(aligned[row][~row_mask].abs().sum()) != 0.0:
                ok_placement = False
        else:
            if float(aligned[row].abs().sum()) != 0.0:
                ok_zero = False
    check("every supervised row carries its own teacher value", ok_values)
    check("skipped rows stay exactly zero (the 777 is never read)", ok_zero)
    check("nothing written outside the output positions", ok_placement)

    print("\n[3] the advantage penalty over the same batch")
    advantages, reverse_kl, active = _compute_opd_advantages(
        old_logp,
        aligned,
        batch["opd_token_weight"].float(),
        batch["opd_reward_clip"].float(),
        rolled_loss_mask,
    )
    active_rows = {int(r) for r in active.any(dim=-1).nonzero().flatten()}
    check(
        "exactly the supervised rows are active",
        active_rows == set(signature),
        f"{sorted(active_rows)} vs {sorted(signature)}",
    )
    check(
        "skipped rows get no advantage",
        all(float(advantages[r].abs().sum()) == 0.0 for r in range(n_rows) if r not in signature),
    )
    check("supervised rows do get one", all(float(advantages[r].abs().sum()) > 0 for r in signature))
    check(
        "active token count equals the supervised output tokens",
        int(active.count_nonzero())
        == sum(output_len for (_, output_len, s), _ in zip(rows, range(n_rows)) if s),
        f"{int(active.count_nonzero())}",
    )

    print("\n[4] an off-by-one in the teacher layout must be caught")
    shifted = torch.roll(teacher, shifts=1, dims=-1)
    aligned_shifted = _realign_opd_teacher_logp(
        shifted, batch["opd_loss_mask"], rolled_loss_mask, old_logp
    )
    check(
        "a one-token shift changes the result",
        not torch.allclose(aligned, aligned_shifted),
        "the check would be blind to an offset error",
    )

    print("\n[5] a genuinely inconsistent batch still raises")
    broken = batch["opd_loss_mask"].clone()
    first = next(iter(signature))
    idx = broken[first].nonzero().flatten()
    broken[first, idx[0]] = 0  # one fewer teacher token than training expects
    try:
        _realign_opd_teacher_logp(teacher, broken, rolled_loss_mask, old_logp)
    except RuntimeError as exc:
        check("count mismatch raises", True)
        check(
            "the message names the supervised-row count",
            "supervised rows" in str(exc),
            str(exc)[:120],
        )
    else:
        check("count mismatch raises", False, "silently copied a partial set")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all batch checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
