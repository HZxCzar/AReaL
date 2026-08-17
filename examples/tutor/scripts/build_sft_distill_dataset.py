#!/usr/bin/env python
"""Build the SFT baseline's training data: the teacher's own solutions.

The third arm of the head-to-head asks what pure distillation is worth. No
dialogue, no teaching, no RL: the teacher answers each training problem under the
standard math prompt, and the student is fine-tuned on those answers. Whatever
the taught arms gain above this number is what teaching bought.

    teacher (Qwen3-8B)  ->  up to --attempts tries per train problem
    judge               ->  stop at the first accepted solution
    tokenize            ->  a DatasetDict the AReaL SFT trainer loads directly

EVERY ROW IS CORRECT AND EVERY PROBLEM WEIGHS THE SAME. Sampling stops at the
first solution the judge accepts, so the dataset needs no downstream filter, and
--per-task 1 keeps exactly one row per solved problem. Keeping every correct
sample instead would hand a problem the teacher finds easy several rows and a
hard one a single row, training the student mostly on what was already easy --
which is the part of the split where teaching has least to add. One row per
problem is also the unit the RL arms train on, so "all three arms train on the
same 759 rows" is literally true.

A problem the teacher cannot solve within --attempts is SKIPPED, not trained on
with a wrong target. summary.json reports `coverage` and `problems_skipped`; quote
them next to whatever the arm scores, because a floor built on 95% of the split
is a floor on 95% of the split.

THE PROMPT IS THE RE-TEST INPUT, BYTE FOR BYTE. Training pairs use exactly what
the student is shown in the zero-dialogue re-test -- the system turn "You are
Sam, a student talking with a teacher." and then
FREE_CHAT_STUDENT_RETEST_TEMPLATE with the problem. Both come from
cross_eval.retest_messages(transcript=[], ...), the same function the scorer
calls, so the training input cannot drift from the eval input. That makes this
the best case for our setting: the student is fine-tuned on the very prompt it is
then tested under.

The consequence, which belongs in the writeup rather than in a flag: this floor
is now optimally aligned to OUR scorer. Their interview scorer prompts the
student a different way, so xeval/no_dialogue/interview is measured under a
prompt this student never saw, and it is the weaker of its two cells by
construction. --train-prompt math restores the previous behaviour if the neutral
version is wanted alongside.

WHICH TEACHER. The endpoint one, which is the same base Qwen3-8B both RL arms
start from. That is the point: this arm is the no-teaching control, so its
teacher must be the teacher before any teaching-specific training. Distilling a
*trained* teacher is a different question and would need its checkpoint served.

WHICH JUDGE. The same one both RL arms score with -- AReaL's exact math scorer,
then its answer-judge prompt on the same auxiliary model -- so "correct" means
here what it means there. --per-task 0 keeps every accepted solution found within
--attempts if the larger, imbalanced set is wanted for comparison.

WHAT IS DELIBERATELY NOT USED. Every row already carries a human-written
solution in metadata['solution']. Training on that would be a different and
probably stronger baseline, and it would not answer the question this arm is for,
which is what the teacher in this experiment can transfer without teaching.

Usage:

    set -a && . ./.env && set +a
    PYTHONPATH=$PWD .venv/bin/python examples/tutor/scripts/build_sft_distill_dataset.py \
        --out /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/sft/data/distill_qwen3_8b

NOTE ON --out. Do not write under examples/tutor/data: that path is a symlink to
the shared AReaL data directory every worktree reads, so a dataset built there
lands outside this experiment and is visible to every other arm. The default
above is this run's own fileroot, which is what the SFT config expects.

Writes, under --out:

    samples.jsonl        every generation with its verdict, for inspection
    correct/             the tokenized dataset; every row judged correct
    summary.json         counts and the teacher's pass rate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

# The official math prompt template, verbatim from examples/tir/prompts.py. Kept
# as the alternative to the re-test prompt: it is the neutral format, and the
# teacher may answer better under it because it is not wearing a student persona.
MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def math_prompt(task: str) -> list[dict[str, str]]:
    """One user turn: the problem, then the standard instruction."""

    return [{"role": "user", "content": f"{task}\n\n{MATH_INSTRUCTION}"}]


def retest_prompt(task: str) -> list[dict[str, str]]:
    """The zero-dialogue re-test input, exactly as the scorer builds it.

    Not a copy of those two strings: it calls the same function
    cross_eval.score_retest calls, with an empty transcript, so the prompt the
    student is trained on cannot drift from the prompt it is tested under.
    """

    from examples.pedagogical_rl.cross_eval import retest_messages

    return retest_messages(transcript=[], task=task)


PROMPTS = {"retest": retest_prompt, "math": math_prompt}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="examples/tutor/data/math_1.7b_8b/math_pass@2",
        help="The split both RL arms train on.",
    )
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--generate-prompt",
        choices=sorted(PROMPTS),
        default="retest",
        help=(
            "What the TEACHER is asked. 'retest' is the eval input; 'math' is "
            "the standard template, under which the teacher may answer better "
            "because it is not wearing a student persona. Compare the "
            "teacher_pass_rate in summary.json before choosing."
        ),
    )
    parser.add_argument(
        "--train-prompt",
        choices=sorted(PROMPTS),
        default="retest",
        help=(
            "What the STUDENT is trained on. 'retest' matches the eval input "
            "byte for byte and is the point of this arm; it does not have to "
            "equal --generate-prompt, since only the target text comes from the "
            "teacher."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help=(
            "Tries per problem. Sampling stops at the first accepted solution, "
            "so at the measured 0.90 pass rate this costs about 1.1 calls a "
            "problem rather than a fixed --attempts."
        ),
    )
    parser.add_argument(
        "--per-task",
        type=int,
        default=1,
        help=(
            "Accepted solutions to keep per problem. 1 weights every problem "
            "equally, which is the fair default: keeping every correct sample "
            "would give a problem the teacher finds easy several rows and a hard "
            "one a single row, training the student mostly on what was already "
            "easy. 0 keeps every accepted solution found within --attempts."
        ),
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=-1,
        help="-1 uses the whole train split. Set small for a smoke run.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Matches the RL arms' auxiliary_model.max_concurrent_calls.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--student-path",
        default=os.environ.get(
            "TUTOR_STUDENT_BASE_MODEL", "examples/tutor/models/student"
        ),
        help="Student tokenizer: the SFT sequence must use its chat template.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Drop tokenized examples longer than this.",
    )
    return parser.parse_args(argv)


async def _generate(
    args: argparse.Namespace,
    rows: list[dict],
    *,
    teacher=None,
    judge=None,
    score=None,
) -> list[dict]:
    """Sample the teacher until the judge accepts, per problem.

    ``teacher`` / ``judge`` / ``score`` are injectable so the loop -- which owns
    the "every kept row is correct" and "an unsolved problem is skipped"
    invariants -- can be tested without an endpoint.
    """

    from examples.pedagogical_rl.api import PedagogicalAPIClient
    from examples.pedagogical_rl.config import PedagogicalAPIModelConfig
    from examples.pedagogical_rl.scoring import score_unified_answer

    score_unified_answer = score or score_unified_answer
    if teacher is not None and judge is not None:
        return await _sample_rows(args, rows, teacher, judge, score_unified_answer)

    base_url = os.environ.get("TUTOR_QWEN3_8B_BASE_URL")
    api_key = os.environ.get("INF_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "TUTOR_QWEN3_8B_BASE_URL and INF_API_KEY are required; "
            "run `set -a && . ./.env && set +a` first."
        )
    # The same endpoint, key and sampling knobs the RL arms use for this model,
    # so the teacher answering here is the teacher they start from.
    teacher_config = PedagogicalAPIModelConfig(
        base_url=base_url,
        model=os.environ.get("PED_AUX_API_MODEL", "qwen3-8b"),
        api_key=api_key,
        timeout=120.0,
        max_concurrent_calls=args.concurrency,
        extra_headers={"x-inspire-inference-key": "tutor-train-qwen8b-auxiliary"},
    )
    teacher = PedagogicalAPIClient(teacher_config)
    judge = PedagogicalAPIClient(teacher_config)
    return await _sample_rows(args, rows, teacher, judge, score_unified_answer)


async def _sample_rows(args, rows, teacher, judge, score_unified_answer):
    wanted = max(0, int(args.per_task))
    max_attempts = max(1, int(args.attempts))

    async def one(row: dict) -> list[dict]:
        """Sample until the judge accepts, then stop.

        Every kept record is correct by construction, so the SFT set needs no
        downstream filter. A problem the teacher cannot solve in max_attempts
        tries is skipped outright rather than contributing a wrong target: the
        arm is a distillation floor, and training the student on the teacher's
        mistakes would measure something else.
        """

        task = str(row["task"])
        ground_truth = str(row["ground_truth"])
        records: list[dict] = []
        kept = 0
        for attempt in range(1, max_attempts + 1):
            try:
                solutions = await teacher.generate(
                    PROMPTS[args.generate_prompt](task),
                    n=1,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as exc:  # noqa: BLE001 - a dead try is not fatal
                records.append(
                    {
                        "id": row.get("id", ""),
                        "task": task,
                        "ground_truth": ground_truth,
                        "attempt": attempt,
                        "solution": "",
                        "correct": False,
                        "kept": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            solution = solutions[0] if solutions else ""
            try:
                verdict = await score_unified_answer(
                    task=task,
                    ground_truth=ground_truth,
                    student_answer=solution,
                    judge=judge,
                )
                correct, error = bool(verdict.correct), None
            except Exception as exc:  # noqa: BLE001
                correct, error = False, f"judge failed: {type(exc).__name__}: {exc}"
            keep = correct and (wanted == 0 or kept < wanted)
            if keep:
                kept += 1
            records.append(
                {
                    "id": row.get("id", ""),
                    "task": task,
                    "ground_truth": ground_truth,
                    "attempt": attempt,
                    "solution": solution,
                    "correct": correct,
                    "kept": keep,
                    "error": error,
                }
            )
            if wanted and kept >= wanted:
                break
        return records

    # The client holds its own semaphore at max_concurrent_calls, so every
    # problem can be launched at once and the endpoint is what paces it.
    batches = await asyncio.gather(*(one(row) for row in rows))
    return [record for batch in batches for record in batch]


def _tokenize(records: list[dict], *, tokenizer, max_length: int, prompt="retest"):
    """Prompt tokens masked out, solution tokens trained on.

    The same shape areal/dataset/gsm8k.py:get_gsm8k_sft_dataset produces, built
    through the student's chat template so the sequence the student is trained on
    is the shape it will be prompted in.
    """

    from datasets import Dataset

    rows = []
    dropped_long = 0
    for record in records:
        solution = str(record.get("solution") or "").strip()
        if not solution:
            continue
        prompt_ids = tokenizer.apply_chat_template(
            PROMPTS[prompt](record["task"]),
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        solution_ids = tokenizer.encode(solution, add_special_tokens=False)
        eos = tokenizer.eos_token_id
        seq = list(prompt_ids) + list(solution_ids) + ([eos] if eos is not None else [])
        if len(seq) > max_length:
            dropped_long += 1
            continue
        loss_mask = [0] * len(prompt_ids) + [1] * (len(seq) - len(prompt_ids))
        rows.append({"input_ids": seq, "loss_mask": loss_mask})
    return Dataset.from_list(rows), dropped_long


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    from datasets import DatasetDict, load_from_disk

    from areal.utils.hf_utils import load_hf_tokenizer

    dataset = load_from_disk(args.dataset)
    train_rows = list(dataset["train"])
    if args.max_problems > 0:
        train_rows = train_rows[: args.max_problems]
    print(
        f"sampling up to {args.attempts} tries for {len(train_rows)} problems, "
        f"keeping {args.per_task or 'every'} accepted solution per problem; "
        f"generate prompt {args.generate_prompt}, train prompt {args.train_prompt}"
    )

    records = asyncio.run(_generate(args, train_rows))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    kept = [r for r in records if r.get("kept")]
    scored = [r for r in records if not r.get("error")]
    correct_ids = {r["id"] for r in kept}
    all_ids = {r["id"] for r in records}
    # A problem the teacher never solved within --attempts. It contributes
    # nothing rather than a wrong target, so this number is the honest coverage
    # of the arm and belongs next to any score it produces.
    skipped = sorted(all_ids - correct_ids)
    attempts_used = {}
    for record in records:
        attempts_used[record["id"]] = max(
            attempts_used.get(record["id"], 0), int(record.get("attempt", 0))
        )
    tokenizer = load_hf_tokenizer(args.student_path)
    summary = {
        "generate_prompt": args.generate_prompt,
        "train_prompt": args.train_prompt,
        "attempts_allowed": args.attempts,
        "kept_per_task": args.per_task,
        "problems": len(train_rows),
        "problems_solved": len(correct_ids),
        "problems_skipped": len(skipped),
        "coverage": len(correct_ids) / len(train_rows) if train_rows else 0.0,
        "generated": len(records),
        "errors": len(records) - len(scored),
        # Per ATTEMPT, so it is comparable across --attempts settings.
        "teacher_pass_rate": (
            sum(1 for r in scored if r["correct"]) / len(scored) if scored else 0.0
        ),
        "mean_attempts_used": (
            sum(attempts_used.values()) / len(attempts_used) if attempts_used else 0.0
        ),
        "examples": len(kept),
        "skipped_ids": skipped[:50],
    }

    # One dataset, and every row in it is judged correct. The valid split is the
    # SFT trainer's loss-only holdout, carved out of the TRAIN problems: the 528
    # test problems are what all three arms are measured on and must never be
    # trained or tuned on.
    tokenized, dropped = _tokenize(
        kept,
        tokenizer=tokenizer,
        max_length=args.max_length,
        prompt=args.train_prompt,
    )
    summary["tokenized_examples"] = len(tokenized)
    summary["dropped_too_long"] = dropped
    if len(tokenized) < 10:
        print(f"  only {len(tokenized)} examples, not writing a dataset")
    else:
        split = tokenized.train_test_split(test_size=0.02, seed=42)
        DatasetDict({"train": split["train"], "test": split["test"]}).save_to_disk(
            str(out / "correct")
        )
        print(
            f"  correct/: {len(split['train'])} train / {len(split['test'])} holdout"
            f" ({dropped} dropped over {args.max_length} tokens)"
        )

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped_ids"}, indent=2))
    if skipped:
        print(f"\n{len(skipped)} problems the teacher never solved in {args.attempts} tries")


if __name__ == "__main__":
    main(sys.argv[1:])
