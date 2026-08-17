#!/usr/bin/env python
"""Score a student checkpoint with no dialogue, under both scorers.

The SFT arm's row of the head-to-head table. The student is handed the problem
and nothing else, and both measurements run on it:

    xeval/no_dialogue/retest/success      our scorer:  4 solo attempts, LLM judge
    xeval/no_dialogue/interview/success   their scorer: 8 attempts, exact boxed

Both come from examples/pedagogical_rl/cross_eval.py, the same functions the two
RL arms are scored with, so a distilled student and a taught student are compared
on one instrument rather than on two that happen to look alike.

ALWAYS EVALUATE THE BASE MODEL TOO -- that is what --base is for, and it is on by
default. The RL arms reach their student over the INF endpoint; this script
serves a checkpoint locally with SGLang, and the two are not bit-identical
(different batching, different kernels, and the endpoint pins a seed that does
not pin sampling). Running the base student through THIS harness gives the SFT
gain as a within-harness difference, which is the number to quote. The absolute
values are not directly comparable with the endpoint-measured arms.

Usage:

    set -a && . ./.env && set +a
    PYTHONPATH=$PWD .venv/bin/python examples/tutor/scripts/eval_sft_student.py \
        --checkpoint /path/to/sft/checkpoint \
        --out /path/to/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="SFT'd student. Omit to measure only the base model.",
    )
    parser.add_argument(
        "--base",
        default=os.environ.get(
            "TUTOR_STUDENT_BASE_MODEL", "examples/tutor/models/student"
        ),
        help="Base student, measured through this same harness as the control.",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--dataset",
        default="examples/tutor/data/math_1.7b_8b/math_pass@2",
        help="The 528-problem test split all three arms are measured on.",
    )
    parser.add_argument("--out", default="", help="Write the full results here.")
    parser.add_argument(
        "--max-problems",
        type=int,
        default=-1,
        help="-1 uses the whole test split. Set small for a smoke run.",
    )
    # Defaults match student_models[0] in the RL arms' config, so the student
    # being measured here is sampled the way it is sampled there.
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--judge-concurrency", type=int, default=32)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    return parser.parse_args(argv)


def _make_student_call(engine, tokenizer, args):
    """Adapt an SGLang offline engine into cross_eval's student callable."""

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_tokens,
    }

    async def student_call(
        messages, *, n=1, max_tokens=None, rid_prefix="sft-eval", timeout=None
    ) -> list[str]:
        del rid_prefix, timeout
        # enable_thinking=False, matching how the endpoint serves this student in
        # the RL arms; leaving it on would change what is being measured.
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        params = dict(sampling)
        if max_tokens:
            params["max_new_tokens"] = int(max_tokens)
        params["n"] = int(n)
        outputs = await engine.async_generate(prompt=prompt, sampling_params=params)
        if isinstance(outputs, dict):
            outputs = [outputs]
        texts = [str(item.get("text", "")) for item in outputs]
        if len(texts) != n:
            raise RuntimeError(f"engine returned {len(texts)} samples, expected {n}")
        return texts

    return student_call


async def _score_model(*, path: str, label: str, rows: list[dict], args) -> dict:
    import sglang

    from examples.pedagogical_rl import cross_eval as ce
    from examples.pedagogical_rl.api import PedagogicalAPIClient
    from examples.pedagogical_rl.config import PedagogicalAPIModelConfig
    from examples.pedagogical_rl.scoring import score_unified_answer
    from areal.utils.hf_utils import load_hf_tokenizer

    base_url = os.environ.get("TUTOR_QWEN3_8B_BASE_URL")
    api_key = os.environ.get("INF_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "TUTOR_QWEN3_8B_BASE_URL and INF_API_KEY are required for the answer "
            "judge; run `set -a && . ./.env && set +a` first."
        )
    judge = PedagogicalAPIClient(
        PedagogicalAPIModelConfig(
            base_url=base_url,
            model=os.environ.get("PED_AUX_API_MODEL", "qwen3-8b"),
            api_key=api_key,
            timeout=120.0,
            max_concurrent_calls=args.judge_concurrency,
            extra_headers={"x-inspire-inference-key": "tutor-train-qwen8b-auxiliary"},
        )
    )

    async def answer_judge(*, task: str, ground_truth: str, answer: str) -> bool:
        result = await score_unified_answer(
            task=task,
            ground_truth=ground_truth,
            student_answer=answer,
            judge=judge,
        )
        return bool(result.correct)

    print(f"[{label}] loading {path}")
    tokenizer = load_hf_tokenizer(path)
    engine = sglang.Engine(
        model_path=path,
        mem_fraction_static=args.mem_fraction_static,
        skip_tokenizer_init=False,
        log_level="error",
    )
    try:
        student_call = _make_student_call(engine, tokenizer, args)
        retest_spec = ce.RetestSpec(replays=args.replays, max_tokens=args.max_tokens)
        interview_spec = ce.InterviewSpec(
            attempts=args.attempts, max_tokens=args.max_tokens
        )

        async def one(row):
            return await ce.score_no_dialogue(
                task=str(row["task"]),
                ground_truth=str(row["ground_truth"]),
                retest=retest_spec,
                interview=interview_spec,
                student_call=student_call,
                answer_judge=answer_judge,
            )

        results = await asyncio.gather(*(one(row) for row in rows))
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    per_problem = []
    for row, scored in zip(rows, results):
        per_problem.append(
            {
                "id": row.get("id", ""),
                "retest": scored["retest"].score,
                "retest_any": scored["retest"].any_correct,
                "retest_incomplete": not scored["retest"].complete,
                "interview": scored["interview"].score,
                "interview_any": scored["interview"].any_correct,
                "interview_math": scored["interview"].score_math,
                "interview_incomplete": not scored["interview"].complete,
            }
        )

    def mean(key: str) -> float:
        values = [
            float(item[key]) for item in per_problem if item[key] is not None
        ]
        return statistics.fmean(values) if values else 0.0

    summary = {
        "label": label,
        "path": path,
        "problems": len(rows),
        "xeval/no_dialogue/retest/success": mean("retest"),
        "xeval/no_dialogue/retest/success_any": mean("retest_any"),
        "xeval/no_dialogue/retest/incomplete": mean("retest_incomplete"),
        "xeval/no_dialogue/interview/success": mean("interview"),
        "xeval/no_dialogue/interview/success_any": mean("interview_any"),
        "xeval/no_dialogue/interview/success_math": mean("interview_math"),
        "xeval/no_dialogue/interview/incomplete": mean("interview_incomplete"),
    }
    print(f"[{label}] " + json.dumps({k: round(v, 4) for k, v in summary.items() if isinstance(v, float)}))
    return {"summary": summary, "per_problem": per_problem}


def main(argv: list[str]) -> None:
    args = _parse_args(argv)
    from datasets import load_from_disk

    rows = list(load_from_disk(args.dataset)["test"])
    if args.max_problems > 0:
        rows = rows[: args.max_problems]
    print(f"scoring {len(rows)} test problems, no dialogue")

    payload: dict = {"problems": len(rows), "models": []}
    # The two engines are loaded one after the other, not together: a single GPU
    # cannot hold both at mem_fraction_static 0.8, and interleaving them would
    # only make the comparison depend on which was resident.
    if not args.skip_base:
        payload["models"].append(
            asyncio.run(
                _score_model(path=args.base, label="base", rows=rows, args=args)
            )
        )
    if args.checkpoint:
        payload["models"].append(
            asyncio.run(
                _score_model(
                    path=args.checkpoint, label="sft", rows=rows, args=args
                )
            )
        )

    summaries = {item["summary"]["label"]: item["summary"] for item in payload["models"]}
    if "base" in summaries and "sft" in summaries:
        payload["gain"] = {
            key: summaries["sft"][key] - summaries["base"][key]
            for key in summaries["base"]
            if isinstance(summaries["base"][key], float)
        }
        print("\nSFT gain, within this harness:")
        print(json.dumps({k: round(v, 4) for k, v in payload["gain"].items()}, indent=2))

    if args.out:
        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
