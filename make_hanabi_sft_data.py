#!/usr/bin/env python
"""
Utility script to convert Hanabi workflow logs into SFT-ready samples.

Args:
log-dir: The directory containing .josnl log files
output: The destination jsonl to write to
min-score: The minimum final score threshold to accept a trajectory
limit: The maximum number of trajectories to read from
dry-run: If true, only print the statistics of the log files; min-score will be ignored

Example:
python make_hanabi_sft_data.py \
    --log-dir /storage/openpsi/experiments/logs/admin/xmy-hanabi-gather-data/Qwen3-32b-large-2/generated/0/ /storage/openpsi/experiments/logs/admin/xmy-hanabi-gather-data/Qwen3-32b-large-3/generated/0/ \
    --output /storage/openpsi/data/hanabi-sft/Qwen3_data_Qwen3_template_small.jsonl \
    --tokenizer-path /storage/openpsi/models/Qwen__Qwen3-4B \
    --min-score 15 \
    --limit 400 \
    --dry-run
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
from realhf.api.core.data_api import load_hf_tokenizer
from transformers import PreTrainedTokenizerBase

def _iter_episode_records(log_dirs: Sequence[Path], limit: int = 0) -> Iterable[Tuple[Path, int, dict]]:
    """Yield parsed JSON objects from every log file under the given ``log_dirs``."""

    resolved_dirs = [Path(path) for path in log_dirs]
    file_paths: list[Path] = []
    for log_dir in resolved_dirs:
        if not log_dir.exists():
            print(f"[warn] Log directory '{log_dir}' does not exist; skipping.", file=sys.stderr)
            continue
        file_paths.extend(sorted(log_dir.rglob("*.jsonl")))

    print(f"Found {len(file_paths)} logs to read from across {len(resolved_dirs)} directories.", flush=True)

    for idx, file_path in enumerate(file_paths):
        if idx % 200 == 1:
            print(
                f"Processing {idx}/{len(file_paths) if limit == 0 else min(limit, len(file_paths))} json log...",
                flush=True,
            )
        if limit != 0 and idx >= limit:
            return
        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue

        if not content.strip():
            continue

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

        if parsed is not None:
            if isinstance(parsed, list):
                for offset, record in enumerate(parsed, 1):
                    yield file_path, offset, record
                continue
            if isinstance(parsed, dict):
                yield file_path, 1, parsed
                continue
            print(
                f"[warn] Parsed JSON root in {file_path} is neither a list nor an object; skipping.",
                file=sys.stderr,
            )
            continue

        decoder = json.JSONDecoder()
        idx_start = 0
        content_length = len(content)
        while idx_start < content_length:
            while idx_start < content_length and content[idx_start].isspace():
                idx_start += 1
            if idx_start >= content_length:
                break
            try:
                record, idx_end = decoder.raw_decode(content, idx_start)
            except json.JSONDecodeError as exc:  # pragma: no cover - logged for debugging
                line_no = content.count("\n", 0, idx_start) + 1
                print(
                    f"[warn] Failed to parse JSON at {file_path}:{line_no}: {exc}",
                    file=sys.stderr,
                )
                break
            line_no = content.count("\n", 0, idx_start) + 1
            yield file_path, line_no, record
            idx_start = idx_end


def _resolve_score(summary: dict) -> float | None:
    score = summary.get("final_score")
    if score is not None:
        return float(score)
    stats = summary.get("stats", {})
    if isinstance(stats, dict) and stats.get("score") is not None:
        try:
            return float(stats.get("score"))
        except (TypeError, ValueError):
            return None
    return None


def _extract_leading_think_block(text: str | None) -> tuple[str, str | None]:
    if not text:
        return "", text

    match = re.match(r"^\s*(<think>.*?</think>)\s*", text, flags=re.DOTALL)
    if not match:
        return "", text

    return match.group(1), text[match.end() :]


def _apply_chat_template(
    prompt: str, response: str, tokenizer: PreTrainedTokenizerBase | None
) -> Tuple[str, str]:
    if tokenizer is None:
        return prompt, response

    user_message = [{"role": "user", "content": prompt}]
    prompt_with_template = tokenizer.apply_chat_template(
        user_message, tokenize=False, add_generation_prompt=True
    )
    response_with_template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_with_template, response_with_template


def _append_sample(
    samples: list[dict],
    prompt: str | None,
    response: str | None,
    *,
    task_type: str,
    turn: int | None,
    player: int | None,
    metadata: dict,
    reward: float | None,
    discounted_return: float | None,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    if not prompt or not response:
        return

    prompt, response = _apply_chat_template(prompt, response, tokenizer)
    if response.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n"):
        # Wash away the system prompt manually
        len_sys_prompt = len("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n")
        response = response[len_sys_prompt:]

    if task_type == "action":
        if response.startswith("<think>"):
            prompt += "\n<think>"
            response = response[len("<think>"):]
        if response.count("</think>") >= 2:
            return
    else:
        prompt += "\n<think>    </think>"

    samples.append(
        {
            "prompt": prompt,
            "response": response,
            "task_type": task_type,
            "turn_index": turn,
            "player": player,
            "discounted_return": discounted_return,
            "step_reward": reward,
            **metadata,
        }
    )


def _collect_score_stats(log_dirs: Sequence[Path], *, limit: int = 0) -> dict:
    score_counts: Counter[float] = Counter()
    nonzero_scores = 0
    nonzero_rewards = 0
    total = 0
    total_steps = 0
    missing_scores = 0

    for _, _, record in _iter_episode_records(log_dirs, limit):
        total += 1
        steps = record.get("steps", []) or []
        total_steps += len(steps)

        summary = record.get("summary", {}) or {}
        score = _resolve_score(summary)
        if score is None:
            missing_scores += 1
        else:
            score_counts[score] += 1
            if score != 0:
                nonzero_scores += 1

        total_reward = summary.get("total_reward")
        if total_reward is not None:
            try:
                reward_val = float(total_reward)
            except (TypeError, ValueError):
                reward_val = 0
            if reward_val != 0:
                nonzero_rewards += 1

    return {
        "total": total,
        "nonzero_scores": nonzero_scores,
        "nonzero_rewards": nonzero_rewards,
        "missing_scores": missing_scores,
        "avg_traj_length": total_steps / total if total else 0,
        "score_counts": score_counts,
    }


def _plot_score_histogram(score_counts: Counter[float], *, output: Path | None) -> Path:
    if not score_counts:
        raise ValueError("No scores to plot.")

    scores = sorted(score_counts.items(), key=lambda kv: kv[0])
    labels = [str(score) for score, _ in scores]
    values = [count for _, count in scores]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xlabel("Final score")
    plt.ylabel("Trajectory count")
    plt.title("Hanabi trajectory score distribution")
    plt.tight_layout()

    if output is None:
        output_path = Path("hanabi_score_hist.png")
    else:
        output_path = output.with_suffix(".score_hist.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()
    return output_path


def build_samples(
    record: dict,
    min_score: float,
    include_summary: bool,
    tokenizer: PreTrainedTokenizerBase | None,
) -> Iterable[dict]:
    summary = record.get("summary", {}) or {}
    score = _resolve_score(summary)
    if score is None or score < min_score:
        return []

    base_id = summary.get("episode_id")
    if not base_id:
        base_id = f"{record.get('query_id', 'unknown')}-{record.get('episode_index', 0)}"

    steps = record.get("steps", []) or []
    metadata = {
        "trajectory_id": base_id,
        "final_score": score,
        "query_id": record.get("query_id"),
        "episode_index": record.get("episode_index"),
        "total_reward": summary.get("total_reward"),
    }

    samples: list[dict] = []
    for step in steps:
        reward = step.get("adjusted_reward")
        discounted_return = step.get("discounted_return")
        turn = step.get("turn")
        player = step.get("player")

        _append_sample(
            samples,
            step.get("action_prompt"),
            step.get("action_completion"),
            task_type="action",
            turn=turn,
            player=player,
            metadata=metadata,
            reward=reward,
            discounted_return=discounted_return,
            tokenizer=tokenizer,
        )

        _append_sample(
            samples,
            step.get("qgen_prompt"),
            step.get("qgen_response"),
            task_type="question_generation",
            turn=turn,
            player=player,
            metadata=metadata,
            reward=reward,
            discounted_return=discounted_return,
            tokenizer=tokenizer,
        )

        aprompts = step.get("agent_question_prompts") or []
        aanswers = step.get("agent_answers") or []
        for aprompt, ans in zip(aprompts, aanswers):
            _append_sample(
                samples,
                aprompt,
                ans,
                task_type="question_answering",
                turn=turn,
                player=player,
                metadata=metadata,
                reward=reward,
                discounted_return=discounted_return,
                tokenizer=tokenizer,
            )

        if include_summary:
            _append_sample(
                samples,
                step.get("summary_prompt"),
                step.get("agent_summary"),
                task_type="summary",
                turn=turn,
                player=player,
                metadata=metadata,
                reward=reward,
                discounted_return=discounted_return,
                tokenizer=tokenizer,
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        dest="log_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories containing Hanabi JSONL logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        help="Where to write the curated SFT data (JSONL). Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=15.0,
        help="Keep trajectories whose final score is at least this value.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If not zero, limit the number of jsonl logs to read from.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="",
        help=(
            "Optional tokenizer path used to wrap prompts and responses with the tokenizer's chat template."
        ),
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Exclude summary generation samples and keep only action prediction pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect logs without writing SFT data; report non-zero reward/score trajectories and plot score distribution."
        ),
    )
    args = parser.parse_args()
    log_dirs: list[Path] = args.log_dirs
    output: Path | None = args.output
    include_summary = not args.no_summary
    tokenizer: PreTrainedTokenizerBase | None = None

    if not log_dirs:
        parser.error("At least one --log-dir must be provided.")

    if not args.dry_run and output is None:
        parser.error("--output is required when not running with --dry-run.")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    if args.tokenizer_path:
        tokenizer = load_hf_tokenizer(args.tokenizer_path)

    kept_episodes = 0
    total_episodes = 0
    total_samples = 0

    if args.dry_run:
        print(f"Doing dry run...")
        stats = _collect_score_stats(log_dirs, limit=args.limit)
        score_counts = {str(k): v for k, v in sorted(stats["score_counts"].items(), key=lambda kv: kv[0])}
        plot_path = None
        if stats["score_counts"]:
            plot_path = _plot_score_histogram(stats["score_counts"], output=output)

        print(
            json.dumps(
                {
                    "log_dirs": [str(p) for p in log_dirs],
                    "min_score": args.min_score,
                    "include_summary": include_summary,
                    "limit": args.limit,
                    "avg_traj_length": stats["avg_traj_length"],
                    "total_trajectories": stats["total"],
                    "trajectories_with_nonzero_score": stats["nonzero_scores"],
                    "trajectories_with_nonzero_total_reward": stats["nonzero_rewards"],
                    "trajectories_missing_score": stats["missing_scores"],
                    "score_counts": score_counts,
                    "histogram_path": str(plot_path) if plot_path else None,
                },
                ensure_ascii=False,
            )
        )
        return

    with output.open("w", encoding="utf-8") as sink:
        for file_path, line_no, record in _iter_episode_records(log_dirs, limit=args.limit):
            total_episodes += 1
            samples = build_samples(record, args.min_score, include_summary, tokenizer)
            if not samples:
                continue
            kept_episodes += 1
            for sample in samples:
                sink.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_samples += 1

    print(
        json.dumps(
            {
                "log_dirs": [str(p) for p in log_dirs],
                "output": str(output),
                "min_score": args.min_score,
                "include_summary": include_summary,
                "total_episodes": total_episodes,
                "kept_episodes": kept_episodes,
                "total_samples": total_samples,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()