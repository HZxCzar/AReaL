from __future__ import annotations

import argparse
import json

from .env import HiddenRuleEnv
from .inference_engine import VLLMEngine
from .students import LLMStudent, MemoryStudent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", choices=["memory", "llm"], default="memory")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--lora-name", type=str, default="student_lora")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--jsonl-log", type=str, default=None)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    return parser.parse_args()


def build_student(args: argparse.Namespace) -> object:
    if args.student == "memory":
        return MemoryStudent()
    if not args.model:
        raise ValueError("--model is required when --student llm")
    return LLMStudent(
        VLLMEngine(
            args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            lora_path=args.lora_path,
            lora_name=args.lora_name,
        )
    )


def main() -> None:
    args = parse_args()
    env = HiddenRuleEnv(seed=args.seed)
    student = build_student(args)
    log_f = open(args.jsonl_log, "w", encoding="utf-8") if args.jsonl_log else None

    try:
        rewards = []
        for idx in range(args.episodes):
            result = env.run_episode(student, rounds=args.rounds)
            rewards.append(result.reward)
            row = {
                "episode": idx,
                "rule_id": result.rule_id,
                "true_rule": result.true_rule,
                "guessed_rule": result.guessed_rule,
                "heldout_accuracy": result.heldout_accuracy,
                "reward": result.reward,
                "turns_taken": result.turns_taken,
                "examples_used": result.examples_used,
                "rule_matched": result.rule_matched,
                "success": result.heldout_accuracy >= args.success_threshold,
                "transcript": result.transcript,
            }
            print(json.dumps(row, ensure_ascii=False, indent=2))
            if log_f:
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Mean reward: {sum(rewards) / len(rewards):.3f}")
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    main()
