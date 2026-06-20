from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .rules import default_rules, sample_examples


def build_prompt(examples: list[dict[str, object]]) -> str:
    lines = [
        "Infer the hidden rule that maps strings to valid or invalid.",
        "Examples:",
    ]
    for item in examples:
        label = "valid" if item["y"] else "invalid"
        lines.append(f"- {item['x']}: {label}")
    lines.append("Hidden rule:")
    return "\n".join(lines)


def generate_dataset(
    out: Path,
    num_rule_instances: int,
    examples_per_rule: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    rules = default_rules()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i in range(num_rule_instances):
            rule = rules[i % len(rules)]
            examples = sample_examples(rule, rng, examples_per_rule)
            train_examples = examples[: max(4, int(examples_per_rule * 0.75))]
            eval_examples = examples[len(train_examples) :]
            row = {
                "rule_id": rule.rule_id,
                "rule_description": rule.description,
                "examples": train_examples,
                "eval_examples": eval_examples,
                "prompt": build_prompt(train_examples),
                "answer": rule.description,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--rules", type=int, default=200, help="Number of rule episodes to emit.")
    parser.add_argument("--examples-per-rule", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.out, args.rules, args.examples_per_rule, args.seed)
    print(f"Wrote {args.rules} rows to {args.out}")


if __name__ == "__main__":
    main()

