from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_tutor_common import (
    TeacherEvaluator,
    apply_cli_overrides,
    load_config,
    parse_args,
    write_report,
)


from hanabi_env import HanabiEnv  # noqa: E402


def build_env(env_kwargs: dict[str, Any]) -> HanabiEnv:
    return HanabiEnv(**env_kwargs)


def build_prompt(env: HanabiEnv, obs: str, guide: str, teacher_advice: str, memory: str) -> str:
    return (
        f"{obs}\n\n"
        f"{memory}\n\n"
        f"Teacher guidance:\n{teacher_advice or 'No teacher guidance. Decide from your own observation.'}\n\n"
        f"{guide}\n"
        "Choose exactly one legal Hanabi action."
    ).strip()


def score(env: HanabiEnv, _last_reward: Any) -> float:
    return float(env.get_stats().get("score", 0.0))


async def main() -> None:
    args = parse_args(str(Path(__file__).with_name("teacher_eval_config.yaml")))
    config = apply_cli_overrides(load_config(args.config), args)
    evaluator = TeacherEvaluator(
        env_factory=build_env,
        prompt_builder=build_prompt,
        score_getter=score,
        config=config,
    )
    report = await evaluator.run_pair()
    output_path = config.get("output_path", "examples/hanabi-tutor/outputs/teacher_eval.json")
    write_report(report, output_path)
    print(f"Wrote teacher evaluation report to {output_path}")
    print(f"Average score improvement: {report['improvement']['avg_score']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
