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


from werewolf_env import WerewolfEnv  # noqa: E402


def build_env(env_kwargs: dict[str, Any]) -> WerewolfEnv:
    return WerewolfEnv(**env_kwargs)


def build_prompt(env: WerewolfEnv, obs: str, guide: str, teacher_advice: str, memory: str) -> str:
    return (
        f"{obs}\n\n"
        f"{memory}\n\n"
        f"{guide}\n"
        f"Teacher guidance:\n{teacher_advice or 'No teacher guidance. Decide from your own observation.'}\n\n"
        "Choose exactly one valid action for the current phase. "
        "Do not answer with skip when a concrete vote, target, or discussion statement is useful. "
        "Keep the final <answer> concise."
    ).strip()


def score(env: WerewolfEnv, _last_reward: Any) -> float:
    stats = env.get_stats()
    return float(
        20.0 * stats.get("vill_wins", 0)
        - 20.0 * stats.get("were_wins", 0)
        + stats.get("villager_correct_votes", 0)
        - stats.get("villager_wrong_votes", 0)
        + stats.get("witch_correct_heals", 0)
        + stats.get("witch_correct_poisons", 0)
        + stats.get("hunter_correct_shots", 0)
    )


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
    output_path = config.get("output_path", "examples/werewolf-tutor/outputs/teacher_eval.json")
    write_report(report, output_path)
    print(f"Wrote teacher evaluation report to {output_path}")
    print(f"Average score improvement: {report['improvement']['avg_score']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
