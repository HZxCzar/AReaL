"""Compose the shared evaluator with the approved short teacher strategies."""

import json
from pathlib import Path

from examples.tutor.scripts.student_sim_eval import runner as shared

PACKAGE = Path(__file__).resolve().parent
STRATEGIES = json.loads((PACKAGE / "teacher_strategies.json").read_text())
REMINDER = "\n\nNo matter what, strictly maintain this teaching method throughout the entire conversation."


def compile_cell(config: dict, row: str, strategy: str, env: dict) -> tuple[dict, dict]:
    """Change student identity only; no preference prompt or external gate."""
    if config["method"] != "different-models" or strategy not in STRATEGIES:
        raise ValueError(
            "diverse_models requires different-models and a named strategy"
        )
    experiment, protocol = shared.compile_cell(config, row, strategy, env)
    experiment["run_name"] = f"diverse-models-{row}-{strategy}"
    protocol["teacher_system_prompt"] = STRATEGIES[strategy] + REMINDER
    assert protocol["student_system_prompt"] == ""
    assert protocol["personality"]["gate_sample_rate"] == 0
    return experiment, protocol


def main():
    shared.main(config_directory=PACKAGE, compile_cell_fn=compile_cell)


if __name__ == "__main__":
    main()
