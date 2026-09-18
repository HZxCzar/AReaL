"""Use the shared matrix runner with the strict, turn-local student setting."""

import json
from pathlib import Path

from examples.tutor.scripts.student_sim_eval import runner as shared

from .prompts import student_instruction

PACKAGE = Path(__file__).resolve().parent
TEACHER_STRATEGIES = json.loads((PACKAGE / "teacher_strategies.json").read_text())
TEACHER_REMINDER = "\n\nNo matter what, strictly maintain this teaching method throughout the entire conversation."


def compile_cell(config: dict, row: str, strategy: str, env: dict) -> tuple[dict, dict]:
    """Only simulation instructions differ; reuse preparation, dialogue and tests."""
    if config["method"] != "prompt-only":
        raise ValueError("prompted_strict requires method=prompt-only")
    if row == "none" or strategy not in TEACHER_STRATEGIES:
        raise ValueError("This setting evaluates the six named preferences/strategies")
    experiment, protocol = shared.compile_cell(config, row, strategy, env)
    experiment["run_name"] = f"prompted-strict-{row}-{strategy}"
    # This existing protocol field stores the text; our evaluator applies it ONLY
    # to the current tutoring user message, not to any system or test prompt.
    protocol["student_system_prompt"] = student_instruction(
        shared.REPO / protocol["personality"]["prompts_path"],
        row,
        shared.REPO / protocol["personality"]["complaints_path"],
    )
    protocol["teacher_system_prompt"] = TEACHER_STRATEGIES[strategy] + TEACHER_REMINDER
    return experiment, protocol


def main():
    shared.main(
        config_directory=PACKAGE,
        compile_cell_fn=compile_cell,
        cell_module="examples.tutor.scripts.student_sim_eval.prompted_strict.cell",
    )


if __name__ == "__main__":
    main()
