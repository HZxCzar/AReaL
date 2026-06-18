from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict[str, Any], key: str) -> tuple[Any, Any, Any]:
    baseline = report.get("baseline", {})
    teacher = report.get("teacher", {})
    improvement = report.get("improvement", {})
    return baseline.get(key), teacher.get(key), improvement.get(key)


def _row(game: str, report: dict[str, Any] | None) -> str:
    if report is None:
        return f"| {game} | missing | - | - | - |"
    if "avg_score" in report.get("improvement", {}):
        b, t, d = _metric(report, "avg_score")
        return f"| {game} | avg_score | {b:.4g} | {t:.4g} | {d:+.4g} |"
    b, t, d = _metric(report, "mean_reward")
    acc_b, acc_t, acc_d = _metric(report, "mean_heldout_accuracy")
    return (
        f"| {game} | mean_reward / accuracy | {b:.4g} / {acc_b:.4g} | "
        f"{t:.4g} / {acc_t:.4g} | {d:+.4g} / {acc_d:+.4g} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/tutoring-s0.6b-t2b"))
    parser.add_argument("--output", type=Path, default=Path("results/teacher_model_training_report.md"))
    args = parser.parse_args()

    reports = {
        "werewolf": _load(args.results_dir / "werewolf_best.json"),
        "hidden-rule": _load(args.results_dir / "hidden_rule_best.json"),
        "hanabi": _load(args.results_dir / "hanabi_best.json"),
    }

    lines = [
        "# Teacher Tutor Training Report",
        "",
        "## Added training entrypoints",
        "",
        "- `examples/werewolf-tutor/train_teacher_grpo.py` with `train_teacher_grpo.yaml`",
        "- `examples/hidden-rule-tutor/train_teacher_grpo.py` with `train_teacher_grpo.yaml`",
        "- `examples/hanabi-tutor/train_teacher_grpo.py` with `train_teacher_grpo.yaml`",
        "",
        "The GRPO teacher reward combines student-output consistency, explicit reasoning, leakage penalties, and final task reward. "
        "The public default teacher is `Qwen/Qwen3-1.7B` because `Qwen/Qwen3.5-2B-Instruct` was not resolvable locally; override `actor.path=...` if you have a private 2B checkpoint.",
        "",
        "## Current Evaluation Snapshot",
        "",
        f"Source directory: `{args.results_dir}`",
        "",
        "| game | metric | baseline | teacher | delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(_row(game, report) for game, report in reports.items())
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "Start a student OpenAI-compatible server on `127.0.0.1:30001`, then run one of:",
            "",
            "```bash",
            "conda activate areal",
            "python examples/werewolf-tutor/train_teacher_grpo.py --config examples/werewolf-tutor/train_teacher_grpo.yaml",
            "python examples/hidden-rule-tutor/train_teacher_grpo.py --config examples/hidden-rule-tutor/train_teacher_grpo.yaml",
            "python examples/hanabi-tutor/train_teacher_grpo.py --config examples/hanabi-tutor/train_teacher_grpo.yaml",
            "```",
            "",
            "Evaluate the trained checkpoint by serving it on the teacher port used by each `teacher_eval_config.yaml`, then run the existing `evaluate_teacher.py` script for the matching game.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
