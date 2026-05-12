from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve()))

from game_tutor_common import (  # noqa: E402
    LLMConfig,
    OpenAIChatClient,
    apply_cli_overrides,
    load_config,
    normalize_messages,
    parse_args,
    write_report,
)
from hidden_rule_game.env import HiddenRuleEnv  # noqa: E402
from hidden_rule_game.rules import Rule  # noqa: E402
from hidden_rule_game.students import MemoryStudent  # noqa: E402


class OpenAITeacher:
    def __init__(self, cfg: dict[str, Any], *, icl_turns: int = 5):
        self.client = OpenAIChatClient(LLMConfig(**cfg))
        self.icl_turns = max(1, int(icl_turns))
        self.compressed_memory = ""
        self.calls = 0

    def answer(
        self,
        rule: Rule,
        student_message: str,
        transcript: list[dict[str, str]],
        labeled_examples: list[dict[str, object]],
    ) -> str:
        self.calls += 1
        if self.calls % self.icl_turns == 0 and transcript:
            self._compress(transcript)
        examples_text = "\n".join(
            f"{item['x']} -> {'valid' if item['y'] else 'invalid'}"
            for item in labeled_examples
        )
        recent = "\n".join(
            f"Student: {turn['student']}\nTutor: {turn['tutor']}"
            for turn in transcript[-self.icl_turns :]
        )
        prompt = (
            "You are being evaluated as a hidden-rule tutor.\n"
            "Goal: help the student infer the Boolean string rule with as few examples as possible.\n"
            "Do not reveal the private rule verbatim. Provide useful labeled examples and short strategic hints.\n\n"
            f"Private rule, not to reveal verbatim: {rule.description}\n\n"
            f"Compressed memory:\n{self.compressed_memory or 'None'}\n\n"
            f"Recent dialogue:\n{recent or 'None'}\n\n"
            f"Student message: {student_message}\n\n"
            "Fresh labeled examples you may show exactly:\n"
            f"{examples_text}\n\n"
            "Tutor reply:"
        )
        reply = self.client._complete_blocking(
            normalize_messages("You are a concise teacher.", prompt),
            max_tokens=512,
        ).strip()
        if not any(str(item["x"]) in reply for item in labeled_examples):
            reply = "Here are labeled examples:\n" + examples_text + "\nLook for the shared pattern."
        return reply

    def _compress(self, transcript: list[dict[str, str]]) -> None:
        recent = "\n".join(
            f"Student: {turn['student']}\nTutor: {turn['tutor']}"
            for turn in transcript[-self.icl_turns :]
        )
        prompt = (
            "Compress this hidden-rule tutoring dialogue for future tutoring decisions. "
            "Keep the student's hypotheses, examples already shown, and likely misconceptions.\n\n"
            f"Previous compressed memory:\n{self.compressed_memory or 'None'}\n\n"
            f"Recent dialogue:\n{recent}"
        )
        self.compressed_memory = self.client._complete_blocking(
            normalize_messages("You compress tutoring state.", prompt),
            max_tokens=512,
        ).strip()


def run_condition(config: dict[str, Any], *, use_teacher: bool) -> dict[str, Any]:
    icl_cfg = config.get("icl_simulation") or {}
    icl_turns = int(icl_cfg.get("turns", 5))
    teacher = (
        OpenAITeacher(config["teacher_model"], icl_turns=icl_turns)
        if use_teacher
        else None
    )
    env = HiddenRuleEnv(
        seed=int(config.get("seed", 7)),
        examples_per_episode=int(config.get("examples_per_episode", 48)),
        teacher=teacher,
    )
    rounds = int(config.get("rounds", 8))
    episodes = int(config.get("episodes", 50))
    threshold = float(config.get("success_threshold", 0.95))
    rows = []
    rewards = []
    accuracies = []
    successes = []
    examples_used = []
    turns_taken = []
    for idx in range(episodes):
        student = MemoryStudent()
        result = env.run_episode(student, rounds=rounds)
        success = result.heldout_accuracy >= threshold
        row = {
            "episode": idx,
            "rule_id": result.rule_id,
            "true_rule": result.true_rule,
            "guessed_rule": result.guessed_rule,
            "heldout_accuracy": result.heldout_accuracy,
            "reward": result.reward,
            "success": success,
            "turns_taken": result.turns_taken,
            "examples_used": result.examples_used,
            "rule_matched": result.rule_matched,
            "transcript": result.transcript,
        }
        rows.append(row)
        rewards.append(result.reward)
        accuracies.append(result.heldout_accuracy)
        successes.append(float(success))
        examples_used.append(result.examples_used)
        turns_taken.append(result.turns_taken)
    return {
        "episodes": episodes,
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "mean_heldout_accuracy": sum(accuracies) / max(1, len(accuracies)),
        "success_rate": sum(successes) / max(1, len(successes)),
        "avg_examples_used": sum(examples_used) / max(1, len(examples_used)),
        "avg_turns_taken": sum(turns_taken) / max(1, len(turns_taken)),
        "traces": rows,
    }


def main() -> None:
    args = parse_args(str(Path(__file__).with_name("teacher_eval_config.yaml")))
    config = apply_cli_overrides(load_config(args.config), args)
    if config.get("num_turns") is not None:
        config["episodes"] = max(1, int(config["num_turns"]) // max(1, int(config.get("rounds", 8))))
    baseline = run_condition(config, use_teacher=False)
    teacher = run_condition(config, use_teacher=True)
    report = {
        "config": config,
        "baseline": baseline,
        "teacher": teacher,
        "improvement": {
            "mean_reward": teacher["mean_reward"] - baseline["mean_reward"],
            "mean_heldout_accuracy": teacher["mean_heldout_accuracy"] - baseline["mean_heldout_accuracy"],
            "success_rate": teacher["success_rate"] - baseline["success_rate"],
            "avg_examples_used": teacher["avg_examples_used"] - baseline["avg_examples_used"],
            "avg_turns_taken": teacher["avg_turns_taken"] - baseline["avg_turns_taken"],
        },
    }
    output_path = config.get("output_path", "examples/hidden-rule-tutor/outputs/teacher_eval.json")
    write_report(report, output_path)
    print(json.dumps(report["improvement"], indent=2))
    print(f"Wrote teacher evaluation report to {output_path}")


if __name__ == "__main__":
    main()
