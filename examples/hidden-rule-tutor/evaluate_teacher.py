from __future__ import annotations

import json
import re
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
from hidden_rule_game.rules import Rule, count_vowels, has_double_letter  # noqa: E402
from hidden_rule_game.students import MemoryStudent, extract_examples  # noqa: E402


def _clip_text(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"final_rule:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _strip_thinking(text: str) -> str:
    cleaned = THINK_RE.sub("", text).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _as_tentative_reply(text: str) -> str:
    cleaned = _strip_thinking(text)
    match = FINAL_RE.search(cleaned)
    hypothesis = (match.group(1) if match else cleaned).strip()
    hypothesis = hypothesis.splitlines()[0].strip("` ").strip()
    if not hypothesis:
        hypothesis = "a simple property of the string such as length, vowels, repeated letters, or a special character"
    return (
        f"Tentative hypothesis: {hypothesis}\n"
        "I need a contrastive labeled mini-batch that tests this guess against at least two alternatives, "
        "especially examples matched for length while changing letters, vowel pattern, repetition, or order."
    )


def _needs_student_fallback(reply: str, example_count: int) -> bool:
    lowered = reply.lower()
    if example_count == 0 and "tentative hypothesis:" not in lowered:
        return True
    return "tell me the rule" in lowered or "let me know what the rule is" in lowered


def _format_transcript(transcript: list[dict[str, str]]) -> str:
    return "\n".join(
        f"Student: {_strip_thinking(turn['student'])}\nTutor: {_strip_thinking(turn['tutor'])}"
        for turn in transcript
    )


class OpenAIStudent:
    def __init__(self, cfg: dict[str, Any]):
        self.client = OpenAIChatClient(LLMConfig(**cfg))

    def reset(self) -> None:
        pass

    def act(self, observation: str, transcript: list[dict[str, str]]) -> str:
        history = _format_transcript(transcript)
        example_count = len(extract_examples(transcript))
        final_rule_instruction = (
            "You have enough evidence to use `FINAL_RULE:` only if you are genuinely confident."
            if example_count >= 12
            else "Do not use `FINAL_RULE:` yet; make a tentative guess and request discriminating labeled tests."
        )
        prompt = (
            "You are a student trying to infer a hidden Boolean rule over strings.\n"
            "Do your own reasoning. The tutor is not allowed to solve the rule for you.\n"
            "Use systematic rule induction: maintain 2-4 competing simple hypotheses, compare them "
            "against the labeled examples, and prefer the simplest hypothesis that explains both "
            "valid and invalid cases. Pay attention to length, vowels/consonants, repeated adjacent "
            "letters, special letters, first/last letters, letter order, and letter position, but do "
            "not assume any one of these must be the rule.\n"
            "If you are uncertain, state your current tentative hypothesis, one or two alternatives, "
            "which examples support or contradict them, and ask for specific additional labeled tests "
            "that would distinguish them. Do not merely ask the tutor to share patterns.\n"
            f"{final_rule_instruction} Keep the reply concise.\n\n"
            f"Parsed labeled examples so far: {example_count}\n\n"
            f"Dialogue so far:\n{_clip_text(history) or 'None'}\n\n"
            f"Latest tutor message:\n{observation}\n\n"
        )
        reply = self.client._complete_blocking(
            normalize_messages("You are a careful rule-induction student.", prompt),
            max_tokens=self.client.cfg.max_tokens,
        )
        reply = _strip_thinking(reply)
        if example_count < 12 and FINAL_RE.search(reply):
            reply = _as_tentative_reply(reply)
        if _needs_student_fallback(reply, example_count):
            reply = (
                "Tentative hypothesis: validity depends on a simple surface feature such as vowel count, "
                "string length, repeated letters, or a special character.\n"
                "Please show a small labeled batch that includes both valid and invalid strings if possible, "
                "so I can rule out at least one of those alternatives."
            )
        return _clip_text(reply, 2000)

    def observe(self, message: str, reply: str) -> None:
        pass

    def guess_rule(self, transcript: list[dict[str, str]]) -> str:
        prompt = (
            "Infer the hidden Boolean string rule from this tutoring dialogue:\n"
            f"{_clip_text(_format_transcript(transcript))}"
            "Choose one simple, testable property that best explains both valid and invalid examples. "
            "Avoid compound rules with multiple conditions unless each condition is directly supported. "
            "Use precise measurable wording: say adjacent repeated letters for doubles, first and last "
            "letters for endpoint rules, third letter for position rules, multiple of three for length "
            "rules, alphabetical/non-decreasing for order rules, and more consonants than vowels for "
            "vowel/consonant balance rules when those ideas fit the evidence. "
            "Reply with one sentence starting with `A string is valid if`.\n\n"
        )
        reply = self.client._complete_blocking(
            normalize_messages("You infer concise hidden string rules.", prompt),
            max_tokens=self.client.cfg.max_tokens,
        )
        reply = _strip_thinking(reply)
        match = FINAL_RE.search(reply)
        if match:
            reply = match.group(1).strip()
        return reply.strip()

    def predict_label(
        self,
        text: str,
        transcript: list[dict[str, str]],
        guessed_rule: str | None = None,
    ) -> bool:
        return _predict_from_rule_text(text, guessed_rule or self.guess_rule(transcript))


class OpenAITeacher:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        icl_turns: int = 5,
        knows_rule: bool = False,
    ):
        self.client = OpenAIChatClient(LLMConfig(**cfg))
        self.icl_turns = max(1, int(icl_turns))
        self.knows_rule = knows_rule
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
        if self.knows_rule:
            role_context = (
                "You know the hidden Boolean string rule, but must not reveal it verbatim.\n"
                f"Private rule: {rule.description}\n"
            )
        else:
            role_context = (
                "You do not know the hidden Boolean string rule. You and the student only know the "
                "labeled examples and the dialogue. Give useful process guidance based on the "
                "student's latest hypothesis and the current evidence. First show the fresh labeled "
                "examples exactly. Then help the student audit their own hypothesis: name one feature "
                "dimension to compare, point to specific examples that are useful contrasts, and suggest "
                "one discriminating next test or request. You may mention broad feature dimensions such "
                "as length, vowels, repeated letters, special letters, first/last letters, order, or "
                "position, but do not pick a final one for the student. Do not solve the task for the "
                "student, do not state a final rule, and do not give a polished hypothesis for them to copy. "
                "Never say the student's guess is correct or incorrect, and never write phrases like "
                "`the rule is`, `you identified`, `I guess`, or `FINAL_RULE`.\n"
            )
        prompt = (
            "You are a collaborative hidden-rule guess game tutor.\n"
            "Goal: help the student infer the Boolean string rule with as few examples as possible.\n"
            "Keep replies concise. The student must make the guesses; your job is to scaffold their "
            "reasoning without doing the final inference for them.\n\n"
            f"{role_context}\n"
            f"Compressed memory:\n{_clip_text(self.compressed_memory, 2000) or 'None'}\n\n"
            f"Recent dialogue:\n{_clip_text(recent) or 'None'}\n\n"
            f"Student message:\n{_clip_text(_strip_thinking(student_message), 2000)}\n\n"
            "Fresh labeled examples you may show exactly:\n"
            f"{examples_text}\n\n"
            "Tutor reply. Include the fresh labeled examples, then give guidance without naming a final rule:"
        )
        reply = self.client._complete_blocking(
            normalize_messages("You are a concise teacher.", prompt),
            max_tokens=self.client.cfg.max_tokens,
        )
        reply = _strip_thinking(reply)
        if self._needs_safe_fallback(reply, labeled_examples):
            reply = self._safe_evidence_reply(examples_text)
        return reply

    def _compress(self, transcript: list[dict[str, str]]) -> None:
        recent = "\n".join(
            f"Student: {turn['student']}\nTutor: {turn['tutor']}"
            for turn in transcript[-self.icl_turns :]
        )
        prompt = (
            "Compress this hidden-rule tutoring dialogue for future tutoring decisions. "
            "Keep the student's hypotheses, examples already shown, and likely misconceptions.\n\n"
            f"Previous compressed memory:\n{_clip_text(self.compressed_memory, 2000) or 'None'}\n\n"
            f"Recent dialogue:\n{_clip_text(recent, 2000)}"
        )
        self.compressed_memory = self.client._complete_blocking(
            normalize_messages("You compress your memory.", prompt),
            max_tokens=min(512, self.client.cfg.max_tokens),
        ).strip()
        self.compressed_memory = _strip_thinking(self.compressed_memory)

    def _needs_safe_fallback(
        self,
        reply: str,
        labeled_examples: list[dict[str, object]],
    ) -> bool:
        lowered = reply.lower()
        forbidden = [
            "final_rule",
            "the rule is",
            "a string is valid if",
            "you are correct",
            "you're correct",
            "correctly identified",
            "you identified",
            "your rule is",
            "valid password",
        ]
        return (
            not any(str(item["x"]) in reply for item in labeled_examples)
            or any(token in lowered for token in forbidden)
        )

    @staticmethod
    def _safe_evidence_reply(examples_text: str) -> str:
        return (
            "Here are labeled examples:\n"
            f"{examples_text}\n"
            "Use these labels to audit your current hypothesis against alternatives. Pick two examples "
            "that are similar in length or letters but have different labels, compare one feature "
            "dimension at a time, and ask for a targeted test that changes only one suspected feature."
        )


def _predict_from_rule_text(text: str, rule: str) -> bool:
    lowered = rule.lower()
    if "even" in lowered and "vowel" in lowered:
        return count_vowels(text) % 2 == 0
    if "first" in lowered and "last" in lowered:
        return len(text) > 0 and text[0] == text[-1]
    if "same letter twice" in lowered or "double" in lowered or "repeated letter" in lowered:
        return has_double_letter(text)
    if "multiple of three" in lowered or "divisible by three" in lowered:
        return len(text) % 3 == 0
    if "z or q" in lowered or "z" in lowered and "q" in lowered:
        return "z" in text or "q" in text
    if "more consonants" in lowered:
        return (len(text) - count_vowels(text)) > count_vowels(text)
    if "non-decreasing" in lowered or "alphabetical order" in lowered or "sorted" in lowered:
        return all(a <= b for a, b in zip(text, text[1:]))
    if "third letter" in lowered and "vowel" in lowered:
        return len(text) >= 3 and text[2] in set("aeiou")
    return False


def run_condition(config: dict[str, Any], *, use_teacher: bool) -> dict[str, Any]:
    icl_cfg = config.get("icl_simulation") or {}
    icl_turns = int(icl_cfg.get("turns", 5))
    teacher = (
        OpenAITeacher(
            config["teacher_model"],
            icl_turns=icl_turns,
            knows_rule=bool(config.get("tutor_knows_rule", False)),
        )
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
        student = (
            OpenAIStudent(config["student_model"])
            if config.get("student_model")
            else MemoryStudent()
        )
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
