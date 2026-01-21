#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed werewolf analyzer with OpenAI-only judging for villager-side reasoning.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Iterable, Iterator
import urllib.request

ROLE_ALIASES = {
    "wolf": "werewolf",
    "seer": "foreseer",
}

VILLAGER_SIDE_ROLES = {"villager", "foreseer", "witch", "hunter"}

REASONING_PATTERNS = {
    "concealment": "Intentionally hiding or disguising identity/intent, avoiding revealing secret information to werewolves, contributing to villager side winning.",
    "cooperation": "Successfully coordinating, supporting allies, or collaboration actions (e.g., healing villagers/shooting werewolves), beneficial to villager side.",
    "bluffing": "Misleading, false claims, deception, or deliberate misinformation that truly contribute to villager side winning.",
    "sacrificing": "Intentional sacrifice or trading a player for strategic gain.",
    "pressure": "Applying pressure, forcing choices, cornering or pushing votes.",
    "consensus_building": "Building agreement, bandwagoning, or forming a voting majority.",
    "role_revealing": "Explicitly signaling or discussing roles/claim, potentially leaking role information to werewolves.",
}



@dataclass
class SentenceJudgeResult:
    file: str
    episode_index: int
    turn: int
    agent: str
    role: str
    sentence: str
    concealment: bool
    cooperation: bool
    bluffing: bool
    sacrificing: bool
    pressure: bool
    consensus_building: bool
    role_signal: bool


def iter_log_paths(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob(pattern))


def iter_records(path: Path) -> Iterator[dict]:
    decoder = JSONDecoder()
    buffer = ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            buffer += line
            while True:
                payload = buffer.lstrip()
                if not payload:
                    buffer = ""
                    break
                try:
                    obj, idx = decoder.raw_decode(payload)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = payload[idx:]
        if buffer.strip():
            raise ValueError(f"Unparsed JSON leftover in {path}")


def normalize_role(role: str | None) -> str:
    if not role:
        return "unknown"
    role = role.lower()
    return ROLE_ALIASES.get(role, role)


def normalize_sentence(text: str) -> str | None:
    cleaned = " ".join(text.split()).strip()
    if len(cleaned) < 5:
        return None
    return cleaned


def build_judge_prompt(sentences: list[str]) -> str:
    pattern_lines = "\n".join(
        f"- {name}: {definition}" for name, definition in REASONING_PATTERNS.items()
    )
    # items = "\n".join(f"{sentence}" for idx, sentence in enumerate(sentences))
    items = sentences[0]
    return (
        "You are judging villager-side reasoning sentences from a Werewolf game.\n"
        "For each sentence, decide whether it shows each reasoning pattern below.\n\n"
        "Definitions:\n"
        f"{pattern_lines}\n\n"
        "Return strict JSON. Each object must contain:\n"
        "concealment, cooperation, bluffing, sacrificing, "
        "pressure, consensus_building, role_revealing (each yes/no).\n\n"
        f"Sentence:\n{items}\n"
    )


def run_openai_judge(
    prompt: str,
    model_name: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a careful evaluator."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    decoded = json.loads(raw)
    return decoded["choices"][0]["message"]["content"]


def parse_judge_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        print(f"\n\n Error Matching: {text} \n\n", flush=True)
        # raise ValueError("No JSON array found in judge response.")
        return {}
    payload = match.group(0)

    return json.loads(payload)


def extract_villager_sentences(record: dict, max_sentences: int | None) -> list[tuple[int, str, str, str]]:
    sentences: list[tuple[int, str, str, str]] = []
    for step in record.get("steps", []):
        role = normalize_role(step.get("role", "unknown"))
        if role not in VILLAGER_SIDE_ROLES:
            continue
        turn = int(step.get("turn", 0))
        agent = str(step.get("agent", "unknown"))
        thought_text = normalize_sentence(str(step.get("agent_thought", "")))
        if thought_text:
            sentences.append((turn, agent, role, thought_text))
            if max_sentences and len(sentences) >= max_sentences:
                return sentences
        qa_pairs = zip(step.get("self_questions", []) or [], step.get("agent_answers", []) or [])
        for question, answer in qa_pairs:
            qa_sentence = normalize_sentence(f"Q: {question} A: {answer}")
            if qa_sentence:
                sentences.append((turn, agent, role, qa_sentence))
                if max_sentences and len(sentences) >= max_sentences:
                    return sentences
    return sentences


def judge_sentences(
    file_name: str,
    episode_index: int,
    sentences: list[tuple[int, str, str, str]],
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    batch_size: int,
) -> list[SentenceJudgeResult]:
    _ = batch_size
    results: list[SentenceJudgeResult] = []
    for item in sentences:
        turn, agent, role, sentence = item
        prompt = build_judge_prompt([sentence])

        print(f"Running judge with prompt len: {len(prompt)}", flush=True)

        response = run_openai_judge(prompt, model, base_url, api_key, max_tokens, temperature)
        # parsed = parse_judge_json(response)
        # verdict = parsed[0] if parsed else {}
        verdict = parse_judge_json(response)
        results.append(
            SentenceJudgeResult(
                file=file_name,
                episode_index=episode_index,
                turn=turn,
                agent=agent,
                role=role,
                sentence=sentence,
                concealment=str(verdict.get("concealment", "no")).lower() == "yes",
                cooperation=str(verdict.get("cooperation", "no")).lower() == "yes",
                bluffing=str(verdict.get("bluffing", "no")).lower() == "yes",
                sacrificing=str(verdict.get("sacrificing", "no")).lower() == "yes",
                pressure=str(verdict.get("pressure", "no")).lower() == "yes",
                consensus_building=str(verdict.get("consensus_building", "no")).lower() == "yes",
                role_signal=str(verdict.get("role_signal", "no")).lower() == "yes",
            )
        )
        time.sleep(0.1)
    return results


def write_sentence_csv(path: Path, results: list[SentenceJudgeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "episode_index",
                "turn",
                "agent",
                "role",
                "sentence",
                "concealment",
                "cooperation",
                "bluffing",
                "sacrificing",
                "pressure",
                "consensus_building",
                "role_signal",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.file,
                    item.episode_index,
                    item.turn,
                    item.agent,
                    item.role,
                    item.sentence,
                    str(item.concealment),
                    str(item.cooperation),
                    str(item.bluffing),
                    str(item.sacrificing),
                    str(item.pressure),
                    str(item.consensus_building),
                    str(item.role_signal),
                ]
            )


def write_summary(path: Path, results: list[SentenceJudgeResult]) -> None:
    total_sentences = len(results)
    counts = Counter()
    for item in results:
        counts["concealment"] += int(item.concealment)
        counts["cooperation"] += int(item.cooperation)
        counts["bluffing"] += int(item.bluffing)
        counts["sacrificing"] += int(item.sacrificing)
        counts["pressure"] += int(item.pressure)
        counts["consensus_building"] += int(item.consensus_building)
        counts["role_signal"] += int(item.role_signal)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Detailed Werewolf Reasoning Report\n\n")
        handle.write(f"Total villager-side sentences judged: {total_sentences}\n\n")
        for name in REASONING_PATTERNS:
            count = counts[name]
            ratio = (count / total_sentences) if total_sentences else 0.0
            handle.write(f"- {name}: {count} ({ratio:.3f})\n")


"""
python examples/werewolf/detailed_analyser.py --root /storage/openpsi/experiments/logs/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-step4/generated/0 \
    --output-dir analysis/detailed/4-2
"""

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed villager-side reasoning analyzer (OpenAI judge only)."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Root directory or file.")
    parser.add_argument("--glob", dest="glob_pattern", default="*.jsonl", help="Glob pattern to match logs.")
    parser.add_argument("--output-dir", type=Path, default=Path("werewolf_detailed"))
    parser.add_argument("--judge-model", default="qwen-turbo")
    parser.add_argument("--judge-base-url", default="https://matrixllm.alipay.com/")
    parser.add_argument("--judge-api-key", default="sk-43dd5f664179406d92fec42a9364f8a5")
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--judge-temperature", type=float, default=0.3)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[SentenceJudgeResult] = []
    all_files = iter_log_paths(args.root, args.glob_pattern)

    print(f"Found {len(all_files)} log files.", flush=True)

    for idx in range(len(all_files)):
        path = all_files[idx]
        records = list(iter_records(path))
        print(f"Evaluating {idx}/{len(all_files)} log file: {len(records)} records.", flush=True)
        for record in records:
            episode_index = int(record.get("episode_index", 0))
            sentences = extract_villager_sentences(record, args.max_sentences)
            if not sentences:
                continue
            all_results.extend(
                judge_sentences(
                    path.name,
                    episode_index,
                    sentences,
                    args.judge_model,
                    args.judge_base_url,
                    args.judge_api_key,
                    args.judge_max_tokens,
                    args.judge_temperature,
                    args.batch_size,
                )
            )

    write_sentence_csv(output_dir / "sentence_judgments.csv", all_results)
    write_summary(output_dir / "summary.md", all_results)

    print(f"Files wrote to: {output_dir}.")


if __name__ == "__main__":
    main()