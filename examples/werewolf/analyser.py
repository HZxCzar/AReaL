#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Werewolf trajectory analyzer for the new JSON log format.

Features:
- Parses pretty-printed JSONL logs from werewolf workflow dump.
- Computes role-claim rates, vote accuracy, and keyword-driven strategy signals.
- Optionally uses an LLM judge to assess villager strategy quality.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import requests
from collections import Counter
from dataclasses import dataclass
from json import JSONDecoder
from pathlib import Path
from typing import Iterable, Iterator
import importlib.util
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

ROLE_ALIASES = {
    "wolf": "werewolf",
    "seer": "foreseer",
}

ROLE_CLAIM_RE = re.compile(
    r"\b(i am|i'm|im|my role is|as the)\s+(a|an|the)?\s*"
    r"(villager|werewolf|wolf|foreseer|seer|witch|hunter)\b",
    re.IGNORECASE,
)

VOTE_RE = re.compile(r"\bvote\s+player(\d+)\b", re.IGNORECASE)
PHASE_RE = re.compile(r"\bphase\s+(day|night)\s+round\s+(\d+)\b", re.IGNORECASE)

STRATEGY_KEYWORDS = sorted(
    {
        "accuse",
        "defend",
        "alibi",
        "vote",
        "voting",
        "wagon",
        "hammer",
        "lynch",
        "eliminate",
        "execute",
        "flip",
        "town",
        "villager",
        "wolf",
        "werewolf",
        "seer",
        "doctor",
        "witch",
        "hunter",
        "claim",
        "counterclaim",
        "fakeclaim",
        "softclaim",
        "hardclaim",
        "clear",
        "frame",
        "pressure",
        "gambit",
        "bluff",
        "distancing",
        "solve",
        "mechanics",
        "plan",
        "strategy",
        "protect",
        "guard",
        "kill",
        "poison",
        "check",
        "probability",
        "odds",
        "confidence",
        "uncertain",
        "maybe",
        "perhaps",
    }
)

HEDGE_TERMS = [
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "unsure",
    "uncertain",
    "not sure",
    "i think",
    "i guess",
    "i feel",
    "lean",
    "seems",
    "appears",
    "could be",
    "might be",
]

CONFIDENCE_TERMS = [
    "certain",
    "sure",
    "confident",
    "confirmed",
    "definitely",
    "absolutely",
    "guaranteed",
    # "locked",
]

REASONING_PATTERNS = {
    "concealment": [
        "hide",
        "conceal",
        "disguise",
        "mask",
        "cover",
        "secret",
        "avoid revealing",
        "keep my role",
        "stay hidden",
        "stay quiet",
        # "silent",
    ],
    "cooperation": [
        "cooperate",
        "coordinate",
        "align",
        "team",
        "support",
        "agree",
        "work together",
        "trust",
        "save",
        "heal",
    ],
    "bluffing": [
        "bluff",
        "fake",
        "pretend",
        "mislead",
        "lie",
        "lying",
        "deceive",
    ],
    "sacrificing": [
        "sacrifice",
        "trade",
        "bus",
        "bussing",
        "throw",
        "give up",
    ],
    "pressure": [
        "pressure",
        "push",
        "force",
        "corner",
    ],
    "consensus_building": [
        "consensus",
        "majority",
        "bandwagon",
        "wagon",
        "vote together",
    ],
    "role_signal": [
        "claim",
        "role",
        "seer",
        "foreseer",
        "witch",
        "hunter",
        "villager",
        "werewolf",
    ],
}


@dataclass
class ClaimEvent:
    file: str
    episode_index: int
    turn: int
    agent: str
    actual_role: str
    claimed_role: str
    phase_label: str | None


@dataclass
class VoteEvent:
    file: str
    episode_index: int
    turn: int
    agent: str
    actual_role: str
    target: str
    target_role: str | None
    phase_label: str | None
    correct_vs_wolves: bool | None


@dataclass
class JudgeResult:
    file: str
    episode_index: int
    information_sharing: str
    coordinated_voting: str
    role_claim_timing: str
    evidence_usage: str
    notes: str


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


def parse_initial_setup(trajectory: list[str]) -> dict[str, str]:
    if not trajectory:
        return {}
    first = trajectory[0]
    if "initial setup" not in first.lower():
        return {}
    assignments: dict[str, str] = {}
    match = re.split(r"->", first, maxsplit=1)
    right = match[1] if len(match) > 1 else first
    for chunk in right.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        found = re.match(r"(player\d+)\s*:\s*([A-Za-z]+)", chunk)
        if not found:
            continue
        assignments[found.group(1).lower()] = found.group(2).lower()
    return assignments


def normalize_role(role: str | None) -> str:
    if not role:
        return "unknown"
    role = role.lower()
    return ROLE_ALIASES.get(role, role)


def extract_claims(text: str) -> list[str]:
    claims = []
    for match in ROLE_CLAIM_RE.finditer(text):
        role = normalize_role(match.group(3))
        claims.append(role)
    return claims


def parse_phase_label(text: str | None) -> str | None:
    if not text:
        return None
    match = PHASE_RE.search(text)
    if not match:
        return None
    phase, number = match.groups()
    return f"{phase.capitalize()} {number}"


def extract_votes(text: str) -> list[str]:
    return [f"player{match.group(1)}" for match in VOTE_RE.finditer(text)]


def count_terms(text: str, terms: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(term) for term in terms)


def extract_strategy_keywords(text: str) -> Counter:
    lower = text.lower()
    return Counter({kw: lower.count(kw) for kw in STRATEGY_KEYWORDS if kw in lower})


def extract_reasoning_patterns(text: str) -> Counter:
    lower = text.lower()
    counts = Counter()
    for pattern_name, terms in REASONING_PATTERNS.items():
        counts[pattern_name] = sum(lower.count(term) for term in terms)
    return counts


def detect_winner(trajectory: list[str]) -> str | None:
    for line in reversed(trajectory):
        lower = line.lower()
        if "villagers win" in lower:
            return "villagers"
        if "werewolves win" in lower:
            return "werewolves"
    return None


def build_judge_prompt(trajectory: list[str]) -> str:
    # Remove any chain-of-thought blocks if they appear in your logs
    trajectory = [traj.split(": \n<think>")[0][:300] for traj in trajectory]
    # Normalize boilerplate / overlapping segments
    trajectory = [
        traj if not traj.startswith("In the last discussion phase")
        else "Discussion over, vote begins."
        for traj in trajectory
    ]
    joined = "\n".join(trajectory)

    return f"""You are an impartial evaluator of a Werewolf (Mafia) game trajectory.
Your job is to judge whether the **villager side** demonstrates useful team strategies.

Evaluate these strategy dimensions (use the definitions below). Then output **STRICT JSON ONLY**.

## Strategy definitions and decision rules
1) information_sharing (yes/no/uncertain)
- What it means: Villager-aligned players share relevant observations that help others update beliefs.
  Examples: quoting past statements, reporting vote intentions, summarizing who accused whom, clarifying timeline, sharing constraints (e.g., "I was targeted last night").
- Mark "yes" if: there is clear, actionable sharing that improves group situational awareness.
- Mark "no" if: discussion is mostly noise, self-focused, or lacks concrete shareable info.
- Mark "uncertain" if: the trajectory is too short/fragmented to tell.

2) coordinated_voting (yes/no/uncertain)
- What it means: Villagers align votes through explicit coordination or convergence.
  Examples: proposing a vote plan, negotiating a consensus target, whip-like “if X then vote Y”, reacting to new evidence and pivoting together.
- Mark "yes" if: there is explicit coordination or visible convergence with stated reasons.
- Mark "no" if: votes appear random, purely individual, or no attempt to align is present.
- Mark "uncertain" if: votes are not shown or the log lacks the voting phase.

3) role_claim_timing (yes/no/uncertain)
- What it means: Role claims (e.g., seer/doctor) are made at a time that helps villagers:
  not too early to be easily eliminated, not too late to be useless; claims are used to guide decisions.
- Mark "yes" if: claims are timely and strategically leveraged (e.g., to prevent misvote, to confirm information, to coordinate protection).
- Mark "no" if: claims are reckless (very early with no need), manipulative without follow-through, or so late they provide no benefit.
- Mark "uncertain" if: there are no claims or role context is missing.

4) evidence_usage (yes/no/uncertain)
- What it means: Villagers use evidence to justify accusations/defenses rather than vibes.
  Evidence includes: contradictions, consistency over time, voting record, night actions, seer checks, probabilistic reasoning, elimination logic.
- Mark "yes" if: arguments cite concrete events or logical chains tied to the trajectory.
- Mark "no" if: accusations are mostly ad hominem, vibes-only, or ignore available evidence.
- Mark "uncertain" if: the log doesn’t contain enough argumentation.

## Output format requirements
Return STRICT JSON with EXACT keys:
- "information_sharing": "yes" | "no" | "uncertain"
- "coordinated_voting": "yes" | "no" | "uncertain"
- "role_claim_timing": "yes" | "no" | "uncertain"
- "evidence_usage": "yes" | "no" | "uncertain"
- "notes": string

Rules:
- Output JSON ONLY (no markdown, no extra text).
- In notes, give 2–5 short bullet-like sentences explaining the main evidence for your labels.
- If villagers are not clearly identifiable, judge based on behavior consistent with villager-side strategy (team-information and truth-seeking).

Trajectory:
{joined}
"""


def ensure_transformers_available() -> None:
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("transformers is required for --judge-backend local")


def run_local_judge(prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
    ensure_transformers_available()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded


def run_openai_judge(prompt: str, model_name: str, base_url: str, api_key: str,
                    max_tokens: int, temperature: float) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a careful evaluator."},
            {"role": "user", "content": prompt},
        ],
        # NOTE: some servers expect "max_tokens" instead of "max_completion_tokens"
        "max_tokens": int(max_tokens * 1.5),
        "thinking": {
            "type": "enabled",
            "budget_tokens": max_tokens,
        },
        "temperature": temperature,
    }

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
        raise_on_status=False,
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # connect timeout, read timeout (seconds)
    r = session.post(url, headers=headers, data=json.dumps(payload), timeout=(10, 120))
    # If you want to see server errors:
    r.raise_for_status()

    decoded = r.json()
    return decoded["choices"][0]["message"]["content"]


def parse_judge_json(text: str) -> tuple[dict[str, str], str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}, "No JSON payload found"
    payload = match.group(0)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}, "Failed to parse JSON"
    verdict = {
        "information_sharing": str(parsed.get("information_sharing", "uncertain")).lower(),
        "coordinated_voting": str(parsed.get("coordinated_voting", "uncertain")).lower(),
        "role_claim_timing": str(parsed.get("role_claim_timing", "uncertain")).lower(),
        "evidence_usage": str(parsed.get("evidence_usage", "uncertain")).lower(),
    }
    notes = str(parsed.get("notes", ""))
    return verdict, notes


def analyze_records(
    records: Iterable[dict], source_file: str
) -> tuple[list[ClaimEvent], list[VoteEvent], Counter, Counter, dict]:
    claims: list[ClaimEvent] = []
    votes: list[VoteEvent] = []
    strategy_counts = Counter()
    reasoning_pattern_counts = Counter()
    summary_counts = Counter()

    for record in records:
        episode_index = int(record.get("episode_index", 0))
        trajectory = record.get("trajectory", []) or []
        role_map = parse_initial_setup(trajectory)
        winner = detect_winner(trajectory)
        if winner:
            summary_counts[f"winner_{winner}"] += 1
        summary_counts["episodes"] += 1

        for step in record.get("steps", []):
            turn = int(step.get("turn", 0))
            agent = str(step.get("agent", "unknown"))
            actual_role = normalize_role(step.get("role", "unknown"))
            action_text = str(step.get("action_completion", ""))
            thought_text = str(step.get("agent_thought", ""))
            qa_text = "\n".join(step.get("self_questions", []) or [])
            answer_text = "\n".join(step.get("agent_answers", []) or [])
            combined_text = "\n".join([action_text, thought_text, qa_text, answer_text])
            phase_label = parse_phase_label(step.get("observation") or step.get("action_prompt"))

            for claimed_role in extract_claims(combined_text):
                claims.append(
                    ClaimEvent(
                        file=source_file,
                        episode_index=episode_index,
                        turn=turn,
                        agent=agent,
                        actual_role=actual_role,
                        claimed_role=claimed_role,
                        phase_label=phase_label,
                    )
                )

            for target in extract_votes(action_text):
                target_role = normalize_role(role_map.get(target.lower())) if role_map else None
                correct = None
                if actual_role in {"villager", "foreseer", "witch", "hunter"} and target_role != "unknown":
                    correct = target_role == "werewolf"
                votes.append(
                    VoteEvent(
                        file=source_file,
                        episode_index=episode_index,
                        turn=turn,
                        agent=agent,
                        actual_role=actual_role,
                        target=target,
                        target_role=target_role,
                        phase_label=phase_label,
                        correct_vs_wolves=correct,
                    )
                )

            strategy_counts.update(extract_strategy_keywords(combined_text))
            reasoning_pattern_counts.update(extract_reasoning_patterns(combined_text))
            summary_counts["hedge_terms"] += count_terms(combined_text, HEDGE_TERMS)
            summary_counts["confidence_terms"] += count_terms(combined_text, CONFIDENCE_TERMS)

    return claims, votes, strategy_counts, reasoning_pattern_counts, summary_counts


def write_claims_csv(path: Path, claims: list[ClaimEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file",
            "episode_index",
            "turn",
            "agent",
            "actual_role",
            "claimed_role",
            "phase",
            "truthful",
        ])
        for claim in claims:
            truthful = claim.claimed_role == claim.actual_role
            writer.writerow([
                claim.file,
                claim.episode_index,
                claim.turn,
                claim.agent,
                claim.actual_role,
                claim.claimed_role,
                claim.phase_label or "",
                str(truthful),
            ])


def write_votes_csv(path: Path, votes: list[VoteEvent]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "file",
            "episode_index",
            "turn",
            "agent",
            "actual_role",
            "target",
            "target_role",
            "phase",
            "villager_vote_correct",
        ])
        for vote in votes:
            writer.writerow([
                vote.file,
                vote.episode_index,
                vote.turn,
                vote.agent,
                vote.actual_role,
                vote.target,
                vote.target_role or "",
                vote.phase_label or "",
                "" if vote.correct_vs_wolves is None else str(vote.correct_vs_wolves),
            ])


def write_keyword_csv(path: Path, counts: Counter) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["keyword", "count"])
        for keyword, count in counts.most_common():
            writer.writerow([keyword, count])


def write_pattern_csv(path: Path, counts: Counter) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pattern", "count"])
        for pattern, count in counts.most_common():
            writer.writerow([pattern, count])


def write_summary(
    path: Path,
    summary_counts: dict,
    claims: list[ClaimEvent],
    votes: list[VoteEvent],
    judge_results: list[JudgeResult],
) -> None:
    total_claims = len(claims)
    truthful_claims = sum(1 for c in claims if c.claimed_role == c.actual_role)
    false_claims = sum(1 for c in claims if c.claimed_role != c.actual_role)
    vill_votes = [v for v in votes if v.correct_vs_wolves is not None]
    vill_vote_accuracy = (
        sum(1 for v in vill_votes if v.correct_vs_wolves) / len(vill_votes)
        if vill_votes
        else float("nan")
    )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Werewolf Analysis Summary\n\n")
        handle.write(f"Episodes analyzed: {summary_counts.get('episodes', 0)}\n\n")
        handle.write(f"Villager wins: {summary_counts.get('winner_villagers', 0)}\n")
        handle.write(f"Werewolf wins: {summary_counts.get('winner_werewolves', 0)}\n\n")
        handle.write(f"Total role claims: {total_claims}\n")
        handle.write(f"Truthful claim rate: {truthful_claims / total_claims if total_claims else 0:.3f}\n\n")
        handle.write(f"False claims: {false_claims}\n\n")
        handle.write(f"Villager vote accuracy: {vill_vote_accuracy:.3f}\n\n")
        handle.write(f"Hedge terms count: {summary_counts.get('hedge_terms', 0)}\n")
        handle.write(f"Confidence terms count: {summary_counts.get('confidence_terms', 0)}\n")
        if judge_results:
            handle.write("\n## Judge Metrics\n")
            totals = {
                "information_sharing": Counter(),
                "coordinated_voting": Counter(),
                "role_claim_timing": Counter(),
                "evidence_usage": Counter(),
            }
            for result in judge_results:
                totals["information_sharing"][result.information_sharing] += 1
                totals["coordinated_voting"][result.coordinated_voting] += 1
                totals["role_claim_timing"][result.role_claim_timing] += 1
                totals["evidence_usage"][result.evidence_usage] += 1
            handle.write(f"information_sharing: {dict(totals['information_sharing'])}\n")
            handle.write(f"coordinated_voting: {dict(totals['coordinated_voting'])}\n")
            handle.write(f"role_claim_timing: {dict(totals['role_claim_timing'])}\n")
            handle.write(f"evidence_usage: {dict(totals['evidence_usage'])}\n")


def run_llm_judge(
    records: Iterable[dict],
    source_file: str,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    sample_limit: int | None,
) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    for record in records:
        episode_index = int(record.get("episode_index", 0))
        trajectory = record.get("trajectory", []) or []
        prompt = build_judge_prompt(trajectory)
        print(f"Calling judge with input len {len(prompt)}...", flush=True)
        if backend == "local":
            response = run_local_judge(prompt, model, max_tokens, temperature)
        elif backend == "openai":
            response = run_openai_judge(prompt, model, base_url, api_key, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown judge backend: {backend}")
        verdict, notes = parse_judge_json(response)
        results.append(
            JudgeResult(
                file=source_file,
                episode_index=episode_index,
                information_sharing=verdict.get("information_sharing", "uncertain"),
                coordinated_voting=verdict.get("coordinated_voting", "uncertain"),
                role_claim_timing=verdict.get("role_claim_timing", "uncertain"),
                evidence_usage=verdict.get("evidence_usage", "uncertain"),
                notes=notes,
            )
        )
        if sample_limit and len(results) >= sample_limit:
            break
        time.sleep(0.1)
    return results


def write_judge_csv(path: Path, results: list[JudgeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "episode_index",
                "information_sharing",
                "coordinated_voting",
                "role_claim_timing",
                "evidence_usage",
                "notes",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.file,
                    result.episode_index,
                    result.information_sharing,
                    result.coordinated_voting,
                    result.role_claim_timing,
                    result.evidence_usage,
                    result.notes,
                ]
            )


"""
python examples/werewolf/analyser.py --root /storage/openpsi/experiments/logs/admin/xmy-werewolf-eval/qwen3-8b-prm2.5-vs-claude-4-5-thinking-noqa-step4/generated/0 \
    --output-dir analysis/4-2
"""

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze werewolf JSON logs.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root directory or file.")
    parser.add_argument("--glob", dest="glob_pattern", default="*.jsonl", help="Glob pattern to match logs.")
    parser.add_argument("--output-dir", type=Path, default=Path("werewolf_analysis"))
    parser.add_argument("--judge-backend", choices=["none", "local", "openai"], default="openai")
    parser.add_argument("--judge-model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--judge-base-url", default="https://matrixllm.alipay.com/")
    parser.add_argument("--judge-api-key", default="sk-43dd5f664179406d92fec42a9364f8a5")
    parser.add_argument("--judge-max-tokens", type=int, default=15000)
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--judge-sample-limit", type=int, default=None)

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_claims: list[ClaimEvent] = []
    all_votes: list[VoteEvent] = []
    keyword_counts = Counter()
    pattern_counts = Counter()
    summary_counts = Counter()
    judge_results: list[JudgeResult] = []

    file_entries = iter_log_paths(args.root, args.glob_pattern)
    print(f"Found {len(file_entries)} log files to analyse.")

    for idx in range(len(file_entries)):
        path = file_entries[idx]
        print(f"Processing {idx}/{len(file_entries)} file entry...", flush=True)

        records = list(iter_records(path))
        claims, votes, strategy_counts, reasoning_patterns, file_summary = analyze_records(
            records, path.name
        )
        all_claims.extend(claims)
        all_votes.extend(votes)
        keyword_counts.update(strategy_counts)
        pattern_counts.update(reasoning_patterns)
        summary_counts.update(file_summary)

        if args.judge_backend != "none":
            judge_results.extend(
                run_llm_judge(
                    records,
                    path.name,
                    args.judge_backend,
                    args.judge_model,
                    args.judge_base_url,
                    args.judge_api_key,
                    args.judge_max_tokens,
                    args.judge_temperature,
                    args.judge_sample_limit,
                )
            )

    write_claims_csv(output_dir / "role_claims.csv", all_claims)
    write_votes_csv(output_dir / "votes.csv", all_votes)
    write_keyword_csv(output_dir / "strategy_keywords.csv", keyword_counts)
    write_pattern_csv(output_dir / "reasoning_patterns.csv", pattern_counts)
    write_summary(output_dir / "summary.md", summary_counts, all_claims, all_votes, judge_results)

    if judge_results:
        write_judge_csv(output_dir / "judge_results.csv", judge_results)


if __name__ == "__main__":
    main()