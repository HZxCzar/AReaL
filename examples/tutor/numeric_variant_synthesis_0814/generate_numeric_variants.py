#!/usr/bin/env python3
"""Generate, audit, solve twice, and student-probe numeric-only variants."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from examples.tutor.prompts import (
    DEFAULT_STUDENT_SYSTEM_PROMPT,
    INITIAL_TEACHER_FEEDBACK_PLACEHOLDER,
    TASK_CONTEXT_TEMPLATE,
    render_prompt,
)

from model_client import SerialTutorClients
from numeric_variants import (
    answers_equivalent,
    apply_numeric_edits,
    atomic_write_json,
    extracted_answer,
    mechanical_report,
    numeric_tokens,
    parse_tagged_json,
    safe_filename,
    strict_audit_pass,
    target_answer,
    token_inventory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = "examples/tutor/data/math_1.7b_8b/math_pass@2"

GENERATOR_SYSTEM = """You synthesize rigorous numeric-only variants of MATH problems.
You may reason privately, but the final response must contain exactly one
<FINAL_JSON>...</FINAL_JSON> block. Return indexed numeric edits, never a rewritten task."""

AUDITOR_SYSTEM = """You are a strict dataset auditor. Compare a source math problem and
a mechanically reconstructed candidate. Reject any semantic drift or unsafe numeric edit.
End with exactly one <FINAL_JSON>...</FINAL_JSON> object and no text after it."""

SOLVER_SYSTEM = """Solve the given math problem independently and carefully. You have
not seen any source problem or claimed answer. Show enough reasoning to self-check the
result and end with exactly one final answer in \\boxed{...}."""

ANSWER_AUDITOR_SYSTEM = """Verify a proposed math answer rigorously. Agreement between
two solvers is not proof. Independently recompute, check feasibility and every constraint,
and reject an ill-posed task or a stable-but-wrong answer. End with one FINAL_JSON block."""

AUDIT_KEYS = (
    "only_numeric_values_changed",
    "same_wording_units_target_constraints",
    "same_computation_graph",
    "same_knowledge_point",
    "structural_numbers_unchanged",
    "dependent_values_consistent",
    "well_posed",
    "similar_difficulty",
)


def generator_prompt(
    source: dict[str, Any], prior_rejections: list[dict[str, Any]]
) -> str:
    history = json.dumps(prior_rejections, ensure_ascii=False, indent=2)
    allowed = [
        token["token_index"]
        for token in numeric_tokens(source["task"])
        if token["editable"]
    ]
    return f"""Create one useful numeric-only variant.

SOURCE TASK:
{source["task"]}

SOURCE ANSWER:
{source["ground_truth"]}

SOURCE REFERENCE SOLUTION:
{source["reference_solution"]}

STABLE INDEXED NUMERIC TOKENS:
{token_inventory(source["task"])}

ALLOWED EDITABLE TOKEN INDICES:
{allowed}

PRIOR REJECTIONS:
{history if prior_rejections else "(none)"}

Rules:
1. Return edits only. The program reconstructs the candidate from the source bytes.
2. Every token_index MUST be in ALLOWED EDITABLE TOKEN INDICES. Never select a
   PROTECTED token, even if it appears in an example, formula, exponent, or subscript.
3. Prefer the minimum intervention: usually change one causally active independent
   given. Change repeated/dependent occurrences together only when consistency requires.
4. Keep every word, name, unit, target object/variables, constraint type, operation
   pattern, relation type, computation graph, knowledge point, and difficulty unchanged.
   A numeric coefficient, bound, or threshold inside the requested expression may change
   when the target variables, operators, expression shape, and direction stay the same.
5. Do not edit structural constants, dimensions, bases/moduli, sequence-rule counters,
   answer-choice labels, rounding precision, problem numbering, or diagram coordinates.
6. Keep each new positive value between 0.5x and 2x its old value. Prefer clean values
   of the same scale. Never collapse a threshold such as 60 to 5.
7. Solve and independently self-check the reconstructed candidate. The final answer
   must differ from SOURCE ANSWER. Pick a value that actually affects the result.
8. Prefer a simple, stable calculation. For interest/rounding problems, a clean target
   scaling is safer than simultaneously changing rate, duration, and target.
9. For a discrete first/largest/smallest threshold question, calculate adjacent outcomes
   and move the threshold far enough to change the answer. A tiny change that leaves the
   same first/largest/smallest object is invalid.

Replace every uppercase placeholder below with actual values. token_index may be an
integer or an integer string, but it must come from ALLOWED EDITABLE TOKEN INDICES.
Do not include a proposed answer. End exactly with:
<FINAL_JSON>
{{
  "edits": [
    {{
      "token_index": "ALLOWED_INTEGER",
      "old": "EXACT_OLD_TOKEN",
      "new": "NEW_NUMERIC_LITERAL",
      "role": "SHORT_SEMANTIC_ROLE"
    }}
  ]
}}
</FINAL_JSON>"""


def audit_prompt(
    source: dict[str, Any],
    candidate: str,
    edits: list[dict[str, Any]],
    proposed_answer: str,
) -> str:
    return f"""Audit this candidate conservatively.

SOURCE TASK:
{source["task"]}

CANDIDATE TASK:
{candidate}

APPLIED EDITS:
{json.dumps(edits, ensure_ascii=False, indent=2)}

GENERATOR'S PROPOSED ANSWER:
{proposed_answer or "(not supplied)"}

The nonnumeric skeleton and Asymptote spans have already been checked mechanically.
Still reject if any changed token is structural rather than a replaceable mathematical
given. Changing the VALUE of an ordinary numeric given, bound, threshold, or numeric
coefficient in the requested expression is intended and is not by itself semantic drift.
For example, changing a rate, price, count, inequality endpoint, target threshold, or
10a+b to 15a+b can pass. "Same target" means the same target variables/object, operator
pattern, expression shape, and optimization direction after numeric parameter
substitution; it does not require identical numeric coefficients. "Same constraints"
means the same constraint types, directions, inclusivity, variable domains, and
relationships after parameter substitution. "Same computation graph" means the same
symbolic solution operations with the new parameters; intermediate numeric results may
differ. Reject if dependent values are inconsistent; if units, target type/expression
shape, constraint types, computation graph, knowledge point, or intended interpretation
changed; if the task is ill posed/degenerate; or if difficulty changed substantially.
Be especially strict about exponents, dimensions, bases/moduli, sequence rules, rounding
precision, diagram/layout constants, and hidden relations.

End with exactly:
<FINAL_JSON>
{{
  "pass": true,
  "only_numeric_values_changed": true,
  "same_wording_units_target_constraints": true,
  "same_computation_graph": true,
  "same_knowledge_point": true,
  "structural_numbers_unchanged": true,
  "dependent_values_consistent": true,
  "well_posed": true,
  "similar_difficulty": true,
  "changed_numeric_roles": ["..."],
  "reason": "concise evidence"
}}
</FINAL_JSON>"""


def solver_messages(task: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SOLVER_SYSTEM},
        {"role": "user", "content": task},
    ]


def answer_audit_prompt(
    task: str, outputs: list[str], answers: list[str]
) -> str:
    return f"""Verify the candidate answer by independently solving the task.

TASK:
{task}

SOLVER 1 ANSWER:
{answers[0]}

SOLVER 1 WORK:
{outputs[0]}

SOLVER 2 ANSWER:
{answers[1]}

SOLVER 2 WORK:
{outputs[1]}

Do not accept merely because the answers agree. Recompute the result, test it against
all strict inequalities, integrality/domain requirements, optimization language, and
rounding rules, and confirm the task has at least one valid answer.

End with:
<FINAL_JSON>
{{
  "pass": true,
  "problem_well_posed": true,
  "answer_correct": true,
  "answer_satisfies_all_constraints": true,
  "reason": "concise independent verification"
}}
</FINAL_JSON>"""


def strict_answer_audit(audit: dict[str, Any]) -> bool:
    return bool(audit.get("pass")) and all(
        audit.get(key) is True
        for key in (
            "problem_well_posed",
            "answer_correct",
            "answer_satisfies_all_constraints",
        )
    )


def student_messages(task: str) -> list[dict[str, str]]:
    context = render_prompt(TASK_CONTEXT_TEMPLATE, task=task)
    return [
        {
            "role": "system",
            "content": f"{DEFAULT_STUDENT_SYSTEM_PROMPT.rstrip()}\n\n{context}",
        },
        {"role": "user", "content": INITIAL_TEACHER_FEEDBACK_PLACEHOLDER},
    ]


def load_sources(dataset_path: Path, ids: list[str]) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(dataset_path))
    wanted = set(ids)
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        if split not in dataset:
            continue
        for raw in dataset[split]:
            source_id = str(raw["id"])
            if wanted and source_id not in wanted:
                continue
            metadata = dict(raw.get("metadata") or {})
            rows.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "task": str(raw["task"]),
                    "ground_truth": str(raw["ground_truth"]),
                    "reference_solution": str(metadata.get("solution", "")),
                    "subject": str(metadata.get("type", "")),
                    "level": str(metadata.get("level", "")),
                }
            )
    if ids:
        by_id = {row["source_id"]: row for row in rows}
        missing = [source_id for source_id in ids if source_id not in by_id]
        if missing:
            raise ValueError(f"unknown source ids: {missing}")
        rows = [by_id[source_id] for source_id in ids]
    return rows


def resolve_run_dir(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = SCRIPT_DIR / candidate
    resolved = candidate.resolve()
    root = SCRIPT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"run directory must stay below {root}")
    if resolved == root:
        raise ValueError("run directory must be a child of the script directory")
    return resolved


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def rebuild_outputs(
    run_dir: Path,
    requested: int,
    source_concurrency: int,
    max_concurrent_calls: int,
) -> dict[str, Any]:
    records = []
    for path in sorted((run_dir / "records").glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    counts = Counter(str(row.get("status", "unknown")) for row in records)
    accepted = [row for row in records if row.get("status") == "accepted"]
    bank = []
    for row in accepted:
        item = row["accepted"]
        bank.append(
            {
                "source_id": row["source"]["source_id"],
                "split": row["source"]["split"],
                "teacher_variant_task": item["task"],
                "teacher_variant_answer": item["answer"],
                "original_retest_task": row["source"]["task"],
                "original_retest_ground_truth": row["source"]["ground_truth"],
                "edits": item["edits"],
                "mechanical_checks": item["mechanical_checks"],
                "semantic_audit": item["semantic_audit"],
                "teacher_solve_answers": item["teacher_solve_answers"],
                "answer_audit": item["answer_audit"],
                "student_bonus": item["student_bonus"],
            }
        )
    write_jsonl_atomic(run_dir / "variant_bank.jsonl", bank)
    summary = {
        "updated_unix": time.time(),
        "requested_sources": requested,
        "completed_records": len(records),
        "status_counts": dict(sorted(counts.items())),
        "accepted_variants": len(accepted),
        "teacher_solve_consistency_required": "2/2",
        "student_failure_is_bonus_not_acceptance_gate": True,
        "source_concurrency": source_concurrency,
        "max_concurrent_calls": max_concurrent_calls,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary


async def process_source(
    client: SerialTutorClients,
    source: dict[str, Any],
    record_path: Path,
    max_attempts: int,
    retry_failed: bool,
) -> dict[str, Any]:
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "accepted":
            return record
        if not retry_failed and record.get("status") in {"failed", "ineligible"}:
            return record
    else:
        record = {
            "source": source,
            "status": "running",
            "attempts": [],
            "policy": {
                "numeric_edits_only": True,
                "asymptote_edits_forbidden": True,
                "teacher_solve_replays": 2,
                "student_is_bonus": True,
            },
        }

    editable = [token for token in numeric_tokens(source["task"]) if token["editable"]]
    if not editable:
        record["status"] = "ineligible"
        record["reason"] = "no editable numeric literal outside Asymptote"
        atomic_write_json(record_path, record)
        return record

    prior = [
        {
            "attempt": item.get("attempt"),
            "rejection": item.get("rejection", item.get("error", "unknown")),
        }
        for item in record["attempts"]
    ][-3:]
    first_attempt = len(record["attempts"]) + 1

    for attempt_number in range(first_attempt, first_attempt + max_attempts):
        stage = f"{source['source_id']}:attempt-{attempt_number}"
        attempt: dict[str, Any] = {"attempt": attempt_number}
        print(f"[{source['source_id']}] generate attempt {attempt_number}", flush=True)
        try:
            generated = await client.teacher_call(
                [
                    {"role": "system", "content": GENERATOR_SYSTEM},
                    {"role": "user", "content": generator_prompt(source, prior)},
                ],
                stage=f"{stage}:generate",
                max_tokens=3072,
                temperature=0.6,
                seed=10000 + attempt_number,
            )
            attempt["generator_raw"] = generated
            parsed = parse_tagged_json(generated)
            candidate, edits = apply_numeric_edits(source["task"], parsed.get("edits"))
            proposed = str(parsed.get("proposed_answer", "")).strip()
            checks = mechanical_report(source["task"], candidate, edits)
            attempt.update(
                {
                    "generator_parsed": parsed,
                    "candidate_task": candidate,
                    "edits": edits,
                    "mechanical_checks": checks,
                }
            )

            audited = await client.teacher_call(
                [
                    {"role": "system", "content": AUDITOR_SYSTEM},
                    {
                        "role": "user",
                        "content": audit_prompt(source, candidate, edits, proposed),
                    },
                ],
                stage=f"{stage}:semantic-audit",
                max_tokens=1536,
                temperature=0.0,
                seed=20000 + attempt_number,
            )
            attempt["semantic_audit_raw"] = audited
            audit = parse_tagged_json(audited)
            attempt["semantic_audit"] = audit
            if not strict_audit_pass(audit):
                raise ValueError(f"semantic audit rejected: {audit.get('reason', '')}")

            solve_outputs = []
            solve_answers = []
            for replay in (1, 2):
                output = await client.teacher_call(
                    solver_messages(candidate),
                    stage=f"{stage}:blind-solve-{replay}",
                    max_tokens=4096,
                    temperature=0.2,
                    seed=30000 + attempt_number * 10 + replay,
                )
                answer = extracted_answer(output)
                solve_outputs.append(output)
                solve_answers.append(answer)
            attempt["teacher_solve_outputs"] = solve_outputs
            attempt["teacher_solve_answers"] = solve_answers

            if not answers_equivalent(solve_answers[0], solve_answers[1]):
                raise ValueError(
                    f"teacher answers are not stable: {solve_answers!r}"
                )
            attempt["proposed_answer_check"] = {
                "provided": bool(proposed),
                "matches_two_solve_answer": (
                    answers_equivalent(target_answer(proposed), solve_answers[0])
                    if proposed
                    else None
                ),
            }
            source_answer = target_answer(source["ground_truth"])
            if answers_equivalent(source_answer, solve_answers[0]):
                raise ValueError("variant answer is equivalent to source answer")

            answer_audit_raw = await client.teacher_call(
                [
                    {"role": "system", "content": ANSWER_AUDITOR_SYSTEM},
                    {
                        "role": "user",
                        "content": answer_audit_prompt(
                            candidate, solve_outputs, solve_answers
                        ),
                    },
                ],
                stage=f"{stage}:answer-audit",
                max_tokens=1536,
                temperature=0.0,
                seed=35000 + attempt_number,
            )
            answer_audit = parse_tagged_json(answer_audit_raw)
            attempt["answer_audit_raw"] = answer_audit_raw
            attempt["answer_audit"] = answer_audit
            if not strict_answer_audit(answer_audit):
                raise ValueError(
                    f"answer audit rejected: {answer_audit.get('reason', '')}"
                )

            student_output = await client.student_call(
                student_messages(candidate),
                stage=f"{stage}:student-bonus",
                seed=40000 + attempt_number,
            )
            student_answer = extracted_answer(student_output)
            student_solved = answers_equivalent(student_answer, solve_answers[0])
            student_bonus = {
                "output": student_output,
                "extracted_answer": student_answer,
                "solved": student_solved,
                "preferred_outcome": "failed" if not student_solved else "solved",
                "acceptance_gate": False,
            }
            attempt["student_bonus"] = student_bonus
            attempt["accepted"] = True
            record["attempts"].append(attempt)
            record["status"] = "accepted"
            record["accepted"] = {
                "attempt": attempt_number,
                "task": candidate,
                "answer": solve_answers[0],
                "edits": edits,
                "mechanical_checks": checks,
                "semantic_audit": audit,
                "teacher_solve_answers": solve_answers,
                "teacher_solve_outputs": solve_outputs,
                "answer_audit": answer_audit,
                "student_bonus": student_bonus,
            }
            atomic_write_json(record_path, record)
            print(
                f"[{source['source_id']}] accepted "
                f"student_solved={student_solved}",
                flush=True,
            )
            return record
        except Exception as error:
            attempt["accepted"] = False
            attempt["error_type"] = error.__class__.__name__
            attempt["rejection"] = str(error)
            record["attempts"].append(attempt)
            prior.append({"attempt": attempt_number, "rejection": str(error)})
            record["status"] = "running"
            atomic_write_json(record_path, record)
            print(f"[{source['source_id']}] rejected: {error}", flush=True)

    record["status"] = "failed"
    record["reason"] = f"no accepted candidate in {max_attempts} new attempts"
    atomic_write_json(record_path, record)
    return record


async def async_main(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir)
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)
    sources = load_sources(dataset_path, args.ids)
    if args.limit:
        sources = sources[: args.limit]
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "dataset": str(dataset_path.resolve()),
            "source_ids": [source["source_id"] for source in sources],
            "max_attempts_per_invocation": args.max_attempts,
            "min_interval_seconds": args.min_interval_seconds,
            "source_concurrency": args.source_concurrency,
            "max_concurrent_calls": args.max_concurrent_calls,
            "teacher_endpoint_env": "TUTOR_QWEN3_8B_BASE_URL",
            "student_endpoint_env": "TUTOR_QWEN3_1_7B_BASE_URL",
            "created_or_resumed_unix": time.time(),
        },
    )

    async with SerialTutorClients(
        run_dir / "calls.jsonl",
        args.min_interval_seconds,
        args.max_concurrent_calls,
    ) as client:
        source_semaphore = asyncio.Semaphore(args.source_concurrency)
        output_lock = asyncio.Lock()

        async def run_one(position: int, source: dict[str, Any]) -> None:
            async with source_semaphore:
                print(
                    f"[source {position}/{len(sources)}] {source['source_id']}",
                    flush=True,
                )
                path = records_dir / f"{safe_filename(source['source_id'])}.json"
                await process_source(
                    client, source, path, args.max_attempts, args.retry_failed
                )
            async with output_lock:
                summary = rebuild_outputs(
                    run_dir,
                    len(sources),
                    args.source_concurrency,
                    args.max_concurrent_calls,
                )
                print(f"[summary] {summary['status_counts']}", flush=True)

        await asyncio.gather(
            *(
                run_one(position, source)
                for position, source in enumerate(sources, start=1)
            )
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", default="runs/smoke_5")
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-interval-seconds", type=float, default=2.0)
    parser.add_argument("--source-concurrency", type=int, default=1)
    parser.add_argument("--max-concurrent-calls", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.min_interval_seconds < 0:
        parser.error("--min-interval-seconds must be nonnegative")
    if args.source_concurrency < 1:
        parser.error("--source-concurrency must be positive")
    if args.max_concurrent_calls < 1:
        parser.error("--max-concurrent-calls must be positive")
    return args


def main() -> int:
    required = (
        "INF_API_KEY",
        "TUTOR_QWEN3_8B_BASE_URL",
        "TUTOR_QWEN3_1_7B_BASE_URL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print(f"missing environment variables: {missing}", file=sys.stderr)
        return 2
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
