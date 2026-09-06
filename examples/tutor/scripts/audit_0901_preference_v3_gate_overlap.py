#!/usr/bin/env python3
"""Cross-score saved 0901 teacher replies with every V3 binary preference gate."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig
from examples.tutor.core.callers import ApiAuxiliaryCaller
from examples.tutor.prompts import (
    PERSONALITY_GATE_V3_NO_LAST_STUDENT_MESSAGE,
    PERSONALITY_GATE_V3_SYSTEM_PROMPT,
    PERSONALITY_GATE_V3_USER_TEMPLATE,
)
from examples.tutor.workflow import (
    _parse_personality_gate_reply,
    load_personality_prompts,
)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_latest_jsonl(
    path: Path,
    *,
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            key = tuple(str(item.get(field) or "") for field in key_fields)
            if not all(key):
                raise ValueError(
                    f"{path}:{line_number}: missing key field(s) {key_fields}"
                )
            latest[key] = item
    return list(latest.values())


def _resolve_trace_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (run_dir / path).resolve()


def _stable_sample(
    samples: list[dict[str, Any]],
    *,
    maximum_per_cell: int,
    seed: int,
    stratify_previous_real_student: bool,
) -> list[dict[str, Any]]:
    if maximum_per_cell <= 0:
        return samples
    by_cell: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        context_stratum = bool(sample["previous_real_student_message"])
        if not stratify_previous_real_student:
            context_stratum = False
        by_cell[
            (sample["teacher_key"], sample["source_student"], context_stratum)
        ].append(sample)
    selected: list[dict[str, Any]] = []
    for cell_samples in by_cell.values():
        ranked = sorted(
            cell_samples,
            key=lambda item: hashlib.sha256(
                f"{seed}:{item['sample_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:maximum_per_cell])
    return selected


def _load_samples(
    run_dir: Path,
    *,
    expected_episodes_per_cell: int,
    max_episodes_per_cell: int,
    include_no_previous_real_student: bool,
    max_samples_per_cell: int,
    seed: int,
    stratify_previous_real_student: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    teachers = manifest.get("teachers")
    students = manifest.get("students")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError(f"no teacher matrix in {manifest_path}")
    if not isinstance(students, list) or not students:
        raise ValueError(f"no student matrix in {manifest_path}")

    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for teacher in teachers:
        teacher_key = str(teacher.get("key") or "").strip()
        if not teacher_key:
            raise ValueError(f"teacher without key in {manifest_path}")
        for student in students:
            student_name = str(student.get("name") or "").strip()
            source_preference = str(student.get("preference") or "none").strip()
            if not student_name:
                raise ValueError(f"student without name in {manifest_path}")
            results_path = (
                run_dir / "cells" / teacher_key / student_name / "results.jsonl"
            )
            episodes = _load_latest_jsonl(results_path, key_fields=("key",))
            if (
                expected_episodes_per_cell > 0
                and len(episodes) != expected_episodes_per_cell
            ):
                raise ValueError(
                    f"{results_path}: expected {expected_episodes_per_cell} "
                    f"episodes, found {len(episodes)}"
                )
            counts["available_episode_count"] += len(episodes)
            if max_episodes_per_cell > 0:
                episodes = sorted(
                    episodes,
                    key=lambda item: hashlib.sha256(
                        (
                            f"{seed}:{teacher_key}:{student_name}:"
                            f"{item['key']}"
                        ).encode()
                    ).hexdigest(),
                )[:max_episodes_per_cell]
            counts["episode_count"] += len(episodes)
            for episode in sorted(
                episodes,
                key=lambda item: (
                    int(item.get("dataset_index", 0) or 0),
                    str(item["key"]),
                ),
            ):
                raw_trace_path = str(episode.get("trace_path") or "").strip()
                if not raw_trace_path:
                    counts["episode_without_trace_count"] += 1
                    continue
                trace_path = _resolve_trace_path(run_dir, raw_trace_path)
                if not trace_path.is_file():
                    raise ValueError(f"missing trace: {trace_path}")
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                turns = trace.get("turns") or []
                if not isinstance(turns, list):
                    raise ValueError(f"turns are not a list: {trace_path}")
                task = str(trace.get("task") or "").strip()
                previous_real_student_message = ""
                for ordinal, turn in enumerate(turns, start=1):
                    counts["saved_turn_count"] += 1
                    turn_idx = int(turn.get("turn_idx", ordinal) or ordinal)
                    teacher_message = str(
                        turn.get("tutor_visible_output") or ""
                    ).strip()
                    skip_reason = ""
                    if not teacher_message or bool(turn.get("teacher_ended")):
                        skip_reason = "empty_or_teacher_end"
                    elif bool(turn.get("tutor_format_error")):
                        skip_reason = "format_error"
                    elif bool(turn.get("leaked")) or bool(turn.get("leak_masked")):
                        skip_reason = "leak"
                    elif (
                        not previous_real_student_message
                        and not include_no_previous_real_student
                    ):
                        skip_reason = "no_previous_real_student"

                    if skip_reason:
                        counts[f"excluded_{skip_reason}_count"] += 1
                    else:
                        source_key = str(episode["key"])
                        sample_id = (
                            f"{teacher_key}:{student_name}:{source_key}:turn-{turn_idx}"
                        )
                        if sample_id in seen_ids:
                            raise ValueError(f"duplicate sample id: {sample_id}")
                        seen_ids.add(sample_id)
                        samples.append(
                            {
                                "sample_id": sample_id,
                                "teacher_key": teacher_key,
                                "teacher_trial": str(teacher.get("trial") or ""),
                                "source_student": student_name,
                                "source_preference": source_preference,
                                "source_key": source_key,
                                "item_id": str(
                                    episode.get("item_id") or source_key
                                ),
                                "dataset_index": int(
                                    episode.get("dataset_index", 0) or 0
                                ),
                                "turn_idx": turn_idx,
                                "task": task,
                                "previous_real_student_message": (
                                    previous_real_student_message or None
                                ),
                                "teacher_message": teacher_message,
                                "original_gate_passed": (
                                    (turn.get("personality_gate_result") or {}).get(
                                        "passed"
                                    )
                                ),
                                "original_personality_gated": bool(
                                    turn.get("personality_gated")
                                ),
                                "teacher_exact_repeat": bool(
                                    turn.get("teacher_exact_repeat")
                                ),
                                "trace_path": str(trace_path),
                            }
                        )
                        counts["eligible_turn_count"] += 1

                    # This mirrors training/evaluation: scripted replies following
                    # a leak or failed preference gate never become student context.
                    if not bool(turn.get("personality_gated")) and not bool(
                        turn.get("leak_masked")
                    ):
                        student_message = str(
                            turn.get("student_output") or ""
                        ).strip()
                        if student_message:
                            previous_real_student_message = student_message

    samples = _stable_sample(
        samples,
        maximum_per_cell=max_samples_per_cell,
        seed=seed,
        stratify_previous_real_student=stratify_previous_real_student,
    )
    samples.sort(
        key=lambda item: (
            item["teacher_key"],
            item["source_student"],
            item["dataset_index"],
            item["source_key"],
            item["turn_idx"],
        )
    )
    if not samples:
        raise ValueError("no eligible saved teacher replies")
    counts["selected_sample_count"] = len(samples)
    return samples, {
        "counts": dict(counts),
        "teachers": teachers,
        "students": students,
        "source_manifest": str(manifest_path.resolve()),
    }


def _decision_keys(
    samples: list[dict[str, Any]], gates: list[str]
) -> set[tuple[str, str]]:
    return {
        (str(sample["sample_id"]), gate)
        for sample in samples
        for gate in gates
    }


def _build_manifest(
    *,
    run_dir: Path,
    prompts_path: Path,
    samples: list[dict[str, Any]],
    source: dict[str, Any],
    prompts: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gates = list(prompts)
    sample_fingerprint = [
        {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "previous_real_student_message": sample[
                "previous_real_student_message"
            ],
            "teacher_message": sample["teacher_message"],
        }
        for sample in samples
    ]
    prompt_fingerprint = {
        "system": PERSONALITY_GATE_V3_SYSTEM_PROMPT,
        "no_previous": PERSONALITY_GATE_V3_NO_LAST_STUDENT_MESSAGE,
        "user": PERSONALITY_GATE_V3_USER_TEMPLATE,
        "preferences": {
            gate: prompts[gate]["preference"] for gate in gates
        },
    }
    return {
        "audit": "same_saved_teacher_reply_cross_scored_by_all_v3_binary_gates",
        "run_dir": str(run_dir.resolve()),
        "prompts_path": str(prompts_path.resolve()),
        "gates": gates,
        "sample_count": len(samples),
        "expected_decision_count": len(samples) * len(gates),
        "samples_sha256": _sha256(sample_fingerprint),
        "prompts_sha256": _sha256(prompt_fingerprint),
        "source": source,
        "selection": {
            "only_nonempty_non_end_non_format_non_leak_teacher_replies": True,
            "include_no_previous_real_student": (
                args.include_no_previous_real_student
            ),
            "max_episodes_per_teacher_student_cell": (
                args.max_episodes_per_cell
            ),
            "max_samples_per_teacher_student_cell": args.max_samples_per_cell,
            "stratify_previous_real_student": (
                args.stratify_previous_real_student
            ),
            "selection_seed": args.selection_seed,
        },
        "judge": {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.judge_seed,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "gate_retries": args.gate_retries,
            "enable_thinking": False,
        },
    }


def _prepare_output(
    output_dir: Path,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    comparable = dict(manifest)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing.pop("created_at", None)
        if existing != comparable:
            raise ValueError(
                f"existing audit differs: {manifest_path}; use another output dir"
            )
    else:
        payload = dict(manifest)
        payload["created_at"] = datetime.now(UTC).isoformat()
        _atomic_write(
            manifest_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
    _atomic_write(
        output_dir / "samples.jsonl",
        "".join(
            json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
            for sample in samples
        ),
    )


async def _score_one(
    *,
    sample: dict[str, Any],
    gate: str,
    preference: str,
    caller: ApiAuxiliaryCaller,
    gate_retries: int,
) -> dict[str, Any]:
    system_prompt = PERSONALITY_GATE_V3_SYSTEM_PROMPT
    user_prompt = PERSONALITY_GATE_V3_USER_TEMPLATE.format(
        preference=preference,
        task=sample["task"],
        last_student_message=(
            sample["previous_real_student_message"]
            or PERSONALITY_GATE_V3_NO_LAST_STUDENT_MESSAGE
        ),
        teacher_message=sample["teacher_message"],
    )
    raw_output = ""
    last_error = "personality gate produced no verdict"
    reason = ""
    attempts = 0
    try:
        for attempts in range(1, gate_retries + 1):
            request_id = hashlib.sha256(
                f"{sample['sample_id']}:{gate}:{attempts}".encode()
            ).hexdigest()[:20]
            result = await caller.call_text(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                rid_prefix=f"v3-overlap-{request_id}",
            )
            raw_output = result.raw_text or result.text
            if result.error:
                last_error = str(result.error)
                continue
            passed, reason, parse_error = _parse_personality_gate_reply(
                result.text
            )
            if parse_error:
                last_error = parse_error
                continue
            return {
                "sample_id": sample["sample_id"],
                "teacher_key": sample["teacher_key"],
                "source_preference": sample["source_preference"],
                "source_student": sample["source_student"],
                "source_key": sample["source_key"],
                "turn_idx": sample["turn_idx"],
                "gate": gate,
                "passed": passed,
                "reason": reason,
                "raw_output": raw_output,
                "error": None,
                "attempts": attempts,
                "completed_at": datetime.now(UTC).isoformat(),
            }
    except Exception as exc:  # Preserve resumability across endpoint failures.
        last_error = f"{type(exc).__name__}: {exc}"
    return {
        "sample_id": sample["sample_id"],
        "teacher_key": sample["teacher_key"],
        "source_preference": sample["source_preference"],
        "source_student": sample["source_student"],
        "source_key": sample["source_key"],
        "turn_idx": sample["turn_idx"],
        "gate": gate,
        "passed": False,
        "reason": reason,
        "raw_output": raw_output,
        "error": last_error,
        "attempts": attempts,
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _summarize_scope(
    samples: list[dict[str, Any]],
    gates: list[str],
    latest: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    complete_ids: list[str] = []
    valid_ids: list[str] = []
    error_decision_count = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        decisions = [latest.get((sample_id, gate)) for gate in gates]
        if all(decision is not None for decision in decisions):
            complete_ids.append(sample_id)
            errors = sum(bool(decision.get("error")) for decision in decisions)
            error_decision_count += errors
            if errors == 0:
                valid_ids.append(sample_id)

    passed = {
        sample_id: {
            gate
            for gate in gates
            if latest[(sample_id, gate)].get("passed") is True
        }
        for sample_id in valid_ids
    }
    gate_rates: dict[str, Any] = {}
    for gate in gates:
        passed_count = sum(gate in passed[sample_id] for sample_id in valid_ids)
        exclusive_count = sum(
            passed[sample_id] == {gate} for sample_id in valid_ids
        )
        gate_rates[gate] = {
            "passed_count": passed_count,
            "pass_rate": _rate(passed_count, len(valid_ids)),
            "exclusive_count": exclusive_count,
            "exclusive_share_of_gate_passes": _rate(
                exclusive_count, passed_count
            ),
        }

    conditional: dict[str, dict[str, float | None]] = {}
    jaccard: dict[str, dict[str, float | None]] = {}
    pair_counts: dict[str, dict[str, Any]] = {}
    for row_gate in gates:
        conditional[row_gate] = {}
        jaccard[row_gate] = {}
        pair_counts[row_gate] = {}
        for column_gate in gates:
            row_ids = {
                sample_id
                for sample_id in valid_ids
                if row_gate in passed[sample_id]
            }
            column_ids = {
                sample_id
                for sample_id in valid_ids
                if column_gate in passed[sample_id]
            }
            intersection = len(row_ids & column_ids)
            union = len(row_ids | column_ids)
            conditional[row_gate][column_gate] = _rate(
                intersection, len(row_ids)
            )
            jaccard[row_gate][column_gate] = _rate(intersection, union)
            pair_counts[row_gate][column_gate] = {
                "both_pass": intersection,
                "row_pass": len(row_ids),
                "column_pass": len(column_ids),
                "either_pass": union,
            }

    histogram = Counter(len(passed[sample_id]) for sample_id in valid_ids)
    combinations = Counter(
        "+".join(gate for gate in gates if gate in passed[sample_id]) or "NONE"
        for sample_id in valid_ids
    )
    samples_by_id = {
        str(sample["sample_id"]): sample for sample in samples
    }
    original_pass_transfer: dict[str, dict[str, Any]] = {}
    for source_gate in gates:
        source_ids = [
            sample_id
            for sample_id in valid_ids
            if samples_by_id[sample_id]["source_preference"] == source_gate
            and samples_by_id[sample_id]["original_gate_passed"] is True
        ]
        original_pass_transfer[source_gate] = {
            "original_pass_count": len(source_ids),
            "rates": {
                target_gate: _rate(
                    sum(target_gate in passed[sample_id] for sample_id in source_ids),
                    len(source_ids),
                )
                for target_gate in gates
            },
        }
    return {
        "selected_sample_count": len(samples),
        "complete_sample_count": len(complete_ids),
        "valid_all_six_gates_sample_count": len(valid_ids),
        "error_decision_count": error_decision_count,
        "gate_rates": gate_rates,
        "pass_count_histogram": {
            str(count): histogram[count] for count in range(len(gates) + 1)
        },
        "exactly_one_pass_rate": _rate(histogram[1], len(valid_ids)),
        "multi_pass_rate": _rate(
            sum(histogram[count] for count in range(2, len(gates) + 1)),
            len(valid_ids),
        ),
        "conditional_overlap": {
            "description": "P(column gate PASS | row gate PASS)",
            "rates": conditional,
        },
        "positive_jaccard": {
            "description": "intersection(PASS sets) / union(PASS sets)",
            "rates": jaccard,
        },
        "pair_counts": pair_counts,
        "original_pass_transfer": {
            "description": (
                "Among replies whose saved source gate originally passed, "
                "fraction that passes each replayed target gate"
            ),
            "by_source_gate": original_pass_transfer,
        },
        "passed_gate_combination_counts": dict(
            sorted(combinations.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _build_summary(
    samples: list[dict[str, Any]],
    gates: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = _decision_keys(samples, gates)
    latest = {
        (str(item["sample_id"]), str(item["gate"])): item
        for item in results
        if (str(item["sample_id"]), str(item["gate"])) in expected
    }
    by_teacher: dict[str, Any] = {}
    for teacher_key in sorted({sample["teacher_key"] for sample in samples}):
        by_teacher[teacher_key] = _summarize_scope(
            [sample for sample in samples if sample["teacher_key"] == teacher_key],
            gates,
            latest,
        )
    by_context = {
        "no_previous_real_student": _summarize_scope(
            [
                sample
                for sample in samples
                if not sample["previous_real_student_message"]
            ],
            gates,
            latest,
        ),
        "post_real_student": _summarize_scope(
            [
                sample
                for sample in samples
                if sample["previous_real_student_message"]
            ],
            gates,
            latest,
        ),
    }
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "complete": len(latest) == len(expected),
        "sample_count": len(samples),
        "expected_decision_count": len(expected),
        "recorded_decision_count": len(latest),
        "progress": _rate(len(latest), len(expected)),
        "gates": gates,
        "overall": _summarize_scope(samples, gates, latest),
        "by_context": by_context,
        "by_teacher": by_teacher,
    }


def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _markdown_matrix(
    title: str,
    description: str,
    rates: dict[str, dict[str, float | None]],
    gates: list[str],
) -> list[str]:
    lines = [f"### {title}", "", description, ""]
    lines.append("| Gate | " + " | ".join(gates) + " |")
    lines.append("|---|" + "---:|" * len(gates))
    for row_gate in gates:
        lines.append(
            f"| {row_gate} | "
            + " | ".join(
                _fmt_rate(rates[row_gate][column_gate])
                for column_gate in gates
            )
            + " |"
        )
    lines.append("")
    return lines


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    gates = list(summary["gates"])
    lines = [
        "# V3 preference-gate overlap on identical saved teacher replies",
        "",
        f"Valid replies scored by all six gates: "
        f"{summary['overall']['valid_all_six_gates_sample_count']} / "
        f"{summary['sample_count']}.",
        "",
        "## Gate pass rate by teacher checkpoint",
        "",
        "| Teacher checkpoint | n | " + " | ".join(gates) + " |",
        "|---|---:|" + "---:|" * len(gates),
    ]
    scopes = {"ALL": summary["overall"], **summary["by_teacher"]}
    for name, scope in scopes.items():
        lines.append(
            f"| {name} | {scope['valid_all_six_gates_sample_count']} | "
            + " | ".join(
                _fmt_rate(scope["gate_rates"][gate]["pass_rate"])
                for gate in gates
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Overall overlap",
            "",
            f"Exactly one gate passes: "
            f"{_fmt_rate(summary['overall']['exactly_one_pass_rate'])}; "
            f"multiple gates pass: "
            f"{_fmt_rate(summary['overall']['multi_pass_rate'])}.",
            "",
        ]
    )
    histogram = summary["overall"]["pass_count_histogram"]
    valid_count = summary["overall"]["valid_all_six_gates_sample_count"]
    lines.extend(
        [
            "### Number of gates passed by one reply",
            "",
            "| Gates passed | Replies | Rate |",
            "|---:|---:|---:|",
        ]
    )
    for count in range(len(gates) + 1):
        value = int(histogram[str(count)])
        lines.append(f"| {count} | {value} | {_fmt_rate(_rate(value, valid_count))} |")
    lines.append("")
    lines += _markdown_matrix(
        "Conditional overlap",
        "Cell = P(column PASS | row PASS). This matrix is directional.",
        summary["overall"]["conditional_overlap"]["rates"],
        gates,
    )
    lines += _markdown_matrix(
        "Positive Jaccard overlap",
        "Cell = |row PASS ∩ column PASS| / |row PASS ∪ column PASS|.",
        summary["overall"]["positive_jaccard"]["rates"],
        gates,
    )
    original = summary["overall"]["original_pass_transfer"]
    lines.extend(
        [
            "### Transfer from saved original gate passes",
            "",
            (
                "Rows use the gate verdict saved in the original trajectory; "
                "cells are the replayed target-gate pass rates for those replies."
            ),
            "",
            "| Original gate | n | " + " | ".join(gates) + " |",
            "|---|---:|" + "---:|" * len(gates),
        ]
    )
    for source_gate in gates:
        row = original["by_source_gate"][source_gate]
        lines.append(
            f"| {source_gate} | {row['original_pass_count']} | "
            + " | ".join(
                _fmt_rate(row["rates"][target_gate]) for target_gate in gates
            )
            + " |"
        )
    lines.append("")
    context_titles = {
        "no_previous_real_student": "No previous real student reply",
        "post_real_student": "After a real student reply",
    }
    for context_key, context_title in context_titles.items():
        scope = summary["by_context"][context_key]
        context_n = scope["valid_all_six_gates_sample_count"]
        lines.extend(
            [
                f"## Context: {context_title}",
                "",
                f"Valid replies: {context_n}.",
                "",
                "### Number of gates passed by one reply",
                "",
                "| Gates passed | Replies | Rate |",
                "|---:|---:|---:|",
            ]
        )
        for count in range(len(gates) + 1):
            value = int(scope["pass_count_histogram"][str(count)])
            lines.append(
                f"| {count} | {value} | "
                f"{_fmt_rate(_rate(value, context_n))} |"
            )
        lines.append("")
        lines += _markdown_matrix(
            "Conditional overlap",
            "Cell = P(column PASS | row PASS).",
            scope["conditional_overlap"]["rates"],
            gates,
        )
        lines += _markdown_matrix(
            "Positive Jaccard overlap",
            "Cell = positive-set Jaccard overlap within this context slice.",
            scope["positive_jaccard"]["rates"],
            gates,
        )
        context_original = scope["original_pass_transfer"]
        lines.extend(
            [
                "### Transfer from saved original gate passes",
                "",
                "| Original gate | n | " + " | ".join(gates) + " |",
                "|---|---:|" + "---:|" * len(gates),
            ]
        )
        for source_gate in gates:
            row = context_original["by_source_gate"][source_gate]
            lines.append(
                f"| {source_gate} | {row['original_pass_count']} | "
                + " | ".join(
                    _fmt_rate(row["rates"][target_gate])
                    for target_gate in gates
                )
                + " |"
            )
        lines.append("")
    for teacher_key, scope in summary["by_teacher"].items():
        lines.extend([f"## Teacher: {teacher_key}", ""])
        lines += _markdown_matrix(
            "Conditional overlap",
            "Cell = P(column PASS | row PASS).",
            scope["conditional_overlap"]["rates"],
            gates,
        )
        lines += _markdown_matrix(
            "Positive Jaccard overlap",
            "Cell = positive-set Jaccard overlap.",
            scope["positive_jaccard"]["rates"],
            gates,
        )
    _atomic_write(output_dir / "overlap.md", "\n".join(lines) + "\n")


def _write_scored_samples(
    output_dir: Path,
    samples: list[dict[str, Any]],
    gates: list[str],
    results: list[dict[str, Any]],
    *,
    examples_per_combination: int,
) -> None:
    latest = {
        (str(item["sample_id"]), str(item["gate"])): item for item in results
    }
    scored: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if not all((sample_id, gate) in latest for gate in gates):
            continue
        judgments = {gate: latest[(sample_id, gate)] for gate in gates}
        item = dict(sample)
        item["passed_gates"] = [
            gate
            for gate in gates
            if judgments[gate].get("passed") is True
            and not judgments[gate].get("error")
        ]
        item["gate_judgments"] = {
            gate: {
                "passed": judgments[gate].get("passed"),
                "reason": judgments[gate].get("reason"),
                "error": judgments[gate].get("error"),
                "raw_output": judgments[gate].get("raw_output"),
            }
            for gate in gates
        }
        scored.append(item)
    _atomic_write(
        output_dir / "scored_teacher_replies.jsonl",
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in scored
        ),
    )

    by_combination: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        combination = tuple(item["passed_gates"])
        if len(combination) > 1:
            by_combination[combination].append(item)
    lines = ["# Raw multi-gate examples", ""]
    for combination, items in sorted(
        by_combination.items(), key=lambda pair: (-len(pair[0]), pair[0])
    ):
        lines.extend(
            [
                "## " + " + ".join(combination),
                "",
                f"Total: {len(items)}",
                "",
            ]
        )
        for item in items[:examples_per_combination]:
            lines.extend(
                [
                    f"### {item['sample_id']}",
                    "",
                    f"Teacher checkpoint: `{item['teacher_key']}`  ",
                    f"Source student preference: `{item['source_preference']}`  ",
                    f"Trace: `{item['trace_path']}`",
                    "",
                    "Previous real student message:",
                    "",
                    str(item["previous_real_student_message"]),
                    "",
                    "Teacher message:",
                    "",
                    str(item["teacher_message"]),
                    "",
                    "Gate judgments:",
                    "",
                ]
            )
            for gate in gates:
                judgment = item["gate_judgments"][gate]
                verdict = "PASS" if judgment["passed"] else "FAIL"
                if judgment["error"]:
                    verdict = "ERROR"
                lines.append(
                    f"- `{gate}`: {verdict} — {judgment['reason']}"
                )
            lines.append("")
    _atomic_write(
        output_dir / "multi_gate_examples.md", "\n".join(lines) + "\n"
    )


async def _run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (run_dir / "v3_gate_overlap").resolve()
    )
    cached_manifest_path = output_dir / "manifest.json"
    cached_samples_path = output_dir / "samples.jsonl"
    if cached_manifest_path.is_file() and cached_samples_path.is_file():
        print("[overlap] loading cached selected replies...", flush=True)
        cached_manifest = json.loads(
            cached_manifest_path.read_text(encoding="utf-8")
        )
        samples = _load_latest_jsonl(
            cached_samples_path, key_fields=("sample_id",)
        )
        source = cached_manifest["source"]
    else:
        print("[overlap] loading saved trajectories...", flush=True)
        samples, source = _load_samples(
            run_dir,
            expected_episodes_per_cell=args.expected_episodes_per_cell,
            max_episodes_per_cell=args.max_episodes_per_cell,
            include_no_previous_real_student=args.include_no_previous_real_student,
            max_samples_per_cell=args.max_samples_per_cell,
            seed=args.selection_seed,
            stratify_previous_real_student=(
                args.stratify_previous_real_student
            ),
        )
    prompts = load_personality_prompts(str(prompts_path))
    gates = list(prompts)
    if len(gates) != 6:
        raise ValueError(f"expected six V3 gates, found {gates}")
    manifest = _build_manifest(
        run_dir=run_dir,
        prompts_path=prompts_path,
        samples=samples,
        source=source,
        prompts=prompts,
        args=args,
    )
    _prepare_output(output_dir, manifest, samples)

    results_path = output_dir / "decisions.jsonl"
    results = _load_latest_jsonl(
        results_path, key_fields=("sample_id", "gate")
    )
    expected = _decision_keys(samples, gates)
    unexpected = {
        (str(item["sample_id"]), str(item["gate"])) for item in results
    } - expected
    if unexpected:
        raise ValueError(
            f"{results_path} contains {len(unexpected)} unexpected decisions"
        )
    latest = {
        (str(item["sample_id"]), str(item["gate"])): item for item in results
    }
    # Existing successful calls are immutable. Existing endpoint/parse errors are
    # retried when the same command is run again.
    pending = [
        (sample, gate)
        for sample in samples
        for gate in gates
        if (sample["sample_id"], gate) not in latest
        or bool(latest[(sample["sample_id"], gate)].get("error"))
    ]
    summary = _build_summary(samples, gates, results)
    _write_summary(output_dir, summary)
    print(
        f"[overlap] {len(samples)} identical replies x {len(gates)} gates; "
        f"{len(pending)} calls pending",
        flush=True,
    )
    if not pending:
        _write_scored_samples(
            output_dir,
            samples,
            gates,
            results,
            examples_per_combination=args.examples_per_combination,
        )
        print(f"[overlap] complete: {output_dir / 'overlap.md'}", flush=True)
        return 0

    request_params = {
        "seed": args.judge_seed,
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    if args.inference_key:
        request_params["extra_headers"] = {
            "x-inspire-inference-key": args.inference_key
        }
    llm_caller = AsyncLLMCaller(
        AuxModelConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_concurrency=args.concurrency,
            request_params=request_params,
        )
    )
    caller = ApiAuxiliaryCaller(llm_caller)
    pending_iter = iter(pending)
    in_flight: set[asyncio.Task[dict[str, Any]]] = set()

    def fill() -> None:
        while len(in_flight) < args.concurrency:
            try:
                sample, gate = next(pending_iter)
            except StopIteration:
                return
            in_flight.add(
                asyncio.create_task(
                    _score_one(
                        sample=sample,
                        gate=gate,
                        preference=prompts[gate]["preference"],
                        caller=caller,
                        gate_retries=args.gate_retries,
                    )
                )
            )

    completed_now = 0
    fill()
    try:
        with results_path.open("a", encoding="utf-8", buffering=1) as destination:
            while in_flight:
                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    item = task.result()
                    destination.write(
                        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    results.append(item)
                    completed_now += 1
                fill()
                if (
                    completed_now == len(done)
                    or completed_now % args.summary_every < len(done)
                    or completed_now == len(pending)
                ):
                    summary = _build_summary(samples, gates, results)
                    _write_summary(output_dir, summary)
                    print(
                        f"[overlap] {completed_now}/{len(pending)} new calls; "
                        f"{summary['recorded_decision_count']}/"
                        f"{summary['expected_decision_count']} recorded",
                        flush=True,
                    )
    finally:
        for task in in_flight:
            task.cancel()
        client = getattr(llm_caller, "_client", None)
        if client is not None:
            await client.close()

    summary = _build_summary(samples, gates, results)
    _write_summary(output_dir, summary)
    _write_scored_samples(
        output_dir,
        samples,
        gates,
        results,
        examples_per_combination=args.examples_per_combination,
    )
    failed = summary["overall"]["error_decision_count"]
    if failed:
        raise RuntimeError(
            f"{failed} gate calls still have errors; rerun the same command"
        )
    print(f"[overlap] complete: {output_dir / 'overlap.md'}", flush=True)
    print(
        f"[overlap] raw judgments: "
        f"{output_dir / 'scored_teacher_replies.jsonl'}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("examples/tutor/prompt_pools/personality_prompts_v3.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-episodes-per-cell", type=int, default=192)
    parser.add_argument(
        "--max-episodes-per-cell",
        type=int,
        default=0,
        help="Deterministic episode cap per teacher x student cell before reading traces.",
    )
    parser.add_argument(
        "--include-no-previous-real-student",
        action="store_true",
        help="Also score first/no-real-student turns; off by default because V3 auto-passes them.",
    )
    parser.add_argument(
        "--max-samples-per-cell",
        type=int,
        default=0,
        help="Deterministic cap per teacher x source-student cell; 0 uses all replies.",
    )
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--stratify-previous-real-student",
        action="store_true",
        help=(
            "Apply the per-cell sample cap separately to turns with and without "
            "a previous real student reply."
        ),
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("TUTOR_QWEN3_8B_BASE_URL", "")
    )
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--api-key", default=os.environ.get("INF_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--gate-retries", type=int, default=3)
    parser.add_argument("--summary-every", type=int, default=100)
    parser.add_argument("--examples-per-combination", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--inference-key", default="tutor-eval-qwen8b-v3-gate-overlap"
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or TUTOR_QWEN3_8B_BASE_URL is required")
    for name in (
        "expected_episodes_per_cell",
        "max_episodes_per_cell",
        "max_samples_per_cell",
        "selection_seed",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "concurrency",
        "gate_retries",
        "summary_every",
        "examples_per_combination",
        "max_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
