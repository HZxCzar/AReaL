#!/usr/bin/env python3
"""Run every saved teacher reply through every applicable binary gate."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from examples.common.openai_utils import AsyncLLMCaller, AuxModelConfig
from examples.tutor.core.callers import ApiAuxiliaryCaller
from examples.tutor.prompts import (
    PERSONALITY_GATE_V2_PREVIOUS_STUDENT_TEMPLATE,
    PERSONALITY_GATE_V2_SYSTEM_PROMPT,
    PERSONALITY_GATE_V2_USER_TEMPLATE,
)
from examples.tutor.workflow import (
    _parse_personality_gate_reply,
    load_personality_prompts,
)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            key = tuple(str(payload.get(field) or "") for field in key_fields)
            if not all(key):
                raise ValueError(
                    f"{path}:{line_number}: missing key field(s) {key_fields}"
                )
            latest[key] = payload
    return list(latest.values())


def _resolve_trace_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _load_all_teacher_turns(
    run_dir: Path,
    *,
    expected_episodes_per_cell: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    teacher = str(run_manifest.get("teacher") or "").strip()
    students = list(run_manifest.get("students") or [])
    if not teacher or not students:
        raise ValueError(f"invalid evaluation manifest: {run_manifest_path}")

    samples: list[dict[str, Any]] = []
    source_counts = Counter()
    seen_sample_ids: set[str] = set()
    for student_entry in students:
        student = str(student_entry.get("name") or "").strip()
        source_preference = str(
            student_entry.get("preference") or "none"
        ).strip()
        if not student:
            raise ValueError(f"student with no name in {run_manifest_path}")
        results_path = run_dir / "cells" / teacher / student / "results.jsonl"
        results = _load_latest_jsonl(results_path, key_fields=("key",))
        if len(results) != expected_episodes_per_cell:
            raise ValueError(
                f"{results_path}: expected {expected_episodes_per_cell} recorded "
                f"episodes, got {len(results)}"
            )
        source_counts["recorded_episode_count"] += len(results)
        results.sort(
            key=lambda item: (
                int(item.get("dataset_index", 0) or 0),
                str(item["key"]),
            )
        )
        for result in results:
            raw_trace_path = str(result.get("trace_path") or "").strip()
            if not raw_trace_path:
                source_counts["episode_without_trace_count"] += 1
                continue
            trace_path = _resolve_trace_path(run_dir, raw_trace_path)
            if not trace_path.is_file():
                raise ValueError(f"missing saved trace: {trace_path}")
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            turns = trace.get("turns") or []
            if not isinstance(turns, list):
                raise ValueError(f"trace turns are not a list: {trace_path}")
            task = str(trace.get("task") or "").strip()
            previous_real_student_message = ""
            for ordinal, turn in enumerate(turns, start=1):
                turn_idx = int(turn.get("turn_idx", ordinal) or ordinal)
                teacher_message = str(
                    turn.get("tutor_visible_output") or ""
                ).strip()
                if not teacher_message:
                    raise ValueError(
                        f"empty teacher output at turn {turn_idx}: {trace_path}"
                    )
                source_key = str(result["key"])
                sample_id = f"{student}:{source_key}:turn-{turn_idx}"
                if sample_id in seen_sample_ids:
                    raise ValueError(f"duplicate teacher-turn id: {sample_id}")
                seen_sample_ids.add(sample_id)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "source_preference": source_preference,
                        "source_student": student,
                        "source_key": source_key,
                        "item_id": str(result.get("item_id") or source_key),
                        "dataset_index": int(
                            result.get("dataset_index", 0) or 0
                        ),
                        "turn_idx": turn_idx,
                        "is_first_turn": turn_idx == 1,
                        "task": task,
                        "teacher_message": teacher_message,
                        "previous_real_student_message": (
                            previous_real_student_message or None
                        ),
                        "trace_path": str(trace_path),
                    }
                )
                source_counts["teacher_turn_count"] += 1
                if turn_idx == 1:
                    source_counts["first_turn_count"] += 1
                if not bool(turn.get("personality_gated")):
                    student_message = str(
                        turn.get("student_output") or ""
                    ).strip()
                    if student_message:
                        previous_real_student_message = student_message

    samples.sort(
        key=lambda item: (
            item["source_preference"],
            item["dataset_index"],
            item["source_key"],
            item["turn_idx"],
        )
    )
    if not samples:
        raise ValueError(f"no saved teacher turns found under {run_dir}")
    return samples, dict(source_counts)


def _gate_is_applicable(sample: dict[str, Any], gate: str) -> bool:
    return gate != "feedback" or bool(sample["previous_real_student_message"])


def _expected_decision_keys(
    samples: list[dict[str, Any]],
    gates: list[str],
) -> set[tuple[str, str]]:
    return {
        (str(sample["sample_id"]), gate)
        for sample in samples
        for gate in gates
        if _gate_is_applicable(sample, gate)
    }


def _build_manifest(
    *,
    run_dir: Path,
    prompts_path: Path,
    samples: list[dict[str, Any]],
    source_counts: dict[str, int],
    prompts: dict[str, dict[str, str]],
    gates: list[str],
    expected_episodes_per_cell: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_payload = {
        "system": PERSONALITY_GATE_V2_SYSTEM_PROMPT,
        "previous_student": PERSONALITY_GATE_V2_PREVIOUS_STUDENT_TEMPLATE,
        "user": PERSONALITY_GATE_V2_USER_TEMPLATE,
        "preferences": {name: prompts[name]["preference"] for name in gates},
    }
    source_payload = [
        {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "teacher_message": sample["teacher_message"],
            "previous_real_student_message": sample[
                "previous_real_student_message"
            ],
        }
        for sample in samples
    ]
    expected_decision_count = sum(
        _gate_is_applicable(sample, gate)
        for sample in samples
        for gate in gates
    )
    return {
        "audit": "all_teacher_turns_all_applicable_binary_gates",
        "run_dir": str(run_dir.resolve()),
        "prompts_path": str(prompts_path.resolve()),
        "expected_episodes_per_cell": expected_episodes_per_cell,
        "source": {
            **source_counts,
            "sample_count": len(samples),
            "sha256": _sha256(source_payload),
        },
        "prompts_sha256": _sha256(prompt_payload),
        "gates": gates,
        "expected_decision_count": expected_decision_count,
        "applicability": {
            "feedback": (
                "Applicable only when a previous real, non-scripted student "
                "message exists; in particular, first turns are N/A."
            ),
            "other_gates": "Applicable to every saved teacher reply.",
        },
        "model": args.model,
        "request": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "gate_retries": args.gate_retries,
            "enable_thinking": False,
            "lora_path": None,
        },
    }


def _prepare_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("created_at", None)
        if existing != manifest:
            raise ValueError(
                f"existing gate-audit manifest differs: {path}; "
                "use a new run directory"
            )
        return
    payload = dict(manifest)
    payload["created_at"] = datetime.now(UTC).isoformat()
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_samples(output_dir: Path, samples: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n"
        for sample in samples
    )
    _atomic_write(output_dir / "samples.jsonl", content)


def _slice_summary(
    *,
    samples: list[dict[str, Any]],
    gates: list[str],
    latest: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    expected_by_sample = {
        sample_id: [
            gate for gate in gates if _gate_is_applicable(sample, gate)
        ]
        for sample_id, sample in sample_by_id.items()
    }
    complete_sample_ids = [
        sample_id
        for sample_id, applicable in expected_by_sample.items()
        if all((sample_id, gate) in latest for gate in applicable)
    ]
    passed_by_sample = {
        sample_id: [
            gate
            for gate in expected_by_sample[sample_id]
            if latest[(sample_id, gate)].get("passed") is True
        ]
        for sample_id in complete_sample_ids
    }

    gate_summaries: dict[str, Any] = {}
    exclusivity: dict[str, Any] = {}
    for gate in gates:
        applicable_ids = [
            sample_id
            for sample_id, applicable in expected_by_sample.items()
            if gate in applicable
        ]
        decisions = [
            latest[(sample_id, gate)]
            for sample_id in applicable_ids
            if (sample_id, gate) in latest
        ]
        passed_count = sum(item.get("passed") is True for item in decisions)
        error_count = sum(bool(item.get("error")) for item in decisions)
        gate_summaries[gate] = {
            "sample_count": len(samples),
            "not_applicable_count": len(samples) - len(applicable_ids),
            "applicable_count": len(applicable_ids),
            "recorded_count": len(decisions),
            "passed_count": passed_count,
            "fail_closed_error_count": error_count,
            "pass_rate": _rate(passed_count, len(decisions)),
            "valid_pass_rate": _rate(
                passed_count,
                len(decisions) - error_count,
            ),
        }

        passed_complete_ids = [
            sample_id
            for sample_id in complete_sample_ids
            if gate in expected_by_sample[sample_id]
            and gate in passed_by_sample[sample_id]
        ]
        exclusive_ids = [
            sample_id
            for sample_id in passed_complete_ids
            if len(passed_by_sample[sample_id]) == 1
        ]
        overlap_ids = [
            sample_id
            for sample_id in passed_complete_ids
            if len(passed_by_sample[sample_id]) > 1
        ]
        additional_passes = sum(
            len(passed_by_sample[sample_id]) - 1
            for sample_id in passed_complete_ids
        )
        exclusivity[gate] = {
            "complete_passed_message_count": len(passed_complete_ids),
            "exclusive_count": len(exclusive_ids),
            "exclusive_share_of_passes": _rate(
                len(exclusive_ids), len(passed_complete_ids)
            ),
            "overlap_count": len(overlap_ids),
            "overlap_share_of_passes": _rate(
                len(overlap_ids), len(passed_complete_ids)
            ),
            "mean_additional_passed_gates": _rate(
                additional_passes, len(passed_complete_ids)
            ),
        }

    conditional_rates: dict[str, dict[str, float | None]] = {}
    conditional_counts: dict[str, dict[str, int]] = {}
    conditional_denominators: dict[str, dict[str, int]] = {}
    for source_gate in gates:
        conditional_rates[source_gate] = {}
        conditional_counts[source_gate] = {}
        conditional_denominators[source_gate] = {}
        for target_gate in gates:
            eligible = [
                sample_id
                for sample_id in complete_sample_ids
                if source_gate in expected_by_sample[sample_id]
                and target_gate in expected_by_sample[sample_id]
                and source_gate in passed_by_sample[sample_id]
            ]
            both_count = sum(
                target_gate in passed_by_sample[sample_id]
                for sample_id in eligible
            )
            conditional_counts[source_gate][target_gate] = both_count
            conditional_denominators[source_gate][target_gate] = len(eligible)
            conditional_rates[source_gate][target_gate] = _rate(
                both_count, len(eligible)
            )

    pass_count_histogram = Counter(
        len(passed_by_sample[sample_id]) for sample_id in complete_sample_ids
    )
    combination_counts = Counter(
        "+".join(passed_by_sample[sample_id]) or "NONE"
        for sample_id in complete_sample_ids
    )
    complete_count = len(complete_sample_ids)
    multi_pass_ids = [
        sample_id
        for sample_id in complete_sample_ids
        if len(passed_by_sample[sample_id]) > 1
    ]
    multi_pass_examples = [
        {
            "sample_id": sample_id,
            "source_preference": sample_by_id[sample_id]["source_preference"],
            "turn_idx": sample_by_id[sample_id]["turn_idx"],
            "passed_gates": passed_by_sample[sample_id],
            "teacher_message": sample_by_id[sample_id]["teacher_message"],
            "trace_path": sample_by_id[sample_id]["trace_path"],
        }
        for sample_id in multi_pass_ids
    ]
    return {
        "sample_count": len(samples),
        "complete_sample_count": complete_count,
        "gates": gate_summaries,
        "exclusivity": exclusivity,
        "pass_count_histogram": {
            str(count): pass_count_histogram[count]
            for count in range(len(gates) + 1)
        },
        "zero_pass_count": pass_count_histogram[0],
        "zero_pass_rate": _rate(pass_count_histogram[0], complete_count),
        "exactly_one_pass_count": pass_count_histogram[1],
        "exactly_one_pass_rate": _rate(pass_count_histogram[1], complete_count),
        "multi_pass_count": len(multi_pass_ids),
        "multi_pass_rate": _rate(len(multi_pass_ids), complete_count),
        "combination_counts": dict(
            sorted(combination_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "conditional_overlap": {
            "description": "P(column gate PASS | row gate PASS)",
            "rates": conditional_rates,
            "both_pass_counts": conditional_counts,
            "denominators": conditional_denominators,
        },
        "multi_pass_examples": multi_pass_examples,
    }


def _build_summary(
    *,
    samples: list[dict[str, Any]],
    gates: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_keys = _expected_decision_keys(samples, gates)
    latest = {
        (str(result["sample_id"]), str(result["gate"])): result
        for result in results
        if (str(result["sample_id"]), str(result["gate"])) in expected_keys
    }
    all_turns = _slice_summary(samples=samples, gates=gates, latest=latest)
    first_turn_samples = [sample for sample in samples if sample["is_first_turn"]]
    first_turns = _slice_summary(
        samples=first_turn_samples,
        gates=gates,
        latest=latest,
    )
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "complete": len(latest) == len(expected_keys),
        "teacher_reply_count": len(samples),
        "first_turn_reply_count": len(first_turn_samples),
        "expected_decision_count": len(expected_keys),
        "recorded_decision_count": len(latest),
        "progress": _rate(len(latest), len(expected_keys)),
        "gates": gates,
        "applicability": {
            "feedback": (
                "N/A without a previous real, non-scripted student message; "
                "therefore N/A on every first turn."
            )
        },
        "slices": {
            "all_turns": all_turns,
            "first_turns": first_turns,
        },
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _matrix_tsv(
    matrix: dict[str, dict[str, Any]],
    gates: list[str],
    *,
    formatter: Any,
) -> str:
    rows = ["source_gate\\target_gate\t" + "\t".join(gates)]
    for source_gate in gates:
        rows.append(
            "\t".join(
                [source_gate]
                + [formatter(matrix[source_gate][target]) for target in gates]
            )
        )
    return "\n".join(rows) + "\n"


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    _atomic_write(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    gates = list(summary["gates"])
    gate_rows = [
        "scope\tgate\tapplicable\tnot_applicable\trecorded\tpassed\t"
        "errors\tpass_rate\tvalid_pass_rate"
    ]
    exclusivity_rows = [
        "scope\tgate\tpassed_complete\texclusive\texclusive_share\t"
        "overlap\toverlap_share\tmean_additional_passed_gates"
    ]
    combination_rows = ["scope\tpassed_gate_combination\tcount\trate"]
    for scope, slice_summary in summary["slices"].items():
        for gate in gates:
            gate_item = slice_summary["gates"][gate]
            gate_rows.append(
                "\t".join(
                    (
                        scope,
                        gate,
                        str(gate_item["applicable_count"]),
                        str(gate_item["not_applicable_count"]),
                        str(gate_item["recorded_count"]),
                        str(gate_item["passed_count"]),
                        str(gate_item["fail_closed_error_count"]),
                        _fmt(gate_item["pass_rate"]),
                        _fmt(gate_item["valid_pass_rate"]),
                    )
                )
            )
            exclusive_item = slice_summary["exclusivity"][gate]
            exclusivity_rows.append(
                "\t".join(
                    (
                        scope,
                        gate,
                        str(exclusive_item["complete_passed_message_count"]),
                        str(exclusive_item["exclusive_count"]),
                        _fmt(exclusive_item["exclusive_share_of_passes"]),
                        str(exclusive_item["overlap_count"]),
                        _fmt(exclusive_item["overlap_share_of_passes"]),
                        _fmt(exclusive_item["mean_additional_passed_gates"]),
                    )
                )
            )
        complete_count = int(slice_summary["complete_sample_count"])
        for combination, count in slice_summary["combination_counts"].items():
            combination_rows.append(
                "\t".join(
                    (
                        scope,
                        combination,
                        str(count),
                        _fmt(_rate(count, complete_count)),
                    )
                )
            )

        conditional = slice_summary["conditional_overlap"]
        stem = "all_turns" if scope == "all_turns" else "first_turns"
        _atomic_write(
            output_dir / f"{stem}_overlap_matrix.tsv",
            _matrix_tsv(conditional["rates"], gates, formatter=_fmt),
        )
        _atomic_write(
            output_dir / f"{stem}_overlap_counts.tsv",
            _matrix_tsv(
                conditional["both_pass_counts"],
                gates,
                formatter=lambda value: str(int(value)),
            ),
        )
        _atomic_write(
            output_dir / f"{stem}_overlap_denominators.tsv",
            _matrix_tsv(
                conditional["denominators"],
                gates,
                formatter=lambda value: str(int(value)),
            ),
        )

    _atomic_write(output_dir / "gate_rates.tsv", "\n".join(gate_rows) + "\n")
    _atomic_write(
        output_dir / "exclusivity.tsv",
        "\n".join(exclusivity_rows) + "\n",
    )
    _atomic_write(
        output_dir / "combinations.tsv",
        "\n".join(combination_rows) + "\n",
    )
    examples = summary["slices"]["all_turns"]["multi_pass_examples"]
    _atomic_write(
        output_dir / "multi_pass_examples.jsonl",
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in examples
        ),
    )


async def _audit_one(
    *,
    sample: dict[str, Any],
    gate: str,
    preference: str,
    caller: ApiAuxiliaryCaller,
    gate_retries: int,
) -> dict[str, Any]:
    previous_student_context = ""
    if gate == "feedback":
        previous_student_context = (
            PERSONALITY_GATE_V2_PREVIOUS_STUDENT_TEMPLATE.format(
                previous_student_message=sample[
                    "previous_real_student_message"
                ]
            )
        )
    system_prompt = PERSONALITY_GATE_V2_SYSTEM_PROMPT.format(task=sample["task"])
    user_prompt = PERSONALITY_GATE_V2_USER_TEMPLATE.format(
        preference=preference,
        previous_student_context=previous_student_context,
        teacher_message=sample["teacher_message"],
    )
    raw_output = ""
    last_error = "personality gate produced no verdict."
    reason = ""
    for attempt in range(1, gate_retries + 1):
        result = await caller.call_text(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            rid_prefix=f"overlap-{sample['sample_id']}-{gate}-{attempt}",
        )
        raw_output = result.raw_text or result.text
        if result.error:
            last_error = str(result.error)
            continue
        passed, reason, parse_error = _parse_personality_gate_reply(result.text)
        if parse_error:
            last_error = parse_error
            continue
        return {
            "sample_id": sample["sample_id"],
            "source_preference": sample["source_preference"],
            "source_key": sample["source_key"],
            "turn_idx": sample["turn_idx"],
            "gate": gate,
            "passed": passed,
            "reason": reason,
            "raw_output": raw_output,
            "error": None,
            "attempts": attempt,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    return {
        "sample_id": sample["sample_id"],
        "source_preference": sample["source_preference"],
        "source_key": sample["source_key"],
        "turn_idx": sample["turn_idx"],
        "gate": gate,
        "passed": False,
        "reason": "",
        "raw_output": raw_output,
        "error": last_error,
        "attempts": gate_retries,
        "completed_at": datetime.now(UTC).isoformat(),
    }


async def _run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, source_counts = _load_all_teacher_turns(
        run_dir,
        expected_episodes_per_cell=args.expected_episodes_per_cell,
    )
    prompts = load_personality_prompts(str(prompts_path))
    gates = list(prompts)
    if not gates:
        raise ValueError("no binary personality gates were found")
    manifest = _build_manifest(
        run_dir=run_dir,
        prompts_path=prompts_path,
        samples=samples,
        source_counts=source_counts,
        prompts=prompts,
        gates=gates,
        expected_episodes_per_cell=args.expected_episodes_per_cell,
        args=args,
    )
    _prepare_manifest(output_dir, manifest)
    _write_samples(output_dir, samples)

    results_path = output_dir / "results.jsonl"
    results = _load_latest_jsonl(
        results_path,
        key_fields=("sample_id", "gate"),
    )
    expected_keys = _expected_decision_keys(samples, gates)
    unexpected_keys = {
        (str(result["sample_id"]), str(result["gate"])) for result in results
    } - expected_keys
    if unexpected_keys:
        raise ValueError(
            f"{results_path} contains {len(unexpected_keys)} unexpected decisions"
        )
    completed_keys = {
        (str(result["sample_id"]), str(result["gate"])) for result in results
    }
    pending = [
        (sample, gate)
        for sample in samples
        for gate in gates
        if _gate_is_applicable(sample, gate)
        and (str(sample["sample_id"]), gate) not in completed_keys
    ]
    summary = _build_summary(samples=samples, gates=gates, results=results)
    _write_summary(output_dir, summary)
    if not pending:
        print(
            f"[overlap] already complete: {summary['recorded_decision_count']}/"
            f"{summary['expected_decision_count']}"
        )
        return 0

    request_params = {
        "seed": args.seed,
        "extra_headers": {
            "x-inspire-inference-key": "tutor-eval-qwen8b-gate-overlap-v2"
        },
        "extra_body": {
            "top_k": args.top_k,
            "min_p": args.min_p,
            "chat_template_kwargs": {"enable_thinking": False},
        },
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

    def fill_in_flight() -> None:
        while len(in_flight) < args.concurrency:
            try:
                sample, gate = next(pending_iter)
            except StopIteration:
                return
            in_flight.add(
                asyncio.create_task(
                    _audit_one(
                        sample=sample,
                        gate=gate,
                        preference=prompts[gate]["preference"],
                        caller=caller,
                        gate_retries=args.gate_retries,
                    )
                )
            )

    fill_in_flight()
    completed = 0
    try:
        with results_path.open("a", encoding="utf-8", buffering=1) as destination:
            while in_flight:
                done, in_flight = await asyncio.wait(
                    in_flight,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    result = task.result()
                    destination.write(
                        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    destination.flush()
                    results.append(result)
                    completed += 1
                fill_in_flight()
                if (
                    completed == len(done)
                    or completed % args.summary_every < len(done)
                    or completed == len(pending)
                ):
                    summary = _build_summary(
                        samples=samples,
                        gates=gates,
                        results=results,
                    )
                    _write_summary(output_dir, summary)
                    print(
                        f"[overlap] completed {completed}/{len(pending)} pending "
                        f"({summary['recorded_decision_count']}/"
                        f"{summary['expected_decision_count']} total)"
                    )
    finally:
        for task in in_flight:
            task.cancel()
        client = getattr(llm_caller, "_client", None)
        if client is not None:
            await client.close()

    summary = _build_summary(samples=samples, gates=gates, results=results)
    _write_summary(output_dir, summary)
    if not summary["complete"]:
        raise RuntimeError("gate overlap audit finished without complete coverage")
    print(f"[overlap] gate rates: {output_dir / 'gate_rates.tsv'}")
    print(f"[overlap] exclusivity: {output_dir / 'exclusivity.tsv'}")
    print(
        f"[overlap] conditional matrix: "
        f"{output_dir / 'all_turns_overlap_matrix.tsv'}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes-per-cell", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--gate-retries", type=int, default=3)
    parser.add_argument("--summary-every", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    args = parser.parse_args()
    if args.expected_episodes_per_cell < 1:
        parser.error("--expected-episodes-per-cell must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.gate_retries < 1:
        parser.error("--gate-retries must be positive")
    if args.summary_every < 1:
        parser.error("--summary-every must be positive")
    return args


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
