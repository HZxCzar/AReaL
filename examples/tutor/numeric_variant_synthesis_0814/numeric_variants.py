"""Pure helpers for mechanically safe numeric-only MATH variants."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from examples.tutor.core.math import (
    extract_ground_truth_answer,
    extract_math_answer,
    is_equiv,
)

NUMBER_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.IGNORECASE | re.DOTALL)
FINAL_JSON_RE = re.compile(
    r"<FINAL_JSON>\s*(.*?)\s*</FINAL_JSON>", re.IGNORECASE | re.DOTALL
)


def _inside(span_start: int, span_end: int, containers: list[tuple[int, int]]) -> bool:
    return any(start <= span_start and span_end <= end for start, end in containers)


def numeric_tokens(task: str) -> list[dict[str, Any]]:
    """Return stable, globally indexed numeric tokens and editability metadata."""
    asy_spans = [(match.start(), match.end()) for match in ASY_RE.finditer(task)]
    result: list[dict[str, Any]] = []
    for index, match in enumerate(NUMBER_RE.finditer(task)):
        prefix = task[max(0, match.start() - 4) : match.start()]
        reason = ""
        if _inside(match.start(), match.end(), asy_spans):
            reason = "inside_asymptote"
        elif re.search(r"(?:\^|_)\{?$", prefix):
            reason = "latex_exponent_or_subscript"
        elif re.search(r"(?:,\\!|,)$", prefix) and set(match.group(0)) <= {"0"}:
            reason = "latex_thousands_group_suffix"
        protected = bool(reason)
        left = max(0, match.start() - 36)
        right = min(len(task), match.end() + 36)
        result.append(
            {
                "token_index": index,
                "value": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "editable": not protected,
                "protection_reason": reason,
                "context": task[left:right].replace("\n", "\\n"),
            }
        )
    return result


def token_inventory(task: str) -> str:
    rows = []
    for token in numeric_tokens(task):
        status = "EDITABLE" if token["editable"] else "PROTECTED"
        rows.append(
            f"[{token['token_index']}] {status} value={token['value']!r} "
            f"context={token['context']!r}"
        )
    return "\n".join(rows) or "(no numeric literals)"


def nonnumeric_skeleton(task: str) -> str:
    return NUMBER_RE.sub("<NUM>", task)


def apply_numeric_edits(
    task: str, raw_edits: Any
) -> tuple[str, list[dict[str, Any]]]:
    """Apply indexed edits and prove that no nonnumeric byte changed."""
    if not isinstance(raw_edits, list) or not raw_edits:
        raise ValueError("edits must be a non-empty list")

    matches = list(NUMBER_RE.finditer(task))
    tokens = numeric_tokens(task)
    seen: set[int] = set()
    edits: list[dict[str, Any]] = []

    for raw in raw_edits:
        if not isinstance(raw, dict):
            raise ValueError("each edit must be a JSON object")
        try:
            index = int(raw["token_index"])
            old = str(raw["old"])
            new = str(raw["new"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each edit needs token_index, old, and new") from exc
        if index in seen:
            raise ValueError(f"duplicate token_index {index}")
        if index < 0 or index >= len(matches):
            raise ValueError(f"token_index {index} is out of range")
        if not tokens[index]["editable"]:
            raise ValueError(
                f"token_index {index} is protected: "
                f"{tokens[index]['protection_reason']}"
            )
        actual = matches[index].group(0)
        if old != actual:
            raise ValueError(
                f"old value {old!r} does not match token {index}={actual!r}"
            )
        if NUMBER_RE.fullmatch(new) is None:
            raise ValueError(f"new value is not one numeric literal: {new!r}")
        if old == new:
            raise ValueError(f"token_index {index} was not changed")
        old_number = float(old.replace(",", ""))
        new_number = float(new.replace(",", ""))
        if old_number > 0:
            ratio = new_number / old_number
            if ratio < 0.5 or ratio > 2.0:
                raise ValueError(
                    f"token_index {index} changes scale too much: ratio={ratio:g}"
                )
        seen.add(index)
        edits.append(
            {
                "token_index": index,
                "old": old,
                "new": new,
                "role": str(raw.get("role", "")).strip(),
            }
        )

    candidate = task
    for edit in sorted(edits, key=lambda item: item["token_index"], reverse=True):
        match = matches[edit["token_index"]]
        candidate = candidate[: match.start()] + edit["new"] + candidate[match.end() :]

    report = mechanical_report(task, candidate, edits)
    if not all(report.values()):
        raise AssertionError(f"mechanical numeric-only proof failed: {report}")
    return candidate, sorted(edits, key=lambda item: item["token_index"])


def mechanical_report(
    source: str, candidate: str, edits: list[dict[str, Any]]
) -> dict[str, bool]:
    source_numbers = NUMBER_RE.findall(source)
    candidate_numbers = NUMBER_RE.findall(candidate)
    edited = {int(edit["token_index"]): str(edit["new"]) for edit in edits}
    expected_numbers = [
        edited.get(index, value) for index, value in enumerate(source_numbers)
    ]
    return {
        "task_changed": candidate != source,
        "nonnumeric_skeleton_exact": (
            nonnumeric_skeleton(candidate) == nonnumeric_skeleton(source)
        ),
        "numeric_token_count_exact": len(candidate_numbers) == len(source_numbers),
        "numeric_sequence_matches_edits": candidate_numbers == expected_numbers,
        "asymptote_spans_exact": ASY_RE.findall(candidate) == ASY_RE.findall(source),
    }


def parse_tagged_json(text: str) -> dict[str, Any]:
    matches = FINAL_JSON_RE.findall(text or "")
    if not matches:
        raise ValueError("missing <FINAL_JSON>...</FINAL_JSON>")
    raw = matches[-1].strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(
            r"^" + re.escape(fence) + r"(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        raw = re.sub(r"\s*" + re.escape(fence) + r"$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
        value = json.loads(repaired)
    if not isinstance(value, dict):
        raise ValueError("final JSON must be an object")
    return value


def strict_audit_pass(audit: dict[str, Any]) -> bool:
    required = (
        "only_numeric_values_changed",
        "same_wording_units_target_constraints",
        "same_computation_graph",
        "same_knowledge_point",
        "structural_numbers_unchanged",
        "dependent_values_consistent",
        "well_posed",
        "similar_difficulty",
    )
    return bool(audit.get("pass")) and all(audit.get(key) is True for key in required)


def extracted_answer(output: str) -> str:
    return extract_math_answer(output or "").strip()


def target_answer(ground_truth: str) -> str:
    return extract_ground_truth_answer(str(ground_truth)).strip()


def answers_equivalent(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return bool(is_equiv(str(left), str(right)))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
