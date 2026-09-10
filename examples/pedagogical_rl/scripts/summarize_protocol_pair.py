"""Validate and summarize two full, matched PedagogicalRL-protocol evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

MODES = ("GUIDED", "ATTEMPTED")
ARMS = ("ours", "pedrl")
RULES = ("does_not_leak_answer", "follows_pedagogical_values")
METRICS = {
    "baseline": "initial_correct",
    "final": "final_correct",
    "raw_improvement": "improvement/raw",
    "leak_aware_improvement": "improvement/leak_aware",
    "leak": "native_leak/rejected",
    "style_rejected": "native_pedagogy/rejected",
    "format_error": "format_errors",
}


def load_arm(root: Path, arm: str) -> dict[str, dict[str, dict[str, Any]]]:
    modes: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in MODES}
    for path in sorted((root / arm / "eval").glob("*.json")):
        trace = json.loads(path.read_text())
        mode = trace.get("conversation_type")
        problem = trace.get("problem")
        if mode not in modes or trace.get("preference") != "none":
            raise ValueError(f"Unexpected mode/preference: {path}")
        if not isinstance(problem, str) or not problem:
            raise ValueError(f"Missing problem: {path}")
        if problem in modes[mode]:
            raise ValueError(f"Duplicate problem in {arm}/{mode}: {path}")
        for key in ("answer", "student_name", "native_judges", "final_solutions"):
            if key not in trace:
                raise ValueError(f"Missing {key}: {path}")
        for key in METRICS.values():
            value = trace.get("metrics", {}).get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing/nonfinite metric {key}: {path}")
        trace["trace_path"] = str(path.relative_to(root))
        modes[mode][problem] = trace
    for mode, traces in modes.items():
        if len(traces) != 528:
            raise ValueError(
                f"{arm}/{mode}: expected 528 unique traces, got {len(traces)}"
            )
    return modes


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics: Counter[str] = Counter()
    affected = []
    for trace in traces:
        problems = []
        format_failed = bool(trace["metrics"]["format_errors"])
        decisions = trace["native_judges"]
        for rule in RULES:
            selected = [d for d in decisions if d.get("rule") == rule]
            if len(selected) != 2:
                kind = "format_stopped" if format_failed else "unexpected"
                key = f"{kind}_judge_count/{rule}"
                diagnostics[key] += 1
                problems.append(key)
            for decision in selected:
                if decision.get("reasoning") == "max turns exceeded":
                    diagnostics["judge_fail_open_verdicts"] += 1
                    problems.append("judge_fail_open")
                if decision.get("decision") not in ("OK", "REJECT"):
                    diagnostics["invalid_judge_verdicts"] += 1
                    problems.append("invalid_judge_verdict")
        if len(trace["final_solutions"]) != 8:
            kind = "format_stopped" if format_failed else "unexpected"
            diagnostics[f"{kind}_retest_count"] += 1
            problems.append(f"{kind}_retest_count")
        if problems:
            affected.append(
                {"trace_path": trace["trace_path"], "issues": sorted(set(problems))}
            )
    return {
        "count": len(traces),
        "means": {
            label: fmean(trace["metrics"][key] for trace in traces)
            for label, key in METRICS.items()
        },
        "diagnostics": dict(diagnostics),
        "diagnostic_traces": affected,
    }


def compare(root: Path) -> dict[str, Any]:
    arms = {arm: load_arm(root, arm) for arm in ARMS}
    reference = arms["ours"]["GUIDED"]
    for arm, modes in arms.items():
        for mode, traces in modes.items():
            if traces.keys() != reference.keys():
                raise ValueError(f"Problem set mismatch: {arm}/{mode}")
            for problem, trace in traces.items():
                for field in ("answer", "student_name"):
                    # Answers match globally; names need only match across
                    # models within the same dialogue protocol.
                    expected = (
                        arms["ours"][mode][problem]
                        if field == "student_name"
                        else reference[problem]
                    )
                    if trace[field] != expected[field]:
                        raise ValueError(
                            f"{field} mismatch: {arm}/{mode}, {problem[:80]}"
                        )
    return {
        "protocol": "PedagogicalRL; none; GUIDED + ATTEMPTED; unified XML",
        "note": (
            "All 528 problems per mode are included, including format failures. "
            "Means are fractions; improvement is a difference of accuracies. "
            "Judges are diagnostic, not dialogue gates. Format errors can stop retests. "
            "Fail-open verdicts are detected from 'max turns exceeded'; individual "
            "failed judge attempts are not present in traces and cannot be counted."
        ),
        "arms": {
            arm: {
                **{
                    mode: summarize(list(traces.values()))
                    for mode, traces in modes.items()
                },
                "combined": summarize(
                    [trace for traces in modes.values() for trace in traces.values()]
                ),
            }
            for arm, modes in arms.items()
        },
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# PedagogicalRL protocol comparison",
        "",
        result["note"],
        "",
        "Accuracy/rates are percentages; improvements are percentage points.",
        "",
        "| Model | Mode | N | Baseline | Final | Raw improvement | Leak-aware improvement | Leak | Style rejected | Format error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, modes in result["arms"].items():
        for mode, summary in modes.items():
            values = " | ".join(f"{summary['means'][key] * 100:.2f}" for key in METRICS)
            lines.append(f"| {arm} | {mode} | {summary['count']} | {values} |")
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "No affected episodes are excluded; trace paths are listed in comparison.json.",
            "",
        ]
    )
    for arm, modes in result["arms"].items():
        lines.append(
            f"- {arm}: `{json.dumps(modes['combined']['diagnostics'], sort_keys=True)}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", type=Path)
    args = parser.parse_args()
    result = compare(args.root_dir)
    (args.root_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.root_dir / "comparison.md").write_text(markdown(result))


if __name__ == "__main__":
    main()
