"""Filter all records using ONLY nonempty problem and valid dialogue fields."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import digest, write_json

SOURCES = {
    "standard": "mathdial_bridge.json",
    "hard": "mathdial_bridge_hard.json",
}


def exclusion_reasons(record):
    reasons = []
    problem = record.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        reasons.append("rule1_invalid_or_empty_problem")
    turns = record.get("dialog_history")
    if not isinstance(turns, list) or not all(
        isinstance(t, dict)
        and isinstance(t.get("user"), str)
        and t["user"] in {"Teacher", "Tutor", "Student"}
        and isinstance(t.get("text"), str)
        and bool(t["text"].strip())
        for t in turns
    ):
        reasons.append("rule2_invalid_dialogue_fields")
    return reasons


def prepare(source_dir):
    cases, excluded, sources, summary = [], [], {}, {}
    for split, filename in SOURCES.items():
        path = Path(source_dir) / filename
        raw = path.read_bytes()
        sources[split] = {
            "filename": filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        records = json.loads(raw)
        counts = Counter(total=len(records), included=0, excluded=0)
        for index, record in enumerate(records):
            metadata = {
                "id": f"{split}-{index:04d}-{digest(record)[:12]}",
                "source_split": split,
                "source_index": index,
                "source_record_sha256": digest(record),
            }
            reasons = exclusion_reasons(record)
            if reasons:
                excluded.append(
                    {**metadata, "reasons": reasons, "source_record": record}
                )
                counts["excluded"] += 1
                counts.update(reasons)
                continue
            # Normal benchmark input construction, not another filter rule.
            cases.append(
                {
                    **metadata,
                    "problem": record["problem"],
                    "history": record["dialog_history"][:-1],
                }
            )
            counts["included"] += 1
        summary[split] = dict(counts)
    data = {
        "schema_version": 2,
        "sources": sources,
        "cases": cases,
        "excluded": excluded,
        "summary": summary,
        "filter_rules": {
            "rule1_invalid_or_empty_problem": "problem must be a nonempty string after whitespace stripping",
            "rule2_invalid_dialogue_fields": "dialog_history must be a list; each turn must have a Teacher/Tutor/Student role and nonempty string text",
        },
        "protocol": "All records; only rules 1 and 2; no pending/manual review, sampling, deduplication, semantic or output-based filtering. Final source teacher reply withheld as standard input construction.",
    }
    data["fingerprint"] = digest(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; use a new dataset version")
    data = prepare(args.source_dir)
    write_json(args.output, data)
    print(json.dumps(data["summary"], indent=2))
    print(f"Included={len(data['cases'])}, excluded={len(data['excluded'])}")


if __name__ == "__main__":
    main()
