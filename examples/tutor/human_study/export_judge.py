"""Export complete matched runs, keeping model identities out of human materials."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from .common import digest, load_dataset, read_json, write_json
from .eval import payload_for, result_from_response


def export(left, right, output, seed=42):
    left, right, output = Path(left), Path(right), Path(output)
    if output.exists():
        raise ValueError("Export exists; refusing to change a published A/B assignment")
    manifests = [read_json(p / "manifest.json") for p in [left, right]]
    if left.resolve() == right.resolve():
        raise ValueError("Two different runs required")
    if any(
        manifests[0][k] != manifests[1][k]
        for k in ["dataset_fingerprint", "study", "code_sha256"]
    ):
        raise ValueError(
            "Cannot compare different data, prompts, generation settings, or runner versions"
        )
    data = load_dataset(left / "dataset.json")
    if (
        data != load_dataset(right / "dataset.json")
        or data["fingerprint"] != manifests[0]["dataset_fingerprint"]
    ):
        raise ValueError("Dataset snapshot mismatch")
    rng = random.Random(seed)
    cases = list(data["cases"])
    rng.shuffle(cases)
    # Balance positions within each split; no dependence on response quality.
    positions = {}
    for split in sorted({c["source_split"] for c in cases}):
        members = [c for c in cases if c["source_split"] == split]
        assignments = [i % 2 for i in range(len(members))]
        rng.shuffle(assignments)
        positions.update({c["id"]: a for c, a in zip(members, assignments)})
    items, mapping = [], {}
    for index, case in enumerate(cases, 1):
        records = []
        for path, manifest in zip([left, right], manifests):
            record = read_json(path / "records" / case["id"] / "result.json")
            if record["input_sha256"] != digest(case) or record[
                "request"
            ] != payload_for(case, manifest["study"], manifest["model_config"]):
                raise ValueError("Result does not match its frozen input")
            original = result_from_response(case, record["request"], record["response"])
            if any(
                record[k] != original[k]
                for k in ["raw_response", "finish_reason", "flags"]
            ):
                raise ValueError(
                    "Result text or flags differ from saved provider response"
                )
            records.append(record)
        a = positions[case["id"]]
        item_id = f"Item-{index:04d}"
        history = "\n".join(f"{t['user']}: {t['text']}" for t in case["history"])
        items.append(
            {
                "id": item_id,
                "problem": case["problem"],
                "history": history,
                "A": records[a]["raw_response"],
                "B": records[1 - a]["raw_response"],
            }
        )
        mapping[item_id] = {
            "case_id": case["id"],
            "source_split": case["source_split"],
            "A_run": str([left, right][a].resolve()),
            "B_run": str([left, right][1 - a].resolve()),
            "A_flags": records[a]["flags"],
            "B_flags": records[1 - a]["flags"],
        }
    output.mkdir(parents=True)
    human, private = output / "human", output / "private"
    human.mkdir()
    private.mkdir(mode=0o700)
    write_json(private / "mapping.json", mapping)
    write_json(
        private / "manifest.json",
        {
            "seed": seed,
            "runs": manifests,
            "items_sha256": digest(items),
            "count": len(items),
        },
    )
    with (human / "items.jsonl").open("w", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    # Labels sheet only: model replies beginning '=' must never become spreadsheet formulas.
    with (human / "judgments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "choice", "reason", "judge_id"])
        writer.writerows([item["id"], "", "", ""] for item in items)
    lines = [
        "# Response comparison",
        "",
        "Given the problem and conversation history, which next teacher reply is better for the student? Choose A, B, tie, or neither. Briefly explain your choice.",
        "",
    ]
    for item in items:
        lines.extend(["## " + item["id"], ""])
        for key in ["problem", "history", "A", "B"]:
            value = item[key]
            # A literal-text block: preserve every character, do not render model HTML.
            fence = "`" * max(
                3, max((len(s) for s in re.findall(r"`+", value)), default=0) + 1
            )
            lines.extend(["### " + key, "", fence + "text", value, fence, ""])
    (human / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        f"Exported {export(args.left, args.right, args.output, args.seed)} pairs; share ONLY output/human/"
    )


if __name__ == "__main__":
    main()
