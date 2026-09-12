#!/usr/bin/env python3
"""CPU-only, non-destructive StepVerify rescoring from saved raw replies."""

import argparse
import ast
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml
from run_task import SCORING_PROTOCOL, parse_task_response, response_for_task

TASKS = ("student_solution_correctness", "mistake_correction")


def correction_parser(upstream):
    """Load only the unchanged upstream parser, without datasets or GPUs."""
    path = upstream / "tasks/mistake_correction.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "parse_response"
    )
    namespace = {"re": re}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(
        parse_response=lambda text: namespace["parse_response"](None, text)
    )


def metrics_for(task, records):
    if task == "mistake_correction":
        return {
            "accuracy": sum(
                r["prediction"] is not None
                and abs(float(r["prediction"]) - float(r["target"])) < 1e-6
                for r in records
            )
            / len(records)
        }
    tp = fp = fn = tn = 0
    for r in records:
        target = str(r["target"]).strip().lower() == "yes"
        pred = bool(r["prediction"])
        tp += pred and target
        fp += pred and not target
        fn += not pred and target
        tn += not pred and not target
    return {
        "accuracy": (tp + tn) / len(records),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
    }


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def rescore(source, upstream):
    source = source.resolve()
    output = source.with_name(source.name + "-" + SCORING_PROTOCOL)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    parser = correction_parser(upstream)
    # Validate inputs before creating any output.
    sources = {task: source / "tasks" / task / "predictions.jsonl" for task in TASKS}
    original = {
        task: [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        for task, path in sources.items()
    }
    for task, rows in original.items():
        assert rows and all("raw_response" in r for r in rows), task
        assert len({r["index"] for r in rows}) == len(rows), task
    output.mkdir()
    manifest = {
        "source_run": str(source),
        "scoring_protocol": SCORING_PROTOCOL,
        "generation_reused": True,
        "source_sha256": {},
        "changes": {},
    }
    for task, rows in original.items():
        manifest["source_sha256"][task] = hashlib.sha256(
            sources[task].read_bytes()
        ).hexdigest()
        config = yaml.safe_load((upstream / "configs" / (task + ".yaml")).read_text())
        records = deepcopy(rows)
        for r in records:
            visible = response_for_task(task, r["raw_response"], config.get("stop"))
            r.update(
                visible_response=visible,
                prediction=parse_task_response(task, visible, parser),
                scoring_protocol=SCORING_PROTOCOL,
            )
        dest = output / "tasks" / task
        dest.mkdir(parents=True)
        (dest / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        old = json.loads((source / "tasks" / task / "metrics.json").read_text())
        new = {
            **old,
            "metrics": metrics_for(task, records),
            "scoring_protocol": SCORING_PROTOCOL,
        }
        save(dest / "metrics.json", new)
        manifest["changes"][task] = {
            "before": old["metrics"],
            "after": new["metrics"],
            "changed_predictions": sum(
                a["prediction"] != b["prediction"] for a, b in zip(rows, records)
            ),
            "samples": len(records),
        }
        assert (
            hashlib.sha256(sources[task].read_bytes()).hexdigest()
            == manifest["source_sha256"][task]
        )
    save(output / "rescore.json", manifest)
    # This is explicitly a composite report: only the two affected tasks were
    # reparsed; every other metric remains from the named source evaluation.
    summary = deepcopy(json.loads((source / "summary.json").read_text()))
    for task in TASKS:
        payload = json.loads((output / "tasks" / task / "metrics.json").read_text())
        summary["official_task_metrics"][payload["task_name"]] = payload
    metrics = summary["official_task_metrics"]
    summary["leaderboard"]["solution_correctness"] = metrics["solution_correctness"][
        "metrics"
    ]["f1"]
    summary["leaderboard"]["mistake_location"] = metrics["mistake_location"]["metrics"][
        "f1_micro"
    ]
    summary["leaderboard"]["mistake_correction"] = metrics["mistake_correction"][
        "metrics"
    ]["accuracy"]
    summary.update(
        scoring_protocol=SCORING_PROTOCOL,
        source_run=str(source),
        rescored_tasks=list(TASKS),
        unchanged_tasks_source=str(source),
    )
    save(output / "summary.json", summary)
    (output / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return manifest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("runs", type=Path, nargs="+")
    cli.add_argument(
        "--upstream", type=Path, default=Path(__file__).parent / ".runtime/upstream"
    )
    args = cli.parse_args()
    for source in args.runs:
        print(json.dumps(rescore(source, args.upstream), ensure_ascii=False))


if __name__ == "__main__":
    main()
