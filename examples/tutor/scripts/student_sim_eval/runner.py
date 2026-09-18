"""Plan, execute and summarize a student × teacher-strategy matrix, one cell at a time."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import html
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from examples.tutor.scripts.api_run import runner as api

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
STRATEGIES = json.loads((PACKAGE / "strategies.json").read_text())
PREFERENCES = json.loads((PACKAGE / "student_preferences.json").read_text())


def load_config(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic configuration inheritance")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    parent = data.pop("extends", None)
    if "protocol" in data:
        data["protocol"] = str((path.parent / data["protocol"]).resolve())
    config = api.merge(
        load_config(path.parent / parent, (*seen, path)) if parent else {}, data
    )
    if seen:
        return config
    required = {
        "version",
        "protocol",
        "method",
        "students",
        "preferences",
        "strategies",
        "expected_questions",
        "question_ids_env",
        "sampling",
        "teacher",
        "roles",
        "execution",
    }
    if set(config) != required or config["version"] != 1:
        raise ValueError("Missing/unknown matrix fields or unsupported version")
    if config["method"] not in {"ours", "prompt-only", "different-models"}:
        raise ValueError("Unknown simulation method")
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", config["question_ids_env"]):
        raise ValueError("question_ids_env must name an environment variable")
    if (
        not isinstance(config["sampling"], dict)
        or set(config["sampling"]) != {"seed", "population_size"}
        or type(config["sampling"]["seed"]) is not int
        or type(config["sampling"]["population_size"]) is not int
        or config["sampling"]["population_size"] < config["expected_questions"]
    ):
        raise ValueError("Invalid sampling seed or population_size")
    for field in ("preferences", "strategies"):
        values = config[field]
        if (
            not isinstance(values, list)
            or not values
            or any(v not in STRATEGIES for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid or duplicate {field}")
    if (
        type(config["expected_questions"]) is not int
        or config["expected_questions"] < 1
    ):
        raise ValueError("expected_questions must be positive")
    students = config["students"]
    if not isinstance(students, list):
        raise ValueError("students must be a list")
    if config["method"] == "different-models":
        if not students:
            raise ValueError("Specify different-model student slots")
        names = []
        for student in students:
            if set(student) - {"request_params"} != {
                "name",
                "model_env",
                "endpoint_env",
                "key_env",
            } or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", student["name"]):
                raise ValueError("Invalid student slot")
            names.append(student["name"])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate student names")
    elif students:
        raise ValueError("ours/prompt-only rows come from preferences, not students")
    return config


def cells(config: dict) -> list[tuple[str, str]]:
    rows = (
        [s["name"] for s in config["students"]]
        if config["method"] == "different-models"
        else config["preferences"]
    )
    return [(row, strategy) for row in rows for strategy in config["strategies"]]


def sample_ids(population: list, count: int, seed: int) -> list[str]:
    """Seeded sampling without replacement, independent of dataset row order."""
    values = sorted(map(str, population))
    if len(set(values)) != len(values):
        raise ValueError("Duplicate dataset question IDs")
    if not 0 < count <= len(values):
        raise ValueError("Sample size exceeds the question population")
    return sorted(random.Random(seed).sample(values, count))


def question_ids(config: dict, env: dict, limit: int) -> list[str]:
    """Use a supplied frozen list, or sample the local evaluation split once."""
    name = config["question_ids_env"]
    if env.get(name):
        selected = json.loads(Path(env[name]).read_text())
    else:
        from datasets import load_from_disk

        protocol = yaml.safe_load(Path(config["protocol"]).read_text())
        # The snapshot names its local source via TUTOR_DATASET. Do not fetch a
        # remote dataset or silently substitute another split for sampling.
        expression = protocol["valid_dataset"]["path"]
        match = re.fullmatch(r"\$\{oc.env:TUTOR_DATASET,([^}]+)\}", expression)
        if not env.get("TUTOR_DATASET") and not match:
            raise ValueError("Set TUTOR_DATASET to the local dataset for sampling")
        source = Path(env.get("TUTOR_DATASET") or match.group(1))
        if not source.is_absolute():
            source = REPO / source
        dataset = load_from_disk(str(source))
        if "test" not in dataset:
            raise ValueError("Sampling requires the original dataset's test split")
        population = dataset["test"]["id"]
        if len(population) != config["sampling"]["population_size"]:
            raise ValueError(
                "Evaluation population size changed; review sampling config"
            )
        selected = sample_ids(
            population, config["expected_questions"], config["sampling"]["seed"]
        )
    if (
        not isinstance(selected, list)
        or len(selected) != config["expected_questions"]
        or any(type(item) not in (str, int) or not str(item) for item in selected)
    ):
        raise ValueError(
            f"Question selection must contain exactly {config['expected_questions']} string/integer IDs"
        )
    selected = list(map(str, selected))
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate question IDs")
    return selected[:limit] if limit else selected


def resolve_role(role: dict, env: dict) -> dict:
    role = copy.deepcopy(role)
    for field in ("model_env", "prices_env"):
        if field not in role:
            continue
        name = role.pop(field)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) or not env.get(name):
            raise ValueError(f"Set environment variable {name}")
        role[field.removesuffix("_env")] = (
            json.loads(env[name]) if field == "prices_env" else env[name]
        )
    return role


def compile_cell(config: dict, row: str, strategy: str, env: dict) -> tuple[dict, dict]:
    """Only simulation, teacher strategy and role deployment differ between cells."""
    if (row, strategy) not in cells(config):
        raise ValueError("Cell is outside configured matrix")
    protocol = yaml.safe_load(Path(config["protocol"]).read_text())
    if (
        "defaults" in protocol
        or len(protocol["student_axes"]) != 1
        or not protocol["free_chat"]["enabled"]
    ):
        raise ValueError("Use a standalone, single-axis free-chat protocol")
    roles = {name: resolve_role(role, env) for name, role in config["roles"].items()}
    if config["method"] == "different-models":
        slot = next(s for s in config["students"] if s["name"] == row)
        roles["student"] = resolve_role(
            {k: v for k, v in slot.items() if k != "name"}, env
        )
    teacher = resolve_role(config["teacher"], env)
    student_params = roles["student"].pop("request_params", None)
    judge_params = roles["judge"].pop("request_params", None)
    for params in (student_params, judge_params):
        if params is not None and not isinstance(params, dict):
            raise ValueError("Role request_params must be a mapping")
    experiment = dict(
        version=1,
        protocol="protocol.yaml",
        run_name=f"{config['method']}-{row}-{strategy}",
        teacher=teacher,
        roles=roles,
        execution=config["execution"],
    )
    api.validate_config(experiment)
    axis = protocol["student_axes"][0]
    if student_params is not None:
        axis["template"]["request_params"] = student_params
    if judge_params is not None:
        protocol["auxiliary_model"]["request_params"] = judge_params
    axis["personalities"] = [row if config["method"] == "ours" else "none"]
    axis["template"].update(
        name=roles["student"]["model"],
        model=roles["student"]["model"],
        max_concurrent_calls=config["execution"]["concurrency"],
    )
    protocol["auxiliary_model"].update(
        model=roles["judge"]["model"],
        max_concurrent_calls=config["execution"]["concurrency"],
    )
    protocol["teacher_response_format"] = teacher["format"]
    protocol["teacher_private_visibility"] = False
    # This experiment varies a fixed style, not an adaptive strategy selector.
    protocol["teacher_adaptive_instruction_enabled"] = False
    # Preserve the training system's student-awareness paragraph. The temporary
    # user message supplies the particular teaching method for this experiment.
    protocol["free_chat"]["student_awareness_prompt_enabled"] = True
    # A strategy is a teacher instruction, not a statement of the student's type.
    protocol["teacher_system_prompt"] = STRATEGIES[strategy]
    preference = PREFERENCES[row] if config["method"] == "prompt-only" else ""
    protocol["student_system_prompt"] = (
        preference
        + "\nLet this preference guide how you engage with the tutor. Ask for clarification when needed. Solve as accurately as you can; do not deliberately give wrong answers or pretend not to understand."
        if preference
        else ""
    )
    if config["method"] != "ours":
        protocol["personality"]["gate_sample_rate"] = 0.0
    return experiment, protocol


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def scientific_identity(manifest: dict) -> dict:
    """Increasing a safety budget must not require discarding collected episodes."""
    identity = copy.deepcopy(manifest)
    for cell in identity["cells"]:
        cell["experiment"]["execution"].pop("budget_usd", None)
    return identity


def cell_metrics(directory: Path, expected: int, replays: int) -> tuple[dict, set]:
    latest = {}
    with (directory / "results.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            latest[record["key"]] = record
    if len(latest) != expected:
        raise ValueError(
            f"{directory}: expected {expected} episodes, found {len(latest)}"
        )
    improvements, before, after, identities = [], [], [], set()
    sampled = passed = leaks = 0
    for record in latest.values():
        gate = record.get("personality_gate") or {}
        retest = record.get("generalization", {}).get("original", {})
        if any(
            record.get(k)
            for k in (
                "error",
                "student_call_failed",
                "answer_judge_failed_count",
                "leak_check_failed_count",
                "teacher_pre_error_count",
            )
        ) or gate.get("gate_error_count"):
            raise ValueError(
                f"Unresolved diagnostic failure: {directory}/{record['key']}"
            )
        baseline, score = record.get("no_teaching_baseline"), retest.get("score")
        if retest.get("replay_count") != replays or any(
            not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in (baseline, score)
        ):
            raise ValueError(f"Missing/invalid baseline or retest: {record['key']}")
        identity = (record["dataset_index"], record["item_id"], record["attempt"])
        if identity in identities:
            raise ValueError("Duplicate question/attempt within cell")
        identities.add(identity)
        before.append(baseline)
        after.append(score)
        improvements.append(100 * (score - baseline))
        sampled += gate.get("sampled_turn_count", 0)
        passed += gate.get("passed_turn_count", 0)
        leaks += record.get("leak_count", 0) > 0
    return {
        "episodes": expected,
        "baseline_percent": 100 * statistics.mean(before),
        "retest_percent": 100 * statistics.mean(after),
        "improvement_pp": statistics.mean(improvements),
        "improvement_se_pp": statistics.stdev(improvements) / math.sqrt(expected)
        if expected > 1
        else None,
        "gate_pass_percent": 100 * passed / sampled if sampled else None,
        "leak_percent": 100 * leaks / expected,
    }, identities


def summarize(output: Path, manifest: dict) -> dict:
    report = {"method": manifest["method"], "complete": True, "cells": []}
    reference = None
    for cell in manifest["cells"]:
        metrics, identities = cell_metrics(
            output / cell["path"] / "evaluation", manifest["questions"], cell["replays"]
        )
        if "question_ids" in manifest and {item[1] for item in identities} != set(
            manifest["question_ids"]
        ):
            raise ValueError("Results do not match the frozen reviewed question IDs")
        if reference is not None and identities != reference:
            raise ValueError("Cells do not contain the same questions/attempts")
        reference = identities
        report["cells"].append(
            dict(student=cell["row"], strategy=cell["strategy"], **metrics)
        )
    atomic_write(output / "report.json", json.dumps(report, indent=2) + "\n")
    with (output / "heatmap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["cells"][0]))
        writer.writeheader()
        writer.writerows(report["cells"])
    render_svg(output / "heatmap.svg", report)
    return report


def render_svg(path: Path, report: dict) -> None:
    """Dependency-free heatmap with the same fixed [-100, 100] pp scale in all methods."""
    rows = list(dict.fromkeys(c["student"] for c in report["cells"]))
    columns = list(dict.fromkeys(c["strategy"] for c in report["cells"]))
    width, height = 250 + 170 * len(columns), 130 + 48 * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="12">',
        f'<text x="12" y="22">{html.escape(report["method"])} — improvement (pp); blue −100, white 0, red +100</text>',
    ]
    for index, name in enumerate(columns):
        parts.append(f'<text x="{250 + 170 * index}" y="65">{html.escape(name)}</text>')
    for index, name in enumerate(rows):
        parts.append(f'<text x="8" y="{111 + 48 * index}">{html.escape(name)}</text>')
    for cell in report["cells"]:
        x, y = (
            245 + 170 * columns.index(cell["strategy"]),
            80 + 48 * rows.index(cell["student"]),
        )
        value = cell["improvement_pp"]
        pale = round(255 * (1 - min(abs(value) / 100, 1)))
        color = f"rgb(255,{pale},{pale})" if value >= 0 else f"rgb({pale},{pale},255)"
        parts.append(
            f'<rect x="{x}" y="{y}" width="168" height="46" fill="{color}"/><text x="{x + 64}" y="{y + 29}">{value:+.1f}</text>'
        )
    parts.append("</g></svg>")
    atomic_write(path, "\n".join(parts))


def main(
    *,
    config_directory: Path | None = None,
    compile_cell_fn=None,
    cell_module: str = "examples.tutor.scripts.student_sim_eval.cell",
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", help="ours, prompt-only, different-models, or YAML path"
    )
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--budget-usd",
        type=float,
        help="Teacher budget per cell; may be raised on resume",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Explicitly enable API calls")
    mode.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_file():
        path = (config_directory or PACKAGE / "configs") / (
            args.config.removesuffix(".yaml") + ".yaml"
        )
    config = load_config(path)
    if args.budget_usd is not None:
        if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
            parser.error("budget-usd must be positive and finite")
        config["execution"]["budget_usd"] = args.budget_usd
    if args.limit < 0:
        parser.error("limit must be non-negative")
    questions = (
        min(args.limit, config["expected_questions"])
        if args.limit
        else config["expected_questions"]
    )
    if not args.run and not args.summarize:
        sys.stdout.write(
            json.dumps(
                dict(
                    method=config["method"],
                    cells=cells(config),
                    questions_per_cell=questions,
                    episodes=len(cells(config)) * questions,
                    simultaneous_cells=1,
                    concurrency=config["execution"]["concurrency"],
                    teacher_budget_usd_per_cell=config["execution"]["budget_usd"],
                ),
                indent=2,
            )
            + "\n"
        )
        return
    if args.output_dir is None:
        parser.error("Specify --output-dir for execution or summarization")
    output = args.output_dir.resolve()
    if args.summarize:
        summarize(output, json.loads((output / "matrix.json").read_text()))
        return
    load_dotenv(args.env_file, override=False)
    env = dict(os.environ)
    selected_ids = question_ids(config, env, args.limit)
    compiled = []
    for row, strategy in cells(config):
        experiment, protocol = (compile_cell_fn or compile_cell)(
            config, row, strategy, env
        )
        for role in [experiment["teacher"], *experiment["roles"].values()]:
            api.endpoint(role, env)
            if not env.get(role["key_env"]):
                raise ValueError(
                    f"Set {role['key_env']} (EMPTY for an unauthenticated API)"
                )
        compiled.append((row, strategy, experiment, protocol))
    if config["method"] == "different-models":
        models = [resolve_role(s, env)["model"] for s in config["students"]]
        if len(set(models)) != len(models):
            raise ValueError("different-models requires distinct model IDs")
    manifest = dict(
        version=1,
        method=config["method"],
        cell_module=cell_module,
        questions=questions,
        limit=args.limit,
        question_ids=selected_ids,
        selection={
            "method": "explicit_ids"
            if env.get(config["question_ids_env"])
            else "seeded_random",
            **config["sampling"],
        },
        cells=[],
        sources={},
    )
    sources = (
        list((REPO / "examples/tutor").rglob("*.py"))
        + list((REPO / "examples/common").glob("*.py"))
        + list(PACKAGE.rglob("*.json"))
    )
    for source in sorted(sources):
        manifest["sources"][str(source.relative_to(REPO))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    for row, strategy, experiment, protocol in compiled:
        role_hashes = {
            name: hashlib.sha256(api.endpoint(role, env).encode()).hexdigest()
            for name, role in {
                "teacher": experiment["teacher"],
                **experiment["roles"],
            }.items()
        }
        gate_hashes = {
            key: hashlib.sha256(
                (REPO / protocol["personality"][key]).read_bytes()
            ).hexdigest()
            for key in ("prompts_path", "complaints_path")
        }
        manifest["cells"].append(
            dict(
                row=row,
                strategy=strategy,
                path=f"{row}/{strategy}",
                experiment=experiment,
                protocol=protocol,
                endpoints=role_hashes,
                gates=gate_hashes,
                replays=protocol["student_generalize"]["replays"],
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = output / "matrix.json"
        if metadata.exists():
            if scientific_identity(
                json.loads(metadata.read_text())
            ) != scientific_identity(manifest):
                raise ValueError("Matrix/config/source changed: refuse mixed resume")
        else:
            if any(p.name != ".matrix.lock" for p in output.iterdir()):
                raise ValueError("Nonempty output has no matrix manifest")
            atomic_write(metadata, json.dumps(manifest, indent=2) + "\n")
        with (output / "invocations.jsonl").open("a") as events:
            events.write(
                json.dumps({"budget_usd_per_cell": config["execution"]["budget_usd"]})
                + "\n"
            )
        for cell in manifest["cells"]:
            directory = output / cell["path"]
            directory.mkdir(parents=True, exist_ok=True)
            atomic_write(
                directory / "protocol.yaml",
                yaml.safe_dump(cell["protocol"], sort_keys=False),
            )
            atomic_write(
                directory / "config.yaml",
                yaml.safe_dump(cell["experiment"], sort_keys=False),
            )
            atomic_write(
                directory / "question_ids.json", json.dumps(selected_ids) + "\n"
            )
            command = [
                sys.executable,
                "-m",
                cell_module,
                str(directory / "config.yaml"),
                "--env-file",
                str(args.env_file.resolve()),
                "--output-dir",
                str(directory / "evaluation"),
                "--limit",
                str(args.limit),
            ]
            sys.stdout.write(f"Starting {cell['path']}\n")
            sys.stdout.flush()
            process = subprocess.Popen(
                command, cwd=REPO, env=env, start_new_session=True
            )

            def stop(signum, frame):
                raise KeyboardInterrupt

            previous = signal.signal(signal.SIGTERM, stop)
            try:
                code = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                signal.signal(signal.SIGTERM, previous)
            if code:
                raise RuntimeError(
                    f"Cell {cell['path']} exited {code}; resume the same output after investigation"
                )
            cell_metrics(directory / "evaluation", questions, cell["replays"])
        summarize(output, manifest)


if __name__ == "__main__":
    main()
