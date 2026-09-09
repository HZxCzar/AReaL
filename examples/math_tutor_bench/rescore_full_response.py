#!/usr/bin/env python3
"""Reparse saved outputs and rescore on eight GPUs without altering source runs."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def full_response_stops(text, stops):
    """Keep paragraph breaks; only recognize dialogue roles at line starts."""
    stop_list = [stops] if isinstance(stops, str) else list(stops or [])
    if "Student:" not in stop_list or "Teacher:" not in stop_list:
        # Structured-answer tasks retain their task-specific parsing protocol.
        from run_task import apply_official_stops

        return apply_official_stops(text, stops)
    text = re.sub(r"^\s*(?:Teacher|Tutor)\s*:\s*", "", text, count=1)
    boundary = re.search(r"(?m)^[ \t]*(?:Student|Teacher|Tutor)\s*:", text)
    return text[: boundary.start()].strip() if boundary else text.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--gpus", default=os.environ.get("GPU_IDS", "0,1,2,3,4,5,6,7"))
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or not all(g.isdigit() for g in gpus):
        parser.error("--gpus requires eight distinct GPU indices")
    import torch

    if not torch.cuda.is_available():
        parser.error("Run this script on a GPU node; no CUDA GPU is available here")

    here = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    env["HF_DATASETS_CACHE"] = str(here / ".runtime/hf_datasets")
    env["HF_HUB_CACHE"] = str(here / ".runtime/hf_hub")
    env["PYTHONPATH"] = os.pathsep.join([
        str(here), str(here / ".runtime/python"), str(here / ".runtime/upstream"),
        env.get("PYTHONPATH", ""),
    ])
    from summarize import CONFIGS
    from score_pedrm import PEDAGOGY_TASKS, resolve_local_model, write_json

    for source in args.runs:
        source = source.resolve()
        manifest = json.loads((source / "run.json").read_text())
        model = str(resolve_local_model(manifest["pedrm_model"]))
        output = source.with_name(source.name + "-full-response-v1")
        if output.exists():
            raise SystemExit(f"Refusing to overwrite an existing result: {output}")
        output.mkdir()
        (output / "logs").mkdir()
        manifest.update(status="rescoring", source_run=str(source),
                        response_processing="full-response-v1: paragraphs preserved; line-start dialogue boundaries",
                        gpu_ids=args.gpus)
        write_json(output / "run.json", manifest)
        print(f"[output] {output}", flush=True)
        # Only copy raw predictions, never cached scores or old summaries.
        for task in CONFIGS:
            dest = output / "tasks" / task
            dest.mkdir(parents=True)
            shutil.copy2(source / "tasks" / task / "predictions.jsonl", dest / "predictions.jsonl")
            code = (
                "import run_task; from rescore_full_response import full_response_stops; "
                "original=run_task.apply_official_stops; "
                "run_task.apply_official_stops=lambda text,stops: "
                "full_response_stops(text,stops) if 'Teacher:' in (stops or []) "
                "and 'Student:' in (stops or []) else original(text,stops); "
                "run_task.main()"
            )
            subprocess.run([
                sys.executable, "-B", "-c", code, "--upstream", str(here / ".runtime/upstream"),
                "--task", task, "--output", str(dest), "--reparse-only",
                "--max-tokens", str(manifest["max_tokens"]),
                "--max-samples", str(manifest.get("max_samples", 0)),
            ], env=env, check=True)

        shards = output / "pedrm/shards"
        shards.mkdir(parents=True)
        workers = []
        try:
            for i, gpu in enumerate(gpus):
                task, shard = PEDAGOGY_TASKS[i // 2], i % 2
                log = (output / "logs" / f"{task}-{shard}.log").open("w")
                try:
                    proc = subprocess.Popen([
                        sys.executable, str(here / "score_pedrm_shard.py"), "--model", model,
                        "--tasks-root", str(output / "tasks"), "--task", task,
                        "--shard-index", str(shard), "--num-shards", "2",
                        "--output", str(shards / f"{task}-{shard}.json"),
                    ], env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
                finally:
                    log.close()
                workers.append(proc)
            codes = [proc.wait() for proc in workers]
            if any(codes):
                raise RuntimeError(f"Ped-RM worker failures {codes}; see {output / 'logs'}")
        finally:
            for proc in workers:
                if proc.poll() is None:
                    proc.terminate()
            for proc in workers:
                proc.wait()
        subprocess.run([
            sys.executable, str(here / "merge_pedrm_shards.py"),
            "--tasks-root", str(output / "tasks"), "--shards-root", str(shards),
            "--output", str(output / "pedrm"), "--num-shards", "2",
        ], env=env, check=True)
        subprocess.run([
            sys.executable, str(here / "summarize.py"), "--run-dir", str(output),
            "--upstream-revision", manifest["math_tutor_bench_revision"],
        ], env=env, check=True)
        report = json.loads((output / "summary.json").read_text())
        metrics = report["official_task_metrics"]
        report["leaderboard"]["solution_correctness"] = metrics["solution_correctness"]["metrics"]["f1"]
        report["leaderboard"]["mistake_location"] = metrics["mistake_location"]["metrics"]["f1_micro"]
        report["pedagogy_average"] = sum(v["win_rate"] for v in report["official_pedrm_metrics"].values()) / 4
        report["response_processing"] = manifest["response_processing"]
        write_json(output / "summary.json", report)
        import yaml

        (output / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
        manifest["status"] = "complete"
        write_json(output / "run.json", manifest)
        print(f"[done] {output / 'summary.json'}", flush=True)
        print(f"Pedagogy Average: {report['pedagogy_average']:.6f}", flush=True)


if __name__ == "__main__":
    main()
