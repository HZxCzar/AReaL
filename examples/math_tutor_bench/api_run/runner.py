"""API generation without GPUs; separate, offline official Ped-RM scoring.

Never modifies the checkpoint runner. Uses its output adapter and the pinned
upstream task prompts/parsers/metrics. All tasks use a single user chat message:
this is a declared transport difference from legacy completion-mode evaluation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml

from examples.math_tutor_bench import run_task
from examples.math_tutor_bench.prepare import UPSTREAM_COMMIT
from examples.math_tutor_bench.summarize import CONFIGS, atomic_text

BENCH = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def read_jsonl(path):
    """Repair only an interrupted final append, never silently skip corruption."""
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as stream:
        end = 0
        while line := stream.readline():
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                if stream.read().strip():
                    raise ValueError(f"Corrupt interior record: {path}") from None
                stream.truncate(end)
                break
            end = stream.tell()
    return records


def append(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def usage_cost(usage, prices):
    if not usage or not {"prompt_tokens", "completion_tokens"} <= usage.keys():
        return None
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    total = usage.get("total_tokens", prompt + completion)
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens", 0) or 0
    written = details.get("cache_write_tokens", 0) or 0
    values = [prompt, completion, total, cached, written]
    if any(
        not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values
    ):
        return None
    counts = [
        max(0, prompt - cached - written),
        cached,
        written,
        max(completion, total - prompt),
    ]
    return sum(n * rate for n, rate in zip(counts, prices, strict=True)) / 1e6


class BudgetStopped(RuntimeError):
    pass


class Journal:
    """One synchronized, append-only HTTP audit and cumulative spend ledger."""

    def __init__(self, root, prices, limit, reserve):
        self.root, self.prices, self.limit, self.reserve = root, prices, limit, reserve
        self.path = root / "api_requests.jsonl"
        self.lock = threading.Lock()
        self.requests, self.responses, self.successes = {}, {}, {}
        self.known_cost, self.unknown_count = 0.0, 0
        self.fatal = False
        for row in read_jsonl(self.path):
            self.consume(row)
        self.status()

    def consume(self, row):
        rid = row["request_id"]
        if row["event"] == "request":
            if rid not in self.requests:
                self.unknown_count += 1
            self.requests[rid] = row
        else:
            old_cost = usage_cost(
                self.responses.get(rid, {}).get("payload", {}).get("usage"), self.prices
            )
            new_cost = usage_cost(row.get("payload", {}).get("usage"), self.prices)
            self.known_cost += (new_cost or 0) - (old_cost or 0)
            self.unknown_count += int(new_cost is None) - int(old_cost is None)
            self.responses[rid] = row
            if row.get("payload", {}).get("choices"):
                self.successes[row["key"]] = row

    def status(self):
        spent, unknown = self.known_cost, self.unknown_count
        result = {
            "limit_usd": self.limit,
            "known_cost_usd": spent,
            "unknown_requests": unknown,
            "reserve_per_unknown_usd": self.reserve,
            "budget_used_usd": spent + unknown * self.reserve,
            "scope": "estimated teacher API spend, not a provider billing cap",
        }
        run_task.write_json(self.root / "budget_status.json", result)
        return result

    def record(self, row):
        append(self.path, {"at": datetime.now(UTC).isoformat(), **row})
        self.consume(row)
        self.status()

    def begin(self, key, payload):
        with self.lock:
            status = self.status()
            if (
                self.fatal
                or (status["unknown_requests"] and self.reserve == 0)
                or status["budget_used_usd"] >= self.limit
            ):
                raise BudgetStopped(
                    "Cumulative budget stopped; inspect budget_status.json"
                )
            rid = uuid.uuid4().hex
            self.record(
                {"event": "request", "request_id": rid, "key": key, "payload": payload}
            )
            return rid

    def finish(self, rid, key, payload, status):
        with self.lock:
            self.record(
                {
                    "event": "response",
                    "request_id": rid,
                    "key": key,
                    "status_code": status,
                    "payload": payload,
                }
            )


def load_config(path):
    config = yaml.safe_load(path.read_text())
    expected = {
        "version",
        "run_name",
        "model",
        "endpoint_env",
        "key_env",
        "reasoning_effort",
        "prices_per_million",
    }
    if (
        not isinstance(config, dict)
        or set(config) != expected
        or config["version"] != 1
    ):
        raise ValueError("Unknown/missing API preset fields")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["run_name"]):
        raise ValueError("Invalid run_name")
    prices = config["prices_per_million"]
    if len(prices) != 4 or any(not math.isfinite(x) or x < 0 for x in prices):
        raise ValueError("Expected four finite nonnegative token prices")
    return config


def request_payload(config, prompt):
    return {
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "messages": [{"role": "user", "content": prompt}],
    }


def setup_runtime():
    runtime = BENCH / ".runtime"
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["HF_DATASETS_CACHE"] = str(runtime / "hf_datasets")
    os.environ["HF_HUB_CACHE"] = str(runtime / "hf_hub")
    sys.path[:0] = [str(runtime / "python"), str(runtime / "upstream")]
    os.chdir(runtime / "upstream")
    revision = (runtime / "upstream/.math_tutor_bench_revision").read_text().strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("Wrong staged upstream revision")


def load_tasks(limit):
    import tasks  # noqa: F401 - upstream registration
    from registry import TaskRegistry
    from tasks.base import TaskConfig

    loaded = []
    for name in CONFIGS:
        raw = yaml.safe_load(
            (BENCH / f".runtime/upstream/configs/{name}.yaml").read_text()
        )
        config = TaskConfig(**raw)
        task = TaskRegistry.get_task(config.name)(config)
        examples = task.get_test_examples()
        if limit:
            examples = examples[:limit]
        for example in examples:
            example["shots"] = config.few_shot_samples
        prompts = [task.get_system_prompt(example) for example in examples]
        loaded.append((name, config, task, examples, prompts))
    return loaded


def build_record(name, config, task, example, index, prompt, response):
    choice = response["choices"][0]
    # NEVER fall back to reasoning_content when content is empty.
    raw = choice["message"].get("content") or ""
    visible = run_task.response_for_task(name, raw, config.stop)
    prediction = run_task.parse_task_response(name, visible, task)
    record = {
        "task_config": name,
        "task_name": config.name,
        "index": index,
        "input_example": run_task.jsonable(example),
        "input_prompt": prompt,
        "request_input": {"messages": [{"role": "user", "content": prompt}]},
        "raw_response": raw,
        "visible_response": visible,
        "finish_reason": choice.get("finish_reason"),
        "prediction": run_task.jsonable(prediction),
        "scoring_protocol": run_task.SCORING_PROTOCOL,
        "response_processing": run_task.RESPONSE_PROCESSING,
        "target": run_task.jsonable(task.format_ground_truth(example)),
    }
    if name in run_task.PEDAGOGY_TASKS:
        record["generation"] = {
            "problem": example.get("question", ""),
            "reference_solution": example.get("reference_solution", "N/A"),
            "dialog_history": example.get("conversation_json", []),
            "dialog_formatted": example.get("dialog_history", ""),
            "ground_truth_response": example.get("ground_truth_response", ""),
            "generated_teacher_utterance": prediction,
        }
    return record


def fetch(client, journal, config, key, prompt, retries):
    if key in journal.successes:
        return journal.successes[key]["payload"]
    import httpx

    payload = request_payload(config, prompt)
    for attempt in range(retries + 1):
        rid = journal.begin(key, payload)
        try:
            response = client.post("chat/completions", json=payload)
        except httpx.TransportError as error:
            journal.finish(rid, key, {"transport_error": type(error).__name__}, None)
            status = 0
        else:
            status = response.status_code
            if response.is_success:
                # Persist before parsing/scoring so a crash never requires regeneration.
                body = response.json()
                journal.finish(rid, key, body, status)
                if not body.get("choices"):
                    raise ValueError("Provider returned no choices; see request log")
                return body
            # Error bodies may echo credentials; record HTTP status only.
            journal.finish(rid, key, {}, status)
            if status in (400, 401, 403, 404):
                journal.fatal = True
        if status not in (0, 408, 429, 500, 502, 503, 504) or attempt == retries:
            raise RuntimeError(f"API request failed: HTTP {status}; key={key}")
        time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def check_manifest(path, manifest):
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Run identity changed; use a NEW output directory")
    else:
        if path.name == "run.json" and (path.parent / "api_requests.jsonl").exists():
            raise ValueError("Refusing to adopt an unmanifested usage ledger")
        run_task.write_json(path, manifest)


def finalize_report(root):
    """Select official leaderboard metrics without changing legacy summaries."""
    report = json.loads((root / "summary.json").read_text())
    tasks = report["official_task_metrics"]
    report["leaderboard"]["solution_correctness"] = tasks["solution_correctness"][
        "metrics"
    ]["f1"]
    report["leaderboard"]["mistake_location"] = tasks["mistake_location"]["metrics"][
        "f1_micro"
    ]
    report["api_evaluation"] = json.loads((root / "run.json").read_text())
    run_task.write_json(root / "summary.json", report)
    atomic_text(
        root / "summary.yaml",
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
    )


def generate(args, config, loaded, root):
    import httpx

    endpoint = os.environ.get(config["endpoint_env"], "").rstrip("/")
    key = os.environ.get(config["key_env"], "")
    if not endpoint or not key:
        raise ValueError(
            "Set the endpoint and key environment variables named in the preset"
        )
    from urllib.parse import urlsplit

    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Endpoint must be a credential-free HTTP(S) base URL")
    manifest = {
        "version": 1,
        "evaluation_mode": "api-chat",
        "response_processing": run_task.RESPONSE_PROCESSING,
        "config": config,
        "math_tutor_bench_revision": UPSTREAM_COMMIT,
        "endpoint_sha256": digest(endpoint),
        "limit_per_task": args.limit,
        "content_sha256": digest(
            [(n, run_task.jsonable(ex), ps) for n, _, _, ex, ps in loaded]
        ),
        "adapter_sha256": digest(
            [Path(__file__).read_text(), Path(run_task.__file__).read_text()]
        ),
        "upstream_source_sha256": digest(
            [
                (str(f.relative_to(BENCH / ".runtime/upstream")), file_digest(f))
                for f in sorted((BENCH / ".runtime/upstream").rglob("*.py"))
            ]
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_manifest(root / "run.json", manifest)
        journal = Journal(
            root,
            config["prices_per_million"],
            args.budget_usd,
            args.unknown_reserve_usd,
        )
        append(
            root / "invocations.jsonl",
            {
                "at": datetime.now(UTC).isoformat(),
                "budget_usd": args.budget_usd,
                "concurrency": args.concurrency,
                "retries": args.retries,
                "unknown_reserve_usd": args.unknown_reserve_usd,
            },
        )
        failed = []
        with httpx.Client(
            base_url=endpoint + "/",
            headers={"Authorization": "Bearer " + key},
            timeout=args.timeout,
        ) as client:
            for name, tc, task, examples, prompts in loaded:
                dest = root / "tasks" / name
                dest.mkdir(parents=True, exist_ok=True)

                def work(index):
                    response = fetch(
                        client,
                        journal,
                        config,
                        f"{name}:{index}",
                        prompts[index],
                        args.retries,
                    )
                    return build_record(
                        name, tc, task, examples[index], index, prompts[index], response
                    )

                records = []
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    # Bounded by one task; budget is checked immediately before every HTTP request.
                    futures = [(i, pool.submit(work, i)) for i in range(len(examples))]
                    for index, future in futures:
                        try:
                            record = future.result()
                        except Exception as error:
                            failed.append(
                                {
                                    "task": name,
                                    "index": index,
                                    "error_type": type(error).__name__,
                                }
                            )
                        else:
                            records.append(record)
                run_task.write_jsonl(dest / "predictions.jsonl", records)
                complete = len(records) == len(examples)
                if complete:
                    metrics = task.compute_metrics(
                        [r["prediction"] for r in records],
                        [r["target"] for r in records],
                    )
                    run_task.write_json(
                        dest / "metrics.json",
                        {
                            "task_config": name,
                            "task_name": tc.name,
                            "num_examples": len(records),
                            "metrics": run_task.jsonable(metrics),
                            "response_processing": run_task.RESPONSE_PROCESSING,
                            "decoding": {
                                "mode": "api-chat",
                                "reasoning_effort": config["reasoning_effort"],
                                "sampling": "provider-default",
                                "max_tokens": None,
                                "seed": None,
                            },
                        },
                    )
                    if name in run_task.PEDAGOGY_TASKS:
                        run_task.write_json(
                            dest / "generations.json",
                            [r["generation"] for r in records],
                        )
                print(f"{name}: {len(records)}/{len(examples)} saved", flush=True)
        run_task.write_json(root / "pending.json", failed)
        if failed:
            raise SystemExit(
                f"Incomplete: {len(failed)} entries; repeat command to resume"
            )
        run_task.write_json(
            root / "generation_complete.json",
            {
                "at": datetime.now(UTC).isoformat(),
                "examples": sum(len(x[3]) for x in loaded),
            },
        )
        print(
            "Generation complete. Transfer this directory to the GPU machine and run score."
        )


def parse_gpu_ids(value):
    """Normalize explicit device IDs; never silently schedule twice on one GPU."""
    ids = [part.strip() for part in value.split(",")]
    if not all(re.fullmatch(r"[0-9]+", part) for part in ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative integers")
    ids = [str(int(part)) for part in ids]
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("GPU IDs must be distinct")
    return ids


def score_pedrm(root, model_path, gpu_ids):
    """Run the existing scorers under the caller's run lock, without reparsing.

    Eight devices run two shards per task concurrently. Smaller device lists
    process batches of tasks; no device runs more than one scoring worker.
    Completed shards are reusable only with identical inputs and scorer code.
    """
    if not gpu_ids or len(gpu_ids) == 1:
        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        subprocess.run(
            [
                sys.executable,
                str(BENCH / "score_pedrm.py"),
                "--model",
                str(model_path),
                "--tasks-root",
                str(root / "tasks"),
                "--output",
                str(root / "pedrm"),
            ],
            env=env,
            check=True,
        )
        return

    from examples.math_tutor_bench.score_pedrm import PEDAGOGY_TASKS

    num_shards = max(1, len(gpu_ids) // len(PEDAGOGY_TASKS))
    identity = {
        "score_identity": file_digest(root / "score_identity.json"),
        "inputs": {
            task: file_digest(root / "tasks" / task / "generations.json")
            for task in PEDAGOGY_TASKS
        },
        "scorers": {
            name: file_digest(BENCH / name)
            for name in ("score_pedrm_shard.py", "merge_pedrm_shards.py")
        },
        "num_shards": num_shards,
    }
    shards = root / "pedrm" / "shards" / digest(identity)
    shards.mkdir(parents=True, exist_ok=True)
    check_manifest(shards / "identity.json", identity)
    jobs = [
        (task, shard)
        for task in PEDAGOGY_TASKS
        for shard in range(num_shards)
        if not (shards / f"{task}-{shard}.json").exists()
    ]
    print(
        f"Ped-RM: {len(jobs)} pending shards on {len(gpu_ids)} GPUs; logs: {shards}",
        flush=True,
    )
    for offset in range(0, len(jobs), len(gpu_ids)):
        workers, logs = [], []
        try:
            for gpu, (task, shard) in zip(
                gpu_ids, jobs[offset : offset + len(gpu_ids)]
            ):
                log = (shards / f"{task}-{shard}.log").open("w")
                logs.append(log)
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "TOKENIZERS_PARALLELISM": "false",
                }
                workers.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(BENCH / "score_pedrm_shard.py"),
                            "--model",
                            str(model_path),
                            "--tasks-root",
                            str(root / "tasks"),
                            "--task",
                            task,
                            "--shard-index",
                            str(shard),
                            "--num-shards",
                            str(num_shards),
                            "--output",
                            str(shards / f"{task}-{shard}.json"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            while any(worker.poll() is None for worker in workers):
                if any(worker.poll() not in (None, 0) for worker in workers):
                    raise RuntimeError(f"Ped-RM worker failed; inspect {shards}")
                time.sleep(0.2)
            if any(worker.returncode != 0 for worker in workers):
                raise RuntimeError(f"Ped-RM worker failed; inspect {shards}")
        finally:
            # Also handles Ctrl-C and partially failed launches: no orphan workers.
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            for log in logs:
                log.close()
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "merge_pedrm_shards.py"),
            "--tasks-root",
            str(root / "tasks"),
            "--shards-root",
            str(shards),
            "--output",
            str(root / "pedrm"),
            "--num-shards",
            str(num_shards),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["estimate", "generate", "score"])
    parser.add_argument(
        "--config", type=Path, default=PACKAGE / "gemini-3.8-flash.yaml"
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit", type=int, default=0, help="Examples PER task; 0 is full"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        help="Score only: comma-separated physical GPU IDs (e.g. 0,1,2,3,4,5,6,7)",
    )
    parser.add_argument("--budget-usd", type=float, default=2)
    parser.add_argument("--unknown-reserve-usd", type=float, default=0.10)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--pedrm-model", default="eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel"
    )
    args = parser.parse_args()
    if args.gpu_ids is not None and args.command != "score":
        parser.error("--gpu-ids is only supported for score")
    if args.limit < 0 or args.concurrency < 1 or args.retries < 0:
        parser.error("Invalid limit/concurrency/retries")
    if (
        any(
            not math.isfinite(v) or v < 0
            for v in [args.budget_usd, args.unknown_reserve_usd, args.timeout]
        )
        or not args.timeout
    ):
        parser.error("Invalid budget/reserve/timeout")
    config = load_config(args.config.resolve())
    root = (args.output_dir or BENCH / "results/api" / config["run_name"]).resolve()
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file.resolve(), override=False)
    # Prefer an explicitly configured HTTP proxy to an optional SOCKS transport.
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)
    setup_runtime()
    if args.command == "score":
        if not (root / "generation_complete.json").exists():
            parser.error(
                "Generation is incomplete; score only a completed exported run"
            )
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from examples.math_tutor_bench.score_pedrm import resolve_local_model

            model_path = resolve_local_model(args.pedrm_model)
            score_identity = {
                "model_config": digest((model_path / "config.json").read_text()),
                "model_files": [
                    (f.name, file_digest(f))
                    for f in sorted(model_path.iterdir())
                    if f.suffix in {".safetensors", ".bin", ".json", ".py"}
                ],
                "generation": digest(json.loads((root / "run.json").read_text())),
                "implementation": digest((BENCH / "score_pedrm.py").read_text()),
            }
            check_manifest(root / "score_identity.json", score_identity)
            score_pedrm(root, model_path, args.gpu_ids)
            subprocess.run(
                [
                    sys.executable,
                    str(BENCH / "summarize.py"),
                    "--run-dir",
                    str(root),
                    "--upstream-revision",
                    UPSTREAM_COMMIT,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            finalize_report(root)
            print(f"Official leaderboard saved: {root / 'summary.yaml'}")
        return
    loaded = load_tasks(args.limit)
    if args.command == "estimate":
        # Deliberately rough and offline: no tokenizer download or paid count call.
        n = sum(len(x[3]) for x in loaded)
        inputs = sum(sum(len(p) / 4 + 8 for p in x[4]) for x in loaded)
        print(
            json.dumps(
                {
                    "task_counts": {x[0]: len(x[3]) for x in loaded},
                    "examples": n,
                    "input_tokens_approx_chars_div4": round(inputs),
                    "no_cache_cost_scenarios_usd": {
                        str(o): round(
                            (
                                inputs * config["prices_per_million"][0]
                                + n * o * config["prices_per_million"][3]
                            )
                            / 1e6,
                            2,
                        )
                        for o in [256, 512, 1024, 2048]
                    },
                    "scenario_key": "average OUTPUT PLUS THINKING tokens per example; not a budget guarantee",
                },
                indent=2,
            )
        )
        return
    generate(args, config, loaded, root)


if __name__ == "__main__":
    main()
