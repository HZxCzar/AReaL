"""Replay recorded student/judge requests only; never calls the paid teacher.

Endpoints/keys come from environment or a private env file. Outputs contain full
request/response bodies, but no credentials or endpoint URLs. This is a load
measurement, not a scored evaluation; warm-prefix reuse can affect throughput.
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


def sample_requests(path, size=256):
    pools = {name: [] for name in ("preference", "leak", "answer", "student")}
    seen = dict.fromkeys(pools, 0)
    rng = random.Random(42)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "request":
                continue
            role = row.get("role", "")
            if role.startswith("student:"):
                kind = "student"
            elif role == "answer_judge":
                kind = "answer"
            elif role == "judge":
                system = str(row["payload"]["messages"][0]["content"])
                kind = (
                    "preference"
                    if "one stated student preference" in system
                    else "leak"
                )
            else:
                continue
            seen[kind] += 1
            pool = pools[kind]
            if len(pool) < size:
                pool.append(row["payload"])
            else:
                index = rng.randrange(seen[kind])
                if index < size:
                    pool[index] = row["payload"]
    if any(not pool for pool in pools.values()):
        raise ValueError("Source needs student, preference, leak and answer requests")
    return pools


def summarize(records, elapsed):
    ok = [r for r in records if r["ok"]]
    latency = sorted(r["seconds"] for r in records)
    return {
        "requests": len(records),
        "ok": len(ok),
        "errors": len(records) - len(ok),
        "seconds": elapsed,
        "requests_per_second": len(ok) / elapsed,
        "output_tokens_per_second": sum(
            (r["response"].get("usage") or {}).get("completion_tokens", 0) for r in ok
        )
        / elapsed,
        "p50_seconds": latency[len(latency) // 2],
        "p95_seconds": latency[min(len(latency) - 1, int(len(latency) * 0.95))],
    }


async def run(args):
    pools = sample_requests(args.source)
    load_dotenv(args.env_file, override=False)
    key = os.environ.get("INF_API_KEY") or "EMPTY"
    roles = {
        "auxiliary": (
            os.environ["BENCH_AUX_BASE_URL"],
            "qwen3.8-27b-fp8",
            os.environ.get("BENCH_AUX_API_KEY") or key,
        ),
        "student": (
            os.environ["STUDENT_BASE_URL"],
            "qwen3-1.7b",
            os.environ.get("STUDENT_API_KEY") or key,
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    stopped = set()
    stages = []
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
    async with httpx.AsyncClient(trust_env=False, timeout=90, limits=limits) as client:
        for concurrency in args.levels:

            async def stage(role):
                url, model, api_key = roles[role]
                semaphore = asyncio.Semaphore(concurrency)
                records = []

                async def request(index):
                    kind = (
                        "student"
                        if role == "student"
                        else ("preference", "leak", "answer")[index % 3]
                    )
                    payload = dict(
                        pools[kind][(index + concurrency * 7) % len(pools[kind])]
                    )
                    payload["model"] = model
                    async with semaphore:
                        start = time.monotonic()
                        record = {
                            "role": role,
                            "kind": kind,
                            "concurrency": concurrency,
                            "index": index,
                            "request": payload,
                        }
                        try:
                            response = await client.post(
                                url.rstrip("/") + "/chat/completions",
                                headers={"Authorization": "Bearer " + api_key},
                                json=payload,
                            )
                            body = response.json() if response.is_success else {}
                            record.update(
                                ok=response.is_success and bool(body.get("choices")),
                                status=response.status_code,
                                response=body,
                            )
                        except Exception as exc:
                            record.update(
                                ok=False, error_type=type(exc).__name__, response={}
                            )
                        record["seconds"] = time.monotonic() - start
                        records.append(record)
                        with (args.output / "requests.jsonl").open("a") as stream:
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

                start = time.monotonic()
                await asyncio.gather(
                    *(request(i) for i in range(max(8, concurrency * 2)))
                )
                result = {
                    "role": role,
                    "concurrency": concurrency,
                    **summarize(records, time.monotonic() - start),
                }
                stages.append(result)
                print(json.dumps(result), flush=True)
                if (
                    result["errors"] / result["requests"] > 0.05
                    or result["p95_seconds"] > 60
                ):
                    stopped.add(role)

            await asyncio.gather(
                *(stage(role) for role in roles if role not in stopped)
            )
            (args.output / "summary.json").write_text(json.dumps(stages, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    args = parser.parse_args()
    if not args.levels or any(c < 1 or c > 128 for c in args.levels):
        parser.error("Concurrency must be between 1 and 128")
    asyncio.run(run(args))
