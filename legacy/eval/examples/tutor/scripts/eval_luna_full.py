"""Full seven-preference Luna evaluation; no training or server deployment.

Run with --help. The $25 default is an estimated stopping threshold, not a
provider hard cap. Prices are the Luna estimate used in our capacity report;
override --teacher-pricing if your proxy charges differently. Old smoke runs
are not included. Resume counts all usage in the same output directory.
Do not reuse a budget-stopped directory to obtain a fresh budget.
"""

import argparse
import fcntl
import math
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def check_local_endpoints(student_url, judge_url, api_key):
    """Check real chat authorization on both local roles before paid calls."""
    import httpx

    with httpx.Client(trust_env=False, timeout=30) as client:
        for role, url, model in (
            ("student", student_url, "qwen3-1.7b"),
            ("judge", judge_url, "qwen3-8b"),
        ):
            response = client.post(
                url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            # Never include request headers or credentials in errors.
            if response.status_code != 200:
                raise RuntimeError(
                    f"{role} preflight failed (HTTP {response.status_code}); no Luna requests sent"
                )
            data = response.json()
            if not data.get("choices"):
                raise RuntimeError(
                    f"{role} preflight returned no choices; no Luna requests sent"
                )


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--student-base-url", required=True)
    parser.add_argument("--aux-base-url", required=True)
    parser.add_argument("--env-file", type=Path, default=repo / ".env")
    parser.add_argument("--budget-usd", type=float, default=25)
    parser.add_argument(
        "--unknown-request-reserve-usd",
        type=float,
        default=0.10,
        help="Estimated budget reservation per missing-usage call; 0 stops on unknown usage.",
    )
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--teacher-model", default="openai/gpt-5.6-luna")
    parser.add_argument(
        "--teacher-pricing", type=float, nargs=4, default=[0.20, 0.02, 0.25, 1.20]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Resume only infrastructure/diagnostic failures, never low scores or format errors alone.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        not math.isfinite(args.unknown_request_reserve_usd)
        or args.unknown_request_reserve_usd < 0
    ):
        parser.error("--unknown-request-reserve-usd must be finite and non-negative")
    load_dotenv(args.env_file, override=False)
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_BASE_URL"):
        parser.error("Set OPENAI_API_KEY and OPENAI_BASE_URL in --env-file")
    env = dict(os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] = str(args.unknown_request_reserve_usd)
    output = args.output_dir.resolve()
    command = [
        sys.executable,
        "-m",
        "examples.tutor.evaluate_teacher_api",
        "--provider",
        "openai",
        "--teacher-model",
        args.teacher_model,
        "--env-file",
        str(args.env_file.resolve()),
        "--config",
        "examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml",
        "--student-base-url",
        args.student_base_url,
        "--aux-base-url",
        args.aux_base_url,
        "--student-api-key-env",
        "INF_API_KEY",
        "--aux-api-key-env",
        "INF_API_KEY",
        "--reasoning-effort",
        "medium",
        "--teacher-format",
        "thinking",
        "--teacher-sampling",
        "provider-default",
        "--teacher-output-limit",
        "provider-default",
        "--concurrency",
        str(args.concurrency),
        "--teacher-budget-usd",
        str(args.budget_usd),
        "--teacher-pricing",
        *map(str, args.teacher_pricing),
        "--output-dir",
        str(output),
    ]
    if args.resume or args.backfill:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    command += [
        "--",
        "--limit",
        "0",
        "--attempts",
        "1",
        "--teacher-presolve",
        "on",
        "--save-traces",
        "all",
        "--skip-preflight",
        "--episode-error-retries",
        "1",
    ]
    if args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=repo, env=env))
    if args.backfill:
        command += [
            "--retry-diagnostic-failures",
            "--allow-diagnostic-backfill-on-resume",
            "--allow-evaluator-code-change-on-resume",
        ]
    # Keep the lock outside the evaluator directory: its empty-dir check remains
    # intact. A second process must not share this directory's budget ledger.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An evaluation already holds this output directory's lock")
        if (output / "run_config.json").exists() and "--resume" not in command:
            command.insert(command.index("--"), "--resume")
        check_local_endpoints(
            args.student_base_url,
            args.aux_base_url,
            os.getenv("INF_API_KEY") or "EMPTY",
        )
        raise SystemExit(subprocess.call(command, cwd=repo, env=env))


if __name__ == "__main__":
    main()
