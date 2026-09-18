"""Full API teacher evaluation, sharing the Luna protocol across providers.

Endpoints and credentials come from .env, never from this script. Re-running
automatically resumes the same output directory and its cumulative budget.
The budget is an estimated stopping threshold, not a provider billing cap.
No teacher calls are made during --dry-run or local-role preflight.
"""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from examples.tutor.core.api_eval_budget import TeacherBudget


def local_preflight(student_url, judge_url, key):
    """Fail before paid teacher requests if either fixed local role is unavailable."""
    import httpx

    with httpx.Client(trust_env=False, timeout=30) as client:
        for role, url, model in (
            ("student", student_url, "qwen3-1.7b"),
            ("judge", judge_url, "qwen3-8b"),
        ):
            response = client.post(
                url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            if response.status_code != 200 or not response.json().get("choices"):
                raise RuntimeError(
                    f"{role} preflight failed (HTTP {response.status_code}); no teacher calls sent"
                )


def evaluation_command(args, repo, env):
    """Resolve one shared protocol; vary only provider, model, prices and transport."""
    student = env.get("STUDENT_BASE_URL", "")
    judge = env.get("AUX_BASE_URL", "")
    for url in (student, judge):
        if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).hostname:
            raise ValueError("Set STUDENT_BASE_URL and AUX_BASE_URL in .env")
    bypass = [
        env.get("NO_PROXY", ""),
        env.get("no_proxy", ""),
        "localhost",
        "127.0.0.1",
        "::1",
        urlsplit(student).hostname,
        urlsplit(judge).hostname,
    ]
    env["NO_PROXY"] = env["no_proxy"] = ",".join(x for x in bypass if x)
    if args.keep_env_proxy:
        if not (env.get("HTTPS_PROXY") or env.get("https_proxy")):
            raise ValueError(
                "Enable clashon / HTTPS_PROXY before using --keep-env-proxy"
            )
        # HTTP(S) proxy is sufficient; avoid requiring the optional SOCKS transport.
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    else:
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
    env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] = "0"
    command = [
        sys.executable,
        "-m",
        "examples.tutor.evaluate_teacher_api",
        "--provider",
        args.provider,
        "--env-file",
        str(args.env_file.resolve()),
        "--config",
        "examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml",
        "--student-base-url",
        student,
        "--aux-base-url",
        judge,
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
        str(args.output_dir.resolve()),
    ]
    if args.teacher_model:
        command += ["--teacher-model", args.teacher_model]
    if (args.output_dir / "run_config.json").exists() or args.resume or args.backfill:
        command += ["--resume"]
    if args.dry_run:
        command += ["--dry-run"]
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
    if args.keep_env_proxy:
        command += ["--keep-env-proxy"]
    if args.backfill:
        command += [
            "--retry-diagnostic-failures",
            "--allow-diagnostic-backfill-on-resume",
            "--allow-evaluator-code-change-on-resume",
        ]
    return command


def main():
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["gemini", "openai", "generic"], required=True
    )
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-pricing", type=float, nargs=4, required=True)
    parser.add_argument("--budget-usd", type=float, default=2)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--env-file", type=Path, default=repo / ".env")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-env-proxy", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    env = dict(os.environ)
    try:
        if args.concurrency < 1:
            raise ValueError("Concurrency must be positive")
        command = evaluation_command(args, repo, env)
        # Unknown usage stops the formal run; do not silently assign it zero cost.
        os.environ["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] = "0"
        budget = TeacherBudget(
            args.output_dir / "teacher_usage.jsonl",
            args.budget_usd,
            args.teacher_pricing,
        )
        if not args.dry_run:
            budget.check()
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=repo, env=env))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir.with_name(args.output_dir.name + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation holds this output directory's lock")
        local_preflight(
            env["STUDENT_BASE_URL"],
            env["AUX_BASE_URL"],
            env.get("INF_API_KEY") or "EMPTY",
        )
        raise SystemExit(subprocess.call(command, cwd=repo, env=env))


if __name__ == "__main__":
    main()
