"""One configuration-driven entrypoint for paper API teacher experiments.

No model-specific code, shell-sourced secrets, or per-run evaluator patches.
The shared tutor API adapter owns request formatting and evaluation semantics.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

from examples.tutor.core.api_eval_budget import TeacherBudget

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


def load_config(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Safe, relative single-parent inheritance; reject cycles and unknown fields."""
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic experiment config inheritance")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Experiment config must be a mapping")
    parent = data.pop("extends", None)
    if "protocol" in data:
        data["protocol"] = str((path.parent / data["protocol"]).resolve())
    result = merge(
        load_config(path.parent / parent, (*seen, path)) if parent else {}, data
    )
    if not seen:
        validate_config(result)
    return result


def validate_config(config: dict) -> None:
    schemas = {
        "root": {"version", "protocol", "run_name", "teacher", "roles", "execution"},
        "teacher": {
            "provider",
            "model",
            "endpoint_env",
            "key_env",
            "reasoning_effort",
            "format",
            "sampling",
            "output_limit",
            "prices",
        },
        "execution": {
            "concurrency",
            "budget_usd",
            "episode_timeout_seconds",
            "episode_error_retries",
            "proxy",
        },
        "roles": {"student", "judge"},
    }
    for name, allowed in schemas.items():
        value = config if name == "root" else config.get(name)
        if not isinstance(value, dict) or set(value) != allowed:
            raise ValueError(f"Missing or unknown fields in {name} config")
    if config["version"] != 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_name"]
    ):
        raise ValueError("Use schema version 1 and a safe run_name")
    teacher = config["teacher"]
    if (
        teacher["provider"] not in {"openai", "gemini", "generic"}
        or not teacher["model"]
    ):
        raise ValueError("Specify a supported provider and exact model ID")
    if (
        teacher["format"] not in {"thinking", "non_thinking"}
        or teacher["sampling"] not in {"config", "provider-default"}
        or teacher["output_limit"] not in {"config", "provider-default"}
    ):
        raise ValueError("Invalid teacher format, sampling or output policy")
    if teacher["reasoning_effort"] not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise ValueError("Invalid reasoning effort")
    for role in config["roles"].values():
        if not isinstance(role, dict) or set(role) != {
            "model",
            "endpoint_env",
            "key_env",
        }:
            raise ValueError("Invalid role config")
    for role in [teacher, *config["roles"].values()]:
        for key in ("endpoint_env", "key_env"):
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", role[key]):
                raise ValueError(
                    "Endpoints and keys must be environment variable NAMES"
                )
    execution = config["execution"]
    if execution["proxy"] not in {"direct", "environment"}:
        raise ValueError("Proxy policy must be direct or environment")
    for key, minimum in (("concurrency", 1), ("episode_error_retries", 0)):
        if type(execution[key]) is not int or execution[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if (
        not math.isfinite(execution["episode_timeout_seconds"])
        or execution["episode_timeout_seconds"] <= 0
    ):
        raise ValueError("Invalid episode timeout")
    if not math.isfinite(execution["budget_usd"]) or execution["budget_usd"] <= 0:
        raise ValueError("Invalid budget")
    # Validation only; never open a ledger or perform API calls here.
    prices = teacher["prices"]
    if (
        not isinstance(prices, list)
        or len(prices) != 4
        or any(
            not isinstance(p, (float, int)) or not math.isfinite(p) or p < 0
            for p in prices
        )
    ):
        raise ValueError("Supply four non-negative finite token prices")


def endpoint(role: dict, env: dict) -> str:
    value = env.get(role["endpoint_env"], "")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"Set a credential-free API base URL in {role['endpoint_env']}"
        )
    return value.rstrip("/")


def prepare(
    config: dict, args: argparse.Namespace, env: dict
) -> tuple[list[str], dict]:
    teacher, execution, roles = config["teacher"], config["execution"], config["roles"]
    protocol = yaml.safe_load(Path(config["protocol"]).read_text())
    student_models = {axis["template"]["model"] for axis in protocol["student_axes"]}
    if (
        student_models != {roles["student"]["model"]}
        or protocol["auxiliary_model"]["model"] != roles["judge"]["model"]
    ):
        raise ValueError(
            "Role model names must match the actual protocol, not just preflight"
        )
    urls = {
        name: endpoint(role, env)
        for name, role in {"teacher": teacher, **roles}.items()
    }
    if not env.get(teacher["key_env"]):
        raise ValueError(
            f"Set {teacher['key_env']} (use EMPTY for an unauthenticated endpoint)"
        )
    for role in roles.values():
        env.setdefault(role["key_env"], "EMPTY")
    if execution["proxy"] == "direct":
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env.pop(key, None)
    elif env.get("HTTPS_PROXY") or env.get("https_proxy"):
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    bypass = [
        env.get("NO_PROXY", ""),
        env.get("no_proxy", ""),
        "localhost",
        "127.0.0.1",
        "::1",
        urlsplit(urls["student"]).hostname,
        urlsplit(urls["judge"]).hostname,
    ]
    env["NO_PROXY"] = env["no_proxy"] = ",".join(x for x in bypass if x)
    env["PYTHONUNBUFFERED"] = "1"
    reserve = getattr(args, "unknown_request_reserve_usd", 0)
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Unknown-request reserve must be finite and non-negative")
    env["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] = f"{reserve:g}"
    command = [
        sys.executable,
        "-m",
        "examples.tutor.evaluate_teacher_api",
        "--config",
        config["protocol"],
        "--env-file",
        str(args.env_file),
        "--provider",
        teacher["provider"],
        "--teacher-model",
        teacher["model"],
        "--teacher-base-url",
        urls["teacher"],
        "--teacher-api-key-env",
        teacher["key_env"],
        "--student-base-url",
        urls["student"],
        "--aux-base-url",
        urls["judge"],
        "--student-api-key-env",
        roles["student"]["key_env"],
        "--aux-api-key-env",
        roles["judge"]["key_env"],
        "--reasoning-effort",
        teacher["reasoning_effort"],
        "--teacher-format",
        teacher["format"],
        "--teacher-sampling",
        teacher["sampling"],
        "--teacher-output-limit",
        teacher["output_limit"],
        "--teacher-budget-usd",
        str(args.budget_usd),
        "--teacher-pricing",
        *map(str, teacher["prices"]),
        "--concurrency",
        str(args.concurrency),
        "--episode-timeout-seconds",
        str(execution["episode_timeout_seconds"]),
        "--output-dir",
        str(args.output_dir),
    ]
    if (args.output_dir / "run_config.json").exists() or args.backfill:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry-run")
    command += [
        "--",
        "--limit",
        str(args.limit),
        "--attempts",
        "1",
        "--teacher-presolve",
        "on",
        "--save-traces",
        "all",
        "--skip-preflight",
        "--save-api-requests",
        "--episode-error-retries",
        str(execution["episode_error_retries"]),
    ]
    if execution["proxy"] == "environment":
        command.append("--keep-env-proxy")
    if args.backfill:
        command += [
            "--retry-diagnostic-failures",
            "--allow-diagnostic-backfill-on-resume",
            "--allow-evaluator-code-change-on-resume",
        ]
    # Endpoint hashes detect accidental deployment changes without publishing URLs.
    manifest = copy.deepcopy(config)
    manifest["protocol"] = Path(config["protocol"]).name
    manifest["protocol_sha256"] = hashlib.sha256(
        Path(config["protocol"]).read_bytes()
    ).hexdigest()
    manifest["endpoint_sha256"] = {
        k: hashlib.sha256(v.encode()).hexdigest() for k, v in urls.items()
    }
    manifest["limit"] = args.limit
    manifest["deployment_path_sha256"] = {
        key: hashlib.sha256(env.get(key, "<protocol-default>").encode()).hexdigest()
        for key in ("TUTOR_TOKENIZER", "TUTOR_DATASET")
    }
    # Operational knobs may change on resume; scientific settings/prices may not.
    for key in ("budget_usd", "concurrency", "proxy"):
        manifest["execution"].pop(key)
    return command, manifest


def check_manifest(path: Path, manifest: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Run configuration changed; refuse to mix experiments in one output directory"
        )


def preflight(config: dict, env: dict) -> None:
    import httpx

    with httpx.Client(trust_env=False, timeout=30) as client:
        for name, role in config["roles"].items():
            response = client.post(
                endpoint(role, env) + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {env.get(role['key_env']) or 'EMPTY'}"
                },
                json={
                    "model": role["model"],
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
            )
            if response.status_code != 200 or not response.json().get("choices"):
                raise RuntimeError(
                    f"{name} preflight failed (HTTP {response.status_code}); no teacher calls sent"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Preset name or path to an experiment YAML")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument(
        "--unknown-request-reserve-usd",
        type=float,
        default=0,
        help="Explicit estimate per missing-usage call, charged to the SAME budget; default 0 stops on unknown usage.",
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Base questions before seven-preference expansion; 0 = full",
    )
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_file():
        path = PACKAGE / "configs" / (args.config.removesuffix(".yaml") + ".yaml")
    try:
        config = load_config(path)
        args.env_file = args.env_file.resolve()
        load_dotenv(args.env_file, override=False)
        args.output_dir = (
            args.output_dir or REPO / "output" / "api-run" / config["run_name"]
        ).resolve()
        args.budget_usd = (
            config["execution"]["budget_usd"]
            if args.budget_usd is None
            else args.budget_usd
        )
        args.concurrency = (
            config["execution"]["concurrency"]
            if args.concurrency is None
            else args.concurrency
        )
        if args.concurrency < 1 or args.limit < 0:
            raise ValueError("Concurrency must be positive and limit non-negative")
        env = dict(os.environ)
        command, manifest = prepare(config, args, env)
        os.environ["TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"] = env[
            "TUTOR_API_UNKNOWN_REQUEST_RESERVE_USD"
        ]
        metadata = args.output_dir.with_name(args.output_dir.name + ".experiment.json")
        if args.dry_run:
            TeacherBudget(
                args.output_dir / "teacher_usage.jsonl",
                args.budget_usd,
                config["teacher"]["prices"],
            )
            sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
            sys.stdout.flush()
            raise SystemExit(subprocess.call(command, cwd=REPO, env=env))
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with args.output_dir.with_name(args.output_dir.name + ".lock").open(
            "a"
        ) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_manifest(metadata, manifest)
            if (args.output_dir / "run_config.json").exists() and not metadata.exists():
                raise ValueError(
                    "Existing output has no formal experiment manifest; use its original entrypoint"
                )
            TeacherBudget(
                args.output_dir / "teacher_usage.jsonl",
                args.budget_usd,
                config["teacher"]["prices"],
            ).check()
            preflight(config, env)
            if not metadata.exists():
                temporary = metadata.with_suffix(".tmp")
                temporary.write_text(json.dumps(manifest, indent=2) + "\n")
                temporary.replace(metadata)
            with args.output_dir.with_name(
                args.output_dir.name + ".invocations.jsonl"
            ).open("a") as events:
                events.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "budget_usd": args.budget_usd,
                            "unknown_request_reserve_usd": args.unknown_request_reserve_usd,
                            "concurrency": args.concurrency,
                            "proxy": config["execution"]["proxy"],
                            "backfill": args.backfill,
                            "resume": "--resume" in command,
                        }
                    )
                    + "\n"
                )
            # Console and file receive identical output; preserve the evaluator exit code.
            with args.output_dir.with_name(args.output_dir.name + ".log").open(
                "ab"
            ) as log:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                try:
                    for line in process.stdout:
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
                        log.write(line)
                        log.flush()
                    raise SystemExit(process.wait())
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
