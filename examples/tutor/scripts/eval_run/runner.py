"""Portable, config-driven evaluation against deployed OpenAI-compatible roles.

No server deployment, cluster paths, model-specific patches, or proxy activation.
Paid API-teacher budgeting remains the responsibility of the api_run entrypoint.
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
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[3]
PREFERENCES = (
    "none",
    "attempt-diagnosis",
    "subgoal-decomposition",
    "contrastive-comparison",
    "causal-justification",
    "step-demonstration",
    "independent-verification",
)


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
    path = path.resolve()
    if path in seen:
        raise ValueError("Cyclic experiment config inheritance")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Experiment YAML must contain a mapping")
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
        "root": {
            "version",
            "protocol",
            "run_name",
            "teacher",
            "roles",
            "evaluation",
            "execution",
        },
        "teacher": {
            "model",
            "endpoint_env",
            "key_env",
            "format",
            "enable_thinking",
            "checkpoint",
            "adapter_env",
            "request_params",
        },
        "roles": {"student", "auxiliary"},
        "evaluation": {
            "preferences",
            "id_preferences",
            "attempts",
            "expected_questions",
        },
        "execution": {
            "concurrency",
            "student_concurrency",
            "auxiliary_concurrency",
            "episode_timeout_seconds",
            "episode_error_retries",
            "proxy",
        },
    }
    for name, fields in schemas.items():
        value = config if name == "root" else config.get(name)
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"Missing or unknown fields in {name}")
    if config["version"] != 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", config["run_name"]
    ):
        raise ValueError("Require schema version 1 and a safe run_name")
    teacher = config["teacher"]
    for role in config["roles"].values():
        if not isinstance(role, dict) or set(role) != {
            "model",
            "endpoint_env",
            "key_env",
        }:
            raise ValueError("Invalid role config")
    for role in [teacher, *config["roles"].values()]:
        if not isinstance(role["model"], str) or not role["model"]:
            raise ValueError("Specify exact served model names")
        for field in ("endpoint_env", "key_env"):
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", role[field]):
                raise ValueError(
                    "Endpoint/key fields must be environment variable NAMES"
                )
    if not isinstance(teacher["checkpoint"], str) or not teacher["checkpoint"]:
        raise ValueError("An explicit checkpoint identity is required")
    if teacher["adapter_env"] is not None and not re.fullmatch(
        r"[A-Z_][A-Z0-9_]*", teacher["adapter_env"]
    ):
        raise ValueError("adapter_env must be an environment variable name or null")
    if (
        teacher["format"] not in {"thinking", "non_thinking"}
        or type(teacher["enable_thinking"]) is not bool
    ):
        raise ValueError("Invalid teacher response format / native thinking toggle")
    params = teacher["request_params"]
    if not isinstance(params, dict) or set(params) - {"seed"}:
        raise ValueError(
            "Only seed is supported in teacher.request_params; generation settings belong in protocol"
        )
    if "seed" in params and type(params["seed"]) is not int:
        raise ValueError("Teacher seed must be an integer")
    evaluation = config["evaluation"]
    for name in ("preferences", "id_preferences"):
        values = evaluation[name]
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or set(values) - set(PREFERENCES)
        ):
            raise ValueError(f"Invalid {name}")
    if not evaluation["preferences"]:
        raise ValueError("Select at least one preference")
    for name in ("attempts", "expected_questions"):
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError(f"Invalid {name}")
    execution = config["execution"]
    for name in (
        "concurrency",
        "student_concurrency",
        "auxiliary_concurrency",
        "episode_error_retries",
    ):
        minimum = 0 if name == "episode_error_retries" else 1
        if type(execution[name]) is not int or execution[name] < minimum:
            raise ValueError(f"Invalid {name}")
    if execution["proxy"] not in {"environment", "direct"}:
        raise ValueError("proxy must be environment or direct")
    if (
        not math.isfinite(execution["episode_timeout_seconds"])
        or execution["episode_timeout_seconds"] <= 0
    ):
        raise ValueError("Invalid episode timeout")


def digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def endpoint(role: dict, env: dict) -> str:
    value = env.get(role["endpoint_env"], "").rstrip("/")
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
    return value


def prepare(
    config: dict, args: argparse.Namespace, env: dict
) -> tuple[list[str], dict, dict]:
    if (
        args.limit < 0
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        raise ValueError("Invalid limit or shard selection")
    protocol = yaml.safe_load(Path(config["protocol"]).read_text())
    if not isinstance(protocol, dict) or "defaults" in protocol:
        raise ValueError(
            "Protocol must be a standalone snapshot, not Hydra inheritance"
        )
    if len(protocol["student_axes"]) != 1:
        raise ValueError(
            "One student axis is required; select preferences in evaluation"
        )
    teacher, roles, execution = config["teacher"], config["roles"], config["execution"]
    axis = protocol["student_axes"][0]
    axis["personalities"] = config["evaluation"]["preferences"]
    axis["template"].update(
        name=roles["student"]["model"],
        model=roles["student"]["model"],
        max_concurrent_calls=execution["student_concurrency"],
    )
    axis["template"]["base_url"] = "${oc.env:EVAL_RUN_STUDENT_URL}"
    axis["template"]["api_key"] = "${oc.env:EVAL_RUN_STUDENT_KEY}"
    protocol["auxiliary_model"].update(
        model=roles["auxiliary"]["model"],
        max_concurrent_calls=execution["auxiliary_concurrency"],
    )
    protocol["auxiliary_model"]["base_url"] = "${oc.env:EVAL_RUN_AUX_URL}"
    protocol["auxiliary_model"]["api_key"] = "${oc.env:EVAL_RUN_AUX_KEY}"
    protocol["teacher_response_format"] = teacher["format"]
    protocol["enable_thinking"] = teacher["enable_thinking"]
    params = copy.deepcopy(teacher["request_params"])
    params["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": teacher["enable_thinking"]}
    }
    adapter = env.get(teacher["adapter_env"], "") if teacher["adapter_env"] else ""
    if teacher["adapter_env"] and not adapter:
        raise ValueError(
            f"Set {teacher['adapter_env']} to the teacher server's adapter selector"
        )
    if adapter:
        params["extra_body"]["lora_path"] = adapter
    # Keep the generated protocol portable and free of private selectors.
    protocol["teacher_api_request_params"] = copy.deepcopy(teacher["request_params"])
    protocol["teacher_api_request_params"]["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": teacher["enable_thinking"]}
    }
    urls = {
        name: endpoint(role, env)
        for name, role in {"teacher": teacher, **roles}.items()
    }
    for role in [teacher, *roles.values()]:
        if not env.get(role["key_env"]):
            raise ValueError(
                f"Set {role['key_env']} (EMPTY for an unauthenticated endpoint)"
            )
    env.update(
        DEEPSEEK_API_KEY=env[teacher["key_env"]],
        EVAL_RUN_STUDENT_URL=urls["student"],
        EVAL_RUN_STUDENT_KEY=env[roles["student"]["key_env"]],
        EVAL_RUN_AUX_URL=urls["auxiliary"],
        EVAL_RUN_AUX_KEY=env[roles["auxiliary"]["key_env"]],
        TUTOR_EVAL_SHARD_COUNT=str(args.shard_count),
        TUTOR_EVAL_SHARD_INDEX=str(args.shard_index),
        PYTHONUNBUFFERED="1",
    )
    if execution["proxy"] == "direct":
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env.pop(name, None)
    protocol_path = args.output_dir / "protocol.yaml"
    command = [
        sys.executable,
        "-m",
        "examples.tutor.scripts.evaluate_api_teacher_sharded",
        "--config",
        str(protocol_path),
        "--teacher-base-url",
        urls["teacher"],
        "--teacher-model",
        teacher["model"],
        "--api-key",
        "EMPTY",
        "--teacher-request-params",
        json.dumps(params),
        "--teacher-presolve",
        "config",
        "--student-generalization",
        "config",
        "--attempts",
        str(config["evaluation"]["attempts"]),
        "--limit",
        str(args.limit),
        "--concurrency",
        str(execution["concurrency"]),
        "--episode-timeout-seconds",
        str(execution["episode_timeout_seconds"]),
        "--episode-error-retries",
        str(execution["episode_error_retries"]),
        "--retry-diagnostic-failures",
        "--save-traces",
        "all",
        "--save-api-requests",
        "--output-dir",
        str(args.output_dir / "evaluation"),
    ]
    if execution["proxy"] == "environment":
        command.append("--keep-env-proxy")
    if (args.output_dir / "evaluation/run_config.json").exists():
        command.append("--resume")
    scientific = copy.deepcopy(config)
    scientific["protocol"] = Path(config["protocol"]).name
    # Operational changes are logged but cannot silently change protocol contents.
    manifest = {
        "schema_version": 2,
        "experiment": scientific,
        "protocol_sha256": digest(yaml.safe_dump(protocol, sort_keys=True)),
        "endpoint_sha256": {name: digest(url) for name, url in urls.items()},
        "adapter_selector_sha256": digest(adapter),
        "asset_location_sha256": {
            name: digest(env.get(name, "<protocol-default>"))
            for name in ("TUTOR_DATASET", "TUTOR_TOKENIZER")
        },
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "source_sha256": {},
    }
    sources = list((REPO / "examples/tutor").rglob("*.py")) + list(
        (REPO / "examples/common").glob("*.py")
    )
    for field in ("prompts_path", "complaints_path"):
        sources.append(REPO / protocol["personality"][field])
    manifest["source_sha256"] = {
        str(p.relative_to(REPO)): digest(p.read_bytes()) for p in sorted(sources)
    }
    if adapter and Path(adapter).is_dir():
        manifest["adapter_files_sha256"] = {
            p.name: digest(p.read_bytes())
            for p in sorted(Path(adapter).iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors"}
        }
    return command, manifest_identity(manifest), protocol


def manifest_identity(manifest: dict) -> dict:
    """Exclude unrelated human-study tools and this orchestration-only runner.

    The generated protocol, request/deployment settings, and actual evaluator
    source remain strictly checked. Normalize legacy manifests the same way.
    """
    result = copy.deepcopy(manifest)
    result["source_sha256"] = {
        path: value
        for path, value in result.get("source_sha256", {}).items()
        if not path.startswith("examples/tutor/human_study/")
        and path != "examples/tutor/scripts/eval_run/runner.py"
    }
    return result


def check_manifest(path: Path, expected: dict) -> None:
    if path.exists() and resume_identity(
        json.loads(path.read_text())
    ) != resume_identity(expected):
        raise ValueError(
            "Experiment/protocol/deployment changed; use a new output directory"
        )


def resume_identity(manifest: dict) -> dict:
    """Episode scheduling concurrency is operational, not protocol identity.

    Keep actual values in manifests and invocation history for reproducibility.
    Per-caller limits, timeout/retry policy, and all scientific fields stay strict.
    """
    result = manifest_identity(manifest)
    result.get("experiment", {}).get("execution", {}).pop("concurrency", None)
    return result


def preflight(config: dict, env: dict) -> None:
    import httpx

    with httpx.Client(
        trust_env=config["execution"]["proxy"] == "environment", timeout=30
    ) as client:
        for name, role in {"teacher": config["teacher"], **config["roles"]}.items():
            response = client.get(
                endpoint(role, env) + "/models",
                headers={"Authorization": f"Bearer {env[role['key_env']]}"},
            )
            if response.status_code != 200 or role["model"] not in {
                v["id"] for v in response.json().get("data", [])
            }:
                raise RuntimeError(
                    f"{name} model catalog preflight failed (HTTP {response.status_code})"
                )


def completion_status(output: Path, config: dict, args: argparse.Namespace) -> dict:
    questions = (
        min(args.limit, config["evaluation"]["expected_questions"])
        if args.limit
        else config["evaluation"]["expected_questions"]
    )
    rows = questions * len(config["evaluation"]["preferences"])
    expected = (
        len(range(args.shard_index, rows, args.shard_count))
        * config["evaluation"]["attempts"]
    )
    path = output / "evaluation/summary.json"
    if not path.exists():
        return {"complete": False, "expected": expected, "reason": "missing summary"}
    summary = json.loads(path.read_text())
    mode = summary.get("modes", {}).get("presolve_on", {})
    completed = mode.get("completed_attempts", 0)
    pending = summary.get("pending_backfill", {}).get("count", 0)
    return {
        "complete": completed == expected and pending == 0,
        "expected": expected,
        "completed": completed,
        "pending": pending,
    }


def main() -> None:
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Preset name or YAML path")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--dry-run", action="store_true", help="No API calls or output writes"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check model catalogs; no generation or output writes",
    )
    args = parser.parse_args()
    try:
        path = Path(args.config)
        if not path.is_file():
            path = PACKAGE / "configs" / (args.config.removesuffix(".yaml") + ".yaml")
        config = load_config(path)
        load_dotenv(args.env_file, override=False)
        args.output_dir = (
            args.output_dir or REPO / "output/eval-run" / config["run_name"]
        ).resolve()
        env = dict(os.environ)
        command, manifest, protocol = prepare(config, args, env)
        if args.dry_run or args.preflight:
            if args.preflight:
                preflight(config, env)
            sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
            return
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_path = args.output_dir.with_name(args.output_dir.name + ".lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            marker = args.output_dir / "experiment.json"
            check_manifest(marker, manifest)
            if (
                args.output_dir.exists()
                and any(args.output_dir.iterdir())
                and not marker.exists()
            ):
                raise ValueError(
                    "Refusing to adopt an existing/legacy output without a matching manifest"
                )
            preflight(config, env)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            protocol_path = args.output_dir / "protocol.yaml"
            text = yaml.safe_dump(protocol, sort_keys=False)
            if protocol_path.exists() and protocol_path.read_text() != text:
                raise ValueError("Generated protocol was modified; refusing resume")
            if not protocol_path.exists():
                protocol_path.write_text(text)
            marker.write_text(json.dumps(manifest, indent=2) + "\n")
            with (args.output_dir / "invocations.jsonl").open("a") as events:
                events.write(
                    json.dumps(
                        {
                            "at": datetime.now(UTC).isoformat(),
                            "resume": "--resume" in command,
                            "execution": config["execution"],
                        }
                    )
                    + "\n"
                )
            with (args.output_dir / "console.log").open("ab") as log:
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
                    status = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            if status:
                raise SystemExit(status)
            completion = completion_status(args.output_dir, config, args)
            (args.output_dir / "completion.json").write_text(
                json.dumps(completion, indent=2)
            )
            if not completion["complete"]:
                raise RuntimeError(
                    "Evaluation has missing/pending episodes; repeat the same command to resume"
                )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
