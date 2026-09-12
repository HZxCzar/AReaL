"""Generate next teacher replies; no reward model, parsing, or response filtering."""

import argparse
import fcntl
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .common import digest, load_dataset, load_yaml, read_json, write_json


def messages(case, study):
    history = "\n".join(f"{t['user']}: {t['text']}" for t in case["history"])
    return [
        {
            "role": "user",
            "content": study["teacher_instruction"]
            + "\nProblem: "
            + case["problem"]
            + "\nConversation:\n"
            + history
            + "\nTeacher (maximum two sentences): ",
        }
    ]


def validate_configs(study, model):
    if (
        set(study) != {"version", "teacher_instruction", "generation"}
        or study["version"] != 1
    ):
        raise ValueError("Invalid study config")
    if (
        not isinstance(study["teacher_instruction"], str)
        or not study["teacher_instruction"].strip()
    ):
        raise ValueError("Missing teacher instruction")
    generation = study["generation"]
    if not isinstance(generation, dict) or set(generation) - {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "chat_template_kwargs",
    }:
        raise ValueError(
            "Unsupported generation settings (no custom stops or model overrides)"
        )
    if (
        not isinstance(generation.get("max_tokens"), int)
        or generation["max_tokens"] <= 0
    ):
        raise ValueError("Positive max_tokens required")
    expected = {
        "version",
        "name",
        "model",
        "endpoint_env",
        "key_env",
        "lora_path",
        "identity",
    }
    if set(model) != expected or model["version"] != 1:
        raise ValueError(
            "Invalid model config; endpoints and credentials must use environment variables"
        )
    for field in ["name", "model", "endpoint_env", "key_env"]:
        if not isinstance(model[field], str) or not model[field]:
            raise ValueError(f"Missing model {field}")
    identity = model["identity"]
    if identity.get("kind") not in {"models", "sglang"}:
        raise ValueError("identity.kind must be models or sglang")
    if model["lora_path"] is not None and (
        identity["kind"] != "sglang" or not identity.get("adapter_path_contains")
    ):
        raise ValueError("LoRA runs require SGLang adapter identity checks")


def payload_for(case, study, model):
    result = {
        "model": model["model"],
        "messages": messages(case, study),
        **study["generation"],
        "stream": False,
    }
    if model["lora_path"] is not None:
        result["lora_path"] = model["lora_path"]
    return result


def http_json(url, key, payload=None, timeout=240):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_identity(base, key, model, request=http_json):
    if model["identity"]["kind"] == "models":
        data = request(base + "/models", key)
        if model["model"] not in {m["id"] for m in data["data"]}:
            raise ValueError("Requested model not registered")
        return data
    info = request(base.removesuffix("/v1") + "/get_server_info", key)
    fields = {
        k: info.get(k)
        for k in [
            "model_path",
            "served_model_name",
            "enable_lora",
            "lora_paths",
            "version",
        ]
    }
    expected = model["identity"]
    if not expected.get("base_path_contains") or expected[
        "base_path_contains"
    ] not in str(fields["model_path"]):
        raise ValueError("Base model identity mismatch")
    if model["lora_path"] is None:
        if fields["lora_paths"]:
            raise ValueError(
                "Baseline server unexpectedly has adapters; use a dedicated base endpoint"
            )
    else:
        adapters = fields["lora_paths"] or []
        adapter = next(
            (a for a in adapters if a["lora_name"] == model["lora_path"]), None
        )
        if (
            not fields["enable_lora"]
            or adapter is None
            or not all(
                fragment in adapter["lora_path"]
                for fragment in expected["adapter_path_contains"]
            )
        ):
            raise ValueError("Adapter identity mismatch")
    return fields


def result_from_response(case, payload, response):
    choices = response.get("choices", [])
    if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
        raise ValueError("Expected exactly one chat completion")
    choice = choices[0]
    content = choice["message"].get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError(
            "Non-text content cannot be exported without a protocol change"
        )
    raw = content if content is not None else ""
    return {
        "case_id": case["id"],
        "input_sha256": digest(case),
        "request": payload,
        "response": response,
        "raw_response": raw,
        "finish_reason": choice.get("finish_reason"),
        "flags": {
            "empty": not raw.strip(),
            "length_limited": choice.get("finish_reason") == "length",
            "has_training_tags": any(
                t in raw for t in ["<output>", "<reasoning>", "<think>"]
            ),
            "separate_reasoning": bool(choice["message"].get("reasoning_content")),
        },
    }


def generate_case(root, case, study, model, base, key, retry=False, request=http_json):
    folder = root / "records" / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    payload = payload_for(case, study, model)
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["request"] != payload or result["input_sha256"] != digest(case):
            raise ValueError("Saved result does not match current input")
        return "cached"
    attempts = sorted(folder.glob("attempt-*"))
    for attempt in attempts:
        if not (attempt / "request.json").exists():
            # Crash between mkdir and the atomic request write: HTTP not started.
            if not retry:
                return "needs_retry_approval"
            continue
        if read_json(attempt / "request.json") != payload:
            raise ValueError("Attempt input mismatch")
        if (attempt / "response.json").exists():
            try:
                result = result_from_response(
                    case, payload, read_json(attempt / "response.json")
                )
            except (ValueError, KeyError, TypeError):
                if not retry:
                    return "needs_retry_approval"
                continue
            write_json(result_path, result)
            return "recovered"
    if attempts and not retry:
        return "needs_retry_approval"
    attempt = folder / ("attempt-" + uuid.uuid4().hex)
    attempt.mkdir()
    write_json(attempt / "request.json", payload)
    write_json(attempt / "started.json", {"at": datetime.now(UTC).isoformat()})
    started = time.monotonic()
    try:
        response = request(base + "/chat/completions", key, payload)
        # Save full provider response before any interpretation.
        write_json(attempt / "response.json", response)
        result = result_from_response(case, payload, response)
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(result_path, result)
    except Exception as error:
        # No exception message/body: a provider may echo credentials in errors.
        write_json(
            attempt / "error.json",
            {
                "type": type(error).__name__,
                "http_status": error.code
                if isinstance(error, urllib.error.HTTPError)
                else None,
            },
        )
        return "failed"
    return "generated"


def run(
    dataset_path,
    study_path,
    model_path,
    output,
    workers=2,
    retry=False,
    request=http_json,
):
    if workers < 1:
        raise ValueError("workers must be positive")
    data = load_dataset(dataset_path)
    study, model = load_yaml(study_path), load_yaml(model_path)
    validate_configs(study, model)
    base = os.environ[model["endpoint_env"]].rstrip("/")
    if not base.startswith(("https://", "http://")) or not base.endswith("/v1"):
        raise ValueError("Endpoint must be an HTTP(S) URL ending in /v1")
    parsed = urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Do not embed credentials or query parameters in endpoints")
    key = os.environ[model["key_env"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = check_identity(base, key, model, request)
        manifest = {
            "schema_version": 1,
            "dataset_fingerprint": data["fingerprint"],
            "study": study,
            "model_config": model,
            "endpoint": base,
            "identity": identity,
            "code_sha256": hashlib_code(),
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError(
                    "Run inputs/code/server identity changed; use a new output directory"
                )
            if read_json(output / "dataset.json") != data:
                raise ValueError("Run dataset snapshot changed")
        else:
            if (output / "records").exists():
                raise ValueError("Records without a manifest cannot be resumed")
            write_json(output / "dataset.json", data)
            write_json(manifest_path, manifest)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            statuses = []
            for case, status in zip(
                data["cases"],
                pool.map(
                    lambda c: generate_case(
                        output, c, study, model, base, key, retry, request
                    ),
                    data["cases"],
                ),
            ):
                statuses.append(status)
                print(f"{case['id']}: {status}", flush=True)
        counts = {s: statuses.count(s) for s in sorted(set(statuses))}
        complete = all(s in {"cached", "generated", "recovered"} for s in statuses)
        write_json(
            output / "status.json",
            {"complete": complete, "counts": counts, "total": len(statuses)},
        )
        print(json.dumps(counts), flush=True)
        return complete


def hashlib_code():
    here = Path(__file__).parent
    return digest(
        {name: (here / name).read_text() for name in ["eval.py", "common.py"]}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--retry-incomplete",
        action="store_true",
        help="Explicitly retry failed or uncertain requests; may duplicate a billed request",
    )
    args = parser.parse_args()
    if not run(
        args.data,
        args.study,
        args.model_config,
        args.output,
        args.workers,
        args.retry_incomplete,
    ):
        raise SystemExit("Incomplete run; inspect status and attempts before retrying")


if __name__ == "__main__":
    main()
