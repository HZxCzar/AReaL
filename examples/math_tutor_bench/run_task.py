#!/usr/bin/env python3
"""Run one official MathTutorBench task with resumable parallel API calls."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

CHAT_TASKS = {
    "mistake_correction",
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
}
PEDAGOGY_TASKS = {
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
}
NATIVE_NO_THINK = "<think>\n\n</think>\n\n"
OUTPUT_BLOCK = re.compile(r"<output(?:\s[^>]*)?>(.*?)</output\s*>", re.I | re.S)
OPEN_OUTPUT_BLOCK = re.compile(r"<output(?:\s[^>]*)?>(.*)$", re.I | re.S)
REASONING_BLOCK = re.compile(
    r"<reasoning(?:\s[^>]*)?>.*?</reasoning\s*>", re.I | re.S
)
END_ONLY = re.compile(r"\s*<end\s*>\s*(?:</end\s*>)?\s*", re.I | re.S)
LEADING_TEACHER_ROLE = re.compile(r"^\s*Teacher\s*:\s*", re.I)
_THREAD_STATE = threading.local()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def extract_visible_teacher_output(raw: str) -> str:
    """Remove native/custom hidden reasoning without changing ordinary output."""
    text = (raw or "").strip()
    if not text:
        return ""

    # Native Qwen thinking is disabled, but handle a server that still returns it.
    if "</think>" in text:
        text = text.rpartition("</think>")[2].strip()
    elif re.search(r"<think(?:\s[^>]*)?>", text, re.I):
        return ""

    output_matches = OUTPUT_BLOCK.findall(text)
    if output_matches:
        visible = output_matches[-1].strip()
        return "" if END_ONLY.fullmatch(visible) else visible

    # A truncated closing tag should not expose the preceding reasoning. Preserve
    # the text after an opening <output>, which is the actual visible response.
    open_output = OPEN_OUTPUT_BLOCK.search(text)
    if open_output:
        visible = open_output.group(1).strip()
        return "" if END_ONLY.fullmatch(visible) else visible

    if re.search(r"<reasoning(?:\s[^>]*)?>", text, re.I):
        if not re.search(r"</reasoning\s*>", text, re.I):
            return ""
        text = REASONING_BLOCK.sub("", text).strip()

    return "" if END_ONLY.fullmatch(text) else text


def apply_official_stops(text: str, stops: Any) -> str:
    if not text or not stops:
        return text.strip()
    stop_list = [stops] if isinstance(stops, str) else list(stops)

    # Dialogue prompts already establish that the assistant is the teacher. Some
    # chat models harmlessly repeat that role label. MathTutorBench also lists
    # ``Teacher:`` as a stop sequence, so applying the stop literally at offset
    # zero would otherwise erase the entire substantive answer.
    if any(str(stop).strip().lower() == "teacher:" for stop in stop_list):
        text = LEADING_TEACHER_ROLE.sub("", text, count=1)

    cut = len(text)
    for stop in stop_list:
        if not stop:
            continue
        position = text.find(stop)
        if position >= 0:
            cut = min(cut, position)
    return text[:cut].strip()


def client_for(base_url: str, api_key: str, timeout: float) -> OpenAI:
    key = (base_url, api_key, timeout)
    if getattr(_THREAD_STATE, "client_key", None) != key:
        _THREAD_STATE.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        _THREAD_STATE.client_key = key
    return _THREAD_STATE.client


def request_completion(
    *,
    base_url: str,
    api_key: str,
    timeout: float,
    model: str,
    lora_path: str | None,
    prompt: str,
    chat_mode: bool,
    max_tokens: int,
) -> tuple[str, str | None]:
    client = client_for(base_url, api_key, timeout)
    last_error: BaseException | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(min(4 * (2 ** (attempt - 1)), 10))
        try:
            if chat_mode:
                extra_body: dict[str, Any] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
                if lora_path:
                    extra_body["lora_path"] = lora_path
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    seed=42,
                    extra_body=extra_body,
                )
                choice = response.choices[0]
                content = choice.message.content or ""
                reasoning = (
                    getattr(choice.message, "reasoning_content", None)
                    or getattr(choice.message, "reasoning", None)
                    or ""
                )
                # This mirrors upstream's fallback for a reasoning parser that
                # unexpectedly returns an empty content field.
                raw = content if content.strip() or not reasoning else reasoning
            else:
                extra_body = {"lora_path": lora_path} if lora_path else {}
                response = client.completions.create(
                    model=model,
                    prompt=prompt + NATIVE_NO_THINK,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    seed=42,
                    extra_body=extra_body,
                )
                choice = response.choices[0]
                raw = choice.text or ""
            finish_reason = getattr(choice, "finish_reason", None)
            return raw.strip(), str(finish_reason) if finish_reason else None
        except BaseException as error:  # retries match the official runner
            last_error = error
    assert last_error is not None
    raise last_error


def load_existing_records(path: Path, task_config: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    with path.open("rb+") as handle:
        valid_end = 0
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A killed process can leave only its final append incomplete.
                if handle.read().strip():
                    raise RuntimeError(f"corrupt non-final JSONL record in {path}")
                handle.truncate(valid_end)
                break
            if record.get("task_config") != task_config:
                raise RuntimeError(f"record from another task found in {path}")
            index = int(record["index"])
            records[index] = record
            valid_end = handle.tell()
    return records


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--lora-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reparse-only",
        action="store_true",
        help="Rebuild derived artifacts from existing raw responses without generation.",
    )
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    config_file = upstream / "configs" / f"{args.task}.yaml"
    if not config_file.is_file():
        raise SystemExit(f"unknown or missing official task config: {config_file}")
    if args.concurrency < 1 or args.max_tokens < 1 or args.max_samples < 0:
        raise SystemExit("concurrency/max-tokens must be positive and max-samples nonnegative")
    # Official local dataset paths are relative to the upstream repository.
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    import tasks  # noqa: F401,PLC0415 - registers all official task classes
    from registry import TaskRegistry  # noqa: PLC0415
    from tasks.base import TaskConfig  # noqa: PLC0415

    config_dict = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    task_config = TaskConfig(**config_dict)
    task_class = TaskRegistry.get_task(task_config.name)
    task = task_class(task_config)
    examples = task.get_test_examples()
    if args.max_samples:
        examples = examples[: args.max_samples]
    for example in examples:
        example["shots"] = task_config.few_shot_samples

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    existing = load_existing_records(records_path, args.task)
    if any(index < 0 or index >= len(examples) for index in existing):
        raise SystemExit(
            f"existing records in {records_path} do not match the current sample count"
        )
    missing = [index for index in range(len(examples)) if index not in existing]
    if args.reparse_only and missing:
        raise SystemExit(
            f"cannot reparse {args.task}: {len(missing)} raw responses are missing"
        )
    if missing:
        if not args.base_url or not args.model:
            raise SystemExit(
                "--base-url and --model are required for generation"
            )
        if args.lora_path is not None and not args.lora_path.is_dir():
            raise SystemExit(f"LoRA adapter path does not exist: {args.lora_path}")
        # SGLang identifies a preloaded LoRA by the exact path string passed to
        # --lora-paths. Do not resolve this symlink to its checkpoint target.
        lora_request_path = (
            os.path.abspath(args.lora_path) if args.lora_path is not None else None
        )
    else:
        lora_request_path = None
    print(
        f"[task] {args.task}: total={len(examples)} complete={len(existing)} "
        f"remaining={len(missing)} mode={'chat' if args.task in CHAT_TASKS else 'completion'}"
    )

    def build_record(
        index: int, raw: str, finish_reason: str | None
    ) -> dict[str, Any]:
        example = examples[index]
        visible = apply_official_stops(
            extract_visible_teacher_output(raw), task_config.stop
        )
        prediction = task.parse_response(visible)
        target = task.format_ground_truth(example)
        record: dict[str, Any] = {
            "task_config": args.task,
            "task_name": task_config.name,
            "index": index,
            "raw_response": raw,
            "visible_response": visible,
            "finish_reason": finish_reason,
            "prediction": jsonable(prediction),
            "target": jsonable(target),
        }
        if args.task in PEDAGOGY_TASKS:
            record["generation"] = {
                "problem": example.get("question", ""),
                "reference_solution": example.get("reference_solution", "N/A"),
                "dialog_history": example.get("conversation_json", []),
                "dialog_formatted": example.get("dialog_history", ""),
                "ground_truth_response": example.get("ground_truth_response", ""),
                "generated_teacher_utterance": prediction,
            }
        return record

    def evaluate(index: int) -> dict[str, Any]:
        example = examples[index]
        prompt = task.get_system_prompt(example)
        raw, finish_reason = request_completion(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            model=args.model,
            lora_path=lora_request_path,
            prompt=prompt,
            chat_mode=args.task in CHAT_TASKS,
            max_tokens=args.max_tokens,
        )
        return build_record(index, raw, finish_reason)

    if missing:
        pool = ThreadPoolExecutor(max_workers=args.concurrency)
        futures: dict[Future[dict[str, Any]], int] = {
            pool.submit(evaluate, index): index for index in missing
        }
        completed_now = 0
        try:
            with records_path.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    record = future.result()
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[int(record["index"])] = record
                    completed_now += 1
                    if completed_now % 50 == 0 or completed_now == len(missing):
                        print(
                            f"[task] {args.task}: {len(existing)}/{len(examples)} complete",
                            flush=True,
                        )
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    # Always derive parsed responses and metrics from the persisted raw output.
    # This makes post-processing fixes resumable without another model rollout.
    ordered = [
        build_record(
            index,
            str(existing[index].get("raw_response", "")),
            existing[index].get("finish_reason"),
        )
        for index in range(len(examples))
    ]
    write_jsonl(records_path, ordered)
    predictions = [record["prediction"] for record in ordered]
    targets = [record["target"] for record in ordered]
    metrics = jsonable(task.compute_metrics(predictions, targets))
    write_json(
        output / "metrics.json",
        {
            "task_config": args.task,
            "task_name": task_config.name,
            "num_examples": len(ordered),
            "metrics": metrics,
            "decoding": {
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": args.max_tokens,
                "native_thinking": False,
                "mode": "chat" if args.task in CHAT_TASKS else "completion",
            },
        },
    )
    if args.task in PEDAGOGY_TASKS:
        write_json(
            output / "generations.json",
            [record["generation"] for record in ordered],
        )
    print(f"[task] {args.task} metrics: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
