from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import asyncio
import json
import logging
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    AsyncOpenAI = None

if TYPE_CHECKING:
    from configs import TutorConfig
    from datasets import Dataset
    from workflow import TutorAgentWorkflow


logger = logging.getLogger("TutorLLMJudgeFilter")
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_TUTOR_DIR = _THIS_DIR.parent
_REPO_ROOT = _TUTOR_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_TUTOR_DIR))

from examples.tutor.core.types import PublicHistoryState, StudentTurnState  # noqa: E402


DEFAULT_CONFIG_PATH = "examples/tutor/configs/math/baseline.yaml"
DEFAULT_BASE_URL = "http://127.0.0.1:30008/v1"
DEFAULT_MODEL = "default"
DEFAULT_SOLVER_SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem independently. "
    "Show your reasoning if useful. Put the final answer in the last "
    "\\boxed{...}. Do not use any answer key or hidden solution."
)
DEFAULT_SOLVER_USER_TEMPLATE = """\
Task:
{task}

Solve the problem. Put your final answer in \\boxed{{}}.
"""


@dataclass(slots=True)
class AttemptRecord:
    attempt: int
    answer: str
    error: str | None
    correct: bool
    feedback: str
    raw_result: dict[str, Any]
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class ClassifiedRow:
    index: int
    item_id: int | str
    stage: str
    status: str
    kept: bool
    attempts: list[AttemptRecord]
    error: str | None = None


class TeacherSolverClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        request_params: dict[str, Any],
        concurrency: int,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("The openai package is required for teacher calls.")
        self.model = model
        self.request_params = dict(request_params)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def solve(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, dict[str, Any], str | None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_kwargs, extra_body = request_params_for_create(self.request_params)
        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body=extra_body,
                    **request_kwargs,
                )
        except Exception as exc:
            return "", {}, str(exc)
        content = response.choices[0].message.content or ""
        return content.strip(), response_usage_to_dict(response), None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate tutor train/test filters using the configured LLM answer "
            "judge fallback. Rows solved by the initial student are dropped; rows "
            "not solved by the teacher are dropped."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--input",
        default="",
        help="Input HuggingFace dataset path. Defaults to config.train_dataset.path.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output HuggingFace dataset path. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="JSON report path. Defaults to '<output>_llm_judge_filter_report.json'.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to filter, or 'all'. Unselected splits are copied unchanged.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="", help="Defaults to OPENAI_API_KEY, then EMPTY.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--student-attempts", type=int, default=1)
    parser.add_argument("--teacher-attempts", type=int, default=1)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Maximum concurrent calls. Defaults to config.auxiliary_model.max_concurrent_calls.",
    )
    parser.add_argument(
        "--thinking",
        choices=["config", "on", "off", "unset"],
        default="config",
        help=(
            "Whether to send extra_body.chat_template_kwargs.enable_thinking. "
            "'config' uses config.enable_thinking for teacher and "
            "auxiliary_model.enable_thinking for student/judge."
        ),
    )
    parser.add_argument(
        "--request-params",
        default="",
        help="Additional chat.completions.create kwargs as a JSON object.",
    )
    parser.add_argument(
        "--request-params-file",
        type=Path,
        default=None,
        help="JSON file containing additional chat.completions.create kwargs.",
    )
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--user-template-file", type=Path, default=None)
    parser.add_argument("--keep-on-error", action="store_true")
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_object_arg(value: str, *, label: str) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def load_request_params(args: argparse.Namespace) -> dict[str, Any]:
    params = load_json_object_arg(args.request_params, label="--request-params")
    if args.request_params_file is not None:
        file_params = json.loads(args.request_params_file.read_text(encoding="utf-8"))
        if not isinstance(file_params, dict):
            raise ValueError("--request-params-file must contain a JSON object.")
        params = merge_dicts(params, file_params)
    return params


def resolve_thinking(choice: str, default: bool) -> bool | None:
    if choice == "unset":
        return None
    if choice == "on":
        return True
    if choice == "off":
        return False
    return bool(default)


def with_thinking_param(params: dict[str, Any], enable_thinking: bool | None) -> dict[str, Any]:
    if enable_thinking is None:
        return dict(params)
    return merge_dicts(
        params,
        {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                }
            }
        },
    )


def build_aux_request_params(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    params = merge_dicts(dict(auxiliary_model.request_params), load_request_params(args))
    return with_thinking_param(
        params,
        resolve_thinking(args.thinking, bool(auxiliary_model.enable_thinking)),
    )


def build_teacher_request_params(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    params = load_request_params(args)
    params.setdefault("temperature", float(config.gconfig.temperature))
    if getattr(config.gconfig, "top_p", None) is not None:
        params.setdefault("top_p", float(config.gconfig.top_p))
    if "max_tokens" in params and "max_completion_tokens" not in params:
        params["max_completion_tokens"] = int(params.pop("max_tokens"))
    params.setdefault("max_completion_tokens", int(config.gconfig.max_new_tokens))
    return with_thinking_param(
        params,
        resolve_thinking(args.thinking, bool(config.enable_thinking)),
    )


def request_params_for_create(params: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    request_kwargs = {
        key: value for key, value in params.items() if key != "extra_body" and value is not None
    }
    extra_body = params.get("extra_body")
    return request_kwargs, extra_body


def response_usage_to_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return dict(usage.model_dump())
    if isinstance(usage, dict):
        return dict(usage)
    return dict(getattr(usage, "__dict__", {}))


def format_user_prompt(template: str, task: str) -> str:
    try:
        return template.format(task=task)
    except KeyError as exc:
        raise ValueError("User template may only reference the '{task}' placeholder.") from exc


def load_prompt(path: Path | None, default: str) -> str:
    if path is None:
        return default
    return path.read_text(encoding="utf-8").strip()


def item_id_from_row(row: dict[str, Any], index: int) -> int | str:
    value = row.get("id", index)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def resolve_splits(requested: list[str], dataset: Any) -> list[str]:
    if "all" in requested:
        return list(dataset.keys())
    missing = [split for split in requested if split not in dataset]
    if missing:
        raise ValueError(
            f"Unknown split(s) {missing}; available splits are {list(dataset.keys())}"
        )
    return requested


def build_workflow(
    config: TutorConfig,
    args: argparse.Namespace,
    *,
    aux_request_params: dict[str, Any],
    max_concurrency: int,
) -> TutorAgentWorkflow:
    from workflow import TutorAgentWorkflow

    auxiliary_model = config.auxiliary_model
    reward = config.reward
    pairwise = reward.pairwise
    aux_thinking = resolve_thinking(args.thinking, bool(auxiliary_model.enable_thinking))
    return TutorAgentWorkflow(
        answer_scorer=config.answer_scorer,
        max_turns=config.max_turns,
        enable_thinking=config.enable_thinking,
        enable_leak_check=config.enable_leak_check,
        aux_mode="api",
        aux_enable_thinking=bool(aux_thinking),
        aux_base_url=args.base_url or auxiliary_model.base_url,
        aux_model=args.model or auxiliary_model.model,
        aux_api_key=args.api_key or auxiliary_model.api_key,
        aux_timeout=int(args.timeout or auxiliary_model.timeout),
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=max_concurrency,
        aux_request_params=aux_request_params,
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        assign_success_reward=reward.assign_success_reward,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        enable_turn_penalty=reward.enable_turn_penalty,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        teacher_system_prompt=config.teacher_system_prompt,
        teacher_show_ground_truth=config.teacher_show_ground_truth,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        answer_judge_system_prompt=config.answer_judge_system_prompt,
        summary_system_prompt=config.summary_system_prompt,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        pairwise_reward_enabled=pairwise.enabled,
        pairwise_reference_lag_steps=pairwise.reference_lag_steps,
        pairwise_reward_scale=pairwise.scale,
        pairwise_compare_all_turns=pairwise.compare_all_turns,
        pairwise_judge_both_incorrect=pairwise.judge_both_incorrect,
    )


async def classify_pre_solved_row(
    *,
    workflow: TutorAgentWorkflow,
    answer_judge_caller: Any,
    row: dict[str, Any],
    index: int,
    attempts: int,
    keep_on_error: bool,
) -> ClassifiedRow:
    item_id = item_id_from_row(row, index)
    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    attempt_rows: list[AttemptRecord] = []

    for attempt_idx in range(1, attempts + 1):
        answer, error = await workflow._run_student(
            StudentTurnState(
                task=task,
                public_history=PublicHistoryState(),
                previous_student_output="",
                latest_tutor_visible_output="(none, produce the first answer attempt)",
            )
        )
        if error is not None:
            attempt_rows.append(
                AttemptRecord(
                    attempt=attempt_idx,
                    answer="",
                    error=error,
                    correct=False,
                    feedback="Student call failed.",
                    raw_result={},
                )
            )
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                stage="pre_solve",
                status="error",
                kept=keep_on_error,
                attempts=attempt_rows,
                error=error,
            )

        judge_result = await workflow._score_answer_async(
            task,
            ground_truth,
            answer,
            answer_judge_caller=answer_judge_caller,
        )
        attempt_rows.append(
            AttemptRecord(
                attempt=attempt_idx,
                answer=answer,
                error=None,
                correct=bool(judge_result.correct),
                feedback=judge_result.feedback,
                raw_result=judge_result.raw_result,
            )
        )
        if judge_result.correct:
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                stage="pre_solve",
                status="pre_solved",
                kept=False,
                attempts=attempt_rows,
            )

    return ClassifiedRow(
        index=index,
        item_id=item_id,
        stage="pre_solve",
        status="student_unsolved",
        kept=True,
        attempts=attempt_rows,
    )


async def classify_teacher_solved_row(
    *,
    client: Any,
    workflow: TutorAgentWorkflow,
    answer_judge_caller: Any,
    system_prompt: str,
    user_template: str,
    row: dict[str, Any],
    index: int,
    attempts: int,
    keep_on_error: bool,
) -> ClassifiedRow:
    item_id = item_id_from_row(row, index)
    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    attempt_rows: list[AttemptRecord] = []
    saw_successful_call = False

    for attempt_idx in range(1, attempts + 1):
        answer, usage, error = await client.solve(
            system_prompt=system_prompt,
            user_prompt=format_user_prompt(user_template, task),
        )
        if error is not None:
            attempt_rows.append(
                AttemptRecord(
                    attempt=attempt_idx,
                    answer="",
                    error=error,
                    correct=False,
                    feedback="Teacher call failed.",
                    raw_result={},
                    usage=usage,
                )
            )
            continue

        saw_successful_call = True
        judge_result = await workflow._score_answer_async(
            task,
            ground_truth,
            answer,
            answer_judge_caller=answer_judge_caller,
        )
        attempt_rows.append(
            AttemptRecord(
                attempt=attempt_idx,
                answer=answer,
                error=None,
                correct=bool(judge_result.correct),
                feedback=judge_result.feedback,
                raw_result=judge_result.raw_result,
                usage=usage,
            )
        )
        if judge_result.correct:
            return ClassifiedRow(
                index=index,
                item_id=item_id,
                stage="teacher_solve",
                status="teacher_solved",
                kept=True,
                attempts=attempt_rows,
            )

    if saw_successful_call:
        return ClassifiedRow(
            index=index,
            item_id=item_id,
            stage="teacher_solve",
            status="teacher_unsolved",
            kept=False,
            attempts=attempt_rows,
        )
    return ClassifiedRow(
        index=index,
        item_id=item_id,
        stage="teacher_solve",
        status="error",
        kept=keep_on_error,
        attempts=attempt_rows,
        error="All teacher attempts failed.",
    )


async def classify_split_pre_solved(
    *,
    workflow: TutorAgentWorkflow,
    answer_judge_caller: Any,
    dataset: Dataset,
    split_name: str,
    attempts: int,
    keep_on_error: bool,
    limit: int,
    log_every: int = 10,
) -> list[ClassifiedRow]:
    size = len(dataset) if limit <= 0 else min(limit, len(dataset))
    processed = 0
    log_lock = asyncio.Lock()

    async def _run(index: int) -> ClassifiedRow:
        nonlocal processed
        result = await classify_pre_solved_row(
            workflow=workflow,
            answer_judge_caller=answer_judge_caller,
            row=dict(dataset[index]),
            index=index,
            attempts=attempts,
            keep_on_error=keep_on_error,
        )
        async with log_lock:
            processed += 1
            if processed == size or processed % max(1, log_every) == 0:
                logger.info("Pre-solve classified %s/%s rows in '%s'", processed, size, split_name)
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


async def classify_split_teacher_solved(
    *,
    client: Any,
    workflow: TutorAgentWorkflow,
    answer_judge_caller: Any,
    system_prompt: str,
    user_template: str,
    dataset: Dataset,
    split_name: str,
    attempts: int,
    keep_on_error: bool,
    log_every: int = 10,
) -> list[ClassifiedRow]:
    processed = 0
    log_lock = asyncio.Lock()
    size = len(dataset)

    async def _run(index: int) -> ClassifiedRow:
        nonlocal processed
        result = await classify_teacher_solved_row(
            client=client,
            workflow=workflow,
            answer_judge_caller=answer_judge_caller,
            system_prompt=system_prompt,
            user_template=user_template,
            row=dict(dataset[index]),
            index=index,
            attempts=attempts,
            keep_on_error=keep_on_error,
        )
        async with log_lock:
            processed += 1
            if processed == size or processed % max(1, log_every) == 0:
                logger.info("Teacher-solve classified %s/%s rows in '%s'", processed, size, split_name)
        return result

    return list(await asyncio.gather(*[_run(index) for index in range(size)]))


def attempt_to_report(
    attempt: AttemptRecord,
    *,
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "attempt": attempt.attempt,
        "error": attempt.error,
        "correct": attempt.correct,
        "feedback": attempt.feedback,
        "extracted_answer": attempt.raw_result.get("extracted_answer"),
        "target_answer": attempt.raw_result.get("target_answer"),
        "normalized_prediction": attempt.raw_result.get("normalized_prediction"),
        "normalized_target": attempt.raw_result.get("normalized_target"),
        "answer_judge": attempt.raw_result.get("answer_judge"),
    }
    if attempt.usage is not None:
        row["usage"] = attempt.usage
    if save_outputs:
        row["answer"] = attempt.answer
    else:
        row["answer_preview"] = attempt.answer[: max(0, int(preview_chars))]
        row["answer_chars"] = len(attempt.answer)
    return row


def rows_to_report(
    results: list[ClassifiedRow],
    *,
    save_outputs: bool,
    preview_chars: int,
) -> list[dict[str, Any]]:
    return [
        {
            "index": result.index,
            "id": result.item_id,
            "stage": result.stage,
            "status": result.status,
            "kept": result.kept,
            "error": result.error,
            "attempts": [
                attempt_to_report(
                    attempt,
                    save_outputs=save_outputs,
                    preview_chars=preview_chars,
                )
                for attempt in result.attempts
            ],
        }
        for result in results
    ]


def summarize_pipeline(
    *,
    input_rows: int,
    pre_results: list[ClassifiedRow],
    teacher_results: list[ClassifiedRow],
    final_original_indices: list[int],
    save_outputs: bool,
    preview_chars: int,
) -> dict[str, Any]:
    return {
        "input_rows": input_rows,
        "evaluated": len(pre_results),
        "kept": len(final_original_indices),
        "dropped_pre_solved": sum(result.status == "pre_solved" for result in pre_results),
        "pre_solve_errors": sum(
            result.stage == "pre_solve" and result.status == "error" for result in pre_results
        ),
        "teacher_solved": sum(result.status == "teacher_solved" for result in teacher_results),
        "teacher_unsolved": sum(result.status == "teacher_unsolved" for result in teacher_results),
        "teacher_errors": sum(result.status == "error" for result in teacher_results),
        "kept_original_indices": final_original_indices,
        "kept_ids": [result.item_id for result in teacher_results if result.kept],
        "pre_solve_rows": rows_to_report(
            pre_results,
            save_outputs=save_outputs,
            preview_chars=preview_chars,
        ),
        "teacher_rows": rows_to_report(
            teacher_results,
            save_outputs=save_outputs,
            preview_chars=preview_chars,
        ),
    }


def default_report_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}_llm_judge_filter_report.json")


async def main_async(args: argparse.Namespace) -> None:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to run the tutor filter.") from exc
    from configs import TutorConfig

    from areal.api.cli_args import load_expr_config

    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    input_path = Path(args.input or config.train_dataset.path).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    if output_path is None and not args.dry_run:
        raise ValueError("Pass --output or use --dry-run.")
    report_path = (
        Path(args.report).resolve()
        if args.report
        else default_report_path(output_path if output_path is not None else input_path)
    )
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    max_concurrency = max(
        1,
        int(args.concurrency)
        if args.concurrency and args.concurrency > 0
        else int(config.auxiliary_model.max_concurrent_calls),
    )
    student_attempts = max(1, int(args.student_attempts))
    teacher_attempts = max(1, int(args.teacher_attempts))
    aux_request_params = build_aux_request_params(args, config)
    teacher_request_params = build_teacher_request_params(args, config)
    workflow = build_workflow(
        config,
        args,
        aux_request_params=aux_request_params,
        max_concurrency=max_concurrency,
    )
    answer_judge_caller = workflow._make_answer_judge_caller()
    teacher_client = TeacherSolverClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout=float(args.timeout),
        request_params=teacher_request_params,
        concurrency=max_concurrency,
    )
    system_prompt = load_prompt(args.system_prompt_file, DEFAULT_SOLVER_SYSTEM_PROMPT)
    user_template = load_prompt(args.user_template_file, DEFAULT_SOLVER_USER_TEMPLATE)

    loaded = load_from_disk(str(input_path))
    is_dataset_dict = isinstance(loaded, DatasetDict)
    dataset = loaded if is_dataset_dict else DatasetDict({"train": loaded})
    selected_splits = resolve_splits(args.splits, dataset)

    filtered_splits: dict[str, Any] = {}
    report: dict[str, Any] = {
        "input": str(input_path),
        "output": None if args.dry_run else str(output_path),
        "config": str(Path(args.config).resolve()),
        "answer_scorer": config.answer_scorer,
        "answer_judge_enabled": bool(config.auxiliary_model.answer_judge_enabled),
        "base_url": args.base_url,
        "model": args.model,
        "max_concurrency": max_concurrency,
        "student_attempts": student_attempts,
        "teacher_attempts": teacher_attempts,
        "keep_on_error": bool(args.keep_on_error),
        "save_outputs": bool(args.save_outputs),
        "preview_chars": int(args.preview_chars),
        "auxiliary_request_params": aux_request_params,
        "teacher_request_params": teacher_request_params,
        "prompt": {
            "system_prompt": system_prompt,
            "user_template": user_template,
        },
        "selected_splits": selected_splits,
        "splits": {},
        "dry_run": bool(args.dry_run),
    }

    for split_name, split_dataset in dataset.items():
        if split_name not in selected_splits:
            filtered_splits[split_name] = split_dataset
            report["splits"][split_name] = {
                "input_rows": len(split_dataset),
                "kept": len(split_dataset),
                "copied_unchanged": True,
            }
            continue

        logger.info("Filtering split '%s' with %s rows", split_name, len(split_dataset))
        pre_results = await classify_split_pre_solved(
            workflow=workflow,
            answer_judge_caller=answer_judge_caller,
            dataset=split_dataset,
            split_name=split_name,
            attempts=student_attempts,
            keep_on_error=bool(args.keep_on_error),
            limit=max(0, int(args.limit)),
        )
        pre_keep_indices = [result.index for result in pre_results if result.kept]
        candidate_dataset = split_dataset.select(pre_keep_indices)
        teacher_results = await classify_split_teacher_solved(
            client=teacher_client,
            workflow=workflow,
            answer_judge_caller=answer_judge_caller,
            system_prompt=system_prompt,
            user_template=user_template,
            dataset=candidate_dataset,
            split_name=split_name,
            attempts=teacher_attempts,
            keep_on_error=bool(args.keep_on_error),
        )
        final_original_indices = [
            pre_keep_indices[result.index] for result in teacher_results if result.kept
        ]
        filtered_splits[split_name] = split_dataset.select(final_original_indices)
        split_summary = summarize_pipeline(
            input_rows=len(split_dataset),
            pre_results=pre_results,
            teacher_results=teacher_results,
            final_original_indices=final_original_indices,
            save_outputs=bool(args.save_outputs),
            preview_chars=int(args.preview_chars),
        )
        if args.limit and args.limit > 0:
            split_summary["debug_limit"] = int(args.limit)
        report["splits"][split_name] = split_summary
        logger.info(
            "Split '%s': kept=%s dropped_pre_solved=%s teacher_unsolved=%s errors=%s",
            split_name,
            split_summary["kept"],
            split_summary["dropped_pre_solved"],
            split_summary["teacher_unsolved"],
            split_summary["pre_solve_errors"] + split_summary["teacher_errors"],
        )

    if output_path is not None and not args.dry_run:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. Pass --overwrite to replace it."
                )
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dataset = DatasetDict(filtered_splits) if is_dataset_dict else filtered_splits["train"]
        output_dataset.save_to_disk(str(output_path))
        logger.info("Saved filtered dataset to %s", output_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved filter report to %s", report_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
