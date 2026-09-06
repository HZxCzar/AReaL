#!/usr/bin/env python3
"""Collect matched turn-1 policy-gradient examples for the four ID students.

For each problem, one pre-solve and one exact turn-1 prompt are shared.  Eight
teacher actions sampled from that prompt are then replayed into every student
environment; each branch follows the normal evaluation workflow through retest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from areal import workflow_context
from areal.api import ModelResponse
from areal.infra.workflow_context import WorkflowContext
from areal.utils.hf_utils import load_hf_tokenizer
from examples.tutor import train as tutor_train
from examples.tutor.configs import TUTOR_EVAL_STUDENT_FIELD
from examples.tutor.core.callers import apply_chat_template, encode_text
from examples.tutor.scripts.evaluate_api_teacher import (
    ApiTeacherClient,
    RecordingTutorWorkflow,
    build_eval_workflow_kwargs,
    close_workflow_api_clients,
    effective_eval_presolve_enabled,
    load_experiment_config,
    normalize_base_url,
    prepare_episode_workflow_kwargs,
    resolve_teacher_generation_args,
    stratified_math_subset_indices,
)


ENVIRONMENTS = (
    "none",
    "attempt-diagnosis",
    "contrastive-comparison",
    "subgoal-decomposition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-eval-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--teacher-urls", required=True)
    parser.add_argument("--student-urls", required=True)
    parser.add_argument("--teacher-model", default="qwen3-8b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problems", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--branch-concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--episode-timeout", type=float, default=600.0)
    return parser.parse_args()


class FirstTurnCaptured(RuntimeError):
    def __init__(self, *, messages: list[dict[str, str]], response: ModelResponse, raw: str, pre_solve: Any):
        super().__init__("first teacher turn captured")
        self.messages = deepcopy(messages)
        self.response = response
        self.raw = raw
        self.pre_solve = pre_solve


class CaptureFirstTurnWorkflow(RecordingTutorWorkflow):
    async def _generate_tutor_response(self, tutor_state: Any, **kwargs: Any):
        messages = self._build_tutor_messages(tutor_state)
        response, raw = await super()._generate_tutor_response(
            tutor_state, **kwargs
        )
        if int(tutor_state.turn_idx) == 1:
            raise FirstTurnCaptured(
                messages=messages,
                response=response,
                raw=raw,
                pre_solve=tutor_state.teacher_pre_solve_result,
            )
        return response, raw


class ForcedFirstTurnWorkflow(RecordingTutorWorkflow):
    def __init__(
        self,
        *args: Any,
        forced_raw: str,
        forced_pre_solve: Any,
        expected_input_tokens: list[int],
        fixed_baseline: float | None,
        **kwargs: Any,
    ) -> None:
        self._forced_raw = forced_raw
        self._forced_pre_solve = forced_pre_solve
        self._expected_input_tokens = list(expected_input_tokens)
        self._fixed_baseline = fixed_baseline
        super().__init__(*args, **kwargs)

    async def _teacher_pre_solve_for_group(self, *args: Any, **kwargs: Any):
        del args, kwargs
        return self._forced_pre_solve, True

    async def _no_teaching_baseline(self, *args: Any, **kwargs: Any):
        if self._fixed_baseline is not None:
            return self._fixed_baseline
        return await super()._no_teaching_baseline(*args, **kwargs)

    async def _generate_tutor_response(self, tutor_state: Any, **kwargs: Any):
        if int(tutor_state.turn_idx) != 1:
            return await super()._generate_tutor_response(tutor_state, **kwargs)
        messages = self._build_tutor_messages(tutor_state)
        input_tokens = apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
        )
        if input_tokens != self._expected_input_tokens:
            raise RuntimeError(
                "Matched test drifted: a branch did not reproduce the shared "
                "turn-1 prompt exactly"
            )
        output_tokens = encode_text(self.tokenizer, self._forced_raw)
        response = ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            tokenizer=self.tokenizer,
        )
        return response, self._forced_raw


def split_urls(raw: str) -> list[str]:
    urls = [normalize_base_url(item.strip()) for item in raw.split(",") if item.strip()]
    if len(urls) != len(ENVIRONMENTS):
        raise ValueError(f"Expected {len(ENVIRONMENTS)} comma-separated URLs")
    return urls


def load_source_rows(source_eval_dir: Path, sample_count: int, seed: int) -> list[dict[str, Any]]:
    trace_dir = (
        source_eval_dir
        / "cells/all-id/qwen3-1.7b-text-original/traces/presolve_on"
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(trace_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = payload.get("dataset_row")
        if not isinstance(row, dict):
            continue
        key = str(row.get("id") or row.get("task") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(dict(row))
    if len(rows) < sample_count:
        raise RuntimeError(
            f"Source evaluation has only {len(rows)} usable rows; requested {sample_count}"
        )
    indices = stratified_math_subset_indices(
        rows, sample_count=sample_count, seed=seed
    )
    return [rows[index] for index in indices]


def student_environment(student: dict[str, Any]) -> str:
    personality = str(student.get("personality") or "none")
    return "none" if personality in {"", "none"} else personality


def make_workflow_kwargs(
    *,
    config: Any,
    students: list[dict[str, Any]],
    tokenizer: Any,
    teacher_urls: list[str],
    student_urls: list[str],
) -> dict[str, dict[str, Any]]:
    generation_args = SimpleNamespace(
        teacher_temperature=None,
        teacher_top_p=None,
        teacher_max_tokens=None,
        presolve_attempts=0,
        presolve_max_tokens=None,
    )
    resolve_teacher_generation_args(generation_args, config)
    result: dict[str, dict[str, Any]] = {}
    by_environment = {student_environment(student): student for student in students}
    for env_index, env in enumerate(ENVIRONMENTS):
        if env not in by_environment:
            raise RuntimeError(f"Config has no student for {env}")
        student = deepcopy(by_environment[env])
        student["base_url"] = student_urls[env_index]
        kwargs = build_eval_workflow_kwargs(
            config=config,
            student_models=[student],
            tokenizer=tokenizer,
            args=generation_args,
            presolve_enabled=effective_eval_presolve_enabled(config),
        )
        kwargs["aux_base_url"] = teacher_urls[env_index]
        kwargs["debug_trace_dir"] = None
        result[env] = kwargs
    return result


async def capture_prompt(
    *,
    row: dict[str, Any],
    workflow_kwargs: dict[str, Any],
    teacher_client: ApiTeacherClient,
    task_id: int,
) -> FirstTurnCaptured:
    workflow = CaptureFirstTurnWorkflow(
        **prepare_episode_workflow_kwargs(workflow_kwargs)
    )
    workflow_context.set(
        WorkflowContext(is_eval=True, task_id=task_id, lora_version=None)
    )
    try:
        await workflow._run_episode(dict(row), external_client=teacher_client)
    except FirstTurnCaptured as captured:
        return captured
    finally:
        await close_workflow_api_clients(workflow)
    raise RuntimeError("Workflow ended without producing a first teacher turn")


async def sample_candidates(
    *,
    captured: FirstTurnCaptured,
    teacher_client: ApiTeacherClient,
    count: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[str]:
    outputs = [captured.raw]

    async def sample(index: int) -> str:
        response = await teacher_client.chat.completions.create(
            model="default",
            messages=captured.messages,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
            seed=seed + index,
        )
        return response.choices[0].message.content or ""

    if count > 1:
        outputs.extend(await asyncio.gather(*(sample(index) for index in range(1, count))))
    return outputs


async def run_branch(
    *,
    row: dict[str, Any],
    env: str,
    candidate_index: int,
    raw: str,
    captured: FirstTurnCaptured,
    workflow_kwargs: dict[str, Any],
    teacher_client: ApiTeacherClient,
    fixed_baseline: float | None,
    task_id: int,
    timeout: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for execution_try in range(1, 4):
        workflow = ForcedFirstTurnWorkflow(
            **prepare_episode_workflow_kwargs(workflow_kwargs),
            forced_raw=raw,
            forced_pre_solve=captured.pre_solve,
            expected_input_tokens=list(captured.response.input_tokens),
            fixed_baseline=fixed_baseline,
        )
        workflow_context.set(
            WorkflowContext(
                is_eval=True,
                task_id=task_id * 100 + candidate_index,
                lora_version=None,
            )
        )
        try:
            async with asyncio.timeout(timeout):
                await workflow._run_episode(dict(row), external_client=teacher_client)
            if workflow.captured_stats is None or not workflow.last_traces:
                raise RuntimeError("Branch emitted no complete episode statistics")
            stats = workflow.captured_stats
            baseline = stats.get("no_teaching_baseline")
            if baseline is None:
                raise RuntimeError("Branch emitted no no-teaching baseline")
            outcome = float(
                workflow._free_chat_outcome_score(
                    workflow.last_student_generalization_results
                )
            )
            first_turn = workflow.last_traces[0]
            credit = not bool(first_turn.personality_gated) and not bool(
                first_turn.leak_masked
            )
            improvement = outcome - float(baseline)
            return {
                "environment": env,
                "candidate_index": candidate_index,
                "baseline": float(baseline),
                "outcome": outcome,
                "improvement": improvement,
                "credit": credit,
                "return": improvement if credit else 0.0,
                "personality_gated": bool(first_turn.personality_gated),
                "leak_masked": bool(first_turn.leak_masked),
                "format_error": bool(first_turn.tutor_format_error),
                "termination_reason": str(stats.get("termination_reason") or ""),
                "turns": len(workflow.last_traces),
                "execution_try": execution_try,
            }
        except Exception as exc:  # noqa: BLE001 - whole-episode retry
            last_error = exc
        finally:
            await close_workflow_api_clients(workflow)
    assert last_error is not None
    raise RuntimeError(
        f"{env} candidate {candidate_index} failed three times: {last_error}"
    ) from last_error


async def collect_problem(
    *,
    problem_index: int,
    row: dict[str, Any],
    workflow_kwargs: dict[str, dict[str, Any]],
    teacher_clients: dict[str, ApiTeacherClient],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_row = dict(row)
    source_row[TUTOR_EVAL_STUDENT_FIELD] = workflow_kwargs["none"]["student_models"][0]["name"]
    captured = await capture_prompt(
        row=source_row,
        workflow_kwargs=workflow_kwargs["none"],
        teacher_client=teacher_clients["none"],
        task_id=problem_index,
    )
    gconfig = workflow_kwargs["none"]["gconfig"]
    candidates = await sample_candidates(
        captured=captured,
        teacher_client=teacher_clients["none"],
        count=args.candidates,
        seed=args.seed + problem_index * 1000,
        temperature=float(gconfig.temperature),
        top_p=float(gconfig.top_p),
        max_tokens=int(gconfig.max_new_tokens),
    )

    # Compute the group's no-teaching baseline once, exactly as training caches it.
    none_first = await run_branch(
        row=source_row,
        env="none",
        candidate_index=0,
        raw=candidates[0],
        captured=captured,
        workflow_kwargs=workflow_kwargs["none"],
        teacher_client=teacher_clients["none"],
        fixed_baseline=None,
        task_id=problem_index,
        timeout=args.episode_timeout,
    )
    baseline = float(none_first["baseline"])
    outcomes: dict[str, list[dict[str, Any]]] = {env: [] for env in ENVIRONMENTS}
    outcomes["none"].append(none_first)
    semaphore = asyncio.Semaphore(max(1, args.branch_concurrency))

    async def branch(env: str, candidate_index: int):
        branch_row = dict(row)
        branch_row[TUTOR_EVAL_STUDENT_FIELD] = workflow_kwargs[env]["student_models"][0]["name"]
        async with semaphore:
            return await run_branch(
                row=branch_row,
                env=env,
                candidate_index=candidate_index,
                raw=candidates[candidate_index],
                captured=captured,
                workflow_kwargs=workflow_kwargs[env],
                teacher_client=teacher_clients[env],
                fixed_baseline=baseline,
                task_id=problem_index,
                timeout=args.episode_timeout,
            )

    jobs = [
        (env, candidate_index)
        for candidate_index in range(args.candidates)
        for env in ENVIRONMENTS
        if not (env == "none" and candidate_index == 0)
    ]
    results = await asyncio.gather(*(branch(env, index) for env, index in jobs))
    for result in results:
        outcomes[result["environment"]].append(result)
    for env in ENVIRONMENTS:
        outcomes[env].sort(key=lambda item: item["candidate_index"])

    return {
        "problem_index": problem_index,
        "id": str(row.get("id") or problem_index),
        "task": str(row.get("task") or ""),
        "metadata": row.get("metadata") or {},
        "prompt_tokens": list(captured.response.input_tokens),
        "prompt_sha256": hashlib.sha256(
            json.dumps(captured.messages, sort_keys=True).encode()
        ).hexdigest(),
        "candidate_raw": candidates,
        "candidate_tokens": [encode_text(tokenizer, raw) for raw in candidates],
        "outcomes": outcomes,
    }


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lm = sum(left) / len(left)
    rm = sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0 else None


def build_gradient_manifest(problems: list[dict[str, Any]]) -> dict[str, Any]:
    examples: dict[str, list[dict[str, Any]]] = {env: [] for env in ENVIRONMENTS}
    advantage_vectors: dict[str, list[float]] = {env: [] for env in ENVIRONMENTS}
    for problem in problems:
        prompt_tokens = problem["prompt_tokens"]
        for env in ENVIRONMENTS:
            outcomes = problem["outcomes"][env]
            returns = [float(item["return"]) for item in outcomes]
            total = sum(returns)
            for candidate_index, (outcome, raw_return) in enumerate(
                zip(outcomes, returns, strict=True)
            ):
                advantage = raw_return - (total - raw_return) / (len(returns) - 1)
                advantage_vectors[env].append(advantage)
                if abs(advantage) <= 1e-12:
                    continue
                output_tokens = problem["candidate_tokens"][candidate_index]
                input_ids = prompt_tokens + output_tokens
                examples[env].append(
                    {
                        "environment": env,
                        "task_id": int(problem["problem_index"]),
                        "trajectory_id": int(problem["problem_index"]) * 100 + candidate_index,
                        "turn_idx": 1,
                        "prompt_len": len(prompt_tokens),
                        "seqlen": len(input_ids),
                        "input_ids": input_ids,
                        "advantage": advantage,
                        "raw_return": raw_return,
                        "credit": bool(outcome["credit"]),
                    }
                )

    pairwise: dict[str, dict[str, Any]] = {env: {} for env in ENVIRONMENTS}
    for left in ENVIRONMENTS:
        for right in ENVIRONMENTS:
            lv = advantage_vectors[left]
            rv = advantage_vectors[right]
            nonzero_pairs = [(a, b) for a, b in zip(lv, rv, strict=True) if a and b]
            opposite = sum(a * b < 0 for a, b in nonzero_pairs)
            pairwise[left][right] = {
                "advantage_correlation": correlation(lv, rv),
                "both_nonzero": len(nonzero_pairs),
                "opposite_sign_rate": opposite / len(nonzero_pairs) if nonzero_pairs else None,
            }
    return {
        "diagnostic": "matched turn-1 reward-v3 outcome policy-gradient cosine",
        "groups_per_environment": len(problems),
        "selected_groups": {
            env: [str(problem["id"]) for problem in problems] for env in ENVIRONMENTS
        },
        "environments": examples,
        "advantage_comparison": pairwise,
    }


async def main_async(args: argparse.Namespace) -> None:
    if args.problems < 1 or args.candidates < 2:
        raise ValueError("--problems must be positive and --candidates must be at least 2")
    teacher_urls = split_urls(args.teacher_urls)
    student_urls = split_urls(args.student_urls)
    os.environ.setdefault("INF_API_KEY", "EMPTY")
    os.environ.setdefault("DEEPSEEK_API_KEY", "EMPTY")
    os.environ["TUTOR_QWEN3_8B_BASE_URL"] = teacher_urls[0]
    os.environ["TUTOR_QWEN3_1_7B_BASE_URL"] = student_urls[0]
    config, students = load_experiment_config(args.config, [])
    tutor_train._apply_eval_average_rollouts(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    workflow_kwargs = make_workflow_kwargs(
        config=config,
        students=students,
        tokenizer=tokenizer,
        teacher_urls=teacher_urls,
        student_urls=student_urls,
    )
    rows = load_source_rows(args.source_eval_dir, args.problems, args.seed)
    request_params = {
        "seed": args.seed,
        "extra_body": {
            # Keep the registered SGLang alias path. Resolving the symlink would
            # produce a different adapter identifier from --lora-paths.
            "lora_path": os.path.abspath(args.adapter),
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    teacher_clients = {
        env: ApiTeacherClient(
            base_url=teacher_urls[index],
            api_key=args.api_key,
            model=args.teacher_model,
            timeout=args.timeout,
            max_retries=1,
            request_params=request_params,
        )
        for index, env in enumerate(ENVIRONMENTS)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    problem_dir = args.output_dir / "problems"
    problem_dir.mkdir(exist_ok=True)
    try:
        for problem_index, row in enumerate(rows):
            path = problem_dir / f"problem_{problem_index:03d}.json"
            if path.is_file():
                print(f"[resume] problem {problem_index + 1}/{len(rows)}", flush=True)
                continue
            print(f"[collect] problem {problem_index + 1}/{len(rows)}", flush=True)
            result = await collect_problem(
                problem_index=problem_index,
                row=row,
                workflow_kwargs=workflow_kwargs,
                teacher_clients=teacher_clients,
                tokenizer=tokenizer,
                args=args,
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            temporary.replace(path)
    finally:
        await asyncio.gather(*(client.close() for client in teacher_clients.values()))

    problems = [
        json.loads((problem_dir / f"problem_{index:03d}.json").read_text())
        for index in range(len(rows))
    ]
    manifest = build_gradient_manifest(problems)
    (args.output_dir / "matched_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n"
    )
    print(
        "[done] "
        + ", ".join(
            f"{env}={len(manifest['environments'][env])} nonzero turn-1 examples"
            for env in ENVIRONMENTS
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
