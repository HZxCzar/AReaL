from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(pathlib.Path(__file__).parent))
from configs import TutorConfig
import workflow as tutor_workflow_module
from workflow import TutorAgentWorkflow

from areal.api.cli_args import load_expr_config
from examples.common.trace_utils import (
    LoggedTeacherClient,
    TraceSink,
    load_demo_rows,
    patch_teacher_factory,
)
from examples.common.openai_utils import make_teacher_client


class DemoTutorWorkflow(TutorAgentWorkflow):
    def __init__(self, *args, trace_sink: TraceSink, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink
        self.root_task = ""
        self.last_judge_outputs: list[dict[str, Any]] = []

    async def run(self, data: dict[str, Any], **extra_kwargs):
        self.root_task = str(data["task"])
        return await super().run(data, **extra_kwargs)

    async def _run_student(
        self,
        task: str,
        teacher_action: str | None,
        history: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        if teacher_action is None and task == self.root_task:
            label = "student_init"
        elif teacher_action is None:
            label = "transfer_student"
        else:
            label = "student"
        prompt = self._build_student_prompt(task, teacher_action, history)
        messages = [
            {"role": "system", "content": self.student_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.trace_sink.append_messages(f"{label}_input", messages)
        try:
            answer = await self.aux_caller.call_text(messages)
            self.trace_sink.append(label, answer)
            return answer, None
        except Exception as exc:
            error = f"Student call failed: {exc}"
            self.trace_sink.append(label, error)
            return "", error

    async def _run_leak_check(
        self, task: str, ground_truth: str, teacher_action: str
    ):
        prompt = self._build_leak_check_prompt(task, ground_truth, teacher_action)
        messages = [
            {"role": "system", "content": self.leak_check_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.trace_sink.append_messages("leak_check_input", messages)
        result = await super()._run_leak_check(task, ground_truth, teacher_action)
        self.trace_sink.append("leak_check", result.raw_output or result.feedback)
        return result

    async def _run_transfer_generation(self, task: str, ground_truth: str):
        prompt = self._build_transfer_generation_prompt(task, ground_truth)
        messages = [
            {"role": "system", "content": self.generator_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.trace_sink.append_messages("transfer_generator_input", messages)
        result = await super()._run_transfer_generation(task, ground_truth)
        self.trace_sink.append(
            "transfer_generator", result.raw_output or result.parse_error or "(empty)"
        )
        return result

    def _score_aime_answer(self, task: str, ground_truth: str, student_answer: str):
        result = super()._score_aime_answer(task, ground_truth, student_answer)
        label = "judge" if task == self.root_task else "transfer_judge"
        self.trace_sink.append(label, result.raw_output)
        self.last_judge_outputs.append(
            {
                "label": label,
                "correct": result.correct,
                "feedback": result.feedback,
                "raw_result": result.raw_result,
            }
        )
        return result


def build_workflow_kwargs(config: TutorConfig, trace_sink: TraceSink) -> dict[str, Any]:
    return dict(
        trace_sink=trace_sink,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=config.max_turns,
        aux_base_url=config.aux_base_url,
        aux_model=config.aux_model,
        aux_api_key=config.aux_api_key,
        aux_timeout=config.aux_timeout,
        aux_max_tokens=config.aux_max_tokens,
        aux_temperature=config.aux_temperature,
        aux_top_p=config.aux_top_p,
        max_concurrent_aux_calls=config.max_concurrent_aux_calls,
        api_params_config_path=config.api_params_config_path or None,
        api_params_key=config.api_params_key or None,
        primary_success_reward=config.primary_success_reward,
        transfer_bonus_reward=config.transfer_bonus_reward,
        transfer_success_reward=config.transfer_success_reward,
        transfer_fail_reward=config.transfer_fail_reward,
        token_budget_penalty=config.token_budget_penalty,
        teacher_system_prompt=config.teacher_system_prompt,
        student_system_prompt=config.student_system_prompt,
        judge_system_prompt=config.judge_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        generator_system_prompt=config.generator_system_prompt,
        max_episode_total_tokens=config.gconfig.max_tokens,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
    )


async def _run_one(
    config: TutorConfig,
    row: dict[str, Any],
    output_dir: Path,
    teacher_base_url: str,
    teacher_api_key: str,
    teacher_model: str | None,
):
    sink = TraceSink(output_dir)
    workflow = DemoTutorWorkflow(**build_workflow_kwargs(config, sink))
    teacher_extra_kwargs = {"base_url": teacher_base_url, "api_key": teacher_api_key}
    base_client = make_teacher_client(teacher_extra_kwargs)
    logged_teacher = LoggedTeacherClient(base_client, sink, model_override=teacher_model)

    def _factory(_extra_kwargs):
        return logged_teacher

    with patch_teacher_factory(tutor_workflow_module, _factory):
        rewards = await workflow.run(row, **teacher_extra_kwargs)

    visible_history_summaries = [
        str(record.get("student_visible_summary", ""))
        for record in workflow.last_history
        if not record.get("leak_detected") and record.get("student_visible_summary")
    ]
    turn_totals = [
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        for usage in logged_teacher.logged_usage
    ]
    length_exceeded_turns = [
        idx + 1
        for idx, total in enumerate(turn_totals)
        if total >= config.gconfig.max_tokens
    ]
    summary = {
        "id": row.get("id"),
        "rewards": rewards,
        "teacher_usage": logged_teacher.logged_usage,
        "turn_total_tokens": turn_totals,
        "length_budget": config.gconfig.max_tokens,
        "length_exceeded_turns": length_exceeded_turns,
        "student_visible_history": visible_history_summaries,
        "judge_outputs": workflow.last_judge_outputs,
        "aux_request_config": workflow.aux_caller.request_config,
    }
    meta = {
        "dataset_row": row,
        "teacher_extra_kwargs": teacher_extra_kwargs,
        "teacher_model_override": teacher_model,
        "workflow_config": {
            "max_turns": config.max_turns,
            "max_completion_tokens": config.gconfig.max_new_tokens,
            "max_episode_total_tokens": config.gconfig.max_tokens,
            "aux_base_url": config.aux_base_url,
            "aux_model": config.aux_model,
        },
    }
    sink.dump_json("history.json", workflow.last_history)
    sink.dump_json("summary.json", summary)
    sink.dump_json("meta.json", meta)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-api-key", default="EMPTY")
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    rows = load_demo_rows(config.train_dataset.path, args.split, args.num_rollouts)
    root = Path(args.output_dir or Path(__file__).parent / "demo_output" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, row in enumerate(rows, start=1):
        rollout_dir = root / f"rollout_{index:02d}_{row.get('id', index)}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        summary = asyncio.run(
            _run_one(
                config=config,
                row=row,
                output_dir=rollout_dir,
                teacher_base_url=args.teacher_base_url,
                teacher_api_key=args.teacher_api_key,
                teacher_model=args.teacher_model or None,
            )
        )
        summaries.append(summary)

    (root / "index.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(root)


if __name__ == "__main__":
    main()
