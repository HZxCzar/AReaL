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
from configs import CodeCoachConfig
import workflow as codecoach_workflow_module
from workflow import CodeCoachAgentWorkflow

from areal.api.cli_args import load_expr_config
from examples.common.trace_utils import (
    LoggedTeacherClient,
    TraceSink,
    load_demo_rows,
    patch_teacher_factory,
)
from examples.common.openai_utils import AsyncLLMCaller, make_teacher_client


class LoggedStudentCaller:
    def __init__(self, caller: AsyncLLMCaller, sink: TraceSink):
        self._caller = caller
        self._sink = sink
        self.request_config = caller.request_config

    async def call_text(self, messages: list[dict[str, str]]) -> str:
        self._sink.append_messages("student_input", messages)
        output = await self._caller.call_text(messages)
        self._sink.append("student", output)
        return output


class DemoCodeCoachWorkflow(CodeCoachAgentWorkflow):
    def __init__(self, *args, trace_sink: TraceSink, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink
        self.student_caller = LoggedStudentCaller(self.student_caller, trace_sink)


def build_workflow_kwargs(config: CodeCoachConfig, trace_sink: TraceSink) -> dict[str, Any]:
    return dict(
        trace_sink=trace_sink,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=config.max_turns,
        student_base_url=config.student_base_url,
        student_model=config.student_model,
        student_api_key=config.student_api_key,
        student_timeout=config.student_timeout,
        student_max_tokens=config.student_max_tokens,
        student_temperature=config.student_temperature,
        student_top_p=config.student_top_p,
        max_concurrent_students=config.max_concurrent_students,
        api_params_config_path=config.api_params_config_path or None,
        api_params_key=config.api_params_key or None,
        work_dir_root=config.work_dir_root,
        max_episode_total_tokens=config.gconfig.max_tokens,
    )


async def _run_one(
    config: CodeCoachConfig,
    row: dict[str, Any],
    output_dir: Path,
    teacher_base_url: str,
    teacher_api_key: str,
    teacher_model: str | None,
):
    sink = TraceSink(output_dir)
    workflow = DemoCodeCoachWorkflow(**build_workflow_kwargs(config, sink))
    teacher_extra_kwargs = {"base_url": teacher_base_url, "api_key": teacher_api_key}
    base_client = make_teacher_client(teacher_extra_kwargs)
    logged_teacher = LoggedTeacherClient(base_client, sink, model_override=teacher_model)

    def _factory(_extra_kwargs):
        return logged_teacher

    with patch_teacher_factory(codecoach_workflow_module, _factory):
        rewards = await workflow.run(row, **teacher_extra_kwargs)

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
        "student_request_config": workflow.student_caller.request_config,
    }
    meta = {
        "dataset_row": row,
        "teacher_extra_kwargs": teacher_extra_kwargs,
        "teacher_model_override": teacher_model,
        "workflow_config": {
            "max_turns": config.max_turns,
            "max_completion_tokens": config.gconfig.max_new_tokens,
            "max_episode_total_tokens": config.gconfig.max_tokens,
            "student_base_url": config.student_base_url,
            "student_model": config.student_model,
            "work_dir_root": config.work_dir_root,
        },
    }
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
    config, _ = load_expr_config(config_args, CodeCoachConfig)
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
