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
import workflow as codecoach_workflow_module
from configs import CodeCoachConfig
from workflow import CodeCoachAgentWorkflow

from examples.common.openai_utils import make_teacher_client
from examples.common.trace_utils import (
    LoggedTeacherClient,
    TraceSink,
    load_demo_rows,
    patch_teacher_factory,
)

from areal.api.cli_args import load_expr_config


class LoggedAuxiliaryCaller:
    def __init__(self, caller: Any, sink: TraceSink):
        self._caller = caller
        self._sink = sink

    @property
    def request_config(self) -> dict[str, Any]:
        return self._caller.request_config

    async def call_text(
        self,
        messages: list[dict[str, str]],
        *,
        rid_prefix: str = "auxiliary",
    ):
        self._sink.append_messages(f"{rid_prefix}_input", messages)
        result = await self._caller.call_text(messages, rid_prefix=rid_prefix)
        self._sink.append(rid_prefix, result.raw_text or result.text or result.error or "")
        return result


class DemoCodeCoachWorkflow(CodeCoachAgentWorkflow):
    def __init__(self, *args, trace_sink: TraceSink, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_sink = trace_sink
        if self.api_aux_caller is not None:
            self.api_aux_caller = LoggedAuxiliaryCaller(self.api_aux_caller, trace_sink)


def build_workflow_kwargs(config: CodeCoachConfig, trace_sink: TraceSink) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    pairwise = reward.pairwise
    return dict(
        trace_sink=trace_sink,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=config.max_turns,
        enable_thinking=config.enable_thinking,
        aux_mode=auxiliary_model.mode,
        aux_enable_thinking=auxiliary_model.enable_thinking,
        aux_base_url=auxiliary_model.base_url,
        aux_model=auxiliary_model.model,
        aux_api_key=auxiliary_model.api_key,
        aux_timeout=auxiliary_model.timeout,
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=auxiliary_model.max_concurrent_calls,
        aux_request_params=auxiliary_model.request_params,
        work_dir_root=config.work_dir_root,
        max_train_sample_tokens=config.gconfig.max_tokens,
        token_budget_penalty=reward.token_budget_penalty,
        tokenizer_path=config.tokenizer_path,
        model_context_length=config.sglang.context_length,
        pairwise_reward_enabled=pairwise.enabled,
        pairwise_reference_lag_steps=pairwise.reference_lag_steps,
        pairwise_reward_scale=pairwise.scale,
        pairwise_compare_all_turns=pairwise.compare_all_turns,
        debug_trace_dir=config.debug_trace_dir or None,
        debug_trace_every_n_rollouts=config.debug_trace_every_n_rollouts,
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
        reward = await workflow.run(row, **teacher_extra_kwargs)

    turn_totals = [
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
        for usage in logged_teacher.logged_usage
    ]
    auxiliary_request_config = (
        workflow.api_aux_caller.request_config
        if workflow.api_aux_caller is not None
        else {"mode": "self"}
    )
    summary = {
        "id": row.get("id"),
        "reward": reward,
        "history": workflow.last_history,
        "teacher_usage": logged_teacher.logged_usage,
        "turn_total_tokens": turn_totals,
        "length_budget": config.gconfig.max_tokens,
        "auxiliary_request_config": auxiliary_request_config,
    }
    meta = {
        "dataset_row": row,
        "teacher_extra_kwargs": teacher_extra_kwargs,
        "teacher_model_override": teacher_model,
        "workflow_config": {
            "max_turns": config.max_turns,
            "max_completion_tokens": config.gconfig.max_new_tokens,
            "max_train_sample_tokens": config.gconfig.max_tokens,
            "auxiliary_model": config.auxiliary_model,
            "reward": config.reward,
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
    root = Path(
        args.output_dir
        or Path(__file__).parent
        / "demo_output"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
