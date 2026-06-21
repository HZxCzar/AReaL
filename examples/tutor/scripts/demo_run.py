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
import workflow as tutor_workflow_module
from configs import TutorConfig
from workflow import TutorAgentWorkflow

from examples.common.openai_utils import make_teacher_client
from examples.common.trace_utils import (
    LoggedTeacherClient,
    TraceSink,
    load_demo_rows,
    patch_teacher_factory,
)

from areal.api.cli_args import load_expr_config


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
        state: tutor_workflow_module.StudentTurnState,
        *,
        aux_caller=None,
    ) -> tuple[str, str | None]:
        prompt = self._build_student_prompt_from_state(state)
        if (
            state.task == self.root_task
            and not state.previous_student_output
            and not state.public_history.summary
        ):
            label = "student_init"
        else:
            label = "student"
        messages = [
            {"role": "system", "content": self.student_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.trace_sink.append_messages(f"{label}_input", messages)
        result = await (aux_caller or self._make_auxiliary_caller(engine=None)).call_text(
            messages,
            rid_prefix=label,
        )
        if result.error:
            self.trace_sink.append(label, f"Student call failed: {result.error}")
            return "", result.error
        self.trace_sink.append(label, result.text)
        return result.text, None

    async def _run_leak_check(
        self,
        task: str,
        ground_truth: str,
        teacher_action: str,
        *,
        aux_caller=None,
    ):
        prompt = self._build_leak_check_prompt(task, ground_truth, teacher_action)
        messages = [
            {"role": "system", "content": self.leak_check_system_prompt},
            {"role": "user", "content": prompt},
        ]
        self.trace_sink.append_messages("leak_check_input", messages)
        result = await super()._run_leak_check(
            task,
            ground_truth,
            teacher_action,
            aux_caller=aux_caller,
        )
        self.trace_sink.append("leak_check", result.raw_output or result.feedback)
        return result

    async def _score_answer_async(
        self,
        task: str,
        ground_truth: str,
        student_answer: str,
        *,
        answer_judge_caller=None,
    ):
        result = await super()._score_answer_async(
            task,
            ground_truth,
            student_answer,
            answer_judge_caller=answer_judge_caller,
        )
        self.trace_sink.append("judge", result.raw_output)
        self.last_judge_outputs.append(
            {
                "label": "judge",
                "correct": result.correct,
                "feedback": result.feedback,
                "raw_result": result.raw_result,
            }
        )
        return result


def build_workflow_kwargs(config: TutorConfig, trace_sink: TraceSink) -> dict[str, Any]:
    auxiliary_model = config.auxiliary_model
    reward = config.reward
    return dict(
        trace_sink=trace_sink,
        tokenizer=config.tokenizer_path,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=config.max_turns,
        answer_scorer=config.answer_scorer,
        enable_thinking=config.enable_thinking,
        enable_leak_check=config.enable_leak_check,
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
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        leak_penalty_mode=reward.leak_penalty_mode,
        leak_penalty_final_answer=reward.leak_penalty_final_answer,
        leak_penalty_compute=reward.leak_penalty_compute,
        leak_penalty_formula=reward.leak_penalty_formula,
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
        teacher_user_prompt_template=config.teacher_user_prompt_template,
        teacher_show_ground_truth=config.teacher_show_ground_truth,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        answer_judge_system_prompt=config.answer_judge_system_prompt,
        summary_system_prompt=config.summary_system_prompt,
        max_train_sample_tokens=config.gconfig.max_tokens,
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
        str(record.get("public_history_after", ""))
        for record in workflow.last_history
        if not record.get("leak_detected") and record.get("public_history_after")
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
        "aux_request_config": getattr(workflow.aux_caller, "request_config", {}),
    }
    meta = {
        "dataset_row": row,
        "teacher_extra_kwargs": teacher_extra_kwargs,
        "teacher_model_override": teacher_model,
        "workflow_config": {
            "max_turns": config.max_turns,
            "enable_thinking": config.enable_thinking,
            "max_completion_tokens": config.gconfig.max_new_tokens,
            "max_train_sample_tokens": config.gconfig.max_tokens,
            "aux_base_url": config.auxiliary_model.base_url,
            "aux_model": config.auxiliary_model.model,
            "aux_mode": config.auxiliary_model.mode,
            "aux_enable_thinking": config.auxiliary_model.enable_thinking,
            "answer_scorer": config.answer_scorer,
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
