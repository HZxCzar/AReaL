from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.append(str(_THIS_DIR))
sys.path.append(str(_REPO_ROOT))

from examples.tutor.core.text import (  # noqa: E402
    strip_reasoning_for_context as _strip_reasoning_for_context,
)

DEFAULT_FILTERED_DATASET = _THIS_DIR / "aime_dataset_no_pre_solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively test whether the current auxiliary student can be taught "
            "by a human tutor on rows from the filtered tutor training dataset."
        )
    )
    parser.add_argument("--config", default="examples/tutor/config.yaml")
    parser.add_argument(
        "--dataset",
        default="",
        help=(
            "HuggingFace dataset path. Defaults to examples/tutor/"
            "aime_dataset_no_pre_solve when present, otherwise "
            "config.train_dataset.path."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--id", default="", help="Dataset row id to load.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Sample one row from the selected split instead of using --index/0.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Maximum tutor turns. Defaults to config.max_turns.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory for transcript.json. Defaults to examples/tutor/manual_output/"
            "<timestamp>. Pass 'none' to disable saving."
        ),
    )
    parser.add_argument(
        "--hide-ground-truth",
        action="store_true",
        help="Do not print the ground-truth answer in the terminal.",
    )
    parser.add_argument(
        "--skip-leak-check",
        action="store_true",
        help="Send your tutor messages to the student even if they reveal the answer.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def resolve_dataset_path(config: Any, dataset_arg: str) -> Path:
    if dataset_arg:
        return Path(dataset_arg).expanduser().resolve()
    if DEFAULT_FILTERED_DATASET.exists():
        return DEFAULT_FILTERED_DATASET.resolve()
    return Path(config.train_dataset.path).expanduser().resolve()


def load_split_rows(dataset_path: Path, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required to load tutor datasets."
        ) from exc

    loaded = load_from_disk(str(dataset_path))
    if hasattr(loaded, "keys"):
        if split not in loaded:
            raise ValueError(
                f"Split {split!r} not found in {dataset_path}; "
                f"available splits are {list(loaded.keys())}"
            )
        dataset = loaded[split]
    else:
        dataset = loaded
    return [dict(dataset[index]) for index in range(len(dataset))]


def select_row(
    rows: list[dict[str, Any]],
    *,
    row_id: str,
    index: int | None,
    random_row: bool,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    if not rows:
        raise ValueError("Selected dataset split is empty.")
    if row_id:
        for row_index, row in enumerate(rows):
            if str(row.get("id", "")) == str(row_id):
                return row_index, row
        raise ValueError(f"Could not find dataset row with id={row_id!r}.")
    if random_row:
        row_index = random.Random(seed).randrange(len(rows))
        return row_index, rows[row_index]
    row_index = 0 if index is None else index
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(
            f"Row index {row_index} is out of range for split size {len(rows)}."
        )
    return row_index, rows[row_index]


def resolve_local_tokenizer_path(tokenizer_path: str | None) -> str | None:
    if not tokenizer_path:
        return None
    expanded = Path(tokenizer_path).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    return None


def build_workflow(config: Any, max_turns: int, *, hide_ground_truth: bool) -> Any:
    from workflow import TutorAgentWorkflow

    auxiliary_model = config.auxiliary_model
    reward = config.reward
    teacher_pre = config.teacher_pre
    return TutorAgentWorkflow(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=max_turns,
        dataset_type=config.dataset_type,
        answer_scorer=config.answer_scorer,
        enable_thinking=config.enable_thinking,
        leak_handling_mode=config.leak_handling_mode,
        aux_mode=auxiliary_model.mode,
        aux_enable_thinking=auxiliary_model.enable_thinking,
        aux_base_url=auxiliary_model.base_url,
        aux_model=auxiliary_model.model,
        aux_api_key=auxiliary_model.api_key,
        aux_timeout=auxiliary_model.timeout,
        aux_max_tokens=auxiliary_model.max_tokens,
        aux_temperature=auxiliary_model.temperature,
        aux_top_p=auxiliary_model.top_p,
        max_concurrent_aux_calls=1,
        aux_request_params=auxiliary_model.request_params,
        success_reward=reward.success,
        leak_penalty=reward.leak_penalty,
        leak_penalty_mode=reward.leak_penalty_mode,
        leak_penalty_final_answer=reward.leak_penalty_final_answer,
        leak_penalty_compute=reward.leak_penalty_compute,
        leak_penalty_formula=reward.leak_penalty_formula,
        leak_penalty_aggregation=reward.leak_penalty_aggregation,
        turn_local_reward_components=tuple(reward.turn_local_components),
        turn_local_reward_component_placements=dict(
            reward.turn_local_component_placements
        ),
        turn_local_reward_default_placement=(
            "group_norm"
            if config.actor.group_baseline_local_reward_mode == "include"
            else "pre_std"
        ),
        format_error_penalty=reward.format_error_penalty,
        personality_gate_terminate_penalty=(reward.personality_gate_terminate_penalty),
        personality_gate_fail_penalty=reward.personality_gate_fail_penalty,
        leaked_success_reward_scale=reward.leaked_success_reward_scale,
        assign_success_reward=reward.assign_success_reward,
        outcome_prior_turn_weight=reward.outcome_prior_turn_weight,
        outcome_credit_gamma=reward.outcome_credit_gamma,
        early_success_bonus=reward.early_success_bonus,
        success_turn_shaping=asdict(reward.success_turn_shaping),
        max_turn_penalty=reward.max_turn_penalty,
        enable_turn_penalty=reward.enable_turn_penalty,
        turn_penalty=reward.turn_penalty,
        length_penalty_threshold_chars=reward.length_penalty_threshold_chars,
        length_penalty_per_100_chars=reward.length_penalty_per_100_chars,
        length_penalty_min=reward.length_penalty_min,
        teacher_diversity_reward=asdict(reward.teacher_diversity),
        teacher_context_reward=asdict(reward.teacher_context),
        teacher_system_prompt=config.teacher_system_prompt,
        teacher_anti_leak_instruction_enabled=(
            config.teacher_anti_leak_instruction_enabled
        ),
        teacher_adaptive_instruction_enabled=(
            config.teacher_adaptive_instruction_enabled
        ),
        teacher_user_prompt_template=config.teacher_user_prompt_template,
        teacher_show_ground_truth=(
            bool(config.teacher_show_ground_truth) and not hide_ground_truth
        ),
        teacher_pre_enabled=teacher_pre.enabled,
        teacher_pre_mode=teacher_pre.mode,
        teacher_pre_verify=teacher_pre.verify,
        teacher_pre_attempts=teacher_pre.attempts,
        teacher_pre_max_tokens=teacher_pre.max_tokens,
        student_system_prompt=config.student_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        answer_judge_enabled=auxiliary_model.answer_judge_enabled,
        answer_judge_max_tokens=auxiliary_model.answer_judge_max_tokens,
        answer_judge_system_prompt=config.answer_judge_system_prompt,
        debug_trace_dir="",
        max_train_sample_tokens=config.gconfig.max_tokens,
        tokenizer_path=resolve_local_tokenizer_path(config.tokenizer_path),
        model_context_length=config.sglang.context_length,
    )


def resolve_output_dir(output_dir_arg: str) -> Path | None:
    if output_dir_arg.lower() == "none":
        return None
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (_THIS_DIR / "manual_output" / timestamp).resolve()


def print_block(title: str, body: str) -> None:
    separator = "=" * 80
    print(f"\n{separator}\n{title}\n{separator}")
    print(body.rstrip() or "(empty)")


def print_judge_result(result: Any) -> None:
    extracted = result.raw_result.get("extracted_answer") or "(none)"
    target = result.raw_result.get("normalized_target") or "(none)"
    status = "CORRECT" if result.correct else "INCORRECT"
    print(f"\nJudge: {status} | extracted={extracted!r} | target={target!r}")


def read_tutor_message(turn_idx: int) -> str | None:
    print(f"\nTutor turn {turn_idx}: enter your feedback. End with a single '.' line.")
    print("Commands: /quit exits, /empty sends an empty tutor message.")
    lines: list[str] = []
    while True:
        line = input("> ")
        command = line.strip()
        if command in {"/quit", "/q"}:
            return None
        if command == "/empty" and not lines:
            return ""
        if command == ".":
            return "\n".join(lines).strip()
        lines.append(line)


def to_visible_tutor_message(tutor_message: str) -> str:
    return _strip_reasoning_for_context(tutor_message)


async def run_interactive(args: argparse.Namespace) -> dict[str, Any]:
    from configs import TutorConfig

    from examples.tutor.core.types import (
        PublicHistoryState,
        StudentTurnState,
        TutorPrivateFeedback,
        TutorTurnState,
    )

    from areal.api.cli_args import load_expr_config

    config_args = ["--config", args.config, *args.overrides]
    config, _ = load_expr_config(config_args, TutorConfig)
    dataset_path = resolve_dataset_path(config, args.dataset)
    rows = load_split_rows(dataset_path, args.split)
    row_index, row = select_row(
        rows,
        row_id=args.id,
        index=args.index,
        random_row=bool(args.random),
        seed=int(args.seed),
    )
    max_turns = int(args.max_turns) if args.max_turns > 0 else int(config.max_turns)
    workflow = build_workflow(
        config, max_turns=max_turns, hide_ground_truth=bool(args.hide_ground_truth)
    )
    answer_judge_caller = workflow._make_answer_judge_caller()

    task = str(row["task"])
    ground_truth = str(row["ground_truth"])
    history: list[dict[str, Any]] = []
    transcript: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "split": args.split,
        "row_index": row_index,
        "row": row,
        "max_turns": max_turns,
        "skip_leak_check": bool(args.skip_leak_check),
        "rounds": history,
    }

    row_summary = (
        f"path: {dataset_path}\n"
        f"split: {args.split}\n"
        f"index: {row_index}\n"
        f"id: {row.get('id')}"
    )
    print_block("Dataset Row", row_summary)
    print_block("Task", task)
    if args.hide_ground_truth:
        print("\nGround truth: (hidden)")
    else:
        print_block("Ground Truth", ground_truth)

    print("\nCalling current student for the initial answer...")
    initial_answer, initial_error = await workflow._run_student(
        StudentTurnState(
            task=task,
            public_history=PublicHistoryState(),
            previous_student_output="",
            latest_tutor_visible_output="(none, produce the first answer attempt)",
        )
    )
    initial_judge = await workflow._score_answer_async(
        task,
        ground_truth,
        initial_answer,
        answer_judge_caller=answer_judge_caller,
    )
    transcript["initial_student_answer"] = initial_answer
    transcript["initial_student_error"] = initial_error
    transcript["initial_judge"] = initial_judge.raw_result
    print_block("Initial Student Answer", initial_answer)
    if initial_error:
        print(f"\nStudent error: {initial_error}")
    print_judge_result(initial_judge)

    if initial_judge.correct:
        transcript["termination_reason"] = "pre_solved"
        print("\nThe current student solved this row before tutoring.")
        return transcript

    initial_turns = workflow._initial_conversation(initial_answer)
    initial_turns[-1]["env"] = workflow._teacher_env_feedback(initial_judge, 0)
    public_history = PublicHistoryState(
        summary=workflow._build_initial_public_summary(initial_answer),
        turn_count=0,
        turns=initial_turns,
    )
    termination_reason = "max_turns"
    latest_answer = initial_answer
    previous_student_answer = initial_answer
    previous_tutor_visible_output = ""
    previous_feedback = TutorPrivateFeedback(
        kind="student_judged",
        student_output=initial_answer,
        judge_correct=False,
        judge_feedback=initial_judge.feedback,
    )
    for turn_idx in range(1, max_turns + 1):
        leak_result = None
        print_block("Student-Visible State", public_history.summary)
        tutor_state = TutorTurnState(
            task=task,
            ground_truth=ground_truth,
            public_history=public_history,
            previous_tutor_visible_output=previous_tutor_visible_output,
            previous_feedback=previous_feedback,
            turn_idx=turn_idx,
            max_turns=max_turns,
        )
        tutor_prompt = workflow._build_tutor_prompt(tutor_state)
        print_block("Full Tutor Input", tutor_prompt)
        tutor_message = read_tutor_message(turn_idx)
        if tutor_message is None:
            termination_reason = "user_quit"
            break
        tutor_visible_message = to_visible_tutor_message(tutor_message)
        record: dict[str, Any] = {
            "round_idx": turn_idx,
            "teacher_raw_output": tutor_message,
            "teacher_action": tutor_visible_message,
            "tutor_prompt": tutor_prompt,
            "public_history_before": public_history.summary,
        }

        if config.leak_handling_mode == "terminate" and not args.skip_leak_check:
            print("\nRunning leak check before sending this message to the student...")
            leak_result = await workflow._run_optional_leak_check(
                task, ground_truth, tutor_visible_message
            )
            record["leak_check"] = {
                "leaked": leak_result.leaked,
                "level": leak_result.leak_level,
                "feedback": leak_result.feedback,
                "parse_error": leak_result.parse_error,
                "raw_output": leak_result.raw_output,
            }
            if leak_result.leaked and config.leak_handling_mode == "terminate":
                record["leak_detected"] = True
                record["student_answer"] = ""
                record["judge_correct"] = False
                record["student_visible"] = False
                record["invalid_due_to_leak"] = False
                record["public_history_after"] = public_history.summary
                history.append(record)
                termination_reason = "leak"
                print("\nLeak check: LEAKED. Terminating before student call.")
                print(f"Feedback: {leak_result.feedback}")
                break

        student_state = StudentTurnState(
            task=task,
            public_history=public_history,
            previous_student_output=previous_student_answer,
            latest_tutor_visible_output=tutor_visible_message,
        )
        student_answer, student_error = await workflow._run_student(student_state)
        next_public_history = await workflow._run_public_summary_update(
            old_public_history=public_history,
            previous_student_answer=previous_student_answer,
            tutor_visible_output=tutor_visible_message,
            current_student_answer=student_answer,
        )
        judge_result = await workflow._score_answer_async(
            task,
            ground_truth,
            student_answer,
            answer_judge_caller=answer_judge_caller,
        )
        record.update(
            {
                "leak_detected": False,
                "invalid_due_to_leak": False,
                "student_visible": True,
                "student_answer": student_answer,
                "student_error": student_error,
                "judge_correct": judge_result.correct,
                "judge_feedback": judge_result.feedback,
                "judge_raw_result": judge_result.raw_result,
                "public_history_after": next_public_history.summary,
            }
        )
        history.append(record)
        public_history = next_public_history
        latest_answer = student_answer
        previous_student_answer = student_answer
        previous_tutor_visible_output = tutor_visible_message
        previous_feedback = TutorPrivateFeedback(
            kind="student_judged",
            student_output=student_answer,
            judge_correct=judge_result.correct,
            judge_feedback=judge_result.feedback,
        )

        print_block(f"Student Answer After Turn {turn_idx}", student_answer)
        if student_error:
            print(f"\nStudent error: {student_error}")
        print_judge_result(judge_result)

        if judge_result.correct:
            termination_reason = "success"
            break

    transcript["latest_student_answer"] = latest_answer
    transcript["termination_reason"] = termination_reason
    return transcript


def save_transcript(transcript: dict[str, Any], output_dir: Path | None) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "transcript.json"
    path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved transcript: {path}")


def main() -> None:
    args = parse_args()
    transcript = asyncio.run(run_interactive(args))
    save_transcript(transcript, resolve_output_dir(args.output_dir))
    print(f"\nTermination: {transcript.get('termination_reason')}")


if __name__ == "__main__":
    main()
