from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.append(str(_THIS_DIR))
sys.path.append(str(_REPO_ROOT))


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
    parser.add_argument(
        "--transfer-check",
        action="store_true",
        help="After the original task is solved, also run the workflow transfer check.",
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


def build_workflow(config: Any, max_turns: int) -> Any:
    from workflow import TutorAgentWorkflow

    class ManualTutorWorkflow(TutorAgentWorkflow):
        async def _run_transfer_round(
            self,
            *,
            task: str,
            ground_truth: str,
            initial_student_answer: str,
            history: list[dict[str, Any]],
        ) -> dict[str, Any]:
            generation = await self._run_transfer_generation(task, ground_truth)
            payload: dict[str, Any] = {
                "transfer_triggered": True,
                "transfer_task": generation.task,
                "transfer_ground_truth": generation.ground_truth,
                "transfer_generation_error": generation.parse_error,
                "transfer_generation_raw_output": generation.raw_output,
                "transfer_similarity_notes": generation.similarity_notes,
                "transfer_success": False,
            }
            if not generation.task or not generation.ground_truth:
                return payload
            answer, answer_error = await self._run_transfer_student(
                original_task=task,
                initial_student_answer=initial_student_answer,
                history=history,
                transfer_task=generation.task,
            )
            judge_result = self._score_aime_answer(
                generation.task, generation.ground_truth, answer
            )
            payload.update(
                {
                    "transfer_student_answer": answer,
                    "transfer_student_error": answer_error,
                    "transfer_judge_feedback": judge_result.feedback,
                    "transfer_success": judge_result.correct,
                }
            )
            return payload

    return ManualTutorWorkflow(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_completion_tokens=config.gconfig.max_new_tokens,
        max_turns=max_turns,
        enable_thinking=config.enable_thinking,
        aux_base_url=config.aux_base_url,
        aux_model=config.aux_model,
        aux_api_key=config.aux_api_key,
        aux_timeout=config.aux_timeout,
        aux_max_tokens=config.aux_max_tokens,
        aux_temperature=config.aux_temperature,
        aux_top_p=config.aux_top_p,
        max_concurrent_aux_calls=1,
        api_params_config_path=config.api_params_config_path or None,
        api_params_key=config.api_params_key or None,
        term_success_reward=config.term_success_reward,
        transfer_bonus_reward=config.transfer_bonus_reward,
        token_budget_penalty=config.token_budget_penalty,
        teacher_system_prompt=config.teacher_system_prompt,
        student_system_prompt=config.student_system_prompt,
        judge_system_prompt=config.judge_system_prompt,
        leak_check_system_prompt=config.leak_check_system_prompt,
        generator_system_prompt=config.generator_system_prompt,
        debug_trace_dir="",
        max_episode_total_tokens=config.gconfig.max_tokens,
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
    print(
        f"\nTutor turn {turn_idx}: enter your feedback. End with a single '.' line."
    )
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


async def run_interactive(args: argparse.Namespace) -> dict[str, Any]:
    from areal.api.cli_args import load_expr_config
    from configs import TutorConfig

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
    workflow = build_workflow(config, max_turns=max_turns)

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
        "transfer_check": bool(args.transfer_check),
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
        task, teacher_action=None, history=[]
    )
    initial_judge = workflow._score_aime_answer(task, ground_truth, initial_answer)
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

    termination_reason = "max_turns"
    latest_answer = initial_answer
    for turn_idx in range(1, max_turns + 1):
        visible_history = workflow._student_visible_history_summaries(history)
        if visible_history:
            print_block("Student-Visible History", "\n".join(visible_history))
        tutor_message = read_tutor_message(turn_idx)
        if tutor_message is None:
            termination_reason = "user_quit"
            break
        record: dict[str, Any] = {
            "round_idx": turn_idx,
            "teacher_action": tutor_message,
        }

        if not args.skip_leak_check:
            print("\nRunning leak check before sending this message to the student...")
            leak_result = await workflow._run_leak_check(
                task, ground_truth, tutor_message
            )
            record["leak_check"] = {
                "leaked": leak_result.leaked,
                "feedback": leak_result.feedback,
                "parse_error": leak_result.parse_error,
                "raw_output": leak_result.raw_output,
            }
            if leak_result.leaked:
                record["leak_detected"] = True
                record["student_answer"] = ""
                record["judge_correct"] = False
                history.append(record)
                print("\nLeak check: LEAKED. The student will not see this turn.")
                print(f"Feedback: {leak_result.feedback}")
                continue

        student_answer, student_error = await workflow._run_student(
            task, tutor_message, history=history
        )
        judge_result = workflow._score_aime_answer(task, ground_truth, student_answer)
        latest_answer = student_answer
        record.update(
            {
                "leak_detected": False,
                "student_answer": student_answer,
                "student_error": student_error,
                "judge_correct": judge_result.correct,
                "judge_feedback": judge_result.feedback,
                "judge_raw_result": judge_result.raw_result,
            }
        )
        record["student_visible_summary"] = workflow._build_student_visible_summary(
            record
        )
        history.append(record)

        print_block(f"Student Answer After Turn {turn_idx}", student_answer)
        if student_error:
            print(f"\nStudent error: {student_error}")
        print_judge_result(judge_result)

        if judge_result.correct:
            termination_reason = "success"
            if args.transfer_check:
                print("\nRunning transfer check with the same workflow logic...")
                transfer_result = await workflow._run_transfer_round(
                    task=task,
                    ground_truth=ground_truth,
                    initial_student_answer=initial_answer,
                    history=history,
                )
                record.update(transfer_result)
                print_block(
                    "Transfer Task", str(transfer_result.get("transfer_task", ""))
                )
                print_block(
                    "Transfer Student Answer",
                    str(transfer_result.get("transfer_student_answer", "")),
                )
                generation_error = transfer_result.get("transfer_generation_error")
                if generation_error:
                    print_block("Transfer Generation Error", str(generation_error))
                    print_block(
                        "Transfer Generator Raw Output",
                        str(
                            transfer_result.get(
                                "transfer_generation_raw_output", ""
                            )
                        ),
                    )
                print(
                    "\nTransfer success: "
                    f"{bool(transfer_result.get('transfer_success', False))}"
                )
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
