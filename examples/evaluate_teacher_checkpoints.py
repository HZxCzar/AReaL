from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
RESULTS = REPO / "results" / "train-teacher" / "eval-checkpoints"
BASE_TEACHER = "/data/xmy/models/Qwen3.5-2B"
STUDENT = "/data/xmy/models/Qwen3-0.6B"
CKPT_ROOT = (
    REPO
    / "results/train-teacher/checkpoints/checkpoints/xmy/tutor-grpo"
)


ACTIVE_MODELS: dict[str, "LocalChatModel"] = {}


def step_of(path: Path) -> int:
    match = re.search(r"globalstep(\d+)", path.name)
    return int(match.group(1)) if match else -1


def checkpoint(trial: str, step: int) -> Path:
    path = CKPT_ROOT / trial / "default" / f"epoch0epochstep{step}globalstep{step}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def default_suite() -> dict[str, list[tuple[str, str, int | None]]]:
    hanabi_trial = "hanabi-largelr-qwen3_5-2b-teacher-qwen3-0_6b-student"
    hidden_trial = "hidden-rule-largelr-qwen3_5-2b-teacher-qwen3-0_6b-student"
    werewolf_trial = "werewolf-qwen3_5-2b-teacher-qwen3-0_6b-student"
    return {
        "hanabi": [("base", BASE_TEACHER, None)]
        + [(f"step{step}", str(checkpoint(hanabi_trial, step)), step) for step in range(0, 176, 25)],
        "hidden-rule": [("base", BASE_TEACHER, None)]
        + [(f"step{step}", str(checkpoint(hidden_trial, step)), step) for step in range(0, 301, 50)],
        "werewolf": [("base", BASE_TEACHER, None)]
        + [
            (f"step{step}", str(checkpoint(werewolf_trial, step)), step)
            for step in (5, 25, 59, 100, 134, 179, 221, 271)
        ],
    }


class LocalChatModel:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        template_kwargs = dict(chat_template_kwargs or {})
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **template_kwargs,
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        has_inputs = hasattr(encoded, "__getitem__") and "input_ids" in encoded
        input_ids = encoded["input_ids"] if has_inputs else encoded
        attention_mask = encoded.get("attention_mask") if has_inputs and "attention_mask" in encoded else None
        input_ids = input_ids.to(self.device)
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask.to(self.device)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        else:
            gen_kwargs.update(do_sample=False)
        output = self.model.generate(input_ids, **gen_kwargs)
        text = self.tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True)
        return text.strip()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LocalOpenAIChatClient:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        if cfg.base_url == "local://teacher":
            self.local_model = ACTIVE_MODELS["teacher"]
        elif cfg.base_url == "local://student":
            self.local_model = ACTIVE_MODELS["student"]
        else:
            from game_tutor_common import OpenAIChatClient

            self.remote = OpenAIChatClient(cfg)
            self.local_model = None

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int | None = None) -> str:
        return await asyncio.to_thread(self._complete_blocking, messages, max_tokens)

    def _complete_blocking(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        if self.local_model is None:
            return self.remote._complete_blocking(messages, max_tokens)
        return self.local_model.complete(
            messages,
            max_tokens=int(max_tokens or self.cfg.max_tokens),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            chat_template_kwargs=self.cfg.chat_template_kwargs,
        )


def load_module(name: str, path: Path, game_dir: Path):
    for text in (str(game_dir), str(EXAMPLES), str(REPO)):
        if text not in sys.path:
            sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_eval_modules():
    import game_tutor_common

    game_tutor_common.OpenAIChatClient = LocalOpenAIChatClient
    modules = {
        "hanabi": load_module(
            "hanabi_eval_teacher",
            EXAMPLES / "hanabi-tutor/evaluate_teacher.py",
            EXAMPLES / "hanabi-tutor",
        ),
        "werewolf": load_module(
            "werewolf_eval_teacher",
            EXAMPLES / "werewolf-tutor/evaluate_teacher.py",
            EXAMPLES / "werewolf-tutor",
        ),
        "hidden-rule": load_module(
            "hidden_rule_eval_teacher",
            EXAMPLES / "hidden-rule-tutor/evaluate_teacher.py",
            EXAMPLES / "hidden-rule-tutor",
        ),
    }
    return modules


def config_for(
    game: str,
    output_path: Path,
    game_num_turns: dict[str, int],
    game_max_tokens: dict[str, int],
) -> dict[str, Any]:
    cfg_path = EXAMPLES / f"{game}-tutor/teacher_eval_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    max_tokens = game_max_tokens.get(game, game_max_tokens["default"])
    num_turns = game_num_turns.get(game, game_num_turns["default"])
    cfg["num_turns"] = num_turns
    cfg["output_path"] = str(output_path)
    cfg["teacher_model"]["base_url"] = "local://teacher"
    cfg["teacher_model"]["model"] = "local-teacher"
    cfg["teacher_model"]["max_tokens"] = min(max_tokens, int(cfg["teacher_model"].get("max_tokens", 1024)))
    cfg["teacher_model"]["chat_template_kwargs"] = {"enable_thinking": False}
    cfg["student_model"]["base_url"] = "local://student"
    cfg["student_model"]["model"] = "local-student"
    cfg["student_model"]["max_tokens"] = min(max_tokens, int(cfg["student_model"].get("max_tokens", 1024)))
    cfg["student_model"]["chat_template_kwargs"] = {"enable_thinking": False}
    if game == "hidden-rule":
        cfg["episodes"] = max(1, int(num_turns) // max(1, int(cfg.get("rounds", 8))))
    elif game == "werewolf":
        cfg["max_episode_turns"] = max(
            int(cfg.get("max_episode_turns", 80)),
            min(20, max(10, int(num_turns))),
        )
    return cfg


async def run_game_model(game: str, module: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    if game == "hidden-rule":
        baseline = module.run_condition(cfg, use_teacher=False)
        teacher = module.run_condition(cfg, use_teacher=True)
        report = {
            "config": cfg,
            "baseline": baseline,
            "teacher": teacher,
            "improvement": {
                "mean_reward": teacher["mean_reward"] - baseline["mean_reward"],
                "mean_heldout_accuracy": teacher["mean_heldout_accuracy"]
                - baseline["mean_heldout_accuracy"],
                "success_rate": teacher["success_rate"] - baseline["success_rate"],
                "avg_examples_used": teacher["avg_examples_used"] - baseline["avg_examples_used"],
                "avg_turns_taken": teacher["avg_turns_taken"] - baseline["avg_turns_taken"],
            },
        }
        return report
    evaluator = module.TeacherEvaluator(
        env_factory=module.build_env,
        prompt_builder=module.build_prompt,
        score_getter=module.score,
        config=cfg,
    )
    return await evaluator.run_pair()


def summarize_metric(game: str, report: dict[str, Any]) -> dict[str, float]:
    imp = report["improvement"]
    if game == "hidden-rule":
        return {
            "primary_delta": float(imp["mean_reward"]),
            "teacher_primary": float(report["teacher"]["mean_reward"]),
            "baseline_primary": float(report["baseline"]["mean_reward"]),
            "accuracy_delta": float(imp["mean_heldout_accuracy"]),
            "success_delta": float(imp["success_rate"]),
        }
    return {
        "primary_delta": float(imp["avg_score"]),
        "teacher_primary": float(report["teacher"]["avg_score"]),
        "baseline_primary": float(report["baseline"]["avg_score"]),
        "invalid_action_delta": float(imp["invalid_actions"]),
        "turn_delta": float(imp["turns"]),
    }


def plot_game(game: str, rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["label"] for row in rows]
    deltas = [row["primary_delta"] for row in rows]
    teacher = [row["teacher_primary"] for row in rows]
    baseline = [row["baseline_primary"] for row in rows]
    x = range(len(rows))
    plt.figure(figsize=(8, 4.8))
    plt.axhline(0, color="#444", linewidth=1)
    plt.bar(x, deltas, color="#4C78A8")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Teacher - no-teacher baseline")
    plt.title(f"{game} tutoring gain")
    plt.tight_layout()
    plt.savefig(out_dir / f"{game}_gain.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(labels, baseline, marker="o", label="No teacher")
    plt.plot(labels, teacher, marker="o", label="Teacher")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Primary score")
    plt.title(f"{game} absolute scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{game}_scores.png", dpi=180)
    plt.close()

    base_teacher = next((row["teacher_primary"] for row in rows if row["label"] == "base"), teacher[0])
    vs_base = [row["teacher_primary"] - base_teacher for row in rows]
    plt.figure(figsize=(8, 4.8))
    plt.axhline(0, color="#444", linewidth=1)
    plt.bar(x, vs_base, color="#59A14F")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("Teacher primary - original Qwen3.5 teacher")
    plt.title(f"{game} checkpoint vs original teacher")
    plt.tight_layout()
    plt.savefig(out_dir / f"{game}_vs_base_teacher.png", dpi=180)
    plt.close()

    trend_rows = sorted((row for row in rows if row["step"] is not None), key=lambda row: row["step"])
    if trend_rows:
        steps = [int(row["step"]) for row in trend_rows]
        trend_teacher = [row["teacher_primary"] for row in trend_rows]
        trend_gain = [row["primary_delta"] for row in trend_rows]
        base_teacher_value = next(
            (row["teacher_primary"] for row in rows if row["label"] == "base"),
            trend_teacher[0],
        )
        plt.figure(figsize=(8, 4.8))
        plt.axhline(base_teacher_value, color="#E15759", linestyle="--", linewidth=1.5, label="Original teacher")
        plt.plot(steps, trend_teacher, marker="o", color="#4C78A8", label="Checkpoint teacher")
        plt.plot(steps, trend_gain, marker="s", color="#59A14F", label="Tutoring gain")
        plt.xlabel("Training step")
        plt.ylabel("Primary metric")
        plt.title(f"{game} tutor trend across checkpoints")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{game}_checkpoint_trend.png", dpi=180)
        plt.close()


def write_game_report(game: str, rows: list[dict[str, Any]], out_dir: Path) -> None:
    base_teacher = next((row["teacher_primary"] for row in rows if row["label"] == "base"), rows[0]["teacher_primary"])
    trained_rows = [row for row in rows if row["label"] != "base"] or rows
    best = max(trained_rows, key=lambda row: row["teacher_primary"] - base_teacher)
    lines = [
        f"# {game} Teacher Checkpoint Evaluation",
        "",
        f"Original Qwen3.5 teacher primary score: `{base_teacher:.4f}`.",
        f"Best trained checkpoint vs original teacher: `{best['label']}` "
        f"with delta `{best['teacher_primary'] - base_teacher:.4f}`.",
        "",
        "| Label | Step | Teacher primary | Baseline primary | Tutoring gain | vs original teacher | JSON |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        step = "" if row["step"] is None else str(row["step"])
        lines.append(
            f"| {row['label']} | {step} | {row['teacher_primary']:.4f} | "
            f"{row['baseline_primary']:.4f} | {row['primary_delta']:.4f} | "
            f"{row['teacher_primary'] - base_teacher:.4f} | "
            f"[json]({Path(row['json']).name}) |"
        )
    lines.extend(
        [
            "",
            f"Plots: `{game}_gain.png`, `{game}_scores.png`, `{game}_vs_base_teacher.png`, "
            f"`{game}_checkpoint_trend.png`.",
            "",
            "Evaluation uses local Qwen3-0.6B as the fixed student and compares each teacher model "
            "against the no-teacher baseline under the same evaluator seeds.",
        ]
    )
    (out_dir / f"{game}_report.md").write_text("\n".join(lines) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(RESULTS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-turns", type=int, default=24)
    parser.add_argument("--hanabi-num-turns", type=int, default=None)
    parser.add_argument("--hidden-rule-num-turns", type=int, default=None)
    parser.add_argument("--werewolf-num-turns", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--hanabi-max-tokens", type=int, default=None)
    parser.add_argument("--hidden-rule-max-tokens", type=int, default=None)
    parser.add_argument("--werewolf-max-tokens", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    game_max_tokens = {
        "default": int(args.max_tokens),
        "hanabi": int(args.hanabi_max_tokens or args.max_tokens),
        "hidden-rule": int(args.hidden_rule_max_tokens or args.max_tokens),
        "werewolf": int(args.werewolf_max_tokens or args.max_tokens),
    }
    game_num_turns = {
        "default": int(args.num_turns),
        "hanabi": int(args.hanabi_num_turns or args.num_turns),
        "hidden-rule": int(args.hidden_rule_num_turns or args.num_turns),
        "werewolf": int(args.werewolf_num_turns or args.num_turns),
    }

    modules = load_eval_modules()
    ACTIVE_MODELS["student"] = LocalChatModel(STUDENT, args.device, dtype)
    all_rows: list[dict[str, Any]] = []
    try:
        for game, models in default_suite().items():
            game_rows = []
            for label, model_path, step in models:
                print(f"[eval] {game} {label}: {model_path}", flush=True)
                ACTIVE_MODELS["teacher"] = LocalChatModel(model_path, args.device, dtype)
                json_path = out_dir / f"{game}_{label}.json"
                try:
                    cfg = config_for(game, json_path, game_num_turns, game_max_tokens)
                    report = await run_game_model(game, modules[game], cfg)
                    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2))
                    row = {
                        "game": game,
                        "label": label,
                        "step": step,
                        "model_path": model_path,
                        "json": str(json_path),
                    }
                    row.update(summarize_metric(game, report))
                    game_rows.append(row)
                    all_rows.append(row)
                finally:
                    ACTIVE_MODELS["teacher"].close()
                    ACTIVE_MODELS.pop("teacher", None)
            plot_game(game, game_rows, out_dir)
            write_game_report(game, game_rows, out_dir)
        (out_dir / "summary.json").write_text(json.dumps(all_rows, ensure_ascii=True, indent=2))
        write_overall_report(all_rows, out_dir)
    finally:
        ACTIVE_MODELS["student"].close()


def write_overall_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    lines = [
        "# Teacher Checkpoint Evaluation Summary",
        "",
        "| Game | Original teacher | Best trained | vs original teacher | Best trained tutoring gain |",
        "|---|---:|---|---:|---:|",
    ]
    for game in sorted({row["game"] for row in rows}):
        game_rows = [row for row in rows if row["game"] == game]
        base_teacher = next(
            (row["teacher_primary"] for row in game_rows if row["label"] == "base"),
            game_rows[0]["teacher_primary"],
        )
        trained_rows = [row for row in game_rows if row["label"] != "base"] or game_rows
        best = max(trained_rows, key=lambda row: row["teacher_primary"] - base_teacher)
        lines.append(
            f"| {game} | {base_teacher:.4f} | {best['label']} | "
            f"{best['teacher_primary'] - base_teacher:.4f} | {best['primary_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Per-game reports and plots are in the same directory.",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
