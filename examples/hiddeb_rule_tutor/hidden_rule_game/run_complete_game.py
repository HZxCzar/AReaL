from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .data_generator import generate_dataset
from .env import HiddenRuleEnv
from .inference_engine import VLLMEngine
from .students import LLMStudent, MemoryStudent
from .teachers import LLMTeacher


@dataclass
class RunConfig:
    student: str = "memory"
    quick: bool = False
    skip_train: bool = False
    student_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    teacher_model: str | None = None
    student_lora_path: Path | None = None
    tensor_parallel_size: int = 1
    teacher_tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    vllm_dtype: str = "auto"
    vllm_download_dir: str | None = None
    served_model_name: str | None = None
    train_rules: int = 400
    eval_rules: int = 80
    examples_per_rule: int = 80
    train_seed: int = 7
    eval_seed: int = 1007
    rounds: int = 8
    eval_episodes: int = 50
    epochs: float = 1.0
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    success_threshold: float = 0.95
    wandb_enabled: bool = False
    wandb_project: str = "hidden-rule-game"
    wandb_run_name: str | None = None
    wandb_mode: str = "online"
    run_dir: Path = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: Path | None = None

    def apply_quick_defaults(self) -> None:
        if not self.quick:
            return
        self.train_rules = 24
        self.eval_rules = 8
        self.examples_per_rule = 32
        self.eval_episodes = 4
        self.rounds = 4
        self.epochs = 0.05

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def lora_dir(self) -> Path:
        return self.out_dir or self.run_dir / "outputs" / "lora-student"

    @property
    def eval_lora_path(self) -> Path:
        return self.student_lora_path or self.lora_dir


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full hidden-rule induction pipeline.")
    parser.add_argument("--student", choices=["memory", "llm", "lora"], default=env_str("STUDENT", "memory"))
    parser.add_argument("--quick", action="store_true", default=env_bool("QUICK", False))
    parser.add_argument("--skip-train", action="store_true", default=env_bool("SKIP_TRAIN", False))
    default_model = env_str("STUDENT_MODEL", env_str("MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    parser.add_argument("--student-model", "--model", dest="student_model", default=default_model)
    parser.add_argument("--teacher-model", default=env_str("TEACHER_MODEL", "") or None)
    parser.add_argument(
        "--student-lora-path",
        type=Path,
        default=Path(env_str("STUDENT_LORA_PATH", "")) if env_str("STUDENT_LORA_PATH", "") else None,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=env_int("TENSOR_PARALLEL_SIZE", 1))
    parser.add_argument("--teacher-tensor-parallel-size", type=int, default=env_int("TEACHER_TENSOR_PARALLEL_SIZE", 1))
    parser.add_argument("--trust-remote-code", action="store_true", default=env_bool("TRUST_REMOTE_CODE", False))
    parser.add_argument("--vllm-dtype", default=env_str("VLLM_DTYPE", "auto"))
    parser.add_argument("--vllm-download-dir", default=env_str("VLLM_DOWNLOAD_DIR", "") or None)
    parser.add_argument("--served-model-name", default=env_str("SERVED_MODEL_NAME", "") or None)
    parser.add_argument("--train-rules", type=int, default=env_int("TRAIN_RULES", 400))
    parser.add_argument("--eval-rules", type=int, default=env_int("EVAL_RULES", 80))
    parser.add_argument("--examples-per-rule", type=int, default=env_int("EXAMPLES_PER_RULE", 80))
    parser.add_argument("--train-seed", type=int, default=env_int("TRAIN_SEED", 7))
    parser.add_argument("--eval-seed", type=int, default=env_int("EVAL_SEED", 1007))
    parser.add_argument("--rounds", type=int, default=env_int("ROUNDS", 8))
    parser.add_argument("--eval-episodes", type=int, default=env_int("EVAL_EPISODES", 50))
    parser.add_argument("--epochs", type=float, default=env_float("EPOCHS", 1.0))
    parser.add_argument("--batch-size", type=int, default=env_int("BATCH_SIZE", 1))
    parser.add_argument("--grad-accum", type=int, default=env_int("GRAD_ACCUM", 8))
    parser.add_argument("--learning-rate", type=float, default=env_float("LEARNING_RATE", 2e-4))
    parser.add_argument("--max-seq-length", type=int, default=env_int("MAX_SEQ_LENGTH", 2048))
    parser.add_argument("--success-threshold", type=float, default=env_float("SUCCESS_THRESHOLD", 0.95))
    parser.add_argument("--wandb-enabled", action="store_true", default=env_bool("WANDB_ENABLED", False))
    parser.add_argument("--wandb-project", default=env_str("WANDB_PROJECT", "hidden-rule-game"))
    parser.add_argument("--wandb-run-name", default=env_str("WANDB_RUN_NAME", "") or None)
    parser.add_argument("--wandb-mode", default=env_str("WANDB_MODE", "online"))
    parser.add_argument("--run-dir", type=Path, default=Path(env_str("RUN_DIR", "")) if env_str("RUN_DIR", "") else None)
    parser.add_argument("--out-dir", type=Path, default=Path(env_str("OUT_DIR", "")) if env_str("OUT_DIR", "") else None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    run_dir = args.run_dir or Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = RunConfig(
        student=args.student,
        quick=args.quick,
        skip_train=args.skip_train,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        student_lora_path=args.student_lora_path,
        tensor_parallel_size=args.tensor_parallel_size,
        teacher_tensor_parallel_size=args.teacher_tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        vllm_dtype=args.vllm_dtype,
        vllm_download_dir=args.vllm_download_dir,
        served_model_name=args.served_model_name,
        train_rules=args.train_rules,
        eval_rules=args.eval_rules,
        examples_per_rule=args.examples_per_rule,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        rounds=args.rounds,
        eval_episodes=args.eval_episodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        success_threshold=args.success_threshold,
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        run_dir=run_dir,
        out_dir=args.out_dir,
    )
    cfg.apply_quick_defaults()
    return cfg


def train_if_needed(cfg: RunConfig, train_file: Path) -> None:
    if cfg.student != "lora" or cfg.skip_train:
        print(f"[3/4] Skipping LoRA training for student={cfg.student}")
        return

    print(f"[3/4] Training LoRA student at: {cfg.lora_dir}")
    from .trainer import train_lora

    train_lora(
        argparse.Namespace(
            model=cfg.student_model,
            train_file=train_file,
            out_dir=cfg.lora_dir,
            max_seq_length=cfg.max_seq_length,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            grad_accum=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            trust_remote_code=cfg.trust_remote_code,
            init_lora_path=cfg.student_lora_path,
        )
    )


def build_student(cfg: RunConfig) -> object:
    if cfg.student == "memory":
        return MemoryStudent()
    lora_path = str(cfg.eval_lora_path) if cfg.student == "lora" else None
    return LLMStudent(
        VLLMEngine(
            cfg.student_model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.vllm_dtype,
            lora_path=lora_path,
            lora_name="student_lora",
            trust_remote_code=cfg.trust_remote_code,
            download_dir=cfg.vllm_download_dir,
            served_model_name=cfg.served_model_name,
        )
    )


def build_teacher(cfg: RunConfig) -> object | None:
    if not cfg.teacher_model:
        return None
    return LLMTeacher(
        VLLMEngine(
            cfg.teacher_model,
            tensor_parallel_size=cfg.teacher_tensor_parallel_size,
            dtype=cfg.vllm_dtype,
            trust_remote_code=cfg.trust_remote_code,
            download_dir=cfg.vllm_download_dir,
            served_model_name=f"{cfg.served_model_name}-teacher" if cfg.served_model_name else None,
        )
    )


def init_wandb(cfg: RunConfig) -> object | None:
    if not cfg.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("WANDB_ENABLED=1 requires `pip install wandb`.") from exc

    os.environ["WANDB_MODE"] = cfg.wandb_mode
    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=_jsonable_config(cfg),
        dir=str(cfg.run_dir),
    )


def evaluate(cfg: RunConfig, wandb_run: object | None = None) -> dict[str, float]:
    print("[4/4] Running interactive evaluation")
    teacher = build_teacher(cfg)
    env = HiddenRuleEnv(seed=cfg.eval_seed, examples_per_episode=cfg.examples_per_rule, teacher=teacher)
    student = build_student(cfg)
    log_path = cfg.log_dir / f"game_eval_{cfg.student}.jsonl"
    txt_path = cfg.log_dir / f"game_eval_{cfg.student}.txt"
    rewards: list[float] = []
    accuracies: list[float] = []
    successes: list[float] = []
    turns: list[int] = []
    examples_used: list[int] = []

    with log_path.open("w", encoding="utf-8") as jsonl, txt_path.open("w", encoding="utf-8") as text_log:
        for idx in range(cfg.eval_episodes):
            result = env.run_episode(student, rounds=cfg.rounds)
            success = result.heldout_accuracy >= cfg.success_threshold
            rewards.append(result.reward)
            accuracies.append(result.heldout_accuracy)
            successes.append(float(success))
            turns.append(result.turns_taken)
            examples_used.append(result.examples_used)
            row = {
                "episode": idx,
                "rule_id": result.rule_id,
                "true_rule": result.true_rule,
                "guessed_rule": result.guessed_rule,
                "heldout_accuracy": result.heldout_accuracy,
                "reward": result.reward,
                "success": success,
                "turns_taken": result.turns_taken,
                "examples_used": result.examples_used,
                "rule_matched": result.rule_matched,
                "transcript": result.transcript,
            }
            line = json.dumps(row, ensure_ascii=False)
            pretty = json.dumps(row, ensure_ascii=False, indent=2)
            print(pretty)
            jsonl.write(line + "\n")
            text_log.write(pretty + "\n")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "episode/reward": result.reward,
                        "episode/heldout_accuracy": result.heldout_accuracy,
                        "episode/success": float(success),
                        "episode/turns_taken": result.turns_taken,
                        "episode/examples_used": result.examples_used,
                        "episode/rule_matched": float(result.rule_matched),
                    },
                    step=idx,
                )

    metrics = {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_heldout_accuracy": sum(accuracies) / len(accuracies),
        "success_rate": sum(successes) / len(successes),
        "avg_turns_needed": sum(turns) / len(turns),
        "avg_examples_used": sum(examples_used) / len(examples_used),
        "min_turns_needed": min(turns),
        "max_turns_needed": max(turns),
        "min_examples_used": min(examples_used),
        "max_examples_used": max(examples_used),
    }
    print(json.dumps(metrics, indent=2))
    with (cfg.log_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (cfg.log_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    if wandb_run is not None:
        wandb_run.log({f"eval/{key}": value for key, value in metrics.items()})
    return metrics


def write_run_config(cfg: RunConfig, metrics: dict[str, float]) -> None:
    payload = _jsonable_config(cfg)
    payload["metrics"] = metrics
    with (cfg.run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _jsonable_config(cfg: RunConfig) -> dict[str, object]:
    payload = asdict(cfg)
    payload["run_dir"] = str(cfg.run_dir)
    payload["out_dir"] = str(cfg.lora_dir)
    payload["student_lora_path"] = str(cfg.student_lora_path) if cfg.student_lora_path else None
    payload["eval_lora_path"] = str(cfg.eval_lora_path)
    payload["data_dir"] = str(cfg.data_dir)
    payload["log_dir"] = str(cfg.log_dir)
    return payload


def main() -> None:
    cfg = build_config(parse_args())
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(cfg)

    train_file = cfg.data_dir / "train.jsonl"
    eval_file = cfg.data_dir / "eval.jsonl"

    print(f"[1/4] Generating train set: {train_file}")
    generate_dataset(train_file, cfg.train_rules, cfg.examples_per_rule, cfg.train_seed)
    print(f"[2/4] Generating eval set: {eval_file}")
    generate_dataset(eval_file, cfg.eval_rules, cfg.examples_per_rule, cfg.eval_seed)
    train_if_needed(cfg, train_file)
    try:
        metrics = evaluate(cfg, wandb_run)
        write_run_config(cfg, metrics)
        print(f"Done. Run artifacts are in: {cfg.run_dir}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
