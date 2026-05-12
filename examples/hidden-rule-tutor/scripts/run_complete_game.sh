#!/usr/bin/env bash
set -euo pipefail

# Edit these toggles or override them from the command line.
# Minimal full training run:
#   STUDENT=lora STUDENT_MODEL=/models/student TEACHER_MODEL=/models/teacher bash scripts/run_complete_game.sh

export STUDENT="${STUDENT:-memory}"              # memory | llm | lora
export QUICK="${QUICK:-0}"                      # 1 for a tiny smoke run
export SKIP_TRAIN="${SKIP_TRAIN:-0}"            # 1 to reuse an existing LoRA OUT_DIR

# Model setup.
export STUDENT_MODEL="${STUDENT_MODEL:-${MODEL:-meta-llama/Llama-3.1-8B-Instruct}}"  # HF id or local path
export TEACHER_MODEL="${TEACHER_MODEL:-}"       # empty = deterministic tutor; otherwise HF id or local path
export STUDENT_LORA_PATH="${STUDENT_LORA_PATH:-}" # existing adapter to continue training or evaluate with SKIP_TRAIN=1
export OUT_DIR="${OUT_DIR:-}"                   # output adapter dir; empty = runs/<time>/outputs/lora-student

# vLLM setup.
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export TEACHER_TENSOR_PARALLEL_SIZE="${TEACHER_TENSOR_PARALLEL_SIZE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export VLLM_DTYPE="${VLLM_DTYPE:-auto}"
export VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

# Data and game setup.
export TRAIN_RULES="${TRAIN_RULES:-400}"
export EVAL_RULES="${EVAL_RULES:-80}"
export EXAMPLES_PER_RULE="${EXAMPLES_PER_RULE:-80}"
export EVAL_EPISODES="${EVAL_EPISODES:-50}"
export ROUNDS="${ROUNDS:-8}"

# LoRA training setup. Training starts when STUDENT=lora and SKIP_TRAIN=0.
export EPOCHS="${EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
export SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-0.95}"

export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-hidden-rule-game}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"       # online | offline | disabled

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m hidden_rule_game.run_complete_game "$@"
