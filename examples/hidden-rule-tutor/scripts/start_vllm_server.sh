#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"  # HF id or local model path
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export VLLM_DTYPE="${VLLM_DTYPE:-auto}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
export VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
export LORA_PATH="${LORA_PATH:-}"
export LORA_NAME="${LORA_NAME:-student_lora}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --dtype "$VLLM_DTYPE"
)

[[ -n "$SERVED_MODEL_NAME" ]] && ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
[[ "$TRUST_REMOTE_CODE" == "1" ]] && ARGS+=(--trust-remote-code)
[[ -n "$VLLM_DOWNLOAD_DIR" ]] && ARGS+=(--download-dir "$VLLM_DOWNLOAD_DIR")
[[ -n "$LORA_PATH" ]] && ARGS+=(--lora-path "$LORA_PATH" --lora-name "$LORA_NAME")

python -m hidden_rule_game.serve_vllm "${ARGS[@]}"
