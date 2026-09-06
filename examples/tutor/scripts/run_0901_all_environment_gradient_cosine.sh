#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

GPU_COUNT="${1:-8}"
shift || true
if [[ "$GPU_COUNT" != "4" && "$GPU_COUNT" != "8" ]]; then
  echo "Usage: bash $0 [4|8] [extra analyzer arguments ...]" >&2
  exit 2
fi

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$GPU_COUNT" == "8" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
  fi
fi

RUN_NAME=20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu
OUTPUT_DIR="${GRADIENT_COSINE_OUTPUT_DIR:-$ROOT_DIR/output/gradient_cosine/${RUN_NAME}-step1000}"

exec "$ROOT_DIR/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node="$GPU_COUNT" \
  examples/tutor/scripts/analyze_environment_gradient_cosine.py \
  --checkpoint /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/checkpoints/root/tutor-math-baseline/${RUN_NAME}/default/epoch21epochstep12globalstep999 \
  --rollout-root /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/logs/root/tutor-math-baseline/${RUN_NAME}/rollout \
  --trace-dir /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/debug_traces/tutor-math-baseline/${RUN_NAME}/train \
  --output-dir "$OUTPUT_DIR" \
  --rollout-version-min 1000 \
  --rollout-version-max 1010 \
  --groups-per-environment 5 \
  "$@"
