#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-"examples/tutor/config.yaml"}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${2:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

python examples/tutor/train.py \
  --config "$CONFIG"
