#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

TRIAL_NAME="${TRIAL_NAME:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tutor-grpo}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

python examples/tutor/train.py \
  --config examples/tutor/config.yaml \
  scheduler.type=local \
  cluster.n_gpus_per_node="$N_GPUS_PER_NODE" \
  experiment_name="$EXPERIMENT_NAME" \
  trial_name="$TRIAL_NAME" \
  "$@"
