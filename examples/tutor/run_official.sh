#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-"examples/tutor/config.yaml"}"
TRIAL_NAME="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv under $ROOT_DIR. Run: uv sync --extra cuda" >&2
  exit 1
fi

source ".venv/bin/activate"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${INF_API_KEY:-}" ]]; then
  echo "Missing INF_API_KEY. Add it to $ROOT_DIR/.env or export it before running." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ -n "$TRIAL_NAME" ]]; then
  python examples/tutor/train.py --config "$CONFIG" "trial_name=$TRIAL_NAME"
else
  python examples/tutor/train.py --config "$CONFIG"
fi
