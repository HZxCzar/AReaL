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

# Everything after the trial name is forwarded to hydra untouched, so a one-off
# override -- a warm start, a shorter run -- does not need a config file of its own.
# NOTE that passing a trial name suppresses the automatic timestamp prefix
# (train.py), so reusing one means recover.mode=auto finds that run's own state.
EXTRA=("${@:3}")
if [[ -n "$TRIAL_NAME" ]]; then
  python examples/tutor/train.py --config "$CONFIG" "trial_name=$TRIAL_NAME"     ${EXTRA[@]+"${EXTRA[@]}"}
else
  python examples/tutor/train.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
fi
