#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"

CONFIG="${CONFIG:-${SCRIPT_DIR}/baseline-overfit-1-generalize-001020-lora-batch128-rebn-nomean-5.yaml}"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137}"
TRIAL_NAME="${TRIAL_NAME:-$(date +%Y%m%d_%H%M%S)_qwen32b-baseline-overfit-1-generalization-001020-lora-batch128-rebn-nomean-5}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing tutor config: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Missing Qwen3-32B model: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  echo "Missing .venv under ${ROOT_DIR}. Run: uv sync --extra cuda" >&2
  exit 1
fi

cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${INF_API_KEY:-}" ]]; then
  echo "Missing INF_API_KEY. Add it to ${ROOT_DIR}/.env or export it." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cmd=(
  python examples/tutor/train.py
  --config "${CONFIG}"
  "actor.path=${MODEL_PATH}"
  "trial_name=${TRIAL_NAME}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
