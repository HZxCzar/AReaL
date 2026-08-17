#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-examples/pedagogical_rl/configs/qwen3_8b_qwen3_1_7b_math_pass2_baseline.yaml}"
shift || true

# Any argument containing '=' is a Hydra override; the first that does not is
# the trial name. Extra arguments used to be discarded silently, which made a
# dropped override look like it had been applied.
TRIAL_NAME=""
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == *=* ]]; then
    EXTRA_ARGS+=("${arg}")
  else
    TRIAL_NAME="${arg}"
  fi
done
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv under ${ROOT_DIR}." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 1
fi

source ".venv/bin/activate"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi
if [[ -z "${INF_API_KEY:-}" ]]; then
  echo "Missing INF_API_KEY. Add it to ${ROOT_DIR}/.env or export it." >&2
  exit 1
fi

# The INF endpoint is directly reachable and must bypass inherited proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1"
export no_proxy="${no_proxy:-},localhost,127.0.0.1"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-examples/pedagogical_rl/wandb_offline}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-examples/pedagogical_rl/.flashinfer}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -n "${TRIAL_NAME}" ]]; then
  exec python -m examples.pedagogical_rl.train \
    --config "${CONFIG}" \
    "trial_name=${TRIAL_NAME}" \
    "${EXTRA_ARGS[@]}"
else
  exec python -m examples.pedagogical_rl.train \
    --config "${CONFIG}" \
    "${EXTRA_ARGS[@]}"
fi
