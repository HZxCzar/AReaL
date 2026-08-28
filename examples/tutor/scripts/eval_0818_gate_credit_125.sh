#!/usr/bin/env bash
set -Eeuo pipefail

# Exact 125-step comparison for the surface and all-ID gate-credit runs.
# This wrapper only pins adapters and delegates to the standalone offline evaluator.

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0818_gate_credit_125.sh smoke

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash examples/tutor/scripts/eval_0818_gate_credit_125.sh subset

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0818_gate_credit_125.sh full

smoke starts the exact eight-GPU topology and runs one row in every cell.
subset uses the fixed 192-row type x level stratified sample on four GPUs.
full uses all 528 rows on eight GPUs.
EOF
}

REQUESTED_PHASE="${1:-}"
case "$REQUESTED_PHASE" in
  smoke)
    EXPECTED_GPU_COUNT=8
    EVAL_PHASE=quick
    ;;
  subset)
    EXPECTED_GPU_COUNT=4
    EVAL_PHASE=subset
    ;;
  full)
    EXPECTED_GPU_COUNT=8
    EVAL_PHASE=full
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
BASE_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/base/default.yaml"
EVAL_SCRIPT="$ROOT_DIR/examples/tutor/scripts/eval_0818_personality.sh"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'Set CUDA_VISIBLE_DEVICES to exactly %s GPU ids.\n' \
    "$EXPECTED_GPU_COUNT" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if (( ${#GPU_IDS[@]} != EXPECTED_GPU_COUNT )); then
  printf '%s requires exactly %s GPUs; got %s.\n' \
    "$REQUESTED_PHASE" "$EXPECTED_GPU_COUNT" "${#GPU_IDS[@]}" >&2
  exit 2
fi

if [[ "$REQUESTED_PHASE" == "smoke" ]]; then
  MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-45000}"
  for gpu in "${GPU_IDS[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    gpu_name="$(
      nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader
    )"
    gpu_memory_mib="$(
      nvidia-smi --id="$gpu" --query-gpu=memory.total \
        --format=csv,noheader,nounits
    )"
    gpu_memory_mib="${gpu_memory_mib//[[:space:]]/}"
    printf '[gpu] id=%s name=%s memory_mib=%s\n' \
      "$gpu" "$gpu_name" "$gpu_memory_mib"
    if (( gpu_memory_mib < MIN_GPU_MEMORY_MIB )); then
      printf 'GPU %s has less than %s MiB; refusing the 48G profile.\n' \
        "$gpu" "$MIN_GPU_MEMORY_MIB" >&2
      exit 1
    fi
  done
fi

TUTOR_FILEROOT="${TUTOR_FILEROOT:-$(
  "$PYTHON" -B - "$BASE_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

print(OmegaConf.load(sys.argv[1]).cluster.fileroot)
PY
)}"
CHECKPOINT_ROOT="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline"
STEP125=epoch2epochstep30globalstep124
SURFACE_TRIAL=20260826_070157_0818-personality-gate-credit-penalty-surface-8gpu
FULL_TRIAL=20260826_072605_0818-personality-gate-credit-penalty-all-id-8gpu

export ADAPTER_SURFACE="${ADAPTER_SURFACE:-$CHECKPOINT_ROOT/$SURFACE_TRIAL/default/$STEP125}"
# The generic matrix calls the joint/all-ID teacher "full".
export ADAPTER_FULL="${ADAPTER_FULL:-$CHECKPOINT_ROOT/$FULL_TRIAL/default/$STEP125}"

for adapter in "$ADAPTER_SURFACE" "$ADAPTER_FULL"; do
  for required in adapter_model.safetensors adapter_config.json config.json; do
    if [[ ! -s "$adapter/$required" ]]; then
      printf 'Incomplete 125 checkpoint: %s is missing or empty.\n' \
        "$adapter/$required" >&2
      exit 1
    fi
  done
done

export EVAL_TEACHERS=surface,full
export EVAL_PERSONALITIES=all
if [[ "$REQUESTED_PHASE" == "smoke" ]]; then
  export EVAL_MAX_SAMPLES=1
  export EVAL_STRATIFIED_MAX_SAMPLES=0
  export EVAL_CONCURRENCY=1
  export SAVE_TRACES=none
else
  export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-20}"
  export SAVE_TRACES="${SAVE_TRACES:-errors}"
fi

exec bash "$EVAL_SCRIPT" "$EVAL_PHASE"
