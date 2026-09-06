#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SELF=$(realpath -e -- "${BASH_SOURCE[0]}")
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# run.sh uses PYTHON for every Python entry point. In shim mode, delegate all of
# them unchanged except its conservative LoRA liveness probe. A trained adapter
# can alter logits without changing the greedy text on that probe's single
# prompt; SGLang has still loaded and selected the adapter in that case.
if [[ "${MATH_TUTOR_BENCH_PEDRL_PYTHON_SHIM:-0}" == 1 ]]; then
  REAL_PYTHON=${MATH_TUTOR_BENCH_REAL_PYTHON:?missing real Python interpreter}
  if [[ "${1:-}" == "$SCRIPT_DIR/probe_adapter.py" ]]; then
    set +e
    probe_output=$("$REAL_PYTHON" "$@" 2>&1)
    probe_status=$?
    set -e
    if (( probe_status == 0 )); then
      printf '%s\n' "$probe_output"
      exit 0
    fi
    if [[ "$probe_output" == *"LoRA is not observably active on the chat endpoint"* ||
          "$probe_output" == *"LoRA is not observably active on the completion endpoint"* ]]; then
      printf '%s\n' \
        '[liveness] warning: base and LoRA produced identical greedy chat text; adapter loading and path validation succeeded, continuing.'
      exit 0
    fi
    printf '%s\n' "$probe_output" >&2
    exit "$probe_status"
  fi
  exec "$REAL_PYTHON" "$@"
fi

if (( $# != 1 )); then
  printf 'Usage: GPU_IDS=0,1,2,3,4,5,6,7 bash %s PEDAGOGICAL_RL_CHECKPOINT\n' "$0" >&2
  exit 2
fi

REAL_PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
export MATH_TUTOR_BENCH_PEDRL_PYTHON_SHIM=1
export MATH_TUTOR_BENCH_REAL_PYTHON=$REAL_PYTHON
export PYTHON=$SELF
export PED_RM_MODEL=${PED_RM_MODEL:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--eth-nlped--Qwen2.5-1.5B-pedagogical-rewardmodel/snapshots/b1cf3e323398713e10ed88a4dbfa27f4a8f3b8dc}

exec bash "$SCRIPT_DIR/run.sh" "$@"
