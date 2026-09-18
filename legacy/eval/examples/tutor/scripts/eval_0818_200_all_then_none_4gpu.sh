#!/usr/bin/env bash
set -Eeuo pipefail

# Run the two 200-step full evaluations sequentially on four GPUs. All-ID is
# deliberately first so adding the None baseline cannot delay its result.

usage() {
  cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash examples/tutor/scripts/eval_0818_200_all_then_none_4gpu.sh

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash examples/tutor/scripts/eval_0818_200_all_then_none_4gpu.sh none

The first evaluation is All-ID step 200; the second is None step 200. Each
uses all 528 filtered rows and all eight student personalities.

Modes:
  sequence  Run All-ID and then None (default).
  all       Run only All-ID.
  none      Run only None, for example after All-ID already finished.

For a backfill rerun, ALL_EVAL_RUN_DIR and NONE_EVAL_RUN_DIR may independently
pin the corresponding existing output directory.
EOF
}

MODE="${1:-sequence}"
case "$MODE" in
  sequence|all|none) ;;
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
  printf 'Set CUDA_VISIBLE_DEVICES to exactly four GPU ids.\n' >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if (( ${#GPU_IDS[@]} != 4 )); then
  printf 'This sequence requires exactly four GPUs; got %s.\n' \
    "${#GPU_IDS[@]}" >&2
  exit 2
fi

TUTOR_FILEROOT="${TUTOR_FILEROOT:-$(
  "$PYTHON" -B - "$BASE_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

print(OmegaConf.load(sys.argv[1]).cluster.fileroot)
PY
)}"
CHECKPOINT_ROOT="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline"
STEP200=epoch4epochstep11globalstep199
ALL_TRIAL=20260826_072605_0818-personality-gate-credit-penalty-all-id-8gpu
NONE_TRIAL=20260824_192159_0818-personality-none-8gpu

ALL_ADAPTER="${ADAPTER_FULL:-$CHECKPOINT_ROOT/$ALL_TRIAL/default/$STEP200}"
NONE_ADAPTER="${ADAPTER_NONE:-$CHECKPOINT_ROOT/$NONE_TRIAL/default/$STEP200}"

for adapter in "$ALL_ADAPTER" "$NONE_ADAPTER"; do
  for required in adapter_model.safetensors adapter_config.json config.json; do
    if [[ ! -s "$adapter/$required" ]]; then
      printf 'Incomplete 200 checkpoint: %s is missing or empty.\n' \
        "$adapter/$required" >&2
      exit 1
    fi
  done
done

export EVAL_PERSONALITIES=all
export EVAL_MAX_SAMPLES=0
export EVAL_STRATIFIED_MAX_SAMPLES=0
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-20}"
export SAVE_TRACES="${SAVE_TRACES:-errors}"

ALL_BASE_PORT="${ALL_BASE_PORT:-${BASE_PORT:-33000}}"
NONE_BASE_PORT="${NONE_BASE_PORT:-$((ALL_BASE_PORT + 1000))}"

wait_for_all_ports_to_close() {
  "$PYTHON" -B - "$ALL_BASE_PORT" "$((ALL_BASE_PORT + 1))" \
    "$((ALL_BASE_PORT + 10))" "$((ALL_BASE_PORT + 11))" <<'PY'
import socket
import sys
import time

ports = [int(raw) for raw in sys.argv[1:]]
deadline = time.monotonic() + 180
while True:
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(("127.0.0.1", port))
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"All-ID inference servers did not release ports {ports} in 180s"
            )
        time.sleep(2)
    else:
        print(f"[sequence] All-ID ports released: {ports}")
        break
    finally:
        for sock in sockets:
            sock.close()
PY
}

if [[ "$MODE" == "sequence" || "$MODE" == "all" ]]; then
  printf '[sequence 1/2] All-ID step 200, full 528-row evaluation.\n'
  BASE_PORT="$ALL_BASE_PORT" \
  EVAL_RUN_DIR="${ALL_EVAL_RUN_DIR:-}" \
  EVAL_TEACHERS=full \
  ADAPTER_FULL="$ALL_ADAPTER" \
  bash "$EVAL_SCRIPT" full
fi

if [[ "$MODE" == "sequence" ]]; then
  wait_for_all_ports_to_close
fi

if [[ "$MODE" == "sequence" || "$MODE" == "none" ]]; then
  printf '[sequence 2/2] None step 200, full 528-row evaluation.\n'
  BASE_PORT="$NONE_BASE_PORT" \
  EVAL_RUN_DIR="${NONE_EVAL_RUN_DIR:-}" \
  EVAL_TEACHERS=none \
  ADAPTER_NONE="$NONE_ADAPTER" \
  bash "$EVAL_SCRIPT" full
fi

printf '[sequence done] Requested step-200 full evaluation mode finished: %s.\n' \
  "$MODE"
