#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete 0825 classification evaluation twice on the same eight GPUs:
# first with the trained 0818 LoRA teacher, then with the untrained Qwen3-8B base
# teacher. Gate/leak/answer judges are base Qwen3-8B in both runs; students are
# Qwen3-1.7B in both runs.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0825_personality_trained_vs_untrained_full_8gpu.sh [preflight|run]

Run both complete evaluations sequentially:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0825_personality_trained_vs_untrained_full_8gpu.sh run

The script fixes EVAL_STRATIFIED_MAX_SAMPLES=0, which means the complete test
set. It prints one EVAL_STAMP shared by the trained and untrained output paths.
Reuse that EVAL_STAMP to resume an interrupted pair of runs.
EOF
}

MODE="${1:-preflight}"
case "$MODE" in
  -h|--help)
    usage
    exit 0
    ;;
  preflight|run) ;;
  *)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -n "${EVAL_RUN_DIR:-}" ]]; then
  printf 'Do not set EVAL_RUN_DIR for the two-run wrapper; use EVAL_STAMP to resume.\n' >&2
  exit 2
fi
if [[ -n "${EVAL_STRATIFIED_MAX_SAMPLES:-}" && \
      "$EVAL_STRATIFIED_MAX_SAMPLES" != "0" ]]; then
  printf 'This wrapper is full-eval only: EVAL_STRATIFIED_MAX_SAMPLES must be 0.\n' >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAUNCHER="$ROOT_DIR/examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh"
EVAL_STAMP="${EVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export EVAL_STAMP
export EVAL_STRATIFIED_MAX_SAMPLES=0
export CELL_WALL_TIMEOUT_SECONDS="${CELL_WALL_TIMEOUT_SECONDS:-86400}"

printf '[comparison] full dataset; trained then untrained; stamp=%s\n' "$EVAL_STAMP"

TEACHER_VARIANT=trained bash "$LAUNCHER" "$MODE"
env -u ADAPTER_PATH TEACHER_VARIANT=untrained bash "$LAUNCHER" "$MODE"

printf '[comparison] both teacher variants finished; stamp=%s\n' "$EVAL_STAMP"
