#!/usr/bin/env bash
set -Eeuo pipefail

# Quick classifier-gate evaluation after changing the scripted EXPLAINING
# complaint. EXPLAINING starts immediately; the other six cells remain as small
# controls. live_summary.tsv reports first-post-complaint compliance in `post1`
# and all post-complaint turn compliance in `post_micro`.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0825_explaining_complaint_quick_8gpu.sh [preflight|run|analyze]

Run on eight GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0825_explaining_complaint_quick_8gpu.sh run

Analyze or resume using the output path printed by the run:
  EVAL_RUN_DIR=/path/to/run \
    bash examples/tutor/scripts/eval_0825_explaining_complaint_quick_8gpu.sh analyze

Defaults:
  TEACHER_VARIANT=untrained (fixed)
  EVAL_STRATIFIED_MAX_SAMPLES=12
  EVAL_CONCURRENCY=16
  SAVE_TRACES=all

Both the teacher and auxiliary gate use the untrained base Qwen3-8B without a
LoRA; the student is Qwen3-1.7B. Override the sample count through environment
variables when needed.
EOF
}

MODE="${1:-preflight}"
case "$MODE" in
  -h|--help)
    usage
    exit 0
    ;;
  preflight|run|analyze) ;;
  *)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    usage >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export GATE_DECISION_MODE=classifier
export TEACHER_VARIANT=untrained
unset ADAPTER_PATH
export EVAL_STRATIFIED_MAX_SAMPLES="${EVAL_STRATIFIED_MAX_SAMPLES:-12}"
export EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
export EVAL_FOCUS_PREFERENCE=explaining
export SAVE_TRACES="${SAVE_TRACES:-all}"
export LIVE_SUMMARY_INTERVAL_SECONDS="${LIVE_SUMMARY_INTERVAL_SECONDS:-5}"

exec bash "$ROOT_DIR/examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh" \
  "$MODE"
