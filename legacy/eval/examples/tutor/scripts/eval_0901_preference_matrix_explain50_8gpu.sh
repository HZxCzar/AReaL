#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directly compare against the existing 600-update, explain-ratio-1.0 matrix.
export COMMON_GLOBAL_STEP="${COMMON_GLOBAL_STEP:-599}"
export MATRIX_EVAL_CONFIG="$SCRIPT_DIR/../configs/math/0901/pilot/eval-all-preferences-explain50.yaml"
export MATRIX_EXPECTED_EXPLAIN_RATIO=0.5
export MATRIX_STRATIFIED_SAMPLES=192
export MATRIX_INCLUDE_NONE_STUDENT=0
export MATRIX_OUTPUT_TAG=explain50-n192
export MATRIX_LAUNCHER_SCRIPT=examples/tutor/scripts/eval_0901_preference_matrix_explain50_8gpu.sh

exec bash "$SCRIPT_DIR/eval_0901_preference_matrix_8gpu.sh" "$@"
