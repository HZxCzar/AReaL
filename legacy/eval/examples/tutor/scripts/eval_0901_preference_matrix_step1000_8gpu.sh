#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full 5-checkpoint x 7-student evaluation at 1000 completed updates.
export COMMON_GLOBAL_STEP="${COMMON_GLOBAL_STEP:-999}"
export MATRIX_EVAL_CONFIG="$SCRIPT_DIR/../configs/math/0901/pilot/eval-all-preferences-explain100.yaml"
export MATRIX_EXPECTED_EXPLAIN_RATIO=1.0
export MATRIX_STRATIFIED_SAMPLES=0
export MATRIX_INCLUDE_NONE_STUDENT=1
export MATRIX_OUTPUT_TAG=explain100
export MATRIX_LAUNCHER_SCRIPT=examples/tutor/scripts/eval_0901_preference_matrix_step1000_8gpu.sh

exec bash "$SCRIPT_DIR/eval_0901_preference_matrix_8gpu.sh" "$@"
