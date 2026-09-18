#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paired extreme-setting comparison on the same step-600 checkpoints and the
# same deterministic 192-row subset as the explain-ratio-0.5 evaluation.
export COMMON_GLOBAL_STEP="${COMMON_GLOBAL_STEP:-599}"
export MATRIX_EVAL_CONFIG="$SCRIPT_DIR/../configs/math/0901/pilot/eval-all-preferences-explain0.yaml"
export MATRIX_EXPECTED_EXPLAIN_RATIO=0.0
export MATRIX_STRATIFIED_SAMPLES=192
export MATRIX_INCLUDE_NONE_STUDENT=0
export MATRIX_OUTPUT_TAG=explain0-n192
export MATRIX_LAUNCHER_SCRIPT=examples/tutor/scripts/eval_0901_preference_matrix_explain0_8gpu.sh

exec bash "$SCRIPT_DIR/eval_0901_preference_matrix_8gpu.sh" "$@"
