#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Full evaluation of the ALL teacher after 1500 completed updates on all seven
# students. globalstep1499 is the checkpoint after update 1500. The frozen
# legacy Gate prompt pool keeps these results comparable with earlier matrices.
export COMMON_GLOBAL_STEP=1499
export MATRIX_TEACHER_KEYS=all-id
export MATRIX_EVAL_CONFIG="$SCRIPT_DIR/../configs/math/0901/pilot/eval-all-preferences-explain100-legacy-gates.yaml"
export MATRIX_EXPECTED_EXPLAIN_RATIO=1.0
export MATRIX_STRATIFIED_SAMPLES=0
export MATRIX_INCLUDE_NONE_STUDENT=1
export MATRIX_OUTPUT_TAG=all-only-explain100-legacy-gates
export MATRIX_LAUNCHER_SCRIPT=examples/tutor/scripts/eval_0901_all_step1500_legacy_gates_8gpu.sh

exec bash "$SCRIPT_DIR/eval_0901_preference_matrix_8gpu.sh" "$@"
