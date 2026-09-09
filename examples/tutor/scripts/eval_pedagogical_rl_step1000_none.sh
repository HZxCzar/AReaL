#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluation only: frozen PedagogicalRL checkpoint, our reward-v4 test protocol.
# Usage: bash examples/tutor/scripts/eval_pedagogical_rl_step1000_none.sh preflight
#        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/tutor/scripts/eval_pedagogical_rl_step1000_none.sh run
# To resume/analyze, set EVAL_RUN_DIR to the directory printed by the first run.
# Four independent GPU pairs evaluate disjoint row-index shards (132/528 each).
# Full trajectories remain in cells/pedagogical-rl/qwen3-1.7b-text-original/
# shards/{0,1,2,3}/traces/presolve_on/. Merged results.jsonl links to each trace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Reuse the exact checkpoint recorded by the latest completed MathTutorBench run.
REFERENCE="$REPO_ROOT/examples/math_tutor_bench/results"
REFERENCE="$REFERENCE/0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu"
REFERENCE="$REFERENCE/epoch21epochstep12globalstep999/run.json"
CHECKPOINT_ROOT="$("$REPO_ROOT/.venv/bin/python" -B - "$REFERENCE" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(json.loads(Path(sys.argv[1]).read_text())["checkpoint"])
if checkpoint.name != "epoch21epochstep12globalstep999":
    raise SystemExit(f"Unexpected PedagogicalRL checkpoint: {checkpoint}")
print(checkpoint.parents[2])
PY
)"
export CHECKPOINT_ROOT
export COMMON_GLOBAL_STEP=999
export MATRIX_TEACHER_KEY=pedagogical-rl
export MATRIX_TEACHER_TRIAL=0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu
export MATRIX_TEACHER_KEYS=pedagogical-rl
export MATRIX_INCLUDE_NONE_STUDENT=1
export MATRIX_ONLY_NONE_STUDENT=1
export MATRIX_SHARD_NONE=1
export MATRIX_EVALUATOR="$SCRIPT_DIR/evaluate_api_teacher_sharded.py"
export MATRIX_STRATIFIED_SAMPLES=0
export MATRIX_EXPECTED_EXPLAIN_RATIO=1.0
export MATRIX_OUTPUT_TAG=pedagogical-rl-step1000-none-sharded-current-gates
export MATRIX_LAUNCHER_SCRIPT=examples/tutor/scripts/eval_pedagogical_rl_step1000_none.sh
export SAVE_TRACES=all

exec bash "$SCRIPT_DIR/eval_0901_all_reward_v4_step1000_8gpu.sh" "$@"
