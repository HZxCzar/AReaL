#!/usr/bin/env bash
set -Eeuo pipefail

# Matched step-100 OOD comparison for the three personality teachers that were
# previously evaluated at different checkpoints. The generic evaluator retains
# the standard 0818 protocol; this wrapper only pins the teacher/student subset,
# the shared seeded 300-row sample, and the three LoRA checkpoints.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_SCRIPT="$ROOT_DIR/examples/tutor/scripts/eval_0818_personality.sh"

export EVAL_TEACHERS="none,contrasting_cases,full"
export EVAL_PERSONALITIES="ood"
export EVAL_MAX_SAMPLES="300"
export CELL_WALL_TIMEOUT_SECONDS="7200"
export EVAL_RUN_DIR="${EVAL_RUN_DIR:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/offline_eval/0818-personality/ood-step100-n300}"

export ADAPTER_NONE="/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/checkpoints/root/tutor-math-baseline/20260824_192159_0818-personality-none-8gpu/default/epoch2epochstep5globalstep99"
export ADAPTER_CONTRASTING_CASES="/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/checkpoints/root/tutor-math-baseline/20260824_082351_0818-personality-contrasting-cases-8gpu/default/epoch2epochstep5globalstep99"
export ADAPTER_FULL="/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/checkpoints/root/tutor-math-baseline/20260824_192211_0818-personality-all-8gpu/default/epoch2epochstep5globalstep99"

exec bash "$EVAL_SCRIPT" "${1:-full}"
