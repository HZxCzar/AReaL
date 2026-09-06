#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
UPSTREAM_DIR="$SCRIPT_DIR/.runtime/upstream"
DEPENDENCY_ROOT="$SCRIPT_DIR/.runtime/python"
DATASETS_CACHE="$SCRIPT_DIR/.runtime/hf_datasets"
DATASET_HUB_CACHE="$SCRIPT_DIR/.runtime/hf_hub"
UPSTREAM_REVISION=6faed173ec2bef55cb899b2a3e0f93982f9cb176
PED_RM_MODEL=${PED_RM_MODEL:-}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}

if (( $# != 1 )); then
  printf 'Usage: CUDA_VISIBLE_DEVICES=0 bash %s RUN_DIR\n' "$0" >&2
  exit 2
fi

RUN_DIR=$(realpath -e -- "$1")
if [[ ! -f "$RUN_DIR/run.json" || ! -d "$RUN_DIR/tasks" ]]; then
  printf 'Not a MathTutorBench result directory: %s\n' "$RUN_DIR" >&2
  exit 1
fi

# Reuse the exact absolute Ped-RM snapshot recorded by the completed run. This
# remains valid after the benchmark-specific Hugging Face caches are enabled.
if [[ -z "$PED_RM_MODEL" ]]; then
  PED_RM_MODEL=$(
    "$PYTHON" - "$RUN_DIR/run.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["pedrm_model"])
PY
  )
fi
if [[ "$PED_RM_MODEL" == /* && ! -f "$PED_RM_MODEL/config.json" ]]; then
  printf 'Recorded Ped-RM snapshot is unavailable: %s\n' "$PED_RM_MODEL" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  printf 'GPU_IDS must contain exactly 8 GPU ids for parallel rescoring; got %s\n' "$GPU_IDS" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for index in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$index]//[[:space:]]/}
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    printf 'GPU_IDS must contain 8 distinct integer ids; got %s\n' "$GPU_IDS" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
  GPU_ARRAY[$index]=$gpu
done

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="$DATASETS_CACHE"
export HF_HUB_CACHE="$DATASET_HUB_CACHE"
export PYTHONPATH="$DEPENDENCY_ROOT:$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

TASKS=(
  student_solution_correctness
  mistake_location
  problem_solving
  socratic_questioning
  scaffolding_generation
  pedagogy_following
  mistake_correction
  scaffolding_generation_hard
  pedagogy_following_hard
)

for task in "${TASKS[@]}"; do
  "$PYTHON" "$SCRIPT_DIR/run_task.py" \
    --upstream "$UPSTREAM_DIR" \
    --task "$task" \
    --output "$RUN_DIR/tasks/$task" \
    --reparse-only
done

PED_RM_PATH=$(
  "$PYTHON" "$SCRIPT_DIR/score_pedrm.py" --model "$PED_RM_MODEL" --resolve-only
)
SHARDS_ROOT="$RUN_DIR/pedrm/rescore-shards-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$SHARDS_ROOT"
PEDAGOGY_TASKS=(
  scaffolding_generation
  pedagogy_following
  scaffolding_generation_hard
  pedagogy_following_hard
)
PIDS=()
for worker in "${!GPU_ARRAY[@]}"; do
  task=${PEDAGOGY_TASKS[$((worker / 2))]}
  shard=$((worker % 2))
  gpu=${GPU_ARRAY[$worker]}
  log="$RUN_DIR/logs/pedrm-rescore-$task-$shard.log"
  printf '[pedrm] GPU %s: %s shard %s/2\n' "$gpu" "$task" "$((shard + 1))"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT_DIR/score_pedrm_shard.py" \
    --model "$PED_RM_PATH" \
    --tasks-root "$RUN_DIR/tasks" \
    --task "$task" \
    --shard-index "$shard" \
    --num-shards 2 \
    --output "$SHARDS_ROOT/$task-$shard.json" >"$log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for worker in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$worker]}"; then
    task=${PEDAGOGY_TASKS[$((worker / 2))]}
    shard=$((worker % 2))
    printf 'Ped-RM worker failed: GPU %s, %s shard %s. See logs.\n' \
      "${GPU_ARRAY[$worker]}" "$task" "$shard" >&2
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/merge_pedrm_shards.py" \
  --tasks-root "$RUN_DIR/tasks" \
  --shards-root "$SHARDS_ROOT" \
  --output "$RUN_DIR/pedrm" \
  --num-shards 2

"$PYTHON" "$SCRIPT_DIR/summarize.py" \
  --run-dir "$RUN_DIR" \
  --upstream-revision "$UPSTREAM_REVISION" | tee "$RUN_DIR/leaderboard.txt"

printf '[done] 8-GPU rescored report: %s/summary.yaml\n' "$RUN_DIR"
