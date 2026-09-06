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

usage() {
  printf 'Usage: GPU_IDS=0,1,2,3,4,5,6,7 bash %s MODEL_DIR\n' "$0" >&2
  printf 'MODEL_DIR may be an AReaL LoRA checkpoint or a complete base-model snapshot.\n' >&2
}

if (( $# != 1 )); then
  usage
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  printf 'Python interpreter is not executable: %s\n' "$PYTHON" >&2
  exit 1
fi

MODEL_INPUT=$(realpath -e -- "$1")
if [[ -f "$MODEL_INPUT/adapter_config.json" && -f "$MODEL_INPUT/adapter_model.safetensors" ]]; then
  EVALUATION_MODE=lora
  CHECKPOINT="$MODEL_INPUT"
  INFERRED_BASE=$(
    "$PYTHON" - "$CHECKPOINT/adapter_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["base_model_name_or_path"])
PY
  )
  BASE_MODEL_PATH=${BASE_MODEL_PATH:-$INFERRED_BASE}
elif [[ -f "$MODEL_INPUT/config.json" ]] && \
     { compgen -G "$MODEL_INPUT/*.safetensors" >/dev/null || compgen -G "$MODEL_INPUT/pytorch_model*.bin" >/dev/null; }; then
  EVALUATION_MODE=base
  CHECKPOINT=""
  BASE_MODEL_PATH="$MODEL_INPUT"
else
  printf 'Neither a complete LoRA checkpoint nor a complete base model: %s\n' "$MODEL_INPUT" >&2
  exit 1
fi

if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  printf 'Base model is not a local directory: %s\n' "$BASE_MODEL_PATH" >&2
  printf 'This runner refuses to download a model; set BASE_MODEL_PATH to its local snapshot.\n' >&2
  exit 1
fi
BASE_MODEL_PATH=$(realpath -e -- "$BASE_MODEL_PATH")

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
if (( ${#GPU_ARRAY[@]} < 1 || ${#GPU_ARRAY[@]} > 8 )); then
  printf 'GPU_IDS must contain between 1 and 8 GPU ids; got %s\n' "$GPU_IDS" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for index in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$index]//[[:space:]]/}
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    printf 'Invalid GPU id in GPU_IDS: %s\n' "${GPU_ARRAY[$index]}" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    printf 'Duplicate GPU id in GPU_IDS: %s\n' "$gpu" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
  GPU_ARRAY[$index]=$gpu
done

BASE_PORT=${BASE_PORT:-32100}
REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-16}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_SAMPLES=${MAX_SAMPLES:-0}
SERVER_CONTEXT_LENGTH=${SERVER_CONTEXT_LENGTH:-40960}
SERVER_MEM_FRACTION_STATIC=${SERVER_MEM_FRACTION_STATIC:-0.80}
SERVER_MAX_RUNNING_REQUESTS=${SERVER_MAX_RUNNING_REQUESTS:-32}
SERVED_MODEL=${SERVED_MODEL:-math-tutor-bench-base}
PED_RM_MODEL=${PED_RM_MODEL:-eth-nlped/Qwen2.5-1.5B-pedagogical-rewardmodel}
SKIP_PED_RM=${SKIP_PED_RM:-0}

# Compute workers have no outbound network. Every benchmark asset was staged in
# this directory beforehand; force all Hugging Face libraries to use it directly
# instead of spending minutes on doomed metadata HEAD retries.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for numeric in BASE_PORT REQUEST_CONCURRENCY MAX_TOKENS MAX_SAMPLES SERVER_CONTEXT_LENGTH SERVER_MAX_RUNNING_REQUESTS; do
  value=${!numeric}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a nonnegative integer; got %s\n' "$numeric" "$value" >&2
    exit 2
  fi
done
if (( BASE_PORT < 1 || REQUEST_CONCURRENCY < 1 || MAX_TOKENS < 1 || SERVER_CONTEXT_LENGTH < 1 || SERVER_MAX_RUNNING_REQUESTS < 1 )); then
  printf 'Port, concurrency, token limits, and server limits must be positive.\n' >&2
  exit 2
fi
if (( BASE_PORT + ${#GPU_ARRAY[@]} - 1 > 65535 )); then
  printf 'BASE_PORT leaves too little room for %s endpoints.\n' "${#GPU_ARRAY[@]}" >&2
  exit 2
fi
if [[ ! "$SERVER_MEM_FRACTION_STATIC" =~ ^0\.[0-9]+$|^1\.0+$ ]]; then
  printf 'SERVER_MEM_FRACTION_STATIC must be between 0 and 1; got %s\n' "$SERVER_MEM_FRACTION_STATIC" >&2
  exit 2
fi
if [[ "$SKIP_PED_RM" != 0 && "$SKIP_PED_RM" != 1 ]]; then
  printf 'SKIP_PED_RM must be 0 or 1.\n' >&2
  exit 2
fi

if [[ "$EVALUATION_MODE" == lora ]]; then
  TRIAL_NAME=$(basename -- "$(dirname -- "$(dirname -- "$CHECKPOINT")")")
  CHECKPOINT_NAME=$(basename -- "$CHECKPOINT")
else
  TRIAL_NAME=base-Qwen3-8B
  CHECKPOINT_NAME=$(basename -- "$BASE_MODEL_PATH")
fi
RUN_DIR=${RUN_DIR:-$SCRIPT_DIR/results/$TRIAL_NAME/$CHECKPOINT_NAME}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/tasks" "$RUN_DIR/served_adapter" "$SCRIPT_DIR/.runtime"
RUN_DIR=$(realpath -e -- "$RUN_DIR")

"$PYTHON" "$SCRIPT_DIR/prepare.py" \
  --upstream "$UPSTREAM_DIR" \
  --dependency-root "$DEPENDENCY_ROOT" \
  --python "$PYTHON" \
  --offline

# Resolve the already-cached reward model before isolating benchmark dataset caches.
PED_RM_PATH=$(
  "$PYTHON" "$SCRIPT_DIR/score_pedrm.py" --model "$PED_RM_MODEL" --resolve-only
)

export PYTHONPATH="$DEPENDENCY_ROOT:$UPSTREAM_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$DATASETS_CACHE"
export HF_HUB_CACHE="$DATASET_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# Validate the already-staged official datasets before reserving GPUs. Offline
# mode makes a missing artifact fail immediately without any network retry.
"$PYTHON" "$SCRIPT_DIR/prefetch_datasets.py" --cache-dir "$HF_DATASETS_CACHE"

MANIFEST="$RUN_DIR/run.json"
"$PYTHON" - "$MANIFEST" "$CHECKPOINT" "$BASE_MODEL_PATH" "$PED_RM_PATH" \
  "$UPSTREAM_REVISION" "$GPU_IDS" "$MAX_TOKENS" "$MAX_SAMPLES" "$EVALUATION_MODE" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
new = {
    "checkpoint": sys.argv[2] or None,
    "base_model": sys.argv[3],
    "pedrm_model": sys.argv[4],
    "math_tutor_bench_revision": sys.argv[5],
    "gpu_ids": sys.argv[6],
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": int(sys.argv[7]),
    "max_samples": int(sys.argv[8]),
    "evaluation_mode": sys.argv[9],
    "status": "running",
}
if path.exists():
    old = json.loads(path.read_text(encoding="utf-8"))
    if "evaluation_mode" not in old:
        old["evaluation_mode"] = "lora" if old.get("checkpoint") else "base"
    immutable = ("checkpoint", "base_model", "pedrm_model", "math_tutor_bench_revision", "max_tokens", "evaluation_mode")
    mismatches = [key for key in immutable if old.get(key) != new.get(key)]
    if mismatches:
        raise SystemExit(f"RUN_DIR belongs to incompatible settings: {mismatches}")
    if old.get("max_samples", 0) not in (0, new["max_samples"]) and new["max_samples"]:
        raise SystemExit("cannot mix two nonzero MAX_SAMPLES values in one RUN_DIR")
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

ADAPTER_ALIAS=""
if [[ "$EVALUATION_MODE" == lora ]]; then
  ADAPTER_ALIAS="$RUN_DIR/served_adapter/teacher"
  if [[ -L "$ADAPTER_ALIAS" ]]; then
    if [[ "$(readlink -f -- "$ADAPTER_ALIAS")" != "$CHECKPOINT" ]]; then
      printf 'Existing adapter alias points to another checkpoint: %s\n' "$ADAPTER_ALIAS" >&2
      exit 1
    fi
  elif [[ -e "$ADAPTER_ALIAS" ]]; then
    printf 'Refusing to replace non-symlink adapter alias: %s\n' "$ADAPTER_ALIAS" >&2
    exit 1
  else
    ln -s "$CHECKPOINT" "$ADAPTER_ALIAS"
  fi
fi

PORTS=()
ENDPOINTS=()
for index in "${!GPU_ARRAY[@]}"; do
  port=$((BASE_PORT + index))
  PORTS+=("$port")
  ENDPOINTS+=("http://127.0.0.1:$port/v1")
done
"$PYTHON" - "${PORTS[@]}" <<'PY'
import socket
import sys

for raw_port in sys.argv[1:]:
    port = int(raw_port)
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as error:
        raise SystemExit(f"port {port} is unavailable: {error}") from error
    finally:
        sock.close()
PY

SERVER_PIDS=()
WORKER_PIDS=()
stop_workers() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  WORKER_PIDS=()
}

stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}

cleanup() {
  status=$?
  trap - EXIT INT TERM
  stop_workers
  stop_servers
  exit "$status"
}
trap cleanup EXIT INT TERM

printf '[run] mode:       %s\n' "$EVALUATION_MODE"
printf '[run] model:      %s\n' "$MODEL_INPUT"
printf '[run] output:     %s\n' "$RUN_DIR"
printf '[run] GPUs:       %s\n' "$GPU_IDS"
printf '[server] launching %s independent SGLang replicas\n' "${#GPU_ARRAY[@]}"
for index in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$index]}
  port=${PORTS[$index]}
  log="$RUN_DIR/logs/server-gpu-$gpu.log"
  printf '\n[%s] launching GPU %s on port %s\n' "$(date -u +%FT%TZ)" "$gpu" "$port" >>"$log"
  server_args=(
    --model-path "$BASE_MODEL_PATH"
    --served-model-name "$SERVED_MODEL"
    --host 127.0.0.1
    --port "$port"
    --tp-size 1
    --context-length "$SERVER_CONTEXT_LENGTH"
    --mem-fraction-static "$SERVER_MEM_FRACTION_STATIC"
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS"
    --random-seed 42
  )
  if [[ "$EVALUATION_MODE" == lora ]]; then
    server_args+=(
      --enable-lora
      --lora-paths "$ADAPTER_ALIAS"
      --max-loras-per-batch 1
      --max-loaded-loras 1
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$PYTHON" -m sglang.launch_server \
    "${server_args[@]}" >>"$log" 2>&1 &
  SERVER_PIDS+=("$!")
done

wait_for_server() {
  local pid=$1 endpoint=$2 log=$3 deadline
  deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -fsS --connect-timeout 2 --max-time 5 "$endpoint/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'SGLang exited before becoming ready; tail of %s:\n' "$log" >&2
      tail -80 "$log" >&2 || true
      return 1
    fi
    sleep 2
  done
  printf 'Timed out waiting for %s; tail of %s:\n' "$endpoint" "$log" >&2
  tail -80 "$log" >&2 || true
  return 1
}

for index in "${!GPU_ARRAY[@]}"; do
  gpu=${GPU_ARRAY[$index]}
  wait_for_server "${SERVER_PIDS[$index]}" "${ENDPOINTS[$index]}" \
    "$RUN_DIR/logs/server-gpu-$gpu.log"
  printf '[server] GPU %s ready at %s\n' "$gpu" "${ENDPOINTS[$index]}"
done

if [[ "$EVALUATION_MODE" == lora ]]; then
  "$PYTHON" "$SCRIPT_DIR/probe_adapter.py" \
    --base-url "${ENDPOINTS[0]}" \
    --model "$SERVED_MODEL" \
    --lora-path "$ADAPTER_ALIAS"
fi

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
WORK_ASSIGNMENTS=()
for index in "${!GPU_ARRAY[@]}"; do
  WORK_ASSIGNMENTS[$index]=""
done
if (( ${#GPU_ARRAY[@]} == 8 )); then
  WORK_ASSIGNMENTS[0]="student_solution_correctness"
  WORK_ASSIGNMENTS[1]="mistake_location"
  WORK_ASSIGNMENTS[2]="problem_solving"
  WORK_ASSIGNMENTS[3]="socratic_questioning"
  WORK_ASSIGNMENTS[4]="scaffolding_generation"
  WORK_ASSIGNMENTS[5]="pedagogy_following"
  WORK_ASSIGNMENTS[6]="mistake_correction"
  WORK_ASSIGNMENTS[7]="scaffolding_generation_hard pedagogy_following_hard"
else
  for index in "${!TASKS[@]}"; do
    worker=$((index % ${#GPU_ARRAY[@]}))
    WORK_ASSIGNMENTS[$worker]="${WORK_ASSIGNMENTS[$worker]} ${TASKS[$index]}"
  done
fi

run_worker() {
  local worker=$1 task log current_task_pid status
  current_task_pid=""
  worker_cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$current_task_pid" ]] && kill -0 "$current_task_pid" 2>/dev/null; then
      kill -TERM -- "-$current_task_pid" 2>/dev/null || kill -TERM "$current_task_pid" 2>/dev/null || true
      wait "$current_task_pid" 2>/dev/null || true
    fi
    exit "$status"
  }
  trap worker_cleanup EXIT INT TERM
  for task in ${WORK_ASSIGNMENTS[$worker]}; do
    log="$RUN_DIR/logs/task-$task.log"
    printf '[task] GPU %s starting %s\n' "${GPU_ARRAY[$worker]}" "$task"
    task_args=(
      --upstream "$UPSTREAM_DIR" \
      --task "$task" \
      --base-url "${ENDPOINTS[$worker]}" \
      --model "$SERVED_MODEL" \
      --output "$RUN_DIR/tasks/$task" \
      --concurrency "$REQUEST_CONCURRENCY" \
      --max-tokens "$MAX_TOKENS" \
      --max-samples "$MAX_SAMPLES"
    )
    if [[ "$EVALUATION_MODE" == lora ]]; then
      task_args+=(--lora-path "$ADAPTER_ALIAS")
    fi
    setsid "$PYTHON" "$SCRIPT_DIR/run_task.py" \
      "${task_args[@]}" >>"$log" 2>&1 &
    current_task_pid=$!
    wait "$current_task_pid"
    status=$?
    current_task_pid=""
    if (( status != 0 )); then
      return "$status"
    fi
    printf '[task] GPU %s finished %s\n' "${GPU_ARRAY[$worker]}" "$task"
  done
}

for index in "${!GPU_ARRAY[@]}"; do
  run_worker "$index" &
  WORKER_PIDS+=("$!")
done

worker_failure=0
set +e
for index in "${!WORKER_PIDS[@]}"; do
  wait "${WORKER_PIDS[$index]}"
  status=$?
  if (( status != 0 )); then
    printf 'Task worker on GPU %s failed with status %s. See %s/logs/.\n' \
      "${GPU_ARRAY[$index]}" "$status" "$RUN_DIR" >&2
    worker_failure=1
  fi
done
set -e
WORKER_PIDS=()
if (( worker_failure )); then
  exit 1
fi

printf '[server] generation complete; stopping SGLang replicas\n'
stop_servers

if [[ "$SKIP_PED_RM" == 0 ]]; then
  if (( ${#GPU_ARRAY[@]} == 8 )); then
    printf '[pedrm] scoring four open-ended tasks across all 8 GPUs\n'
    GPU_IDS="$GPU_IDS" PED_RM_MODEL="$PED_RM_PATH" PYTHON="$PYTHON" \
      bash "$SCRIPT_DIR/rescore.sh" "$RUN_DIR"
  else
    printf '[pedrm] scoring four open-ended tasks on GPU %s\n' "${GPU_ARRAY[0]}"
    CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "$PYTHON" "$SCRIPT_DIR/score_pedrm.py" \
      --model "$PED_RM_PATH" \
      --tasks-root "$RUN_DIR/tasks" \
      --output "$RUN_DIR/pedrm" 2>&1 | tee -a "$RUN_DIR/logs/pedrm.log"
  fi
fi

"$PYTHON" "$SCRIPT_DIR/summarize.py" \
  --run-dir "$RUN_DIR" \
  --upstream-revision "$UPSTREAM_REVISION" | tee "$RUN_DIR/leaderboard.txt"

"$PYTHON" - "$MANIFEST" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "complete"
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

printf '[done] full report: %s/summary.yaml\n' "$RUN_DIR"
