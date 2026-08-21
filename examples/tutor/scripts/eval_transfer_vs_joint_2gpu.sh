#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="${EVAL_CONFIG:-$ROOT_DIR/examples/tutor/configs/math/0810/4gpu/two-student-text-code-eval.yaml}"
TRANSFER_CHECKPOINT="${TRANSFER_CHECKPOINT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/checkpoints/root/tutor-math-baseline/20260813_053934_0810fc-leak-local/default/epoch3epochstep8globalstep149}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/checkpoints/root/tutor-math-baseline/20260815_204624_0810fc-two-student-local-format-terminate/default/epoch6epochstep17globalstep299}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
STUDENT_PORT="${STUDENT_PORT:-30001}"
TEACHER_BASE_URL="http://127.0.0.1:${TEACHER_PORT}/v1"
STUDENT_BASE_URL="http://127.0.0.1:${STUDENT_PORT}/v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/offline_eval/fair-transfer-step150-vs-joint-step300-code-freechat-v2}"

# Code-only by default. Use a comma-separated list to opt into more configured
# students without changing the fair protocol.
EVAL_STUDENT_NAMES="${EVAL_STUDENT_NAMES:-qwen3-1.7b-code}"
# Full eval is 528 base rows x selected student modes for each checkpoint.
EVAL_LIMIT="${EVAL_LIMIT:-0}"
# A real end-to-end smoke runs before the full pass and must show five free-chat
# turns, four original retests, and a healthy code channel.
SMOKE_LIMIT="${SMOKE_LIMIT:-2}"
# Two checkpoint jobs run concurrently, so aggregate episode concurrency is 32.
EVAL_CONCURRENCY_PER_RUN="${EVAL_CONCURRENCY_PER_RUN:-16}"
SMOKE_CONCURRENCY_PER_RUN="${SMOKE_CONCURRENCY_PER_RUN:-2}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.80}"
STUDENT_MEM_FRACTION_STATIC="${STUDENT_MEM_FRACTION_STATIC:-0.80}"
TEACHER_MAX_RUNNING_REQUESTS="${TEACHER_MAX_RUNNING_REQUESTS:-32}"
STUDENT_MAX_RUNNING_REQUESTS="${STUDENT_MAX_RUNNING_REQUESTS:-32}"
MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-40960}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
# Successful traces are unnecessary for metrics and thousands of files add I/O.
SAVE_TRACES="${SAVE_TRACES:-errors}"
TEACHER_REQUEST_PARAMS="${TEACHER_REQUEST_PARAMS:-}"
if [[ -z "$TEACHER_REQUEST_PARAMS" ]]; then
  TEACHER_REQUEST_PARAMS='{"seed":42,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
fi

for required in "$PYTHON" "$CONFIG" \
  "$TRANSFER_CHECKPOINT/adapter_model.safetensors" \
  "$TRANSFER_CHECKPOINT/adapter_config.json" \
  "$JOINT_CHECKPOINT/adapter_model.safetensors" \
  "$JOINT_CHECKPOINT/adapter_config.json" \
  "$TEACHER_MODEL_PATH/config.json" \
  "$STUDENT_MODEL_PATH/config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if ! [[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "EVAL_LIMIT must be a non-negative integer, got: $EVAL_LIMIT" >&2
  exit 1
fi
if ! [[ "$SMOKE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_LIMIT must be positive, got: $SMOKE_LIMIT" >&2
  exit 1
fi
for concurrency in "$EVAL_CONCURRENCY_PER_RUN" "$SMOKE_CONCURRENCY_PER_RUN"; do
  if ! [[ "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
    echo "Eval concurrency must be positive, got: $concurrency" >&2
    exit 1
  fi
done

IFS=',' read -r -a RAW_STUDENT_NAMES <<<"$EVAL_STUDENT_NAMES"
STUDENT_ARGS=()
SELECTED_STUDENT_NAMES=()
for name in "${RAW_STUDENT_NAMES[@]}"; do
  name="${name#"${name%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"
  if [[ -n "$name" ]]; then
    STUDENT_ARGS+=(--student-name "$name")
    SELECTED_STUDENT_NAMES+=("$name")
  fi
done
if (( ${#SELECTED_STUDENT_NAMES[@]} == 0 )); then
  echo "EVAL_STUDENT_NAMES selected no students." >&2
  exit 1
fi
for port in "$TEACHER_PORT" "$STUDENT_PORT"; do
  if ! [[ "$port" =~ ^[1-9][0-9]*$ ]] || (( port > 65535 )); then
    echo "Ports must be in 1..65535, got: $port" >&2
    exit 1
  fi
done
if [[ "$TEACHER_PORT" == "$STUDENT_PORT" ]]; then
  echo "TEACHER_PORT and STUDENT_PORT must differ." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

# Deliberately override .env: no external inference endpoint is used in this run.
export INF_API_KEY="${INF_API_KEY:-EMPTY}"
export TUTOR_QWEN3_8B_BASE_URL="$TEACHER_BASE_URL"
export TUTOR_QWEN3_1_7B_BASE_URL="$STUDENT_BASE_URL"

VISIBLE_GPU_SPEC="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a GPU_IDS <<<"$VISIBLE_GPU_SPEC"
if (( ${#GPU_IDS[@]} < 2 )); then
  echo "Need two visible GPUs; CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_SPEC" >&2
  exit 1
fi
TEACHER_GPU="${GPU_IDS[0]}"
STUDENT_GPU="${GPU_IDS[1]}"
if ! CUDA_VISIBLE_DEVICES="$VISIBLE_GPU_SPEC" "$PYTHON" -c \
  'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1)'; then
  echo "PyTorch does not see two GPUs with CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_SPEC" >&2
  exit 1
fi

EXPECTED_BASE_ROWS=528
if (( EVAL_LIMIT > 0 && EVAL_LIMIT < EXPECTED_BASE_ROWS )); then
  EXPECTED_BASE_ROWS=$EVAL_LIMIT
fi
EXPECTED_EPISODES_PER_CHECKPOINT=$((EXPECTED_BASE_ROWS * ${#SELECTED_STUDENT_NAMES[@]}))

TRANSFER_OUTPUT="$OUTPUT_ROOT/transfer_step150"
JOINT_OUTPUT="$OUTPUT_ROOT/joint_step300"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_TRANSFER_OUTPUT="$OUTPUT_ROOT/_smoke/$stamp/transfer_step150"
SMOKE_JOINT_OUTPUT="$OUTPUT_ROOT/_smoke/$stamp/joint_step300"
mkdir -p \
  "$TRANSFER_OUTPUT" "$JOINT_OUTPUT" \
  "$SMOKE_TRANSFER_OUTPUT" "$SMOKE_JOINT_OUTPUT"
TEACHER_SERVER_LOG="$OUTPUT_ROOT/sglang-shared-teacher-$stamp.log"
STUDENT_SERVER_LOG="$OUTPUT_ROOT/sglang-shared-student-$stamp.log"
TRANSFER_EVAL_LOG="$OUTPUT_ROOT/eval-transfer-step150-$stamp.log"
JOINT_EVAL_LOG="$OUTPUT_ROOT/eval-joint-step300-$stamp.log"
SMOKE_TRANSFER_LOG="$OUTPUT_ROOT/smoke-transfer-step150-$stamp.log"
SMOKE_JOINT_LOG="$OUTPUT_ROOT/smoke-joint-step300-$stamp.log"

SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  local pid
  for pid in "${EVAL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

for url in "$TEACHER_BASE_URL" "$STUDENT_BASE_URL"; do
  if curl --silent --show-error --fail --max-time 2 "$url/models" >/dev/null 2>&1; then
    echo "An OpenAI-compatible server already occupies $url; choose different ports." >&2
    exit 1
  fi
done

echo "Starting shared Qwen3-8B teacher + two LoRAs on GPU $TEACHER_GPU."
CUDA_VISIBLE_DEVICES="$TEACHER_GPU" "$PYTHON" -m sglang.launch_server \
  --model-path "$TEACHER_MODEL_PATH" \
  --served-model-name "$TEACHER_MODEL" \
  --host 127.0.0.1 \
  --port "$TEACHER_PORT" \
  --tp-size 1 \
  --context-length "$MODEL_CONTEXT_LENGTH" \
  --mem-fraction-static "$TEACHER_MEM_FRACTION_STATIC" \
  --max-running-requests "$TEACHER_MAX_RUNNING_REQUESTS" \
  --enable-lora \
  --lora-paths "$TRANSFER_CHECKPOINT" "$JOINT_CHECKPOINT" \
  --max-loras-per-batch 2 \
  --max-loaded-loras 2 \
  >"$TEACHER_SERVER_LOG" 2>&1 &
TEACHER_SERVER_PID=$!
SERVER_PIDS+=("$TEACHER_SERVER_PID")

echo "Starting shared Qwen3-1.7B text/code student on GPU $STUDENT_GPU."
CUDA_VISIBLE_DEVICES="$STUDENT_GPU" "$PYTHON" -m sglang.launch_server \
  --model-path "$STUDENT_MODEL_PATH" \
  --served-model-name "$STUDENT_MODEL" \
  --host 127.0.0.1 \
  --port "$STUDENT_PORT" \
  --tp-size 1 \
  --context-length "$MODEL_CONTEXT_LENGTH" \
  --mem-fraction-static "$STUDENT_MEM_FRACTION_STATIC" \
  --max-running-requests "$STUDENT_MAX_RUNNING_REQUESTS" \
  >"$STUDENT_SERVER_LOG" 2>&1 &
STUDENT_SERVER_PID=$!
SERVER_PIDS+=("$STUDENT_SERVER_PID")

wait_ready() {
  local pid=$1
  local url=$2
  local log_path=$3
  local label=$4
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited before becoming ready. Last lines of $log_path:" >&2
      tail -n 100 "$log_path" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 3 "$url/models" >/dev/null 2>&1; then
      echo "$label is ready at $url"
      return 0
    fi
    sleep 2
  done
  echo "$label was not ready after ${SERVER_READY_TIMEOUT}s. Last lines of $log_path:" >&2
  tail -n 100 "$log_path" >&2 || true
  return 1
}

wait_ready "$TEACHER_SERVER_PID" "$TEACHER_BASE_URL" "$TEACHER_SERVER_LOG" "teacher server"
wait_ready "$STUDENT_SERVER_PID" "$STUDENT_BASE_URL" "$STUDENT_SERVER_LOG" "student server"

run_eval() {
  local checkpoint=$1
  local output=$2
  local log_path=$3
  local limit=$4
  local traces=$5
  local concurrency=$6
  "$PYTHON" examples/tutor/scripts/eval_checkpoints.py \
    --checkpoint "$checkpoint" \
    --base-url "$TEACHER_BASE_URL" \
    --model "$TEACHER_MODEL" \
    --api-key EMPTY \
    --config "$CONFIG" \
    --teacher-request-params "$TEACHER_REQUEST_PARAMS" \
    --output-root "$output" \
    --concurrency "$concurrency" \
    --resume \
    -- --limit "$limit" --save-traces "$traces" "${STUDENT_ARGS[@]}" \
    >"$log_path" 2>&1 &
  LAST_EVAL_PID=$!
  EVAL_PIDS+=("$LAST_EVAL_PID")
}

wait_pair() {
  local transfer_pid=$1
  local joint_pid=$2
  local transfer_log=$3
  local joint_log=$4
  local label=$5
  local transfer_status
  local joint_status
  set +e
  wait "$transfer_pid"
  transfer_status=$?
  wait "$joint_pid"
  joint_status=$?
  set -e
  if (( transfer_status != 0 || joint_status != 0 )); then
    if (( transfer_status != 0 )); then
      echo "$label transfer eval failed. Last lines of $transfer_log:" >&2
      tail -n 120 "$transfer_log" >&2 || true
    fi
    if (( joint_status != 0 )); then
      echo "$label joint eval failed. Last lines of $joint_log:" >&2
      tail -n 120 "$joint_log" >&2 || true
    fi
    return 1
  fi
}

validate_smoke() {
  local transfer_results=$1
  local joint_results=$2
  "$PYTHON" - \
    "$transfer_results" "$joint_results" \
    "$SMOKE_LIMIT" "${SELECTED_STUDENT_NAMES[*]}" <<'PY'
import json
import sys
from pathlib import Path

paths = [Path(sys.argv[1]), Path(sys.argv[2])]
limit = int(sys.argv[3])
expected_students = set(sys.argv[4].split())
expected_rows = limit * len(expected_students)
for path in paths:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise SystemExit(f"smoke {path}: expected {expected_rows} rows, got {len(rows)}")
    code_turns = 0
    code_no_program = 0
    code_crashes = 0
    code_silent_cells = 0
    code_constant_prints = 0
    for row in rows:
        label = f"{path}:{row.get('student_name')}:{row.get('item_id')}"
        if row.get("error") is not None:
            raise SystemExit(f"{label}: episode error: {row['error']}")
        if row.get("student_name") not in expected_students:
            raise SystemExit(f"{label}: unexpected student")
        if not row.get("free_chat_enabled"):
            raise SystemExit(f"{label}: free_chat was not enabled")
        if row.get("pre_solved"):
            raise SystemExit(f"{label}: free-chat eval cannot be pre_solved")
        if row.get("termination_reason") != "max_turns" or row.get("num_turns") != 5:
            raise SystemExit(
                f"{label}: expected five-round free chat, got "
                f"termination={row.get('termination_reason')} turns={row.get('num_turns')}"
            )
        generalization = row.get("generalization") or {}
        original = generalization.get("original") or {}
        if not original.get("attempted") or original.get("replay_count") != 4:
            raise SystemExit(f"{label}: original four-replay retest did not run")
        if "level1" in generalization or "level2" in generalization:
            raise SystemExit(f"{label}: disabled transfer variants unexpectedly ran")
        if row.get("student_name", "").endswith("-code"):
            code_stats = row.get("code_stats")
            if not isinstance(code_stats, dict):
                raise SystemExit(f"{label}: code channel stats are missing")
            code_turns += int(row.get("num_turns", 0))
            code_no_program += int(code_stats.get("no_program", 0))
            code_crashes += int(code_stats.get("crashes", 0))
            code_silent_cells += int(code_stats.get("silent_cells", 0))
            code_constant_prints += int(code_stats.get("constant_prints", 0))
    if code_turns:
        # no_program is a regular, scored code-student outcome. The original
        # training run logged it as a channel-health metric (it was not an
        # episode error), so a fair evaluator must report it rather than abort.
        # The smoke still proves that the code workflow, execution stats, and
        # four-replay retest all ran end to end.
        print(
            f"smoke code health {path}: no_program={code_no_program}/{code_turns} "
            f"({code_no_program / code_turns:.1%}), crashes={code_crashes}, "
            f"silent_cells={code_silent_cells}, "
            f"constant_prints={code_constant_prints}"
        )
print("smoke semantic checks passed")
PY
}

echo "Running a ${SMOKE_LIMIT}-row semantic smoke before the full evaluation."
run_eval \
  "$TRANSFER_CHECKPOINT" "$SMOKE_TRANSFER_OUTPUT" "$SMOKE_TRANSFER_LOG" \
  "$SMOKE_LIMIT" all "$SMOKE_CONCURRENCY_PER_RUN"
SMOKE_TRANSFER_PID=$LAST_EVAL_PID
run_eval \
  "$JOINT_CHECKPOINT" "$SMOKE_JOINT_OUTPUT" "$SMOKE_JOINT_LOG" \
  "$SMOKE_LIMIT" all "$SMOKE_CONCURRENCY_PER_RUN"
SMOKE_JOINT_PID=$LAST_EVAL_PID
wait_pair \
  "$SMOKE_TRANSFER_PID" "$SMOKE_JOINT_PID" \
  "$SMOKE_TRANSFER_LOG" "$SMOKE_JOINT_LOG" "Smoke"

SMOKE_TRANSFER_RESULTS="$SMOKE_TRANSFER_OUTPUT/step0149/results.jsonl"
SMOKE_JOINT_RESULTS="$SMOKE_JOINT_OUTPUT/step0299/results.jsonl"
validate_smoke "$SMOKE_TRANSFER_RESULTS" "$SMOKE_JOINT_RESULTS"

echo "Smoke passed. Starting transfer and joint full evals against the same services."
run_eval \
  "$TRANSFER_CHECKPOINT" "$TRANSFER_OUTPUT" "$TRANSFER_EVAL_LOG" \
  "$EVAL_LIMIT" "$SAVE_TRACES" "$EVAL_CONCURRENCY_PER_RUN"
TRANSFER_EVAL_PID=$LAST_EVAL_PID
run_eval \
  "$JOINT_CHECKPOINT" "$JOINT_OUTPUT" "$JOINT_EVAL_LOG" \
  "$EVAL_LIMIT" "$SAVE_TRACES" "$EVAL_CONCURRENCY_PER_RUN"
JOINT_EVAL_PID=$LAST_EVAL_PID

count_results() {
  local path=$1
  if [[ -f "$path" ]]; then
    wc -l <"$path"
  else
    echo 0
  fi
}

TRANSFER_RESULTS="$TRANSFER_OUTPUT/step0149/results.jsonl"
JOINT_RESULTS="$JOINT_OUTPUT/step0299/results.jsonl"
while kill -0 "$TRANSFER_EVAL_PID" 2>/dev/null || kill -0 "$JOINT_EVAL_PID" 2>/dev/null; do
  sleep 60
  echo "progress: transfer=$(count_results "$TRANSFER_RESULTS")/$EXPECTED_EPISODES_PER_CHECKPOINT, joint=$(count_results "$JOINT_RESULTS")/$EXPECTED_EPISODES_PER_CHECKPOINT episodes"
done

wait_pair \
  "$TRANSFER_EVAL_PID" "$JOINT_EVAL_PID" \
  "$TRANSFER_EVAL_LOG" "$JOINT_EVAL_LOG" "Full"

echo "Both fair evals complete:"
echo "  transfer step150: $TRANSFER_OUTPUT/step0149/summary.json"
echo "  joint    step300: $JOINT_OUTPUT/step0299/summary.json"
echo "  selected students: ${SELECTED_STUDENT_NAMES[*]}"
echo "  raw results: $TRANSFER_RESULTS and $JOINT_RESULTS"
