#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="${EVAL_CONFIG:-$ROOT_DIR/examples/tutor/configs/math/0810/4gpu/two-student-text-code-eval.yaml}"
TRANSFER_CHECKPOINT="${TRANSFER_CHECKPOINT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/checkpoints/root/tutor-math-baseline/20260813_053934_0810fc-leak-local/default/epoch3epochstep8globalstep149}"
JOINT_CHECKPOINT="${JOINT_CHECKPOINT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/checkpoints/root/tutor-math-baseline/20260815_204624_0810fc-two-student-local-format-terminate/default/epoch6epochstep17globalstep299}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_BASE_URL="http://127.0.0.1:${TEACHER_PORT}/v1"
OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/output/tutor/offline_eval/fair-transfer-step150-vs-joint-step300-text-code}"
# 0 means the full test split; set EVAL_LIMIT=N for a quick subset.
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.75}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-40960}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-16}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SAVE_TRACES="${SAVE_TRACES:-all}"

for required in "$PYTHON" "$CONFIG" \
  "$TRANSFER_CHECKPOINT/adapter_model.safetensors" \
  "$TRANSFER_CHECKPOINT/adapter_config.json" \
  "$JOINT_CHECKPOINT/adapter_model.safetensors" \
  "$JOINT_CHECKPOINT/adapter_config.json" \
  "$BASE_MODEL_PATH/config.json"; do
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

for variable in INF_API_KEY TUTOR_QWEN3_1_7B_BASE_URL TUTOR_QWEN3_8B_BASE_URL; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Missing $variable; export it or define it in $ROOT_DIR/.env" >&2
    exit 1
  fi
done

if ! [[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]]; then
  echo "EVAL_LIMIT must be a non-negative integer, got: $EVAL_LIMIT" >&2
  exit 1
fi
if ! [[ "$EVAL_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_CONCURRENCY must be a positive integer, got: $EVAL_CONCURRENCY" >&2
  exit 1
fi
if ! [[ "$TEACHER_PORT" =~ ^[1-9][0-9]*$ ]] || (( TEACHER_PORT > 65535 )); then
  echo "TEACHER_PORT must be in 1..65535, got: $TEACHER_PORT" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
  echo "No CUDA GPU is visible to $PYTHON (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
SERVER_LOG="$OUTPUT_ROOT/sglang-fair-comparison-$(date -u +%Y%m%dT%H%M%SZ).log"
SERVER_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if curl --silent --show-error --fail --max-time 2 "$TEACHER_BASE_URL/models" >/dev/null 2>&1; then
  echo "Port $TEACHER_PORT already has an OpenAI-compatible server; choose another TEACHER_PORT." >&2
  exit 1
fi

echo "Starting Qwen3-8B with transfer-step150 and joint-step300 LoRAs on GPU $CUDA_VISIBLE_DEVICES (log: $SERVER_LOG)"
"$PYTHON" -m sglang.launch_server \
  --model-path "$BASE_MODEL_PATH" \
  --served-model-name "$TEACHER_MODEL" \
  --host 127.0.0.1 \
  --port "$TEACHER_PORT" \
  --tp-size 1 \
  --context-length "$SGLANG_CONTEXT_LENGTH" \
  --mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
  --max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS" \
  --enable-lora \
  --lora-paths "$TRANSFER_CHECKPOINT" "$JOINT_CHECKPOINT" \
  --max-loaded-loras 2 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "SGLang exited before becoming ready. Last log lines:" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if curl --silent --show-error --fail --max-time 3 "$TEACHER_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl --silent --show-error --fail --max-time 3 "$TEACHER_BASE_URL/models" >/dev/null 2>&1; then
  echo "SGLang was not ready after ${SERVER_READY_TIMEOUT}s. Last log lines:" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "Running the same text+code eval for both checkpoints (student and auxiliary models use configured endpoints)."
eval_args=(
  "$PYTHON" examples/tutor/scripts/eval_checkpoints.py
  --checkpoint "$TRANSFER_CHECKPOINT"
  --checkpoint "$JOINT_CHECKPOINT"
  --base-url "$TEACHER_BASE_URL"
  --model "$TEACHER_MODEL"
  --api-key EMPTY
  --config "$CONFIG"
  --output-root "$OUTPUT_ROOT"
  --concurrency "$EVAL_CONCURRENCY"
  --resume
)

# eval_checkpoints.py's --max-samples spelling is stale; evaluate_api_teacher.py
# calls this option --limit. Forward the correct option after argparse's separator.
eval_args+=(-- --limit "$EVAL_LIMIT" --save-traces "$SAVE_TRACES")
"${eval_args[@]}"

echo "Eval complete:"
echo "  transfer step150: $OUTPUT_ROOT/step0149/summary.json"
echo "  joint    step300: $OUTPUT_ROOT/step0299/summary.json"
