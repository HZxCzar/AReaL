#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
IFS=',' read -r -a GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
if [[ "${#GPU_IDS[@]}" != "8" ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly 8 GPU ids." >&2
  exit 2
fi

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy PYTHONHOME
export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export INF_API_KEY=EMPTY
export DEEPSEEK_API_KEY=EMPTY
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"

PYTHON="$ROOT_DIR/.venv/bin/python"
RUN_NAME=20260901_182029_0901-preference-v3-reward-v3-all-id-8gpu
CHECKPOINT=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/checkpoints/root/tutor-math-baseline/${RUN_NAME}/default/epoch21epochstep12globalstep999
TEACHER_MODEL_PATH=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
STUDENT_MODEL_PATH=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SOURCE_EVAL_DIR=/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/offline_eval/0901-preference-v3-step1000-explain100-full-matrix/20260904T154202Z
CONFIG="$ROOT_DIR/examples/tutor/configs/math/0901/pilot/eval-all-preferences-explain100.yaml"
OUTPUT_DIR="${MATCHED_TURN1_OUTPUT_DIR:-$ROOT_DIR/output/gradient_cosine/${RUN_NAME}-matched-turn1}"
PROBLEMS="${MATCHED_TURN1_PROBLEMS:-16}"
BASE_PORT="${MATCHED_TURN1_BASE_PORT:-39000}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-192}"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/adapters"

ADAPTER_ALIAS="$OUTPUT_DIR/adapters/all-id"
if [[ -L "$ADAPTER_ALIAS" ]]; then
  if [[ "$(readlink -f "$ADAPTER_ALIAS")" != "$(readlink -f "$CHECKPOINT")" ]]; then
    echo "Existing adapter alias points to a different checkpoint: $ADAPTER_ALIAS" >&2
    exit 1
  fi
elif [[ -e "$ADAPTER_ALIAS" ]]; then
  echo "Adapter alias path exists and is not a symlink: $ADAPTER_ALIAS" >&2
  exit 1
else
  ln -s "$CHECKPOINT" "$ADAPTER_ALIAS"
fi

SERVER_PIDS=()
stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}
trap stop_servers EXIT INT TERM

wait_ready() {
  local pid=$1 url=$2 log=$3 label=$4 deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited during startup; tail of $log:" >&2
      tail -80 "$log" >&2 || true
      return 1
    fi
    if curl --silent --fail --max-time 2 "$url/models" >/dev/null 2>&1; then
      echo "[server] $label ready at $url"
      return 0
    fi
    sleep 2
  done
  echo "$label did not become ready; tail of $log:" >&2
  tail -80 "$log" >&2 || true
  return 1
}

if [[ ! -f "$OUTPUT_DIR/matched_manifest.json" ]]; then
  TEACHER_URLS=()
  STUDENT_URLS=()
  for pair in 0 1 2 3; do
    teacher_port=$((BASE_PORT + pair * 10))
    student_port=$((teacher_port + 1))
    teacher_url="http://127.0.0.1:${teacher_port}/v1"
    student_url="http://127.0.0.1:${student_port}/v1"
    teacher_log="$OUTPUT_DIR/logs/pair-${pair}-teacher.log"
    student_log="$OUTPUT_DIR/logs/pair-${pair}-student.log"
    teacher_gpu="${GPU_IDS[$((pair * 2))]}"
    student_gpu="${GPU_IDS[$((pair * 2 + 1))]}"

    CUDA_VISIBLE_DEVICES="$teacher_gpu" setsid "$PYTHON" -m sglang.launch_server \
      --model-path "$TEACHER_MODEL_PATH" \
      --served-model-name qwen3-8b \
      --host 127.0.0.1 \
      --port "$teacher_port" \
      --tp-size 1 \
      --context-length 40960 \
      --mem-fraction-static 0.80 \
      --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" \
      --enable-lora \
      --lora-paths "$ADAPTER_ALIAS" \
      --max-loras-per-batch 1 \
      --max-loaded-loras 1 >"$teacher_log" 2>&1 &
    teacher_pid=$!
    SERVER_PIDS+=("$teacher_pid")

    CUDA_VISIBLE_DEVICES="$student_gpu" setsid "$PYTHON" -m sglang.launch_server \
      --model-path "$STUDENT_MODEL_PATH" \
      --served-model-name qwen3-1.7b \
      --host 127.0.0.1 \
      --port "$student_port" \
      --tp-size 1 \
      --context-length 40960 \
      --mem-fraction-static 0.80 \
      --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" >"$student_log" 2>&1 &
    student_pid=$!
    SERVER_PIDS+=("$student_pid")

    TEACHER_URLS+=("$teacher_url")
    STUDENT_URLS+=("$student_url")
  done

  for pair in 0 1 2 3; do
    teacher_index=$((pair * 2))
    student_index=$((teacher_index + 1))
    wait_ready "${SERVER_PIDS[$teacher_index]}" "${TEACHER_URLS[$pair]}" \
      "$OUTPUT_DIR/logs/pair-${pair}-teacher.log" "pair $pair teacher"
    wait_ready "${SERVER_PIDS[$student_index]}" "${STUDENT_URLS[$pair]}" \
      "$OUTPUT_DIR/logs/pair-${pair}-student.log" "pair $pair student"
  done

  TEACHER_URL_CSV="$(IFS=,; echo "${TEACHER_URLS[*]}")"
  STUDENT_URL_CSV="$(IFS=,; echo "${STUDENT_URLS[*]}")"
  "$PYTHON" -B examples/tutor/scripts/collect_matched_turn1_gradients.py \
    --config "$CONFIG" \
    --source-eval-dir "$SOURCE_EVAL_DIR" \
    --adapter "$ADAPTER_ALIAS" \
    --teacher-urls "$TEACHER_URL_CSV" \
    --student-urls "$STUDENT_URL_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --problems "$PROBLEMS" \
    --candidates 8 \
    --branch-concurrency 16

  stop_servers
else
  echo "[resume] reusing completed matched rollout: $OUTPUT_DIR/matched_manifest.json"
fi

"$ROOT_DIR/.venv/bin/torchrun" \
  --standalone \
  --nproc_per_node=8 \
  examples/tutor/scripts/analyze_environment_gradient_cosine.py \
  --checkpoint "$CHECKPOINT" \
  --input-manifest "$OUTPUT_DIR/matched_manifest.json" \
  --output-dir "$OUTPUT_DIR/gradient"

echo "[done] $OUTPUT_DIR/gradient/result.md"
