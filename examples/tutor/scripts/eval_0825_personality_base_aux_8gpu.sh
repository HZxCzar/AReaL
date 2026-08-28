#!/usr/bin/env bash
set -Eeuo pipefail

# One selected Qwen3-8B teacher variant against the ungated control and all six
# 0825 preference students. The trained variant carries the 0818 LoRA; the
# untrained variant is the base model. Gate, leak, and answer-judge requests always
# use the untrained base 8B. Runtime uses four teacher/student pairs: eight GPUs.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh [preflight|run|analyze]

Examples:
  # Read-only: resolve the selected teacher, construct config semantics, and select
  # the requested MATH rows. No server or output directory is created.
  bash examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh preflight

  # Full student rollout on exactly eight GPUs.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh run

  # Resume an interrupted run. Use the output path printed by the first command.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  EVAL_RUN_DIR=/path/to/the/existing/run \
    bash examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh run

  # Rebuild the readable summary without starting GPU processes.
  EVAL_RUN_DIR=/path/to/the/existing/run \
    bash examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh analyze

Defaults and useful overrides:
  GATE_DECISION_MODE=classification
  TEACHER_VARIANT=trained           # trained | untrained
  EVAL_STRATIFIED_MAX_SAMPLES=48    # 0 means the complete evaluation set
  EVAL_CONCURRENCY=16
  BASE_PORT=35000
  ADAPTER_PATH=/explicit/checkpoint/path  # trained only
  EVAL_RUN_DIR=/explicit/output/path

During a run, every completed episode is appended immediately to the cell's
results.jsonl. The combined live_summary.json and live_summary.tsv are refreshed
every five seconds. The gate is one seven-way A-G classification over the six
preferences plus NONE, using the untrained base Qwen3-8B auxiliary model.
EOF
}

MODE="${1:-preflight}"
case "$MODE" in
  -h|--help)
    usage
    exit 0
    ;;
  preflight|run|analyze) ;;
  *)
    printf 'Unknown mode: %s\n' "$MODE" >&2
    usage >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
GATE_DECISION_MODE="${GATE_DECISION_MODE:-classification}"
if [[ "$GATE_DECISION_MODE" != "classification" ]]; then
  printf 'The 0825 evaluator is configured for classification; got %q.\n' \
    "$GATE_DECISION_MODE" >&2
  exit 2
fi
TEACHER_VARIANT="${TEACHER_VARIANT:-trained}"
case "$TEACHER_VARIANT" in
  trained|untrained) ;;
  *)
    printf 'TEACHER_VARIANT must be trained or untrained; got %q.\n' \
      "$TEACHER_VARIANT" >&2
    exit 2
    ;;
esac
EVAL_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0825/pilot/eval-all-preferences-base-aux.yaml"
RUN_KIND="0825-personality-classification-base-aux-$TEACHER_VARIANT"
LAUNCHER_SCRIPT="examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh"
BASE_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0825/base/default.yaml"
EVALUATOR="$ROOT_DIR/examples/tutor/scripts/evaluate_api_teacher.py"
PREFLIGHT="$ROOT_DIR/examples/tutor/scripts/preflight_0825_personality_eval.py"
ANALYZER="$ROOT_DIR/examples/tutor/scripts/analyze_0825_personality_eval.py"

for required in "$PYTHON" "$EVAL_CONFIG" "$BASE_CONFIG" "$EVALUATOR" \
  "$PREFLIGHT" "$ANALYZER"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required path: %s\n' "$required" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  printf 'Project Python is not executable: %s\n' "$PYTHON" >&2
  exit 1
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
unset PYTHONHOME
export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export DO_NOT_TRACK=1
export INF_API_KEY=EMPTY
export DEEPSEEK_API_KEY=EMPTY
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
# Config interpolation needs values even during read-only preflight.
export TUTOR_QWEN3_8B_BASE_URL="${TUTOR_QWEN3_8B_BASE_URL:-http://127.0.0.1:1/v1}"
export TUTOR_QWEN3_1_7B_BASE_URL="${TUTOR_QWEN3_1_7B_BASE_URL:-http://127.0.0.1:2/v1}"

if [[ "$MODE" == "analyze" ]]; then
  if [[ -z "${EVAL_RUN_DIR:-}" || ! -d "$EVAL_RUN_DIR" ]]; then
    printf 'analyze requires an existing EVAL_RUN_DIR.\n' >&2
    exit 2
  fi
  exec "$PYTHON" -B "$ANALYZER" --run-dir "$EVAL_RUN_DIR"
fi

EVAL_STRATIFIED_MAX_SAMPLES="${EVAL_STRATIFIED_MAX_SAMPLES:-48}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
GENERALIZE_REPLAYS="${GENERALIZE_REPLAYS:-8}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-192}"
CALLER_MAX_CONCURRENT="${CALLER_MAX_CONCURRENT:-192}"
EPISODE_ERROR_RETRIES="${EPISODE_ERROR_RETRIES:-3}"
EPISODE_RETRY_BACKOFF_SECONDS="${EPISODE_RETRY_BACKOFF_SECONDS:-1}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-300}"
CELL_WALL_TIMEOUT_SECONDS="${CELL_WALL_TIMEOUT_SECONDS:-7200}"
CELL_PROCESS_RESTARTS="${CELL_PROCESS_RESTARTS:-3}"
BASE_PORT="${BASE_PORT:-35000}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
LIVE_SUMMARY_INTERVAL_SECONDS="${LIVE_SUMMARY_INTERVAL_SECONDS:-5}"
SKIP_LIVENESS="${SKIP_LIVENESS:-0}"
SAVE_TRACES="${SAVE_TRACES:-all}"

for integer_name in EVAL_STRATIFIED_MAX_SAMPLES EVAL_CONCURRENCY \
  GENERALIZE_REPLAYS SERVER_MAX_RUNNING_REQUESTS CALLER_MAX_CONCURRENT \
  EPISODE_ERROR_RETRIES EPISODE_RETRY_BACKOFF_SECONDS EPISODE_TIMEOUT_SECONDS \
  CELL_WALL_TIMEOUT_SECONDS CELL_PROCESS_RESTARTS BASE_PORT \
  SERVER_READY_TIMEOUT LIVE_SUMMARY_INTERVAL_SECONDS; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer; got %q.\n' \
      "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
if (( EVAL_STRATIFIED_MAX_SAMPLES < 0 || EVAL_CONCURRENCY < 1 || \
      GENERALIZE_REPLAYS != 8 || SERVER_MAX_RUNNING_REQUESTS < 1 || \
      CALLER_MAX_CONCURRENT < 1 || EPISODE_TIMEOUT_SECONDS < 1 || \
      CELL_WALL_TIMEOUT_SECONDS < 1 || LIVE_SUMMARY_INTERVAL_SECONDS < 1 )); then
  printf 'Samples must be non-negative; concurrency/timeouts must be positive; replays must remain 8.\n' >&2
  exit 2
fi
if (( EVAL_CONCURRENCY * GENERALIZE_REPLAYS > CALLER_MAX_CONCURRENT || \
      EVAL_CONCURRENCY * GENERALIZE_REPLAYS > SERVER_MAX_RUNNING_REQUESTS )); then
  printf 'EVAL_CONCURRENCY x 8 exceeds a caller/server request cap.\n' >&2
  exit 2
fi
case "$SAVE_TRACES" in all|errors|none) ;;
  *) printf 'SAVE_TRACES must be all, errors, or none.\n' >&2; exit 2 ;;
esac
case "$SKIP_LIVENESS" in 0|1) ;;
  *) printf 'SKIP_LIVENESS must be 0 or 1.\n' >&2; exit 2 ;;
esac

mapfile -t BASE_SETTINGS < <(
  "$PYTHON" -B - "$BASE_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
print(config.cluster.fileroot)
print(config.actor.path)
PY
)
if [[ "${#BASE_SETTINGS[@]}" != "2" ]]; then
  printf 'Could not resolve fileroot and teacher model path.\n' >&2
  exit 1
fi
TUTOR_FILEROOT="${TUTOR_FILEROOT:-${BASE_SETTINGS[0]}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${BASE_SETTINGS[1]}}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
TRIAL_NAME="${TRIAL_NAME:-20260824_192159_0818-personality-none-8gpu}"
TRIAL_DIR="${TRIAL_DIR:-$TUTOR_FILEROOT/checkpoints/root/tutor-math-baseline/$TRIAL_NAME}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

if [[ "$TEACHER_VARIANT" == "trained" ]]; then
  if [[ -n "$ADAPTER_PATH" ]]; then
    ADAPTER_PATH="$(readlink -f "$ADAPTER_PATH")"
  else
    ADAPTER_PATH="$($PYTHON -B - "$TRIAL_DIR" <<'PY'
import sys
from pathlib import Path
from examples.tutor.scripts.eval_checkpoints import discover

print(discover(Path(sys.argv[1]))[-1].path)
PY
)"
  fi
  printf '[resolve] teacher=trained trial=%s\n' "$TRIAL_NAME"
  printf '[resolve] adapter=%s\n' "$ADAPTER_PATH"
else
  if [[ -n "$ADAPTER_PATH" ]]; then
    printf 'TEACHER_VARIANT=untrained must not receive ADAPTER_PATH.\n' >&2
    exit 2
  fi
  printf '[resolve] teacher=untrained base=%s\n' "$TEACHER_MODEL_PATH"
fi

for model_path in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
  if [[ ! -f "$model_path/config.json" ]]; then
    printf 'Model path has no config.json: %s\n' "$model_path" >&2
    exit 1
  fi
done
if [[ "$TEACHER_MODEL" != "qwen3-8b" || "$STUDENT_MODEL" != "qwen3-1.7b" ]]; then
  printf 'This evaluator requires teacher=qwen3-8b and student=qwen3-1.7b.\n' >&2
  exit 2
fi
"$PYTHON" -B - "$TEACHER_MODEL_PATH/config.json" \
  "$STUDENT_MODEL_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    teacher = json.load(source)
with open(sys.argv[2], encoding="utf-8") as source:
    student = json.load(source)
expected = {
    "teacher Qwen3-8B": (teacher, 4096, 36),
    "student Qwen3-1.7B": (student, 2048, 28),
}
for label, (config, hidden_size, layers) in expected.items():
    actual = (
        config.get("model_type"),
        int(config.get("hidden_size", 0)),
        int(config.get("num_hidden_layers", 0)),
    )
    wanted = ("qwen3", hidden_size, layers)
    if actual != wanted:
        raise SystemExit(f"{label} model signature is {actual}, expected {wanted}")
print("[preflight] teacher/student model sizes: PASS")
PY

PREFLIGHT_COMMAND=(
  "$PYTHON" -B "$PREFLIGHT"
  --config "$EVAL_CONFIG"
  --teacher-model-path "$TEACHER_MODEL_PATH"
  --teacher-variant "$TEACHER_VARIANT"
  --stratified-max-samples "$EVAL_STRATIFIED_MAX_SAMPLES"
  --replays "$GENERALIZE_REPLAYS"
  --gate-decision-mode "$GATE_DECISION_MODE"
  --require-base-aux
)
if [[ "$TEACHER_VARIANT" == "trained" ]]; then
  PREFLIGHT_COMMAND+=(--adapter "$ADAPTER_PATH")
fi
PREFLIGHT_OUTPUT="$("${PREFLIGHT_COMMAND[@]}")"
printf '%s\n' "$PREFLIGHT_OUTPUT"
PREFLIGHT_JSON="$(printf '%s\n' "$PREFLIGHT_OUTPUT" | sed -n 's/^PREFLIGHT_JSON=//p' | tail -1)"
if [[ -z "$PREFLIGHT_JSON" ]]; then
  printf 'Preflight did not emit its machine-readable summary.\n' >&2
  exit 1
fi
EXPECTED_ROWS="$($PYTHON -B -c \
  'import json,sys; print(json.loads(sys.argv[1])["runtime_dataset_rows"])' \
  "$PREFLIGHT_JSON")"

STAMP="${EVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASET_RUN_KIND=subset
if (( EVAL_STRATIFIED_MAX_SAMPLES == 0 )); then
  DATASET_RUN_KIND=full
fi
RUN_DIR="${EVAL_RUN_DIR:-$TUTOR_FILEROOT/offline_eval/$RUN_KIND/$DATASET_RUN_KIND-$STAMP}"
if [[ "$MODE" == "preflight" ]]; then
  printf '[preflight] no files or GPU processes were created.\n'
  printf '[preflight] prospective output=%s\n' "$RUN_DIR"
  exit 0
fi

for command_name in curl setsid nvidia-smi timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf '%s is required for runtime mode.\n' "$command_name" >&2
    exit 1
  fi
done
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'Set CUDA_VISIBLE_DEVICES to exactly eight free GPU ids.\n' >&2
  exit 2
fi
IFS=',' read -r -a RAW_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
GPU_IDS=()
declare -A SEEN_GPU=()
for raw_gpu in "${RAW_GPU_IDS[@]}"; do
  gpu="${raw_gpu//[[:space:]]/}"
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPU[$gpu]:-}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES needs eight distinct integer ids; got %q.\n' \
      "$CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader >/dev/null
  SEEN_GPU["$gpu"]=1
  GPU_IDS+=("$gpu")
done
if [[ "${#GPU_IDS[@]}" != "8" ]]; then
  printf 'Expose exactly eight GPUs; got %d.\n' "${#GPU_IDS[@]}" >&2
  exit 2
fi
PAIR_COUNT=4
LAST_PORT=$((BASE_PORT + (PAIR_COUNT - 1) * 10 + 1))
if (( BASE_PORT < 1 || LAST_PORT > 65535 )); then
  printf 'BASE_PORT leaves the valid TCP range.\n' >&2
  exit 2
fi
PORTS=()
for ((pair_index = 0; pair_index < PAIR_COUNT; pair_index++)); do
  PORTS+=("$((BASE_PORT + pair_index * 10))")
  PORTS+=("$((BASE_PORT + pair_index * 10 + 1))")
done
"$PYTHON" -B - "${PORTS[@]}" <<'PY'
import socket
import sys

for raw in sys.argv[1:]:
    port = int(raw)
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"TCP port {port} is unavailable: {exc}") from exc
    finally:
        sock.close()
print("[preflight] runtime GPU ids and TCP ports: PASS")
PY

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/cells"
"$PYTHON" -B - "$RUN_DIR/manifest.json" "$EVAL_STRATIFIED_MAX_SAMPLES" \
  "$EVAL_CONCURRENCY" "$GENERALIZE_REPLAYS" "$SERVER_MAX_RUNNING_REQUESTS" \
  "$CALLER_MAX_CONCURRENT" "$SAVE_TRACES" "$PAIR_COUNT" "$BASE_PORT" \
  "$EPISODE_ERROR_RETRIES" "$EPISODE_RETRY_BACKOFF_SECONDS" \
  "$EPISODE_TIMEOUT_SECONDS" "$CELL_WALL_TIMEOUT_SECONDS" \
  "$CELL_PROCESS_RESTARTS" "$SKIP_LIVENESS" "$TEACHER_MODEL_PATH" \
  "$STUDENT_MODEL_PATH" "$PREFLIGHT_JSON" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

(
    output, stratified_rows, concurrency, replays, server_cap, caller_cap,
    save_traces, pair_count, base_port, error_retries, retry_backoff,
    episode_timeout, cell_timeout, cell_restarts, skip_liveness,
    teacher_model_path, student_model_path, preflight_raw,
) = sys.argv[1:]
preflight = json.loads(preflight_raw)
payload = {
    "phase": (
        "full"
        if preflight["dataset_selection"]["strategy"] == "full"
        else "subset"
    ),
    "eval_stratified_max_samples": int(stratified_rows),
    "eval_concurrency_per_pair": int(concurrency),
    "student_generalize_replays": int(replays),
    "server_max_running_requests": int(server_cap),
    "caller_max_concurrent": int(caller_cap),
    "save_traces": save_traces,
    "pair_count": int(pair_count),
    "base_port": int(base_port),
    "teacher": preflight["teacher"],
    "students": preflight["students"],
    "adapter": preflight["adapter"],
    "completed_steps": preflight["completed_steps"],
    "model_paths": {
        "teacher": str(Path(teacher_model_path).resolve()),
        "student": str(Path(student_model_path).resolve()),
    },
    "dataset": {
        "full_rows": preflight["full_dataset_rows"],
        "phase_rows": preflight["runtime_dataset_rows"],
        "sha256": preflight["runtime_dataset_sha256"],
        "seed": preflight["seed"],
        "selection": preflight["dataset_selection"],
    },
    "semantics": preflight["semantics"],
    "input_hashes": {
        "config": preflight["config_sha256"],
        "resolved_config": preflight["resolved_config_sha256"],
        "prompts": preflight["prompts_sha256"],
        "complaints": preflight["complaints_sha256"],
    },
    "reliability": {
        "episode_error_retries": int(error_retries),
        "episode_retry_backoff_seconds": int(retry_backoff),
        "episode_timeout_seconds": int(episode_timeout),
        "cell_wall_timeout_seconds": int(cell_timeout),
        "cell_process_restarts": int(cell_restarts),
        "skip_liveness": bool(int(skip_liveness)),
    },
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous.pop("created_at", None)
    if previous != payload:
        raise SystemExit(
            "Existing manifest differs. Use a new EVAL_RUN_DIR when changing "
            "samples, ports, models, or checkpoint."
        )
else:
    payload["created_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

process_running() {
  kill -0 "$1" 2>/dev/null
}

wait_ready() {
  local pid=$1 url=$2 log_path=$3 label=$4 deadline
  deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! process_running "$pid"; then
      printf '%s exited during startup. Tail of %s:\n' "$label" "$log_path" >&2
      tail -80 "$log_path" >&2 || true
      return 1
    fi
    if curl --silent --fail --max-time 2 "$url/models" >/dev/null 2>&1; then
      printf '[server] %s ready at %s\n' "$label" "$url"
      return 0
    fi
    sleep 2
  done
  printf '%s did not become ready. Tail of %s:\n' "$label" "$log_path" >&2
  tail -80 "$log_path" >&2 || true
  return 1
}

validate_cell() {
  local output=$1 student=$2 preference=$3 adapter=$4
  "$PYTHON" -B - "$output" "$student" "$preference" "$TEACHER_VARIANT" \
    "$adapter" "$GATE_DECISION_MODE" "$EXPECTED_ROWS" \
    "$GENERALIZE_REPLAYS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
student, preference = sys.argv[2:4]
teacher_variant = sys.argv[4]
adapter = sys.argv[5]
gate_decision_mode = sys.argv[6]
expected, replays = map(int, sys.argv[7:9])
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))["signature"]
if set(summary["modes"]) != {"presolve_on"}:
    raise SystemExit(f"{output}: expected only presolve_on")
mode = summary["modes"]["presolve_on"]
pending = int((summary.get("pending_backfill") or {}).get("count", 0) or 0)
checks = {
    "dataset_rows": int(summary["dataset_rows"]) == expected,
    "expected_attempts": int(mode["expected_attempts"]) == expected,
    "recorded_attempts": int(mode["recorded_attempts"]) == expected,
    "accounted_attempts": int(mode["completed_attempts"]) + int(mode["error_count"]) == expected,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"{output}: coverage checks failed: {failed}")
students = run_config["students"]
if len(students) != 1 or students[0]["name"] != student:
    raise SystemExit(f"{output}: evaluator was not singleton for {student}")
effective_preference = str(students[0].get("personality") or "none")
if effective_preference != preference:
    raise SystemExit(
        f"{output}: preference={effective_preference}, expected={preference}"
    )
semantics = run_config["test_semantics"]
personality = semantics.get("personality") or {}
if run_config["presolve"]["verify"]:
    raise SystemExit(f"{output}: teacher pre-solve verification unexpectedly enabled")
if semantics["leak_handling_mode"] != "reward_only" or semantics["format_handling_mode"] != "continue":
    raise SystemExit(f"{output}: deployment semantics drifted")
if personality.get("gate_prompt_version") != "v2":
    raise SystemExit(f"{output}: v2 gate prompt was not forwarded")
if personality.get("gate_decision_mode") != gate_decision_mode:
    raise SystemExit(
        f"{output}: gate mode={personality.get('gate_decision_mode')}, "
        f"expected={gate_decision_mode}"
    )
if int(semantics["student_generalize_replays"]) != replays:
    raise SystemExit(f"{output}: replay count drifted")
if set(semantics["generalization_levels"]) != {"original", "original_preleak"}:
    raise SystemExit(f"{output}: generalization levels drifted")
teacher_extra_body = (
    run_config["teacher"]["request_params"].get("extra_body") or {}
)
teacher_lora = str(teacher_extra_body.get("lora_path") or "")
if teacher_variant == "trained":
    if not teacher_lora or str(Path(teacher_lora).resolve()) != str(
        Path(adapter).resolve()
    ):
        raise SystemExit(f"{output}: teacher uses wrong LoRA: {teacher_lora}")
elif teacher_lora:
    raise SystemExit(f"{output}: untrained teacher unexpectedly uses {teacher_lora}")
auxiliary = run_config["auxiliary"]
if auxiliary["source_mode"] != "api" or auxiliary["effective_mode"] != "api":
    raise SystemExit(f"{output}: auxiliary is not the base-model API path")
aux_extra_body = auxiliary["request_params"].get("extra_body") or {}
if aux_extra_body.get("lora_path"):
    raise SystemExit(
        f"{output}: auxiliary unexpectedly carries LoRA: {aux_extra_body['lora_path']}"
    )
gate = mode.get("personality_gate") or {}
if preference == "none":
    if int(gate.get("active_episode_count", 0) or 0):
        raise SystemExit(f"{output}: none unexpectedly activated the gate")
elif int(gate.get("active_episode_count", 0) or 0) + pending < expected:
    raise SystemExit(f"{output}: demanding episodes are missing gate metrics")
if (
    preference == "feedback"
    and int(gate.get("turn1_sampled_episode_count", 0) or 0) != 0
):
    raise SystemExit(
        f"{output}: {gate_decision_mode} feedback unexpectedly gated turn 1"
    )
print(f"[cell-ok] {output}: rows={expected}, pending={pending}")
PY
}

PREFERENCES=(none feedback hinting instructing explaining modeling questioning)
STUDENTS=(
  qwen3-1.7b-text-original
  qwen3-1.7b-text-original-feedback
  qwen3-1.7b-text-original-hinting
  qwen3-1.7b-text-original-instructing
  qwen3-1.7b-text-original-explaining
  qwen3-1.7b-text-original-modeling
  qwen3-1.7b-text-original-questioning
)
TEACHER_NAME="$TEACHER_VARIANT"

run_pair() (
  set -Eeuo pipefail
  local pair_index=$1 teacher_gpu=$2 student_gpu=$3
  local teacher_port=$((BASE_PORT + pair_index * 10))
  local student_port=$((teacher_port + 1))
  local teacher_url="http://127.0.0.1:${teacher_port}/v1"
  local student_url="http://127.0.0.1:${student_port}/v1"
  local pair_dir="$RUN_DIR/pair-${pair_index}"
  local teacher_log="$RUN_DIR/logs/pair-${pair_index}-teacher.log"
  local student_log="$RUN_DIR/logs/pair-${pair_index}-student.log"
  local teacher_pid="" student_pid="" current_cell_pid=""

  pair_cleanup() {
    local status=$? pid
    trap - EXIT INT TERM
    for pid in "$current_cell_pid" "$teacher_pid" "$student_pid"; do
      [[ -n "$pid" ]] || continue
      if kill -0 -- "-$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      elif kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    for pid in "$current_cell_pid" "$teacher_pid" "$student_pid"; do
      [[ -n "$pid" ]] || continue
      wait "$pid" 2>/dev/null || true
    done
    exit "$status"
  }
  trap pair_cleanup EXIT INT TERM

  local adapter_alias=""
  if [[ "$TEACHER_VARIANT" == "trained" ]]; then
    mkdir -p "$pair_dir/served_adapters"
    adapter_alias="$pair_dir/served_adapters/$TEACHER_NAME"
    if [[ -L "$adapter_alias" ]]; then
      if [[ "$(readlink -f "$adapter_alias")" != "$ADAPTER_PATH" ]]; then
        printf 'Existing adapter alias points elsewhere: %s\n' "$adapter_alias" >&2
        return 1
      fi
    elif [[ -e "$adapter_alias" ]]; then
      printf 'Refusing non-symlink adapter alias: %s\n' "$adapter_alias" >&2
      return 1
    else
      ln -s "$ADAPTER_PATH" "$adapter_alias"
    fi
  fi

  local -a teacher_server_command
  teacher_server_command=(
    "$PYTHON" -m sglang.launch_server
    --model-path "$TEACHER_MODEL_PATH" --served-model-name "$TEACHER_MODEL" \
    --host 127.0.0.1 --port "$teacher_port" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS"
  )
  if [[ "$TEACHER_VARIANT" == "trained" ]]; then
    teacher_server_command+=(
      --enable-lora --lora-paths "$adapter_alias"
      --max-loras-per-batch 1 --max-loaded-loras 1
    )
  fi
  CUDA_VISIBLE_DEVICES="$teacher_gpu" setsid "${teacher_server_command[@]}" \
    >"$teacher_log" 2>&1 &
  teacher_pid=$!

  CUDA_VISIBLE_DEVICES="$student_gpu" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$STUDENT_MODEL_PATH" --served-model-name "$STUDENT_MODEL" \
    --host 127.0.0.1 --port "$student_port" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" \
    >"$student_log" 2>&1 &
  student_pid=$!

  wait_ready "$teacher_pid" "$teacher_url" "$teacher_log" \
    "pair $pair_index teacher (GPU $teacher_gpu)"
  wait_ready "$student_pid" "$student_url" "$student_log" \
    "pair $pair_index student (GPU $student_gpu)"

  if [[ "$SKIP_LIVENESS" == "0" ]]; then
    "$PYTHON" -B - "$teacher_url" "$TEACHER_MODEL" "$TEACHER_VARIANT" \
      "$adapter_alias" <<'PY'
import sys
from examples.tutor.scripts.eval_checkpoints import probe

base_url, model, teacher_variant, adapter = sys.argv[1:]
base = probe(base_url, model, "EMPTY", None, 300.0)
if not base:
    raise SystemExit("base teacher returned an empty liveness response")
if teacher_variant == "trained":
    try:
        probe(base_url, model, "EMPTY", "/nonexistent/tutor-adapter", 300.0)
    except Exception:
        pass
    else:
        raise SystemExit("endpoint accepted a nonexistent lora_path")
    answer = probe(base_url, model, "EMPTY", adapter, 300.0)
    if not answer or answer == base:
        raise SystemExit(f"adapter is not observably active: {adapter}")
    print(f"[liveness] trained adapter active: {adapter}")
else:
    print("[liveness] untrained base teacher active")
PY
  fi

  run_eval_cell() {
    local student=$1 preference=$2 output=$3 log=$4
    local request_params status cell_try max_cell_tries
    local -a command attempt_command
    request_params="$(
      "$PYTHON" -B -c \
        'import json,sys; variant,adapter=sys.argv[1:]; body={"chat_template_kwargs":{"enable_thinking":False}}; body.update({"lora_path":adapter} if variant == "trained" else {}); print(json.dumps({"seed":42,"extra_body":body}))' \
        "$TEACHER_VARIANT" "$adapter_alias"
    )"
    mkdir -p "$output"
    command=(
      "$PYTHON" -B "$EVALUATOR"
      --config "$EVAL_CONFIG"
      --teacher-base-url "$teacher_url"
      --teacher-model "$TEACHER_MODEL"
      --api-key EMPTY
      --teacher-request-params "$request_params"
      --teacher-presolve config
      --student-generalization config
      --student-name "$student"
      --attempts 0
      --stratified-max-samples "$EVAL_STRATIFIED_MAX_SAMPLES"
      --concurrency "$EVAL_CONCURRENCY"
      --episode-error-retries "$EPISODE_ERROR_RETRIES"
      --episode-error-retry-backoff-seconds "$EPISODE_RETRY_BACKOFF_SECONDS"
      --episode-timeout-seconds "$EPISODE_TIMEOUT_SECONDS"
      --retry-diagnostic-failures
      --save-traces "$SAVE_TRACES"
      --log-every 1
      --output-dir "$output"
      "auxiliary_model.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
    )
    printf '[cell] pair=%s preference=%s student=%s\n' \
      "$pair_index" "$preference" "$student"
    touch "$log"
    max_cell_tries=$((CELL_PROCESS_RESTARTS + 1))
    for ((cell_try = 1; cell_try <= max_cell_tries; cell_try++)); do
      attempt_command=("${command[@]}")
      if [[ -f "$output/run_config.json" ]]; then
        attempt_command+=(--resume)
      fi
      printf '[cell-process] try=%s/%s wall_timeout=%ss\n' \
        "$cell_try" "$max_cell_tries" "$CELL_WALL_TIMEOUT_SECONDS" >>"$log"
      set +e
      setsid env TUTOR_QWEN3_8B_BASE_URL="$teacher_url" \
        TUTOR_QWEN3_1_7B_BASE_URL="$student_url" \
        timeout --signal=TERM --kill-after=30s \
          "${CELL_WALL_TIMEOUT_SECONDS}s" "${attempt_command[@]}" \
          >>"$log" 2>&1 &
      current_cell_pid=$!
      wait "$current_cell_pid"
      status=$?
      current_cell_pid=""
      set -e
      if (( status == 0 )); then
        break
      fi
      if (( cell_try == max_cell_tries )); then
        printf 'Cell failed after %s tries; tail of %s:\n' \
          "$max_cell_tries" "$log" >&2
        tail -100 "$log" >&2 || true
        return 1
      fi
      printf '[cell-process-retry] status=%s; resuming\n' "$status" >>"$log"
    done
    validate_cell "$output" "$student" "$preference" "$adapter_alias"
  }

  local cell_index student preference output log
  for cell_index in "${!STUDENTS[@]}"; do
    if (( cell_index % PAIR_COUNT != pair_index )); then
      continue
    fi
    student="${STUDENTS[$cell_index]}"
    preference="${PREFERENCES[$cell_index]}"
    output="$RUN_DIR/cells/$TEACHER_NAME/$student"
    log="$RUN_DIR/logs/$TEACHER_NAME--$preference.log"
    run_eval_cell "$student" "$preference" "$output" "$log"
  done

)

WORKER_PIDS=()
WATCHER_PID=""
cleanup_all() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "$WATCHER_PID" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
    kill -TERM "$WATCHER_PID" 2>/dev/null || true
  fi
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$WATCHER_PID" ]]; then
    wait "$WATCHER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_all EXIT INT TERM

"$PYTHON" -B "$ANALYZER" --run-dir "$RUN_DIR" --watch \
  --interval-seconds "$LIVE_SUMMARY_INTERVAL_SECONDS" \
  >"$RUN_DIR/logs/live-summary.log" 2>&1 &
WATCHER_PID=$!

printf '[run] teacher=%s gate_mode=%s pairs=4 rows_per_cell=%s cells=7 output=%s\n' \
  "$TEACHER_VARIANT" "$GATE_DECISION_MODE" "$EXPECTED_ROWS" "$RUN_DIR"
printf '[live] %s\n' "$RUN_DIR/live_summary.tsv"
for ((pair_index = 0; pair_index < PAIR_COUNT; pair_index++)); do
  run_pair "$pair_index" "${GPU_IDS[$((pair_index * 2))]}" \
    "${GPU_IDS[$((pair_index * 2 + 1))]}" &
  WORKER_PIDS+=("$!")
done

ACTIVE_WORKER_PIDS=("${WORKER_PIDS[@]}")
worker_failed=0
while (( ${#ACTIVE_WORKER_PIDS[@]} > 0 )); do
  finished_pid=""
  set +e
  wait -n -p finished_pid "${ACTIVE_WORKER_PIDS[@]}"
  worker_status=$?
  set -e
  if [[ -z "$finished_pid" ]]; then
    worker_failed=1
    break
  fi
  NEXT_WORKER_PIDS=()
  for worker_pid in "${ACTIVE_WORKER_PIDS[@]}"; do
    if [[ "$worker_pid" != "$finished_pid" ]]; then
      NEXT_WORKER_PIDS+=("$worker_pid")
    fi
  done
  ACTIVE_WORKER_PIDS=("${NEXT_WORKER_PIDS[@]}")
  if (( worker_status != 0 )); then
    worker_failed=1
    break
  fi
done
if (( worker_failed )); then
  printf 'A server pair or cell failed; stopping other pairs. Logs: %s/logs\n' \
    "$RUN_DIR" >&2
  exit 1
fi
WORKER_PIDS=()

if [[ -n "$WATCHER_PID" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
  kill -TERM "$WATCHER_PID" 2>/dev/null || true
fi
wait "$WATCHER_PID" 2>/dev/null || true
WATCHER_PID=""
"$PYTHON" -B "$ANALYZER" --run-dir "$RUN_DIR"

PENDING_COUNT="$($PYTHON -B - "$RUN_DIR/cells" <<'PY'
import sys
from pathlib import Path

count = 0
for path in Path(sys.argv[1]).glob("*/*/pending_backfill.jsonl"):
    count += sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
print(count)
PY
)"
trap - EXIT INT TERM
if (( PENDING_COUNT > 0 )); then
  printf '[done-with-pending] %s episodes need backfill. Rerun with:\n' \
    "$PENDING_COUNT"
  printf 'CUDA_VISIBLE_DEVICES=%s TEACHER_VARIANT=%q EVAL_STRATIFIED_MAX_SAMPLES=%q EVAL_RUN_DIR=%q bash %q run\n' \
    "$CUDA_VISIBLE_DEVICES" "$TEACHER_VARIANT" \
    "$EVAL_STRATIFIED_MAX_SAMPLES" "$RUN_DIR" "$LAUNCHER_SCRIPT"
  exit 0
fi
printf '[done] results and live/final summary: %s\n' "$RUN_DIR"
