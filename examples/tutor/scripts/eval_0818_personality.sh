#!/usr/bin/env bash
set -Eeuo pipefail

# Offline 5-teacher x 8-personality evaluation. This is deliberately standalone:
# it starts only inference servers and evaluate_api_teacher.py processes, and does
# not enter the training launcher or scheduler.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0818_personality.sh \
    [preflight|quick|full|analyze]

Examples:
  # Read-only: checkpoints, config semantics, prompts, and seeded test split.
  bash examples/tutor/scripts/eval_0818_personality.sh preflight

  # 24 seeded rows in every one of the 5 x 8 cells. Eight GPUs form four
  # independent teacher/student server pairs.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0818_personality.sh quick

  # Every filtered test row (currently 528) in every cell.
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0818_personality.sh full

Phases:
  preflight  No output directory and no GPU process.
  quick      24 seeded test rows per cell; saves all traces.
  full       Full filtered test split; saves error traces and all Gate metrics.
  analyze    Rebuild matrix_summary.json for EVAL_RUN_DIR.

Common overrides:
  EVAL_RUN_DIR, BASE_PORT, EVAL_MAX_SAMPLES, EVAL_CONCURRENCY,
  SERVER_MAX_RUNNING_REQUESTS, CALLER_MAX_CONCURRENT, SAVE_TRACES,
  EVAL_TEACHERS (CSV or all), EVAL_PERSONALITIES (CSV, id, ood, or all),
  ADAPTER_NONE, ADAPTER_SURFACE, ADAPTER_EXECUTIVE,
  ADAPTER_CONTRASTING_CASES, ADAPTER_FULL.

The five TRIAL_* variables may override the pinned active run directories. A fresh
run independently selects each trial's highest complete LoRA checkpoint. Reusing an
EVAL_RUN_DIR is strict: phase, sample count, GPU-pair count, ports, and adapters must
all remain identical.
EOF
}

PHASE="${1:-preflight}"
case "$PHASE" in
  -h|--help)
    usage
    exit 0
    ;;
  preflight|quick|full|analyze) ;;
  *)
    printf 'Unknown phase: %s\n' "$PHASE" >&2
    usage >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
EVAL_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/pilot/eval-all-personalities.yaml"
BASE_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/base/default.yaml"
EVALUATOR="$ROOT_DIR/examples/tutor/scripts/evaluate_api_teacher.py"
PREFLIGHT="$ROOT_DIR/examples/tutor/scripts/preflight_0818_personality_eval.py"
ANALYZER="$ROOT_DIR/examples/tutor/scripts/analyze_0818_personality_eval.py"

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
# Config loading needs the interpolations, but preflight performs no request.
export TUTOR_QWEN3_8B_BASE_URL="${TUTOR_QWEN3_8B_BASE_URL:-http://127.0.0.1:1/v1}"
export TUTOR_QWEN3_1_7B_BASE_URL="${TUTOR_QWEN3_1_7B_BASE_URL:-http://127.0.0.1:2/v1}"

if [[ "$PHASE" == "analyze" ]]; then
  if [[ -z "${EVAL_RUN_DIR:-}" || ! -d "$EVAL_RUN_DIR" ]]; then
    printf 'analyze requires an existing EVAL_RUN_DIR.\n' >&2
    exit 2
  fi
  exec "$PYTHON" -B "$ANALYZER" --run-dir "$EVAL_RUN_DIR"
fi

case "$PHASE" in
  preflight|quick) DEFAULT_MAX_SAMPLES=24 ;;
  full) DEFAULT_MAX_SAMPLES=0 ;;
esac
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
GENERALIZE_REPLAYS="${GENERALIZE_REPLAYS:-8}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-192}"
CALLER_MAX_CONCURRENT="${CALLER_MAX_CONCURRENT:-192}"
EPISODE_ERROR_RETRIES="${EPISODE_ERROR_RETRIES:-3}"
EPISODE_RETRY_BACKOFF_SECONDS="${EPISODE_RETRY_BACKOFF_SECONDS:-1}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-300}"
CELL_WALL_TIMEOUT_SECONDS="${CELL_WALL_TIMEOUT_SECONDS:-3600}"
CELL_PROCESS_RESTARTS="${CELL_PROCESS_RESTARTS:-3}"
BASE_PORT="${BASE_PORT:-33000}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SKIP_LIVENESS="${SKIP_LIVENESS:-0}"
if [[ -n "${SAVE_TRACES:-}" ]]; then
  SAVE_TRACES="$SAVE_TRACES"
elif [[ "$PHASE" == "quick" ]]; then
  SAVE_TRACES=all
else
  SAVE_TRACES=errors
fi

for integer_name in EVAL_MAX_SAMPLES EVAL_CONCURRENCY GENERALIZE_REPLAYS \
  SERVER_MAX_RUNNING_REQUESTS CALLER_MAX_CONCURRENT \
  EPISODE_ERROR_RETRIES EPISODE_RETRY_BACKOFF_SECONDS \
  EPISODE_TIMEOUT_SECONDS CELL_WALL_TIMEOUT_SECONDS CELL_PROCESS_RESTARTS \
  BASE_PORT SERVER_READY_TIMEOUT; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer; got %q.\n' \
      "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
if (( EVAL_CONCURRENCY < 1 || GENERALIZE_REPLAYS < 1 || \
      SERVER_MAX_RUNNING_REQUESTS < 1 || CALLER_MAX_CONCURRENT < 1 || \
      EPISODE_TIMEOUT_SECONDS < 1 || CELL_WALL_TIMEOUT_SECONDS < 1 )); then
  printf 'Concurrency, replay, request, and timeout limits must be positive.\n' >&2
  exit 2
fi
if (( GENERALIZE_REPLAYS != 8 )); then
  printf 'GENERALIZE_REPLAYS must remain 8; got %s.\n' "$GENERALIZE_REPLAYS" >&2
  exit 2
fi
if (( EVAL_CONCURRENCY * GENERALIZE_REPLAYS > CALLER_MAX_CONCURRENT || \
      EVAL_CONCURRENCY * GENERALIZE_REPLAYS > SERVER_MAX_RUNNING_REQUESTS )); then
  printf 'EVAL_CONCURRENCY x GENERALIZE_REPLAYS exceeds a caller/server cap.\n' >&2
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
CHECKPOINT_ROOT="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline"

ALL_TEACHERS=(none surface executive contrasting_cases full)
ALL_PERSONALITIES=(
  none executive surface contrasting_cases
  analogy error_focused example_first rule_first
)
declare -A STUDENT_BY_PERSONALITY=(
  [none]=qwen3-1.7b-text-original
  [executive]=qwen3-1.7b-text-original-executive
  [surface]=qwen3-1.7b-text-original-surface
  [contrasting_cases]=qwen3-1.7b-text-original-contrasting_cases
  [analogy]=qwen3-1.7b-text-original-analogy
  [error_focused]=qwen3-1.7b-text-original-error_focused
  [example_first]=qwen3-1.7b-text-original-example_first
  [rule_first]=qwen3-1.7b-text-original-rule_first
)

parse_selection() {
  local label=$1 raw=$2 allowed_name=$3 output_name=$4 item candidate found
  local -n allowed="$allowed_name"
  local -n output="$output_name"
  local -a requested=()
  IFS=',' read -r -a requested <<<"$raw"
  output=()
  for item in "${requested[@]}"; do
    candidate="${item//[[:space:]]/}"
    if [[ -z "$candidate" ]]; then
      printf '%s contains an empty selection: %q\n' "$label" "$raw" >&2
      exit 2
    fi
    found=0
    for item in "${allowed[@]}"; do
      if [[ "$candidate" == "$item" ]]; then
        found=1
        break
      fi
    done
    if (( ! found )); then
      printf '%s contains unknown value %q. Allowed: %s\n' \
        "$label" "$candidate" "${allowed[*]}" >&2
      exit 2
    fi
    for item in "${output[@]}"; do
      if [[ "$candidate" == "$item" ]]; then
        printf '%s contains duplicate value %q.\n' "$label" "$candidate" >&2
        exit 2
      fi
    done
    output+=("$candidate")
  done
}

EVAL_TEACHERS="${EVAL_TEACHERS:-all}"
EVAL_PERSONALITIES="${EVAL_PERSONALITIES:-all}"
case "$EVAL_TEACHERS" in
  all) EVAL_TEACHERS="none,surface,executive,contrasting_cases,full" ;;
esac
case "$EVAL_PERSONALITIES" in
  all) EVAL_PERSONALITIES="none,executive,surface,contrasting_cases,analogy,error_focused,example_first,rule_first" ;;
  id) EVAL_PERSONALITIES="none,executive,surface,contrasting_cases" ;;
  ood) EVAL_PERSONALITIES="analogy,error_focused,example_first,rule_first" ;;
esac
TEACHERS=()
PERSONALITIES=()
parse_selection EVAL_TEACHERS "$EVAL_TEACHERS" ALL_TEACHERS TEACHERS
parse_selection EVAL_PERSONALITIES "$EVAL_PERSONALITIES" ALL_PERSONALITIES PERSONALITIES
STUDENTS=()
for personality in "${PERSONALITIES[@]}"; do
  STUDENTS+=("${STUDENT_BY_PERSONALITY[$personality]}")
done
printf '[selection] teachers=%s personalities=%s\n' \
  "$(IFS=,; echo "${TEACHERS[*]}")" "$(IFS=,; echo "${PERSONALITIES[*]}")"

declare -A DEFAULT_TRIALS=(
  [surface]=20260824_082332_0818-personality-surface-8gpu
  [contrasting_cases]=20260824_082351_0818-personality-contrasting-cases-8gpu
  [executive]=20260824_082352_0818-personality-executive-8gpu
  [none]=20260824_192159_0818-personality-none-8gpu
  [full]=20260824_192211_0818-personality-all-8gpu
)
declare -A TRIALS=()
declare -A ADAPTERS=()

discover_adapter() {
  local label=$1 variable explicit trial_variable trial_name trial_dir
  variable="ADAPTER_${label^^}"
  explicit="${!variable:-}"
  if [[ -n "$explicit" ]]; then
    readlink -f "$explicit"
    return
  fi
  trial_variable="TRIAL_${label^^}"
  trial_name="${!trial_variable:-${DEFAULT_TRIALS[$label]}}"
  TRIALS["$label"]="$trial_name"
  trial_dir="$CHECKPOINT_ROOT/$trial_name"
  "$PYTHON" -B - "$trial_dir" <<'PY'
import sys
from pathlib import Path
from examples.tutor.scripts.eval_checkpoints import discover

print(discover(Path(sys.argv[1]))[-1].path)
PY
}

for teacher in "${TEACHERS[@]}"; do
  ADAPTERS["$teacher"]="$(discover_adapter "$teacher")"
  if [[ -z "${TRIALS[$teacher]:-}" ]]; then
    TRIALS["$teacher"]="explicit"
  fi
  printf '[resolve] %-18s %s\n' "$teacher" "${ADAPTERS[$teacher]}"
done

PREFLIGHT_COMMAND=(
  "$PYTHON" -B "$PREFLIGHT"
  --config "$EVAL_CONFIG"
  --teacher-model-path "$TEACHER_MODEL_PATH"
  --max-samples "$EVAL_MAX_SAMPLES"
  --replays "$GENERALIZE_REPLAYS"
)
for teacher in "${TEACHERS[@]}"; do
  PREFLIGHT_COMMAND+=(--adapter "$teacher=${ADAPTERS[$teacher]}")
done
for personality in "${PERSONALITIES[@]}"; do
  PREFLIGHT_COMMAND+=(--personality "$personality")
done
PREFLIGHT_OUTPUT="$("${PREFLIGHT_COMMAND[@]}")"
printf '%s\n' "$PREFLIGHT_OUTPUT"
PREFLIGHT_JSON="$(printf '%s\n' "$PREFLIGHT_OUTPUT" | sed -n 's/^PREFLIGHT_JSON=//p' | tail -1)"
if [[ -z "$PREFLIGHT_JSON" ]]; then
  printf 'Preflight did not emit its machine-readable summary.\n' >&2
  exit 1
fi
EXPECTED_ROWS="$($PYTHON -B -c 'import json,sys; print(json.loads(sys.argv[1])["runtime_dataset_rows"])' "$PREFLIGHT_JSON")"

STAMP="${EVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${EVAL_RUN_DIR:-$TUTOR_FILEROOT/offline_eval/0818-personality/$PHASE-$STAMP}"
if [[ "$PHASE" == "preflight" ]]; then
  printf '[preflight] no files or GPU processes were created.\n'
  printf '[preflight] prospective output=%s\n' "$RUN_DIR"
  exit 0
fi

for command_name in curl setsid nvidia-smi timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf '%s is required for runtime phases.\n' "$command_name" >&2
    exit 1
  fi
done
for model_path in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
  if [[ ! -f "$model_path/config.json" ]]; then
    printf 'Model path has no config.json: %s\n' "$model_path" >&2
    exit 1
  fi
done
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  printf 'Set CUDA_VISIBLE_DEVICES to exactly 4 or 8 free GPU ids.\n' >&2
  exit 2
fi
IFS=',' read -r -a RAW_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
GPU_IDS=()
declare -A SEEN_GPU=()
for raw_gpu in "${RAW_GPU_IDS[@]}"; do
  gpu="${raw_gpu//[[:space:]]/}"
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN_GPU[$gpu]:-}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES needs distinct integer ids; got %q.\n' \
      "$CUDA_VISIBLE_DEVICES" >&2
    exit 2
  fi
  nvidia-smi --id="$gpu" --query-gpu=name --format=csv,noheader >/dev/null
  SEEN_GPU["$gpu"]=1
  GPU_IDS+=("$gpu")
done
if [[ "${#GPU_IDS[@]}" != "4" && "${#GPU_IDS[@]}" != "8" ]]; then
  printf 'Expose exactly 4 or 8 GPUs; got %d.\n' "${#GPU_IDS[@]}" >&2
  exit 2
fi
PAIR_COUNT=$((${#GPU_IDS[@]} / 2))
LAST_PORT=$((BASE_PORT + (PAIR_COUNT - 1) * 10 + 1))
if (( BASE_PORT < 1 || LAST_PORT > 65535 )); then
  printf 'BASE_PORT leaves the valid TCP range for %s pairs.\n' "$PAIR_COUNT" >&2
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
"$PYTHON" -B - "$RUN_DIR/manifest.json" "$PHASE" "$EVAL_MAX_SAMPLES" \
  "$EVAL_CONCURRENCY" "$GENERALIZE_REPLAYS" "$SERVER_MAX_RUNNING_REQUESTS" \
  "$CALLER_MAX_CONCURRENT" "$SAVE_TRACES" "$PAIR_COUNT" "$BASE_PORT" \
  "$EPISODE_ERROR_RETRIES" "$EPISODE_RETRY_BACKOFF_SECONDS" \
  "$EPISODE_TIMEOUT_SECONDS" "$CELL_WALL_TIMEOUT_SECONDS" \
  "$CELL_PROCESS_RESTARTS" "$SKIP_LIVENESS" \
  "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH" "$PREFLIGHT_JSON" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

(
    output, phase, max_samples, concurrency, replays, server_cap, caller_cap,
    save_traces, pair_count, base_port, episode_error_retries,
    episode_retry_backoff_seconds, episode_timeout_seconds,
    cell_wall_timeout_seconds, cell_process_restarts, skip_liveness,
    teacher_model_path, student_model_path, preflight_raw,
) = sys.argv[1:]
preflight = json.loads(preflight_raw)
payload = {
    "phase": phase,
    "eval_max_samples": int(max_samples),
    "eval_concurrency_per_pair": int(concurrency),
    "student_generalize_replays": int(replays),
    "server_max_running_requests": int(server_cap),
    "caller_max_concurrent": int(caller_cap),
    "save_traces": save_traces,
    "pair_count": int(pair_count),
    "base_port": int(base_port),
    "model_paths": {
        "teacher": str(Path(teacher_model_path).resolve()),
        "student": str(Path(student_model_path).resolve()),
    },
    "reliability": {
        "episode_error_retries": int(episode_error_retries),
        "episode_retry_backoff_seconds": int(episode_retry_backoff_seconds),
        "episode_timeout_seconds": int(episode_timeout_seconds),
        "cell_wall_timeout_seconds": int(cell_wall_timeout_seconds),
        "cell_process_restarts": int(cell_process_restarts),
        "skip_liveness": bool(int(skip_liveness)),
    },
    "teachers": preflight["teachers"],
    "students": preflight["students"],
    "dataset": {
        "full_rows": preflight["full_dataset_rows"],
        "phase_rows": preflight["runtime_dataset_rows"],
        "sha256": preflight["runtime_dataset_sha256"],
        "seed": preflight["seed"],
    },
    "semantics": preflight["semantics"],
    "input_hashes": {
        "config": preflight["config_sha256"],
        "resolved_config": preflight["resolved_config_sha256"],
        "prompts": preflight["prompts_sha256"],
        "complaints": preflight["complaints_sha256"],
    },
    "adapters": preflight["adapters"],
    "completed_steps": preflight["completed_steps"],
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous.pop("created_at", None)
    if previous != payload:
        raise SystemExit(
            "Existing manifest differs. Do not mix quick/full, GPU counts, ports, "
            "or checkpoint versions in one EVAL_RUN_DIR."
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
  local output=$1 student=$2 personality=$3 adapter=$4
  "$PYTHON" -B - "$output" "$student" "$personality" "$adapter" \
    "$EXPECTED_ROWS" "$GENERALIZE_REPLAYS" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
student, personality = sys.argv[2:4]
adapter = str(Path(sys.argv[4]).resolve())
expected, replays = map(int, sys.argv[5:7])
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
effective_personality = str(students[0].get("personality") or "none")
if effective_personality != personality:
    raise SystemExit(
        f"{output}: personality={effective_personality}, expected={personality}"
    )
semantics = run_config["test_semantics"]
if run_config["presolve"]["verify"] or semantics["leak_handling_mode"] != "reward_only" \
        or semantics["format_handling_mode"] != "continue":
    raise SystemExit(f"{output}: deployment semantics drifted")
if not semantics["personality"].get("prompts_path"):
    raise SystemExit(f"{output}: personality settings were not forwarded")
if int(semantics["student_generalize_replays"]) != replays:
    raise SystemExit(f"{output}: replay count drifted")
if set(semantics["generalization_levels"]) != {"original", "original_preleak"}:
    raise SystemExit(f"{output}: generalization levels drifted")
for caller in ("teacher", "auxiliary"):
    actual = run_config[caller]["request_params"]["extra_body"]["lora_path"]
    if str(Path(actual).resolve()) != adapter:
        raise SystemExit(f"{output}: {caller} uses wrong LoRA: {actual}")
gate = mode.get("personality_gate") or {}
if personality == "none":
    if int(gate.get("active_episode_count", 0) or 0):
        raise SystemExit(f"{output}: none unexpectedly activated the gate")
elif int(gate.get("active_episode_count", 0) or 0) + pending < expected:
    raise SystemExit(f"{output}: demanding episodes are missing Gate metrics")
print(f"[cell-ok] {output}: rows={expected}, pending={pending}")
PY
}

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

  mkdir -p "$pair_dir/served_adapters"
  local -a served_paths=()
  for teacher in "${TEACHERS[@]}"; do
    target="$pair_dir/served_adapters/$teacher"
    if [[ -L "$target" ]]; then
      if [[ "$(readlink -f "$target")" != "${ADAPTERS[$teacher]}" ]]; then
        printf 'Existing adapter alias points elsewhere: %s\n' "$target" >&2
        return 1
      fi
    elif [[ -e "$target" ]]; then
      printf 'Refusing non-symlink adapter alias: %s\n' "$target" >&2
      return 1
    else
      ln -s "${ADAPTERS[$teacher]}" "$target"
    fi
    served_paths+=("$target")
  done

  CUDA_VISIBLE_DEVICES="$teacher_gpu" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_PATH" --served-model-name "$TEACHER_MODEL" \
    --host 127.0.0.1 --port "$teacher_port" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" --enable-lora \
    --lora-paths "${served_paths[@]}" \
    --max-loras-per-batch 5 --max-loaded-loras 5 \
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
    "$PYTHON" -B - "$teacher_url" "$TEACHER_MODEL" "${served_paths[@]}" <<'PY'
import sys
from examples.tutor.scripts.eval_checkpoints import probe

base_url, model, *paths = sys.argv[1:]
base = probe(base_url, model, "EMPTY", None, 300.0)
try:
    probe(base_url, model, "EMPTY", "/nonexistent/tutor-adapter", 300.0)
except Exception:
    pass
else:
    raise SystemExit("endpoint accepted a nonexistent lora_path")
for path in paths:
    answer = probe(base_url, model, "EMPTY", path, 300.0)
    if not answer or answer == base:
        raise SystemExit(f"adapter is not observably active: {path}")
    print(f"[liveness] active: {path}")
PY
  fi

  run_eval_cell() {
    local teacher=$1 student=$2 personality=$3 output=$4 log=$5
    local alias request_params status cell_try max_cell_tries
    local -a command attempt_command
    alias="$pair_dir/served_adapters/$teacher"
    request_params="$(
      "$PYTHON" -B -c \
        'import json,sys; print(json.dumps({"seed":42,"extra_body":{"lora_path":sys.argv[1],"chat_template_kwargs":{"enable_thinking":False}}}))' \
        "$alias"
    )"
    mkdir -p "$output"
    command=(
      "$PYTHON" -B "$EVALUATOR"
      --config "$EVAL_CONFIG"
      --teacher-base-url "$teacher_url"
      --teacher-model "$TEACHER_MODEL"
      --api-key EMPTY
      --teacher-request-params "$request_params"
      --self-aux-via-teacher
      --teacher-presolve config
      --student-generalization config
      --student-name "$student"
      --attempts 0
      --concurrency "$EVAL_CONCURRENCY"
      --episode-error-retries "$EPISODE_ERROR_RETRIES"
      --episode-error-retry-backoff-seconds "$EPISODE_RETRY_BACKOFF_SECONDS"
      --episode-timeout-seconds "$EPISODE_TIMEOUT_SECONDS"
      --retry-diagnostic-failures
      --save-traces "$SAVE_TRACES"
      --output-dir "$output"
      "auxiliary_model.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
    )
    if (( EVAL_MAX_SAMPLES > 0 )); then
      command+=("+evaluator.max_samples=$EVAL_MAX_SAMPLES")
    fi
    printf '[cell] pair=%s teacher=%s student=%s\n' \
      "$pair_index" "$teacher" "$personality"
    : >"$log"
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
    validate_cell "$output" "$student" "$personality" "$alias"
  }

  local cell_index=0 student_index teacher student personality output log
  for teacher in "${TEACHERS[@]}"; do
    for student_index in "${!STUDENTS[@]}"; do
      student="${STUDENTS[$student_index]}"
      personality="${PERSONALITIES[$student_index]}"
      if (( cell_index % PAIR_COUNT == pair_index )); then
        output="$RUN_DIR/cells/$teacher/$student"
        log="$RUN_DIR/logs/$teacher--$personality.log"
        run_eval_cell "$teacher" "$student" "$personality" "$output" "$log"
      fi
      cell_index=$((cell_index + 1))
    done
  done
)

WORKER_PIDS=()
cleanup_workers() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

printf '[run] phase=%s pairs=%s rows_per_cell=%s output=%s\n' \
  "$PHASE" "$PAIR_COUNT" "$EXPECTED_ROWS" "$RUN_DIR"
for ((pair_index = 0; pair_index < PAIR_COUNT; pair_index++)); do
  run_pair "$pair_index" "${GPU_IDS[$((pair_index * 2))]}" \
    "${GPU_IDS[$((pair_index * 2 + 1))]}" &
  WORKER_PIDS+=("$!")
done

remaining="${#WORKER_PIDS[@]}"
worker_failed=0
while (( remaining > 0 )); do
  if wait -n; then
    remaining=$((remaining - 1))
  else
    worker_failed=1
    break
  fi
done
if (( worker_failed )); then
  printf 'A server pair/cell failed; stopping the other pairs. Logs: %s/logs\n' \
    "$RUN_DIR" >&2
  exit 1
fi
WORKER_PIDS=()
trap - EXIT INT TERM

PENDING_COUNT="$($PYTHON -B - "$RUN_DIR/cells" <<'PY'
import json
import sys
from pathlib import Path

count = 0
for path in Path(sys.argv[1]).glob("*/*/pending_backfill.jsonl"):
    count += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
print(count)
PY
)"
if (( PENDING_COUNT > 0 )); then
  printf '[done-with-pending] %s episodes need backfill. Rerun with the same EVAL_RUN_DIR.\n' \
    "$PENDING_COUNT"
  exit 0
fi
"$PYTHON" -B "$ANALYZER" --run-dir "$RUN_DIR"
printf '[done] personality matrix and analysis: %s\n' "$RUN_DIR"
