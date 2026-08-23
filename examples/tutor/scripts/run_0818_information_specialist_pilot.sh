#!/usr/bin/env bash
set -Eeuo pipefail

# A screening experiment, not the confirmatory specialist run.
#
# It branches four text specialists from one protocol-safe 0818 adapter, then
# evaluates every branch against the four text (ID) and four code (OOD) masks on
# exactly the same held-out items. This removes the cold-start format/leak phase
# that dominates roughly the first 15-20 updates of a base-model run.
# The auto-discovered branch point was trained with a type-probe reward. Because
# every row starts from that identical adapter and the adapter is evaluated as a
# common baseline, this measures local mask-specific divergence; it is not a
# substitute for the eventual from-base specialist comparison.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/run_0818_information_specialist_pilot.sh \
    [all|train|eval|analyze] [2|4] [RUN_DIR]

Recommended:
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
    examples/tutor/scripts/run_0818_information_specialist_pilot.sh all 4

Phases:
  all      train, safety-gate, evaluate, and analyze (default)
  train    produce four branch adapters and RUN_DIR/manifest.env
  eval     use an existing RUN_DIR/manifest.env and run the 4x4 + 4x4 matrix
  analyze  recompute matrix_summary.json from existing results

Profiles:
  4 GPUs  screening profile: two 2-GPU arms at once, 12 branch updates,
          24 held-out items, 2 ID attempts and 1 OOD attempt.
  2 GPUs  smoke profile: four arms in sequence, 8 branch updates,
          16 held-out items and 1 attempt. It is a direction check only.

Important environment variables:
  WARM_START              Common 0818 PEFT adapter. If omitted, discover the
                          latest joint-probe-stratified globalstep99 adapter.
  PILOT_RUN_DIR           Output directory (alternative to positional RUN_DIR).
  STUDENT_MODEL_PATH      Local Qwen3-1.7B snapshot; otherwise reuse the default
                          written in examples/tutor/run_offline.sh.
  TRAIN_STEPS, TRAIN_BATCH_SIZE, TRAIN_N_SAMPLES, TRAIN_REPLAYS
  EVAL_LIMIT, ID_ATTEMPTS, OOD_ATTEMPTS, EVAL_CONCURRENCY
  ALLOW_UNSAFE_EVAL=1     Continue if the common-step safety gate fails.
  DRY_RUN=1               Validate and print training commands; launch nothing.

The default safety gate requires the same final checkpoint for all four arms and
stops the whole matrix if any arm's last four training steps average leak > .15,
format errors > .05, or show a leak swing > .10. It never selects checkpoints by
their held-out score.
EOF
}

PHASE="${1:-all}"
if [[ "$PHASE" == "-h" || "$PHASE" == "--help" ]]; then
  usage
  exit 0
fi
case "$PHASE" in
  all|train|eval|analyze) ;;
  *)
    printf 'Unknown phase: %s\n' "$PHASE" >&2
    usage >&2
    exit 2
    ;;
esac
if (( $# > 0 )); then shift; fi

GPU_COUNT="${1:-4}"
case "$GPU_COUNT" in
  2|4) ;;
  *)
    printf 'GPU count must be 2 or 4; got %s.\n' "$GPU_COUNT" >&2
    exit 2
    ;;
esac
if (( $# > 0 )); then shift; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing venv Python: %s\n' "$PYTHON" >&2
  exit 1
fi

if [[ "$(basename "$(dirname "$ROOT_DIR")")" == "AReaL.worktrees" ]]; then
  PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
else
  PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
fi
TUTOR_FILEROOT="${TUTOR_FILEROOT:-$PROJECT_ROOT/output/tutor}"
STAMP="${PILOT_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
POSITIONAL_RUN_DIR="${1:-}"
if [[ -n "$POSITIONAL_RUN_DIR" ]]; then shift; fi
RUN_DIR="${POSITIONAL_RUN_DIR:-${PILOT_RUN_DIR:-$TUTOR_FILEROOT/information_specialist_pilot/$STAMP}}"
RUN_DIR="$(mkdir -p "$RUN_DIR" && cd "$RUN_DIR" && pwd)"
MANIFEST="$RUN_DIR/manifest.env"

EVAL_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/pilot/eval-all-students.yaml"
ANALYZER="$ROOT_DIR/examples/tutor/scripts/analyze_information_specialist_matrix.py"
TRAIN_LAUNCHER="$ROOT_DIR/examples/tutor/run_offline.sh"
for required in "$EVAL_CONFIG" "$ANALYZER" "$TRAIN_LAUNCHER"; do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 1
  fi
done

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export INF_API_KEY="EMPTY"

if [[ "$GPU_COUNT" == "4" ]]; then
  DEFAULT_TRAIN_STEPS=12
  DEFAULT_SAVE_EVERY=6
  DEFAULT_EVAL_LIMIT=24
  DEFAULT_ID_ATTEMPTS=2
  DEFAULT_OOD_ATTEMPTS=1
  PROFILE=screen
else
  DEFAULT_TRAIN_STEPS=8
  DEFAULT_SAVE_EVERY=4
  DEFAULT_EVAL_LIMIT=16
  DEFAULT_ID_ATTEMPTS=1
  DEFAULT_OOD_ATTEMPTS=1
  PROFILE=smoke
fi
TRAIN_STEPS="${TRAIN_STEPS:-$DEFAULT_TRAIN_STEPS}"
SAVE_EVERY="${SAVE_EVERY:-$DEFAULT_SAVE_EVERY}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_N_SAMPLES="${TRAIN_N_SAMPLES:-4}"
TRAIN_REPLAYS="${TRAIN_REPLAYS:-4}"
EVAL_LIMIT="${EVAL_LIMIT:-$DEFAULT_EVAL_LIMIT}"
ID_ATTEMPTS="${ID_ATTEMPTS:-$DEFAULT_ID_ATTEMPTS}"
OOD_ATTEMPTS="${OOD_ATTEMPTS:-$DEFAULT_OOD_ATTEMPTS}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-3}"
MAX_TRAIN_LEAK="${MAX_TRAIN_LEAK:-0.15}"
MAX_TRAIN_FORMAT="${MAX_TRAIN_FORMAT:-0.05}"
MAX_TRAIN_LEAK_SWING="${MAX_TRAIN_LEAK_SWING:-0.10}"
ALLOW_UNSAFE_EVAL="${ALLOW_UNSAFE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

for integer_name in TRAIN_STEPS SAVE_EVERY TRAIN_BATCH_SIZE TRAIN_N_SAMPLES \
  TRAIN_REPLAYS EVAL_LIMIT ID_ATTEMPTS OOD_ATTEMPTS EVAL_CONCURRENCY; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %q.\n' "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
if (( TRAIN_STEPS % SAVE_EVERY != 0 )); then
  printf 'TRAIN_STEPS (%d) must be divisible by SAVE_EVERY (%d).\n' \
    "$TRAIN_STEPS" "$SAVE_EVERY" >&2
  exit 2
fi
for flag_name in ALLOW_UNSAFE_EVAL DRY_RUN; do
  flag_value="${!flag_name}"
  if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
    printf '%s must be 0 or 1; got %q.\n' "$flag_name" "$flag_value" >&2
    exit 2
  fi
done

LABELS=(original student_fade teacher_fade long_drop)
declare -A CONFIGS=()
CONFIGS[original]="$ROOT_DIR/examples/tutor/configs/math/0818/2gpu/life-t10-r8-single-student/text-original.yaml"
CONFIGS[student_fade]="$ROOT_DIR/examples/tutor/configs/math/0818/2gpu/life-t10-r8-single-student/text-student-fade.yaml"
CONFIGS[teacher_fade]="$ROOT_DIR/examples/tutor/configs/math/0818/2gpu/life-t10-r8-single-student/text-teacher-fade.yaml"
CONFIGS[long_drop]="$ROOT_DIR/examples/tutor/configs/math/0818/2gpu/life-t10-r8-single-student/text-long-drop.yaml"
for label in "${LABELS[@]}"; do
  if [[ ! -f "${CONFIGS[$label]}" ]]; then
    printf 'Missing training config: %s\n' "${CONFIGS[$label]}" >&2
    exit 1
  fi
done

discover_warm_start() {
  local checkpoint_root candidate
  checkpoint_root="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline"
  candidate="$(find "$checkpoint_root" -type f \
    -path '*0818-life-joint-probe-stratified-8gpu/default/*globalstep99/adapter_model.safetensors' \
    -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$candidate" ]]; then
    return 1
  fi
  printf '%s\n' "$candidate"
}

WARM_START="${WARM_START:-}"
if [[ "$PHASE" == "train" || "$PHASE" == "all" ]]; then
  if [[ -z "$WARM_START" ]]; then
    WARM_START="$(discover_warm_start || true)"
  fi
  if [[ ! -f "$WARM_START/adapter_model.safetensors" ]]; then
    printf 'Set WARM_START to a compatible PEFT adapter; got %q.\n' "$WARM_START" >&2
    exit 1
  fi
fi

CHILD_PIDS=()
SERVER_PIDS=()
cleanup() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${CHILD_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${CHILD_PIDS[@]}" "${SERVER_PIDS[@]}"; do
    [[ -n "$pid" ]] || continue
    wait "$pid" 2>/dev/null || true
  done
  exit "$status"
}
trap cleanup EXIT INT TERM

detect_gpu_ids() {
  local count index
  count="$($PYTHON -B -c 'import torch; print(torch.cuda.device_count())')"
  for ((index = 0; index < count; index++)); do printf '%s\n' "$index"; done
}

GPU_IDS=()
if [[ "$PHASE" != "analyze" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a RAW_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
  else
    mapfile -t RAW_GPU_IDS < <(detect_gpu_ids)
  fi
  for raw_gpu in "${RAW_GPU_IDS[@]}"; do
    gpu="${raw_gpu//[[:space:]]/}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      printf 'CUDA_VISIBLE_DEVICES must contain integer ids; got %q.\n' "$raw_gpu" >&2
      exit 2
    fi
    GPU_IDS+=("$gpu")
  done
  if (( ${#GPU_IDS[@]} < GPU_COUNT )); then
    printf 'Need %d GPUs, but CUDA_VISIBLE_DEVICES exposes %d.\n' \
      "$GPU_COUNT" "${#GPU_IDS[@]}" >&2
    exit 1
  fi
  GPU_IDS=("${GPU_IDS[@]:0:GPU_COUNT}")
fi

join_pair() {
  printf '%s,%s' "$1" "$2"
}

declare -A TRIALS=()
declare -A ADAPTERS=()
for label in "${LABELS[@]}"; do
  hyphen_label="${label//_/-}"
  TRIALS[$label]="${STAMP}_0818-info-pilot-text-${hyphen_label}"
done

launch_training_arm() {
  local label=$1 gpu_spec=$2 student_port=$3 log_path
  local -a command overrides
  log_path="$RUN_DIR/train-${label}.launcher.log"
  overrides=(
    "trial_name=${TRIALS[$label]}"
    "+actor.init_lora_path=$WARM_START"
    "total_train_steps=$TRAIN_STEPS"
    "saver.freq_steps=$SAVE_EVERY"
    "student_generalize.replays=$TRAIN_REPLAYS"
    "train_dataset.batch_size=$TRAIN_BATCH_SIZE"
    "gconfig.n_samples=$TRAIN_N_SAMPLES"
    "recover.freq_epochs=null"
    "recover.freq_steps=null"
    "recover.freq_secs=null"
    "debug_trace_every_n_rollouts=1000000"
    "stats_logger.wandb.group=0818-information-specialist-pilot"
  )
  command=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu_spec"
    "STUDENT_PORT=$student_port"
    "DRY_RUN=$DRY_RUN"
    bash "$TRAIN_LAUNCHER" 2 "${CONFIGS[$label]}" "${overrides[@]}"
  )
  printf '[train] %-12s GPUs=%s port=%s log=%s\n' \
    "$label" "$gpu_spec" "$student_port" "$log_path"
  if [[ "$DRY_RUN" == "1" ]]; then
    "${command[@]}"
    return
  fi
  setsid "${command[@]}" >"$log_path" 2>&1 &
  CHILD_PIDS+=("$!")
}

wait_training_wave() {
  local pid failed=0
  for pid in "${CHILD_PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  CHILD_PIDS=()
  if (( failed )); then
    printf 'A training arm failed; inspect %s/train-*.launcher.log.\n' "$RUN_DIR" >&2
    return 1
  fi
}

find_final_adapter() {
  local trial=$1 checkpoint_root expected best_path="" best_step=-1 path base step
  checkpoint_root="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline/$trial/default"
  expected=$((TRAIN_STEPS - 1))
  while IFS= read -r path; do
    base="$(basename "$path")"
    if [[ "$base" =~ globalstep([0-9]+)$ ]]; then
      step="${BASH_REMATCH[1]}"
      if (( step > best_step )); then best_step=$step; best_path=$path; fi
    fi
  done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d \
    -exec test -f '{}/adapter_model.safetensors' \; -print 2>/dev/null | sort)
  if [[ -z "$best_path" || "$best_step" != "$expected" ]]; then
    printf 'Expected globalstep%d adapter under %s; newest was globalstep%d.\n' \
      "$expected" "$checkpoint_root" "$best_step" >&2
    return 1
  fi
  printf '%s\n' "$best_path"
}

write_manifest() {
  local temporary="$MANIFEST.tmp"
  {
    printf 'PILOT_PROFILE=%q\n' "$PROFILE"
    printf 'PILOT_GPU_COUNT=%q\n' "$GPU_COUNT"
    printf 'PILOT_STAMP=%q\n' "$STAMP"
    printf 'PILOT_RUN_DIR=%q\n' "$RUN_DIR"
    printf 'PILOT_WARM_START=%q\n' "$WARM_START"
    printf 'PILOT_TRAIN_STEPS=%q\n' "$TRAIN_STEPS"
    printf 'PILOT_SAVE_EVERY=%q\n' "$SAVE_EVERY"
    for label in "${LABELS[@]}"; do
      printf 'PILOT_TRIAL_%s=%q\n' "$label" "${TRIALS[$label]}"
      printf 'PILOT_ADAPTER_%s=%q\n' "$label" "${ADAPTERS[$label]}"
    done
  } >"$temporary"
  mv "$temporary" "$MANIFEST"
}

run_train_phase() {
  printf '[pilot] profile=%s run_dir=%s warm_start=%s\n' \
    "$PROFILE" "$RUN_DIR" "$WARM_START"
  if [[ "$GPU_COUNT" == "4" ]]; then
    launch_training_arm original "$(join_pair "${GPU_IDS[0]}" "${GPU_IDS[1]}")" 31101
    launch_training_arm student_fade "$(join_pair "${GPU_IDS[2]}" "${GPU_IDS[3]}")" 31102
    wait_training_wave
    launch_training_arm teacher_fade "$(join_pair "${GPU_IDS[0]}" "${GPU_IDS[1]}")" 31101
    launch_training_arm long_drop "$(join_pair "${GPU_IDS[2]}" "${GPU_IDS[3]}")" 31102
    wait_training_wave
  else
    pair="$(join_pair "${GPU_IDS[0]}" "${GPU_IDS[1]}")"
    for label in "${LABELS[@]}"; do
      launch_training_arm "$label" "$pair" 31101
      wait_training_wave
    done
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[pilot] dry run complete; no manifest or adapters were created.\n'
    return
  fi
  for label in "${LABELS[@]}"; do
    ADAPTERS[$label]="$(find_final_adapter "${TRIALS[$label]}")"
  done
  write_manifest
  printf '[pilot] training complete; manifest=%s\n' "$MANIFEST"
}

load_manifest() {
  if [[ ! -f "$MANIFEST" ]]; then
    printf 'Missing pilot manifest: %s\n' "$MANIFEST" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$MANIFEST"
  WARM_START="$PILOT_WARM_START"
  for label in "${LABELS[@]}"; do
    trial_var="PILOT_TRIAL_${label}"
    adapter_var="PILOT_ADAPTER_${label}"
    TRIALS[$label]="${!trial_var}"
    ADAPTERS[$label]="${!adapter_var}"
    if [[ ! -f "${ADAPTERS[$label]}/adapter_model.safetensors" ]]; then
      printf 'Manifest adapter is missing: %s\n' "${ADAPTERS[$label]}" >&2
      exit 1
    fi
  done
}

check_training_safety() {
  local -a logs=()
  for label in "${LABELS[@]}"; do
    logs+=("$TUTOR_FILEROOT/logs/$(id -un)/tutor-math-baseline/${TRIALS[$label]}/merged.log")
  done
  if "$PYTHON" -B - "$MAX_TRAIN_LEAK" "$MAX_TRAIN_FORMAT" \
      "$MAX_TRAIN_LEAK_SWING" "${logs[@]}" <<'PY'
import re
import sys
from pathlib import Path

max_leak, max_format, max_swing = map(float, sys.argv[1:4])
ansi = re.compile(r"\x1b\[[0-9;:]*m")
step_re = re.compile(r"Train step (\d+)/")
metrics = {
    "leak": re.compile(r"rollout/leaks\s*[^-+0-9]*([-+0-9.eE]+)"),
    "format": re.compile(r"rollout/format_errors\s*[^-+0-9]*([-+0-9.eE]+)"),
}
failed = False
for raw_path in sys.argv[4:]:
    path = Path(raw_path)
    if not path.is_file():
        print(f"[safety] missing log: {path}", file=sys.stderr)
        failed = True
        continue
    rows = {}
    current = None
    for raw in path.open(errors="replace"):
        line = ansi.sub("", raw)
        match = step_re.search(line)
        if match:
            current = int(match.group(1))
            rows[current] = {}
            continue
        if current is None:
            continue
        for name, pattern in metrics.items():
            match = pattern.search(line)
            if match:
                rows[current][name] = float(match.group(1))
    complete = [(step, row) for step, row in sorted(rows.items()) if len(row) == 2]
    if len(complete) < 4:
        print(f"[safety] {path.parent.name}: fewer than four complete steps", file=sys.stderr)
        failed = True
        continue
    tail = complete[-4:]
    leak = sum(row["leak"] for _, row in tail) / 4
    fmt = sum(row["format"] for _, row in tail) / 4
    first = sum(row["leak"] for _, row in tail[:2]) / 2
    last = sum(row["leak"] for _, row in tail[2:]) / 2
    swing = abs(last - first)
    ok = leak <= max_leak and fmt <= max_format and swing <= max_swing
    print(
        f"[safety] {path.parent.name}: steps {tail[0][0]}-{tail[-1][0]} "
        f"leak={leak:.4f} format={fmt:.4f} leak_swing={swing:.4f} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    failed |= not ok
raise SystemExit(1 if failed else 0)
PY
  then
    return 0
  fi
  if [[ "$ALLOW_UNSAFE_EVAL" == "1" ]]; then
    printf '[safety] WARNING: gate failed; continuing because ALLOW_UNSAFE_EVAL=1.\n' >&2
    return 0
  fi
  printf '[safety] gate failed; no held-out matrix was run. Set ALLOW_UNSAFE_EVAL=1 to override.\n' >&2
  return 1
}

process_running() {
  local pid=$1
  kill -0 "$pid" 2>/dev/null
}

wait_ready() {
  local pid=$1 url=$2 log_path=$3 label=$4 deadline
  deadline=$((SECONDS + ${SERVER_READY_TIMEOUT:-900}))
  while (( SECONDS < deadline )); do
    if ! process_running "$pid"; then
      printf '%s server exited during startup. Tail of %s:\n' "$label" "$log_path" >&2
      tail -80 "$log_path" >&2 || true
      return 1
    fi
    if curl --silent --fail --max-time 2 "$url/models" >/dev/null 2>&1; then
      printf '[eval] %s server ready: %s\n' "$label" "$url"
      return 0
    fi
    sleep 2
  done
  printf '%s server did not become ready; tail of %s:\n' "$label" "$log_path" >&2
  tail -80 "$log_path" >&2 || true
  return 1
}

resolve_student_model_path() {
  if [[ -n "${STUDENT_MODEL_PATH:-}" ]]; then
    printf '%s\n' "$STUDENT_MODEL_PATH"
    return
  fi
  "$PYTHON" -B - "$TRAIN_LAUNCHER" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r'^STUDENT_MODEL_PATH="\$\{STUDENT_MODEL_PATH:-(.+)\}"$',
    text,
    flags=re.MULTILINE,
)
if match is None:
    raise SystemExit("Could not discover STUDENT_MODEL_PATH from run_offline.sh")
print(match.group(1))
PY
}

resolve_teacher_model_path() {
  if [[ -n "${TEACHER_MODEL_PATH:-}" ]]; then
    printf '%s\n' "$TEACHER_MODEL_PATH"
    return
  fi
  "$PYTHON" -B - "$EVAL_CONFIG" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

eval_config = Path(sys.argv[1])
base_config = eval_config.parent.parent / "base" / "default.yaml"
print(OmegaConf.load(base_config).actor.path)
PY
}

link_served_adapter() {
  local label=$1 source=$2 target="$RUN_DIR/served_adapters/$label"
  mkdir -p "$RUN_DIR/served_adapters"
  if [[ -L "$target" ]]; then
    if [[ "$(readlink -f "$target")" != "$(readlink -f "$source")" ]]; then
      printf 'Adapter link points elsewhere: %s\n' "$target" >&2
      return 1
    fi
  elif [[ -e "$target" ]]; then
    printf 'Refusing to replace non-symlink path: %s\n' "$target" >&2
    return 1
  else
    ln -s "$source" "$target"
  fi
  printf '%s\n' "$target"
}

launch_eval_job() {
  local teacher=$1 split=$2 adapter=$3 attempts=$4 output log request_params
  local -a students command
  output="$RUN_DIR/eval/final/teachers/$teacher/$split"
  log="$RUN_DIR/eval-${teacher}-${split}.log"
  mkdir -p "$output"
  request_params="$($PYTHON -B -c \
    'import json,sys; print(json.dumps({"seed":42,"extra_body":{"lora_path":sys.argv[1],"chat_template_kwargs":{"enable_thinking":False}}}))' \
    "$adapter")"
  if [[ "$split" == "id" ]]; then
    students=(
      qwen3-1.7b-text-original
      qwen3-1.7b-text-student_fade
      qwen3-1.7b-text-teacher_fade
      qwen3-1.7b-text-long_drop
    )
  else
    students=(
      qwen3-1.7b-code-original
      qwen3-1.7b-code-student_fade
      qwen3-1.7b-code-teacher_fade
      qwen3-1.7b-code-long_drop
    )
  fi
  command=(
    "$PYTHON" examples/tutor/scripts/evaluate_api_teacher.py
    --config "$EVAL_CONFIG"
    --teacher-base-url "$TEACHER_BASE_URL"
    --teacher-model "$TEACHER_MODEL"
    --api-key EMPTY
    --teacher-request-params "$request_params"
    --teacher-presolve on
    --presolve-attempts 1
    --attempts "$attempts"
    --limit "$EVAL_LIMIT"
    --concurrency "$EVAL_CONCURRENCY"
    --save-traces errors
    --output-dir "$output"
  )
  for student in "${students[@]}"; do command+=(--student-name "$student"); done
  printf '[eval] %-12s %-3s attempts=%s output=%s\n' \
    "$teacher" "$split" "$attempts" "$output"
  setsid "${command[@]}" >"$log" 2>&1 &
  CHILD_PIDS+=("$!")
}

wait_eval_jobs() {
  local pid failed=0
  for pid in "${CHILD_PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  CHILD_PIDS=()
  if (( failed )); then
    printf 'An evaluation job failed; inspect %s/eval-*.log.\n' "$RUN_DIR" >&2
    return 1
  fi
}

run_eval_phase() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN covers the train phase; eval would start model servers.\n' >&2
    return 2
  fi
  load_manifest
  check_training_safety
  unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
  export NO_PROXY="127.0.0.1,localhost"
  export no_proxy="$NO_PROXY"
  export DEEPSEEK_API_KEY="EMPTY"

  TEACHER_PORT="${TEACHER_PORT:-32000}"
  EVAL_STUDENT_PORT="${EVAL_STUDENT_PORT:-32001}"
  TEACHER_BASE_URL="http://127.0.0.1:${TEACHER_PORT}/v1"
  STUDENT_BASE_URL="http://127.0.0.1:${EVAL_STUDENT_PORT}/v1"
  TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
  STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
  export TUTOR_QWEN3_8B_BASE_URL="$TEACHER_BASE_URL"
  export TUTOR_QWEN3_1_7B_BASE_URL="$STUDENT_BASE_URL"

  for occupied_url in "$TEACHER_BASE_URL" "$STUDENT_BASE_URL"; do
    if curl --silent --fail --max-time 2 "$occupied_url/models" >/dev/null 2>&1; then
      printf 'An OpenAI-compatible server already occupies %s. Choose other ports.\n' \
        "$occupied_url" >&2
      return 1
    fi
  done

  STUDENT_MODEL_PATH="$(resolve_student_model_path)"
  TEACHER_MODEL_PATH="$(resolve_teacher_model_path)"
  for model_path in "$STUDENT_MODEL_PATH" "$TEACHER_MODEL_PATH"; do
    if [[ ! -f "$model_path/config.json" ]]; then
      printf 'Model path has no config.json: %s\n' "$model_path" >&2
      exit 1
    fi
  done

  declare -A served=()
  served[baseline]="$(link_served_adapter baseline "$WARM_START")"
  for label in "${LABELS[@]}"; do
    served[$label]="$(link_served_adapter "$label" "${ADAPTERS[$label]}")"
  done
  served_paths=(
    "${served[baseline]}"
    "${served[original]}"
    "${served[student_fade]}"
    "${served[teacher_fade]}"
    "${served[long_drop]}"
  )

  teacher_log="$RUN_DIR/sglang-teacher.log"
  student_log="$RUN_DIR/sglang-student.log"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_PATH" \
    --served-model-name "$TEACHER_MODEL" \
    --host 127.0.0.1 --port "$TEACHER_PORT" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests 32 --enable-lora \
    --lora-paths "${served_paths[@]}" \
    --max-loras-per-batch 5 --max-loaded-loras 5 \
    >"$teacher_log" 2>&1 &
  teacher_pid=$!
  SERVER_PIDS+=("$teacher_pid")

  CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$STUDENT_MODEL_PATH" \
    --served-model-name "$STUDENT_MODEL" \
    --host 127.0.0.1 --port "$EVAL_STUDENT_PORT" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests 32 \
    >"$student_log" 2>&1 &
  student_pid=$!
  SERVER_PIDS+=("$student_pid")

  wait_ready "$teacher_pid" "$TEACHER_BASE_URL" "$teacher_log" teacher
  wait_ready "$student_pid" "$STUDENT_BASE_URL" "$student_log" student

  "$PYTHON" -B - "$TEACHER_BASE_URL" "$TEACHER_MODEL" "${served_paths[@]}" <<'PY'
import sys
from pathlib import Path

from examples.tutor.scripts.eval_checkpoints import Checkpoint, verify_adapter_is_live

base_url, model, *paths = sys.argv[1:]
checkpoints = [Checkpoint(step=index, path=Path(path)) for index, path in enumerate(paths)]
verify_adapter_is_live(checkpoints, base_url, model, "EMPTY", 300.0)
PY

  for teacher in baseline "${LABELS[@]}"; do
    launch_eval_job "$teacher" id "${served[$teacher]}" "$ID_ATTEMPTS"
    launch_eval_job "$teacher" ood "${served[$teacher]}" "$OOD_ATTEMPTS"
  done
  wait_eval_jobs

  "$PYTHON" -B "$ANALYZER" \
    --eval-root "$RUN_DIR/eval/final" \
    --output "$RUN_DIR/eval/final/matrix_summary.json"
}

run_analyze_phase() {
  "$PYTHON" -B "$ANALYZER" \
    --eval-root "$RUN_DIR/eval/final" \
    --output "$RUN_DIR/eval/final/matrix_summary.json"
}

case "$PHASE" in
  train)
    run_train_phase
    ;;
  eval)
    run_eval_phase
    ;;
  analyze)
    run_analyze_phase
    ;;
  all)
    run_train_phase
    if [[ "$DRY_RUN" != "1" ]]; then run_eval_phase; fi
    ;;
esac

printf '[pilot] done: %s\n' "$RUN_DIR"
