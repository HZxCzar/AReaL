#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf '%s\n' \
    'Self-hosted tutor training: local Qwen3-1.7B student + in-engine Qwen3-8B auxiliary.' \
    '' \
    'Usage:' \
    '  bash examples/tutor/run_offline.sh [2|4|8] [CONFIG.yaml] [CONFIG_OVERRIDE ...]' \
    '' \
    'Examples:' \
    '  bash examples/tutor/run_offline.sh' \
    '  bash examples/tutor/run_offline.sh 4' \
    '  bash examples/tutor/run_offline.sh 2 examples/tutor/configs/math/0818/2gpu/base.yaml' \
    '  DRY_RUN=1 bash examples/tutor/run_offline.sh 2 CONFIG.yaml total_train_steps=2' \
    '' \
    'The generation/training GPU split is NOT a launcher option. It is read out' \
    'of the config backends, sglang:d<gen>p1t1 and fsdp:d<actor>p1t1, which must' \
    'sum to <n>; the allocation file is where the split is written down.' \
    '' \
    'Main environment overrides:' \
    '  CONFIG                         Fallback when positional CONFIG.yaml is omitted.' \
    '  CUDA_VISIBLE_DEVICES           At least <n> integer GPU ids; first <n> are used.' \
    '  STUDENT_MODEL_PATH             Local Qwen3-1.7B snapshot path.' \
    '  STUDENT_PORT                   Local OpenAI port (default 30001).' \
    '  STUDENT_MEM_FRACTION_STATIC    Student server allocation (default 0.20).' \
    '  DRY_RUN=1                      Validate and print commands without launching.'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GPU_COUNT="${1:-2}"
if (( $# > 0 )); then
  shift
fi
case "$GPU_COUNT" in
  2|4|8) ;;
  *)
    printf 'GPU count must be 2, 4, or 8; got %q.\n' "$GPU_COUNT" >&2
    usage >&2
    exit 2
    ;;
esac

POSITIONAL_CONFIG=""
if (( $# > 0 )); then
  case "$1" in
    *.yaml|*.yml|*.YAML|*.YML)
      POSITIONAL_CONFIG="$1"
      shift
      ;;
  esac
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="${POSITIONAL_CONFIG:-${CONFIG:-examples/tutor/configs/math/0818/${GPU_COUNT}gpu/base.yaml}}"
# Everything after CONFIG is user-authored and is forwarded verbatim. This
# launcher must never synthesize training-config overrides.
USER_OVERRIDES=("$@")

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing executable venv Python: %s\nRun: uv sync --extra cuda\n' "$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Config does not exist: %s\n' "$CONFIG" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  printf 'curl is required for local model health checks.\n' >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  printf 'setsid is required to clean up all local SGLang worker processes.\n' >&2
  exit 1
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

# AReaL's local scheduler launches RPC and SGLang workers with the literal
# `python3` command. Calling the trainer through .venv/bin/python is not enough:
# without an activated PATH those children fall back to /usr/bin/python3 and mix
# the system importlib with the venv's Python 3.12 standard library.
unset PYTHONHOME
export VIRTUAL_ENV="$ROOT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r
if [[ "$(command -v python3)" != "$VIRTUAL_ENV/bin/python3" ]]; then
  printf 'Worker python3 did not resolve to the project venv: %s\n' \
    "$(command -v python3)" >&2
  exit 1
fi

# All inference is local. Keep HTTP clients from routing localhost through a proxy,
# and satisfy legacy config/wrapper fields without carrying a real credential.
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export INF_API_KEY="EMPTY"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export DO_NOT_TRACK=1
export PYTHONPATH="$ROOT_DIR"
export PYTHONUNBUFFERED=1

DRY_RUN="${DRY_RUN:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  printf 'DRY_RUN must be 0 or 1; got %q.\n' "$DRY_RUN" >&2
  exit 2
fi

detect_gpu_ids() {
  local detected_count index
  detected_count="$($PYTHON -B -c 'import torch; print(torch.cuda.device_count())')"
  for ((index = 0; index < detected_count; index++)); do
    printf '%s\n' "$index"
  done
}

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a RAW_GPU_IDS <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t RAW_GPU_IDS < <(detect_gpu_ids)
  if [[ "$DRY_RUN" == "1" && ${#RAW_GPU_IDS[@]} -lt $GPU_COUNT ]]; then
    RAW_GPU_IDS=()
    for ((gpu_index = 0; gpu_index < GPU_COUNT; gpu_index++)); do
      RAW_GPU_IDS+=("$gpu_index")
    done
  fi
fi

declare -A SEEN_GPU_IDS=()
for raw_gpu_id in "${RAW_GPU_IDS[@]}"; do
  gpu_id="${raw_gpu_id//[[:space:]]/}"
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    printf 'CUDA_VISIBLE_DEVICES must contain integer GPU ids; got %q.\n' "$raw_gpu_id" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES contains duplicate GPU id %s.\n' "$gpu_id" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$gpu_id]=1
  GPU_IDS+=("$gpu_id")
done

if (( ${#GPU_IDS[@]} < GPU_COUNT )); then
  printf 'Requested %d GPUs but only %d are visible: %s\n' \
    "$GPU_COUNT" "${#GPU_IDS[@]}" "${CUDA_VISIBLE_DEVICES:-<none>}" >&2
  exit 1
fi

SELECTED_GPU_IDS=("${GPU_IDS[@]:0:GPU_COUNT}")

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

SELECTED_GPU_SPEC="$(join_by_comma "${SELECTED_GPU_IDS[@]}")"
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU_SPEC"

if [[ "$DRY_RUN" != "1" ]]; then
  VISIBLE_COUNT="$($PYTHON -B -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "$VISIBLE_COUNT" != "$GPU_COUNT" ]]; then
    printf 'PyTorch sees %s GPUs after CUDA_VISIBLE_DEVICES=%s; expected %s.\n' \
      "$VISIBLE_COUNT" "$CUDA_VISIBLE_DEVICES" "$GPU_COUNT" >&2
    exit 1
  fi
  "$PYTHON" -B - <<'PY'
import sys
import torch

low_vram = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    gib = props.total_memory / 2**30
    print(f"GPU logical {index}: {props.name}, {gib:.1f} GiB")
    if gib < 40:
        low_vram.append((index, gib))
if low_vram:
    print(
        "WARNING: colocated BF16 Qwen3-8B + Qwen3-1.7B is tight below 40 GiB; "
        "reduce context/concurrency or use a quantized student.",
        file=sys.stderr,
    )
PY
fi

STUDENT_HOST="127.0.0.1"
STUDENT_PORT="${STUDENT_PORT:-30001}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
STUDENT_CONTEXT_LENGTH="${STUDENT_CONTEXT_LENGTH:-40960}"
STUDENT_MEM_FRACTION_STATIC="${STUDENT_MEM_FRACTION_STATIC:-0.20}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
STUDENT_PROBE_TIMEOUT="${STUDENT_PROBE_TIMEOUT:-120}"

if [[ ! "$STUDENT_PORT" =~ ^[1-9][0-9]*$ ]] || (( STUDENT_PORT > 65535 )); then
  printf 'STUDENT_PORT must be in 1..65535; got %q.\n' "$STUDENT_PORT" >&2
  exit 2
fi
for integer_name in \
  STUDENT_CONTEXT_LENGTH SERVER_READY_TIMEOUT STUDENT_PROBE_TIMEOUT; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %q.\n' "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
if [[ ! "$STUDENT_MODEL" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  printf 'STUDENT_MODEL contains unsupported characters: %q.\n' "$STUDENT_MODEL" >&2
  exit 2
fi
if [[ "$STUDENT_MODEL_PATH" == /* || "$STUDENT_MODEL_PATH" == ./* ]]; then
  if [[ ! -f "$STUDENT_MODEL_PATH/config.json" ]]; then
    printf 'Local STUDENT_MODEL_PATH has no config.json: %s\n' "$STUDENT_MODEL_PATH" >&2
    exit 1
  fi
fi

STUDENT_BASE_URL="http://${STUDENT_HOST}:${STUDENT_PORT}/v1"
export TUTOR_QWEN3_1_7B_BASE_URL="$STUDENT_BASE_URL"
# Unused when the config selects mode=self; this prevents stale external resolution.
export TUTOR_QWEN3_8B_BASE_URL="http://127.0.0.1:1/v1"
export STUDENT_MEM_FRACTION_STATIC

# Resolve the final config before reserving GPU memory. The generation/training
# split is read OUT of the config's backends rather than imposed on them, so the
# allocation file stays the single place the split is written down.
CONFIG_SUMMARY="$("$PYTHON" -B - "$CONFIG" "$GPU_COUNT" "$STUDENT_BASE_URL" \
  "$STUDENT_MODEL" "${USER_OVERRIDES[@]}" <<'PY'
import json
import re
import sys
from urllib.parse import urlparse

from omegaconf import OmegaConf

from areal.api.cli_args import parse_cli_args, to_structured_cfg
from examples.tutor.configs import TutorConfig

(
    config_path,
    gpu_count,
    student_url,
    served_model,
    *overrides,
) = sys.argv[1:]
raw_config, _ = parse_cli_args(["--config", config_path, *overrides])
structured_config = to_structured_cfg(raw_config, TutorConfig)
config = OmegaConf.to_object(structured_config)
if not isinstance(config, TutorConfig):
    raise SystemExit(f"Expected TutorConfig, got {type(config).__name__}.")
gpu_count = int(gpu_count)


def _backend_devices(role, backend, prefix):
    """The device count a d<N>p1t1 backend string asks for."""
    match = re.fullmatch(rf"{prefix}:d(\d+)p(\d+)t(\d+)", backend)
    if match is None:
        raise SystemExit(
            f"{role} backend must look like {prefix}:d<N>p1t1, got {backend!r}."
        )
    devices, pipeline, tensor = (int(part) for part in match.groups())
    if pipeline != 1 or tensor != 1:
        raise SystemExit(
            f"{role} backend must be p1t1 for the offline launcher, got {backend!r}."
        )
    return devices


# THE SPLIT COMES FROM THE CONFIG, and half-and-half is no longer assumed. The
# trainer is the critical path at every allocation measured -- timeperf/rollout,
# the time the trainer waits on generated data, is 0.0 at 2 and 4 GPUs -- so the
# 4- and 8-GPU allocations hand generation one and two cards and give the rest to
# the actor. Reading the counts out of the backends keeps the allocation file the
# only place the split is stated.
gen_gpu_count = _backend_devices("rollout", config.rollout.backend, "sglang")
actor_gpu_count = _backend_devices("actor", config.actor.backend, "fsdp")
if gen_gpu_count + actor_gpu_count != gpu_count:
    raise SystemExit(
        f"Config splits {actor_gpu_count} actor + {gen_gpu_count} generation GPUs "
        f"= {actor_gpu_count + gen_gpu_count}, but the launcher was asked for "
        f"{gpu_count}. Fix the allocation file or the GPU count."
    )

if int(config.cluster.n_nodes) != 1:
    raise SystemExit(
        f"Offline launcher requires cluster.n_nodes=1, got {config.cluster.n_nodes}."
    )
if int(config.cluster.n_gpus_per_node) != gpu_count:
    raise SystemExit(
        f"Config resolves cluster.n_gpus_per_node={config.cluster.n_gpus_per_node}, "
        f"but launcher requested {gpu_count}."
    )
if config.scheduler.type != "local":
    raise SystemExit(
        f"Offline launcher requires scheduler.type=local, got {config.scheduler.type}."
    )
expected_actor_backend = f"fsdp:d{actor_gpu_count}p1t1"
if config.actor.scheduling_strategy.type != "separation":
    raise SystemExit("Actor scheduling_strategy must be separation.")
if config.rollout.scheduling_strategy.type != "separation":
    raise SystemExit("Rollout scheduling_strategy must be separation.")
if config.critic is not None or config.teacher is not None:
    raise SystemExit("Offline launcher does not allow critic or PPO teacher engines.")
if config.ref is not None:
    ref_strategy = config.ref.scheduling_strategy
    if (
        config.ref.backend != expected_actor_backend
        or ref_strategy.type != "colocation"
        or ref_strategy.target != "actor"
    ):
        raise SystemExit(
            "Reference engine must use the actor backend and colocate with actor."
        )
expected_workflow = "examples.tutor.workflow.TutorAgentWorkflow"
if config.workflow != expected_workflow or config.eval_workflow != expected_workflow:
    raise SystemExit(
        "Unexpected train/eval workflow for strict offline launch: "
        f"{config.workflow}, {config.eval_workflow}."
    )
wandb_mode = config.stats_logger.wandb.mode
swanlab_mode = config.stats_logger.swanlab.mode
trackio = config.stats_logger.trackio
if wandb_mode not in {"offline", "disabled"}:
    raise SystemExit(f"W&B must be offline or disabled, got {wandb_mode}.")
if swanlab_mode not in {"local", "offline", "disabled"}:
    raise SystemExit(f"SwanLab must be local/offline/disabled, got {swanlab_mode}.")
if trackio.mode not in {"local", "disabled"} or trackio.space_id is not None:
    raise SystemExit(
        "Trackio must be local/disabled with no remote space_id for offline launch."
    )
if config.auxiliary_model.mode != "self":
    raise SystemExit("Offline launcher requires auxiliary_model.mode=self.")
students = list(config.student_models)
if not students:
    raise SystemExit("Offline launcher requires at least one student profile.")
for index, student in enumerate(students):
    if student.base_url.rstrip("/") != student_url.rstrip("/"):
        raise SystemExit(
            f"student_models[{index}] endpoint must be local {student_url}, "
            f"got {student.base_url}."
        )
    parsed = urlparse(student.base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit(
            f"Refusing non-local student endpoint: {student.base_url}"
        )
    if student.model != served_model:
        raise SystemExit(
            f"student_models[{index}].model={student.model!r} does not match "
            f"STUDENT_MODEL={served_model!r}."
        )

student_concurrency_caps = [int(student.max_concurrent_calls) for student in students]
if any(cap <= 0 for cap in student_concurrency_caps):
    raise SystemExit(
        f"Student concurrency caps must be positive: {student_concurrency_caps}."
    )
total_student_concurrency = sum(student_concurrency_caps)
first_student = students[0]

teacher_fraction = float(config.sglang.mem_fraction_static)

print(
    json.dumps(
        {
            "config": config_path,
            "experiment_name": config.experiment_name,
            "trial_name": config.trial_name,
            "gpus": gpu_count,
            "generation_gpus": gen_gpu_count,
            "actor_gpus": actor_gpu_count,
            "actor_backend": config.actor.backend,
            "rollout_backend": config.rollout.backend,
            "auxiliary": "self (rollout base model, LoRA disabled)",
            "student": first_student.model,
            "student_profiles": len(students),
            "student_url": first_student.base_url,
            "rollout_mem_fraction_static_from_config": teacher_fraction,
            "rollout_concurrency": config.rollout.max_concurrent_rollouts,
            "student_client_concurrency_per_profile": sorted(
                set(student_concurrency_caps)
            ),
            "student_client_concurrency_total_cap": total_student_concurrency,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
)"
printf '%s\n' "$CONFIG_SUMMARY"

GEN_GPU_COUNT="$(printf '%s\n' "$CONFIG_SUMMARY" \
  | sed -n 's/^[[:space:]]*"generation_gpus":[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
ACTOR_GPU_COUNT="$(printf '%s\n' "$CONFIG_SUMMARY" \
  | sed -n 's/^[[:space:]]*"actor_gpus":[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
if [[ ! "$GEN_GPU_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$ACTOR_GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Could not read the GPU split out of the config summary above.\n' >&2
  exit 1
fi

# WHICH ids go to which role is not this script's choice -- it has to match what
# AReaL's scheduler does, or the student server reserves memory on a card the
# rollout engine is not on. _allocate_gpus hands ids out with a monotonic counter
# in role-creation order and the actor role is created before the rollout role,
# so the actor takes the leading ACTOR_GPU_COUNT ids and the rollout engine the
# trailing GEN_GPU_COUNT. The student is pinned to that same tail on purpose:
# sglang's mem_fraction_static and the student's have to land on one card
# together, or each of them collides with the actor instead of with each other.
ACTOR_GPU_IDS=("${SELECTED_GPU_IDS[@]:0:ACTOR_GPU_COUNT}")
ROLLOUT_GPU_IDS=("${SELECTED_GPU_IDS[@]:ACTOR_GPU_COUNT:GEN_GPU_COUNT}")
ACTOR_GPU_SPEC="$(join_by_comma "${ACTOR_GPU_IDS[@]}")"
ROLLOUT_GPU_SPEC="$(join_by_comma "${ROLLOUT_GPU_IDS[@]}")"

if [[ "$(basename "$(dirname "$ROOT_DIR")")" == "AReaL.worktrees" ]]; then
  PROJECT_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
else
  PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
fi
LOCAL_MODEL_LOG_ROOT="${LOCAL_MODEL_LOG_ROOT:-$PROJECT_ROOT/output/tutor/local_models}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
STUDENT_SERVER_LOG="$LOCAL_MODEL_LOG_ROOT/student-${GPU_COUNT}gpu-${stamp}.log"

STUDENT_CMD=(
  "$PYTHON" -m sglang.launch_server
  --model-path "$STUDENT_MODEL_PATH"
  --served-model-name "$STUDENT_MODEL"
  --host "$STUDENT_HOST"
  --port "$STUDENT_PORT"
  --tp-size 1
  --dp-size "$GEN_GPU_COUNT"
  --load-balance-method round_robin
  --context-length "$STUDENT_CONTEXT_LENGTH"
  --mem-fraction-static "$STUDENT_MEM_FRACTION_STATIC"
)

# Intentionally no launcher-generated Hydra overrides: YAML plus only the
# explicitly supplied trailing arguments controls training.
TRAIN_CMD=("$PYTHON" examples/tutor/train.py --config "$CONFIG" "${USER_OVERRIDES[@]}")

printf 'Python runtime: trainer=%s, workers=%s\n' "$PYTHON" "$(command -v python3)"
printf 'GPU plan: actor=[%s], rollout+student=[%s]\n' "$ACTOR_GPU_SPEC" "$ROLLOUT_GPU_SPEC"
printf 'Student endpoint: %s\n' "$STUDENT_BASE_URL"
printf 'Student log: %s\n' "$STUDENT_SERVER_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY RUN: student command:\n  CUDA_VISIBLE_DEVICES=%q setsid ' \
    "$ROLLOUT_GPU_SPEC"
  printf '%q ' "${STUDENT_CMD[@]}"
  printf '\nDRY RUN: training command:\n  CUDA_VISIBLE_DEVICES=%q setsid ' \
    "$SELECTED_GPU_SPEC"
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

STUDENT_SERVER_PID=""
TRAIN_PID=""
OWNS_STUDENT_SERVER=0
OWNS_TRAIN=0

process_is_running() {
  local pid=$1 state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
  state="${state//[[:space:]]/}"
  [[ -n "$state" && "$state" != Z* ]]
}

session_has_live_processes() {
  local session_id=$1 state
  while read -r state; do
    state="${state//[[:space:]]/}"
    if [[ -n "$state" && "$state" != Z* ]]; then
      return 0
    fi
  done < <(ps -o stat= --sid "$session_id" 2>/dev/null || true)
  return 1
}

stop_process_group() {
  local label=$1 pid=$2
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    printf 'Stopping %s process group (pgid %s).\n' "$label" "$pid"
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {1..50}; do
      if ! session_has_live_processes "$pid"; then
        break
      fi
      sleep 0.1
    done
    if session_has_live_processes "$pid"; then
      printf '%s did not stop after 5 seconds; sending group-wide SIGKILL.\n' \
        "$label" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$OWNS_TRAIN" == "1" ]]; then
    stop_process_group training "$TRAIN_PID"
  fi
  if [[ "$OWNS_STUDENT_SERVER" == "1" ]]; then
    stop_process_group 'local student server' "$STUDENT_SERVER_PID"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
  local pid=$1
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! process_is_running "$pid"; then
      printf 'Student server exited before becoming ready. Last log lines:\n' >&2
      tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
      return 1
    fi
    if curl --silent --show-error --fail --max-time 3 \
        "$STUDENT_BASE_URL/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  printf 'Student server was not ready after %s seconds. Last log lines:\n' \
    "$SERVER_READY_TIMEOUT" >&2
  tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
  return 1
}

if "$PYTHON" -B - "$STUDENT_HOST" "$STUDENT_PORT" <<'PY'
import socket
import sys

try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
        pass
except OSError:
    raise SystemExit(1)
PY
then
  printf 'Port %s:%s is already occupied; refusing to reuse an unverified server.\n' \
    "$STUDENT_HOST" "$STUDENT_PORT" >&2
  exit 1
fi

mkdir -p "$LOCAL_MODEL_LOG_ROOT"
printf 'Starting Qwen3-1.7B student with DP=%s on GPUs [%s].\n' \
  "$GEN_GPU_COUNT" "$ROLLOUT_GPU_SPEC"
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU_SPEC" setsid "${STUDENT_CMD[@]}" \
  >"$STUDENT_SERVER_LOG" 2>&1 &
STUDENT_SERVER_PID=$!
OWNS_STUDENT_SERVER=1
wait_ready "$STUDENT_SERVER_PID"

# /models only proves the HTTP frontend is alive. Force one real decode before
# claiming GPUs for the trainer, so model-load or DP failures surface immediately.
if ! "$PYTHON" -B - "$STUDENT_BASE_URL" "$STUDENT_MODEL" \
    "$STUDENT_PROBE_TIMEOUT" <<'PY'
import json
import sys
import urllib.request

base_url, expected_model, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])


def get_json(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base_url + path,
        data=body,
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


models = get_json("/models")
model_ids = [item.get("id") for item in models.get("data", [])]
if model_ids != [expected_model]:
    raise SystemExit(
        f"Expected exactly model {expected_model!r} from /models, got {model_ids!r}."
    )

completion = get_json(
    "/chat/completions",
    {
        "model": expected_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "temperature": 0,
        "top_p": 1,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 2,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
if completion.get("model") != expected_model:
    raise SystemExit(
        f"Decode response model mismatch: {completion.get('model')!r}."
    )
choices = completion.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit(f"Decode response has no choices: {completion!r}")
message = choices[0].get("message", {})
if not isinstance(message.get("content"), str):
    raise SystemExit(f"Decode response has no text content: {completion!r}")
print(f"Local student decode probe passed for {expected_model}.")
PY
then
  printf 'Student decode probe failed. Last log lines:\n' >&2
  tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
  exit 1
fi

printf 'Student server is healthy; starting %s-GPU tutor training.\n' "$GPU_COUNT"
CUDA_VISIBLE_DEVICES="$SELECTED_GPU_SPEC" setsid "${TRAIN_CMD[@]}" &
TRAIN_PID=$!
OWNS_TRAIN=1

FINISHED_PID=""
if wait -n -p FINISHED_PID "$TRAIN_PID" "$STUDENT_SERVER_PID"; then
  FINISHED_STATUS=0
else
  FINISHED_STATUS=$?
fi

if [[ "$FINISHED_PID" == "$TRAIN_PID" ]]; then
  exit "$FINISHED_STATUS"
fi

printf 'Local student server exited while training was still running.\n' >&2
tail -n 120 "$STUDENT_SERVER_LOG" >&2 || true
exit 1
