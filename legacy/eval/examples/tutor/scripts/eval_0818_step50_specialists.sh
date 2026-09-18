#!/usr/bin/env bash
set -Eeuo pipefail

# Offline 4-teacher x 8-student evaluation for saved 0818 specialist checkpoints.
#
# Each teacher/student cell runs in its own evaluate_api_teacher.py process. This
# is required for prompt fidelity: TutorAgentWorkflow's mask note is run-level, so
# putting masked and unmasked students in one workflow would add the note to the
# original student even though its singleton training/eval prompt did not have it.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0818_step50_specialists.sh \
    [preflight|smoke|benchmark|screen|full|analyze]

Examples:
  # Read-only: resolve checkpoints, load the real test split, and audit prompts.
  bash examples/tutor/scripts/eval_0818_step50_specialists.sh preflight

  # Fail-closed gate: local fresh-session regression, semantic/liveness checks,
  # then one seeded test item in a balanced 8-cell prompt/adapter cover.
  CUDA_VISIBLE_DEVICES=0,1 bash \
    examples/tutor/scripts/eval_0818_step50_specialists.sh smoke

  # One code cell, same 16 eval rows at concurrency 3, 6, 12, and 16.
  CUDA_VISIBLE_DEVICES=0,1 bash \
    examples/tutor/scripts/eval_0818_step50_specialists.sh benchmark

  # Seeded 128-item screen. Two GPUs use one server pair; four use two pairs.
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
    examples/tutor/scripts/eval_0818_step50_specialists.sh screen

  # All filtered test items (currently 528).
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash \
    examples/tutor/scripts/eval_0818_step50_specialists.sh full

Phases:
  preflight  Default. No output directory, model server, or GPU job is created.
  smoke      Complete pre-full gate, then 1 item in 8 balanced cells.
  benchmark  Time one code cell at concurrency 3, 6, 12, and 16 on two GPUs.
  screen     128 seeded test items x 4 teachers x 8 students.
  full       Every filtered test item x 4 teachers x 8 students.
  analyze    Rebuild the 4x4 ID and OOD summaries in EVAL_RUN_DIR.

Important environment variables:
  EVAL_RUN_DIR                Reuse/select an output root.
  GLOBAL_STEP_INDEX           Checkpoint directory index; 49 means completed step 50.
  ADAPTER_ORIGINAL            Explicit selected-step adapter directory.
  ADAPTER_STUDENT_FADE        Explicit selected-step adapter directory.
  ADAPTER_TEACHER_FADE        Explicit selected-step adapter directory.
  ADAPTER_LONG_DROP           Explicit selected-step adapter directory.
  EVAL_MAX_SAMPLES            Override phase sample count; 0 means all.
  EVAL_CONCURRENCY            Episode concurrency per two-GPU server pair (default 3).
  SERVER_MAX_RUNNING_REQUESTS SGLang request cap per endpoint (default 160).
  CALLER_MAX_CONCURRENT       API caller semaphore per endpoint (default 160).
  EPISODE_ERROR_RETRIES       Extra whole-episode retries after infra errors (default 3).
  EPISODE_RETRY_BACKOFF_SECONDS Initial retry delay; doubles each time (default 1).
  EPISODE_TIMEOUT_SECONDS     Hard timeout for one complete episode (default 300).
  CELL_WALL_TIMEOUT_SECONDS   Hard timeout for one evaluator process (default 3600).
  CELL_PROCESS_RESTARTS       Resume a timed-out/failed cell this many times (default 3).
  BENCHMARK_CONCURRENCIES     Comma list for benchmark (default 3,6,12,16).
  GENERALIZE_REPLAYS          Must remain 8 for standard 0818 evaluation.
  BASE_PORT                   First teacher port (default 33000).
  SAVE_TRACES                 all, errors, or none; smoke requires all.
  SKIP_LIVENESS=1             Skip LoRA liveness; rejected by smoke.

The teacher is Qwen3-8B with the selected LoRA. The student endpoint is
Qwen3-1.7B. "code" is the Qwen3-1.7B CodeAct/Python behavior, not a Codex model.
EOF
}

PHASE="${1:-preflight}"
case "$PHASE" in
  -h|--help)
    usage
    exit 0
    ;;
  preflight|smoke|benchmark|screen|full|analyze) ;;
  *)
    printf 'Unknown phase: %s\n' "$PHASE" >&2
    usage >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
EVAL_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/pilot/eval-all-students.yaml"
BASE_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0818/base/default.yaml"
SINGLE_CONFIG_ROOT="$ROOT_DIR/examples/tutor/configs/math/0818/8gpu/single"
EVALUATOR="$ROOT_DIR/examples/tutor/scripts/evaluate_api_teacher.py"
ANALYZER="$ROOT_DIR/examples/tutor/scripts/analyze_information_specialist_matrix.py"
SMOKE_VALIDATOR="$ROOT_DIR/examples/tutor/scripts/validate_information_specialist_smoke.py"
RETEST_REGRESSION="$ROOT_DIR/tests/test_tutor_code_retest.py"
TRAIN_LAUNCHER="$ROOT_DIR/examples/tutor/run_offline.sh"

for required in \
  "$PYTHON" "$EVAL_CONFIG" "$BASE_CONFIG" "$EVALUATOR" "$ANALYZER" \
  "$SMOKE_VALIDATOR" "$RETEST_REGRESSION" "$TRAIN_LAUNCHER"; do
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
export TUTOR_QWEN3_8B_BASE_URL="${TUTOR_QWEN3_8B_BASE_URL:-http://127.0.0.1:1/v1}"
export TUTOR_QWEN3_1_7B_BASE_URL="${TUTOR_QWEN3_1_7B_BASE_URL:-http://127.0.0.1:2/v1}"

GLOBAL_STEP_INDEX="${GLOBAL_STEP_INDEX:-49}"
if [[ ! "$GLOBAL_STEP_INDEX" =~ ^[0-9]+$ ]]; then
  printf 'GLOBAL_STEP_INDEX must be a non-negative integer; got %q.\n' \
    "$GLOBAL_STEP_INDEX" >&2
  exit 2
fi
COMPLETED_TRAIN_STEP=$((GLOBAL_STEP_INDEX + 1))

case "$PHASE" in
  smoke) DEFAULT_MAX_SAMPLES=1 ;;
  benchmark) DEFAULT_MAX_SAMPLES=16 ;;
  preflight|screen) DEFAULT_MAX_SAMPLES=128 ;;
  full|analyze) DEFAULT_MAX_SAMPLES=0 ;;
esac
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-$DEFAULT_MAX_SAMPLES}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-3}"
GENERALIZE_REPLAYS="${GENERALIZE_REPLAYS:-8}"
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-160}"
CALLER_MAX_CONCURRENT="${CALLER_MAX_CONCURRENT:-160}"
EPISODE_ERROR_RETRIES="${EPISODE_ERROR_RETRIES:-3}"
EPISODE_RETRY_BACKOFF_SECONDS="${EPISODE_RETRY_BACKOFF_SECONDS:-1}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-300}"
CELL_WALL_TIMEOUT_SECONDS="${CELL_WALL_TIMEOUT_SECONDS:-3600}"
CELL_PROCESS_RESTARTS="${CELL_PROCESS_RESTARTS:-3}"
BENCHMARK_CONCURRENCIES="${BENCHMARK_CONCURRENCIES:-3,6,12,16}"
BASE_PORT="${BASE_PORT:-33000}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SKIP_LIVENESS="${SKIP_LIVENESS:-0}"
if [[ -n "${SAVE_TRACES:-}" ]]; then
  SAVE_TRACES="$SAVE_TRACES"
elif [[ "$PHASE" == "smoke" ]]; then
  SAVE_TRACES=all
elif [[ "$PHASE" == "benchmark" ]]; then
  SAVE_TRACES=none
else
  SAVE_TRACES=errors
fi

for integer_name in EVAL_MAX_SAMPLES EVAL_CONCURRENCY GENERALIZE_REPLAYS \
  SERVER_MAX_RUNNING_REQUESTS CALLER_MAX_CONCURRENT BASE_PORT \
  SERVER_READY_TIMEOUT EPISODE_ERROR_RETRIES EPISODE_RETRY_BACKOFF_SECONDS; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer; got %q.\n' \
      "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
for integer_name in EPISODE_TIMEOUT_SECONDS CELL_WALL_TIMEOUT_SECONDS \
  CELL_PROCESS_RESTARTS; do
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
if [[ "$GENERALIZE_REPLAYS" != "8" ]]; then
  printf 'GENERALIZE_REPLAYS must be 8 to match the 0818 standard; got %s.\n' \
    "$GENERALIZE_REPLAYS" >&2
  exit 2
fi
if (( EVAL_CONCURRENCY * GENERALIZE_REPLAYS > CALLER_MAX_CONCURRENT || \
      EVAL_CONCURRENCY * GENERALIZE_REPLAYS > SERVER_MAX_RUNNING_REQUESTS )); then
  printf 'EVAL_CONCURRENCY x GENERALIZE_REPLAYS exceeds a configured request cap: %s x %s, caller=%s, server=%s.\n' \
    "$EVAL_CONCURRENCY" "$GENERALIZE_REPLAYS" "$CALLER_MAX_CONCURRENT" \
    "$SERVER_MAX_RUNNING_REQUESTS" >&2
  exit 2
fi
IFS=',' read -r -a BENCHMARK_CONCURRENCY_VALUES <<<"$BENCHMARK_CONCURRENCIES"
if [[ "$PHASE" == "benchmark" ]]; then
  if [[ "${#BENCHMARK_CONCURRENCY_VALUES[@]}" == "0" ]]; then
    printf 'BENCHMARK_CONCURRENCIES must not be empty.\n' >&2
    exit 2
  fi
  for benchmark_concurrency in "${BENCHMARK_CONCURRENCY_VALUES[@]}"; do
    if [[ ! "$benchmark_concurrency" =~ ^[1-9][0-9]*$ ]] || \
        (( benchmark_concurrency * GENERALIZE_REPLAYS > CALLER_MAX_CONCURRENT )) || \
        (( benchmark_concurrency * GENERALIZE_REPLAYS > SERVER_MAX_RUNNING_REQUESTS )); then
      printf 'Invalid benchmark concurrency %q for replay=%s, caller=%s, server=%s.\n' \
        "$benchmark_concurrency" "$GENERALIZE_REPLAYS" \
        "$CALLER_MAX_CONCURRENT" "$SERVER_MAX_RUNNING_REQUESTS" >&2
      exit 2
    fi
  done
fi
if (( BASE_PORT < 1 || BASE_PORT + 11 > 65535 )); then
  printf 'BASE_PORT leaves the valid TCP port range: %s.\n' "$BASE_PORT" >&2
  exit 2
fi
case "$SAVE_TRACES" in all|errors|none) ;;
  *) printf 'SAVE_TRACES must be all, errors, or none.\n' >&2; exit 2 ;;
esac
case "$SKIP_LIVENESS" in 0|1) ;;
  *) printf 'SKIP_LIVENESS must be 0 or 1.\n' >&2; exit 2 ;;
esac
if [[ "$PHASE" == "smoke" ]]; then
  if (( EVAL_MAX_SAMPLES != 1 )); then
    printf 'smoke requires EVAL_MAX_SAMPLES=1; got %s.\n' \
      "$EVAL_MAX_SAMPLES" >&2
    exit 2
  fi
  if [[ "$SAVE_TRACES" != "all" ]]; then
    printf 'smoke requires SAVE_TRACES=all for trace-completeness checks.\n' >&2
    exit 2
  fi
  if [[ "$SKIP_LIVENESS" != "0" ]]; then
    printf 'smoke refuses SKIP_LIVENESS=1; every LoRA must be exercised.\n' >&2
    exit 2
  fi
fi

mapfile -t BASE_SETTINGS < <(
  "$PYTHON" -B - "$BASE_CONFIG" "$TRAIN_LAUNCHER" <<'PY'
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

base_path, launcher_path = map(Path, sys.argv[1:])
base = OmegaConf.load(base_path)
print(base.cluster.fileroot)
print(base.actor.path)
text = launcher_path.read_text(encoding="utf-8")
match = re.search(
    r'^STUDENT_MODEL_PATH="\$\{STUDENT_MODEL_PATH:-(.+)\}"$',
    text,
    flags=re.MULTILINE,
)
if match is None:
    raise SystemExit("Could not discover STUDENT_MODEL_PATH from run_offline.sh")
print(match.group(1))
PY
)
if [[ "${#BASE_SETTINGS[@]}" != "3" ]]; then
  printf 'Could not resolve fileroot and model paths from the current configs.\n' >&2
  exit 1
fi
TUTOR_FILEROOT="${TUTOR_FILEROOT:-${BASE_SETTINGS[0]}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${BASE_SETTINGS[1]}}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${BASE_SETTINGS[2]}}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"
CHECKPOINT_ROOT="$TUTOR_FILEROOT/checkpoints/$(id -un)/tutor-math-baseline"

MECHANISMS=(original student_fade teacher_fade long_drop)
STUDENTS=(
  qwen3-1.7b-text-original
  qwen3-1.7b-text-student_fade
  qwen3-1.7b-text-teacher_fade
  qwen3-1.7b-text-long_drop
  qwen3-1.7b-code-original
  qwen3-1.7b-code-student_fade
  qwen3-1.7b-code-teacher_fade
  qwen3-1.7b-code-long_drop
)
declare -A ADAPTERS=()

discover_adapter() {
  local label=$1 variable explicit trial_label candidate
  variable="ADAPTER_${label^^}"
  explicit="${!variable:-}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$(readlink -f "$explicit")"
    return
  fi
  trial_label="${label//_/-}"
  candidate="$(
    find "$CHECKPOINT_ROOT" -type f \
      -path "*0818-text-${trial_label}-8gpu/default/*globalstep${GLOBAL_STEP_INDEX}/adapter_model.safetensors" \
      -printf '%T@ %h\n' 2>/dev/null |
      sort -nr |
      head -1 |
      cut -d' ' -f2-
  )"
  if [[ -z "$candidate" ]]; then
    printf 'No globalstep%s adapter found for %s under %s.\n' \
      "$GLOBAL_STEP_INDEX" "$label" "$CHECKPOINT_ROOT" >&2
    return 1
  fi
  readlink -f "$candidate"
}

for mechanism in "${MECHANISMS[@]}"; do
  ADAPTERS["$mechanism"]="$(discover_adapter "$mechanism")"
  for needed in adapter_model.safetensors adapter_config.json; do
    if [[ ! -f "${ADAPTERS[$mechanism]}/$needed" ]]; then
      printf 'Incomplete %s adapter: %s\n' \
        "$mechanism" "${ADAPTERS[$mechanism]}" >&2
      exit 1
    fi
  done
done

run_semantic_preflight() {
  "$PYTHON" -B - \
    "$EVAL_CONFIG" "$SINGLE_CONFIG_ROOT" "$GENERALIZE_REPLAYS" \
    "$EVAL_MAX_SAMPLES" "$CALLER_MAX_CONCURRENT" "$TEACHER_MODEL_PATH" \
    "$GLOBAL_STEP_INDEX" \
    "${ADAPTERS[original]}" "${ADAPTERS[student_fade]}" \
    "${ADAPTERS[teacher_fade]}" "${ADAPTERS[long_drop]}" <<'PY'
import copy
import hashlib
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from examples.tutor import train as tutor_train
from examples.tutor.core.attention_mask import is_identity, normalize_mask
from examples.tutor.prompts import FREE_CHAT_STUDENT_MASK_NOTE
from examples.tutor.scripts.evaluate_api_teacher import (
    dataset_sha256,
    load_experiment_config,
    load_hf_tokenizer,
    prepare_test_dataset,
)
from examples.tutor.workflow import TutorAgentWorkflow

(
    eval_path_raw,
    singleton_root_raw,
    replays_raw,
    max_samples_raw,
    caller_max_concurrent_raw,
    teacher_model_path_raw,
    global_step_index_raw,
    *adapter_paths_raw,
) = sys.argv[1:]
eval_path = Path(eval_path_raw)
singleton_root = Path(singleton_root_raw)
replays = int(replays_raw)
max_samples = int(max_samples_raw)
caller_max_concurrent = int(caller_max_concurrent_raw)
teacher_model_path = Path(teacher_model_path_raw)
global_step_index = int(global_step_index_raw)
mechanisms = ("original", "student_fade", "teacher_fade", "long_drop")
behaviors = ("text", "code")
expected_names = [
    f"qwen3-1.7b-{behavior}-{mechanism}"
    for behavior in behaviors
    for mechanism in mechanisms
]
overrides = [
    f"student_generalize.replays={replays}",
    f"auxiliary_model.max_concurrent_calls={caller_max_concurrent}",
    f"student_axes.0.template.max_concurrent_calls={caller_max_concurrent}",
]
if max_samples > 0:
    overrides.append(f"+evaluator.max_samples={max_samples}")
config, pool = load_experiment_config(str(eval_path), overrides)
tutor_train._apply_eval_average_rollouts(config)

names = [str(student["name"]) for student in pool]
if names != expected_names:
    raise SystemExit(f"Eval pool order/content drifted: {names!r}")
if config.auxiliary_model.mode != "self":
    raise SystemExit("Expected the 0818 source auxiliary_model.mode to be 'self'.")
if config.student_generalize.replays != 8:
    raise SystemExit("The effective standard evaluation must use 8 replays.")
if not config.student_generalize.retest_original:
    raise SystemExit("The standard original-problem re-test is disabled.")
if config.teacher_private_visibility:
    raise SystemExit("Teacher-private student labels must stay disabled.")
if any(config.prompt_pool.student_eval_paths.values()):
    raise SystemExit("Student eval persona pools are non-empty; this matrix expects none.")
expected_evaluator = {
    "teacher_pre_enabled": True,
    "teacher_pre_verify": False,
    "leak_terminate": False,
    "format_terminate": False,
    "average_rollouts": 1,
}
for key, expected in expected_evaluator.items():
    actual = getattr(config.evaluator, key)
    if actual != expected:
        raise SystemExit(
            f"Evaluator {key} drifted: expected {expected!r}, got {actual!r}."
        )

def plain(value):
    if is_dataclass(value):
        value = asdict(value)
    elif not isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
        value = OmegaConf.to_container(OmegaConf.structured(value), resolve=True)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value

def scrub_student(student):
    result = copy.deepcopy(dict(student))
    for key in ("weight", "base_url", "api_key", "max_concurrent_calls"):
        result.pop(key, None)
    params = result.get("request_params") or {}
    params.pop("extra_headers", None)
    return plain(result)

relevant_config_fields = (
    "dataset_type",
    "answer_scorer",
    "max_turns",
    "enable_thinking",
    "leak_handling_mode",
    "format_handling_mode",
    "teacher_show_ground_truth",
    "teacher_anti_leak_instruction_enabled",
    "teacher_private_visibility",
    "teacher_history_tags",
    "free_chat",
    "teacher_pre",
    "student_generalize",
    "auxiliary_model",
    "valid_dataset",
    "reward",
    "prompt_pool",
)
prompts = {}
for behavior in behaviors:
    for mechanism in mechanisms:
        name = f"qwen3-1.7b-{behavior}-{mechanism}"
        split_dir = "ID" if behavior == "text" else "OOD"
        native_path = singleton_root / split_dir / (
            f"{behavior}-{mechanism.replace('_', '-')}.yaml"
        )
        native_config, native_students = load_experiment_config(str(native_path), [])
        tutor_train._apply_eval_average_rollouts(native_config)
        if len(native_students) != 1 or native_students[0]["name"] != name:
            raise SystemExit(f"{native_path}: expected the singleton {name}.")
        selected = next(student for student in pool if student["name"] == name)
        if scrub_student(selected) != scrub_student(native_students[0]):
            raise SystemExit(
                f"{name}: eval-pool student settings differ from its 8-GPU singleton."
            )
        for field in relevant_config_fields:
            left = plain(getattr(config, field))
            right = plain(getattr(native_config, field))
            if field == "auxiliary_model":
                left.pop("max_concurrent_calls", None)
                right.pop("max_concurrent_calls", None)
            if left != right:
                raise SystemExit(
                    f"{name}: non-allocation config field {field!r} drifted."
                )
        # Native validation asks for three independent episode repeats. The quick
        # matrix intentionally uses one; compare the generation distribution for
        # each episode after removing only that repeat-count encoding.
        if plain(config.eval_gconfig.new(n_samples=1)) != plain(
            native_config.eval_gconfig.new(n_samples=1)
        ):
            raise SystemExit(f"{name}: per-attempt eval generation settings drifted.")

        mask = normalize_mask(selected.get("mask"))
        probe = object.__new__(TutorAgentWorkflow)
        probe.free_chat_transfer_prompts = bool(
            getattr(config.free_chat, "transfer_prompts", False)
        )
        probe.student_mask_active = not is_identity(mask)
        system_prompt, retest_prompt = TutorAgentWorkflow._free_chat_student_prompts(
            probe, behavior
        )
        has_note = FREE_CHAT_STUDENT_MASK_NOTE in system_prompt
        if has_note != (mechanism != "original"):
            raise SystemExit(f"{name}: mask-note presence does not match its mask.")
        prompts[name] = (system_prompt, retest_prompt, has_note)

for behavior in behaviors:
    rows = [prompts[f"qwen3-1.7b-{behavior}-{m}"] for m in mechanisms]
    base_systems = [
        system.replace(FREE_CHAT_STUDENT_MASK_NOTE, "").strip()
        for system, _, _ in rows
    ]
    if len(set(base_systems)) != 1:
        raise SystemExit(f"{behavior}: information masks changed the base prompt family.")
    if len({retest for _, retest, _ in rows}) != 1:
        raise SystemExit(f"{behavior}: information masks changed the re-test template.")
if prompts["qwen3-1.7b-text-original"][0] == prompts["qwen3-1.7b-code-original"][0]:
    raise SystemExit("Text and code students unexpectedly share a system prompt.")
if prompts["qwen3-1.7b-text-original"][1] == prompts["qwen3-1.7b-code-original"][1]:
    raise SystemExit("Text and code students unexpectedly share a re-test prompt.")

# Load the same filtered test split used by evaluate_api_teacher.py, including the
# sidecar generalization eligibility filter. No train item enters this path.
if config.student_generalize.enabled:
    tutor_train._prepare_math_generalization_data(config)
tokenizer = load_hf_tokenizer(config.tokenizer_path)
student_prompts = tutor_train._load_eval_student_prompts(config)
if student_prompts:
    raise SystemExit("Unexpected forced student prompt/persona rows.")
selected_original = [next(student for student in pool if student["name"] == expected_names[0])]
saved_max_samples = config.evaluator.max_samples
config.evaluator.max_samples = None
full_dataset = prepare_test_dataset(
    config,
    selected_original,
    tokenizer=tokenizer,
    limit=0,
    student_prompts=student_prompts,
)
config.evaluator.max_samples = saved_max_samples
runtime_dataset = prepare_test_dataset(
    config,
    selected_original,
    tokenizer=tokenizer,
    limit=0,
    student_prompts=student_prompts,
)
if len(full_dataset) < 1 or len(runtime_dataset) < 1:
    raise SystemExit("The filtered test split is empty.")
if max_samples > 0 and len(runtime_dataset) != min(max_samples, len(full_dataset)):
    raise SystemExit("evaluator.max_samples did not select the expected number of rows.")

adapter_paths = [Path(value) for value in adapter_paths_raw]
if len(adapter_paths) != len(mechanisms):
    raise SystemExit("Expected four adapter paths.")
for mechanism, adapter in zip(mechanisms, adapter_paths, strict=True):
    if adapter.name.rsplit("globalstep", 1)[-1] != str(global_step_index):
        raise SystemExit(
            f"{mechanism}: expected a globalstep{global_step_index} directory, "
            f"got {adapter}."
        )
    manifest = json.loads((adapter / "adapter_config.json").read_text())
    if manifest.get("peft_type") != "LORA" or int(manifest.get("r", 0)) != 16:
        raise SystemExit(f"{mechanism}: incompatible LoRA manifest.")
    if Path(manifest.get("base_model_name_or_path", "")).resolve() != teacher_model_path.resolve():
        raise SystemExit(f"{mechanism}: LoRA base model differs from actor.path.")

print("[preflight] standard evaluation semantics: PASS")
print(f"[preflight] source split=test, filtered rows={len(full_dataset)}")
print(
    f"[preflight] phase rows={len(runtime_dataset)}, seed={config.seed}, "
    f"dataset_sha256={dataset_sha256(runtime_dataset)}"
)
print("[preflight] one evaluator process per teacher/student cell: PASS")
print("[preflight] prompt persona pools: empty")
for behavior in behaviors:
    original = prompts[f"qwen3-1.7b-{behavior}-original"]
    masked = prompts[f"qwen3-1.7b-{behavior}-student_fade"]
    print(
        f"[preflight] {behavior}: system={hashlib.sha256(original[0].encode()).hexdigest()[:12]} "
        f"retest={hashlib.sha256(original[1].encode()).hexdigest()[:12]} "
        f"original_mask_note={original[2]} masked_mask_note={masked[2]}"
    )
print(
    "[preflight] all-pool evaluation would alter both original prompts by adding "
    "the run-level mask note; singleton cells avoid that drift."
)
PY
}

printf '[resolve] completed_step=%s (directory globalstep%s)\n' \
  "$COMPLETED_TRAIN_STEP" "$GLOBAL_STEP_INDEX"
printf '[resolve] fileroot=%s\n' "$TUTOR_FILEROOT"
printf '[resolve] teacher_model=%s\n' "$TEACHER_MODEL_PATH"
printf '[resolve] student_model=%s\n' "$STUDENT_MODEL_PATH"
for mechanism in "${MECHANISMS[@]}"; do
  printf '[resolve] %-12s %s\n' "$mechanism" "${ADAPTERS[$mechanism]}"
done
if [[ "$PHASE" == "smoke" ]]; then
  printf '[smoke] checking fresh code re-test execution before starting GPUs\n'
  "$PYTHON" -B -m pytest -q -p no:cacheprovider "$RETEST_REGRESSION"
  printf '[smoke] fresh code re-test execution: PASS\n'
fi
run_semantic_preflight

STAMP="${EVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${EVAL_RUN_DIR:-$TUTOR_FILEROOT/offline_eval/0818-step${COMPLETED_TRAIN_STEP}/$STAMP}"
MATRIX_ROOT="$RUN_DIR/matrix"

if [[ "$PHASE" == "preflight" ]]; then
  printf '[preflight] no files or GPU jobs were created.\n'
  printf '[preflight] prospective output=%s\n' "$RUN_DIR"
  exit 0
fi
if [[ "$PHASE" == "analyze" ]]; then
  if [[ -z "${EVAL_RUN_DIR:-}" || ! -d "$MATRIX_ROOT/teachers" ]]; then
    printf 'analyze requires EVAL_RUN_DIR containing matrix/teachers.\n' >&2
    exit 1
  fi
  "$PYTHON" -B "$ANALYZER" \
    --eval-root "$MATRIX_ROOT" \
    --no-baseline \
    --output "$MATRIX_ROOT/matrix_summary.json"
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
  printf 'Set CUDA_VISIBLE_DEVICES explicitly to exactly 2 or 4 free GPU ids.\n' >&2
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
  SEEN_GPU["$gpu"]=1
  GPU_IDS+=("$gpu")
done
if [[ "${#GPU_IDS[@]}" != "2" && "${#GPU_IDS[@]}" != "4" ]]; then
  printf 'Expose exactly 2 or 4 GPUs; got %d.\n' "${#GPU_IDS[@]}" >&2
  exit 2
fi
if [[ "$PHASE" == "benchmark" && "${#GPU_IDS[@]}" != "2" ]]; then
  printf 'benchmark requires exactly 2 GPUs so every candidate measures one identical server pair.\n' >&2
  exit 2
fi
PAIR_COUNT=$((${#GPU_IDS[@]} / 2))

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/cells" "$RUN_DIR/progress" "$MATRIX_ROOT"
BARRIER_ROOT="$RUN_DIR/.barriers/${BASHPID}-$(date +%s%N)"
mkdir -p "$BARRIER_ROOT"
"$PYTHON" -B - "$RUN_DIR/manifest.json" "$PHASE" "$EVAL_MAX_SAMPLES" \
  "$EVAL_CONCURRENCY" "$GENERALIZE_REPLAYS" \
  "$SERVER_MAX_RUNNING_REQUESTS" "$CALLER_MAX_CONCURRENT" \
  "$EPISODE_ERROR_RETRIES" "$EPISODE_RETRY_BACKOFF_SECONDS" \
  "$BENCHMARK_CONCURRENCIES" \
  "${ADAPTERS[original]}" "${ADAPTERS[student_fade]}" \
  "${ADAPTERS[teacher_fade]}" "${ADAPTERS[long_drop]}" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

(
    output,
    phase,
    max_samples,
    concurrency,
    replays,
    server_max_running_requests,
    caller_max_concurrent,
    episode_error_retries,
    episode_retry_backoff_seconds,
    benchmark_concurrencies,
    original,
    student_fade,
    teacher_fade,
    long_drop,
) = sys.argv[1:]
payload = {
    "phase": phase,
    "eval_max_samples": int(max_samples),
    "eval_concurrency_per_pair": int(concurrency),
    "student_generalize_replays": int(replays),
    "server_max_running_requests": int(server_max_running_requests),
    "caller_max_concurrent": int(caller_max_concurrent),
    "episode_error_retries": int(episode_error_retries),
    "episode_retry_backoff_seconds": int(episode_retry_backoff_seconds),
    "benchmark_concurrencies": benchmark_concurrencies,
    "schedule": "teacher_then_code_then_text",
    "completed_train_step": 50,
    "checkpoint_directory_index": 49,
    "code_retest_execution": "fresh_session_with_visible_transcript",
    "adapters": {
        "original": original,
        "student_fade": student_fade,
        "teacher_fade": teacher_fade,
        "long_drop": long_drop,
    },
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text())
    previous_without_time = {
        key: value for key, value in previous.items() if key != "created_at"
    }
    if previous_without_time != payload:
        raise SystemExit(f"Existing manifest differs: {path}")
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
  local output=$1 student=$2 adapter=$3
  "$PYTHON" -B - "$output" "$student" "$adapter" \
    "$GENERALIZE_REPLAYS" "$EVAL_MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
student = sys.argv[2]
adapter = str(Path(sys.argv[3]).resolve())
replays = int(sys.argv[4])
expected_rows = int(sys.argv[5])
summary = json.loads((output / "summary.json").read_text())
run_config = json.loads((output / "run_config.json").read_text())["signature"]
if set(summary["modes"]) != {"presolve_on"}:
    raise SystemExit(f"{output}: expected only presolve_on.")
mode = summary["modes"]["presolve_on"]
dataset_rows = int(summary["dataset_rows"])
if expected_rows > 0 and dataset_rows != expected_rows:
    raise SystemExit(f"{output}: expected {expected_rows} rows, got {dataset_rows}.")
expected = dataset_rows
completed = int(mode["completed_attempts"])
error_count = int(mode["error_count"])
pending = summary.get("pending_backfill") or {}
pending_path = Path(str(pending.get("path") or output / "pending_backfill.jsonl"))
pending_records = [
    json.loads(line)
    for line in pending_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
] if pending_path.is_file() else []
pending_keys = {record["key"] for record in pending_records}
pending_count = int(pending.get("count", -1))
checks = {
    "expected_total_attempts": int(summary["expected_total_attempts"]) == expected,
    "recorded_total_attempts": int(summary["recorded_total_attempts"]) == expected,
    "recorded_attempts": int(mode["recorded_attempts"]) == expected,
    "accounted_attempts": completed + error_count == expected,
    "pending_backfill_manifest": pending_count == len(pending_records),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"{output}: health gate failed: {failed}.")
if pending_count:
    print(
        f"[cell-pending] {output}: {pending_count} episode(s) exhausted retries; "
        f"saved in {pending_path} and continuing."
    )

signature_student = run_config["students"]
if len(signature_student) != 1 or signature_student[0]["name"] != student:
    raise SystemExit(f"{output}: evaluator was not singleton for {student}.")
semantics = run_config["test_semantics"]
if int(semantics["max_turns"]) != 10:
    raise SystemExit(f"{output}: max_turns drifted.")
if not semantics["free_chat"]["enabled"]:
    raise SystemExit(f"{output}: free_chat is disabled.")
if semantics["teacher_history_tags"] != "masked":
    raise SystemExit(f"{output}: teacher_history_tags drifted.")
if semantics["leak_handling_mode"] != "reward_only":
    raise SystemExit(f"{output}: deployment leak semantics drifted.")
if semantics["format_handling_mode"] != "continue":
    raise SystemExit(f"{output}: deployment format semantics drifted.")
if int(semantics["student_generalize_replays"]) != replays:
    raise SystemExit(f"{output}: replay count drifted.")
if not semantics["student_generalize_retest_original"]:
    raise SystemExit(f"{output}: original re-test is disabled.")
if set(semantics["generalization_levels"]) != {"original", "original_preleak"}:
    raise SystemExit(f"{output}: generalization levels drifted.")
if "student_prompt_pools" in semantics:
    raise SystemExit(f"{output}: unexpected prompt persona expansion.")
aux = run_config["auxiliary"]
if aux["source_mode"] != "self" or aux["effective_mode"] != "api":
    raise SystemExit(f"{output}: self auxiliary was not bridged to the teacher.")
for caller_name in ("teacher", "auxiliary"):
    actual = run_config[caller_name]["request_params"]["extra_body"]["lora_path"]
    if str(Path(actual).resolve()) != adapter:
        raise SystemExit(f"{output}: {caller_name} uses the wrong LoRA: {actual}.")

latest = {}
for raw in (output / "results.jsonl").read_text().splitlines():
    if raw.strip():
        record = json.loads(raw)
        latest[record["key"]] = record
if len(latest) != expected:
    raise SystemExit(f"{output}: latest result count is {len(latest)}, expected {expected}.")
if not pending_keys.issubset(latest):
    raise SystemExit(f"{output}: pending backfill contains unknown result keys.")
is_code = "-code-" in student
for record in latest.values():
    if record["key"] in pending_keys:
        if record["student_name"] != student:
            raise SystemExit(f"{output}: invalid pending result {record['key']}.")
        continue
    if record.get("error") is not None or record["student_name"] != student:
        raise SystemExit(f"{output}: invalid result {record.get('key')}.")
    levels = record.get("generalization") or {}
    for level in ("original", "original_preleak"):
        result = levels.get(level) or {}
        if result.get("score") is None or int(result.get("replay_count", -1)) != replays:
            raise SystemExit(
                f"{output}: {record['key']} has invalid {level} re-test data."
            )
    if is_code and not isinstance(record.get("code_stats"), dict):
        raise SystemExit(f"{output}: {record['key']} has no code execution stats.")
print(
    f"[cell-ok] {student}: {expected - pending_count} verified episodes, "
    f"{pending_count} pending backfill"
)
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
  local teacher_pid="" student_pid="" telemetry_pid=""

  pair_cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "$teacher_pid" "$student_pid" "$telemetry_pid"; do
      [[ -n "$pid" ]] || continue
      if kill -0 -- "-$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
      elif kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    for pid in "$teacher_pid" "$student_pid" "$telemetry_pid"; do
      [[ -n "$pid" ]] || continue
      wait "$pid" 2>/dev/null || true
    done
    exit "$status"
  }
  trap pair_cleanup EXIT INT TERM

  mkdir -p "$pair_dir/served_adapters"
  local -a served_paths=()
  declare -A served=()
  for mechanism in "${MECHANISMS[@]}"; do
    target="$pair_dir/served_adapters/$mechanism"
    if [[ -L "$target" ]]; then
      if [[ "$(readlink -f "$target")" != "${ADAPTERS[$mechanism]}" ]]; then
        printf 'Existing adapter link points elsewhere: %s\n' "$target" >&2
        return 1
      fi
    elif [[ -e "$target" ]]; then
      printf 'Refusing non-symlink adapter alias: %s\n' "$target" >&2
      return 1
    else
      ln -s "${ADAPTERS[$mechanism]}" "$target"
    fi
    served["$mechanism"]="$(readlink -f "$target")"
    # Keep the alias, not the resolved checkpoint basename: SGLang keys LoRAs
    # by the registration path and all four checkpoint basenames are identical.
    served_paths+=("$target")
  done

  for occupied in "$teacher_url" "$student_url"; do
    if curl --silent --fail --max-time 2 "$occupied/models" >/dev/null 2>&1; then
      printf 'Port already serves an API: %s\n' "$occupied" >&2
      return 1
    fi
  done

  CUDA_VISIBLE_DEVICES="$teacher_gpu" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$TEACHER_MODEL_PATH" \
    --served-model-name "$TEACHER_MODEL" \
    --host 127.0.0.1 --port "$teacher_port" --tp-size 1 \
    --context-length 40960 --mem-fraction-static 0.80 \
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS" --enable-lora \
    --lora-paths "${served_paths[@]}" \
    --max-loras-per-batch 4 --max-loaded-loras 4 \
    >"$teacher_log" 2>&1 &
  teacher_pid=$!

  CUDA_VISIBLE_DEVICES="$student_gpu" setsid "$PYTHON" -m sglang.launch_server \
    --model-path "$STUDENT_MODEL_PATH" \
    --served-model-name "$STUDENT_MODEL" \
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
from pathlib import Path

from examples.tutor.scripts.eval_checkpoints import Checkpoint, verify_adapter_is_live

base_url, model, *paths = sys.argv[1:]
checkpoints = [
    Checkpoint(step=index, path=Path(path))
    for index, path in enumerate(paths)
]
verify_adapter_is_live(checkpoints, base_url, model, "EMPTY", 300.0)
PY
  fi

  barrier_wait() {
    local label=$1 ready index
    mkdir -p "$BARRIER_ROOT/$label"
    : >"$BARRIER_ROOT/$label/pair-$pair_index"
    while true; do
      ready=0
      for ((index = 0; index < PAIR_COUNT; index++)); do
        [[ -f "$BARRIER_ROOT/$label/pair-$index" ]] && ready=$((ready + 1))
      done
      (( ready == PAIR_COUNT )) && return 0
      sleep 1
    done
  }

  run_eval_cell() {
    local teacher=$1 student=$2 concurrency=$3 output=$4 log=$5
    local alias request_params started status cell_try max_cell_tries
    local -a command
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
      --concurrency "$concurrency"
      --episode-error-retries "$EPISODE_ERROR_RETRIES"
      --episode-error-retry-backoff-seconds "$EPISODE_RETRY_BACKOFF_SECONDS"
      --episode-timeout-seconds "$EPISODE_TIMEOUT_SECONDS"
      --retry-diagnostic-failures
      --save-traces "$SAVE_TRACES"
      --output-dir "$output"
    )
    command+=(
      "student_generalize.replays=$GENERALIZE_REPLAYS"
      "auxiliary_model.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
      "student_axes.0.template.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
    )
    if (( EVAL_MAX_SAMPLES > 0 )); then
      command+=("+evaluator.max_samples=$EVAL_MAX_SAMPLES")
    fi
    printf '[cell] pair=%s teacher=%-12s student=%s concurrency=%s\n' \
      "$pair_index" "$teacher" "$student" "$concurrency"
    started=$SECONDS
    : >"$log"
    max_cell_tries=$((CELL_PROCESS_RESTARTS + 1))
    for ((cell_try = 1; cell_try <= max_cell_tries; cell_try++)); do
      local -a attempt_command=("${command[@]}")
      if [[ -f "$output/run_config.json" ]]; then
        attempt_command+=(--resume --allow-evaluator-code-change-on-resume)
      fi
      printf '[cell-process] try=%s/%s wall_timeout=%ss\n' \
        "$cell_try" "$max_cell_tries" "$CELL_WALL_TIMEOUT_SECONDS" >>"$log"
      set +e
      TUTOR_QWEN3_8B_BASE_URL="$teacher_url" \
      TUTOR_QWEN3_1_7B_BASE_URL="$student_url" \
        timeout --signal=TERM --kill-after=30s \
          "${CELL_WALL_TIMEOUT_SECONDS}s" "${attempt_command[@]}" >>"$log" 2>&1
      status=$?
      set -e
      if (( status == 0 )); then
        break
      fi
      if (( cell_try == max_cell_tries )); then
        printf 'Cell failed after %s process tries (last status %s); tail of %s:\n' \
          "$max_cell_tries" "$status" "$log" >&2
        tail -100 "$log" >&2 || true
        return 1
      fi
      printf '[cell-process-retry] status=%s; preserving results and resuming in %ss\n' \
        "$status" "$cell_try" | tee -a "$log" >&2
      sleep "$cell_try"
    done
    LAST_CELL_SECONDS=$((SECONDS - started))
    validate_cell "$output" "$student" "$alias"
  }

  write_teacher_progress() {
    local teacher=$1
    "$PYTHON" -B - "$RUN_DIR" "$teacher" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
teacher = sys.argv[2]
mechanisms = ("original", "student_fade", "teacher_fade", "long_drop")
payload = {"teacher": teacher, "students": {}}
totals = {
    "records": 0,
    "pending_backfill": 0,
    "leak_episodes": 0,
    "leak_turns": 0,
    "original_correct": 0,
    "original_replays": 0,
    "preleak_correct": 0,
    "preleak_replays": 0,
}
for behavior in ("code", "text"):
    for mechanism in mechanisms:
        student = f"qwen3-1.7b-{behavior}-{mechanism}"
        path = run_dir / "cells" / teacher / student / "results.jsonl"
        latest = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                record = json.loads(raw)
                latest[record["key"]] = record
        pending_path = path.parent / "pending_backfill.jsonl"
        pending_keys = {
            json.loads(raw)["key"]
            for raw in pending_path.read_text(encoding="utf-8").splitlines()
            if raw.strip()
        } if pending_path.is_file() else set()
        cell = {key: 0 for key in totals}
        cell["records"] = len(latest)
        cell["pending_backfill"] = len(pending_keys)
        for record in latest.values():
            leaks = int(record.get("leak_count", 0) or 0)
            original = (record.get("generalization") or {}).get("original") or {}
            preleak = (
                (record.get("generalization") or {}).get("original_preleak") or {}
            )
            cell["leak_episodes"] += int(leaks > 0)
            cell["leak_turns"] += leaks
            cell["original_correct"] += int(original.get("replay_correct", 0) or 0)
            cell["original_replays"] += int(original.get("replay_count", 0) or 0)
            cell["preleak_correct"] += int(preleak.get("replay_correct", 0) or 0)
            cell["preleak_replays"] += int(preleak.get("replay_count", 0) or 0)
        payload["students"][student] = cell
        for key, value in cell.items():
            totals[key] += value
payload["totals"] = totals
target = run_dir / "progress" / f"{teacher}.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(
    f"[teacher-done] {teacher}: original "
    f"{totals['original_correct']}/{totals['original_replays']}, preleak "
    f"{totals['preleak_correct']}/{totals['preleak_replays']}, "
    f"leak_episodes={totals['leak_episodes']}, "
    f"pending_backfill={totals['pending_backfill']}; progress={target}"
)
PY
  }

  if [[ "$PHASE" == "benchmark" ]]; then
    printf 'concurrency\trows\tseconds\tepisodes_per_minute\n' >"$RUN_DIR/benchmark.tsv"
    for concurrency in "${BENCHMARK_CONCURRENCY_VALUES[@]}"; do
      output="$RUN_DIR/benchmark/concurrency-$concurrency"
      log="$RUN_DIR/logs/benchmark--concurrency-$concurrency.log"
      nvidia-smi --id="$teacher_gpu,$student_gpu" \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw \
        --format=csv,noheader,nounits --loop=1 \
        >"$RUN_DIR/logs/benchmark--concurrency-$concurrency.gpu.csv" 2>&1 &
      telemetry_pid=$!
      run_eval_cell original qwen3-1.7b-code-original "$concurrency" \
        "$output" "$log"
      kill -TERM "$telemetry_pid" 2>/dev/null || true
      wait "$telemetry_pid" 2>/dev/null || true
      telemetry_pid=""
      "$PYTHON" -B - "$concurrency" "$EVAL_MAX_SAMPLES" \
        "$LAST_CELL_SECONDS" >>"$RUN_DIR/benchmark.tsv" <<'PY'
import sys

concurrency, rows, seconds = map(int, sys.argv[1:])
rate = 60.0 * rows / seconds if seconds else 0.0
print(f"{concurrency}\t{rows}\t{seconds}\t{rate:.3f}")
PY
    done
    return 0
  fi

  # Globally teacher-major. With four GPUs, both pairs split the four mask cells
  # for the same behavior; the barrier completes all code before text, and all
  # eight students before either pair advances to the next teacher.
  for teacher_index in "${!MECHANISMS[@]}"; do
    teacher="${MECHANISMS[$teacher_index]}"
    for behavior in code text; do
      for mechanism_index in "${!MECHANISMS[@]}"; do
        if [[ "$PHASE" == "smoke" ]] && \
            (( mechanism_index != teacher_index )); then
          continue
        fi
        if (( mechanism_index % PAIR_COUNT != pair_index )); then
          continue
        fi
        student="qwen3-1.7b-${behavior}-${MECHANISMS[$mechanism_index]}"
        output="$RUN_DIR/cells/$teacher/$student"
        log="$RUN_DIR/logs/$teacher--$student.log"
        run_eval_cell "$teacher" "$student" "$EVAL_CONCURRENCY" \
          "$output" "$log"
      done
      barrier_wait "teacher-${teacher_index}-${behavior}"
      if (( pair_index == 0 )); then
        printf '[behavior-done] teacher=%s behavior=%s\n' "$teacher" "$behavior"
      fi
    done
    if (( pair_index == 0 )) && [[ "$PHASE" != "smoke" ]]; then
      write_teacher_progress "$teacher"
    fi
    barrier_wait "teacher-${teacher_index}-reported"
  done
)

merge_cells() {
  "$PYTHON" -B - "$RUN_DIR" "$MATRIX_ROOT" "$EVAL_MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
matrix_root = Path(sys.argv[2])
expected_rows = int(sys.argv[3])
teachers = ("original", "student_fade", "teacher_fade", "long_drop")
mechanisms = teachers
pending_backfill = []
for teacher in teachers:
    for split, behavior in (("id", "text"), ("ood", "code")):
        merged = []
        for mechanism in mechanisms:
            student = f"qwen3-1.7b-{behavior}-{mechanism}"
            source = run_dir / "cells" / teacher / student / "results.jsonl"
            latest = {}
            for raw in source.read_text().splitlines():
                if raw.strip():
                    record = json.loads(raw)
                    latest[record["key"]] = record
            cell_pending_path = source.parent / "pending_backfill.jsonl"
            cell_pending_keys = {
                json.loads(raw)["key"]
                for raw in cell_pending_path.read_text(encoding="utf-8").splitlines()
                if raw.strip()
            } if cell_pending_path.is_file() else set()
            if expected_rows > 0 and len(latest) != expected_rows:
                raise SystemExit(
                    f"{source}: {len(latest)} latest records, expected {expected_rows}."
                )
            for key in sorted(latest):
                record = latest[key]
                merged.append(record)
                if key in cell_pending_keys:
                    pending_record = dict(record)
                    pending_record["teacher_specialist"] = teacher
                    pending_record["matrix_split"] = split
                    pending_backfill.append(pending_record)
        target = matrix_root / "teachers" / teacher / split / "results.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in merged)
        )
        temporary.replace(target)
        print(f"[merge] {target}: {len(merged)} records")
pending_path = run_dir / "pending_backfill.jsonl"
pending_temporary = pending_path.with_suffix(".jsonl.tmp")
pending_temporary.write_text(
    "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in pending_backfill
    ),
    encoding="utf-8",
)
pending_temporary.replace(pending_path)
print(f"[backfill] {pending_path}: {len(pending_backfill)} pending episodes")
PY
}

WORKER_PIDS=()
cleanup_workers() {
  local status=$? pid
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "${WORKER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit "$status"
}
trap cleanup_workers EXIT INT TERM

printf '[run] phase=%s pairs=%s rows_per_cell=%s output=%s\n' \
  "$PHASE" "$PAIR_COUNT" "${EVAL_MAX_SAMPLES:-all}" "$RUN_DIR"
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
  printf 'A server pair/cell failed; stopping the other pair. Inspect %s/logs.\n' \
    "$RUN_DIR" >&2
  exit 1
fi
WORKER_PIDS=()
trap - EXIT INT TERM

if [[ "$PHASE" == "benchmark" ]]; then
  printf '[benchmark-done] identical code/original rows; GPU telemetry is under %s/logs\n' \
    "$RUN_DIR"
  column -t -s $'\t' "$RUN_DIR/benchmark.tsv" 2>/dev/null || \
    sed -n '1,20p' "$RUN_DIR/benchmark.tsv"
  printf '[done] concurrency benchmark: %s\n' "$RUN_DIR"
  exit 0
fi

if [[ "$PHASE" == "smoke" ]]; then
  "$PYTHON" -B "$SMOKE_VALIDATOR" \
    --run-dir "$RUN_DIR" \
    --expected-rows "$EVAL_MAX_SAMPLES" \
    --replays "$GENERALIZE_REPLAYS" \
    --diagonal-only \
    --output "$RUN_DIR/smoke_gate.json"
  printf '[smoke-pass] all pre-full gates passed: %s\n' \
    "$RUN_DIR/smoke_gate.json"
  printf '[done] balanced 8-cell smoke: %s\n' "$RUN_DIR"
  exit 0
fi

merge_cells
PENDING_COUNT="$(awk 'NF {count += 1} END {print count + 0}' \
  "$RUN_DIR/pending_backfill.jsonl")"
if (( PENDING_COUNT > 0 )); then
  printf '[done-with-pending] sweep completed with %s episode(s) awaiting backfill: %s\n' \
    "$PENDING_COUNT" "$RUN_DIR/pending_backfill.jsonl"
  printf '[analysis-deferred] rerun this same EVAL_RUN_DIR to retry only pending episodes; the final matrix is intentionally not scored yet.\n'
  exit 0
fi
"$PYTHON" -B "$ANALYZER" \
  --eval-root "$MATRIX_ROOT" \
  --no-baseline \
  --output "$MATRIX_ROOT/matrix_summary.json"
printf '[done] cells, merged matrix, and analysis: %s\n' "$RUN_DIR"
