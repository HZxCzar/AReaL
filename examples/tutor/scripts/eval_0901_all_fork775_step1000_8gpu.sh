#!/usr/bin/env bash
set -Eeuo pipefail

# Full recovered step-demonstration checkpoint x seven students.
# Four independent teacher/student pairs use all eight H200 GPUs.

usage() {
  cat <<'EOF'
Usage:
  bash examples/tutor/scripts/eval_0901_all_fork775_step1000_8gpu.sh [preflight|run|analyze]

Run the complete 1 x 7 matrix on exactly eight GPUs:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash examples/tutor/scripts/eval_0901_all_fork775_step1000_8gpu.sh run

Resume an interrupted evaluation:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  EVAL_RUN_DIR=/path/printed/by/the/first/run \
    bash examples/tutor/scripts/eval_0901_all_fork775_step1000_8gpu.sh run

Rebuild the matrix summary:
  EVAL_RUN_DIR=/path/printed/by/the/run \
    bash examples/tutor/scripts/eval_0901_all_fork775_step1000_8gpu.sh analyze

Useful overrides:
  COMMON_GLOBAL_STEP=999       1000 completed training updates.
  MATRIX_TEACHER_KEYS=step-demonstration   Evaluate only the listed comma-separated teachers.
  EVAL_CONCURRENCY=16          Concurrent episodes per GPU pair.
  BASE_PORT=37000              Teacher ports are BASE_PORT + 0/10/20/30.
  SAVE_TRACES=all              all | errors | none.
  EVAL_RUN_DIR=/explicit/path  Required to resume or analyze an existing run.

The default is the recovered step-demonstration checkpoint globalstep999:
1000 completed training updates, evaluated on the full test split.
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
# shellcheck source=examples/tutor/scripts/eval_process_cleanup.sh
source "$ROOT_DIR/examples/tutor/scripts/eval_process_cleanup.sh"
PYTHON="$ROOT_DIR/.venv/bin/python"
EVAL_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0901/pilot/eval-step-demo-step1000.yaml"
BASE_CONFIG="$ROOT_DIR/examples/tutor/configs/math/0901/base/default.yaml"
EVALUATOR="$ROOT_DIR/examples/tutor/scripts/evaluate_api_teacher_train_aligned.py"
LAUNCHER_SCRIPT="${MATRIX_LAUNCHER_SCRIPT:-examples/tutor/scripts/eval_0901_all_fork775_step1000_8gpu.sh}"
EXPECTED_EXPLAIN_RATIO="${MATRIX_EXPECTED_EXPLAIN_RATIO:-1.0}"
OUTPUT_TAG="${MATRIX_OUTPUT_TAG:-all-fork775-current-gates}"
STRATIFIED_SAMPLES="${MATRIX_STRATIFIED_SAMPLES:-0}"
INCLUDE_NONE_STUDENT="${MATRIX_INCLUDE_NONE_STUDENT:-1}"

for required in "$PYTHON" "$EVAL_CONFIG" "$BASE_CONFIG" "$EVALUATOR"; do
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
# Hydra resolves these while composing the config; run mode replaces them per pair.
export TUTOR_QWEN3_8B_BASE_URL="${TUTOR_QWEN3_8B_BASE_URL:-http://127.0.0.1:1/v1}"
export TUTOR_QWEN3_1_7B_BASE_URL="${TUTOR_QWEN3_1_7B_BASE_URL:-http://127.0.0.1:2/v1}"

write_matrix_summary() {
  local run_dir=$1
  "$PYTHON" -B - "$run_dir" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from examples.tutor.scripts.analyze_0825_personality_eval import _cell_summary

run_dir = Path(sys.argv[1]).resolve()
manifest_path = run_dir / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = int(manifest["dataset"]["rows"])
rows = []
cells = {}
for teacher in manifest["teachers"]:
    teacher_key = str(teacher["key"])
    teacher_cells = {}
    for student in manifest["students"]:
        preference = str(student["preference"])
        cell_dir = run_dir / "cells" / teacher_key / str(student["name"])
        cell = _cell_summary(
            cell_dir,
            preference=preference,
            expected=expected,
        )
        evaluator_summary_path = cell_dir / "summary.json"
        evaluator_mode = {}
        if evaluator_summary_path.is_file():
            evaluator_summary = json.loads(
                evaluator_summary_path.read_text(encoding="utf-8")
            )
            evaluator_mode = (
                (evaluator_summary.get("modes") or {}).get("presolve_on") or {}
            )
        teacher_cells[preference] = cell
        gate = cell["gate"]
        teaching = cell["teaching"]
        diagnostics = cell["diagnostics"]
        rows.append(
            {
                "teacher": teacher_key,
                "student": preference,
                "split": student["split"],
                "progress": cell["progress"],
                "recorded": cell["recorded_episode_count"],
                "errors": cell["error_count"],
                "gate_compliance": gate["micro_compliance"],
                "baseline_accuracy": evaluator_mode.get(
                    "no_teaching_baseline_mean"
                ),
                "taught_success": teaching["taught_success_rate"],
                "retest_accuracy": teaching["original_retest"]["accuracy_on_replays"],
                "improvement": evaluator_mode.get(
                    "improvement_over_no_teaching_baseline_mean"
                ),
                "leak_episodes": diagnostics["leaked_episode_count"],
                "format_error_episodes": diagnostics["format_error_episode_count"],
            }
        )
    cells[teacher_key] = teacher_cells

report = {
    "updated_at": datetime.now(UTC).isoformat(),
    "run_dir": str(run_dir),
    "common_global_step": manifest["common_global_step"],
    "completed_train_steps": manifest["completed_train_steps"],
    "expected_cells": len(manifest["teachers"]) * len(manifest["students"]),
    "expected_episodes": expected * len(manifest["teachers"]) * len(manifest["students"]),
    "recorded_episodes": sum(row["recorded"] for row in rows),
    "complete": all(row["recorded"] >= expected for row in rows),
    "rows": rows,
    "cells": cells,
}
(run_dir / "matrix_summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

def fmt(value):
    return "-" if value is None else f"{float(value):.6f}"

columns = (
    "teacher", "student", "split", "progress", "recorded", "errors",
    "gate_compliance", "baseline_accuracy", "taught_success", "retest_accuracy",
    "improvement", "leak_episodes", "format_error_episodes",
)
lines = ["\t".join(columns)]
for row in rows:
    lines.append(
        "\t".join(
            str(row[column])
            if column in {"teacher", "student", "split", "recorded", "errors",
                          "leak_episodes", "format_error_episodes"}
            else fmt(row[column])
            for column in columns
        )
    )
(run_dir / "matrix_summary.tsv").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print("\n".join(lines))
print(f"[analysis] wrote {run_dir / 'matrix_summary.json'}")
print(f"[analysis] wrote {run_dir / 'matrix_summary.tsv'}")
PY
}

if [[ "$MODE" == "analyze" ]]; then
  if [[ -z "${EVAL_RUN_DIR:-}" || ! -d "$EVAL_RUN_DIR" ]]; then
    printf 'analyze requires an existing EVAL_RUN_DIR.\n' >&2
    exit 2
  fi
  write_matrix_summary "$EVAL_RUN_DIR"
  exit 0
fi

EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
GENERALIZE_REPLAYS=8
SERVER_MAX_RUNNING_REQUESTS="${SERVER_MAX_RUNNING_REQUESTS:-192}"
CALLER_MAX_CONCURRENT="${CALLER_MAX_CONCURRENT:-192}"
EPISODE_ERROR_RETRIES="${EPISODE_ERROR_RETRIES:-3}"
EPISODE_RETRY_BACKOFF_SECONDS="${EPISODE_RETRY_BACKOFF_SECONDS:-1}"
EPISODE_TIMEOUT_SECONDS="${EPISODE_TIMEOUT_SECONDS:-300}"
CELL_WALL_TIMEOUT_SECONDS="${CELL_WALL_TIMEOUT_SECONDS:-86400}"
CELL_PROCESS_RESTARTS="${CELL_PROCESS_RESTARTS:-3}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
BASE_PORT="${BASE_PORT:-37000}"
SAVE_TRACES="${SAVE_TRACES:-all}"
COMMON_GLOBAL_STEP="${COMMON_GLOBAL_STEP:-999}"

for integer_name in EVAL_CONCURRENCY SERVER_MAX_RUNNING_REQUESTS CALLER_MAX_CONCURRENT EPISODE_ERROR_RETRIES EPISODE_RETRY_BACKOFF_SECONDS EPISODE_TIMEOUT_SECONDS CELL_WALL_TIMEOUT_SECONDS CELL_PROCESS_RESTARTS SERVER_READY_TIMEOUT BASE_PORT STRATIFIED_SAMPLES; do
  integer_value="${!integer_name}"
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer; got %q.\n' "$integer_name" "$integer_value" >&2
    exit 2
  fi
done
if [[ "$INCLUDE_NONE_STUDENT" != "0" && "$INCLUDE_NONE_STUDENT" != "1" ]]; then
  printf 'MATRIX_INCLUDE_NONE_STUDENT must be 0 or 1; got %q.\n' "$INCLUDE_NONE_STUDENT" >&2
  exit 2
fi
if [[ -n "$COMMON_GLOBAL_STEP" && ! "$COMMON_GLOBAL_STEP" =~ ^[0-9]+$ ]]; then
  printf 'COMMON_GLOBAL_STEP must be a non-negative integer; got %q.\n' "$COMMON_GLOBAL_STEP" >&2
  exit 2
fi
if (( EVAL_CONCURRENCY < 1 || SERVER_MAX_RUNNING_REQUESTS < 1 || CALLER_MAX_CONCURRENT < 1 || EPISODE_TIMEOUT_SECONDS < 1 || CELL_WALL_TIMEOUT_SECONDS < 1 || SERVER_READY_TIMEOUT < 1 )); then
  printf 'Concurrency, request limits, and timeouts must be positive.\n' >&2
  exit 2
fi
if (( EVAL_CONCURRENCY * GENERALIZE_REPLAYS > CALLER_MAX_CONCURRENT || EVAL_CONCURRENCY * GENERALIZE_REPLAYS > SERVER_MAX_RUNNING_REQUESTS )); then
  printf 'EVAL_CONCURRENCY x 8 exceeds a caller/server request cap.\n' >&2
  exit 2
fi
case "$SAVE_TRACES" in
  all|errors|none) ;;
  *)
    printf 'SAVE_TRACES must be all, errors, or none.\n' >&2
    exit 2
    ;;
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
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$TUTOR_FILEROOT/checkpoints/root/tutor-math-baseline}"
TEACHER_MODEL="${TEACHER_MODEL:-qwen3-8b}"
STUDENT_MODEL="${STUDENT_MODEL:-qwen3-1.7b}"

TEACHERS=(all-id)
declare -A TRIALS=(
  [all-id]=20260908_0901-reward-v4-all-id-fork775-8gpu
)
if [[ -n "${MATRIX_TEACHER_KEYS:-}" ]]; then
  IFS=',' read -r -a SELECTED_TEACHERS <<<"$MATRIX_TEACHER_KEYS"
  TEACHERS=()
  for teacher in "${SELECTED_TEACHERS[@]}"; do
    teacher="${teacher//[[:space:]]/}"
    if [[ -z "$teacher" || -z "${TRIALS[$teacher]:-}" ]]; then
      printf 'Unknown MATRIX_TEACHER_KEYS entry: %q\n' "$teacher" >&2
      exit 2
    fi
    TEACHERS+=("$teacher")
  done
fi
PREFERENCES=(
  none
  attempt-diagnosis
  subgoal-decomposition
  contrastive-comparison
  causal-justification
  step-demonstration
  independent-verification
)
STUDENT_SPLITS=(ID OOD ID ID OOD ID OOD)
STUDENTS=(
  qwen3-1.7b-text-original
  qwen3-1.7b-text-original-attempt-diagnosis
  qwen3-1.7b-text-original-subgoal-decomposition
  qwen3-1.7b-text-original-contrastive-comparison
  qwen3-1.7b-text-original-causal-justification
  qwen3-1.7b-text-original-step-demonstration
  qwen3-1.7b-text-original-independent-verification
)
if [[ "$INCLUDE_NONE_STUDENT" == "0" ]]; then
  PREFERENCES=("${PREFERENCES[@]:1}")
  STUDENT_SPLITS=("${STUDENT_SPLITS[@]:1}")
  STUDENTS=("${STUDENTS[@]:1}")
fi

# A resumed run must keep its original common checkpoint even if training has
# since created a newer checkpoint shared by all selected run.
if [[ "$MODE" == "run" && -n "${EVAL_RUN_DIR:-}" && -f "$EVAL_RUN_DIR/manifest.json" && -z "$COMMON_GLOBAL_STEP" ]]; then
  COMMON_GLOBAL_STEP="$("$PYTHON" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["common_global_step"])' "$EVAL_RUN_DIR/manifest.json")"
fi

RESOLVE_ARGS=()
for teacher in "${TEACHERS[@]}"; do
  RESOLVE_ARGS+=("$teacher=${TRIALS[$teacher]}")
done
RESOLUTION="$("$PYTHON" -B - "$CHECKPOINT_ROOT" "$COMMON_GLOBAL_STEP" "${RESOLVE_ARGS[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
requested = sys.argv[2].strip()
specs = [value.split("=", 1) for value in sys.argv[3:]]
by_teacher = {}
for key, trial in specs:
    directory = root / trial / "default"
    if not directory.is_dir():
        raise SystemExit(f"Missing checkpoint directory: {directory}")
    candidates = {}
    for path in directory.iterdir():
        match = re.search(r"globalstep([0-9]+)$", path.name)
        if (
            path.is_dir()
            and match
            and (path / "adapter_model.safetensors").is_file()
            and (path / "adapter_config.json").is_file()
        ):
            candidates.setdefault(int(match.group(1)), []).append(path.resolve())
    if not candidates:
        raise SystemExit(f"No complete LoRA checkpoints in {directory}")
    by_teacher[key] = candidates

common = set.intersection(*(set(value) for value in by_teacher.values()))
if not common:
    details = {key: sorted(value) for key, value in by_teacher.items()}
    raise SystemExit(f"The selected runs have no complete checkpoint in common: {details}")
step = int(requested) if requested else max(common)
if step not in common:
    raise SystemExit(
        f"globalstep{step} is not complete in every run; common={sorted(common)}"
    )
print(f"COMMON\t{step}")
for key, _ in specs:
    paths = by_teacher[key][step]
    if len(paths) != 1:
        raise SystemExit(f"{key}: expected one globalstep{step}, found {paths}")
    path = paths[0]
    manifest = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if manifest.get("peft_type") != "LORA" or int(manifest.get("r", 0)) != 16:
        raise SystemExit(f"{key}: expected rank-16 LoRA at {path}")
    print(f"ADAPTER\t{key}\t{path}")
PY
)"

declare -A ADAPTERS=()
while IFS=$'\t' read -r kind first second; do
  case "$kind" in
    COMMON) COMMON_GLOBAL_STEP="$first" ;;
    ADAPTER) ADAPTERS["$first"]="$second" ;;
  esac
done <<<"$RESOLUTION"
if [[ -z "$COMMON_GLOBAL_STEP" || "${#ADAPTERS[@]}" != "${#TEACHERS[@]}" ]]; then
  printf 'Checkpoint resolution returned incomplete data.\n' >&2
  exit 1
fi
COMPLETED_TRAIN_STEPS=$((COMMON_GLOBAL_STEP + 1))

for model_path in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
  if [[ ! -f "$model_path/config.json" ]]; then
    printf 'Model path has no config.json: %s\n' "$model_path" >&2
    exit 1
  fi
done

printf '[resolve] common checkpoint=globalstep%s (%s completed updates)\n' "$COMMON_GLOBAL_STEP" "$COMPLETED_TRAIN_STEPS"
for teacher in "${TEACHERS[@]}"; do
  printf '[resolve] %-24s %s\n' "$teacher" "${ADAPTERS[$teacher]}"
done

PREFLIGHT_ARGS=()
for teacher in "${TEACHERS[@]}"; do
  PREFLIGHT_ARGS+=("$teacher" "${TRIALS[$teacher]}" "${ADAPTERS[$teacher]}")
done
PREFLIGHT_OUTPUT="$("$PYTHON" -B - "$EVAL_CONFIG" "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH" "$COMMON_GLOBAL_STEP" "$EXPECTED_EXPLAIN_RATIO" "$STRATIFIED_SAMPLES" "$INCLUDE_NONE_STUDENT" "${PREFLIGHT_ARGS[@]}" <<'PY'
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

from areal.utils.hf_utils import load_hf_tokenizer
from examples.tutor import train as tutor_train
from examples.tutor.scripts.evaluate_api_teacher_train_aligned import (
    build_eval_workflow_kwargs,
    dataset_sha256,
    effective_eval_presolve_enabled,
    load_experiment_config,
    prepare_test_dataset,
    resolve_teacher_generation_args,
)
from examples.tutor.workflow import (
    load_personality_complaints,
    load_personality_prompts,
)

config_path = Path(sys.argv[1]).resolve()
teacher_model_path = Path(sys.argv[2]).resolve()
student_model_path = Path(sys.argv[3]).resolve()
common_step = int(sys.argv[4])
expected_explain_ratio = float(sys.argv[5])
stratified_samples = int(sys.argv[6])
include_none_student = bool(int(sys.argv[7]))
teacher_args = sys.argv[8:]
if not teacher_args or len(teacher_args) % 3:
    raise SystemExit("Expected one or more teacher checkpoint triples")

expected_students = [
    "qwen3-1.7b-text-original",
    "qwen3-1.7b-text-original-attempt-diagnosis",
    "qwen3-1.7b-text-original-subgoal-decomposition",
    "qwen3-1.7b-text-original-contrastive-comparison",
    "qwen3-1.7b-text-original-causal-justification",
    "qwen3-1.7b-text-original-step-demonstration",
    "qwen3-1.7b-text-original-independent-verification",
]
preferences = [
    "none",
    "attempt-diagnosis",
    "subgoal-decomposition",
    "contrastive-comparison",
    "causal-justification",
    "step-demonstration",
    "independent-verification",
]
config, students = load_experiment_config(str(config_path), [])
tutor_train._apply_eval_average_rollouts(config)
actual_students = [str(student["name"]) for student in students]
if actual_students != expected_students:
    raise SystemExit(f"Seven-student pool drifted: {actual_students}")
actual_preferences = [str(student.get("personality") or "none") for student in students]
if actual_preferences != preferences:
    raise SystemExit(f"Student preferences drifted: {actual_preferences}")
if any(str(student.get("model", "")).lower() != "qwen3-1.7b" for student in students):
    raise SystemExit("A student does not use Qwen3-1.7B")

checks = {
    "max_turns": int(config.max_turns) == 10,
    "free_chat": bool(config.free_chat.enabled) and int(config.free_chat.budget) == 10,
    "teacher_end": bool(config.teacher_end_enabled),
    "leak_mode": config.leak_handling_mode == "masked_continue",
    "format_mode": config.format_handling_mode == "terminate",
    "pre_solve": bool(config.teacher_pre.enabled),
    "eval_no_preverify": config.evaluator.teacher_pre_verify is False,
    "eval_one_rollout": int(config.evaluator.average_rollouts) == 1,
    "teacher_sampling_temperature": (
        float(config.eval_gconfig.temperature) == float(config.gconfig.temperature)
        == 1.0
    ),
    "original_retest": bool(config.student_generalize.retest_original),
    "eight_retests": int(config.student_generalize.replays) == 8,
    "no_extra_retest_levels": (
        not config.student_generalize.level1_enabled
        and not config.student_generalize.level2_enabled
    ),
    "gate_v3": config.personality.gate_prompt_version == "v3",
    "binary_gate": config.personality.gate_decision_mode == "binary",
    "gate_every_turn": float(config.personality.gate_sample_rate) == 1.0,
    "teacher_only_failed_turns": (
        config.personality.gated_turn_visibility == "teacher_only"
    ),
    "explain_ratio": (
        float(config.personality.explain_ratio) == expected_explain_ratio
    ),
    "gate_retries": int(config.personality.gate_retries) == 3,
    "gate_nonterminating": (
        not config.personality.terminate_after_explained_failure
    ),
    "base_auxiliary": config.auxiliary_model.mode == "api",
    "repeat_terminate": bool(config.reward.teacher_exact_repeat_terminate),
    "length_retry": bool(config.length_retry.enabled) and config.length_retry.attempts == 3,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"Evaluation semantics drifted: {failed}")

prompts = load_personality_prompts(config.personality.prompts_path)
bare, explained = load_personality_complaints(config.personality.complaints_path)
demanding = preferences[1:]
if any(preference not in prompts for preference in demanding):
    raise SystemExit("V3 Gate prompt bank is incomplete")
if any(preference not in explained for preference in demanding) or not bare:
    raise SystemExit("V3 scripted complaint bank is incomplete")

eval_args = Namespace(
    teacher_temperature=None,
    teacher_top_p=None,
    teacher_max_tokens=None,
    presolve_attempts=0,
    presolve_max_tokens=None,
)
resolve_teacher_generation_args(eval_args, config)
for student in students:
    effective = build_eval_workflow_kwargs(
        config=config,
        student_models=[student],
        tokenizer=object(),
        args=eval_args,
        presolve_enabled=effective_eval_presolve_enabled(config),
    )
    personality = effective["personality"]
    effective_checks = {
        "pre_solve": effective["teacher_pre_enabled"] is True,
        "length_retry": effective["length_retry_enabled"] == config.length_retry.enabled and effective["length_retry_attempts"] == config.length_retry.attempts,
        "soft_overlong": effective["soft_overlong_penalty"]["enabled"] == config.reward.soft_overlong.enabled,
        "teacher_end": effective["teacher_end_enabled"] is True,
        "no_preverify": effective["teacher_pre_verify"] is False,
        "masked_leak_continue": effective["leak_handling_mode"] == "masked_continue",
        "format_terminate": effective["format_handling_mode"] == "terminate",
        "base_auxiliary": effective["aux_mode"] == "api",
        "auxiliary_without_lora": not (
            (effective.get("aux_request_params", {}).get("extra_body") or {}).get(
                "lora_path"
            )
        ),
        "gate_v3_binary": (
            personality["gate_prompt_version"] == "v3"
            and personality["gate_decision_mode"] == "binary"
        ),
        "teacher_only": personality["gated_turn_visibility"] == "teacher_only",
        "explain_ratio": (
            float(personality["explain_ratio"]) == expected_explain_ratio
        ),
        "original_retest": effective["student_generalize_retest_original"] is True,
        "eight_retests": int(effective["student_generalize_replays"]) == 8,
        "teacher_end": effective["teacher_end_enabled"] is True,
        "teacher_sampling_temperature": (
            float(effective["gconfig"].temperature) == 1.0
        ),
    }
    effective_failed = [name for name, passed in effective_checks.items() if not passed]
    if effective_failed:
        raise SystemExit(
            f"{student['name']}: effective rollout drifted: {effective_failed}"
        )

if config.student_generalize.enabled:
    tutor_train._prepare_math_generalization_data(config)
tokenizer = load_hf_tokenizer(config.tokenizer_path)
student_prompts = tutor_train._load_eval_student_prompts(config)
if student_prompts:
    raise SystemExit("Unexpected extra eval-student prompt pool")
dataset = prepare_test_dataset(
    config,
    [students[0]],
    tokenizer=tokenizer,
    limit=0,
    stratified_max_samples=stratified_samples,
    student_prompts=student_prompts,
)
if not dataset:
    raise SystemExit("The full evaluation dataset is empty")
if stratified_samples > 0 and len(dataset) != stratified_samples:
    raise SystemExit(
        f"Stratified selection returned {len(dataset)} rows, "
        f"expected {stratified_samples}"
    )

teachers = []
for index in range(0, len(teacher_args), 3):
    key, trial, raw_adapter = teacher_args[index:index + 3]
    adapter = Path(raw_adapter).resolve()
    if not adapter.name.endswith(f"globalstep{common_step}"):
        raise SystemExit(f"{key}: wrong checkpoint step: {adapter}")
    manifest = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if manifest.get("peft_type") != "LORA" or int(manifest.get("r", 0)) != 16:
        raise SystemExit(f"{key}: incompatible LoRA manifest")
    if Path(str(manifest.get("base_model_name_or_path", ""))).resolve() != teacher_model_path:
        raise SystemExit(f"{key}: LoRA base model differs from Qwen3-8B")
    teachers.append({"key": key, "trial": trial, "adapter": str(adapter)})

for label, path, hidden_size, layers in (
    ("teacher", teacher_model_path, 4096, 36),
    ("student", student_model_path, 2048, 28),
):
    model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    signature = (
        model_config.get("model_type"),
        int(model_config.get("hidden_size", 0)),
        int(model_config.get("num_hidden_layers", 0)),
    )
    if signature != ("qwen3", hidden_size, layers):
        raise SystemExit(f"{label} model signature is wrong: {signature}")

selected_students = [
    {
        "preference": preference,
        "name": name,
        "split": "ID" if preferences[index] in {"none", "contrastive-comparison", "subgoal-decomposition", "step-demonstration"} else "OOD",
    }
    for index, (preference, name) in enumerate(zip(preferences, expected_students))
]
if not include_none_student:
    selected_students = selected_students[1:]

report = {
    "common_global_step": common_step,
    "completed_train_steps": common_step + 1,
    "teachers": teachers,
    "students": selected_students,
    "dataset_rows": len(dataset),
    "dataset_sha256": dataset_sha256(dataset),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "prompts_sha256": hashlib.sha256(
        Path(config.personality.prompts_path).read_bytes()
    ).hexdigest(),
    "complaints_sha256": hashlib.sha256(
        Path(config.personality.complaints_path).read_bytes()
    ).hexdigest(),
    "semantics": {
        "teacher_pre_enabled": True,
        "teacher_pre_verify": False,
        "teacher_temperature": 1.0,
        "leak_handling_mode": "masked_continue",
        "format_handling_mode": "terminate",
        "teacher_end_enabled": True,
        "gate_prompt_version": "v3",
        "gate_decision_mode": "binary",
        "gate_sample_rate": 1.0,
        "gated_turn_visibility": "teacher_only",
        "scripted_leak_reply": True,
        "scripted_gate_reply": True,
        "student_generalize_retest_original": True,
        "student_generalize_replays": 8,
        "auxiliary": "fixed_base_qwen3_8b_without_lora",
    },
}
print(
    "[preflight] rollout semantics PASS; "
    f"teachers={len(teachers)}, students={len(selected_students)}, "
    f"cells={len(teachers) * len(selected_students)}, rows_per_cell={len(dataset)}"
)
print(f"[preflight] dataset_sha256={report['dataset_sha256']}")
print("PREFLIGHT_JSON=" + json.dumps(report, sort_keys=True, separators=(",", ":")))
PY
)"
printf '%s\n' "$PREFLIGHT_OUTPUT"
PREFLIGHT_JSON="$(printf '%s\n' "$PREFLIGHT_OUTPUT" | sed -n 's/^PREFLIGHT_JSON=//p' | tail -1)"
if [[ -z "$PREFLIGHT_JSON" ]]; then
  printf 'Semantic preflight did not emit its machine-readable report.\n' >&2
  exit 1
fi
EXPECTED_ROWS="$("$PYTHON" -B -c 'import json,sys; print(json.loads(sys.argv[1])["dataset_rows"])' "$PREFLIGHT_JSON")"

if [[ -n "$OUTPUT_TAG" && ! "$OUTPUT_TAG" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  printf 'MATRIX_OUTPUT_TAG contains unsupported characters: %q.\n' "$OUTPUT_TAG" >&2
  exit 2
fi
OUTPUT_TAG_PART="${OUTPUT_TAG:+-$OUTPUT_TAG}"
STAMP="${EVAL_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${EVAL_RUN_DIR:-$TUTOR_FILEROOT/offline_eval/0901-preference-v3-step$COMPLETED_TRAIN_STEPS$OUTPUT_TAG_PART-full-matrix/$STAMP}"
if [[ "$MODE" == "preflight" ]]; then
  printf '[preflight] no files or GPU processes were created.\n'
  printf '[preflight] prospective output=%s\n' "$RUN_DIR"
  exit 0
fi

for command_name in curl setsid nvidia-smi timeout; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf '%s is required for run mode.\n' "$command_name" >&2
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
    printf 'CUDA_VISIBLE_DEVICES needs eight distinct integer ids; got %q.\n' "$CUDA_VISIBLE_DEVICES" >&2
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
print("[preflight] eight GPU ids and eight local ports: PASS")
PY

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/cells" "$RUN_DIR/pairs"
"$PYTHON" -B - "$RUN_DIR/manifest.json" "$PREFLIGHT_JSON" "$EVAL_CONCURRENCY" "$SERVER_MAX_RUNNING_REQUESTS" "$CALLER_MAX_CONCURRENT" "$SAVE_TRACES" "$BASE_PORT" "$EPISODE_ERROR_RETRIES" "$EPISODE_RETRY_BACKOFF_SECONDS" "$EPISODE_TIMEOUT_SECONDS" "$CELL_WALL_TIMEOUT_SECONDS" "$CELL_PROCESS_RESTARTS" "$STRATIFIED_SAMPLES" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

(
    output,
    preflight_raw,
    concurrency,
    server_cap,
    caller_cap,
    save_traces,
    base_port,
    error_retries,
    retry_backoff,
    episode_timeout,
    cell_timeout,
    cell_restarts,
    stratified_samples,
) = sys.argv[1:]
preflight = json.loads(preflight_raw)
dataset_rows = preflight.pop("dataset_rows")
dataset_hash = preflight.pop("dataset_sha256")
payload = {
    **preflight,
    "dataset": {
        "selection": (
            "math_type_level_stratified"
            if int(stratified_samples) > 0
            else "full"
        ),
        "rows": dataset_rows,
        "sha256": dataset_hash,
    },
    "runtime": {
        "gpu_layout": "4 x (1 teacher H200 + 1 student H200)",
        "pair_count": 4,
        "eval_concurrency_per_pair": int(concurrency),
        "server_max_running_requests": int(server_cap),
        "caller_max_concurrent": int(caller_cap),
        "save_traces": save_traces,
        "base_port": int(base_port),
        "episode_error_retries": int(error_retries),
        "episode_retry_backoff_seconds": int(retry_backoff),
        "episode_timeout_seconds": int(episode_timeout),
        "cell_wall_timeout_seconds": int(cell_timeout),
        "cell_process_restarts": int(cell_restarts),
    },
}
path = Path(output)
if path.exists():
    previous = json.loads(path.read_text(encoding="utf-8"))
    previous.pop("created_at", None)
    if previous != payload:
        raise SystemExit(
            "Existing manifest differs; use the original settings or a new "
            "EVAL_RUN_DIR."
        )
else:
    payload["created_at"] = datetime.now(UTC).isoformat()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
  local output=$1 teacher=$2 student=$3 preference=$4 adapter=$5
  "$PYTHON" -B - "$output" "$teacher" "$student" "$preference" "$adapter" "$EXPECTED_ROWS" "$EXPECTED_EXPLAIN_RATIO" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
teacher, student, preference = sys.argv[2:5]
adapter = str(Path(sys.argv[5]).resolve())
expected = int(sys.argv[6])
expected_explain_ratio = float(sys.argv[7])
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
signature = json.loads(
    (output / "run_config.json").read_text(encoding="utf-8")
)["signature"]
if set(summary["modes"]) != {"presolve_on"}:
    raise SystemExit(f"{output}: expected only presolve_on")
mode = summary["modes"]["presolve_on"]
checks = {
    "dataset_rows": int(summary["dataset_rows"]) == expected,
    "expected_attempts": int(mode["expected_attempts"]) == expected,
    "recorded_attempts": int(mode["recorded_attempts"]) == expected,
    "accounted_attempts": (
        int(mode["completed_attempts"]) + int(mode["error_count"]) == expected
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"{output}: coverage checks failed: {failed}")
students = signature["students"]
if len(students) != 1 or students[0]["name"] != student:
    raise SystemExit(f"{output}: evaluator is not singleton for {student}")
actual_preference = str(students[0].get("personality") or "none")
if actual_preference != preference:
    raise SystemExit(
        f"{output}: preference={actual_preference}, expected={preference}"
    )
if signature["presolve"]["verify"]:
    raise SystemExit(f"{output}: pre-solve verification is enabled")
semantics = signature["test_semantics"]
personality = semantics["personality"]
semantic_checks = {
    "max_turns": int(semantics["max_turns"]) == 10,
    "free_chat": (
        semantics["free_chat"]["enabled"] is True
        and int(semantics["free_chat"]["budget"]) == 10
    ),
    "masked_continue": semantics["leak_handling_mode"] == "masked_continue",
    "format_terminate": semantics["format_handling_mode"] == "terminate",
    "gate_v3": personality["gate_prompt_version"] == "v3",
    "binary_gate": personality["gate_decision_mode"] == "binary",
    "gate_every_turn": float(personality["gate_sample_rate"]) == 1.0,
    "teacher_only": personality["gated_turn_visibility"] == "teacher_only",
    "explain_ratio": (
        float(personality["explain_ratio"]) == expected_explain_ratio
    ),
    "gate_retries": int(personality["gate_retries"]) == 3,
    "original_retest": semantics["student_generalize_retest_original"] is True,
    "eight_retests": int(semantics["student_generalize_replays"]) == 8,
    "original_only": set(semantics["generalization_levels"]) == {"original"},
}
semantic_failed = [name for name, passed in semantic_checks.items() if not passed]
if semantic_failed:
    raise SystemExit(f"{output}: rollout semantics drifted: {semantic_failed}")
teacher_body = signature["teacher"]["request_params"].get("extra_body") or {}
teacher_lora = str(teacher_body.get("lora_path") or "")
if not teacher_lora or str(Path(teacher_lora).resolve()) != adapter:
    raise SystemExit(f"{output}: {teacher} uses wrong LoRA: {teacher_lora}")
auxiliary = signature["auxiliary"]
if auxiliary["source_mode"] != "api" or auxiliary["effective_mode"] != "api":
    raise SystemExit(f"{output}: auxiliary is not the fixed base API path")
aux_body = auxiliary["request_params"].get("extra_body") or {}
if aux_body.get("lora_path"):
    raise SystemExit(f"{output}: auxiliary unexpectedly carries a LoRA")
pending = int((summary.get("pending_backfill") or {}).get("count", 0) or 0)
print(f"[cell-ok] {teacher} x {preference}: rows={expected}, pending={pending}")
PY
}

run_pair() (
  set -Eeuo pipefail
  local pair_index=$1 teacher_gpu=$2 student_gpu=$3
  local teacher_port=$((BASE_PORT + pair_index * 10))
  local student_port=$((teacher_port + 1))
  local teacher_url="http://127.0.0.1:${teacher_port}/v1"
  local student_url="http://127.0.0.1:${student_port}/v1"
  local pair_dir="$RUN_DIR/pairs/pair-$pair_index"
  local teacher_log="$RUN_DIR/logs/pair-$pair_index-teacher.log"
  local student_log="$RUN_DIR/logs/pair-$pair_index-student.log"
  local teacher_pid="" student_pid="" current_cell_pid=""

  pair_cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM
    eval_stop_process_groups "$current_cell_pid" "$teacher_pid" "$student_pid"
    exit "$status"
  }
  trap pair_cleanup EXIT INT TERM

  mkdir -p "$pair_dir/adapters"
  local -a served_paths=()
  local teacher target
  for teacher in "${TEACHERS[@]}"; do
    target="$pair_dir/adapters/$teacher"
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

  local -a teacher_server_command=(
    "$PYTHON" -m sglang.launch_server
    --model-path "$TEACHER_MODEL_PATH"
    --served-model-name "$TEACHER_MODEL"
    --host 127.0.0.1
    --port "$teacher_port"
    --tp-size 1
    --context-length 40960
    --mem-fraction-static 0.80
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS"
    --enable-lora
    --lora-paths "${served_paths[@]}"
    --max-loras-per-batch 1
    --max-loaded-loras 5
  )
  CUDA_VISIBLE_DEVICES="$teacher_gpu" setsid "${teacher_server_command[@]}" >"$teacher_log" 2>&1 &
  teacher_pid=$!

  local -a student_server_command=(
    "$PYTHON" -m sglang.launch_server
    --model-path "$STUDENT_MODEL_PATH"
    --served-model-name "$STUDENT_MODEL"
    --host 127.0.0.1
    --port "$student_port"
    --tp-size 1
    --context-length 40960
    --mem-fraction-static 0.80
    --max-running-requests "$SERVER_MAX_RUNNING_REQUESTS"
  )
  CUDA_VISIBLE_DEVICES="$student_gpu" setsid "${student_server_command[@]}" >"$student_log" 2>&1 &
  student_pid=$!

  wait_ready "$teacher_pid" "$teacher_url" "$teacher_log" "pair $pair_index teacher (GPU $teacher_gpu)"
  wait_ready "$student_pid" "$student_url" "$student_log" "pair $pair_index student (GPU $student_gpu)"

  run_eval_cell() {
    local teacher=$1 student=$2 preference=$3 output=$4 log=$5
    local alias request_params status cell_try max_cell_tries
    local -a command attempt_command
    alias="$pair_dir/adapters/$teacher"
    request_params="$("$PYTHON" -B -c 'import json,sys; print(json.dumps({"seed":42,"extra_body":{"lora_path":sys.argv[1],"chat_template_kwargs":{"enable_thinking":False}}}))' "$alias")"
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
      --stratified-max-samples "$STRATIFIED_SAMPLES"
      --concurrency "$EVAL_CONCURRENCY"
      --episode-error-retries "$EPISODE_ERROR_RETRIES"
      --episode-error-retry-backoff-seconds "$EPISODE_RETRY_BACKOFF_SECONDS"
      --episode-timeout-seconds "$EPISODE_TIMEOUT_SECONDS"
      --retry-diagnostic-failures
      --save-traces "$SAVE_TRACES"
      --log-every 10
      --output-dir "$output"
      "auxiliary_model.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
      "student_axes.0.template.max_concurrent_calls=$CALLER_MAX_CONCURRENT"
    )
    printf '[cell] pair=%s teacher=%s student=%s\n' "$pair_index" "$teacher" "$preference"
    touch "$log"
    max_cell_tries=$((CELL_PROCESS_RESTARTS + 1))
    for ((cell_try = 1; cell_try <= max_cell_tries; cell_try++)); do
      attempt_command=("${command[@]}")
      if [[ -f "$output/run_config.json" ]]; then
        attempt_command+=(--resume)
      fi
      printf '[cell-process] try=%s/%s wall_timeout=%ss\n' "$cell_try" "$max_cell_tries" "$CELL_WALL_TIMEOUT_SECONDS" >>"$log"
      set +e
      setsid env TUTOR_QWEN3_8B_BASE_URL="$teacher_url" TUTOR_QWEN3_1_7B_BASE_URL="$student_url" timeout --signal=TERM --kill-after=30s "${CELL_WALL_TIMEOUT_SECONDS}s" "${attempt_command[@]}" >>"$log" 2>&1 &
      current_cell_pid=$!
      wait "$current_cell_pid"
      status=$?
      current_cell_pid=""
      set -e
      if (( status == 0 )); then
        break
      fi
      if (( cell_try == max_cell_tries )); then
        printf 'Cell failed after %s tries; tail of %s:\n' "$max_cell_tries" "$log" >&2
        tail -100 "$log" >&2 || true
        return 1
      fi
      printf '[cell-process-retry] status=%s; preserving results and resuming\n' "$status" >>"$log"
    done
    validate_cell "$output" "$teacher" "$student" "$preference" "${ADAPTERS[$teacher]}"
  }

  local teacher_index student_index cell_index student preference output log
  cell_index=0
  for teacher_index in "${!TEACHERS[@]}"; do
    teacher="${TEACHERS[$teacher_index]}"
    for student_index in "${!STUDENTS[@]}"; do
      if (( cell_index % PAIR_COUNT == pair_index )); then
        student="${STUDENTS[$student_index]}"
        preference="${PREFERENCES[$student_index]}"
        output="$RUN_DIR/cells/$teacher/$student"
        log="$RUN_DIR/logs/$teacher--$preference.log"
        run_eval_cell "$teacher" "$student" "$preference" "$output" "$log"
      fi
      cell_index=$((cell_index + 1))
    done
  done
)

WORKER_PIDS=()
cleanup_all() {
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
trap cleanup_all EXIT INT TERM

TEACHER_COUNT="${#TEACHERS[@]}"
STUDENT_COUNT="${#STUDENTS[@]}"
CELL_COUNT=$((TEACHER_COUNT * STUDENT_COUNT))
printf '[run] %s teachers x %s students x %s rows; %s cells over 4 GPU pairs\n' "$TEACHER_COUNT" "$STUDENT_COUNT" "$EXPECTED_ROWS" "$CELL_COUNT"
printf '[run] output=%s\n' "$RUN_DIR"
for ((pair_index = 0; pair_index < PAIR_COUNT; pair_index++)); do
  run_pair "$pair_index" "${GPU_IDS[$((pair_index * 2))]}" "${GPU_IDS[$((pair_index * 2 + 1))]}" &
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
  printf 'A server pair or cell failed; stopping the other pairs. Logs: %s/logs\n' "$RUN_DIR" >&2
  exit 1
fi
WORKER_PIDS=()

write_matrix_summary "$RUN_DIR"
PENDING_COUNT="$("$PYTHON" -B - "$RUN_DIR/cells" <<'PY'
import json
import sys
from pathlib import Path

count = 0
for path in Path(sys.argv[1]).glob("*/*/pending_backfill.jsonl"):
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            json.loads(raw)
            count += 1
print(count)
PY
)"
trap - EXIT INT TERM
if (( PENDING_COUNT > 0 )); then
  printf '[done-with-pending] %s episodes still need backfill. Resume with:\n' "$PENDING_COUNT"
  printf 'CUDA_VISIBLE_DEVICES=%q EVAL_RUN_DIR=%q bash %q run\n' "$CUDA_VISIBLE_DEVICES" "$RUN_DIR" "$LAUNCHER_SCRIPT"
  exit 0
fi
printf '[done] full matrix and summaries: %s\n' "$RUN_DIR"
