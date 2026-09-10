#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen-checkpoint comparison under PedagogicalRL's native dialogue protocol.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$0" run
# Optional: PAIR_RUN_DIR=/new/output/directory; mode preflight or analyze.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
MODE="${1:-preflight}"
case "$MODE" in
  preflight|run|analyze) ;;
  *) printf 'Usage: bash %s [preflight|run|analyze]\n' "$0" >&2; exit 2 ;;
esac
PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ "$MODE" == analyze ]]; then
  : "${PAIR_RUN_DIR:?Set PAIR_RUN_DIR to the original comparison directory}"
  exec "$PYTHON" -B "$SCRIPT_DIR/scripts/summarize_protocol_pair.py" "$PAIR_RUN_DIR"
fi

REFERENCE=examples/math_tutor_bench/results/0906-pedagogical-rl-qwen3-8b-lr5e-5-8gpu
REFERENCE="$REFERENCE/epoch21epochstep12globalstep999/run.json"
RESOLVED="$("$PYTHON" -B - "$REFERENCE" <<'PY'
import json
import sys
from pathlib import Path

ped = Path(json.loads(Path(sys.argv[1]).read_text())["checkpoint"]).resolve()
if ped.name != "epoch21epochstep12globalstep999":
    raise SystemExit(f"Unexpected PedRL checkpoint: {ped}")
output = ped.parents[6]
ours = output / "tutor/checkpoints/root/tutor-math-baseline"
ours /= "20260908_0901-reward-v4-all-id-fork775-8gpu/default/epoch21epochstep12globalstep999"
bases = []
for path in (ours, ped):
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file():
            raise SystemExit(f"Missing checkpoint artifact: {path / name}")
    manifest = json.loads((path / "adapter_config.json").read_text())
    if manifest.get("peft_type") != "LORA" or manifest.get("r") != 16:
        raise SystemExit(f"Expected rank-16 LoRA: {path}")
    bases.append(Path(manifest["base_model_name_or_path"]).resolve())
if bases[0] != bases[1] or not bases[0].is_dir():
    raise SystemExit("Teacher checkpoints must use the same available base model")
print(ours)
print(ped)
print(output / "pedagogical_rl/protocol_pair_eval")
PY
)"
mapfile -t PATHS <<<"$RESOLVED"
[[ ${#PATHS[@]} == 3 ]] || { printf 'Checkpoint resolution failed.\n' >&2; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PAIR_RUN_DIR="${PAIR_RUN_DIR:-${PATHS[2]}/$STAMP}"
CONFIG=examples/pedagogical_rl/configs/comparison/eval_8gpu.yaml
printf '[checkpoints] ours=%s\n[checkpoints] pedrl=%s\n' "${PATHS[0]}" "${PATHS[1]}"
printf '[comparison] output=%s\n' "$PAIR_RUN_DIR"

if [[ "$MODE" == run ]]; then
  : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to eight free GPU ids}"
  if [[ -e "$PAIR_RUN_DIR" ]]; then
    printf 'Refusing to mix results in existing PAIR_RUN_DIR: %s\n' "$PAIR_RUN_DIR" >&2
    exit 1
  fi
fi

for index in 0 1; do
  LABEL=ours
  [[ "$index" == 0 ]] || LABEL=pedrl
  COMMAND=(
    bash "$SCRIPT_DIR/run_comparison.sh" "$CONFIG"
    "trial_name=ped-protocol-pair-$STAMP-$LABEL"
    "actor.init_lora_path=${PATHS[$index]}"
    total_train_steps=0
    evaluator.eval_before_train=true
    evaluator.average_rollouts=1
    recover.mode=disabled
    max_eval_examples=-1
    teacher_pre.enabled=false
    teacher_pre.verify=false
    evaluation.matrix_enabled=true
    evaluation.compute_initial_attempts=true
    'evaluation.conversation_types=[GUIDED,ATTEMPTED]'
    'evaluation.preference_names=[none]'
    evaluation.record_turn_leak_diagnostic=false
    generation.teacher_output_format=unified_xml
    generation.leak_judge_mode=pedagogical_rl
    cross_eval.enabled=false
    "debug_trace_dir=$PAIR_RUN_DIR/$LABEL"
    debug_trace_every_n_rollouts=1
  )
  if [[ "$MODE" == preflight ]]; then
    DRY_RUN=1 "${COMMAND[@]}"
  else
    mkdir -p "$PAIR_RUN_DIR"
    printf '[run] %s: evaluation only, 528 questions x 2 protocols\n' "$LABEL"
    DRY_RUN=0 "${COMMAND[@]}" 2>&1 | tee "$PAIR_RUN_DIR/$LABEL.log"
  fi
done
if [[ "$MODE" == run ]]; then
  "$PYTHON" -B "$SCRIPT_DIR/scripts/summarize_protocol_pair.py" "$PAIR_RUN_DIR"
  printf '[done] Comparison and full trajectories: %s\n' "$PAIR_RUN_DIR"
fi
