#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/pedagogical_rl/run_tutor_protocol_eval.sh CHECKPOINT [TRIAL_NAME] [CONFIG_OVERRIDE ...]

Evaluate one rank-16 teacher adapter under the complete 0901 tutor protocol and
all seven V3 students. This is evaluation-only and performs no policy update.
EOF
}

if (( $# < 1 )); then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="$(readlink -f -- "$1")"
shift
if [[ ! -f "$CHECKPOINT/adapter_model.safetensors" || ! -f "$CHECKPOINT/adapter_config.json" ]]; then
  printf 'Not a complete LoRA checkpoint: %s\n' "$CHECKPOINT" >&2
  exit 1
fi

TRIAL_NAME=""
if (( $# > 0 )) && [[ "$1" != *=* ]]; then
  TRIAL_NAME="$1"
  shift
fi
TRIAL_NAME="${TRIAL_NAME:-$(date -u +%Y%m%d_%H%M%S)_pedagogical-rl-under-0901-eval}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec bash "$ROOT_DIR/examples/tutor/run_offline.sh" 8 \
  "$ROOT_DIR/examples/tutor/configs/math/0901/pilot/eval-all-preferences.yaml" \
  "trial_name=$TRIAL_NAME" \
  "+actor.init_lora_path=$CHECKPOINT" \
  total_train_epochs=1 \
  total_train_steps=0 \
  cluster.n_gpus_per_node=8 \
  rollout.backend=sglang:d4p1t1 \
  actor.backend=fsdp:d4p1t1 \
  rollout.max_concurrent_rollouts=64 \
  sglang.mem_fraction_static=0.50 \
  auxiliary_model.mode=self \
  evaluator.eval_before_train=true \
  evaluator.freq_epochs=null \
  evaluator.freq_steps=null \
  evaluator.freq_secs=null \
  evaluator.average_rollouts=1 \
  evaluator.teacher_pre_enabled=true \
  evaluator.teacher_pre_verify=false \
  personality.explain_ratio=1.0 \
  recover.mode=disabled \
  saver.freq_epochs=null \
  saver.freq_steps=null \
  saver.freq_secs=null \
  stats_logger.wandb.group=0904-pedagogical-rl-under-0901-eval \
  "$@"
