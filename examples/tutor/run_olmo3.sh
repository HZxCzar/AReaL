#!/usr/bin/env bash
# Self-hosted tutor training with Olmo-3-7B-Instruct as the teacher.
#
#   bash examples/tutor/run_olmo3.sh [2|4|8] [CONFIG.yaml] [CONFIG_OVERRIDE ...]
#
# Same signature as run_offline.sh, which this delegates to for everything it
# already owns: GPU selection, the local Qwen3-1.7B student server, the dead
# TUTOR_QWEN3_8B_BASE_URL that forces the auxiliary in-engine, the offline env,
# the venv PATH check, and the generation/training split read out of the config
# backends. run_offline.sh is unchanged apart from one line, noted below.
#
# DRY_RUN=1 works, since it is run_offline.sh's own flag.
#
# WHAT THIS ADDS is only environment. The model, its stop token ids and
# everything else belong to the config -- see
# examples/tutor/configs/math/0901/8gpu/olmo3/base.yaml. Two things cannot be
# expressed in a config file:
#
#   1. SGLANG_EXTERNAL_MODEL_PACKAGE=examples.tutor.olmo3.sglang_olmo3_ext
#      sglang 0.5.9 ships no olmo3 module and olmo2.py registers only
#      Olmo2ForCausalLM, so an OLMo 3 checkpoint resolves to nothing native and
#      falls back to the generic transformers backend -- which has no LoRA
#      support at all, while this loop is rank-16 LoRA with hot
#      /load_lora_adapter. Given as a dotted package name so it resolves off
#      $ROOT_DIR, which run_offline.sh already puts on PYTHONPATH; nothing extra
#      is needed for this half. sglang reads the variable once at registry
#      import (sglang/srt/models/registry.py:131) and it defaults to empty.
#
#   2. TAGENT_OLMO3_ROPE_FIX=1
#      Read by examples/tutor/olmo3/sitecustomize.py, which casts
#      Olmo3RotaryEmbedding's float32 cos/sin back to the input dtype. Without
#      it ppo_update dies in the actor workers on a bf16-vs-fp32
#      gradient-checkpoint metadata mismatch. It has to be a sitecustomize
#      module because the patch is needed in every python process AReaL spawns,
#      not just the trainer -- the failing forward/backward runs in the
#      actor/N rpc workers. PYTHONNOUSERSITE=1 rules out usercustomize.
#
# THE ONE SHARED-FILE CHANGE. run_offline.sh used to do
# `export PYTHONPATH="$ROOT_DIR"`, discarding whatever it inherited, so the
# entry needed for sitecustomize could not survive. It now appends instead:
# `export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"`. That is a no-op
# for every existing caller, since they run with PYTHONPATH unset, and it
# matches what run_official.sh already did.
#
# Pointing this at a Qwen config is harmless rather than wrong. The external
# package names only Olmo3ForCausalLM and sglang resolves by the architecture a
# checkpoint declares, and the rope fix touches no class but OLMo 3's. It simply
# would not switch models -- the config decides that.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT_DIR="$ROOT_DIR/examples/tutor/olmo3"
HF_HUB="${HF_HUB_CACHE:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/.cache/huggingface/hub}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,50p' "${BASH_SOURCE[0]}"
  exit 0
fi

for f in __init__.py sitecustomize.py preflight.py sglang_olmo3_ext/olmo3.py; do
  if [[ ! -f "$EXT_DIR/$f" ]]; then
    printf 'Missing %s\n' "$EXT_DIR/$f" >&2
    exit 1
  fi
done

# run_offline.sh appends to an inherited PYTHONPATH, so this entry survives into
# the trainer and, from there, into the sglang server and the actor/N workers.
export PYTHONPATH="$EXT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_EXTERNAL_MODEL_PACKAGE=examples.tutor.olmo3.sglang_olmo3_ext
export TAGENT_OLMO3_ROPE_FIX=1

# Resolve a snapshot for the pre-flight to build its rotary from. The config
# owns actor.path; this only makes the dtype assertion run against the real
# rope_scaling rather than a synthetic config.
mapfile -t SNAPS < <(find "$HF_HUB/models--allenai--Olmo-3-7B-Instruct/snapshots" \
  -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
if (( ${#SNAPS[@]} == 0 )); then
  printf 'No Olmo-3-7B-Instruct snapshot under %s.\n' "$HF_HUB" >&2
  printf 'Run examples/tutor/olmo3/download_olmo3.sh first.\n' >&2
  exit 1
fi

# PRE-FLIGHT. An unapplied rope fix does not fail at startup; it fails deep in
# the first ppo_update with an error that looks nothing like its cause.
printf 'pre-flight  : OLMo 3 rope dtype fix\n'
if ! "$ROOT_DIR/.venv/bin/python" "$EXT_DIR/preflight.py" "${SNAPS[-1]}"; then
  printf '\nRefusing to launch: the rope fix is not active, so ppo_update would\n' >&2
  printf 'die on a bf16-vs-fp32 gradient-checkpoint metadata mismatch.\n' >&2
  printf 'Check that %s/sitecustomize.py is on PYTHONPATH.\n' "$EXT_DIR" >&2
  exit 1
fi
printf 'teacher     : Olmo-3-7B-Instruct\n'
printf 'sglang ext  : %s\n\n' "$SGLANG_EXTERNAL_MODEL_PACKAGE"

exec bash "$ROOT_DIR/examples/tutor/run_offline.sh" "$@"
