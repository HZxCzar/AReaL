#!/usr/bin/env bash
set -Eeuo pipefail

# Convenience launcher for the official reasoning-plus-decision classifier.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export GATE_DECISION_MODE=classifier
exec bash "$ROOT_DIR/examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh" "$@"
