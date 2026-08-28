#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility alias for the 0825 A-G next-token-logprob classifier launcher.
# The shared launcher now uses classification directly.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export GATE_DECISION_MODE=classification
exec bash "$ROOT_DIR/examples/tutor/scripts/eval_0825_personality_base_aux_8gpu.sh" "$@"
