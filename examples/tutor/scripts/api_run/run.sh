#!/usr/bin/env bash
# Usage: bash run.sh MODEL_OR_CONFIG [runner options]
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_DIR"
exec "${TUTOR_PYTHON:-$REPO_DIR/.venv/bin/python}" -m examples.tutor.scripts.api_run.runner "$@"
