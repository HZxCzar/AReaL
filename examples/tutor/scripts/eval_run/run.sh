#!/usr/bin/env bash
# Portable entrypoint. Deployment, credentials and proxy setup belong outside.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_DIR"
exec "${TUTOR_PYTHON:-$REPO_DIR/.venv/bin/python}" -m examples.tutor.scripts.eval_run.runner "$@"
