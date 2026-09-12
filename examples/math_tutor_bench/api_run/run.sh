#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-$REPO_ROOT/.venv/bin/python}" -m examples.math_tutor_bench.api_run.runner "$@"
