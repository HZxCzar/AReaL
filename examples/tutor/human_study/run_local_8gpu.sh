#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-$REPO_ROOT/.venv/bin/python}" -B -m examples.tutor.human_study.local_8gpu "$@"
