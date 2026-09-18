#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" -B examples/tutor/scripts/eval_suite.py "$@"
