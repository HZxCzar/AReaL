#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

set -a
source .env
set +a
export PYTHONPATH=.
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$HERE/runs/smoke_5"
exec .venv/bin/python -B \
  "$HERE/generate_numeric_variants.py" \
  --run-dir runs/smoke_5 \
  --ids train-9 train-59 test-36 test-54 test-85 \
  --max-attempts 4 \
  --min-interval-seconds 2 \
  --retry-failed \
  >> "$HERE/runs/smoke_5/runner.log" 2>&1
