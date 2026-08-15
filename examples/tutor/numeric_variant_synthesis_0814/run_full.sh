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

mkdir -p "$HERE/runs/full_train_test"
exec 9>"$HERE/runs/full_train_test.lock"
if ! flock -n 9; then
  echo "Another full numeric-variant process owns the lock." >&2
  exit 3
fi

exec .venv/bin/python -B \
  "$HERE/generate_numeric_variants.py" \
  --run-dir runs/full_train_test \
  --max-attempts 3 \
  --source-concurrency 32 \
  --max-concurrent-calls 32 \
  --min-interval-seconds 0 \
  --retry-failed \
  >> "$HERE/runs/full_train_test/runner.log" 2>&1
