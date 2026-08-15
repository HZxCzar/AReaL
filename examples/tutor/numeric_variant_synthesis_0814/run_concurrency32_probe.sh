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

RUN_DIR="$HERE/runs/concurrency32_probe"
mkdir -p "$RUN_DIR"
exec 9>"$HERE/runs/concurrency32_probe.lock"
if ! flock -n 9; then
  echo "Another concurrency-32 probe owns the lock." >&2
  exit 3
fi

exec .venv/bin/python -B \
  "$HERE/generate_numeric_variants.py" \
  --run-dir runs/concurrency32_probe \
  --limit 32 \
  --max-attempts 1 \
  --source-concurrency 32 \
  --max-concurrent-calls 32 \
  --min-interval-seconds 0 \
  >> "$RUN_DIR/runner.log" 2>&1
