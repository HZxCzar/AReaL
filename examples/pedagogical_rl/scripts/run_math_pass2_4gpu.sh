#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint. New jobs should call ../run_official.sh.
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${EXAMPLE_DIR}/run_official.sh" "$@"
