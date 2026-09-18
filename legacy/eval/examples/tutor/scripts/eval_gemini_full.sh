#!/usr/bin/env bash
# Full seven-preference evaluation. The safe default budget remains $2;
# pass --budget-usd 100 explicitly for the suggested full-evaluation allowance.
# Prices: standard tier through 2026-12-31; review before later runs.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

# Interactive bash loads the user's clashon function. Do not alter other panes.
if [[ "${TUTOR_PROXY_READY:-0}" != 1 && " $* " != *" --dry-run "* && " $* " != *" --help "* ]]; then
  exec bash -ic 'clashon && export TUTOR_PROXY_READY=1 && exec bash "$@"' bash "$SCRIPT_DIR/eval_gemini_full.sh" "$@"
fi

PROXY_ARGS=(--keep-env-proxy)
if [[ " $* " == *" --dry-run "* ]]; then PROXY_ARGS=(); fi
.venv/bin/python -m examples.tutor.scripts.eval_api_full \
  --provider gemini --teacher-model gemini-3.8-flash \
  --teacher-pricing 0.75 0.075 0.75 3.75 \
  --output-dir output/api-eval/gemini-3.8-flash-full-medium \
  "${PROXY_ARGS[@]}" "$@" \
  2>&1 | tee -a gemini-3.8-flash-full-medium.log
