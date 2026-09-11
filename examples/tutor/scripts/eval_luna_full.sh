#!/usr/bin/env bash
# This experiment's endpoints; credentials are loaded from the repository .env.
# Optional: append --resume to continue, or --dry-run to inspect without API calls.
# v1 is retained for audit. Reserve $1 of the $25 total for v1's known and
# unknown charges; the fresh v2 usage guard stops at $24 (an estimate, not billing).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_DIR"

STUDENT_URL="https://gqb5oejpqmdccaq5j9cae5qkmm59jjqd.openapi-qb-nat2.sii.edu.cn/v1"
JUDGE_URL="https://ke85ckbqoh5ecjq8jmjdm5jjhgbda8eg.openapi-qb-nat2.sii.edu.cn/v1"

.venv/bin/python examples/tutor/scripts/eval_luna_full.py \
  --student-base-url "$STUDENT_URL" \
  --aux-base-url "$JUDGE_URL" \
  --output-dir output/api-eval/luna-full-medium-v2 \
  --budget-usd 24 \
  --concurrency 64 \
  "$@" \
  2>&1 | tee -a luna-full-medium-v2.log
