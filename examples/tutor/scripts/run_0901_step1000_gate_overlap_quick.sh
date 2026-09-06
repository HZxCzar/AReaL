#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

exec "$ROOT_DIR/.venv/bin/python" -B \
  examples/tutor/scripts/audit_0901_preference_v3_gate_overlap.py \
  --run-dir /inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/offline_eval/0901-preference-v3-step1000-explain100-full-matrix/20260904T154202Z \
  --output-dir /inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/gate_overlap_results/step1000_explain100_quick_balanced30 \
  --expected-episodes-per-cell 0 \
  --max-episodes-per-cell 128 \
  --base-url https://ke85ckbqoh5ecjq8jmjdm5jjhgbda8eg.openapi-qb-nat2.sii.edu.cn/v1 \
  --model qwen3-8b \
  --inference-key '' \
  --include-no-previous-real-student \
  --max-samples-per-cell 30 \
  --stratify-previous-real-student \
  --concurrency 64 \
  --summary-every 250
