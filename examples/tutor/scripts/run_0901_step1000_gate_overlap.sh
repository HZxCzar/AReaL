#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
SOURCE_RUN="${SOURCE_RUN:-/inspire/hdd/project/qproject-fundationmodel/public/wxxu/TAgent/output_hdd/tutor/offline_eval/0901-preference-v3-step1000-explain100-full-matrix/20260904T154202Z}"
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/gate_overlap_results/step1000_explain100_full_all_contexts}"
GATE_BASE_URL="${GATE_BASE_URL:-https://ke85ckbqoh5ecjq8jmjdm5jjhgbda8eg.openapi-qb-nat2.sii.edu.cn/v1}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
unset ALL_PROXY HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"

while true; do
  readiness="$($PYTHON -B - "$SOURCE_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
manifest_path = run / "manifest.json"
if not manifest_path.is_file():
    print("waiting: manifest missing")
    raise SystemExit
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = int(manifest["dataset"]["rows"])
statuses = []
for teacher in manifest["teachers"]:
    for student in manifest["students"]:
        cell = run / "cells" / teacher["key"] / student["name"]
        results = cell / "results.jsonl"
        keys = set()
        if results.is_file():
            with results.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        keys.add(str(json.loads(line)["key"]))
        statuses.append((len(keys), (cell / "summary.json").is_file()))
complete = sum(count >= expected and has_summary for count, has_summary in statuses)
recorded = sum(min(count, expected) for count, _ in statuses)
if complete == len(statuses) == 35:
    print("READY")
else:
    print(
        f"waiting: complete_cells={complete}/35 "
        f"recorded_episodes={recorded}/{35 * expected}"
    )
PY
)"
  printf '[%(%FT%TZ)T] %s\n' -1 "$readiness"
  [[ "$readiness" == "READY" ]] && break
  sleep 60
done

exec "$PYTHON" -B examples/tutor/scripts/audit_0901_preference_v3_gate_overlap.py \
  --run-dir "$SOURCE_RUN" \
  --output-dir "$OUTPUT_DIR" \
  --expected-episodes-per-cell 528 \
  --base-url "$GATE_BASE_URL" \
  --model qwen3-8b \
  --inference-key '' \
  --include-no-previous-real-student \
  --concurrency 64 \
  --summary-every 500
