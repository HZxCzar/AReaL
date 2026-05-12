#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
python examples/hanabi-tutor/evaluate_teacher.py --config examples/hanabi-tutor/teacher_eval_config.yaml "$@"
