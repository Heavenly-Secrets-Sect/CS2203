#!/usr/bin/env bash
# Đóng gói sau mỗi job. Dùng: pack_runs.sh <session>   (cần FOOD101_PERSIST_ROOT, CI_OUTPUT_DIR)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$here/pack_runs.py" --persist "${FOOD101_PERSIST_ROOT:?}" --out-dir "${CI_OUTPUT_DIR:-/kaggle/working}" --session "$1"
