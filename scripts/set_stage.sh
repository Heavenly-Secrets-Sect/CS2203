#!/usr/bin/env bash
# Đặt stage cho kaggle-run.yml (sessions + STAGE=). Không commit, không tạo release.
# Dùng: bash scripts/set_stage.sh smoke|sweep|p0|explore
set -euo pipefail
[ $# -eq 1 ] || { echo "dùng: bash scripts/set_stage.sh <smoke|sweep|p0|explore>" >&2; exit 2; }
cd "$(dirname "$0")/.."
python3 scripts/plan.py set-stage "$1"
