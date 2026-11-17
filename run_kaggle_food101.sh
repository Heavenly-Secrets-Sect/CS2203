#!/usr/bin/env bash
# Chạy MỘT session của một stage cho đồ án Food-101 + ViT trên Kaggle (CI của nhóm).
#   STAGE=smoke bash run_kaggle_food101.sh <session>        # session = 1..N, N = n_sessions của STAGE trong scripts/run_plan.json
# Biến môi trường từ launcher: CI_REPO CI_SHA CI_SESSION BUDGET_HOURS CI_OUTPUT_DIR (=/kaggle/working).
# Của project: STAGE (bắt buộc, KHÔNG có mặc định ngầm), FOOD101_PERSIST_ROOT (mặc định /kaggle/temp/food101-persist).
# Chế độ thử không cần GPU: DRY=1 (chỉ in kế hoạch) | SMOKE_LOCAL=1 (bỏ GPU/pip/dựng cache dữ liệu) và FOOD101_RUN_CMD="python3 scripts/fake_run_job.py".
# Quy ước: 2 job 1-GPU chạy song song (lane 0 -> GPU 0, lane 1 -> GPU 1); lane xong sớm thì "ăn" job cuối của lane kia (khoá mkdir).
# Đóng gói zip SAU MỖI JOB. /kaggle/working chỉ có đúng một file: food101_s<session>.zip. Không checkpoint, không cache.
# Thoát: 0 = xong hết; 1 = có job lỗi; 3 = không lỗi nhưng có job bị hoãn vì hết giờ; 2 = lỗi môi trường/cấu hình.
set -uo pipefail

SESSION="${1:-${CI_SESSION:-}}"
[ -n "$SESSION" ] || { echo "thiếu số session (đối số 1 hoặc CI_SESSION)" >&2; exit 2; }
[ -n "${STAGE:-}" ] || { echo "thiếu STAGE (smoke|sweep|p0|explore); không đoán stage" >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_DIR="$PWD"

PERSIST="${FOOD101_PERSIST_ROOT:-/kaggle/temp/food101-persist}"
OUTDIR="${CI_OUTPUT_DIR:-/kaggle/working}"
BUDGET_H="${BUDGET_HOURS:-10}"
DRY="${DRY:-0}"
LOCAL="${SMOKE_LOCAL:-0}"
RUN_CMD="${FOOD101_RUN_CMD:-python3 -m food101.run_job}"
export FOOD101_PERSIST_ROOT="$PERSIST" CI_OUTPUT_DIR="$OUTDIR"
export HF_HOME="$PERSIST/hf" TORCH_HOME="$PERSIST/torch" HF_HUB_DISABLE_TELEMETRY=1
export FOOD101_CACHE_DIR="${FOOD101_CACHE_DIR:-$PERSIST/cache}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"
T0=$(date +%s)
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- kế hoạch của session
log "stage=$STAGE session=$SESSION budget=${BUDGET_H}h sha=${CI_SHA:-?} persist=$PERSIST"
JOBS_TSV="$(python3 scripts/plan.py session-jobs --stage "$STAGE" --session "$SESSION")" || { echo "không lấy được job của session" >&2; exit 2; }
[ -n "$JOBS_TSV" ] || { echo "session $SESSION của stage $STAGE rỗng" >&2; exit 2; }
if [ "$DRY" = "1" ]; then
  log "DRY=1: kế hoạch (id | lane | ước tính phút, chưa đo | out | đối số)"; printf '%s\n' "$JOBS_TSV"; exit 0
fi
if [ "$LOCAL" != "1" ]; then python3 scripts/plan.py check || exit 2; fi

mkdir -p "$PERSIST/outputs" "$PERSIST/logs" "$OUTDIR"
rm -rf "$PERSIST/locks" "$PERSIST/pack.lock"; mkdir -p "$PERSIST/locks"
rm -f "$OUTDIR"/.food101_s*.zip.tmp "$PERSIST/failed.txt" "$PERSIST/deferred.txt"
touch "$PERSIST/failed.txt" "$PERSIST/deferred.txt"
[ -f "$PERSIST/job_times.csv" ] || echo "id,minutes,exit_code,est_min,lane,gpu" > "$PERSIST/job_times.csv"

# ---------------------------------------------------------------- GPU
NGPU=2
if [ "$LOCAL" != "1" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "không có nvidia-smi: session phải bật accelerator gpu" >&2; exit 2; }
  nvidia-smi -L || { echo "nvidia-smi lỗi" >&2; exit 2; }
  NGPU="$(nvidia-smi -L | grep -c '^GPU')"
  [ "$NGPU" -ge 1 ] || { echo "không thấy GPU" >&2; exit 2; }
  [ "$NGPU" -ge 2 ] && NGPU=2 || log "CẢNH BÁO: chỉ 1 GPU, chạy tuần tự (kế hoạch giả định 2)"
fi

# ---------------------------------------------------------------- môi trường (KHÔNG cài lại torch)
if [ "$LOCAL" != "1" ]; then
  python3 -c "import torch,sys; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "torch không thấy CUDA" >&2; exit 2; }
  TORCH_BEFORE="$(python3 -c 'import torch;print(torch.__version__)')"
  python3 -m pip install -q -r requirements.txt || { echo "pip install lỗi" >&2; exit 2; }
  TORCH_AFTER="$(python3 -c 'import torch;print(torch.__version__)')"
  [ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || log "CẢNH BÁO: pip đã đổi torch $TORCH_BEFORE -> $TORCH_AFTER (requirements.txt không được kéo torch)"
  # Dựng cache ảnh từ HF (ethz/food101, ~5 GB) MỘT lần trước khi chia 2 lane. CHƯA KIỂM CHỨNG trong session CI
  # (cần internet, không HF_TOKEN). Thất bại là lỗi cấu hình vì mọi job đều cần dữ liệu.
  python3 - <<'PY' || { echo "dựng cache food101 lỗi (xem log phía trên)" >&2; exit 2; }
from food101 import data as D
from food101.utils import load_configs
cfg, _ = load_configs()
meta = D.ensure_cache(cfg)
D.check_counts(meta, cfg)
print({k: v["n"] for k, v in meta["splits"].items()})
PY
fi

# 2 job chia 4 CPU: chia num_workers (chưa đo ảnh hưởng tới throughput)
if [ -z "${FOOD101_NUM_WORKERS:-}" ]; then
  NCPU="$(nproc 2>/dev/null || echo 4)"; export FOOD101_NUM_WORKERS=$(( NCPU / NGPU > 0 ? NCPU / NGPU : 1 ))
fi

python3 - <<'PY'
import json, os, platform, subprocess
from importlib import metadata
def v(p):
    try: return metadata.version(p)
    except Exception: return None
try: gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
except Exception: gpus = []
info = {k: os.environ.get(k) for k in ("CI_REPO", "CI_SHA", "CI_SESSION", "STAGE", "BUDGET_HOURS")}
info.update(python=platform.python_version(), gpus=gpus, num_workers=os.environ.get("FOOD101_NUM_WORKERS"),
            versions={p: v(p) for p in ("torch", "torchvision", "timm", "open_clip_torch", "datasets", "numpy", "scikit-learn", "scipy")})
open(os.path.join(os.environ["FOOD101_PERSIST_ROOT"], "ci_env.json"), "w").write(json.dumps(info, indent=1))
PY

# ---------------------------------------------------------------- chạy job
IDS=(); LANES=(); ESTS=(); OUTS=(); ARGS=()
while IFS=$'\t' read -r id lane est out args; do
  IDS+=("$id"); LANES+=("$lane"); ESTS+=("$est"); OUTS+=("$out"); ARGS+=("$args")
done <<< "$JOBS_TSV"
N=${#IDS[@]}
log "$N job, $NGPU lane"

pack() {  # có khoá để 2 lane không ghi đè nhau
  until mkdir "$PERSIST/pack.lock" 2>/dev/null; do sleep 1; done
  bash scripts/pack_runs.sh "$SESSION" || log "CẢNH BÁO: pack lỗi"
  rmdir "$PERSIST/pack.lock"
}

run_one() {  # idx gpu
  local i="$1" gpu="$2" id est out start now rc
  id="${IDS[$i]}"; est="${ESTS[$i]}"; out="$PERSIST/outputs/${OUTS[$i]}"
  now=$(date +%s)
  if awk -v e="$((now - T0))" -v est="$est" -v b="$BUDGET_H" 'BEGIN{exit !(e + est*60 > b*3600)}'; then
    log "HOÃN $id (ước tính ${est} phút, đã dùng $(( (now - T0) / 60 )) phút / budget ${BUDGET_H}h)"
    echo "$id" >> "$PERSIST/deferred.txt"; return
  fi
  if [ -f "$out/DONE" ]; then log "bỏ qua $id (đã xong)"; return; fi
  mkdir -p "$out"
  local -a cmd args
  read -r -a cmd <<< "$RUN_CMD"; read -r -a args <<< "${ARGS[$i]}"
  log "BẮT ĐẦU $id gpu=$gpu (ước tính ${est} phút, chưa đo)"
  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" "${args[@]}" --out "$out" > "$PERSIST/logs/$id.log" 2>&1
  rc=$?
  now=$(date +%s)
  printf '%s,%s,%s,%s,%s,%s\n' "$id" "$(awk -v s="$((now - start))" 'BEGIN{printf "%.1f", s/60}')" "$rc" "$est" "${LANES[$i]}" "$gpu" >> "$PERSIST/job_times.csv"
  if [ "$rc" -eq 0 ]; then log "XONG $id"; else echo "$id" >> "$PERSIST/failed.txt"; log "LỖI $id rc=$rc (xem logs/$id.log)"; tail -n 5 "$PERSIST/logs/$id.log" | sed 's/^/    | /'; fi
  pack
}

lane_worker() {  # lane gpu: job của lane mình trước theo thứ tự plan, rồi "ăn" job của lane khác từ cuối
  local lane="$1" gpu="$2" i order=()
  for ((i = 0; i < N; i++)); do [ "$NGPU" -lt 2 ] || [ "${LANES[$i]}" = "$lane" ] && order+=("$i"); done
  for ((i = N - 1; i >= 0; i--)); do [ "$NGPU" -ge 2 ] && [ "${LANES[$i]}" != "$lane" ] && order+=("$i"); done
  for i in "${order[@]}"; do
    mkdir "$PERSIST/locks/${IDS[$i]}" 2>/dev/null || continue
    run_one "$i" "$gpu"
  done
}

pids=()
for ((l = 0; l < NGPU; l++)); do lane_worker "$l" "$l" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p"; done

# ---------------------------------------------------------------- tổng kết
NFAIL=$(grep -c . "$PERSIST/failed.txt" || true); NDEF=$(grep -c . "$PERSIST/deferred.txt" || true)
{ echo "stage=$STAGE session=$SESSION jobs=$N failed=$NFAIL deferred=$NDEF elapsed_min=$(( ($(date +%s) - T0) / 60 ))"
  [ "$NFAIL" -eq 0 ] || { echo "FAILED:"; cat "$PERSIST/failed.txt"; }
  [ "$NDEF" -eq 0 ] || { echo "DEFERRED (hết giờ, cần plan lại):"; cat "$PERSIST/deferred.txt"; }
} | tee "$PERSIST/logs/_summary.txt"
pack
ls -la "$OUTDIR"
if [ "$NFAIL" -gt 0 ]; then exit 1; fi
if [ "$NDEF" -gt 0 ]; then exit 3; fi
exit 0
