# Food-101 + Vision Transformer (CS2203)

Nghiên cứu thực nghiệm có kiểm soát: LP / FT / LP-FT theo quy mô ViT-Ti→L, và ViT so với CNN trên Food-101. Kế hoạch: [docs/PLAN.md](docs/PLAN.md); tiền đăng ký: [PREREG.md](PREREG.md); sai lệch khi chạy: [DEVIATIONS.md](DEVIATIONS.md).

## Hạ tầng chạy (CI của nhóm, KHÔNG `kaggle kernels push`)

```
commit -> release tag -> workflow kaggle-run -> Worker message-talisman -> session Kaggle clone đúng SHA
       -> run_kaggle_food101.sh <session> -> MỘT zip trong /kaggle/working -> fetch-results -> audit
```

| File | Việc |
|---|---|
| `kaggle-run.yml` | manifest cho bot: notebook `food101-ci`, `sessions`, `STAGE=...` trong `run` |
| `.github/workflows/kaggle-run.yml` | gọi Worker khi publish release (pre-release chỉ dry-run) hoặc chạy tay |
| `run_kaggle_food101.sh` | kiểm GPU, pip (không cài torch), tải dữ liệu, 2 job 1-GPU song song, đóng gói zip sau mỗi job |
| `scripts/plan.py` | sinh / kiểm `scripts/run_plan.json` (đóng băng), `set-stage`, `summary` |
| `scripts/set_stage.sh` | đổi stage trong `kaggle-run.yml` (sessions và STAGE phải khớp plan) |
| `scripts/pick_lr.py` | chọn LR theo dev acc từ stage sweep -> `configs/selected_lr.json` |
| `scripts/audit_runs.py` | kiểm kết quả (đủ seed, 25,250 dòng, không NaN, nhãn 0-100, acc khớp logits) |
| `scripts/stats.py` | khung scorer / Welch / Holm / paired bootstrap (chưa dùng cho kết quả) |
| `configs/estimates.yaml` | throughput và hằng số ước tính, **tất cả chưa đo** |

**Quy tắc round:** mỗi release là MỘT round gồm nhiều session; bot xếp hàng session chưa có tài khoản (FIFO) và không resume round đã kết thúc. Việc độc lập chạy cùng lúc phải gom thành MỘT stage, không tách release (r3/r4/r7 là ngoại lệ, DEVIATIONS D7). Chỉ tách round khi stage sau cần kết quả stage trước.

Các stage (mỗi release chạy **đúng một**): `smoke` -> `sweep` -> `sweep2` -> `sweep3` (mở rộng LR, DEVIATIONS D2/D4) và `p0a`, `explore` (chạy song song, không chờ sweep3), rồi `p0b` sau sweep3 (DEVIATIONS D5). Xem bảng: `python3 scripts/plan.py summary`.

**Đánh số theo round và session** (quy ước của nhóm, giống `fox`): mỗi stage có một số round cố định, tag release là `r<N>`, zip kết quả là `food101_r<N>_s<M>.zip`.

| Round (tag) | Stage | Session | Zip |
|---|---|---|---|
| `r1` | smoke | 1 | `food101_r1_s1.zip` |
| `r2` | sweep | 1 | `food101_r2_s1.zip` |
| `r3` | p0a (ViT-B/L: FT, LP, LP-FT) | 2 (s2 = ViT-L) | `food101_r3_s1.zip`, `food101_r3_s2.zip` |
| `r4` | explore (DINOv2/CLIP FT+LP, DeiT LP, CLIP zero-shot) | 1 | `food101_r4_s1.zip` |
| `r5` | sweep2 (sweep mở rộng của r2) | 1 | `food101_r5_s1.zip` |
| `r6` | sweep3 (mở rộng lần cuối) | 1 | `food101_r6_s1.zip` |
| `r7` | p0b (ViT-Ti/S, ConvNeXt-B, ResNet-50 FT; DeiT-B FT) | 1 | `food101_r7_s1.zip` |

Gửi lại phần lỗi của round N: `r<N>.1`, `r<N>.2`... Gọi tắt trong báo cáo: "round 3, session 2"; số round lấy từ `python3 scripts/plan.py summary`.

## Điều kiện trước khi chạy (checklist)

- [ ] Repo `Heavenly-Secrets-Sect/CS2203` là **public** (launcher clone bằng git không token). Kiểm: mở repo khi chưa đăng nhập.
- [ ] Secret `CI_RUN_SECRET` và variable `KAGGLE_BOT_URL` (không có `/` cuối) đặt ở repo hoặc org. Chuyển repo sang org làm mất chúng.
- [ ] Repo thuộc `CI_ALLOWED_REPOS` của Worker (org `Heavenly-Secrets-Sect/*` đã được phép).
- [ ] Quota GPU còn: xem `considered` trong log dry-run. Mỗi tài khoản tối đa 2 session. Số GPU-giờ ước tính: `plan.py summary` (chưa đo).
- [ ] `PREREG.md` đã được chốt; `python3 scripts/plan.py check --regen` báo OK; `src/food101` đã có `run_job`.
- [ ] Người dùng đã **duyệt** việc tạo release (mỗi release tốn quota và đẩy mã ra ngoài).

> **Cảnh báo.** Không tạo release, không `workflow_dispatch` với dry_run tắt, khi chưa được duyệt. Không đặt token/secret vào file nào. Tick pre-release ép dry-run và không bật lại được (phải xoá release, tạo lại); dùng `gh release create` không kèm `-p` cho release thật.

## Quy trình từng bước

1. **Chọn stage** (mặc định `smoke`): `bash scripts/set_stage.sh smoke` rồi commit `kaggle-run.yml`. Release dùng đúng commit đó. Chỉ chuyển stage sau khi người dùng duyệt kết quả stage trước; script không tự đoán stage kế tiếp.
2. **Dry-run** (không tốn quota, không để lại release): `gh workflow run kaggle-run.yml -f dry_run=true`, rồi đọc `launched/queued/considered` trong log run.
3. **Smoke** (sau khi được duyệt; chạy 20 bước thật mỗi model bằng `--max-steps`, không phải cờ `--smoke` CPU của run_job): `gh release create r1 --title "r1 smoke" --notes "round 1: smoke, 1 session"`. Tag phải là `r<N>` (hoặc `r<N>.<k>` khi gửi lại) với N là round của `STAGE` trong manifest (bảng bên dưới), workflow chặn nếu lệch.
4. **Lấy kết quả** (chỉ khi người dùng yêu cầu, chỉ trên máy chủ dự án):
   ```bash
   MT=/Users/kietnt/Desktop/Workspace/message-talisman
   node --env-file=$MT/.env $MT/bot/scripts/fetch-results.mjs r1 --out runs_inbox
   ```
   Zip tải về là dữ liệu không tin cậy: giải nén vào thư mục riêng. `job_times.csv` trong zip có cột `id,minutes,exit_code,est_min,lane,gpu`.
5. **Hiệu chỉnh ước tính** từ `job_times.csv` của smoke: sửa `configs/estimates.yaml`, `python3 scripts/plan.py generate`, xem `summary`; plan đổi thì `plan_hash` đổi -> ghi vào DEVIATIONS.md. Nếu job không vừa session, chia lại bằng cách sinh plan mới trước khi phóng.
6. **Duyệt -> stage kế tiếp**: `bash scripts/set_stage.sh sweep`, commit, dry-run, release `r2`. Sau sweep và sweep2: `python3 scripts/pick_lr.py food101_r2_s1.zip food101_r5_s1.zip` ghi `configs/selected_lr.json` (cần commit trước khi phóng `p0`/`explore`, vì session chỉ thấy file trong commit). LR ở rìa lưới: mở thêm một bậc và ghi DEVIATIONS.md.
7. **Audit** trước khi vào bảng: `python3 scripts/audit_runs.py --runs runs_inbox/.../food101_r3_s1.zip runs_inbox/.../food101_r3_s2.zip --stage p0`. Điều tra mọi lỗi từ predictions trước khi sửa bảng (PREREG).
8. Session lỗi (`ERROR`) vẫn có zip các job đã xong. Không có retry tự động: đọc `logs/*.log`, `job_times.csv`, `logs/_summary.txt`, lập plan cho phần còn thiếu, gửi lại bằng tag `r<N>.<k>` (ví dụ `r3.1`). Mã thoát: 1 có job lỗi, 3 có job bị hoãn do hết giờ, 2 lỗi cấu hình/môi trường.

## Chạy thử cục bộ (không GPU, không mạng)

```bash
python3 scripts/plan.py check --regen && python3 scripts/plan.py summary
python3 -m unittest scripts/test_scripts.py
STAGE=smoke DRY=1 bash run_kaggle_food101.sh 1                      # chỉ in kế hoạch session
STAGE=smoke SMOKE_LOCAL=1 FOOD101_RUN_CMD="python3 scripts/fake_run_job.py" \
  FOOD101_PERSIST_ROOT=/tmp/f101p CI_OUTPUT_DIR=/tmp/f101w bash run_kaggle_food101.sh 1
```

## Biến môi trường của `run_kaggle_food101.sh`

`STAGE` (bắt buộc), `FOOD101_PERSIST_ROOT` (mặc định `/kaggle/temp/food101-persist`: outputs, logs, cache HF, checkpoint), `BUDGET_HOURS` (không nhận job mới nếu ước tính vượt giờ còn lại), `FOOD101_CACHE_DIR` (mặc định `$PERSIST/cache`), `FOOD101_NUM_WORKERS` (mặc định nproc/số GPU), `FOOD101_RUN_CMD`, `DRY`, `SMOKE_LOCAL`.

## Chưa kiểm chứng

Mọi thời gian trong plan là suy ra từ benchmark RTX 3090 (chưa đo T4). Việc dựng cache từ `ethz/food101` (`food101.data.ensure_cache`, ~5 GB tải + 101k ảnh) chạy được trong session CI (internet bật, không HF_TOKEN, giới hạn tốc độ) chưa thử. Lược đồ output mà audit và pick_lr đọc khớp mã `src/food101` ở thời điểm viết (id/gold/pred; metrics.dev.top1, metrics.test.top1); đổi mã thì kiểm lại. Kích thước zip mỗi session chưa đo. Checkpoint XAI (ViT-B/L seed 42) không nằm trong zip: cần quyết định riêng (xem DEVIATIONS.md).
