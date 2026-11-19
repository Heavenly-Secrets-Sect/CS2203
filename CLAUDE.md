# CS2203: Food-101 + Vision Transformer

Nghiên cứu thực nghiệm có kiểm soát: LP, FT, LP-FT theo quy mô ViT-Ti đến L, và ViT so với CNN trên Food-101. Đồ án môn CS2203. **Đọc `README.md` trước** (hạ tầng chạy, bảng round và session, quy trình từng bước), rồi `PREREG.md` và `DEVIATIONS.md`. Repo **public** (session Kaggle clone đúng SHA không cần token), nên đừng đưa gì riêng tư vào đây.

## Lệnh (chạy cục bộ không cần GPU hay mạng)
- Kiểm tra kế hoạch khớp manifest: `python3 scripts/plan.py check --regen && python3 scripts/plan.py summary`
- Test script: `python3 -m unittest scripts/test_scripts.py`
- Test mô hình: `python3 -m pytest` (cần cài `torch` và `torchvision` theo hướng dẫn của PyTorch cho máy bạn; xem `requirements-dev.txt`). Không có torch thì chỉ chạy hai lệnh trên.
- Xem kế hoạch session mà không chạy: `STAGE=smoke DRY=1 bash run_kaggle_food101.sh 1`
- Đổi stage: `bash scripts/set_stage.sh <stage>` rồi commit `kaggle-run.yml` (mỗi release chạy **đúng một** stage).
- Tải kết quả của một round về `runs_inbox/` (cần `.env` của `message-talisman` trên máy chủ repo): `node --env-file=<message-talisman>/.env <message-talisman>/bot/scripts/fetch-results.mjs r<N> --out runs_inbox`. Kiểm kết quả trước khi vào bảng: `python3 scripts/audit_runs.py --runs <zip> --stage <stage>`.

## Quy tắc cứng
- **`sessions` và `STAGE` trong `kaggle-run.yml` phải khớp `scripts/run_plan.json`** (`plan.py check --regen` báo OK). Đổi stage bằng `set_stage.sh`, không sửa tay.
- **`PREREG.md` viết trước kết quả; sau khi chốt chỉ bổ sung phụ lục có ngày, không sửa ngược.** Mọi thay đổi so với kế hoạch ghi vào `DEVIATIONS.md` (D1, D2...) kèm lý do. Hiện `PREREG.md` còn là bản nháp chờ chủ repo chốt: đừng tự coi là đã chốt.
- **Publish một release tốn quota GPU của thành viên**: không tạo release, không chạy workflow `kaggle-run` khi chưa được hỏi. Dry-run (workflow chạy tay với `dry_run`) thì không tốn quota.
- Không dùng `kaggle kernels push` tay: chạy qua CI của nhóm (release, Worker `message-talisman`, session clone đúng SHA).
- Không commit: `.env*`, `kaggle.json`, checkpoint (`*.pt`, `*.ckpt`, `*.pth`), `runs_inbox/`, và các file riêng của người làm (`research/`, `plan.md`, `topic.md`, `PhieudanhgiaMonhoc.pdf`, đã nằm trong `.gitignore`).
- Zip kết quả `food101_r<N>_s<M>.zip` ở gốc repo là có chủ đích (README gọi `pick_lr.py` trên chúng); chúng bị `*.zip` ignore, đừng ép commit.
- Không in token, URL tải đã ký hay nội dung `.env`.

## Quy ước
- Đánh số theo **round và session** (cùng quy ước với `CS2229-fox`): tag release là `r<N>`, gửi lại phần lỗi của round N là `r<N>.1`, `r<N>.2`... Số round lấy từ `python3 scripts/plan.py summary`.
- PREREG quy định **một scorer duy nhất** tính chỉ số từ `test_logits`; khung của nó nằm ở `scripts/stats.py` (README ghi chưa dùng cho kết quả). Đừng tự tính chỉ số ở chỗ khác.

## Liên kết
- Note dự án trong vault KB: `knowledge-base/20_Nghien-cuu/cs2203-food101/cs2203-food101-index.md`
- Quy trình chạy Kaggle CI chung: `knowledge-base/70_Van-hanh/kaggle-ci-pipeline.md`
