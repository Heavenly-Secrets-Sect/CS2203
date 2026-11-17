# DEVIATIONS (khung)

Ghi mọi thay đổi so với `docs/PLAN.md` / `PREREG.md` **khi chúng xảy ra**, kèm ngày, lý do, tác động, và commit/`plan_hash`. Không sửa ngược mục cũ; thêm mục mới.

| # | Ngày | Thay đổi | Lý do | Tác động lên kết luận | Commit / plan_hash |
|---|---|---|---|---|---|
| D0 | 2026-11-18 | LR sweep dùng proxy chung (20% train, 2 epoch) cho mọi model thay vì riêng ViT-L | cùng ngân sách tune | không | (đã ghi trong PLAN.md mục 4) |
| D1 | | Checkpoint ViT-B/ViT-L seed 42 cho XAI không nằm trong zip CI | yêu cầu "một zip, không checkpoint" | cần huấn luyện lại seed 42 trên máy khác hoặc quyết định riêng | |

## Mẫu cho mục mới
- **Điều gì đổi** (so với bản nào):
- **Vì sao** (số liệu/log chứng minh):
- **Ảnh hưởng** (job/model/bảng nào; có phải chạy lại không):
- **Có sau khi thấy kết quả không?** (nếu có, đánh dấu thăm dò):

## Cần ghi nếu xảy ra
- LR tốt nhất ở rìa lưới và mở thêm một bậc (PREREG).
- Đổi `configs/estimates.yaml` làm `plan_hash` đổi (xếp lại session).
- Job bị hoãn do hết giờ, chạy lại bằng plan bổ sung.
- Revision HF của dataset/model thật sự dùng (xem `ci_env.json` trong zip).
- DINOv2 chạy ở 224 (khác 518 gốc); crop 256->224 cho mọi model khác `crop_pct` gốc.
