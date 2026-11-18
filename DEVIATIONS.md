# DEVIATIONS (khung)

Ghi mọi thay đổi so với `docs/PLAN.md` / `PREREG.md` **khi chúng xảy ra**, kèm ngày, lý do, tác động, và commit/`plan_hash`. Không sửa ngược mục cũ; thêm mục mới.

| # | Ngày | Thay đổi | Lý do | Tác động lên kết luận | Commit / plan_hash |
|---|---|---|---|---|---|
| D0 | 2026-11-18 | LR sweep dùng proxy chung (20% train, 2 epoch) cho mọi model thay vì riêng ViT-L | cùng ngân sách tune | không | (đã ghi trong PLAN.md mục 4) |
| D1 | | Checkpoint ViT-B/ViT-L seed 42 cho XAI không nằm trong zip CI | yêu cầu "một zip, không checkpoint" | cần huấn luyện lại seed 42 trên máy khác hoặc quyết định riêng | |
| D2 | 2026-11-18 | Thêm stage `sweep2` (round **5**): sweep mở rộng của round 2. 3e-4 cho ViT-Ti/S/B/L, ConvNeXt-B, ResNet-50, DeiT-B; thêm 1e-3 cho ResNet-50. DINOv2 và CLIP giữ lưới cũ. Chạy TRƯỚC `p0` (r3) và `explore` (r4) | Round 2: 7/9 model có LR tốt nhất = 1e-4 (rìa lưới); ResNet-50 chỉ đạt 0.143 dev acc ở 1e-4. PREREG yêu cầu mở thêm một bậc cho model đó | Giá trị LR mở rộng được chọn **sau khi thấy dev acc của round 2** (chỉ dev, không dùng test), theo đúng quy tắc đã đăng ký; không đổi giao thức huấn luyện của p0. Nếu best vẫn ở rìa sau round 5 thì mở thêm (round 6) hoặc ghi hạn chế. Sweep proxy chỉ ~200 bước nên thiên về LR lớn: nêu trong Threats to validity | `plan_hash` 9cfa169b1d03c6d8 |
| D3 | 2026-11-18 | Sau smoke r1: `estimates.yaml` cập nhật bằng throughput đo; `open_clip` CLIP dùng `ViT-B-16-quickgelu` (trước là `ViT-B-16`); `.detach()` ở `logit_scale` | Số đo smoke r1; cảnh báo QuickGELU mismatch trong log | Ước tính thời gian/xếp session đổi (plan_hash đổi, config_hash job không đổi). CLIP zero-shot trong r1 chạy với arch cũ: không dùng số r1 | `b1ed2bd` |

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
