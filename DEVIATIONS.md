# DEVIATIONS (khung)

Ghi mọi thay đổi so với `docs/PLAN.md` / `PREREG.md` **khi chúng xảy ra**, kèm ngày, lý do, tác động, và commit/`plan_hash`. Không sửa ngược mục cũ; thêm mục mới.

| # | Ngày | Thay đổi | Lý do | Tác động lên kết luận | Commit / plan_hash |
|---|---|---|---|---|---|
| D0 | 2026-11-18 | LR sweep dùng proxy chung (20% train, 2 epoch) cho mọi model thay vì riêng ViT-L | cùng ngân sách tune | không | (đã ghi trong PLAN.md mục 4) |
| D1 | | Checkpoint ViT-B/ViT-L seed 42 cho XAI không nằm trong zip CI | yêu cầu "một zip, không checkpoint" | cần huấn luyện lại seed 42 trên máy khác hoặc quyết định riêng | |
| D2 | 2026-11-18 | Thêm stage `sweep2` (round **5**): sweep mở rộng của round 2. 3e-4 cho ViT-Ti/S/B/L, ConvNeXt-B, ResNet-50, DeiT-B; thêm 1e-3 cho ResNet-50. DINOv2 và CLIP giữ lưới cũ. Chạy TRƯỚC `p0` (r3) và `explore` (r4) | Round 2: 7/9 model có LR tốt nhất = 1e-4 (rìa lưới); ResNet-50 chỉ đạt 0.143 dev acc ở 1e-4. PREREG yêu cầu mở thêm một bậc cho model đó | Giá trị LR mở rộng được chọn **sau khi thấy dev acc của round 2** (chỉ dev, không dùng test), theo đúng quy tắc đã đăng ký; không đổi giao thức huấn luyện của p0. Nếu best vẫn ở rìa sau round 5 thì mở thêm (round 6) hoặc ghi hạn chế. Sweep proxy chỉ ~200 bước nên thiên về LR lớn: nêu trong Threats to validity | `plan_hash` 9cfa169b1d03c6d8 |
| D3 | 2026-11-18 | Sau smoke r1: `estimates.yaml` cập nhật bằng throughput đo; `open_clip` CLIP dùng `ViT-B-16-quickgelu` (trước là `ViT-B-16`); `.detach()` ở `logit_scale` | Số đo smoke r1; cảnh báo QuickGELU mismatch trong log | Ước tính thời gian/xếp session đổi (plan_hash đổi, config_hash job không đổi). CLIP zero-shot trong r1 chạy với arch cũ: không dùng số r1 | `b1ed2bd` |
| D4 | 2026-11-19 | Thêm stage `sweep3` (round **6**): mở rộng LR **lần cuối**. 1e-3 cho ViT-Ti/S, ConvNeXt-B, DeiT-B; 3e-3 cho ResNet-50 | Sau round 5: ViT-Ti/S, ConvNeXt-B, DeiT-B vẫn best=3e-4 và ResNet-50 best=1e-3 (0.585, còn tăng mạnh) đều ở rìa; ViT-B/L (best 1e-4, 3e-4 thấp hơn) và DINOv2/CLIP đã ở trong lưới | **Quy tắc cứng:** r5 và r6 là hai round mở rộng duy nhất. Sau r6 model nào vẫn ở rìa thì chấp nhận LR đó và ghi vào Threats to validity (nguy cơ tune thiếu cho CNN và ViT nhỏ làm lệch RQ2 và khoảng cách Ti/S→B). Giá trị mở rộng chọn sau khi thấy dev acc (không dùng test) | `plan_hash` 20a96037ba11d4fa |
| D5 | 2026-11-19 | Tách `p0` thành `p0a` (round **3**) và `p0b` (round **7**); `explore` (r4) rút gọn còn 6 job, DeiT-B FT chuyển sang `p0b` | `p0a`/`explore` chỉ gồm model đã có LR chốt trong lưới (ViT-B/L, DINOv2, CLIP) nên chạy song song với r6, không chờ. Phần còn lại cần LR sau r6 | Tổng job không đổi (35). `selected_lr.json` ghi từng phần (`pick_lr.py --models ready`); `plan.py check` chặn stage thiếu LR. Không ảnh hưởng giao thức | `plan_hash` 20a96037ba11d4fa |
| D6 | 2026-11-19 | Chốt LR sau round 6: `selected_lr.json` đủ 9 model. ViT-Ti/S, ConvNeXt-B, DeiT-B = 3e-4; ViT-B/L = 1e-4; DINOv2/CLIP = 3e-5; ResNet-50 = 3e-3 (vẫn ở rìa lưới, dev 0.6417, còn tăng từ 0.5847 ở 1e-3) | Theo quy tắc cứng D4: không mở round thêm, chấp nhận LR rìa | ResNet-50 có thể bị tune thiếu, nên khoảng cách ViT so với CNN (RQ2) có thể bị phóng đại: bắt buộc nêu trong Threats to validity. Sweep proxy ~200 bước thiên về LR lớn | `food101_r6_s1.zip`, audit ĐẠT 5/5 job |
| D7 | 2026-11-19 | Tách `p0a`/`explore`/`p0b` thành ba release r3, r4, r7 (D5) trái quy ước của bot: mỗi release là MỘT round, nhiều session, bot xếp hàng FIFO và không resume round đã kết thúc | Lúc phóng còn đủ tài khoản nên cả bốn session chạy ngay (r3: ZERNEAS, BI10; r4: BIN2; r7: BIN3), nếu thiếu tài khoản thì r4/r7 không xếp hàng cùng round với r3 | Không ảnh hưởng kết quả. Quy tắc từ nay: các việc độc lập phóng cùng lúc gom thành MỘT stage, MỘT round, nhiều session; chỉ stage phụ thuộc kết quả mới tách round | `25e5c41` và các commit stage r3, r4, r7 |
| D8 | 2026-11-19 | Thêm stage `xai` (round **8**, 2 session): huấn luyện lại Full FT seed 42 cho ViT-B, ConvNeXt-B, ViT-L bằng LR đã chốt, lưu `model_fp16.pt` (state_dict fp16) và chạy XAI ngay trong job (`--xai`): Attention Rollout (ViT), Grad-CAM, Integrated Gradients, deletion/insertion AUC trên 202 ảnh test cố định (2/lớp), sanity random hoá head + khối cuối. `pack_runs.py` cho phép đúng `model_fp16.pt` và `xai/*.npz` vào zip | D1: zip CI không có checkpoint nên không chạy được XAI ở máy khác; PREREG định ViT-B/L và ConvNeXt-B seed 42 làm model cho lỗi/XAI | Đây là huấn luyện LẠI cùng cấu hình với p0a/p0b (seed 42), không phải run thuộc bảng chính: số của bảng chính vẫn lấy từ r3/r7. Kết quả test top-1 của run lại được ghi trong `xai/summary.json` để đối chiếu; nếu lệch (GPU không tất định) thì báo cáo mức lệch và dùng run lại cho XAI/lỗi, không thay số bảng. Mẫu XAI, mốc, lưới 14x14 và baseline cố định trong mã trước khi chạy; kết quả XAI là giả thuyết, không phải nhân quả. Zip lớn hơn (~1 GB), `fetch-results --max-mb` mặc định 2000 | `plan_hash` 6ed1b24d7b409b03 |

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
