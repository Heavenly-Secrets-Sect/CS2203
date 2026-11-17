# PREREG (bản nháp, chờ người dùng chốt)

> Viết **trước** khi có kết quả đầu tiên. Sau khi chốt, chỉ bổ sung bằng phụ lục có ngày, không sửa ngược. Ngày bản nháp: 2026-11-18.

## Câu hỏi
- **RQ1:** LP / FT / LP-FT khác nhau thế nào, và thay đổi ra sao theo quy mô Ti→S→B→L.
- **RQ2:** ViT-B so với ConvNeXt-B và ResNet-50.
- **RQ3:** Lỗi còn lại là thiếu năng lực hay nhãn nhập nhằng/nhiễu; model nhìn vào đâu.

## Chỉ số
Chính: top-1 trên test (25,250 ảnh). Phụ: top-5, macro-F1, NLL, ECE (15 bin), accuracy theo lớp. Tính từ `test_logits` bằng một scorer duy nhất.

## Tổng hợp
Mean ± SD mẫu (ddof=1) theo seed, chênh lệch tính từ điểm chưa làm tròn.

## Các family xác nhận (Holm trong từng family)
| Family | So sánh | Kiểm định |
|---|---|---|
| F1 Quy mô, Full FT | Ti vs S, S vs B, B vs L | Welch hai phía, n=3 mỗi nhóm |
| F3 Kiến trúc, Full FT | ViT-B vs ConvNeXt-B, ViT-B vs ResNet-50 | Welch hai phía, n=3 |

Với n=3, p lớn **không** được diễn giải là "không khác" (không có TOST ở mức B; nếu cần, ghi là hạn chế).

## So sánh bằng bootstrap (RQ1)
Paired bootstrap trên test, 10,000 lần, cùng ảnh, cho FT vs LP và FT vs LP-FT của ViT-B (từng seed, rồi trung bình), và ViT-L (FT vs LP). Báo cáo hiệu và CI 95%. Ghi rõ: bootstrap chỉ đo bất định do lấy mẫu test, không đo phương sai do huấn luyện. Paired t-test theo seed (df=2) chỉ là phụ.

## Thăm dò (không gắn "có ý nghĩa")
Pretraining (DeiT, DINOv2, CLIP, 1 seed), LP-FT ViT-L (1 seed), CLIP zero-shot, mọi phân tích ablation hoặc phát sinh sau khi thấy kết quả.

## Model chính cho lỗi và XAI
ViT-B/16 augreg Full FT và ViT-L/16 augreg Full FT, **seed 42** (run thuộc bảng chính). Đối chứng XAI: ConvNeXt-B seed 42.

## Quy tắc chọn LR
Sweep proxy chung: 20% train, 2 epoch, lưới {1e-5, 3e-5, 1e-4}, 1 seed (42), chọn theo **dev acc**. LR ở rìa lưới thì mở thêm một bậc cho model đó và ghi lại. Không dùng test để chọn bất cứ thứ gì.

## Từ ngữ
"Có mean cao hơn" khi chưa kiểm định hoặc không đạt ý nghĩa. "Khác biệt có ý nghĩa sau Holm" chỉ khi vượt tiêu chí trên. "Tốt nhất" phải so với mọi hàng trong bảng.

## Điều kiện loại hoặc điều tra
Acc gần ngẫu nhiên, NaN, hoặc một seed lệch xa: điều tra từ predictions trước khi sửa bảng, báo cáo như một phát hiện.

## Phụ lục 2026-11-18 (trước khi có kết quả)
- `grad_clip=1.0` được dùng cho **mọi model và mọi chế độ fine-tune** (thêm khi cài code; lý do: ViT-L fp16 dễ NaN). Không bật riêng cho model nào.
- Linear probe dùng đặc trưng không augment, weight decay L2 tune trên dev với lưới {0, 1e-6, 1e-5, 1e-4, 1e-3}.
- LP-FT dùng cùng LR với Full FT của model đó.
