# Adidas Authenticity Classification (GAN-augmented)

Dự án này huấn luyện bộ phân loại hình ảnh + văn bản (fusion) để phân biệt giày thật / giả (hoặc nhiều lớp) trên bộ `adidas_dataset`, với **GAN generator luôn bật** để sinh thêm ảnh synthetic cho lớp mục tiêu (thường là lớp thiểu số). Mã nguồn đã được tối giản: chỉ giữ lại các thành phần thiết yếu cho huấn luyện, đánh giá, trực quan hoá.

## 1. Kiến trúc tổng quan
- **Discriminator (classifier)**: Chọn một trong các mô hình trong thư mục `discriminator/`:
  - `Q_cons_fusion`, `MLP_fusion`, `Q_former_fusion`, `Q_bottleneck`, `MoE` (được wrap để trả về chuẩn `(logits, img_feat, txt_feat)` khi cần ITC).
- **Generator (always-on)**: Chuẩn hoá interface với các lựa chọn: `infogan`, `clustergan`, `pix2pix`, `bicyclegan`, `discogan` (chỉ dùng forward sinh ảnh từ latent `z`).
- **Huấn luyện kết hợp**: Mỗi batch:
  1. Forward batch thật → tính classification loss (+ optional ITC)
  2. Sinh ảnh synthetic cho label mục tiêu để tăng cường (weighted bởi `--gen_weight` & `--synthetic_ratio`)
  3. Cập nhật generator theo chế độ `reinforce` (tăng xác suất lớp target) hoặc `confuse` (giảm xác suất – tạo nhiễu) bằng tín hiệu CE từ discriminator (đã freeze tạm thời).
- **Metric lựa chọn checkpoint**: `--select_metric` trong `{acc, macro_f1, bal_acc, recall_minority}`.

## 2. Cấu trúc dữ liệu
```
adidas_dataset/
  labels.csv
  real/ ... (ảnh)
  fake/ ... (ảnh)
```
File `labels.csv` (phân tách bằng dấu chấm phẩy `;`) tối thiểu cần cột:
- `img_path`: đường dẫn tương đối hoặc tuyệt đối đến ảnh
- `label` hoặc `category_name`: tên lớp
- (Tùy chọn) `category_name` nếu bạn muốn dùng làm text input

Ví dụ (giản lược):
```
img_path;label;category_name
real/001.jpg;real;real
fake/010.jpg;fake;fake
```
Script nội suy `label_map` theo thứ tự gặp trong file.

## 3. Cài đặt môi trường
1. Cài đúng phiên bản PyTorch phù hợp phần cứng (ví dụ CUDA 12.1):
```
pip install torch==2.6.0+cu121 torchvision==0.21.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```
2. Cài các phụ thuộc còn lại:
```
pip install -r requirements.txt
```
3. (Tuỳ chọn) Tạo virtual env hoặc Conda env trước khi cài.

## 4. Huấn luyện
Script chính: `utils/train.py`

Ví dụ cơ bản:
```
python utils/train.py \
  --disc_model Q_cons_fusion \
  --gen_model infogan \
  --epochs 20 \
  --batch_size 16 \
  --synthetic_ratio 0.25 \
  --adv_mode reinforce \
  --select_metric macro_f1 \
  --class_weight auto
```
Các tham số chính:
- `--disc_model` : mô hình phân loại (xem danh sách ở trên)
- `--gen_model` : loại generator
- `--gan_latent` : kích thước vector nhiễu z (mặc định 128)
- `--gan_lr` : learning rate cho generator
- `--gan_steps` : số bước cập nhật generator mỗi batch
- `--synthetic_ratio` : tỷ lệ số mẫu synthetic / batch thật (0.25 nghĩa là ~25% số lượng batch real)
- `--gen_label` : tên lớp mà generator nhắm tới (mặc định lớp thiểu số tự động)
- `--adv_mode` : `reinforce` hoặc `confuse`
- `--gen_weight` : trọng số loss synthetic cộng vào classification loss
- `--loss_type` : `ce` hoặc `focal`
- `--class_weight` : `none` hoặc `auto` (inverse frequency cho CE)
- `--gamma` : tham số focal loss
- `--select_metric` : chọn metric để lưu checkpoint tốt nhất
- `--target_size` : resize ảnh về kích thước vuông (<=0 giữ kích thước gốc có giới hạn `--max_raw_hw`)
- `--no_itc` : tắt ITC loss nếu mô hình không trả về feature cặp

Checkpoints xuất vào `CONFIG.checkpoint_dir` (xem `utils/config.py`):
- `best_model_gan.pth` : checkpoint tốt nhất theo metric đã chọn
- `last_model_gan.pth` : epoch cuối cùng (ghi đè mỗi epoch)
- `early_stop_model_gan.pth` : nếu early stopping xảy ra

Log huấn luyện: `CONFIG.log_dir/train_log.txt`.

### Metric định nghĩa
- `acc`: accuracy
- `macro_f1`: trung bình F1 không trọng số giữa các lớp
- `bal_acc`: balanced accuracy (trung bình recall mỗi lớp)
- `recall_minority`: recall của lớp thiểu số (xác định lại mỗi phiên theo thống kê train)

### Gợi ý cân bằng & tốc độ
- Giảm `--synthetic_ratio` hoặc `--gan_steps` nếu GPU thiếu VRAM
- Dùng `--class_weight auto` + `focal` (khi imbalance nặng; focal bỏ qua weight per-class)
- Giảm `--target_size` (ví dụ 160/192) để tăng tốc

## 5. Trực quan hoá & Báo cáo
Script: `utils/visualize_adidas.py`

Ví dụ lấy 12 mẫu ngẫu nhiên và xuất overlay + grid + JSON:
```
python utils/visualize_adidas.py \
  --checkpoint checkpoints/best_model_gan.pth \
  --disc_model Q_cons_fusion \
  --num-samples 12 \
  --export-grid \
  --save-dir viz_out \
  --json-report viz_out/samples.json
```
Chế độ đánh giá toàn bộ (full eval + metrics JSON):
```
python utils/visualize_adidas.py \
  --checkpoint checkpoints/best_model_gan.pth \
  --disc_model Q_cons_fusion \
  --full-eval \
  --metrics-json viz_out/metrics.json
```
Tham số đáng chú ý:
- `--threshold` + `--decision-mode {threshold|top1}`: quyết định nhị phân (áp dụng khi thực sự chỉ có 2 lớp). Với multi-class + `top1` → decision = lớp xác suất cao nhất.
- `--positive-label`: tên lớp dương (mặc định "fake")
- `--no-softmax`: in raw logits (debug)
- `--no-overlay`: không vẽ thông tin lên ảnh
- `--export-grid`: tạo ảnh grid `_grid.jpg`
- `--json-report` / `--html-report`: xuất metadata hoặc báo cáo HTML
- `--full-eval`: duyệt toàn bộ test_csv (bỏ qua sampling) và in metrics
- `--metrics-json`: lưu metrics khi `--full-eval`

Trả về nội bộ (khi gọi như module) bộ 3 `(results_meta, list_PIL_images, grid_image_or_None)`.

## 6. ITC Loss (tuỳ chọn)
Nếu discriminator trả về tuple `(logits, img_feat, txt_feat)` thì script training có thể tính ITC loss (image-text contrastive). Dùng `--no_itc` để tắt. Với các wrapper MoE / bottleneck, feature có thể trùng lặp (giả lập) nên ITC không thật sự mang ý nghĩa lớn.

## 7. Cấu hình
Xem `utils/config.py` để điều chỉnh mặc định: batch_size, epochs, lr, patience, grad_clip, đường dẫn checkpoint/log, v.v. Tham số CLI sẽ ghi đè.

## 8. Xử lý lỗi thường gặp
| Vấn đề | Nguyên nhân | Cách khắc phục |
|--------|-------------|----------------|
| `FileNotFoundError: adidas_dataset/labels.csv` | Sai đường dẫn | Đảm bảo chạy từ root hoặc chỉnh `--train-csv` / `--test-csv` |
| Thiếu GPU / OOM | Batch & synthetic quá lớn | Giảm `--batch_size`, `--synthetic_ratio`, hoặc `--target_size` |
| `Missing keys` khi load checkpoint | Khác phiên bản mô hình | Bỏ qua nếu chỉ cảnh báo không ảnh hưởng logits, hoặc huấn luyện lại |
| Font `arial.ttf` không tồn tại | PIL không tìm thấy font | Thư viện fallback sẽ dùng font mặc định, hoặc cài thêm font |
| Hiệu năng thấp | I/O hoặc augment nặng | Giảm kích thước ảnh, giảm epochs, tắt ITC |

## 9. Mở rộng
- Có thể thêm generator mới chỉ cần cùng signature: `__init__(latent_dim, img_size)` và `forward(z)` → trả về batch ảnh `(B,3,H,W)`.
- Thêm discriminator mới: đảm bảo forward trả về `logits` hoặc tuple `(logits, feat_img, feat_txt)` để ITC chạy được.

## 10. License
Phân phối dưới giấy phép trong file `LICENSE`.

## 11. Gợi ý tái lập nhanh
```
# 1. Cài torch phù hợp (ví dụ CPU):
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
# 2. Cài phụ thuộc
pip install -r requirements.txt
# 3. Huấn luyện nhanh thử
python utils/train.py --epochs 2 --batch_size 4 --synthetic_ratio 0.1
# 4. Trực quan
python utils/visualize_adidas.py --checkpoint checkpoints/last_model_gan.pth --num-samples 4 --export-grid --save-dir demo_viz
```

---
Nếu cần bổ sung / rút gọn thêm cho README, cứ yêu cầu thêm nhé.
