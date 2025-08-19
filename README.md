# Phân Loại Đa Phương Thức Adidas Fake vs Real

Pipeline phân loại đa phương thức (ảnh + văn bản) được chuyển đổi từ prototype GLAMI-1M sang bộ dữ liệu adidas (fake / real). Hỗ trợ nhiều kiến trúc hợp nhất (contrastive, MLP fusion, Q-former style, bottleneck MoE, baseline đơn modality).

## Dữ Liệu
Một file CSV duy nhất tại `adidas_dataset/labels.csv` với dấu phân tách `;`:
```
img_path;description;label_id;label;split
```
- `img_path`: đường dẫn tương đối tới ảnh (bên trong `adidas_dataset/`)
- `description`: mô tả văn bản sản phẩm (có thể trống)
- `label`: tên lớp (`fake` / `real`)
- `split`: `train`, `test` (và có thể `val` nếu bổ sung sau)

Ảnh nằm trong các thư mục con (ví dụ `adidas_dataset/fake/`, `adidas_dataset/real/`). Code chỉ cần đường dẫn tương đối từ CSV.

## Cài Đặt
```bash
# (Tuỳ chọn) tạo môi trường mới
conda create -n adidas_vlm python=3.10 -y
conda activate adidas_vlm

# Cài thư viện còn lại
pip install -r requirements.txt
```

## Huấn Luyện (Đa Phương Thức)
Lệnh cơ bản:
```bash
python train.py --model Q_cons_fusion --epochs 15
```
### Kiến trúc (`--model`)
- `Q_cons_fusion` (mặc định): ViT + BERT + loss contrastive (ITC) + CE/Focal
- `MLP_fusion`: ResNet18 + BERT (fusion tuyến tính)
- `Q_former_fusion`: Query cross-attention style
- `Q_bottleneck`: Q-bottleneck + MoE nội bộ (bọc lại để có triple output)
- `MoE`: MoE độc lập (bọc để tương thích ITC loss)

Tự suy ra `num_classes` từ CSV.

### Điều Khiển Kích Thước Ảnh
| Tham số | Mô tả |
|---------|-------|
| `--target_size 224` | Resize đồng nhất (mặc định 224). |
| `--target_size 0` hoặc giá trị âm | Giữ kích thước gốc (native). Áp dụng downscale sớm nếu quá lớn. |
| `--max_raw_hw 1600` | Giới hạn cạnh dài ảnh gốc trước augment (native mode) để tiết kiệm VRAM/RAM. |
| `--no_return_original` | Bỏ trả metadata ảnh gốc (orig_size, orig_image_tensor) để giảm RAM. |

### Giảm Lệch Lớp & Nâng Cao Loss
| Tham số | Mặc định | Chức năng |
|---------|----------|-----------|
| `--use_class_weights` | off | Áp dụng trọng số nghịch đảo tần suất vào CE / Focal. |
| `--weighted_sampler` | off | Dùng `WeightedRandomSampler` cân bằng sampling per-batch. |
| `--loss_type ce|focal` | `ce` | Chọn CrossEntropy hoặc Focal. |
| `--gamma` | 2.0 | Tham số gamma của Focal. |
| `--positive_label real` | `real` | Đặt tên lớp dương để tính threshold sweep & thống kê. |
| `--threshold_sweep` | off | Quét ngưỡng (0.05→0.95) tìm F1 tốt nhất cho lớp dương mỗi epoch. |

Mỗi epoch log thêm: Balanced Accuracy, confusion matrix, precision/recall/F1 từng lớp, mean prob lớp dương, (nếu bật) ngưỡng & F1 tối ưu.

Ví dụ bật tối đa biện pháp giảm lệch:
```bash
python train.py \
   --model Q_cons_fusion \
   --epochs 20 \
   --loss_type focal --gamma 2.0 \
   --use_class_weights --weighted_sampler \
   --threshold_sweep --positive_label fake
```

### Ghi Log & Checkpoint
- Checkpoint tốt nhất: `CONFIG.checkpoint_dir/best_model.pth`
- Log huấn luyện: `CONFIG.log_dir/train_log.txt` + `history.json`
- Model attention/đơn giản (train1) ghi vào `attention_log.txt` / `attention_history.json`.

## Baseline Đơn Modal
```bash
python single.py --model-train Single_Text  --epochs 10
python single.py --model-train Single_Image --epochs 10
```

## Trực Quan Hoá / Suy Luận (`visualize_adidas.py`)
Script hiện tại tập trung hiển thị trực tiếp xác suất & quyết định và (tuỳ chọn) lưu / mở ảnh **màu gốc** kèm overlay, KHÔNG còn dùng ASCII mặc định.

### Thay đổi chính (so với phiên bản cũ)
- Mặc định `--threshold` = **0.45**.
- Mặc định `--positive-label` = **fake** (nếu xác suất lớp "fake" ≥ 0.45 → quyết định fake, ngược lại real khi chỉ có 2 lớp).
- Bỏ các tham số ASCII (`--ascii-preview`, `--color-ascii`, ...). Có thể tái bổ sung sau nếu cần.
- Thêm hỗ trợ hiển thị / lưu ảnh thật: `--show`, `--save-dir`, `--no-overlay`.

### Ví dụ cơ bản (in thông tin ra console)
```bash
python visualize_adidas.py --checkpoint checkpoints/best_model.pth
```

### Lưu ảnh có overlay dự đoán
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --model-type Q_cons_fusion \
   --num-samples 12 --top-k 5 \
   --save-dir viz_outputs
```

### Mở cửa sổ xem nhanh (nếu môi trường hỗ trợ GUI / desktop)
```bash
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --show --num-samples 4
```

### Chỉ copy ảnh gốc (không vẽ overlay)
```bash
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --save-dir raw_exports --no-overlay
```

### Chỉ in quyết định nhị phân (một dòng / ảnh)
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --only-decision --num-samples 20
```
Đầu ra dạng TSV:
```
index<TAB>decision<TAB>p_pos<TAB>label1:prob;label2:prob;...
```

### Các tham số quan trọng
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--checkpoint` | bắt buộc | Đường dẫn file `.pth` (chứa `model_state`). |
| `--model-type` | Q_cons_fusion | Kiểu mô hình: `Q_cons_fusion` hoặc `MLP_fusion`. |
| `--num-samples` | 8 | Số mẫu hiển thị / xử lý từ CSV test. |
| `--top-k` | 5 | Số lớp top-k in ra. |
| `--threshold` | 0.45 | Ngưỡng xác suất cho lớp dương (decision-mode=threshold). |
| `--positive-label` | fake | Tên lớp dương. Không phân biệt hoa thường. |
| `--decision-mode` | threshold | `threshold` hoặc `top1`. |
| `--only-decision` | off | Bật: chỉ một dòng / mẫu. |
| `--show` | off | Mở ảnh (GUI). |
| `--save-dir` | None | Lưu ảnh overlay vào thư mục. Tạo nếu chưa có. |
| `--no-overlay` | off | Khi lưu / show chỉ dùng ảnh gốc, không vẽ text. |

Overlay gồm: GT, Pred(top1), Decision, p_pos, TopK.

> Ghi chú: Nếu nhiều hơn 2 lớp, logic decision threshold sẽ so sánh p_pos với ngưỡng; phần còn lại được liệt kê trong TopK.

## Cấu Hình (`config.py`)
Mọi siêu tham số nằm trong `config.py` (dataclass `GlamiConfig`). Sửa giá trị rồi chạy lại `train.py`.

Trường chính:
- `batch_size`, `lr`, `epochs`, `weight_decay`
- `scheduler`: `cosine | plateau | none`
- `patience`: early stopping (dựa test accuracy)
- `grad_clip`: cắt gradient nếu > 0
- `mixed_precision`: bật AMP (autocast + GradScaler)
- `num_workers`, `pin_memory`: DataLoader
- `log_dir`, `checkpoint_dir`

Có thể mở rộng để truyền qua CLI (chưa hiện thực).

## Các File Chính
- `process_data.py`: Dataset & label_map
- `train.py`: Điểm vào huấn luyện đa modal
- `single.py`: Huấn luyện baseline đơn modal
- `Contrastive.py`, `MLP.py`, `Q_former.py`, `Q_bottleneck.py`, `MoE.py`: Kiến trúc mô hình
- `config.py`: Siêu tham số & đường dẫn

## Thêm Mô Hình Mới
1. Tạo lớp có `forward(image, input_ids, attention_mask)` trả về:
    - `(logits, img_feat, text_feat)` nếu muốn dùng thêm contrastive loss, hoặc
    - `logits` (khi đó dùng hàm huấn luyện kiểu `train1` hay wrapper tương tự).
2. Thêm import và tên vào danh sách `--model` trong `train.py`.

## Checkpoint
Checkpoint tốt nhất lưu tại `CONFIG.checkpoint_dir/best_model.pth` gồm:
- `model_state`
- `optimizer_state`
- `scaler_state` (nếu AMP)
- `best_acc`
- `config`

Lịch sử epoch đầy đủ: `history.json` (đa modal) hoặc `attention_history.json` (train1).

## Gợi Ý Tối Ưu Ngưỡng
Nếu bật `--threshold_sweep`, mỗi epoch sẽ đề xuất `Thr*` và `F1*` (F1 lớp dương). Dùng giá trị đó để điều chỉnh `--threshold` (mặc định 0.45) khi suy luận.

## Phát Hiện Lệch Lớp
Huấn luyện in cảnh báo nếu >90% dự đoán rơi vào lớp dương. Khi gặp:
1. Dùng `--weighted_sampler` hoặc `--use_class_weights`.
2. Chuyển sang `--loss_type focal`.
3. Kiểm tra lại cân bằng dữ liệu train (CSV). 
4. Điều chỉnh ngưỡng `--threshold` (xem sweep) hoặc chọn lớp dương khác.



## Giấy Phép
Phát hành theo MIT License. Xem chi tiết trong file `LICENSE`.
