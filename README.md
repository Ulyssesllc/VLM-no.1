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
Hiển thị dự đoán trên mẫu test bằng ASCII (đen trắng hoặc true‑color). Không tạo file.

Ví dụ cơ bản:
```bash
python visualize_adidas.py --checkpoint checkpoints/best_model.pth
```

Ví dụ đầy đủ với quyết định nhị phân và ngưỡng:
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --model-type Q_cons_fusion \
   --train-csv adidas_dataset/labels.csv \
   --test-csv adidas_dataset/labels.csv \
   --images adidas_dataset \
   --num-samples 8 --top-k 5 \
   --ascii-preview --ascii-width 48 --ascii-source original \
   --threshold 0.55 --positive-label fake --decision-mode threshold
```

Chỉ in một dòng/ảnh (phục vụ quét nhanh):
```bash
python visualize_adidas.py --checkpoint checkpoints/best_model.pth \
   --only-decision --num-samples 20 --threshold 0.6 --positive-label fake
```

### Các Tham Số Mới / Quan Trọng
| Tham số | Mặc định | Giải thích |
|---------|----------|------------|
| `--threshold 0.5` | 0.5 | Ngưỡng xác suất lớp dương (mode=threshold). |
| `--positive_label real` | real | Tên lớp coi là dương. Không phân biệt hoa thường. |
| `--decision-mode threshold|top1` | threshold | threshold: so sánh p_pos với ngưỡng; top1: luôn chọn lớp xác suất cao nhất. |
| `--only-decision` | off | Ghi 1 dòng: index, quyết định, p_pos, topK (label:prob;...). |
| `--ascii-source original|transformed` | original | Chọn ảnh gốc hay ảnh đã resize/augment cho ASCII. |
| `--ascii-preview` | off | Hiển thị ASCII (đen trắng hoặc màu). |
| `--color-ascii` | off | Bật true‑color block (môi trường cần hỗ trợ 24‑bit). |

Định dạng `--only-decision`:
```
idx<TAB>decision<TAB>p_pos<TAB>label1:prob;label2:prob;...
```

Mẹo: Nếu terminal bị “kẹt” màu sau preview màu, chạy `reset`.

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
Nếu bật `--threshold_sweep`, mỗi epoch sẽ đề xuất `Thr*` và `F1*` (F1 lớp dương). Dùng giá trị đó để đặt `--threshold` trong suy luận/triển khai.

## Phát Hiện Lệch Lớp
Huấn luyện in cảnh báo nếu >90% dự đoán rơi vào lớp dương. Khi gặp:
1. Dùng `--weighted_sampler` hoặc `--use_class_weights`.
2. Chuyển sang `--loss_type focal`.
3. Kiểm tra lại cân bằng dữ liệu train (CSV). 
4. Điều chỉnh ngưỡng `--threshold` (xem sweep) hoặc chọn lớp dương khác.



## Giấy Phép
Phát hành theo MIT License. Xem chi tiết trong file `LICENSE`.
