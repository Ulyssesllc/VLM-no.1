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
```bash
python train.py --model Q_cons_fusion --epochs 15
```
Các tuỳ chọn `--model`:
- `Q_cons_fusion` (mặc định): ViT + BERT đa ngôn ngữ (loss contrastive + CrossEntropy)
- `MLP_fusion`: ResNet18 + BERT (fusion đơn giản)
- `Q_former_fusion`: Hợp nhất qua query cross-attention
- `Q_bottleneck`: Kiến trúc Q-bottleneck + MoE nội bộ (wrapper)
- `MoE`: Mô hình Cross-IT + MoE độc lập (wrapper)

Tất cả tự động suy ra số lớp từ CSV.

Log và checkpoint lưu tại thư mục trong `config.py` (`CONFIG.log_dir`, `CONFIG.checkpoint_dir`).

## Baseline Đơn Modal
```bash
python single.py --model-train Single_Text  --epochs 10
python single.py --model-train Single_Image --epochs 10
```

## Trực Quan Hoá
`visualize_adidas.py` hiển thị dự đoán trên một tập mẫu bằng ASCII (đen trắng hoặc true‑color) không tạo file.

Ví dụ cơ bản (dùng checkpoint tốt nhất):
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth
```

Ví dụ đầy đủ:
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --model-type Q_cons_fusion \
   --train-csv adidas_dataset/labels.csv \
   --test-csv adidas_dataset/labels.csv \
   --images adidas_dataset \
   --num-samples 6 \
   --top-k 5 \
   --sample-strategy first \
   --ascii-preview --color-ascii --ascii-width 48
```

Tham số chính:
- `--checkpoint`: (bắt buộc) đường dẫn file `.pth`.
- `--model-type`: `Q_cons_fusion` | `MLP_fusion`.
- `--num-samples`: số mẫu hiển thị.
- `--sample-strategy`: `random` hoặc `first`.
- `--ascii-preview`: bật hiển thị ASCII.
- `--color-ascii`: dùng block màu (cần terminal 24‑bit color).
- `--ascii-width`: độ rộng ký tự khi scale ảnh.

Mẹo: nếu màu sắc vẫn còn trong terminal sau khi in block màu, chạy `reset`.

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
- (nếu dùng AMP) `scaler_state`
- `best_acc`



## Giấy Phép
Phát hành theo MIT License. Xem chi tiết trong file `LICENSE`.
