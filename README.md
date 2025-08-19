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
Script hiển thị xác suất & quyết định, có thể tạo overlay, grid, báo cáo JSON / HTML, hiển thị inline notebook (ảnh gốc hoặc overlay), hoặc kết xuất matplotlib. Không còn dùng ASCII mặc định.

### Điểm nổi bật
* Ngưỡng mặc định `--threshold = 0.40`.
* `--positive-label = fake` (p(fake) ≥ threshold => fake nếu bài toán 2 lớp).
* Overlay: GT, Pred(top1), Decision, p_pos, TopK.
* Chế độ quyết định: `threshold` hoặc `top1`.
* Xuất đa định dạng: ảnh rời, ảnh grid, JSON metadata, HTML nhúng base64, grid matplotlib, HTML inline notebook ảnh gốc.
* API lập trình: `run_visualization(args)` trả về `(meta_list, list_PIL_images, grid_image_or_None)`.

### Ví dụ nhanh
```bash
# In thông tin lên console (8 mẫu mặc định)
python visualize_adidas.py --checkpoint checkpoints/best_model.pth

# Lưu ảnh overlay + grid + JSON + HTML (không mở cửa sổ)
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --model-type Q_cons_fusion \
   --num-samples 12 --top-k 5 \
   --save-dir viz_out --export-grid \
   --json-report viz_out/results.json \
   --html-report viz_out/report.html

# Chỉ quyết định nhị phân (1 dòng / ảnh)
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --only-decision --num-samples 20

# Ảnh gốc không overlay
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --save-dir raw_out --no-overlay

# Hiển thị grid matplotlib (ví dụ notebook hoặc có display)
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --mpl-grid --grid-cols 6 --num-samples 18

# Inline HTML (ảnh gốc + thông số) trong notebook, không in console
python visualize_adidas.py --checkpoint checkpoints/best_model.pth --inline-html --no-console
```

### Dùng trong Notebook (API)
```python
from visualize_adidas import parse_args, run_visualization
args = parse_args()              # hoặc tự tạo namespace
args.num_samples = 10
args.no_console = True           # tránh spam
meta, imgs, grid = run_visualization(args)

# Hiển thị ảnh đầu tiên
display(imgs[0])

# Truy cập metadata một mẫu
print(meta[0])
```

Trường trong `meta[i]`:
```
{
   'index': int,
   'image': 'đường_dẫn_ảnh',
   'gt': 'nhãn_thật',
   'pred_top1': 'nhãn_top1',
   'decision': 'kết_quả_cuối',
   'pos_prob': float,            # p(positive_label)
   'topk': [ {'label': str, 'prob': float}, ... ],
   'text': 'nội_dung_văn_bản',
   'saved_path': '...'(tuỳ có nếu --save-dir)
}
```

### Tham số chính
| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--checkpoint` | (bắt buộc) | File `.pth` có `model_state`. |
| `--model-type` | Q_cons_fusion | `Q_cons_fusion` hoặc `MLP_fusion`. |
| `--train-csv` / `--test-csv` | adidas_dataset/labels.csv | CSV dữ liệu (phân tách `;`). |
| `--images` | adidas_dataset | Thư mục gốc ảnh. |
| `--num-samples` | 8 | Số mẫu lấy từ test CSV. |
| `--sample-strategy` | random | `random` hoặc `first`. |
| `--top-k` | 5 | Số lớp top-k hiển thị. |
| `--threshold` | 0.40 | Ngưỡng lớp dương (decision-mode=threshold). |
| `--positive-label` | fake | Tên lớp dương (case-insensitive). |
| `--decision-mode` | threshold | `threshold` / `top1`. |
| `--only-decision` | off | In gọn: 1 dòng / mẫu. |
| `--no-console` | off | Không in ra stdout. |
| `--show` | off | Mở ảnh (GUI) hoặc inline notebook (từng ảnh). |
| `--save-dir` | None | Lưu ảnh (overlay hoặc gốc nếu `--no-overlay`). |
| `--no-overlay` | off | Không vẽ khung text lên ảnh. |
| `--export-grid` | off | Tạo ảnh ghép `_grid.jpg` (khi có ảnh). |
| `--grid-cols` | 4 | Cột cho grid & matplotlib. |
| `--mpl-grid` | off | Grid matplotlib (inline hoặc cửa sổ). |
| `--json-report` | None | Ghi JSON metadata. |
| `--html-report` | None | Ghi HTML embedded base64. |
| `--inline-html` | off | Hiển thị HTML grid ảnh gốc + thông số (notebook). |
| `--text-column` | category_name | Cột văn bản đầu vào tokenizer. |

### Ghi chú quyết định
* 2 lớp: nếu `decision-mode=threshold` → so sánh p(positive) với ngưỡng.
* `decision-mode=top1`: chọn lớp có xác suất cao nhất (không dùng ngưỡng) nhưng vẫn báo cáo p_pos.

### Khi nào dùng từng tuỳ chọn
| Mục tiêu | Bật các cờ |
|----------|------------|
| Chỉ xem nhanh console | (mặc định) |
| Tạo bộ ảnh review | `--save-dir viz_out` |
| Một ảnh tổng hợp | `--export-grid --save-dir viz_out` |
| Phân tích / post-processing | `--json-report out.json` |
| Báo cáo chia sẻ nhanh | `--html-report report.html` |
| Notebook gallery (ảnh gốc) | `--inline-html --no-console` |
| Grid khoa học (matplotlib) | `--mpl-grid --grid-cols 6` |
| So sánh nhị phân gọn | `--only-decision` |

### Ví dụ kết hợp tối đa
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth \
   --num-samples 24 --top-k 5 \
   --save-dir viz_all --export-grid \
   --json-report viz_all/results.json \
   --html-report viz_all/report.html \
   --mpl-grid --inline-html --no-console
```

> Nếu chỉ muốn ảnh gốc + thông số trong notebook (không sửa file trên đĩa): dùng `--inline-html --no-overlay --no-console` và bỏ `--save-dir`.

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
Nếu bật `--threshold_sweep`, mỗi epoch sẽ đề xuất `Thr*` và `F1*` (F1 lớp dương). Dùng giá trị đó để điều chỉnh `--threshold` (mặc định 0.40) khi suy luận.

## Phát Hiện Lệch Lớp
Huấn luyện in cảnh báo nếu >90% dự đoán rơi vào lớp dương. Khi gặp:
1. Dùng `--weighted_sampler` hoặc `--use_class_weights`.
2. Chuyển sang `--loss_type focal`.
3. Kiểm tra lại cân bằng dữ liệu train (CSV). 
4. Điều chỉnh ngưỡng `--threshold` (xem sweep) hoặc chọn lớp dương khác.



## Giấy Phép
Phát hành theo MIT License. Xem chi tiết trong file `LICENSE`.
