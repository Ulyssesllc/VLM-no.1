# Adidas Fake vs Real Multimodal Classification

Multimodal (image + text) classification pipeline migrated from a GLAMI-1M prototype to an adidas fake/real dataset. Supports multiple fusion architectures (contrastive, MLP fusion, Q-former style, bottleneck MoE variants, single-modality baselines).

## Dataset
Single CSV at `adidas_dataset/labels.csv` with semicolon delimiter:
```
img_path;description;label_id;label;split
```
- `img_path`: relative path to image inside `adidas_dataset/`
- `description`: product textual description (can be empty)
- `label`: class name (e.g. `fake` / `real`)
- `split`: `train`, `test` (and optionally `val` if added later)

Images are stored in subfolders (e.g. `adidas_dataset/fake/`, `adidas_dataset/real/`). The code only needs the relative path from the CSV.

## Installation
```bash
# (Optional) create a fresh environment
conda create -n adidas_vlm python=3.10 -y
conda activate adidas_vlm

# Install requirements
pip install -r requirements.txt
```

## Training (Multimodal)
```bash
python train.py --model Q_cons_fusion --epochs 15
```
Available `--model` options:
- `Q_cons_fusion` (default): ViT + multilingual BERT with contrastive + CE loss
- `MLP_fusion`: ResNet18 + BERT (simple fusion)
- `Q_former_fusion`: Query-based cross-attention fusion
- `Q_bottleneck`: Q-bottleneck + internal MoE (wrapper for training loop)
- `MoE`: Separate Cross-IT + MoE model (wrapped)

All use dynamic number of classes inferred from CSV.

Logs and checkpoints go to directories defined in `config.py` (see `CONFIG.log_dir`, `CONFIG.checkpoint_dir`).

## Single-Modality Baselines
```bash
python single.py --model-train Single_Text  --epochs 10
python single.py --model-train Single_Image --epochs 10
```

## Visualization
`visualize_adidas.py` hiển thị dự đoán trên một tập mẫu bằng ASCII (đen trắng hoặc màu true‑color) không tạo file.

Ví dụ cơ bản (dùng checkpoint tốt nhất sau huấn luyện):
```bash
python visualize_adidas.py \
   --checkpoint checkpoints/best_model.pth
```

Ví dụ đầy đủ hơn:
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
- `--checkpoint`: (bắt buộc) đường dẫn file mô hình `.pth`.
- `--model-type`: `Q_cons_fusion` | `MLP_fusion`.
- `--num-samples`: số mẫu hiển thị.
- `--sample-strategy`: `random` hoặc `first`.
- `--ascii-preview`: bật render ASCII.
- `--color-ascii`: dùng block màu (cần terminal hỗ trợ 24‑bit color).
- `--ascii-width`: chiều rộng ký tự khi scale ảnh.

Mẹo: nếu terminal bị “loang” màu sau block màu, chạy `reset` hoặc đảm bảo script in `\x1b[0m` (đã có sẵn).

## Cấu hình (config.py)
Tất cả siêu tham số tập trung trong `config.py` (dataclass `GlamiConfig`). Bạn thay đổi trực tiếp giá trị, sau đó chạy lại `train.py`.

Các trường chính:
- `batch_size`, `lr`, `epochs`, `weight_decay`
- `scheduler`: `cosine | plateau | none`
- `patience`: early stopping (dựa test acc hiện tại)
- `grad_clip`: gradient clipping nếu > 0
- `mixed_precision`: bật AMP (autocast + GradScaler)
- `num_workers`, `pin_memory`: DataLoader
- `log_dir`, `checkpoint_dir`

Muốn override qua CLI? (hiện chưa hỗ trợ) → có thể mở rộng bằng cách thêm các `add_argument` và gán vào `CONFIG` trước khi tạo DataLoader.

## Key Files
- `process_data.py`: Dataset + label_map utilities
- `train.py`: Main multimodal training entry point
- `single.py`: Image-only or text-only baseline trainer
- `Contrastive.py`, `MLP.py`, `Q_former.py`, `Q_bottleneck.py`, `MoE.py`: Model architectures
- `config.py`: Hyperparameters and paths

## Adding a New Model
1. Implement a class with forward signature `(image, input_ids, attention_mask)` returning either:
   - `(logits, img_feat, text_feat)` for contrastive loss, or
   - `logits` (then adapt `train1` or create a wrapper like existing ones).
2. Add import and option to the `--model` choices in `train.py`.

## Checkpoints
Best model saved as `CONFIG.checkpoint_dir/best_model.pth` with:
- model_state
- optimizer_state
- (optionally) scaler_state
- best_acc


## License
Released under the MIT License. See the `LICENSE` file for full text.
