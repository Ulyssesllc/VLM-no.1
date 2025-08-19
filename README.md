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
`visualize_adidas.py` currently prints predictions with ASCII previews.

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


## Troubleshooting
| Issue | Fix |
|-------|-----|
| OOM (CUDA) | Reduce `CONFIG.batch_size`, disable mixed precision, or use smaller model |
| Tokenizer download failure | Ensure internet for first run or cache the HuggingFace models |
| Class mismatch | Regenerate label_map by re-running `train.py` after editing CSV |

## License
Released under the MIT License. See the `LICENSE` file for full text.
