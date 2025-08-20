# VLM no.1 – Adidas Fake vs Real Multimodal Classifier (with Synthetic GAN Augmentation)

This repository provides a simplified training pipeline for classifying adidas shoe images (fake vs real – or extended labels) using multimodal discriminators (image + text) with always-on synthetic image augmentation from lightweight GAN-style generators.

## Key Features
- Always-on generator producing synthetic images to bolster minority class.
- Selectable generator architectures: `infogan`, `clustergan`, `pix2pix`, `bicyclegan`, `discogan` (all simplified to noise→image).
- Selectable discriminator (fusion/classifier) models: `Q_cons_fusion`, `MLP_fusion`, `Q_former_fusion`, `Q_bottleneck`, `MoE`.
- Mixed precision (AMP) support.
- Early stopping + best/last/early-stop checkpoints.
- Visualization utility to inspect predictions / overlay results.

## Repository Structure
```
adidas_dataset/
  labels.csv              # CSV with metadata (semicolon separated) including columns: img_path;description;label;split
  fake/ ... real/ ...     # Image folders
utils/
  train.py                # Main training script (GAN integrated)
  process_data.py         # Dataset & tokenization
  config.py               # Global defaults (overridable via CLI)
  visualize_adidas.py     # Inference / reporting
generator/                # Simplified generator definitions
  infogan.py, clustergan.py, pix2pix.py, bicyclegan.py, discogan.py
  __init__.py
discriminator/            # Discriminator (fusion) models
requirements.txt
```

## Installation
1. (Recommended) Create a fresh environment:
```bash
conda create -n sentinels_v1 python=3.12 -y
conda activate sentinels_v1
```
2. Install PyTorch matching your hardware (examples – choose ONE):
```bash
# CPU only
pip install torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```
3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## Data Format
`adidas_dataset/labels.csv` must be semicolon (`;`) separated and contain at least:
- `img_path`: relative path to image (e.g. `fake/fake120.jpg`)
- `description`: (text column – can be empty string)
- `label`: class name (e.g. `fake`, `real`)
- `split`: one of `train` / `test` (or additional splits if extended)

## Training
Basic run (defaults):
```bash
python utils/train.py \
  --disc_model Q_cons_fusion \
  --gen_model infogan \
  --epochs 15 \
  --batch_size 16 \
  --lr 1e-4
```
Common useful flags:
- `--gen_model`: infogan|clustergan|pix2pix|bicyclegan|discogan
- `--gan_latent`: latent vector size (default 128)
- `--synthetic_ratio`: fraction of synthetic samples per batch (0.25 = 25%)
- `--gen_weight`: relative weight of synthetic classification loss
- `--adv_mode`: `reinforce` (push toward target class) or `confuse` (adversarial)
- `--gen_label`: specify target class name for generation (otherwise auto minority)
- `--loss_type`: `ce` or `focal` (with `--gamma`)
- `--target_size`: resize to fixed square (<=0 keeps original size with minimal augs)
- `--seed`, `--epochs`, `--batch_size`, `--lr`: override config defaults

Example (focus on minority class with more aggressive synthetic mix):
```bash
python utils/train.py \
  --disc_model Q_former_fusion \
  --gen_model pix2pix \
  --synthetic_ratio 0.4 \
  --gen_weight 0.4 \
  --gen_label fake \
  --epochs 20 \
  --batch_size 16
```

### Checkpoints
Saved to `checkpoints/`:
- `best_model_gan.pth`: highest validation/test accuracy.
- `last_model_gan.pth`: last epoch (always overwritten).
- `early_stop_model_gan.pth`: if early stopping triggered.

Each best checkpoint includes:
```
{
  'epoch', 'model_state', 'optimizer_state',
  'generator_state', 'g_optimizer_state',
  'scaler_state' (maybe), 'g_scaler_state' (maybe),
  'best_acc', 'config'
}
```

## Visualization / Inference
Use `visualize_adidas.py` after training:
```bash
python utils/visualize_adidas.py \
  --checkpoint checkpoints/best_model_gan.pth \
  --disc_model Q_cons_fusion \
  --num-samples 8 \
  --sample-strategy random \
  --show
```
Optional flags:
- `--threshold` & `--decision-mode` (threshold | top1)
- `--save-dir out_viz/` (saves individual images + optional grid)
- `--export-grid` (compose grid image)
- `--json-report results.json` or `--html-report report.html`

## Generator Details
All generator variants share a simple interface:
```python
gen = InfoGANGenerator(latent_dim=128, img_size=224)
imgs = gen(torch.randn(batch, 128))  # returns normalized tensor (ImageNet stats)
```
They're lightweight (no internal discriminator). Training loop treats them as conditional-on-text via reuse of sampled text tokens from real batch.

## Discriminator Interface
All discriminators are wrapped so that forward returns either:
- Tuple `(logits, img_repr, text_repr)`
- Or just `logits`
The training loop extracts `logits` for loss + uses optional reprs for contrastive ITC loss.

## Reproducibility
Set `--seed` to control Python, Torch (CPU/GPU) deterministic behavior (CuDNN deterministic mode enabled). Note: determinism can reduce speed.

## Extending
Add a new generator:
1. Create `generator/newgan.py` with a class `NewGANGenerator(latent_dim, img_size, out_channels=3)` and `.forward(z)`.
2. Import and add option name in `train.py` argument parser.

Add a new discriminator: place file in `discriminator/` and update selection logic in `train.py`.

## Troubleshooting
| Issue | Cause / Fix |
|-------|-------------|
| Import errors for torch/torchvision | Install PyTorch first (see Installation). |
| CUDA OOM | Reduce `--batch_size`, lower `--synthetic_ratio`, or decrease `--target_size`. |
| Training too slow | Use smaller `--batch_size`, fewer `--epochs`, faster generator (e.g. `pix2pix`). |
| No improvement / early stop | Lower `--gen_weight`, try different `--gen_model`, verify class balance. |
| Class imbalance still high | Increase `--synthetic_ratio` gradually (e.g. 0.25 -> 0.5). |

## Minimal Sanity Test
After install:
```bash
python utils/train.py --epochs 1 --batch_size 8 --gen_model infogan --disc_model Q_cons_fusion --synthetic_ratio 0.1
python utils/visualize_adidas.py --checkpoint checkpoints/last_model_gan.pth --disc_model Q_cons_fusion --num-samples 4 --show
```

## License
See `LICENSE` file.

## Acknowledgements
Simplified generator ideas adapted from classical GAN architectures (InfoGAN, ClusterGAN, Pix2Pix, BicycleGAN, DiscoGAN) but heavily reduced to inference-only modules.
