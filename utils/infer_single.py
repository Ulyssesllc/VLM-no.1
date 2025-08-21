"""Single image inference utility.

Usage example:
    python utils/infer_single.py \
        --checkpoint checkpoints/best_model_gan.pth \
        --disc_model Q_cons_fusion \
        --image path/to/image.jpg \
        --train-csv adidas_dataset/labels.csv \
        --top-k 5 --positive-label fake --threshold 0.4

Outputs:
 - Prints top-k predictions + decision line
 - Optional JSON metadata via --json-out
 - Optional overlay image via --save-overlay

Decision logic:
 - If decision-mode=top1: choose highest prob class (multi-class) OR positive vs negative in binary implicitly.
 - If decision-mode=threshold and binary: use p_pos >= threshold → positive-label else other.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
from typing import Dict, List
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import pandas as pd  # type: ignore
from transformers import BertTokenizer  # type: ignore

# --- Path bootstrap (ensure project root with discriminator/ exists) ---
_CWD = os.getcwd()
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOTS = {
    _CWD,
    os.path.dirname(_FILE_DIR),
    os.path.dirname(os.path.dirname(_FILE_DIR)),
}
for _p in list(_ROOTS):
    if _p and _p not in sys.path and os.path.isdir(os.path.join(_p, "discriminator")):
        sys.path.insert(0, _p)


try:
    from discriminator import (  # type: ignore
        Q_cons_fusion,
        MLP_fusion,
        Q_former_fusion,
        Q_bottleneck,
        MoE,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    print("[ImportError] Could not import discriminator modules. sys.path preview:")
    for i, p in enumerate(sys.path[:20]):
        print(f" {i}: {p}")
    raise e


def parse_args():
    p = argparse.ArgumentParser(description="Single image inference")
    p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    p.add_argument(
        "--disc_model",
        "--model",
        "--model-type",
        dest="disc_model",
        choices=[
            "Q_cons_fusion",
            "MLP_fusion",
            "Q_former_fusion",
            "Q_bottleneck",
            "MoE",
        ],
        default=None,
        help="(Optional) Discriminator name; if omitted sẽ đọc từ checkpoint hoặc tự đoán.",
    )
    p.add_argument(
        "--train-csv",
        default="adidas_dataset/labels.csv",
        help="CSV to build label map order",
    )
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--text", default="", help="Optional text/caption")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--positive-label",
        type=str,
        default="fake",
        help="Positive class name (binary)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Threshold for binary decision-mode=threshold",
    )
    p.add_argument(
        "--decision-mode", choices=["threshold", "top1"], default="threshold"
    )
    p.add_argument(
        "--no-softmax", action="store_true", help="Show logits instead of probabilities"
    )
    p.add_argument(
        "--save-overlay",
        type=str,
        default=None,
        help="Path to save overlay image (jpg/png)",
    )
    p.add_argument(
        "--json-out", type=str, default=None, help="Path to save JSON result"
    )
    p.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    return p.parse_args()


def build_label_map(train_csv: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv, sep=";")
    label_col = "label" if "label" in df.columns else "category_name"
    labels = list(df[label_col].unique())
    return {lab: i for i, lab in enumerate(labels)}


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image(path: str, transform):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = Image.open(path).convert("RGB")
    return transform(img), img


def strip_module(sd):
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


class _QBottleWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Q_bottleneck()

    def forward(self, img, input_ids, attention_mask):  # type: ignore
        logits, _aux, q1, q2 = self.inner(input_ids, attention_mask, img)
        return logits, q1, q2


class _MoEWrapper(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.inner = MoE(num_classes=num_classes)

    def forward(self, img, input_ids, attention_mask):  # type: ignore
        logits, _aux = self.inner(input_ids, attention_mask, img)
        return logits, logits, logits


def _build_model_by_name(name: str, num_classes: int):
    if name == "Q_cons_fusion":
        return Q_cons_fusion(num_classes=num_classes)
    if name == "MLP_fusion":
        return MLP_fusion(num_classes=num_classes)
    if name == "Q_former_fusion":
        return Q_former_fusion(num_classes=num_classes)
    if name == "Q_bottleneck":
        return _QBottleWrapper()
    if name == "MoE":
        return _MoEWrapper(num_classes=num_classes)
    raise ValueError(f"Unknown discriminator name: {name}")


def load_model(disc_name: str | None, ckpt_path: str, num_classes: int, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer explicit name: CLI > checkpoint meta > auto-detect
    if disc_name is None:
        disc_name = ckpt.get("disc_model")
    candidate_names = [
        "Q_cons_fusion",
        "MLP_fusion",
        "Q_former_fusion",
        "Q_bottleneck",
        "MoE",
    ]
    if disc_name is not None and disc_name not in candidate_names:
        print(
            f"[Warn] disc_model '{disc_name}' không hợp lệ trong checkpoint -> auto detect"
        )
        disc_name = None
    sd = strip_module(ckpt.get("model_state", ckpt))
    chosen = None
    load_missing = None
    load_unexpected = None
    # Auto detection: choose model minimizing (missing + unexpected)
    if disc_name is None:
        best_score = 1e18
        for name in candidate_names:
            model_try = _build_model_by_name(name, num_classes)
            missing, unexpected = model_try.load_state_dict(sd, strict=False)
            score = len(missing) + len(unexpected)
            if score < best_score:
                best_score = score
                chosen = (name, model_try, missing, unexpected)
        if chosen is not None:
            disc_name, model, load_missing, load_unexpected = chosen
        else:  # fallback
            disc_name = candidate_names[0]
            model = _build_model_by_name(disc_name, num_classes)
            load_missing, load_unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[AutoDetect] Dùng mô hình '{disc_name}' (missing={len(load_missing)}, unexpected={len(load_unexpected)})"
        )
    else:
        model = _build_model_by_name(disc_name, num_classes)
        load_missing, load_unexpected = model.load_state_dict(sd, strict=False)
    if load_missing:
        print(
            f"[LoadInfo] Missing keys: {load_missing[:5]}{'...' if len(load_missing) > 5 else ''}"
        )
    if load_unexpected:
        print(
            f"[LoadInfo] Unexpected keys: {load_unexpected[:5]}{'...' if len(load_unexpected) > 5 else ''}"
        )
    model.eval().to(device)
    # Print checkpoint metric if available
    best_metric = ckpt.get("best_metric") or ckpt.get("best_acc")
    sel = ckpt.get("select_metric")
    if best_metric is not None and sel:
        print(f"[CKPT] best {sel}: {best_metric:.4f}")
    return model, disc_name


def draw_overlay(img: Image.Image, lines: List[str]) -> Image.Image:
    base = img.convert("RGBA") if img.mode != "RGBA" else img.copy()
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, base.width // 40))
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    # measure
    blocks: List[tuple[str, int, int]] = []
    max_w = 0
    tot_h = 0
    for line in lines:
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            try:
                w, h = font.getsize(line)  # type: ignore[attr-defined]
            except Exception:
                w, h = (len(line) * 8, 14)
        blocks.append((line, w, h))
        max_w = max(max_w, w)
        tot_h += h + 2
    box_w = max_w + pad * 2
    box_h = tot_h + pad * 2
    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 150))
    base.paste(overlay, (0, 0), overlay)
    y = pad
    for line, w, h in blocks:
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += h + 2
    return base.convert("RGB")


def main():
    args = parse_args()
    device = (
        torch.device("cuda")
        if args.device in ("auto", "cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )
    label_map = build_label_map(args.train_csv)
    inv_map = {v: k for k, v in label_map.items()}
    transform = build_transform()

    # Text tokenizer (fallback silent)
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    except Exception:
        print("[Warn] Could not load BertTokenizer, using empty token stream.")
        tokenizer = None

    t_tensor, pil_img = load_image(args.image, transform)
    batch_img = t_tensor.unsqueeze(0).to(device)
    if tokenizer:
        tok = tokenizer(
            [args.text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
    else:
        # minimal fake tokens (assume model can handle zeros)
        input_ids = torch.zeros((1, 8), dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

    model, detected_name = load_model(
        args.disc_model, args.checkpoint, num_classes=len(label_map), device=device
    )

    with torch.no_grad():
        out = model(batch_img, input_ids, attention_mask)
        logits = out[0] if isinstance(out, tuple) else out
    probs = F.softmax(logits, dim=1) if not args.no_softmax else logits

    k = min(args.top_k, probs.size(1))
    topv, topi = probs.topk(k, dim=1)
    top_pairs = [(float(v), int(i)) for v, i in zip(topv[0], topi[0])]
    pred_idx = top_pairs[0][1]
    pred_name = inv_map.get(pred_idx, str(pred_idx))

    # Positive label logic
    pos_norm = args.positive_label.lower().strip()
    positive_index = next(
        (i for i, n in inv_map.items() if str(n).lower() == pos_norm), 0
    )
    if len(inv_map) == 2:
        negative_label = inv_map[1 - positive_index]
    else:
        negative_label = "other"
    if args.no_softmax:
        pos_prob = (
            float(F.softmax(probs[0], dim=0)[positive_index])
            if positive_index < probs.size(1)
            else 0.0
        )
    else:
        pos_prob = (
            float(probs[0, positive_index]) if positive_index < probs.size(1) else 0.0
        )
    if args.decision_mode == "top1":
        decision = (
            pred_name
            if len(inv_map) > 2
            else (
                inv_map[positive_index]
                if pred_idx == positive_index
                else negative_label
            )
        )
    else:
        decision = (
            inv_map[positive_index] if pos_prob >= args.threshold else negative_label
        )

    print("Image:", args.image)
    print("Prediction (top1):", pred_name)
    print(f"p_pos({inv_map.get(positive_index, positive_index)}): {pos_prob:.4f}")
    print("Decision:", decision)
    print("TopK:")
    for v, i in top_pairs:
        print(f"  {inv_map.get(i, i)}: {v:.4f}")

    overlay_path = None
    if args.save_overlay:
        lines = [
            f"Pred: {pred_name}",
            f"Decision: {decision}",
            f"p_pos: {pos_prob:.3f}",
            "TopK: "
            + ", ".join([f"{inv_map.get(i, i)}:{v:.2f}" for v, i in top_pairs]),
        ]
        ov = draw_overlay(pil_img, lines)
        os.makedirs(os.path.dirname(args.save_overlay) or ".", exist_ok=True)
        ov.save(args.save_overlay)
        overlay_path = args.save_overlay
        print("Overlay saved ->", overlay_path)

    if args.json_out:
        res = {
            "image": args.image,
            "pred_top1": pred_name,
            "decision": decision,
            "p_pos": pos_prob,
            "topk": [{"label": inv_map.get(i, i), "prob": v} for v, i in top_pairs],
            "overlay_path": overlay_path,
            "positive_label": inv_map.get(positive_index, positive_index),
            "threshold": args.threshold,
            "decision_mode": args.decision_mode,
            "no_softmax": args.no_softmax,
            "text": args.text,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print("JSON saved ->", args.json_out)


if __name__ == "__main__":
    main()
