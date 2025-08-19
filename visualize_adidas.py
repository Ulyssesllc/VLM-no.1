"""Console visualization for adidas_dataset models (Q_cons_fusion / MLP_fusion).


Example:
    python visualize_adidas.py \
        --checkpoint checkpoints/best_model.pth \
        --model-type Q_cons_fusion \
        --train-csv adidas_dataset/labels.csv \
        --test-csv adidas_dataset/labels.csv \
        --images adidas_dataset \
        --num-samples 6 --top-k 5 --ascii-preview --color-ascii


"""

from __future__ import annotations
import os
import random
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertTokenizer

from Contrastive import Q_cons_fusion
from PIL import ImageDraw, ImageFont

try:
    from MLP import MLP_fusion  # type: ignore
except Exception:
    MLP_fusion = None  # fallback if import fails


def parse_args():
    p = argparse.ArgumentParser(
        description="Console visualize adidas_dataset predictions (no file outputs)"
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    p.add_argument(
        "--model-type", choices=["Q_cons_fusion", "MLP_fusion"], default="Q_cons_fusion"
    )
    p.add_argument("--train-csv", default="adidas_dataset/labels.csv")
    p.add_argument("--test-csv", default="adidas_dataset/labels.csv")
    p.add_argument(
        "--images", default="adidas_dataset", help="Directory with image files"
    )
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--text-column",
        default="category_name",
        help="Column used as text input where needed",
    )
    p.add_argument("--sample-strategy", choices=["random", "first"], default="random")
    # Console visualization flags (mirroring amazon script)
    # Tắt toàn bộ ASCII (người dùng không cần). Các tham số cũ bỏ.
    # Thêm tuỳ chọn hiển thị / lưu ảnh màu thật kèm overlay.
    p.add_argument(
        "--show", action="store_true", help="Mở cửa sổ xem ảnh với overlay kết quả"
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Thư mục để lưu ảnh gốc kèm overlay kết quả (tạo nếu chưa có)",
    )
    p.add_argument(
        "--no-overlay", action="store_true", help="Không vẽ overlay, chỉ copy ảnh"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Ngưỡng xác suất để phân loại (>= threshold => positive-label). Mặc định 0.45 theo yêu cầu.",
    )
    p.add_argument(
        "--positive-label",
        type=str,
        default="fake",
        help="Tên lớp dương (ví dụ 'fake'). Không phân biệt hoa thường.",
    )
    p.add_argument(
        "--only-decision",
        action="store_true",
        help="Chỉ in ra một dòng quyết định (real/fake) cho mỗi sample",
    )
    p.add_argument(
        "--decision-mode",
        choices=["threshold", "top1"],
        default="threshold",
        help="threshold: dùng ngưỡng p_pos; top1: luôn chọn lớp xác suất cao nhất làm decision",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_label_map(train_csv: str) -> Dict[str, int]:
    df = pd.read_csv(train_csv, sep=";")
    label_col = "label" if "label" in df.columns else "category_name"
    labels = list(df[label_col].unique())
    return {lab: i for i, lab in enumerate(labels)}


def load_samples(test_csv: str, num: int, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(test_csv, sep=";")
    if strategy == "first" or len(df) <= num:
        return df.head(num).copy()
    return df.sample(n=num, random_state=42).copy()


def load_image(path: str, transform):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), (0, 0, 0))
    return transform(img), img


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def strip_module(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_model(args, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "Q_cons_fusion":
        model = Q_cons_fusion()
    else:
        if MLP_fusion is None:
            raise RuntimeError(
                "MLP_fusion not available; cannot visualize this model type."
            )
        model = MLP_fusion()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model_state", ckpt)
    sd = strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warn] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(
            f"[Warn] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval().to(device)
    return model, device


def predict(model, device, batch_imgs, batch_text_inputs, model_type):
    with torch.no_grad():
        if model_type == "Q_cons_fusion":
            logits, _, _ = model(
                batch_imgs,
                batch_text_inputs["input_ids"],
                batch_text_inputs["attention_mask"],
            )
        else:
            try:
                logits = model(
                    batch_imgs,
                    batch_text_inputs["input_ids"],
                    batch_text_inputs["attention_mask"],
                )
            except Exception:
                logits = model(batch_imgs, batch_text_inputs["raw_text"])  # type: ignore
    return F.softmax(logits, dim=1)


def draw_overlay(pil_img: Image.Image, lines: List[str]) -> Image.Image:
    if pil_img.mode != "RGBA":
        base = pil_img.convert("RGBA")
    else:
        base = pil_img.copy()
    draw = ImageDraw.Draw(base)
    # Try to load a truetype font; fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, base.width // 40))
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    # Compute box size
    text_blocks = []
    max_w = 0
    total_h = 0
    for line in lines:
        w, h = draw.textsize(line, font=font)
        max_w = max(max_w, w)
        total_h += h + 2
        text_blocks.append((line, w, h))
    box_w = max_w + pad * 2
    box_h = total_h + pad * 2
    # Semi-transparent rectangle
    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
    base.paste(overlay, (0, 0), overlay)
    y = pad
    for line, w, h in text_blocks:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += h + 2
    return base.convert("RGB")


def main():
    args = parse_args()
    set_seed(args.seed)

    label_map = build_label_map(args.train_csv)
    label_map_inv = {v: k for k, v in label_map.items()}
    # Xác định lớp dương (positive)
    pos_label_norm = args.positive_label.strip().lower()
    positive_index = None
    for idx, name in label_map_inv.items():
        if str(name).lower() == pos_label_norm:
            positive_index = idx
            break
    if positive_index is None:
        # fallback: nếu không tìm thấy, chọn index 0
        positive_index = 0
        print(
            f"[Warn] Không tìm thấy lớp '{args.positive_label}', dùng lớp index 0: {label_map_inv[0]}"
        )
    # Xác định tên lớp còn lại (negative)
    if len(label_map_inv) == 2:
        negative_index = 1 - positive_index
        negative_label = label_map_inv[negative_index]
    else:
        # nếu nhiều hơn 2 lớp, negative_label chỉ để hiển thị
        negative_label = ",".join(
            [label_map_inv[i] for i in label_map_inv if i != positive_index]
        )
    samples = load_samples(args.test_csv, args.num_samples, args.sample_strategy)

    transform = build_transform()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    images = []
    origs = []  # (original_path, PIL.Image)
    texts: List[str] = []
    for _, row in samples.iterrows():
        img_path = (
            os.path.join(args.images, row["img_path"])
            if not os.path.isfile(row["img_path"])
            else row["img_path"]
        )
        t_img, pil_img = load_image(img_path, transform)
        images.append(t_img)
        origs.append((img_path, pil_img))
        texts.append(str(row.get(args.text_column, "")))

    batch_imgs = torch.stack(images)
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    tokenized["raw_text"] = texts

    model, device = load_model(args, num_classes=len(label_map))
    batch_imgs = batch_imgs.to(device)
    tokenized = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokenized.items()
    }
    probs = predict(model, device, batch_imgs, tokenized, args.model_type)

    for i, (_idx, row) in enumerate(samples.iterrows()):
        p = probs[i]
        topv, topi = p.topk(min(args.top_k, p.size(0)))
        topk_pairs = list(zip(topv.tolist(), topi.tolist()))
        pred_idx = int(topi[0])
        gt_col = "label" if "label" in row else "category_name"
        gt = row[gt_col]
        pred_name = label_map_inv.get(pred_idx, str(pred_idx))
        # Tính xác suất lớp dương & quyết định nhị phân
        pos_prob = (
            float(p[positive_index]) if positive_index < p.size(0) else float(p.max())
        )
        if args.decision_mode == "top1":
            # Quyết định = top1 luôn, p_pos vẫn báo cáo theo positive_index
            decision = (
                pred_name
                if len(label_map_inv) > 2 or pred_idx == positive_index
                else (
                    label_map_inv[positive_index]
                    if pred_idx == positive_index
                    else negative_label
                )
            )
        else:
            decision = (
                label_map_inv[positive_index]
                if pos_prob >= args.threshold
                else negative_label
            )
        topk_str = ", ".join(
            [
                f"{label_map_inv.get(idx, str(idx))}({val:.2f})"
                for val, idx in topk_pairs
            ]
        )

        if args.only_decision:
            # In một dòng: index, decision, p_pos, topk (label:prob,...)
            topk_inline = ";".join(
                [f"{label_map_inv.get(idx, idx)}:{val:.3f}" for val, idx in topk_pairs]
            )
            print(f"{i}\t{decision}\t{pos_prob:.4f}\t{topk_inline}")
        else:
            img_path_display = origs[i][0]
            print(f"--- Sample {i} ---")
            print(f"Image: {img_path_display}")
            print(f"GT | Pred(top1): {gt} | {pred_name}")
            print(
                f"Decision(threshold={args.threshold:.2f} on '{label_map_inv[positive_index]}'): {decision} (p_pos={pos_prob:.3f})"
            )
            print(f"TopK: {topk_str}")
            # In toàn bộ xác suất các lớp (nếu số lớp nhỏ hợp lý)
            if p.numel() <= 20:
                all_probs_line = ", ".join(
                    [
                        f"{label_map_inv.get(ci, str(ci))}:{p[ci].item():.3f}"
                        for ci in range(p.numel())
                    ]
                )
                print(f"All probs: {all_probs_line}")
            txt = texts[i]
            trunc_txt = txt[:300] + ("..." if len(txt) > 300 else "")
            print(f"Text: {trunc_txt}")

            # Lưu / hiển thị ảnh màu gốc nếu người dùng yêu cầu
            if args.show or args.save_dir:
                pil_img = origs[i][1].copy()
                if not args.no_overlay:
                    overlay_lines = [
                        f"GT: {gt}",
                        f"Pred: {pred_name}",
                        f"Decision: {decision}",
                        f"p_pos: {pos_prob:.3f}",
                    ]
                    # Thêm top1 / topk line
                    overlay_lines.append(
                        "TopK: "
                        + ", ".join(
                            [
                                f"{label_map_inv.get(idx, idx)}:{val:.2f}"
                                for val, idx in topk_pairs
                            ]
                        )
                    )
                    pil_img = draw_overlay(pil_img, overlay_lines)
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    out_name = os.path.basename(img_path_display)
                    out_path = os.path.join(args.save_dir, f"{i:02d}_" + out_name)
                    try:
                        pil_img.save(out_path)
                    except Exception as e:
                        print(f"[Warn] Không lưu được {out_path}: {e}")
                if args.show:
                    try:
                        pil_img.show(title=f"sample_{i}")
                    except Exception:
                        pass

    # Không lưu file – chỉ hiển thị theo yêu cầu


if __name__ == "__main__":
    main()
