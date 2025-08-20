"""Visualization / reporting for adidas_dataset models (simplified to match new training workflow).

Thay đổi chính so với phiên bản cũ:
 - Import mô hình từ package `discriminator.*` (cùng như train.py mới).
 - Thêm hỗ trợ các mô hình: Q_cons_fusion, MLP_fusion, Q_former_fusion, Q_bottleneck, MoE.
 - Tham số `--disc_model` (alias: `--model-type` hoặc `--model`) để tương thích ngược.
 - Tự xử lý tuple output (logits, ...); chỉ cần logits để suy luận.
 - Lược bỏ một số tuỳ chọn ít dùng; vẫn giữ threshold / top1 decision.

Ví dụ:
    python visualize_adidas.py --checkpoint checkpoints/best_model_gan.pth \
        --disc_model Q_cons_fusion --images adidas_dataset --num-samples 8 --show
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

# Discriminator imports (phù hợp train.py mới)
from discriminator.Contrastive import Q_cons_fusion  # type: ignore
from discriminator.MLP import MLP_fusion  # type: ignore
from discriminator.Q_former import Q_former_fusion  # type: ignore
from discriminator.Q_bottleneck import Q_bottleneck  # type: ignore
from discriminator.MoE import MoE as MoE_model  # type: ignore
from PIL import ImageDraw, ImageFont
import io
import base64
import json


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _notebook_display(img):  # pragma: no cover
    if not _in_notebook():
        return
    try:
        from IPython.display import display  # type: ignore

        display(img)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Console visualize adidas_dataset predictions (no file outputs)"
    )
    p.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint (.pth)"
    )
    p.add_argument(
        "--disc_model",
        "--model-type",
        "--model",
        dest="disc_model",
        choices=[
            "Q_cons_fusion",
            "MLP_fusion",
            "Q_former_fusion",
            "Q_bottleneck",
            "MoE",
        ],
        default="Q_cons_fusion",
        help="Tên mô hình discriminator dùng khi suy luận",
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
        "--export-grid",
        action="store_true",
        help="Xuất 1 ảnh tổng hợp (grid) các mẫu (dùng overlay nếu không tắt)",
    )
    p.add_argument(
        "--mpl-grid",
        action="store_true",
        help="Hiển thị grid bằng matplotlib (inline nếu notebook)",
    )
    p.add_argument(
        "--inline-html",
        action="store_true",
        help="Hiển thị inline HTML grid ảnh gốc + thông số (notebook).",
    )
    p.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Số cột cho grid khi --export-grid",
    )
    p.add_argument(
        "--html-report",
        type=str,
        default=None,
        help="Tạo file HTML nhúng base64 ảnh & thông tin dự đoán",
    )
    p.add_argument(
        "--json-report",
        type=str,
        default=None,
        help="Tạo file JSON metadata kết quả (đường dẫn / xác suất)",
    )
    p.add_argument(
        "--no-console",
        action="store_true",
        help="Không in thông tin mẫu ra stdout (dùng khi chỉ tạo báo cáo)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Ngưỡng xác suất quyết định lớp dương (binary).",
    )
    p.add_argument(
        "--positive-label",
        type=str,
        default="fake",
        help="Tên lớp dương (binary).",
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
    if torch.cuda.is_available():
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
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


class _QBottleWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Q_bottleneck()

    def forward(self, img, input_ids, attention_mask):  # type: ignore
        logits, aux, q1, q2 = self.inner(input_ids, attention_mask, img)
        return logits, q1, q2


class _MoEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = MoE_model()

    def forward(self, img, input_ids, attention_mask):  # type: ignore
        logits, aux = self.inner(input_ids, attention_mask, img)
        # Duplicate logits to mimic (logits, img_feat, txt_feat)
        return logits, logits, logits


def load_model(args, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.disc_model
    if name == "Q_cons_fusion":
        model = Q_cons_fusion(num_classes=num_classes)
    elif name == "MLP_fusion":
        model = MLP_fusion(num_classes=num_classes)
    elif name == "Q_former_fusion":
        model = Q_former_fusion(num_classes=num_classes)
    elif name == "Q_bottleneck":
        model = _QBottleWrapper()
    else:  # MoE
        model = _MoEWrapper()
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model_state", ckpt)
    sd = strip_module(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing and not args.no_console:
        print(
            f"[Warn] Missing keys (partial): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected and not args.no_console:
        print(
            f"[Warn] Unexpected keys (partial): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    model.eval().to(device)
    return model, device


def predict(model, device, batch_imgs, batch_text_inputs):
    with torch.no_grad():
        out = model(
            batch_imgs,
            batch_text_inputs["input_ids"],
            batch_text_inputs["attention_mask"],
        )
        logits = out[0] if isinstance(out, tuple) else out
    return F.softmax(logits, dim=1)


def draw_overlay(pil_img: Image.Image, lines: List[str]) -> Image.Image:
    if pil_img.mode != "RGBA":
        base = pil_img.convert("RGBA")
    else:
        base = pil_img.copy()
    draw = ImageDraw.Draw(base)
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
        # Pillow >=10: use textbbox; fallback to font.getbbox / getsize
        try:
            bbox = draw.textbbox((0, 0), line, font=font)  # (left, top, right, bottom)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            try:
                bbox = font.getbbox(line)  # type: ignore[attr-defined]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                try:
                    w, h = font.getsize(line)  # deprecated but fallback
                except Exception:
                    w, h = (len(line) * 8, 14)
        max_w = max(max_w, w)
        total_h += h + 2
        text_blocks.append((line, w, h))
    box_w = max_w + pad * 2
    box_h = total_h + pad * 2
    overlay = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 140))
    base.paste(overlay, (0, 0), overlay)
    y = pad
    for line, w, h in text_blocks:
        draw.text((pad, y), line, fill=(255, 255, 255, 255), font=font)
        y += h + 2
    return base.convert("RGB")


def run_visualization(args):
    """Trả về (list_meta, list_PIL_images, grid_image|None)."""
    set_seed(args.seed)
    label_map = build_label_map(args.train_csv)
    label_map_inv = {v: k for k, v in label_map.items()}
    pos_label_norm = args.positive_label.strip().lower()
    positive_index = next(
        (i for i, n in label_map_inv.items() if str(n).lower() == pos_label_norm), 0
    )
    if (
        positive_index == 0
        and str(label_map_inv[0]).lower() != pos_label_norm
        and not args.no_console
    ):
        print(f"[Warn] Không tìm thấy '{args.positive_label}', dùng {label_map_inv[0]}")
    if len(label_map_inv) == 2:
        negative_label = label_map_inv[1 - positive_index]
    else:
        negative_label = ",".join(
            [label_map_inv[i] for i in label_map_inv if i != positive_index]
        )
    samples = load_samples(args.test_csv, args.num_samples, args.sample_strategy)
    transform = build_transform()
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    images_t, origs, texts = [], [], []
    for _, row in samples.iterrows():
        raw_path = row["img_path"]
        img_path = (
            raw_path
            if os.path.isfile(raw_path)
            else os.path.join(args.images, raw_path)
        )
        t_img, pil_img = load_image(img_path, transform)
        images_t.append(t_img)
        origs.append((img_path, pil_img))
        texts.append(str(row.get(args.text_column, "")))
    if not images_t:
        return [], [], None
    batch_imgs = torch.stack(images_t)
    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
    )
    tokenized["raw_text"] = texts
    model, device = load_model(args, num_classes=len(label_map))
    batch_imgs = batch_imgs.to(device)
    tokenized = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tokenized.items()
    }
    probs = predict(model, device, batch_imgs, tokenized)
    results_meta, out_images = [], []
    for i, (_idx, row) in enumerate(samples.iterrows()):
        p = probs[i]
        topv, topi = p.topk(min(args.top_k, p.size(0)))
        topk_pairs = list(zip(topv.tolist(), topi.tolist()))
        pred_idx = int(topi[0])
        gt_col = "label" if "label" in row else "category_name"
        gt = row[gt_col]
        pred_name = label_map_inv.get(pred_idx, str(pred_idx))
        pos_prob = (
            float(p[positive_index]) if positive_index < p.size(0) else float(p.max())
        )
        if args.decision_mode == "top1":
            decision = (
                pred_name
                if (len(label_map_inv) > 2 or pred_idx == positive_index)
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
        rec = {
            "index": i,
            "image": origs[i][0],
            "gt": str(gt),
            "pred_top1": pred_name,
            "decision": decision,
            "pos_prob": pos_prob,
            "topk": [
                {"label": label_map_inv.get(idx, str(idx)), "prob": float(val)}
                for val, idx in topk_pairs
            ],
            "text": texts[i],
        }
        results_meta.append(rec)
        if not args.no_console:
            if args.only_decision:
                topk_inline = ";".join(
                    [
                        f"{label_map_inv.get(idx, idx)}:{val:.3f}"
                        for val, idx in topk_pairs
                    ]
                )
                print(f"{i}\t{decision}\t{pos_prob:.4f}\t{topk_inline}")
            else:
                topk_str = ", ".join(
                    [
                        f"{label_map_inv.get(idx, str(idx))}({val:.2f})"
                        for val, idx in topk_pairs
                    ]
                )
                print(
                    f"--- Sample {i} ---\nImage: {origs[i][0]}\nGT | Pred: {gt} | {pred_name}\nDecision(thr={args.threshold:.2f} '{label_map_inv[positive_index]}'): {decision} (p_pos={pos_prob:.3f})\nTopK: {topk_str}"
                )
        if (
            args.save_dir or args.export_grid or args.html_report or args.show
        ) and not args.no_overlay:
            pil_img = origs[i][1].copy()
            pil_img = draw_overlay(
                pil_img,
                [
                    f"GT: {gt}",
                    f"Pred: {pred_name}",
                    f"Decision: {decision}",
                    f"p_pos: {pos_prob:.3f}",
                    "TopK: "
                    + ", ".join(
                        [
                            f"{label_map_inv.get(idx, idx)}:{val:.2f}"
                            for val, idx in topk_pairs
                        ]
                    ),
                ],
            )
        else:
            pil_img = origs[i][1]
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out_path = os.path.join(
                args.save_dir, f"{i:02d}_" + os.path.basename(origs[i][0])
            )
            try:
                pil_img.save(out_path)
                rec["saved_path"] = out_path
            except Exception as e:
                if not args.no_console:
                    print(f"[Warn] Không lưu được {out_path}: {e}")
        out_images.append(pil_img)
    grid_img = None
    if args.export_grid and out_images:
        import math

        cols = getattr(args, "grid_cols", 4)
        w, h = out_images[0].size
        rows = math.ceil(len(out_images) / cols)
        grid_img = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
        for idx, im in enumerate(out_images):
            r, c = divmod(idx, cols)
            grid_img.paste(im, (c * w, r * h))
        if args.save_dir:
            try:
                grid_path = os.path.join(args.save_dir, "_grid.jpg")
                grid_img.save(grid_path)
            except Exception as e:
                if not args.no_console:
                    print(f"[Warn] Không lưu grid: {e}")
    # optional reports
    if args.json_report:
        try:
            with open(args.json_report, "w", encoding="utf-8") as f:
                json.dump(results_meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if not args.no_console:
                print(f"[Warn] Không ghi JSON: {e}")
    if args.html_report:
        try:
            rows_html = []
            for rec, img in zip(results_meta, out_images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                topk_html = ", ".join(
                    [f"{tk['label']}:{tk['prob']:.2f}" for tk in rec["topk"]]
                )
                safe_text = rec["text"][:120].replace("<", "&lt;")
                rows_html.append(
                    "<tr>"
                    f"<td>{rec['index']}</td>"
                    f"<td><img width='140' src='data:image/png;base64,{b64}'/></td>"
                    f"<td>{rec['gt']}</td>"
                    f"<td>{rec['pred_top1']}</td>"
                    f"<td>{rec['decision']}</td>"
                    f"<td>{rec['pos_prob']:.3f}</td>"
                    f"<td>{topk_html}</td>"
                    f"<td>{safe_text}</td>"
                    "</tr>"
                )
            html = (
                "<html><head><meta charset='utf-8'><title>Viz</title>"
                "<style>table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px}"
                "td,th{border:1px solid #999;padding:4px 6px;vertical-align:top}</style></head><body>"
                "<h2>Visualization Report</h2><table><thead><tr>"
                "<th>#</th><th>Image</th><th>GT</th><th>Pred</th><th>Decision</th><th>p_pos</th><th>TopK</th><th>Text</th>"
                "</tr></thead><tbody>"
                + "".join(rows_html)
                + "</tbody></table></body></html>"
            )
            with open(args.html_report, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:
            if not args.no_console:
                print(f"[Warn] Không tạo HTML: {e}")
    # inline HTML (original images + text, no overlay) for notebook
    if args.inline_html and out_images:
        if _in_notebook():
            try:
                from IPython.display import HTML, display  # type: ignore

                html_cards = []
                for rec, (orig_path, _orig_pil), p in zip(results_meta, origs, probs):
                    # original image (not overlay)
                    try:
                        with Image.open(orig_path).convert("RGB") as _tmp_im:
                            buf = io.BytesIO()
                            _tmp_im.save(buf, format="JPEG")
                            b64o = base64.b64encode(buf.getvalue()).decode("utf-8")
                    except Exception:
                        b64o = ""
                    topk_html = "<br>".join(
                        [f"{t['label']}:{t['prob']:.2f}" for t in rec["topk"]]
                    )
                    html_cards.append(
                        "<div style='margin:6px;border:1px solid #ccc;padding:6px;width:180px;font-size:12px;font-family:Arial;'>"
                        f"<div style='font-weight:bold'>{rec['decision']} ({rec['pos_prob']:.2f})</div>"
                        + (
                            f"<img src='data:image/jpeg;base64,{b64o}' style='width:160px;display:block;margin:4px auto;'/>"
                            if b64o
                            else "<div style='width:160px;height:120px;background:#eee'></div>"
                        )
                        + f"<div style='color:#555'>GT: {rec['gt']}</div>"
                        + f"<div>TopK:<br>{topk_html}</div>"
                        + "</div>"
                    )
                html_block = (
                    "<div style='display:flex;flex-wrap:wrap'>"
                    + "".join(html_cards)
                    + "</div>"
                )
                display(HTML(html_block))
            except Exception:
                if not args.no_console:
                    print("[Warn] Inline HTML thất bại")

    # show images if requested (after grid maybe)
    if args.show:
        if _in_notebook():
            for i, im in enumerate(out_images):
                _notebook_display(im)
            if grid_img:
                _notebook_display(grid_img)
        else:
            try:
                for im in out_images:
                    im.show()
                if grid_img:
                    grid_img.show()
            except Exception:
                if not args.no_console:
                    print("[Info] Headless environment: skip .show()")
    # Optional matplotlib grid
    if args.mpl_grid and out_images:
        try:
            import math
            import matplotlib.pyplot as plt  # type: ignore

            cols = getattr(args, "grid_cols", 4)
            rows = math.ceil(len(out_images) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            if not isinstance(axes, (list, tuple)):
                axes = axes.reshape(-1)
            axes_flat = axes.ravel()
            for idx, (ax, im, rec) in enumerate(
                zip(axes_flat, out_images, results_meta)
            ):
                ax.imshow(im)
                ax.set_title(f"{rec['decision']} ({rec['pos_prob']:.2f})", fontsize=8)
                ax.axis("off")
            for j in range(len(out_images), len(axes_flat)):
                axes_flat[j].axis("off")
            plt.tight_layout()
            try:
                plt.show()
            except Exception:
                if not args.no_console:
                    print("[Info] Headless environment: skip matplotlib show")
        except Exception as e:
            if not args.no_console:
                print(f"[Warn] matplotlib grid thất bại: {e}")
    return results_meta, out_images, grid_img


def main():
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
