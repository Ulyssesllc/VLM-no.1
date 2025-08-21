"""Simplified training script: always-on GAN + selectable discriminator.

Removed features: threshold sweep, bias metrics, original size metadata, dynamic discriminator
discovery, history JSON, class weighting, weighted sampler. Focus on core training speed.
"""

import os
import sys
import time
import random
import argparse
import multiprocessing

"""Path & import setup"""
# (imports kept at top to satisfy linters; path injection after stdlib imports)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:  # ensure project root resolvable
    sys.path.insert(0, ROOT_DIR)

# Third-party imports (after path injection so local packages resolve)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.cuda.amp import autocast, GradScaler  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
)  # noqa: E402
from tqdm import tqdm  # noqa: E402

from discriminator import (  # type: ignore  # noqa: E402
    Q_cons_fusion,
    compute_itc_loss,
    MLP_fusion,
    Q_former_fusion,
    Q_bottleneck,
    MoE,
)
from generator import (  # type: ignore  # noqa: E402
    InfoGANGenerator,
    ClusterGANGenerator,
    Pix2PixGenerator,
    BicycleGANGenerator,
    DiscoGANGenerator,
)
from process_data import MyData, build_label_map  # type: ignore  # noqa: E402
from config import CONFIG  # type: ignore  # noqa: E402

# ============================= Helper / Utilities ============================= #

os.makedirs(CONFIG.checkpoint_dir, exist_ok=True)
os.makedirs(CONFIG.log_dir, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)

MASTER_CSV = os.path.join("adidas_dataset", "labels.csv")
if not os.path.isfile(MASTER_CSV):
    raise FileNotFoundError(MASTER_CSV)
label_map = build_label_map(MASTER_CSV, label_column="label")


def args():  # kept name for backward CLI compatibility
    parser = argparse.ArgumentParser(description="Simplified training (GAN always on)")
    disc_choices = [
        "Q_cons_fusion",
        "MLP_fusion",
        "Q_former_fusion",
        "Q_bottleneck",
        "MoE",
    ]
    parser.add_argument(
        "--disc_model",
        "--model",
        dest="disc_model",
        type=str,
        default="Q_cons_fusion",
        choices=disc_choices,
        help="Tên mô hình discriminator (trong thư mục discriminator)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (override CONFIG.epochs)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=CONFIG.batch_size,
        help="Batch size (override CONFIG.batch_size)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=CONFIG.lr,
        help="Classifier learning rate (override CONFIG.lr)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.seed,
        help="Random seed (override CONFIG.seed)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=224,
        help="Kích thước resize ảnh (VD 224). Đặt <=0 để giữ kích thước gốc (có thể khác nhau).",
    )
    parser.add_argument(
        "--max_raw_hw",
        type=int,
        default=1600,
        help="Giới hạn cạnh dài tối đa ảnh gốc để downscale sớm tiết kiệm RAM (áp dụng khi giữ kích thước gốc).",
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "focal"],
        default="ce",
        help="Chọn hàm loss: ce hoặc focal (tùy chọn, vẫn nhẹ).",
    )
    parser.add_argument(
        "--class_weight",
        choices=["none", "auto"],
        default="none",
        help="Tự động gán trọng số lớp (inverse freq) cho CrossEntropy (bỏ qua khi focal).",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0, help="Gamma cho focal (nếu dùng)"
    )
    # --- GAN joint training options ---
    # GAN luôn bật: giữ cờ để không phá script cũ nhưng bỏ ý nghĩa (luôn True nội bộ)
    # --with_gan deprecated & ignored; GAN is always enabled now.
    parser.add_argument(
        "--gan_latent",
        type=int,
        default=128,
        help="Kích thước vector nhiễu cho generator",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="infogan",
        choices=["infogan", "clustergan", "pix2pix", "bicyclegan", "discogan"],
        help="Chọn kiến trúc generator (infogan | clustergan | pix2pix | bicyclegan | discogan)",
    )
    parser.add_argument(
        "--gan_lr", type=float, default=2e-4, help="Learning rate cho generator"
    )
    parser.add_argument(
        "--gan_steps",
        type=int,
        default=1,
        help="Số bước cập nhật generator mỗi batch classifier",
    )
    parser.add_argument(
        "--gen_label",
        type=str,
        default=None,
        help="Tên lớp target để generator sinh (mặc định lớp thiểu số)",
    )
    parser.add_argument(
        "--gen_weight",
        type=float,
        default=0.3,
        help="Trọng số loss synthetic vào tổng loss classifier",
    )
    parser.add_argument(
        "--adv_mode",
        choices=["reinforce", "confuse"],
        default="reinforce",
        help="Chiến lược huấn luyện generator: reinforce (củng cố lớp target) hoặc confuse (gây nhiễu)",
    )
    parser.add_argument(
        "--synthetic_ratio",
        type=float,
        default=0.25,
        help="Tỷ lệ số mẫu synthetic so với batch size thật mỗi batch",
    )
    parser.add_argument(
        "--select_metric",
        choices=["acc", "macro_f1", "bal_acc", "recall_minority"],
        default="acc",
        help="Tiêu chí chọn checkpoint tốt nhất.",
    )
    parser.add_argument(
        "--no_itc",
        action="store_true",
        help="Tắt ITC loss (bắt buộc nếu mô hình không trả về image/text feat).",
    )
    return parser.parse_args()


# train_size = int(0.8 * len(df))

# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])
# Datasets & loaders sẽ được khởi tạo sau khi parse args để dùng target_size động
train_data = None
test_data = None


def build_scheduler(optimizer, total_epochs: int):
    sched = CONFIG.scheduler
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
    return None


def build_class_weights(*_args, **_kwargs):  # deprecated placeholder
    return None


def focal_loss_fn(logits, targets, gamma=2.0):
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    probs = nn.functional.softmax(logits, dim=1)
    pt = probs[torch.arange(logits.size(0)), targets]
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def evaluate_simple(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            out = model(img, input_ids, attention_mask)
            logits = out[0] if isinstance(out, tuple) else out
            preds.extend(torch.argmax(logits, 1).cpu().tolist())
            targets.extend(label.cpu().tolist())
    return accuracy_score(targets, preds), preds, targets


def compute_metrics(preds, targets, num_classes, minority_index=None):
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    pr, rc, f1, sup = precision_recall_fscore_support(
        targets, preds, labels=list(range(num_classes)), zero_division=0
    )
    macro_f1 = f1.mean()
    if minority_index is None:
        # minority = class with min support
        minority_index = int(min(range(num_classes), key=lambda i: sup[i]))
    recall_min = rc[minority_index]
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    metrics = {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "recall_minority": recall_min,
        "per_class_precision": pr.tolist(),
        "per_class_recall": rc.tolist(),
        "per_class_f1": f1.tolist(),
        "support": sup.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def unified_train(model, dataloader, eval_loader, args, gen_label_index=None):
    """Train classifier with optional focal loss + always-on GAN augmentation.

    Only essential logic retained: core forward, optional synthetic batch each step,
    generator adversarial reinforcement/confuse modes, basic metric selection.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_focal = args.loss_type == "focal"
    # Placeholder for (optionally) weighted CE (weights injected via global _CLASS_WEIGHTS)
    ce_loss = None
    if not use_focal:
        weights = globals().get("_CLASS_WEIGHTS", None)
        if weights is not None:
            device_w = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ce_loss = nn.CrossEntropyLoss(weight=weights.to(device_w))
        else:
            ce_loss = nn.CrossEntropyLoss()

    params = (
        model.module.parameters() if hasattr(model, "module") else model.parameters()
    )
    optimizer = AdamW(params, lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = build_scheduler(optimizer, total_epochs=CONFIG.epochs)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)

    # GAN luôn bật
    img_size = args.target_size if args.target_size > 0 else 224

    GEN_CLASS = {
        "infogan": InfoGANGenerator,
        "clustergan": ClusterGANGenerator,
        "pix2pix": Pix2PixGenerator,
        "bicyclegan": BicycleGANGenerator,
        "discogan": DiscoGANGenerator,
    }
    gen_cls = GEN_CLASS[args.gen_model]
    generator = gen_cls(latent_dim=args.gan_latent, img_size=img_size)
    generator = generator.to(device)
    g_opt = AdamW(generator.parameters(), lr=args.gan_lr, weight_decay=1e-4)
    g_scaler = GradScaler(enabled=CONFIG.mixed_precision)

    best_score, best_epoch, no_improve = 0.0, 0, 0
    batch_size = CONFIG.batch_size

    stop_training = False
    for epoch in range(CONFIG.epochs):
        t0 = time.time()
        model.train()
        generator.train()
        total_loss = 0.0
        total_g_loss = 0.0
        train_preds = []
        train_targets = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG.epochs}"):
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=CONFIG.mixed_precision):
                out_all = model(img, input_ids, attention_mask)
                if isinstance(out_all, tuple):
                    logits = out_all[0]
                    logit_img = out_all[1] if len(out_all) > 1 else None
                    logit_text = out_all[2] if len(out_all) > 2 else None
                else:
                    logits = out_all
                    logit_img = logit_text = None
                if use_focal:
                    cls_loss = focal_loss_fn(logits, label, gamma=args.gamma)
                else:
                    cls_loss = ce_loss(logits, label)
                itc = (
                    compute_itc_loss(logit_img, logit_text)
                    if (
                        not args.no_itc
                        and logit_img is not None
                        and logit_text is not None
                    )
                    else 0.0
                )
                loss = cls_loss + itc

                # Synthetic augmentation
                if args.synthetic_ratio > 0 and gen_label_index is not None:
                    syn_count = int(batch_size * args.synthetic_ratio)
                    if syn_count > 0:
                        z = torch.randn(syn_count, args.gan_latent, device=device)
                        syn_imgs = generator(z)
                        mask_target = (
                            (label == gen_label_index).nonzero(as_tuple=False).flatten()
                        )
                        if len(mask_target) == 0:
                            ref_idx = torch.arange(
                                min(len(label), syn_count), device=device
                            )
                        else:
                            repeat_needed = (syn_count + len(mask_target) - 1) // len(
                                mask_target
                            )
                            ref_idx = mask_target.repeat(repeat_needed)[:syn_count]
                        syn_in_ids = input_ids[ref_idx]
                        syn_att = attention_mask[ref_idx]
                        syn_labels = torch.full(
                            (syn_count,),
                            gen_label_index,
                            dtype=torch.long,
                            device=device,
                        )
                        syn_out_all = model(syn_imgs, syn_in_ids, syn_att)
                        syn_logits = (
                            syn_out_all[0]
                            if isinstance(syn_out_all, tuple)
                            else syn_out_all
                        )
                        syn_cls = (
                            focal_loss_fn(syn_logits, syn_labels, gamma=args.gamma)
                            if use_focal
                            else ce_loss(syn_logits, syn_labels)
                        )
                        loss = loss + args.gen_weight * syn_cls

            scaler.scale(loss).backward()
            if CONFIG.grad_clip and CONFIG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            train_preds.extend(torch.argmax(logits, 1).cpu().tolist())
            train_targets.extend(label.cpu().tolist())

            # Generator update (freeze discriminator to avoid accumulating its grads)
            for p in model.parameters():
                p.requires_grad = False
            for _ in range(max(1, args.gan_steps)):
                g_opt.zero_grad(set_to_none=True)
                with autocast(enabled=CONFIG.mixed_precision):
                    z = torch.randn(batch_size, args.gan_latent, device=device)
                    fake_imgs = generator(z)
                    ref_idx2 = torch.randint(
                        0, input_ids.size(0), (batch_size,), device=device
                    )
                    ref_ids = input_ids[ref_idx2]
                    ref_att = attention_mask[ref_idx2]
                    fake_labels = torch.full(
                        (batch_size,), gen_label_index, dtype=torch.long, device=device
                    )
                    # Forward with grad (model params frozen) so generator gets signal
                    out_eval = model(fake_imgs, ref_ids, ref_att)
                    logits_eval = (
                        out_eval[0] if isinstance(out_eval, tuple) else out_eval
                    )
                    ce_g = nn.functional.cross_entropy(logits_eval, fake_labels)
                    g_loss = ce_g if args.adv_mode == "reinforce" else -ce_g
                g_scaler.scale(g_loss).backward()
                g_scaler.step(g_opt)
                g_scaler.update()
                total_g_loss += float(g_loss.detach())
            for p in model.parameters():
                p.requires_grad = True

        # ---- Evaluation ----
        train_acc = accuracy_score(train_targets, train_preds)
        test_acc, test_preds, test_targets = evaluate_simple(model, eval_loader, device)
        test_metrics = compute_metrics(
            test_preds,
            test_targets,
            num_classes=len(set(test_targets + train_targets)),
            minority_index=gen_label_index,
        )
        select_value = test_metrics[args.select_metric]
        if scheduler:
            if CONFIG.scheduler == "plateau":
                scheduler.step(1 - test_acc)
            else:
                scheduler.step()
        elapsed = time.time() - t0
        avg_loss = total_loss / len(dataloader)
        avg_g = total_g_loss / len(dataloader)
        log_message = (
            f"[GAN] Ep {epoch + 1}/{CONFIG.epochs} | GLoss {avg_g:.4f} | Loss {avg_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"TestAcc {test_metrics['acc']:.4f} | MacroF1 {test_metrics['macro_f1']:.4f} | BalAcc {test_metrics['bal_acc']:.4f} | "
            f"RecMin {test_metrics['recall_minority']:.4f} | Sel({args.select_metric}) {select_value:.4f} | LR {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )
        print(log_message)
        with open(
            os.path.join(CONFIG.log_dir, "train_log.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(log_message + "\n")
        if select_value > best_score:
            best_score = select_value
            best_epoch = epoch + 1
            no_improve = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if CONFIG.mixed_precision else None,
                "generator_state": generator.state_dict(),
                "g_optimizer_state": g_opt.state_dict(),
                "g_scaler_state": g_scaler.state_dict()
                if CONFIG.mixed_precision
                else None,
                "best_metric": best_score,
                "select_metric": args.select_metric,
                "config": CONFIG.__dict__,
                "disc_model": args.disc_model,
                "gen_model": args.gen_model,
            }
            torch.save(ckpt, os.path.join(CONFIG.checkpoint_dir, "best_model_gan.pth"))
            print(
                f"Saved best model ({args.select_metric} {best_score:.4f}) at epoch {epoch + 1}"
            )
        else:
            no_improve += 1
            if no_improve >= CONFIG.patience:
                print(
                    f"Early stopping tại epoch {epoch + 1} (best {args.select_metric} {best_score:.4f} @ epoch {best_epoch})"
                )
                # Save final state before break
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "generator_state": generator.state_dict(),
                        "best_metric": best_score,
                        "select_metric": args.select_metric,
                        "disc_model": args.disc_model,
                        "gen_model": args.gen_model,
                    },
                    os.path.join(CONFIG.checkpoint_dir, "early_stop_model_gan.pth"),
                )
                stop_training = True
        # Always save last checkpoint (lightweight overwrite)
        ckpt_last = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.state_dict(),
            "g_optimizer_state": g_opt.state_dict(),
            "best_metric": best_score,
            "select_metric": args.select_metric,
            "disc_model": args.disc_model,
            "gen_model": args.gen_model,
        }
        torch.save(ckpt_last, os.path.join(CONFIG.checkpoint_dir, "last_model_gan.pth"))
        if stop_training:
            break

    return best_score


def test(model, dataloader):  # retained for API compatibility
    return evaluate_simple(
        model, dataloader, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def build_discriminator(name: str, num_classes: int, device: torch.device):
    class QBottleWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Q_bottleneck()

        def forward(self, img, input_ids, attention_mask):
            logits, _aux, q1, q2 = self.inner(input_ids, attention_mask, img)
            return logits, q1, q2

    class MoEWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = MoE(num_classes=num_classes)

        def forward(self, img, input_ids, attention_mask):
            logits, _aux = self.inner(input_ids, attention_mask, img)
            return logits, logits, logits

    base = {
        "Q_cons_fusion": lambda: Q_cons_fusion(num_classes=num_classes),
        "MLP_fusion": lambda: MLP_fusion(num_classes=num_classes),
        "Q_former_fusion": lambda: Q_former_fusion(num_classes=num_classes),
        "Q_bottleneck": QBottleWrap,
        "MoE": MoEWrap,
    }
    if name not in base:
        raise ValueError(f"Unknown discriminator: {name}")
    return torch.nn.DataParallel(base[name]()).to(device)


if __name__ == "__main__":
    args = args()
    # Override CONFIG with CLI for reproducibility consistency
    CONFIG.seed = args.seed
    CONFIG.epochs = args.epochs
    CONFIG.batch_size = args.batch_size
    CONFIG.lr = args.lr
    seed_everything(CONFIG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_map)
    # Khởi tạo dataset với tham số độ phân giải động
    tgt_size = None if args.target_size <= 0 else args.target_size
    return_original = False  # metadata disabled in simplified version

    def collate_fn(batch):
        # Chuẩn hóa tensor chính
        images = torch.stack([b["image"] for b in batch])
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        out = {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }
        # Metadata (dạng list, không stack) nếu có
        if return_original and "orig_size" in batch[0]:
            out["orig_size"] = [b["orig_size"] for b in batch]
            out["orig_path"] = [b["orig_path"] for b in batch]
        return out

    train_dataset = MyData(
        MASTER_CSV,
        images_root="adidas_dataset",
        label_map=label_map,
        split="train",
        target_size=tgt_size,
        max_raw_hw=args.max_raw_hw,
        return_original=return_original,
    )
    test_dataset = MyData(
        MASTER_CSV,
        images_root="adidas_dataset",
        label_map=label_map,
        split="test",
        target_size=tgt_size,
        max_raw_hw=args.max_raw_hw,
        return_original=return_original,
    )

    # Positive label index no longer required (kept for backward compatibility in calls)
    positive_index = 0
    label_map_inv = {v: k for k, v in label_map.items()}

    # (Sẽ xác định gen_label_index sau khi có train_labels_list)

    # Class weights & sampler
    train_labels_list = list(train_dataset.df[train_dataset.label_column])
    # Determine target label for generator (luôn cần) + chuẩn bị class counts
    gen_label_index = None
    if args.gen_label is not None:
        name_norm = args.gen_label.lower().strip()
        for name, idx in label_map.items():
            if str(name).lower() == name_norm:
                gen_label_index = idx
                break
        if gen_label_index is None:
            print(
                f"[WARN] gen_label '{args.gen_label}' không tìm thấy -> dùng lớp thiểu số"
            )
    counts_tmp2 = {k: 0 for k in label_map.keys()}
    for lab in train_labels_list:
        counts_tmp2[lab] += 1
    if gen_label_index is None:
        minority_name = min(counts_tmp2, key=counts_tmp2.get)
        gen_label_index = label_map[minority_name]
    # Build inverse frequency weights if requested
    if args.class_weight == "auto":
        total = sum(counts_tmp2.values())
        class_weights = []
        for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
            freq = counts_tmp2[name]
            w = total / (len(label_map) * max(1, freq))
            class_weights.append(w)
        wt_tensor = torch.tensor(class_weights, dtype=torch.float32)
        # inject into CONFIG for visibility (optional)
        CONFIG.class_weights = class_weights  # type: ignore
    else:
        wt_tensor = None
    print(
        f"Generator target label index: {gen_label_index} ({list(label_map.keys())[gen_label_index]}) | adv_mode={args.adv_mode}"
    )
    sampler = None  # weighted sampler removed

    cpu_ct = multiprocessing.cpu_count()
    train_data = DataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=(sampler is None),
        num_workers=min(CONFIG.num_workers, cpu_ct),
        pin_memory=CONFIG.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    test_data = DataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=min(CONFIG.num_workers, cpu_ct),
        pin_memory=CONFIG.pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Original size stats omitted in simplified version

    print(
        f"CPUs: {cpu_ct} | Train batches: {len(train_data)} | Test batches: {len(test_data)} | target_size={tgt_size if tgt_size else 'native'} | "
        f"Epochs={CONFIG.epochs} | BatchSize={CONFIG.batch_size} | LR={CONFIG.lr} | Gen={args.gen_model} | Disc={args.disc_model}"
    )
    # Chọn discriminator (mapping-based)
    model = build_discriminator(args.disc_model, num_classes, device)

    # Provide weighted CE to unified_train via closure hack (monkey patch) if needed
    if "ce_loss" in unified_train.__code__.co_varnames:
        pass  # leave as is
    # Simpler: attach global for weights (used inside unified_train)
    if wt_tensor is not None:
        # Move to device later inside training if needed
        global _CLASS_WEIGHTS
        _CLASS_WEIGHTS = wt_tensor
    else:
        _CLASS_WEIGHTS = None

    # Monkey patch: redefine ce_loss inside unified_train scope not trivial; easier to wrap model training by adjusting loss calc
    # So we adapt by setting CONFIG flag and modify ce_loss creation earlier if needed (light approach)

    unified_train(model, train_data, test_data, args, gen_label_index)
