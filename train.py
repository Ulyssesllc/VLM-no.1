import os
import json
import time
import random
import argparse
import multiprocessing
import pandas as pd  # noqa: F401 (được dùng gián tiếp khi build_label_map / đọc CSV)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from Contrastive import Q_cons_fusion, compute_itc_loss
from MLP import MLP_fusion
from Q_former import Q_former_fusion
from Q_bottleneck import Q_bottleneck
from MoE import MoE as MoE_model
from process_data import MyData, build_label_map
from config import CONFIG

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


def args():
    parser = argparse.ArgumentParser(description="Training script for multimodal model")
    parser.add_argument(
        "--model",
        type=str,
        default="Q_cons_fusion",
        choices=[
            "Q_cons_fusion",
            "MLP_fusion",
            "Q_former_fusion",
            "Q_bottleneck",
            "MoE",
        ],
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
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
        "--no_return_original",
        action="store_true",
        help="Không trả metadata ảnh gốc trong batch (tiết kiệm RAM).",
    )
    parser.add_argument(
        "--positive_label",
        type=str,
        default="real",
        help="Tên lớp coi là dương để tính threshold / metrics chi tiết",
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "focal"],
        default="ce",
        help="Chọn hàm loss: ce (CrossEntropy) hoặc focal",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma cho focal loss",
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Dùng trọng số lớp nghịch đảo tần suất trong loss",
    )
    parser.add_argument(
        "--weighted_sampler",
        action="store_true",
        help="Dùng WeightedRandomSampler để cân bằng batch",
    )
    parser.add_argument(
        "--threshold_sweep",
        action="store_true",
        help="Quét ngưỡng tối ưu F1 cho lớp dương mỗi epoch (validation/test)",
    )
    return parser.parse_args()


# train_size = int(0.8 * len(df))

# test_size = len(df) - train_size
# train_dataset, test_dataset = random_split(df, [train_size, test_size])
# Datasets & loaders sẽ được khởi tạo sau khi parse args để dùng target_size động
train_data = None
test_data = None


def build_scheduler(optimizer):
    if CONFIG.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG.epochs
        )
    if CONFIG.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
    return None


def build_class_weights(label_names_list, label_map):
    counts = {k: 0 for k in label_map.keys()}
    for lab in label_names_list:
        counts[lab] += 1
    total = sum(counts.values())
    freqs = {k: counts[k] / total for k in counts}
    inv = {k: 1.0 / (freqs[k] + 1e-9) for k in freqs}
    mean_inv = sum(inv.values()) / len(inv)
    scaled = {k: inv[k] / mean_inv for k in inv}
    return torch.tensor([scaled[k] for k in label_map.keys()], dtype=torch.float)


def focal_loss_fn(logits, targets, gamma=2.0, weight=None):
    ce = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
    probs = nn.functional.softmax(logits, dim=1)
    pt = probs[torch.arange(logits.size(0)), targets]
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def evaluate(model, dataloader, device, return_probs=True):
    model.eval()
    preds = []
    targets = []
    probs_all = []
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            out = model(img, input_ids, attention_mask)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            probs = nn.functional.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).detach().cpu().tolist())
            targets.extend(label.detach().cpu().tolist())
            if return_probs:
                probs_all.append(probs.cpu())
    acc = accuracy_score(targets, preds)
    if return_probs:
        probs_all = torch.cat(probs_all, dim=0)
        return acc, preds, targets, probs_all
    return acc, preds, targets, None


def sweep_threshold(probs, targets, positive_index):
    pos_scores = probs[:, positive_index].numpy()
    y_true = np.array([1 if t == positive_index else 0 for t in targets])
    best_f1 = -1
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (pos_scores >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def compute_bias_metrics(targets, preds, probs, positive_index, label_map_inv):
    label_indices = list(label_map_inv.keys())
    cm = confusion_matrix(targets, preds, labels=label_indices)
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=label_indices, zero_division=0
    )
    bal_acc = rec.mean() if len(rec) > 0 else 0.0
    pred_counts = np.bincount(preds, minlength=len(label_indices))
    pred_ratio = pred_counts / (pred_counts.sum() + 1e-9)
    pos_prob_mean = (
        probs[:, positive_index].mean().item()
        if (probs is not None and positive_index < probs.size(1))
        else 0.0
    )
    return {
        "confusion_matrix": cm.tolist(),
        "precision": {label_map_inv[i]: float(prec[i]) for i in range(len(prec))},
        "recall": {label_map_inv[i]: float(rec[i]) for i in range(len(rec))},
        "f1": {label_map_inv[i]: float(f1[i]) for i in range(len(f1))},
        "balanced_accuracy": float(bal_acc),
        "pred_ratio": {
            label_map_inv[i]: float(pred_ratio[i]) for i in range(len(pred_ratio))
        },
        "pos_prob_mean": pos_prob_mean,
    }


def train(
    model,
    dataloader,
    eval_loader,
    args,
    label_map_inv,
    positive_index,
    class_weight_tensor=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss_type == "focal":
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weight_tensor.to(device)
            if class_weight_tensor is not None
            else None
        )
    params = (
        model.module.parameters() if hasattr(model, "module") else model.parameters()
    )
    optimizer = AdamW(params, lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scheduler = build_scheduler(optimizer)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)
    best_acc = 0.0
    best_epoch = 0
    history = []
    no_improve = 0

    for epoch in range(CONFIG.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG.epochs}"):
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=CONFIG.mixed_precision):
                output, logit_img, logit_text = model(img, input_ids, attention_mask)
                if args.loss_type == "focal":
                    cls_loss = focal_loss_fn(
                        output,
                        label,
                        gamma=args.gamma,
                        weight=class_weight_tensor.to(device)
                        if class_weight_tensor is not None
                        else None,
                    )
                else:
                    cls_loss = criterion(output, label)
                loss = cls_loss + compute_itc_loss(logit_img, logit_text)
            scaler.scale(loss).backward()
            if CONFIG.grad_clip and CONFIG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            train_preds.extend(output.argmax(1).detach().cpu().tolist())
            train_targets.extend(label.detach().cpu().tolist())

        train_acc = accuracy_score(train_targets, train_preds)
        test_acc, preds_eval, targets_eval, probs_eval = evaluate(
            model, eval_loader, device
        )

        if scheduler:
            if CONFIG.scheduler == "plateau":
                scheduler.step(1 - test_acc)
            else:
                scheduler.step()

        # Extra metrics
        metrics_extra = {}
        metrics_bias = compute_bias_metrics(
            targets_eval, preds_eval, probs_eval, positive_index, label_map_inv
        )
        if args.threshold_sweep:
            best_thr, best_f1 = sweep_threshold(
                probs_eval, targets_eval, positive_index
            )
            metrics_extra["best_threshold"] = best_thr
            metrics_extra["best_f1_pos"] = best_f1
        if (
            len(label_map_inv) == 2
            and metrics_bias["pred_ratio"][label_map_inv[positive_index]] > 0.9
        ):
            print(
                "[Warn] Mô hình có dấu hiệu lệch về lớp dương -> cân nhắc sampler/weights/threshold."
            )

        elapsed = time.time() - t0
        avg_train_loss = total_loss / len(dataloader)
        lr_current = optimizer.param_groups[0]["lr"]
        log_message = (
            f"Epoch {epoch + 1}/{CONFIG.epochs} | Loss {avg_train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"TestAcc {test_acc:.4f} | BalAcc {metrics_bias['balanced_accuracy']:.4f} | PosProbMean {metrics_bias['pos_prob_mean']:.3f} | LR {lr_current:.2e} | {elapsed:.1f}s"
        )
        if "best_threshold" in metrics_extra:
            log_message += f" | Thr* {metrics_extra['best_threshold']:.2f} F1* {metrics_extra['best_f1_pos']:.3f}"
        print(log_message)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "balanced_accuracy": metrics_bias["balanced_accuracy"],
                "pos_prob_mean": metrics_bias["pos_prob_mean"],
                "precision": metrics_bias["precision"],
                "recall": metrics_bias["recall"],
                "f1": metrics_bias["f1"],
                "confusion_matrix": metrics_bias["confusion_matrix"],
                "lr": lr_current,
                "time_sec": elapsed,
                **metrics_extra,
            }
        )

        with open(
            os.path.join(CONFIG.log_dir, "train_log.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(log_message + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict()
                    if CONFIG.mixed_precision
                    else None,
                    "best_acc": best_acc,
                    "config": CONFIG.__dict__,
                },
                os.path.join(CONFIG.checkpoint_dir, "best_model.pth"),
            )
            print(f"Saved best model (acc {best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= CONFIG.patience:
                print(
                    f"Early stopping tại epoch {epoch + 1} (best acc {best_acc:.4f} @ epoch {best_epoch})"
                )
                break

    with open(os.path.join(CONFIG.log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return best_acc


def test(model, dataloader, check=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            if check:
                output, _, _ = model(img, input_ids, attention_mask)
            else:
                output = model(img, input_ids, attention_mask)
            preds.extend(output.argmax(1).detach().cpu().tolist())
            targets.extend(label.detach().cpu().tolist())
    return accuracy_score(targets, preds)


def train1(
    model,
    dataloader,
    eval_loader,
    args,
    label_map_inv,
    positive_index,
    class_weight_tensor=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.loss_type == "focal":
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weight_tensor.to(device)
            if class_weight_tensor is not None
            else None
        )
    optimizer = Adam(model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
    scaler = GradScaler(enabled=CONFIG.mixed_precision)
    history = []
    for epoch in range(CONFIG.epochs):
        model.train()
        total_loss = 0.0
        preds = []
        targets = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CONFIG.epochs}"):
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast(enabled=CONFIG.mixed_precision):
                output = model(img, input_ids, attention_mask)
                if isinstance(output, tuple):
                    output = output[0]
                if args.loss_type == "focal":
                    loss = focal_loss_fn(
                        output,
                        label,
                        gamma=args.gamma,
                        weight=class_weight_tensor.to(device)
                        if class_weight_tensor is not None
                        else None,
                    )
                else:
                    loss = criterion(output, label)
            scaler.scale(loss).backward()
            if CONFIG.grad_clip and CONFIG.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            preds.extend(output.argmax(1).detach().cpu().tolist())
            targets.extend(label.detach().cpu().tolist())
        train_acc = accuracy_score(targets, preds)
        test_acc, preds_eval, targets_eval, probs_eval = evaluate(
            model, eval_loader, device
        )
        metrics_bias = compute_bias_metrics(
            targets_eval, preds_eval, probs_eval, positive_index, label_map_inv
        )
        avg_loss = total_loss / len(dataloader)
        log_message = f"Epoch {epoch + 1}/{CONFIG.epochs} | Loss {avg_loss:.4f} | TrainAcc {train_acc:.4f} | TestAcc {test_acc:.4f} | BalAcc {metrics_bias['balanced_accuracy']:.4f}"
        print(log_message)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "balanced_accuracy": metrics_bias["balanced_accuracy"],
                "precision": metrics_bias["precision"],
                "recall": metrics_bias["recall"],
                "f1": metrics_bias["f1"],
                "confusion_matrix": metrics_bias["confusion_matrix"],
            }
        )
        with open(
            os.path.join(CONFIG.log_dir, "attention_log.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(log_message + "\n")
    with open(
        os.path.join(CONFIG.log_dir, "attention_history.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_map)
    # Khởi tạo dataset với tham số độ phân giải động
    tgt_size = None if args.target_size <= 0 else args.target_size
    return_original = not args.no_return_original

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

    # Positive label index detection
    label_map_inv = {v: k for k, v in label_map.items()}
    pos_name_norm = args.positive_label.lower().strip()
    positive_index = 0
    for name, idx in label_map.items():
        if str(name).lower() == pos_name_norm:
            positive_index = idx
            break

    # Class weights & sampler
    class_weight_tensor = None
    train_labels_list = list(train_dataset.df[train_dataset.label_column])
    if args.use_class_weights:
        class_weight_tensor = build_class_weights(train_labels_list, label_map)
    sampler = None
    if args.weighted_sampler:
        counts_tmp = {k: 0 for k in label_map.keys()}
        for lab in train_labels_list:
            counts_tmp[lab] += 1
        sample_weights = [1.0 / counts_tmp[lab] for lab in train_labels_list]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

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

    if tgt_size is None:
        # Thống kê kích thước gốc một số mẫu đầu
        ws, hs = [], []
        sample_n = min(32, len(train_dataset))
        for i in range(sample_n):
            s = train_dataset[i]
            if "orig_size" in s and s["orig_size"] is not None:
                w, h = s["orig_size"]
                ws.append(w)
                hs.append(h)
        if ws:
            print(
                f"Original size stats (first {len(ws)} imgs): mean {sum(ws) / len(ws):.1f}x{sum(hs) / len(hs):.1f} | min {min(ws)}x{min(hs)} | max {max(ws)}x{max(hs)}"
            )

    print(
        f"CPUs: {cpu_ct} | Train batches: {len(train_data)} | Test batches: {len(test_data)} | target_size={tgt_size if tgt_size else 'native'}"
    )
    if args.model == "Q_cons_fusion":
        model = Q_cons_fusion(num_classes=num_classes)
        model = torch.nn.DataParallel(model).to(device)
        train(
            model,
            train_data,
            test_data,
            args,
            label_map_inv,
            positive_index,
            class_weight_tensor,
        )
    elif args.model == "MLP_fusion":
        model = MLP_fusion(num_classes=num_classes)
        model = torch.nn.DataParallel(model).to(device)
        train1(
            model,
            train_data,
            test_data,
            args,
            label_map_inv,
            positive_index,
            class_weight_tensor,
        )
    elif args.model == "Q_former_fusion":
        model = Q_former_fusion(num_classes=num_classes)
        model = torch.nn.DataParallel(model).to(device)
        train1(
            model,
            train_data,
            test_data,
            args,
            label_map_inv,
            positive_index,
            class_weight_tensor,
        )
    elif args.model == "Q_bottleneck":

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, img, input_ids, attention_mask):
                logits, aux_loss, q1, q2 = self.inner(input_ids, attention_mask, img)
                return logits, q1, q2

        inner = Q_bottleneck()
        model = Wrapper(inner)
        model = torch.nn.DataParallel(model).to(device)
        train(
            model,
            train_data,
            test_data,
            args,
            label_map_inv,
            positive_index,
            class_weight_tensor,
        )
    else:  # MoE standalone

        class MoEWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, img, input_ids, attention_mask):
                logits, aux_loss = self.inner(input_ids, attention_mask, img)
                return logits, logits, logits

        inner = MoE_model(num_classes=num_classes)
        model = MoEWrapper(inner)
        model = torch.nn.DataParallel(model).to(device)
        train(
            model,
            train_data,
            test_data,
            args,
            label_map_inv,
            positive_index,
            class_weight_tensor,
        )
