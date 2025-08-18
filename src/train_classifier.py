import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from data import RealFakeDataset, make_collate_fn
from tqdm import tqdm
from model_utils import resolve_model_name, load_processor_and_classifier


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        default=None,
        help="Direct HF model id (overrides model_profile if set)",
    )
    ap.add_argument(
        "--model_profile",
        default="vl_base",
        help="Shortcut profile key (vl_base, vl_small, vl_medium, vl_large, text_small, text_large)",
    )
    ap.add_argument("--data_root", default="datasets Adidas Samba")
    ap.add_argument("--csv", default="labels.csv")
    ap.add_argument("--output_dir", default="outputs/classifier")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--eval_split", type=float, default=0.1)
    ap.add_argument("--max_samples", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    resolved = resolve_model_name(args.model_name, args.model_profile)
    processor, model = load_processor_and_classifier(resolved, num_labels=2)

    csv_path = os.path.join(args.data_root, args.csv)
    dataset = RealFakeDataset(args.data_root, csv_path, args.max_samples)
    eval_len = int(len(dataset) * args.eval_split)
    train_len = len(dataset) - eval_len
    train_ds, eval_ds = random_split(dataset, [train_len, eval_len])

    collate = make_collate_fn(processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    eval_loader = (
        DataLoader(eval_ds, batch_size=args.batch_size, collate_fn=collate)
        if eval_len > 0
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * batch["labels"].size(0)
            preds = outputs.logits.argmax(-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
        train_acc = correct / total if total else 0

        if eval_loader:
            model.eval()
            eval_total, eval_correct, eval_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for batch in eval_loader:
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss_sum += loss.item() * batch["labels"].size(0)
                    preds = outputs.logits.argmax(-1)
                    eval_correct += (preds == batch["labels"]).sum().item()
                    eval_total += batch["labels"].size(0)
            eval_acc = eval_correct / eval_total if eval_total else 0
        else:
            eval_acc = train_acc

        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.print(
                f"Epoch {epoch}: train_acc={train_acc:.3f} eval_acc={eval_acc:.3f} model={resolved}"
            )
            if eval_acc > best_acc:
                best_acc = eval_acc
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(args.output_dir)
                processor.save_pretrained(args.output_dir)

    accelerator.print("Training complete. Best acc: %.3f" % best_acc)


if __name__ == "__main__":
    main()
