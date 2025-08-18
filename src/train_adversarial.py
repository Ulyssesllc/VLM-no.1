import os
import io
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from data import RealFakeDataset, make_collate_fn
from gan import DCGenerator, weights_init
from tqdm import tqdm

# Simple transform to resize GAN output to expected size (assuming 64x64 generation -> upsample if model expects larger)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2-VL")
    ap.add_argument("--data_root", default="datasets Adidas Samba")
    ap.add_argument("--csv", default="labels.csv")
    ap.add_argument("--output_dir", default="outputs/adversarial")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gan_latent", type=int, default=100)
    ap.add_argument(
        "--gan_ratio",
        type=float,
        default=0.3,
        help="fraction of batch replaced with GAN fake images",
    )
    ap.add_argument("--max_samples", type=int, default=None)
    return ap.parse_args()


def tensor_to_pil(t):
    t = (t.clamp(-1, 1) + 1) / 2  # to [0,1]
    t = t.mul(255).byte()
    return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, trust_remote_code=True
    )

    csv_path = os.path.join(args.data_root, args.csv)
    dataset = RealFakeDataset(args.data_root, csv_path, args.max_samples)
    eval_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - eval_len
    train_ds, eval_ds = random_split(dataset, [train_len, eval_len])

    collate = make_collate_fn(processor)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, collate_fn=collate)

    generator = DCGenerator(latent_dim=args.gan_latent)
    generator.apply(weights_init)
    generator.to(device)
    generator.eval()  # we are not training GAN here (could extend to update GAN)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            # Insert GAN fake samples
            bsz = batch["labels"].size(0)
            k = int(bsz * args.gan_ratio)
            if k > 0:
                with torch.no_grad():
                    z = torch.randn(k, args.gan_latent, 1, 1, device=device)
                    fake_imgs = generator(z)  # range [-1,1]
                    fake_pil = [tensor_to_pil(img) for img in fake_imgs]
                    fake_texts = ["synthetic counterfeit sample" for _ in range(k)]
                    proc = processor(
                        text=fake_texts, images=fake_pil, return_tensors="pt"
                    ).to(device)
                # Replace first k entries with synthetic (keys align with processor outputs)
                for key in proc:
                    if key in batch and proc[key].shape[0] == k:
                        batch[key][:k] = proc[key]
                batch["labels"][:k] = 0  # fake label
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
        # (Optional) evaluation omitted for brevity
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
    accelerator.print("Adversarial training complete")


if __name__ == "__main__":
    main()
