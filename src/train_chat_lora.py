import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from model_utils import resolve_model_name

# Expected JSONL records: {"image": "path/to/img.jpg", "instruction": "...", "output": "..."}


@dataclass
class ChatSample:
    image_path: str
    instruction: str
    output: str


class ChatDataset(Dataset):
    def __init__(self, root: str, jsonl_file: str, processor):
        self.root = root
        self.samples: List[ChatSample] = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                import json

                obj = json.loads(line)
                self.samples.append(
                    ChatSample(
                        image_path=obj["image"],
                        instruction=obj.get("instruction", ""),
                        output=obj.get("output", ""),
                    )
                )
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(os.path.join(self.root, s.image_path)).convert("RGB")
        # Build a single-turn prompt (adapt as needed for multi-turn)
        # Some Qwen instruct formats may require special tokens; keep simple here.
        prompt = s.instruction
        target = s.output
        return {"image": img, "prompt": prompt, "target": target}


def collate_fn(batch: List[Dict[str, Any]], processor, max_target_len=256):
    images = [b["image"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]
    # Combine prompt + expected output into one sequence for causal LM: "<prompt>\n<answer>"
    full_texts = [p + "\n" + t for p, t in zip(prompts, targets)]
    proc = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    # Labels identical to input_ids but mask prompt part (optional improvement: mask only prompt tokens)
    labels = proc["input_ids"].clone()
    return {**proc, "labels": labels}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        default=None,
        help="Direct HF model id (overrides model_profile if set)",
    )
    ap.add_argument(
        "--model_profile",
        default="vl_large",
        help="Profile key (vl_small, vl_medium, vl_large, text_small, text_large, vl_base)",
    )
    ap.add_argument("--data_root", default="datasets Adidas Samba")
    ap.add_argument("--jsonl", default="chat_dataset.jsonl")
    ap.add_argument("--output_dir", default="outputs/chat_lora")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--use_4bit", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    resolved = resolve_model_name(args.model_name, args.model_profile)
    processor = AutoProcessor.from_pretrained(resolved, trust_remote_code=True)
    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    dataset = ChatDataset(
        args.data_root, os.path.join(args.data_root, args.jsonl), processor
    )

    def _collate(b):
        return collate_fn(b, processor)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            if accelerator.is_local_main_process and step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} step {step} loss {loss.item():.4f} model={resolved}"
                )
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
    accelerator.print("LoRA chat fine-tune complete.")


if __name__ == "__main__":
    main()
