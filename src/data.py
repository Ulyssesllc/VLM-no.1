import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Dict, Any, List
import torch


class RealFakeDataset(Dataset):
    """Dataset for image + optional text (loaded later by processor).

    Expected CSV columns: image_path,text,label (0=fake,1=real)
    image_path is relative to root directory passed in.
    """

    def __init__(self, root: str, csv_file: str, max_samples: Optional[int] = None):
        self.root = root
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.head(max_samples)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path_rel = row["image_path"]
        img_path = os.path.join(self.root, img_path_rel)
        image = Image.open(img_path).convert("RGB")
        text = row.get("text", "") if not pd.isna(row.get("text", "")) else ""
        label = int(row["label"])
        return {"image": image, "text": text, "label": label}


def make_collate_fn(processor):
    def _collate(batch: List[Dict[str, Any]]):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        proc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        proc["labels"] = labels
        return proc

    return _collate
