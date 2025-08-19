"""Dataset utilities for adidas_dataset.

This replaces prior GLAMI-1M specific dataset handling.

Expected CSV schema (semicolon separated):
    img_path;description;label_id;label;split

Returned sample dictionary keys:
    image: FloatTensor [3,224,224]
    input_ids: LongTensor [seq_len]
    attention_mask: LongTensor [seq_len]
    label: LongTensor scalar (class index)

Splits: use split column values (train / val / test). The constructor's
`split` argument selects which rows to keep.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd
import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow loading truncated images safely
from transformers import BertTokenizer


class MyData(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_root: str = "adidas_dataset",
        label_map: Optional[Dict[str, int]] = None,
        split: str = "train",
        text_column: str = "description",
        label_column: str = "label",
        img_col: str = "img_path",
        max_length: int = 64,
        use_augmentation: Optional[bool] = None,
        max_raw_hw: int = 1600,  # if either H or W > this, downscale before augment
        target_size: Optional[
            int
        ] = 224,  # None -> giữ nguyên kích thước gốc (chỉ giới hạn bằng max_raw_hw)
        return_original: bool = True,  # kèm thêm ảnh gốc (dạng bytes) & kích thước vào sample
    ) -> None:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        # Explicit sep=';' for provided adidas labels file.
        self.df = pd.read_csv(csv_path, sep=";")
        if "split" in self.df.columns:
            self.df = self.df[
                self.df["split"].str.lower() == split.lower()
            ].reset_index(drop=True)
        self.images_root = images_root
        self.text_column = text_column
        self.label_column = label_column
        self.img_col = img_col
        # Build / use label map
        if label_map is None:
            labels = list(self.df[label_column].unique())
            self.label_map = {lab: i for i, lab in enumerate(labels)}
        else:
            self.label_map = label_map
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        # Decide augmentation usage: default -> only for train split
        if use_augmentation is None:
            use_augmentation = split.lower() == "train"
        self.max_raw_hw = max_raw_hw
        self.target_size = target_size
        self.return_original = return_original
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if self.target_size is None:
            # Keep original size (after potential max_raw_hw downscale), only lightweight augs
            if use_augmentation:
                aug_list = [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.15,
                                contrast=0.15,
                                saturation=0.15,
                                hue=0.03,
                            )
                        ],
                        p=0.4,
                    ),
                    transforms.RandomRotation(degrees=6),
                ]
            else:
                aug_list = []
            self.transform = transforms.Compose(
                aug_list + [transforms.ToTensor(), norm]
            )
        else:
            if use_augmentation:
                self.transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            self.target_size, scale=(0.85, 1.0)
                        ),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomApply(
                            [
                                transforms.ColorJitter(
                                    brightness=0.15,
                                    contrast=0.15,
                                    saturation=0.15,
                                    hue=0.03,
                                )
                            ],
                            p=0.5,
                        ),
                        transforms.RandomRotation(degrees=8),
                        transforms.ToTensor(),
                        norm,
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize((self.target_size, self.target_size)),
                        transforms.ToTensor(),
                        norm,
                    ]
                )

    def _load_image(self, rel_path: str):
        full = (
            rel_path
            if os.path.isabs(rel_path)
            else os.path.join(self.images_root, rel_path)
        )
        try:
            img = Image.open(full)
            raw_img = img.copy()  # giữ bản sao gốc trước augment/resize (có thể lớn)
            # Auto-orientation (EXIF)
            try:
                img = ImageOps.exif_transpose(img)  # type: ignore
                raw_img = ImageOps.exif_transpose(raw_img)  # type: ignore
            except Exception:
                pass
            if img.mode != "RGB":
                img = img.convert("RGB")
            if raw_img.mode != "RGB":
                raw_img = raw_img.convert("RGB")
            # Downscale very large raw images early to save RAM/time
            w, h = img.size
            if max(w, h) > self.max_raw_hw:
                scale = self.max_raw_hw / float(max(w, h))
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.BILINEAR)
        except Exception:
            # fallback empty image
            base_size = self.target_size if self.target_size else 224
            img = Image.new("RGB", (base_size, base_size), (0, 0, 0))
            raw_img = img.copy()

        transformed = self.transform(img)
        if not self.return_original:
            return transformed, None, None, None
        # tensor cho ảnh gốc (sau EXIF và chuyển RGB, trước augment crop/resize)
        orig_tensor = transforms.ToTensor()(raw_img)
        # serialize original (raw) image to bytes (PNG) to avoid size stacking issues
        try:
            with io.BytesIO() as buf:
                raw_img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
        except Exception:
            raw_bytes = None
        return transformed, raw_bytes, raw_img.size, orig_tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row.get(self.text_column, ""))
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze batch dim
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        label_name = row[self.label_column]
        label = self.label_map[label_name]
        image, orig_bytes, orig_size, orig_tensor = self._load_image(row[self.img_col])
        sample = {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.return_original:
            sample["orig_image_bytes"] = orig_bytes  # có thể None nếu lỗi đọc
            sample["orig_size"] = orig_size  # (W,H)
            sample["orig_path"] = row[self.img_col]
            sample["orig_image_tensor"] = orig_tensor  # Tensor [3,H,W]
        return sample


def build_label_map(csv_path: str, label_column: str = "label") -> Dict[str, int]:
    df = pd.read_csv(csv_path, sep=";")
    labels = list(df[label_column].unique())
    return {lab: i for i, lab in enumerate(labels)}
