"""Configuration for adidas_dataset (fake vs real) multimodal training workflow."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class config:
    seed: int = 42
    # Reduced default batch size for safer GPU memory usage; override via CLI if needed
    batch_size: int = 16
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 5  # early stopping on val (test) accuracy
    scheduler: str = "cosine"  # cosine | plateau | none
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    log_dir: str = "adidas_logs"
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None  # path to checkpoint to resume


CONFIG = config()
