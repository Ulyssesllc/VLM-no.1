import os
from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers.modeling_outputs import SequenceClassifierOutput

MODEL_PROFILES: Dict[str, str] = {
    # classification / vision-language
    "vl_base": "Qwen/Qwen2-VL",
    "vl_small": "Qwen/Qwen2.5-VL-0.5B-Instruct",
    "vl_medium": "Qwen/Qwen2.5-VL-1.5B-Instruct",
    "vl_large": "Qwen/Qwen2.5-VL-7B-Instruct",
    # text-only instruct
    "text_small": "Qwen/Qwen2.5-1.5B-Instruct",
    "text_large": "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_model_name(model_name: Optional[str], model_profile: Optional[str]) -> str:
    if model_profile:
        if model_profile not in MODEL_PROFILES:
            raise ValueError(
                f"Unknown model_profile '{model_profile}'. Available: {list(MODEL_PROFILES.keys())}"
            )
        return MODEL_PROFILES[model_profile]
    if not model_name:
        return MODEL_PROFILES["vl_base"]
    return model_name


class CustomQwenClassifier(nn.Module):
    """Fallback classifier if sequence classification head absent."""

    def __init__(self, base_model: AutoModelForCausalLM, num_labels: int = 2):
        super().__init__()
        self.base_model = base_model
        hidden_size = (
            getattr(base_model.config, "hidden_size", None)
            or getattr(base_model.config, "hidden_sizes", [None])[0]
        )
        if hidden_size is None:
            raise ValueError("Cannot determine hidden size from base model config")
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, labels=None, **kwargs):
        outputs = self.base_model(output_hidden_states=True, **kwargs)
        hidden = outputs.hidden_states[-1]
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            mask = kwargs["attention_mask"]
            lengths = mask.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]
        else:
            pooled = hidden[:, -1]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return SequenceClassifierOutput(
            logits=logits,
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=getattr(outputs, "attentions", None),
        )


def load_processor_and_classifier(model_id: str, num_labels: int = 2):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels, trust_remote_code=True
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model = CustomQwenClassifier(base, num_labels=num_labels)
    return processor, model
