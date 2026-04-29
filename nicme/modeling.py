"""Model construction helpers for custom and Hugging Face backbones."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput


class AutoBackboneForImageClassification(nn.Module):
    """Classifier wrapper for AutoModel image backbones such as DINOv3 pretrain models."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_state = outputs.last_hidden_state
            pooled = hidden_state.mean(dim=(-2, -1)) if hidden_state.dim() == 4 else hidden_state[:, 0]
        elif hasattr(outputs, "feature_maps") and outputs.feature_maps:
            pooled = outputs.feature_maps[-1].mean(dim=(-2, -1))
        else:
            raise ValueError("Backbone output does not expose pooler_output, last_hidden_state, or feature_maps")
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits, hidden_states=getattr(outputs, "hidden_states", None))


class TimmForImageClassification(nn.Module):
    """Hugging Face Trainer-compatible wrapper for timm image classifiers."""

    def __init__(self, model: nn.Module, num_labels: int):
        super().__init__()
        self.model = model
        self.num_labels = num_labels

    def forward(self, pixel_values=None, labels=None, **kwargs):
        if pixel_values is None:
            raise ValueError("pixel_values must be provided for timm image classification models")
        logits = self.model(pixel_values)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return ImageClassifierOutput(loss=loss, logits=logits)


def _parse_csv_list(value: str | list[str] | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return default
    return [part.strip() for part in text.split(",") if part.strip()]


def apply_lora_if_requested(model: nn.Module, script_args: Any) -> nn.Module:
    enabled = bool(getattr(script_args, "peft_enabled", False))
    if not enabled:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required when peft_enabled=True") from exc

    target_modules = _parse_csv_list(getattr(script_args, "peft_target_modules", None), ["query", "value"])
    modules_to_save = _parse_csv_list(getattr(script_args, "peft_modules_to_save", None), ["classifier"])
    config = LoraConfig(
        r=int(getattr(script_args, "peft_r", 8)),
        lora_alpha=int(getattr(script_args, "peft_alpha", 16)),
        target_modules=target_modules,
        lora_dropout=float(getattr(script_args, "peft_dropout", 0.1)),
        bias="none",
        modules_to_save=modules_to_save,
    )
    return get_peft_model(model, config)


def build_model(script_args: Any, class_names: list[str] | None = None) -> nn.Module:
    """Build a model from legacy custom or new Hugging Face backends."""
    backend = getattr(script_args, "model_backend", "custom").lower()
    num_labels = int(script_args.num_labels)
    id2label = {idx: name for idx, name in enumerate(class_names or [str(i) for i in range(num_labels)])}
    label2id = {name: idx for idx, name in id2label.items()}

    if backend == "custom":
        from model.convnext import ConvNextConfig, ConvNextForImageClassification
        from model.ResNet import ResNetConfig, ResNetForImageClassification

        if script_args.model_type.lower() == "resnet":
            config = ResNetConfig(num_labels=num_labels, depths=[3, 4, 6, 3])
            model = ResNetForImageClassification(config)
        elif script_args.model_type.lower() == "convnext":
            config = ConvNextConfig(num_labels=num_labels)
            model = ConvNextForImageClassification(config)
        else:
            raise ValueError(f"Unsupported custom model_type: {script_args.model_type}")
        return model

    if backend == "hf_auto":
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            script_args.model,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        return apply_lora_if_requested(model, script_args)

    if backend in {"hf_backbone", "dinov3_feature"}:
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(script_args.model)
        hidden_size = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "hidden_sizes", [768])[-1]))
        model = AutoBackboneForImageClassification(backbone, hidden_size, num_labels)
        return apply_lora_if_requested(model, script_args)

    if backend == "timm":
        import timm

        model_name = str(script_args.model)
        if model_name.startswith("timm/"):
            model_name = model_name.split("/", 1)[1]
        model = timm.create_model(
            model_name,
            pretrained=bool(getattr(script_args, "timm_pretrained", True)),
            num_classes=num_labels,
        )
        wrapped = TimmForImageClassification(model, num_labels)
        return apply_lora_if_requested(wrapped, script_args)

    raise ValueError(f"Unsupported model_backend: {backend}")
