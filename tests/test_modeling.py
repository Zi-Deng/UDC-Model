from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
timm = pytest.importorskip("timm")
transformers = pytest.importorskip("transformers")

from nicme.modeling import (  # noqa: E402
    AutoBackboneForImageClassification,
    TimmForImageClassification,
    apply_lora_if_requested,
    build_model,
)
from scripts.run_binary_experiments import MODEL_PRESETS  # noqa: E402
from utils.image_processor import CustomImageProcessor  # noqa: E402
from utils.utils import validate_training_config  # noqa: E402


def test_dinov3_vit_lora_targets_current_transformers_qv_projection_names():
    config = transformers.DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
    )
    backbone = transformers.DINOv3ViTModel(config)
    model = AutoBackboneForImageClassification(backbone, hidden_size=32, num_labels=2)
    args = SimpleNamespace(
        peft_enabled=True,
        peft_r=2,
        peft_alpha=4,
        peft_dropout=0.0,
        peft_target_modules="q_proj,v_proj",
        peft_modules_to_save="classifier",
    )

    lora_model = apply_lora_if_requested(model, args)
    trainable = [name for name, param in lora_model.named_parameters() if param.requires_grad]

    assert any("q_proj" in name and "lora" in name for name in trainable)
    assert any("v_proj" in name and "lora" in name for name in trainable)
    assert any("classifier" in name for name in trainable)


def test_timm_wrapper_returns_trainer_compatible_logits():
    base = timm.create_model("convnext_tiny.dinov3_lvd1689m", pretrained=False, num_classes=2)
    model = TimmForImageClassification(base, num_labels=2)

    outputs = model(pixel_values=torch.randn(2, 3, 224, 224), labels=torch.tensor([0, 1]))

    assert outputs.logits.shape == (2, 2)
    assert outputs.loss is not None


def test_build_model_supports_timm_backend_without_downloading_weights():
    args = SimpleNamespace(
        model="timm/convnext_tiny.dinov3_lvd1689m",
        model_backend="timm",
        model_type="timm_dinov3_convnext",
        num_labels=2,
        timm_pretrained=False,
        peft_enabled=False,
    )

    model = build_model(args, class_names=["benign", "malignant"])
    outputs = model(pixel_values=torch.randn(1, 3, 224, 224))

    assert outputs.logits.shape == (1, 2)


def test_training_config_validation_accepts_timm_backend():
    validate_training_config(
        {
            "model_backend": "timm",
            "model_type": "timm_dinov3_vit",
            "num_labels": 2,
            "cost_matrix": [[0.0, 10.0], [1.0, 0.0]],
        }
    )


def test_custom_image_processor_uses_timm_dinov3_transforms():
    vit_processor = CustomImageProcessor.from_pretrained("timm/vit_small_patch16_dinov3.lvd1689m")
    convnext_processor = CustomImageProcessor.from_pretrained("timm/convnext_tiny.dinov3_lvd1689m")

    assert vit_processor.size == {"height": 256, "width": 256}
    assert vit_processor.image_mean == [0.485, 0.456, 0.406]
    assert vit_processor.crop_pct == 1.0
    assert convnext_processor.size == {"height": 224, "width": 224}
    assert convnext_processor.crop_pct == 1.0


def test_timm_dinov3_lora_presets_attach_expected_adapters():
    cases = {
        "timm_dinov3_vit_lora": "qkv",
        "timm_dinov3_convnext_lora": "mlp.fc",
    }
    for preset_name, expected_fragment in cases.items():
        cfg = dict(MODEL_PRESETS[preset_name])
        cfg["num_labels"] = 2
        cfg["timm_pretrained"] = False
        args = SimpleNamespace(**cfg)

        model = build_model(args, class_names=["a", "b"])
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]

        assert any("lora_" in name and expected_fragment in name for name in trainable)
        assert any("head" in name for name in trainable)
