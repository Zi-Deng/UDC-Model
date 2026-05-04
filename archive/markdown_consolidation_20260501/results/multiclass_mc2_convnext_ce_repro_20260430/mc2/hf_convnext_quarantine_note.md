# HF ConvNeXt MC2 Quarantine Note

The initial MC2 campaign included the Hugging Face `facebook/convnext-tiny-224`
baseline. The first full EyePACS balanced CE run failed in PyTorch autograd with:

```text
RuntimeError: isDifferentiableType(grad.scalar_type()) INTERNAL ASSERT FAILED
```

After patching the training loss boundary to use contiguous float logits, a
focused MC2-equivalent CE repro still froze after the first training log while
holding GPU memory. Because MC1 already passed the scientific smoke matrix and
MC3 was scoped to the stable timm DINOv3 LoRA backbones, the paper-grade MC2
v2 campaign excludes this unstable HF ConvNeXt path and runs:

- `timm_dinov3_vit_lora`
- `timm_dinov3_convnext_lora`

The excluded run family should be treated as infrastructure/model-backend
instability, not as a scientific method failure.
