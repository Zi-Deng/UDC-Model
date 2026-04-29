# Hugging Face DINOv3 Access Memory

Updated: 2026-04-28

- Stop 1 DINOv3 failed because the official Meta DINOv3 repositories are gated and the machine was not authenticated.
- The user should accept access/license conditions on:
  - `facebook/dinov3-vits16-pretrain-lvd1689m`
  - `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- A read token is sufficient for downloading gated models.
- Authenticate with `micromamba run -n ml hf auth login` and verify with `micromamba run -n ml hf auth whoami`.
- `whoami` proves login only. A `403 Forbidden` DINOv3 download error means the user/token is authenticated but not authorized for the gated repository yet; visit the DINOv3 model page as the same HF user and request/accept access, or adjust fine-grained token repository permissions.
- Do not store or commit tokens in repo files.
- Intended 5090-friendly DINOv3 sizes:
  - DINOv3 ViT: `facebook/dinov3-vits16-pretrain-lvd1689m`, ViT-S/16, 21.6M params.
  - DINOv3 ConvNeXt: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`, ConvNeXt Tiny, 27.8M params on the HF card, approximately 29M in the DINOv3 technical summary.
- Interim non-gated timm fallbacks:
  - `timm/vit_small_patch16_dinov3.lvd1689m`
  - `timm/convnext_tiny.dinov3_lvd1689m`
  - Label these as timm DINOv3 results; rerun/append official Meta DINOv3 after gated access approval.
- timm DINOv3 LoRA presets are part of the research path:
  - `timm_dinov3_vit_lora`: target `qkv`, save `model.head`.
  - `timm_dinov3_convnext_lora`: target `mlp.fc1,mlp.fc2`, save `model.head`; this is ConvNeXt MLP-LoRA/PEFT, not attention LoRA.
- Current Transformers DINOv3 ViT projection modules are `q_proj` and `v_proj`; NICME configs and runner should use `peft_target_modules="q_proj,v_proj"` rather than `query,value`.
