# Hugging Face DINOv3 Access And Model-Size Notes

Updated: 2026-04-28

## Why Stop 1 DINOv3 Failed

The official Meta DINOv3 Hugging Face repositories are gated. The model page
requires the user to log in and agree to share contact information before model
files can be downloaded. The failed Stop 1 rows reached Hugging Face model
metadata, but could not download gated files without an authenticated account
that has accepted the DINOv3 access conditions.

## Authentication Procedure

1. Open these model pages in a browser while logged into Hugging Face and accept
   the access/license conditions:
   - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
   - https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
2. Create a read token at https://huggingface.co/settings/tokens.
3. Authenticate the local `ml` environment:

```bash
micromamba run -n ml hf auth login
```

Paste the token when prompted. Do not commit the token, put it in a JSON config,
or paste it into a shared log.

Non-interactive alternative for a private shell:

```bash
export HF_TOKEN='hf_...'
micromamba run -n ml hf auth login --token "$HF_TOKEN"
unset HF_TOKEN
```

Check login status:

```bash
micromamba run -n ml hf auth whoami
```

`whoami` only proves that the token is valid. For gated DINOv3 access, the
account/token must also be authorized for the specific model repositories. If
the access check returns `403 Forbidden` with "not in the authorized list", log
into the Hugging Face website as the same user, visit the model pages, request
or accept access, and make sure a fine-grained token includes read permission
for those repositories.

Check DINOv3 file access without starting training:

```bash
micromamba run -n ml python - <<'PY'
from transformers import AutoConfig, AutoImageProcessor

for model_id in [
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
]:
    cfg = AutoConfig.from_pretrained(model_id)
    _ = AutoImageProcessor.from_pretrained(model_id)
    print(model_id, cfg.model_type)
PY
```

## Current NICME DINOv3 Size Choices

These are the intended 5090-friendly DINOv3 choices:

| NICME family | Hugging Face model id | Intended size |
| --- | --- | --- |
| `dinov3_vit` | `facebook/dinov3-vits16-pretrain-lvd1689m` | ViT-S/16, 21.6M parameters |
| `dinov3_convnext` | `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` | ConvNeXt Tiny, 27.8M parameters on the HF card; about 29M in the DINOv3 technical summary |
| `timm_dinov3_vit` | `timm/vit_small_patch16_dinov3.lvd1689m` | Non-gated timm ViT-S/16 fallback, 21.6M parameters |
| `timm_dinov3_convnext` | `timm/convnext_tiny.dinov3_lvd1689m` | Non-gated timm ConvNeXt Tiny fallback, 27.8M parameters |
| `timm_dinov3_vit_lora` | `timm/vit_small_patch16_dinov3.lvd1689m` | LoRA variant of the non-gated timm ViT-S/16 fallback |
| `timm_dinov3_convnext_lora` | `timm/convnext_tiny.dinov3_lvd1689m` | LoRA variant of the non-gated timm ConvNeXt Tiny fallback |

The repo intentionally uses the smallest official DINOv3 ViT and ConvNeXt
families for Stop 1/2 feasibility on a single RTX 5090. Larger choices exist
but should be reserved for later ablation only after runtime is measured.

While official Meta DINOv3 access is pending, interim Stop 1/2 runs may use the
non-gated timm fallbacks. These should be labeled as timm DINOv3 results, not
as official `facebook/dinov3-*` results. Official Meta checkpoints should still
be rerun and appended after gated access approval.

## LoRA Target Modules

Current Transformers DINOv3 ViT modules expose attention projections as
`q_proj` and `v_proj`. NICME therefore uses:

```json
"peft_target_modules": "q_proj,v_proj"
```

This differs from older generic ViT PEFT examples that use `query,value`.

For timm DINOv3 models, the module names differ:

- timm DINOv3 ViT uses fused attention projection modules named `qkv`, so
  `timm_dinov3_vit_lora` targets `qkv`.
- timm DINOv3 ConvNeXt is not an attention model, so `timm_dinov3_convnext_lora`
  targets the block MLP linear layers `mlp.fc1,mlp.fc2`; depthwise convolution
  layers remain frozen. Report this as ConvNeXt MLP-LoRA/PEFT rather than
  attention LoRA.
- Both timm LoRA presets save the classifier head through `model.head`.
