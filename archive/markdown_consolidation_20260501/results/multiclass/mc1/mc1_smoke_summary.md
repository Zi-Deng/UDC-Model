# MC1 Multiclass Smoke Summary

- Dataset: `eyepacs_dr`
- Variant: `balanced`
- Runs planned: 9
- Runs completed: 7
- Runs failed: 2
- Ledger: `results/multiclass/mc1/run_ledger.csv`
- Note: metrics below are smoke-test sanity metrics on capped samples, not paper-grade estimates.

| Run | Status | Accuracy | Balanced Accuracy | DR4 Recall | Normalized ATC |
|---|---:|---:|---:|---:|---:|
| `convnext + ce` | completed | 0.3750 | 0.3785 | 0.6000 | 0.1867 |
| `convnext + ce_calibrated_cost_min` | completed | 0.2625 | 0.2538 | 0.2667 | 0.2801 |
| `convnext + nicme_hybrid` | completed | 0.3750 | 0.3785 | 0.6000 | 0.1867 |
| `timm_dinov3_vit_lora + ce` | completed | 0.2375 | 0.2301 | 0.0000 | 0.2355 |
| `timm_dinov3_vit_lora + ce_calibrated_cost_min` | completed | 0.2188 | 0.2000 | 0.0000 | 0.3527 |
| `timm_dinov3_vit_lora + nicme_hybrid` | completed | 0.2375 | 0.2301 | 0.0000 | 0.2355 |
| `timm_dinov3_convnext_lora + ce` | failed |  |  |  |  |
| `timm_dinov3_convnext_lora + ce_calibrated_cost_min` | failed |  |  |  |  |
| `timm_dinov3_convnext_lora + nicme_hybrid` | completed | 0.1500 | 0.1601 | 0.6333 | 0.3676 |

Failures:
- `timm_dinov3_convnext_lora + ce`: manually terminated after a 416 second smoke-run hang before completion.
- `timm_dinov3_convnext_lora + ce_calibrated_cost_min`: process exited with `-11` before writing stdout/stderr.

Current decision:
- Do not proceed to full MC2/MC3 paper-grade EyePACS runs until the unstable `timm_dinov3_convnext_lora` path is either fixed, excluded with justification, or replaced by the stable DINOv3-ViT LoRA path.
