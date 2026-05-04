# MC-FB1 Official Facebook DINOv3 LoRA Smoke Summary

- Scope: EyePACS DR + PMI Pills, balanced splits, official Facebook DINOv3 ViT/ConvNeXt LoRA, CE + NICME hybrid.
- Ledger: `results/multiclass_mc_fb1_official_smoke_20260501/mc1/run_ledger.csv`
- Completed rows: `8/8`
- Failed rows: `0`
- Storage policy: `/mnt/storage` Hugging Face/Torch/temp caches; storage preflight passed.
- GPU health after run: no D-state user processes, no runner taint reasons, CUDA canary passed.
- Operational note: first invocation stopped before row 6 due a transient CUDA-canary `torch` import failure; immediate canary rerun passed, the canary was hardened with retry-once for non-timeout failures, and the campaign resumed successfully without reboot.

| # | Dataset | Model | Method | Status | Seconds | Accuracy | Balanced Acc. | Expected Cost | Result Dir |
|---:|---|---|---|---|---:|---:|---:|---:|---|
| 1 | eyepacs_dr | facebook_dinov3_vit_lora | ce | completed | 56.982 | 0.1437 | 0.1495 | 7.6437 | `results/facebook_dinov3_vit_lora_test/mc1_eyepacs_dr_balanced_fb_d3vit_lora_ce_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_879827c3_04-30_23-30` |
| 2 | eyepacs_dr | facebook_dinov3_vit_lora | nicme_hybrid | completed | 56.879 | 0.1437 | 0.1476 | 7.6937 | `results/facebook_dinov3_vit_lora_test/mc1_eyepacs_dr_balanced_fb_d3vit_lora_nicme_hybrid_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_7e711e4d_04-30_23-31` |
| 3 | eyepacs_dr | facebook_dinov3_convnext_lora | ce | completed | 56.678 | 0.1812 | 0.1906 | 3.1063 | `results/facebook_dinov3_convnext_lora_test/mc1_eyepacs_dr_balanced_fb_d3cnx_lora_ce_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_537cb0f5_04-30_23-32` |
| 4 | eyepacs_dr | facebook_dinov3_convnext_lora | nicme_hybrid | completed | 56.626 | 0.1875 | 0.1915 | 3.0812 | `results/facebook_dinov3_convnext_lora_test/mc1_eyepacs_dr_balanced_fb_d3cnx_lora_nicme_hybrid_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_4c7b29ea_04-30_23-33` |
| 5 | pmi_pills | facebook_dinov3_vit_lora | ce | completed | 19.024 | 0.0125 | 0.0500 | 0.9875 | `results/facebook_dinov3_vit_lora_test/mc1_pmi_pills_balanced_fb_d3vit_lora_ce_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_7b54e0f4_04-30_23-33` |
| 6 | pmi_pills | facebook_dinov3_vit_lora | nicme_hybrid | completed | 17.620 | 0.0125 | 0.0500 | 0.9875 | `results/facebook_dinov3_vit_lora_test/mc1_pmi_pills_balanced_fb_d3vit_lora_nicme_hybrid_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_e07be8c6_04-30_23-35` |
| 7 | pmi_pills | facebook_dinov3_convnext_lora | ce | completed | 18.776 | 0.0500 | 0.0444 | 1.3438 | `results/facebook_dinov3_convnext_lora_test/mc1_pmi_pills_balanced_fb_d3cnx_lora_ce_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_e6433bf8_04-30_23-35` |
| 8 | pmi_pills | facebook_dinov3_convnext_lora | nicme_hybrid | completed | 18.818 | 0.0500 | 0.0444 | 1.2312 | `results/facebook_dinov3_convnext_lora_test/mc1_pmi_pills_balanced_fb_d3cnx_lora_nicme_hybrid_s42_multiclass_mc_fb1_official_smoke_20260501_mc1_a1_e5fabb61_04-30_23-36` |
