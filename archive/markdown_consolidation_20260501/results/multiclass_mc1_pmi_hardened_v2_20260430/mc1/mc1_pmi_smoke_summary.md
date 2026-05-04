# PMI MC1 Hardened Smoke Summary

- Dataset/variant: `pmi_pills` / `balanced`
- Scientific configs required: 6
- Completed configs: 6 / 6
- Retry policy active: retry once for zero-log timeouts, early-startup timeouts, and segfault exits; fresh output namespace per attempt.
- Optional confusion-matrix visualizations skipped for MC1; JSON/TXT metrics still exported.

| Model | Method | Status | Attempts | Returncodes | Target recall | Norm ATC | Balanced acc | Metric file |
|---|---|---:|---:|---|---:|---:|---:|---|
| convnext | ce | completed | 1 | 0 | 0.0000 | 0.0919 | 0.1077 | `results/convnext_test/mc1_pmi_pills_balanced_cnx_ce_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a1_d802031f_04-30_15-57/metrics_20260430_155731_ce.json` |
| convnext | nicme_hybrid | completed | 1 | 0 | 0.0000 | 0.0919 | 0.1039 | `results/convnext_test/mc1_pmi_pills_balanced_cnx_nicme_hybrid_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a1_e4118b6c_04-30_15-57/metrics_20260430_155751_nicme_hybrid.json` |
| timm_dinov3_vit_lora | ce | completed | 2 | -11,0 | 0.0000 | 0.1113 | 0.1792 | `results/timm_dinov3_vit_lora_test/mc1_pmi_pills_balanced_d3vit_lora_ce_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a2_f3eccd62_04-30_15-58/metrics_20260430_155818_ce.json` |
| timm_dinov3_vit_lora | nicme_hybrid | completed | 1 | 0 | 0.0000 | 0.0762 | 0.2623 | `results/timm_dinov3_vit_lora_test/mc1_pmi_pills_balanced_d3vit_lora_nicme_hybrid_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a1_fed57e79_04-30_15-58/metrics_20260430_155839_nicme_hybrid.json` |
| timm_dinov3_convnext_lora | ce | completed | 1 | 0 | 0.0000 | 0.0975 | 0.0236 | `results/timm_dinov3_convnext_lora_test/mc1_pmi_pills_balanced_d3cnx_lora_ce_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a1_ccc8cf32_04-30_15-58/metrics_20260430_155900_ce.json` |
| timm_dinov3_convnext_lora | nicme_hybrid | completed | 2 | 124,0 | 0.0000 | 0.0963 | 0.0379 | `results/timm_dinov3_convnext_lora_test/mc1_pmi_pills_balanced_d3cnx_lora_nicme_hybrid_s42_multiclass_mc1_pmi_hardened_v2_20260430_mc1_a2_6705d4b3_04-30_16-04/metrics_20260430_160421_nicme_hybrid.json` |
