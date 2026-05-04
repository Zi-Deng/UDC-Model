# MC1 Hardened Smoke Summary

- Dataset/variant: `eyepacs_dr` / `balanced`
- Scientific configs required: 6
- Completed configs: 6 / 6
- Retry policy active: retry once for zero-log timeouts and segfault exits; fresh output namespace per attempt.
- Optional confusion-matrix visualizations skipped for MC1 to avoid native Matplotlib smoke-test flakiness; JSON/TXT metrics still exported.

| Model | Method | Status | Attempts | Target recall | Norm ATC | Balanced acc | QWK | Metric file |
|---|---|---:|---:|---:|---:|---:|---:|---|
| convnext | ce | completed | 1 | 0.2667 | 0.2801 | 0.2538 | 0.1346 | `results/convnext_test/mc1_eyepacs_dr_balanced_cnx_ce_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_66fff911_04-30_12-59/metrics_20260430_125932_ce.json` |
| convnext | nicme_hybrid | completed | 1 | 0.6000 | 0.1867 | 0.3785 | 0.3267 | `results/convnext_test/mc1_eyepacs_dr_balanced_cnx_nicme_hybrid_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_0086a9ba_04-30_13-00/metrics_20260430_130037_nicme_hybrid.json` |
| timm_dinov3_vit_lora | ce | completed | 1 | 0.0000 | 0.3527 | 0.2000 | 0.0000 | `results/timm_dinov3_vit_lora_test/mc1_eyepacs_dr_balanced_d3vit_lora_ce_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_1415240e_04-30_13-01/metrics_20260430_130140_ce.json` |
| timm_dinov3_vit_lora | nicme_hybrid | completed | 1 | 0.0000 | 0.2355 | 0.2301 | 0.2280 | `results/timm_dinov3_vit_lora_test/mc1_eyepacs_dr_balanced_d3vit_lora_nicme_hybrid_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_98d98a9d_04-30_13-02/metrics_20260430_130244_nicme_hybrid.json` |
| timm_dinov3_convnext_lora | ce | completed | 1 | 0.6667 | 0.3758 | 0.1668 | -0.0276 | `results/timm_dinov3_convnext_lora_test/mc1_eyepacs_dr_balanced_d3cnx_lora_ce_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_47bb365f_04-30_13-03/metrics_20260430_130352_ce.json` |
| timm_dinov3_convnext_lora | nicme_hybrid | completed | 1 | 0.6333 | 0.3676 | 0.1601 | -0.0225 | `results/timm_dinov3_convnext_lora_test/mc1_eyepacs_dr_balanced_d3cnx_lora_nicme_hybrid_s42_multiclass_mc1_hardened_v2_20260430_mc1_a1_ec5dfc5c_04-30_13-04/metrics_20260430_130459_nicme_hybrid.json` |
