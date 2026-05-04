# MC2 Selection

- Recall tie window: 0.0050

## eyepacs_dr

- Completed candidates: 10
- Failed candidates: 0

| Model | Method | Target Recall | Normalized ATC | Balanced Accuracy | Metrics |
|---|---|---:|---:|---:|---|
| facebook_dinov3_convnext_lora | ce | 0.7432 | 0.1199 | 0.4243 | `results/facebook_dinov3_convnext_lora_test/mc2_eyepacs_dr_balanced_fb_d3cnx_lora_ce_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_cd8fb9f4_05-01_00-46/metrics_20260501_004727_ce.json` |
| facebook_dinov3_convnext_lora | cs_regularized_ce | 0.7568 | 0.1174 | 0.4432 | `results/facebook_dinov3_convnext_lora_test/mc2_eyepacs_dr_balanced_fb_d3cnx_lora_cs_ce_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_5b3178fa_05-01_01-08/metrics_20260501_010848_cs_regularized_ce.json` |
| facebook_dinov3_convnext_lora | nicme_hybrid | 0.6892 | 0.1225 | 0.4108 | `results/facebook_dinov3_convnext_lora_test/mc2_eyepacs_dr_balanced_fb_d3cnx_lora_nicme_hybrid_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_3483e35e_05-01_01-32/metrics_20260501_013321_nicme_hybrid.json` |

## pmi_pills

- Completed candidates: 10
- Failed candidates: 0

| Model | Method | Target Recall | Normalized ATC | Balanced Accuracy | Metrics |
|---|---|---:|---:|---:|---|
| facebook_dinov3_convnext_lora | ce | 0.4062 | 0.0505 | 0.7078 | `results/facebook_dinov3_convnext_lora_test/mc2_pmi_pills_balanced_fb_d3cnx_lora_ce_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_9753b4e1_05-01_01-42/metrics_20260501_014301_ce.json` |
| facebook_dinov3_convnext_lora | cs_regularized_ce | 0.4062 | 0.0503 | 0.7094 | `results/facebook_dinov3_convnext_lora_test/mc2_pmi_pills_balanced_fb_d3cnx_lora_cs_ce_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_09d239dd_05-01_01-51/metrics_20260501_015212_cs_regularized_ce.json` |
| facebook_dinov3_convnext_lora | nicme_logit_adjustment | 0.3750 | 0.0348 | 0.7562 | `results/facebook_dinov3_convnext_lora_test/mc2_pmi_pills_balanced_fb_d3cnx_lora_nicme_logit_s42_multiclass_mc2_official_dinov3_lora_balanced_20260501_mc2_a1_95c4908b_05-01_01-54/metrics_20260501_015442_nicme_logit_adjustment.json` |

