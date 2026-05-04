# PMI-10 Balanced SOTA Pretty Dual-LR Comparison

Generated: 2026-05-02T19:50:40

Post-HPO note, 2026-05-03: this dual-LR baseline summary was generated before the completed NICME v3 alpha/lambda HPO at LR 5e-5. For current post-HPO results, use `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`.

- `lr5e5`: conservative clean fine-tuning LR, `5e-5`.
- `lr1e4`: aggressive LR sensitivity chain, `1e-4`.

## Overall Ranking

| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | csada | lr5e5 | 1e-05 | argmax | 0.9688 | 0.015625 | 0.8719 | 1 |
| 2 | ap_csada | lr1e4 | 1e-05 | argmax | 0.9375 | 0.005000 | 0.9781 | 1 |
| 3 | ce_anchor_pretty | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005312 | 0.9750 | 1 |
| 4 | nicme_v2_hybrid_pretty | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 5 | cost_weighted_ce | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 6 | class_balanced_focal | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005938 | 0.9688 | 1 |
| 7 | csada | lr1e4 | 1e-05 | argmax | 0.9375 | 0.007187 | 0.9500 | 1 |
| 8 | ce_cost_min_inference_pretty | lr1e4 | 0.0001 | cost_min | 0.9062 | 0.002500 | 0.9750 | 0 |
| 9 | ce_cost_min_inference_pretty | lr5e5 | 5e-05 | cost_min | 0.9062 | 0.006563 | 0.9625 | 1 |
| 10 | nicme_v3_hybrid_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 11 | balanced_softmax | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 12 | ce_anchor_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008438 | 0.9719 | 2 |
| 13 | nicme_v2_hybrid_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.010312 | 0.9750 | 3 |
| 14 | menon_logit_adjusted | lr5e5 | 5e-05 | argmax | 0.9062 | 0.011250 | 0.9719 | 3 |
| 15 | ldam_drw | lr5e5 | 5e-05 | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 16 | ap_csada | lr5e5 | 1e-05 | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 17 | menon_logit_adjusted | lr1e4 | 0.0001 | argmax | 0.9062 | 0.015000 | 0.9062 | 2 |
| 18 | nicme_v3_hybrid_pretty | lr1e4 | 0.0001 | argmax | 0.9062 | 0.018438 | 0.9219 | 4 |
| 19 | balanced_softmax | lr1e4 | 0.0001 | argmax | 0.8750 | 0.008125 | 0.9750 | 2 |
| 20 | ldam_drw | lr1e4 | 0.0001 | argmax | 0.8750 | 0.015625 | 0.9563 | 4 |
| 21 | cost_weighted_ce | lr5e5 | 5e-05 | argmax | 0.8750 | 0.040000 | 0.7125 | 4 |
| 22 | class_balanced_focal | lr5e5 | 5e-05 | argmax | 0.8438 | 0.011562 | 0.9688 | 3 |
| 23 | sosr_cnn | lr1e4 | 1e-05 | sosr_argmin | 0.0000 | 0.119687 | 0.0000 | 7 |
| 24 | sosr_cnn | lr5e5 | 1e-05 | sosr_argmin | 0.0000 | 0.139375 | 0.0000 | 14 |

## LR 5e-5

| Rank | Method | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---:|---:|---:|
| 1 | csada | argmax | 0.9688 | 0.015625 | 0.8719 | 1 |
| 2 | ce_cost_min_inference_pretty | cost_min | 0.9062 | 0.006563 | 0.9625 | 1 |
| 3 | nicme_v3_hybrid_pretty | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 4 | balanced_softmax | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 5 | ce_anchor_pretty | argmax | 0.9062 | 0.008438 | 0.9719 | 2 |
| 6 | nicme_v2_hybrid_pretty | argmax | 0.9062 | 0.010312 | 0.9750 | 3 |
| 7 | menon_logit_adjusted | argmax | 0.9062 | 0.011250 | 0.9719 | 3 |
| 8 | ldam_drw | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 9 | ap_csada | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 10 | cost_weighted_ce | argmax | 0.8750 | 0.040000 | 0.7125 | 4 |
| 11 | class_balanced_focal | argmax | 0.8438 | 0.011562 | 0.9688 | 3 |
| 12 | sosr_cnn | sosr_argmin | 0.0000 | 0.139375 | 0.0000 | 14 |

## LR 1e-4 (aggressive LR)

| Rank | Method | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---:|---:|---:|
| 1 | ap_csada | argmax | 0.9375 | 0.005000 | 0.9781 | 1 |
| 2 | ce_anchor_pretty | argmax | 0.9375 | 0.005312 | 0.9750 | 1 |
| 3 | nicme_v2_hybrid_pretty | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 4 | cost_weighted_ce | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 5 | class_balanced_focal | argmax | 0.9375 | 0.005938 | 0.9688 | 1 |
| 6 | csada | argmax | 0.9375 | 0.007187 | 0.9500 | 1 |
| 7 | ce_cost_min_inference_pretty | cost_min | 0.9062 | 0.002500 | 0.9750 | 0 |
| 8 | menon_logit_adjusted | argmax | 0.9062 | 0.015000 | 0.9062 | 2 |
| 9 | nicme_v3_hybrid_pretty | argmax | 0.9062 | 0.018438 | 0.9219 | 4 |
| 10 | balanced_softmax | argmax | 0.8750 | 0.008125 | 0.9750 | 2 |
| 11 | ldam_drw | argmax | 0.8750 | 0.015625 | 0.9563 | 4 |
| 12 | sosr_cnn | sosr_argmin | 0.0000 | 0.119687 | 0.0000 | 7 |
