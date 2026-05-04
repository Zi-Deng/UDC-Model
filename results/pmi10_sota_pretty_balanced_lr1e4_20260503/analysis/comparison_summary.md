# PMI-10 Balanced SOTA Baseline Comparison

Generated: 2026-05-02T19:50:40

Post-HPO note, 2026-05-03: this LR 1e-4 baseline summary was generated before the completed NICME v3 alpha/lambda HPO at LR 5e-5. For current post-HPO results, use `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`.

| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | ap_csada | lr1e4 | 1e-05 | argmax | 0.9375 | 0.005000 | 0.9781 | 1 |
| 2 | ce_anchor_pretty | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005312 | 0.9750 | 1 |
| 3 | nicme_v2_hybrid_pretty | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 4 | cost_weighted_ce | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005625 | 0.9719 | 1 |
| 5 | class_balanced_focal | lr1e4 | 0.0001 | argmax | 0.9375 | 0.005938 | 0.9688 | 1 |
| 6 | csada | lr1e4 | 1e-05 | argmax | 0.9375 | 0.007187 | 0.9500 | 1 |
| 7 | ce_cost_min_inference_pretty | lr1e4 | 0.0001 | cost_min | 0.9062 | 0.002500 | 0.9750 | 0 |
| 8 | menon_logit_adjusted | lr1e4 | 0.0001 | argmax | 0.9062 | 0.015000 | 0.9062 | 2 |
| 9 | nicme_v3_hybrid_pretty | lr1e4 | 0.0001 | argmax | 0.9062 | 0.018438 | 0.9219 | 4 |
| 10 | balanced_softmax | lr1e4 | 0.0001 | argmax | 0.8750 | 0.008125 | 0.9750 | 2 |
| 11 | ldam_drw | lr1e4 | 0.0001 | argmax | 0.8750 | 0.015625 | 0.9563 | 4 |
| 12 | sosr_cnn | lr1e4 | 1e-05 | sosr_argmin | 0.0000 | 0.119687 | 0.0000 | 7 |
