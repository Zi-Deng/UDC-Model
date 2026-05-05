# PMI-10 Balanced SOTA Baseline Comparison

Generated: 2026-05-02T22:47:40

Post-HPO note, 2026-05-03: this LR 1e-5 baseline summary was generated before the completed NICME v3 alpha/lambda HPO at LR 5e-5. For current post-HPO results, use `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`.

| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | ap_csada | lr1e5 | 1e-05 | argmax | 0.9062 | 0.015000 | 0.9563 | 4 |
| 2 | nicme_v3_hybrid_pretty | lr1e5 | 1e-05 | argmax | 0.8750 | 0.013750 | 0.9688 | 4 |
| 3 | cost_weighted_ce | lr1e5 | 1e-05 | argmax | 0.8750 | 0.014375 | 0.9406 | 3 |
| 4 | ldam_drw | lr1e5 | 1e-05 | argmax | 0.8750 | 0.023125 | 0.9469 | 7 |
| 5 | csada | lr1e5 | 1e-05 | argmax | 0.8750 | 0.037187 | 0.7063 | 3 |
| 6 | ce_anchor_pretty | lr1e5 | 1e-05 | argmax | 0.8438 | 0.016875 | 0.9656 | 5 |
| 7 | menon_logit_adjusted | lr1e5 | 1e-05 | argmax | 0.8438 | 0.016875 | 0.9656 | 5 |
| 8 | balanced_softmax | lr1e5 | 1e-05 | argmax | 0.8438 | 0.016875 | 0.9656 | 5 |
| 9 | class_balanced_focal | lr1e5 | 1e-05 | argmax | 0.8125 | 0.020938 | 0.9250 | 5 |
| 10 | nicme_v2_hybrid_pretty | lr1e5 | 1e-05 | argmax | 0.7812 | 0.024375 | 0.9062 | 6 |
| 11 | ce_cost_min_inference_pretty | lr1e5 | 1e-05 | cost_min | 0.3438 | 0.011875 | 0.8812 | 0 |
| 12 | sosr_cnn | lr1e5 | 1e-05 | sosr_argmin | 0.2188 | 0.026250 | 0.7656 | 1 |
