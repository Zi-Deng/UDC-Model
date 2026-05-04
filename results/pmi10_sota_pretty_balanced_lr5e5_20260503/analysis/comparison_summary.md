# PMI-10 Balanced SOTA Baseline Comparison

Generated: 2026-05-02T19:08:33

Post-HPO note, 2026-05-03: this LR 5e-5 baseline summary was generated before the completed NICME v3 alpha/lambda HPO. For current post-HPO results, use `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`. The completed HPO's best observed test row is run 95 (`alpha=0.5`, `cs_lambda=0.07`) with target-min recall `0.96875`, normalized ATC `0.0046875`, balanced accuracy `0.98125`, and `1` critical-pair error.

| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---|---:|---:|---:|---:|
| 1 | csada | lr5e5 | 1e-05 | argmax | 0.9688 | 0.015625 | 0.8719 | 1 |
| 2 | ce_cost_min_inference_pretty | lr5e5 | 5e-05 | cost_min | 0.9062 | 0.006563 | 0.9625 | 1 |
| 3 | nicme_v3_hybrid_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 4 | balanced_softmax | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008125 | 0.9750 | 2 |
| 5 | ce_anchor_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.008438 | 0.9719 | 2 |
| 6 | nicme_v2_hybrid_pretty | lr5e5 | 5e-05 | argmax | 0.9062 | 0.010312 | 0.9750 | 3 |
| 7 | menon_logit_adjusted | lr5e5 | 5e-05 | argmax | 0.9062 | 0.011250 | 0.9719 | 3 |
| 8 | ldam_drw | lr5e5 | 5e-05 | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 9 | ap_csada | lr5e5 | 1e-05 | argmax | 0.9062 | 0.012188 | 0.9719 | 4 |
| 10 | cost_weighted_ce | lr5e5 | 5e-05 | argmax | 0.8750 | 0.040000 | 0.7125 | 4 |
| 11 | class_balanced_focal | lr5e5 | 5e-05 | argmax | 0.8438 | 0.011562 | 0.9688 | 3 |
| 12 | sosr_cnn | lr5e5 | 1e-05 | sosr_argmin | 0.0000 | 0.139375 | 0.0000 | 14 |
