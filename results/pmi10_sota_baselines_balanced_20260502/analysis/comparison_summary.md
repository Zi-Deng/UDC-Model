# PMI-10 Balanced SOTA Baseline Comparison

Generated: 2026-05-02T12:26:43

| Rank | Method | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |
|---:|---|---|---:|---:|---:|---:|
| 1 | cost_sensitive_regularized_ce | argmax | 0.9375 | 0.005000 | 0.9781 | 1 |
| 2 | ce | argmax | 0.9375 | 0.005938 | 0.9688 | 1 |
| 3 | class_balanced_focal | argmax | 0.9375 | 0.006563 | 0.9625 | 1 |
| 4 | menon_logit_adjusted | argmax | 0.9375 | 0.008125 | 0.9750 | 2 |
| 5 | ce_cost_min_inference | cost_min | 0.9375 | 0.008438 | 0.9437 | 1 |
| 6 | ce_anchor | argmax | 0.9375 | 0.008750 | 0.9688 | 2 |
| 7 | cost_weighted_ce | argmax | 0.9375 | 0.010312 | 0.9469 | 2 |
| 8 | csada | argmax | 0.9375 | 0.014688 | 0.8812 | 1 |
| 9 | nicme_v3_hybrid | argmax | 0.9062 | 0.007812 | 0.9781 | 2 |
| 10 | ce | argmax | 0.9062 | 0.009375 | 0.9625 | 2 |
| 11 | balanced_softmax | argmax | 0.8750 | 0.007812 | 0.9781 | 2 |
| 12 | nicme_hybrid | argmax | 0.8750 | 0.013750 | 0.9750 | 4 |
| 13 | ldam_drw | argmax | 0.7812 | 0.011875 | 0.9531 | 3 |
| 14 | sosr_cnn | sosr_argmin | 0.0000 | 0.105625 | 0.0000 | 2 |
