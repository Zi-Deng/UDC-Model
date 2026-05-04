# PMI-10 SOTA Pretty Hyperparameters

Generated: 2026-05-02T19:50:40

- Result root: `/mnt/storage/github/NICME/results/pmi10_sota_pretty_balanced_lr1e4_20260503`
- Comparison profile: `pretty_balanced`
- Learning-rate profile: `lr1e4`
- Chain learning rate: `0.0001`
- Shared model: `convnext_base.fb_in22k_ft_in1k`
- Shared split: balanced PMI-10 no-calibration

| Method | LR | Parent LR | Notes |
|---|---:|---:|---|
| ap_csada | `1e-05` | `0.0001` | CE-anchor adaptation; native adaptation LR `1e-5` |
| ce_anchor_pretty | `0.0001` | `0.0001` |  |
| nicme_v2_hybrid_pretty | `0.0001` | `0.0001` | `alpha=0.4`, `cs_lambda=0.25`, `cs_warmup_epochs=2` |
| cost_weighted_ce | `0.0001` | `0.0001` |  |
| class_balanced_focal | `0.0001` | `0.0001` |  |
| csada | `1e-05` | `0.0001` | CE-anchor adaptation; native adaptation LR `1e-5` |
| ce_cost_min_inference_pretty | `0.0001` | `0.0001` | Inference-only cost-min report from CE anchor |
| menon_logit_adjusted | `0.0001` | `0.0001` |  |
| nicme_v3_hybrid_pretty | `0.0001` | `0.0001` | `alpha=0.4`, `cs_lambda=0.25`, `cs_warmup_epochs=2` |
| balanced_softmax | `0.0001` | `0.0001` |  |
| ldam_drw | `0.0001` | `0.0001` |  |
| sosr_cnn | `1e-05` | `0.0001` | CE-anchor adaptation; native adaptation LR `1e-5` |
