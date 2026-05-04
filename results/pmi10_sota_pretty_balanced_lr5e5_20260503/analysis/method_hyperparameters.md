# PMI-10 SOTA Pretty Hyperparameters

Generated: 2026-05-02T19:08:33

- Result root: `/mnt/storage/github/NICME/results/pmi10_sota_pretty_balanced_lr5e5_20260503`
- Comparison profile: `pretty_balanced`
- Learning-rate profile: `lr5e5`
- Chain learning rate: `5e-05`
- Shared model: `convnext_base.fb_in22k_ft_in1k`
- Shared split: balanced PMI-10 no-calibration

| Method | LR | Parent LR | Notes |
|---|---:|---:|---|
| csada | `1e-05` | `5e-05` | CE-anchor adaptation; native adaptation LR `1e-5` |
| ce_cost_min_inference_pretty | `5e-05` | `5e-05` | Inference-only cost-min report from CE anchor |
| nicme_v3_hybrid_pretty | `5e-05` | `5e-05` | `alpha=0.4`, `cs_lambda=0.25`, `cs_warmup_epochs=2` |
| balanced_softmax | `5e-05` | `5e-05` |  |
| ce_anchor_pretty | `5e-05` | `5e-05` |  |
| nicme_v2_hybrid_pretty | `5e-05` | `5e-05` | `alpha=0.4`, `cs_lambda=0.25`, `cs_warmup_epochs=2` |
| menon_logit_adjusted | `5e-05` | `5e-05` |  |
| ldam_drw | `5e-05` | `5e-05` |  |
| ap_csada | `1e-05` | `5e-05` | CE-anchor adaptation; native adaptation LR `1e-5` |
| cost_weighted_ce | `5e-05` | `5e-05` |  |
| class_balanced_focal | `5e-05` | `5e-05` |  |
| sosr_cnn | `1e-05` | `5e-05` | CE-anchor adaptation; native adaptation LR `1e-5` |
