# PMI-20 Paper Method Hyperparameters

Generated: 2026-05-04T18:01:44

- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared split: `data/prepared/pmi_pills/splits/balanced`.
- Shared training protocol: 32 epochs, patience 5, batch 16, gradient accumulation 4, cosine LR, weight decay 0.005, warmup ratio 0.10.
- cost-sensitive regularized CE and CSADA use native adaptation LR `1e-5` with parent LR `5e-5`.

| Method | Loss / Decision | LR | Parent LR | Alpha | Lambda | Notes |
|---|---|---:|---:|---:|---:|---|
| NICME (alpha=0.5, lambda=0.1) | `nicme_hybrid` / `argmax` | `5e-05` | `5e-05` | `0.5` | `0.1` | Main paper NICME row from six-candidate PMI-20 rerun |
| cost-sensitive regularized CE | `cost_sensitive_regularized_ce` / `argmax` | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation baseline |
| Menon logit adjustment | `menon_logit_adjusted` / `argmax` | `5e-05` | `5e-05` | `` | `` |  |
| CE | `cross_entropy` / `argmax` | `5e-05` | `5e-05` | `` | `` |  |
| Cost-weighted CE | `cost_weighted_ce` / `argmax` | `5e-05` | `5e-05` | `` | `` |  |
| CSADA | `csada` / `argmax` | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation baseline |
| CE + cost-min inference | `cross_entropy` / `cost_min` | `5e-05` | `5e-05` | `` | `` | Inference-only cost-min report from CE probabilities |
