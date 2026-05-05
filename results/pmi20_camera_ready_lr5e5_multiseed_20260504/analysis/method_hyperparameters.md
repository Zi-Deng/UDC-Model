# PMI-20 Camera-Ready Method Hyperparameters

Generated: 2026-05-04T10:33:40

- Shared LR profile: `lr5e5`.
- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared split: balanced full 20-class PMI.

| Method | LR | Parent LR | Alpha | Lambda | Notes |
|---|---:|---:|---:|---:|---|
| CE | `5e-05` | `5e-05` | `` | `` |  |
| CE + cost-min inference | `5e-05` | `5e-05` | `` | `` | Inference-only report from CE anchor |
| NICME v3 (alpha=0.09, lambda=0.07) | `5e-05` | `5e-05` | `0.09` | `0.07` | PMI-10 run-50 alpha/lambda reused; no PMI-20 HPO |
| Menon logit adjustment | `5e-05` | `5e-05` | `` | `` |  |
| Cost-weighted CE | `5e-05` | `5e-05` | `` | `` |  |
| AP-CSADA | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
| CSADA | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
