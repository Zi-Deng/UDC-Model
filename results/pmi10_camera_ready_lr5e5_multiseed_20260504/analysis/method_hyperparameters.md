# PMI-10 Camera-Ready Method Hyperparameters

Generated: 2026-05-03T23:45:54

- Shared LR profile: `lr5e5`.
- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared split: balanced PMI-10 no-calibration.

| Method | LR | Parent LR | Alpha | Lambda | Notes |
|---|---:|---:|---:|---:|---|
| CE | `5e-05` | `5e-05` | `` | `` |  |
| CE + cost-min inference | `5e-05` | `5e-05` | `` | `` | Inference-only report from CE anchor |
| NICME v3 (alpha=0.5, lambda=0.07) | `5e-05` | `5e-05` | `0.5` | `0.07` | Pilot-best fixed config; no post-rerun HPO |
| NICME v2 hybrid | `5e-05` | `5e-05` | `0.4` | `0.25` | Existing LR 5e-5 baseline hyperparameters |
| Cost-weighted CE | `5e-05` | `5e-05` | `` | `` |  |
| Menon logit adjustment | `5e-05` | `5e-05` | `` | `` |  |
| Balanced softmax | `5e-05` | `5e-05` | `` | `` |  |
| Class-balanced focal | `5e-05` | `5e-05` | `` | `` |  |
| LDAM-DRW | `5e-05` | `5e-05` | `` | `` |  |
| cost-sensitive regularized CE | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
| SOSR-CNN | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
| CSADA | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
