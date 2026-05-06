# Binary Camera-Ready Method Hyperparameters

Generated: 2026-05-05T04:48:28

- Shared LR profile: `lr5e5`.
- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared splits: balanced Spider and balanced BreaKHis.
- Primary matrices: Spider `[[0,8],[1,0]]`; BreaKHis `[[0,1],[7,0]]`.

| Method | LR | Parent LR | Alpha | Lambda | Notes |
|---|---:|---:|---:|---:|---|
| CE | `5e-05` | `5e-05` | `` | `` |  |
| CE + cost-min inference | `5e-05` | `5e-05` | `` | `` | Inference-only report from CE anchor |
| NICME (alpha=0.5, lambda=0.1) | `5e-05` | `5e-05` | `0.5` | `0.1` | Primary NICME setting from PMI-20 camera-ready protocol |
| Menon logit adjustment | `5e-05` | `5e-05` | `` | `` |  |
| Cost-weighted CE | `5e-05` | `5e-05` | `` | `` |  |
| cost-sensitive regularized CE | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
| CSADA | `1e-05` | `5e-05` | `` | `` | CE-anchor adaptation; native adaptation LR `1e-5` |
