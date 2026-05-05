# NICME Top-5 Hyperparameters

Generated: 2026-05-04T04:23:32

- Shared LR profile: `lr5e5`.
- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared split: balanced PMI-10 no-calibration.
- Loss: `nicme_hybrid`.

| HPO Run | Pilot Rank | Method | LR | alpha | lambda |
|---:|---:|---|---:|---:|---:|
| 95 | 1 | NICME run 95 (alpha=0.5, lambda=0.07) | `5e-05` | `0.5` | `0.07` |
| 69 | 2 | NICME run 69 (alpha=0.2, lambda=0.09) | `5e-05` | `0.2` | `0.09` |
| 20 | 3 | NICME run 20 (alpha=0.06, lambda=0.03) | `5e-05` | `0.06` | `0.03` |
| 50 | 4 | NICME run 50 (alpha=0.09, lambda=0.07) | `5e-05` | `0.09` | `0.07` |
| 53 | 5 | NICME run 53 (alpha=0.09, lambda=0.2) | `5e-05` | `0.09` | `0.2` |
