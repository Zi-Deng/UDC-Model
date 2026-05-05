# PMI-20 NICME Six-Candidate Hyperparameters

Generated: 2026-05-04T16:09:48

- Shared LR profile: `lr5e5`.
- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.
- Shared split: balanced full 20-class PMI.
- Loss: `nicme_hybrid`.

| Source | HPO Run | Candidate Order | Method | LR | alpha | lambda |
|---|---:|---:|---|---:|---:|---:|
| PMI-10 run 53 | 53 | 1 | NICME PMI-10 run 53 (alpha=0.09, lambda=0.2) | `5e-05` | `0.09` | `0.2` |
| PMI-10 run 20 | 20 | 2 | NICME PMI-10 run 20 (alpha=0.06, lambda=0.03) | `5e-05` | `0.06` | `0.03` |
| PMI-10 run 50 | 50 | 3 | NICME PMI-10 run 50 (alpha=0.09, lambda=0.07) | `5e-05` | `0.09` | `0.07` |
| PMI-10 run 95 | 95 | 4 | NICME PMI-10 run 95 (alpha=0.5, lambda=0.07) | `5e-05` | `0.5` | `0.07` |
| PMI-10 run 69 | 69 | 5 | NICME PMI-10 run 69 (alpha=0.2, lambda=0.09) | `5e-05` | `0.2` | `0.09` |
| added PMI-20 candidate |  | 6 | NICME added PMI-20 candidate (alpha=0.5, lambda=0.1) | `5e-05` | `0.5` | `0.1` |
