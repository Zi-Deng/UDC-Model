# PMI-20 NICME Component Ablation

Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44.

| Component | alpha | lambda | Target-Min Recall | Norm. ATC | Balanced Acc. | Macro F1 | Critical Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE | 0 | 0 | 0.9167 +/- 0.0180 | 0.005677 +/- 0.003103 | 0.9714 +/- 0.0174 | 0.9711 +/- 0.0179 | 2.0000 +/- 1.0000 (total 6) |
| Regularizer-only | 0 | 0.1 | 0.8958 +/- 0.0180 | 0.006615 +/- 0.001329 | 0.9667 +/- 0.0055 | 0.9666 +/- 0.0056 | 2.3333 +/- 0.5774 (total 7) |
| Margin-only | 0.5 | 0 | 0.8750 +/- 0.0541 | 0.007500 +/- 0.004752 | 0.9474 +/- 0.0574 | 0.9472 +/- 0.0577 | 1.6667 +/- 0.5774 (total 5) |
| Full NICME | 0.5 | 0.1 | 0.9167 +/- 0.0180 | 0.003698 +/- 0.000955 | 0.9771 +/- 0.0106 | 0.9771 +/- 0.0106 | 1.0000 +/- 1.0000 (total 3) |
