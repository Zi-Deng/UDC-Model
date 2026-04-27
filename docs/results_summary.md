# NICME Results Summary

This document summarizes the current local results used for NICME paper-facing reporting. Values are copied from the CSV/JSON artifacts in `results/`.

## Hyperparameter Optimization

Source: `results/hpo_results/best_hyperparameters.json`

| Field | Value |
|---|---:|
| Best trial | 4 |
| Best eval accuracy | 0.937037 |
| Learning rate | 0.0003167583 |
| Weight decay | 0.009658 |
| Batch size | 32 |
| Warmup ratio | 0.086561 |
| LR scheduler | linear |
| Frozen ResNet stages | 3 |

## Hybrid NICME Sweep Results

Source files:

- `results/sweep_cost_0_1_reg/sweep_results.csv`
- `results/sweep_cost_1_0_reg/sweep_results.csv`

| Sweep cell | Meaning | Best accuracy | Best class-0 recall | Key observation |
|---|---|---:|---:|---|
| `M[0][1]` | Black Widow predicted as False Widow | 0.896667 at cost 5 | 0.980000 at cost 6 | High class-0 recall without the collapse seen in the playground baseline |
| `M[1][0]` | False Widow predicted as Black Widow | 0.896667 at cost 1 | 0.900000 at cost 1 | Best operating point stayed at low cost; high reverse penalties were not useful for class-0 recall |

## Three-Way Comparison

Source files:

- `results/sweep_comparison/comparison_summary.md`
- `results/sweep_comparison_1_0/comparison_summary.md`

| Sweep | Parent LogitAdj | Playground CE+CS | Hybrid NICME LogitAdj+CS |
|---|---:|---:|---:|
| `M[0][1]` best accuracy | 0.8933 | 0.8886 | 0.8967 |
| `M[0][1]` best class-0 recall | 0.9467 | 1.0000 | 0.9800 |
| `M[0][1]` collapse detected | No | Yes, at costs 80, 90, 100 | No |
| `M[1][0]` best accuracy | 0.8767 | 0.8686 | 0.8967 |
| `M[1][0]` best class-0 recall | 0.8667 | 0.7822 | 0.9000 |
| `M[1][0]` collapse detected | No | Yes, at costs 7, 50, 80, 90, 100 | No |

## Interpretation

The current evidence supports the hybrid NICME regularized method as the best practical operating point among the evaluated pipelines: it improves or preserves accuracy, offers high class-0 recall, and avoids the single-class collapse behavior observed in the playground CE+CS baseline at high costs.

