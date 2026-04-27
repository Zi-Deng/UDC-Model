# NICME Cost-Matrix Sweep Guide

NICME supports two sweep paths:

| Feature | Bash sweep | Python sweep |
|---|---|---|
| Entry point | `examples/run_cost_matrix_sweep.sh` | `scripts/cost_matrix_sweep.py` or `nicme-sweep` |
| Storage | CSV summary | CSV plus DuckDB |
| Plots | Matplotlib/seaborn | Plotly HTML/PNG |
| Best use | Simple local iteration | Paper-facing sweeps and comparisons |

## Python/Optuna Sweep

```bash
micromamba activate ml
python scripts/cost_matrix_sweep.py \
  --config config/nicme_2class_spiders_regularized.json \
  --row 0 \
  --col 1 \
  --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" \
  --output-dir results/nicme_sweep_cost_0_1_reg
```

Equivalent package command after editable install:

```bash
nicme-sweep \
  --config config/nicme_2class_spiders_regularized.json \
  --row 0 \
  --col 1 \
  --values "1,2,3"
```

Outputs:

```text
results/<sweep-output>/
|-- sweep_config.json
|-- sweep_results.csv
|-- cost_matrix_sweep.duckdb
`-- graphs/
```

## Bash Sweep

```bash
micromamba activate ml
bash examples/run_cost_matrix_sweep.sh config/nicme_sweep_2class_bw_cost.json
```

The bash sweep mutates the base config during each iteration and restores it on exit. Prefer the Python sweep for paper-facing experiments.

## Metrics Collected

- accuracy
- balanced accuracy when available
- macro F1
- kappa and AUC when available
- per-class precision, recall, F1, FPR, FNR
- confusion matrix cells
- expected cost for configured cost matrices

## Important Semantics

For the parent LogitAdj and hybrid NICME regularized losses, cost matrices are interpreted as:

```text
cost_matrix[true_label][predicted_label]
```

For the binary spider task:

```text
class 0 = Black Widow
class 1 = False Widow
```

Thus `M[0][1]` penalizes predicting Black Widow as False Widow.

