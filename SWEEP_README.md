# Cost Matrix Sweep Guide

This guide covers both sweep implementations for automating cost matrix experiments.

## Which Sweep to Use

| Feature | Bash sweep (`run_cost_matrix_sweep.sh`) | Python sweep (`scripts/cost_matrix_sweep.py`) |
|---|---|---|
| **Setup** | Zero dependencies beyond training | Requires Optuna, DuckDB, Polars, Plotly |
| **Storage** | CSV file | DuckDB database |
| **Analysis** | Matplotlib static graphs | Plotly interactive charts |
| **Configuration** | Separate sweep JSON config | CLI flags + training config |
| **Resume support** | No (restarts from scratch) | Yes (DuckDB persists trials) |
| **Best for** | Simple, one-off sweeps | Repeated/resumable experiments |

## Bash Sweep

### Quick Start

```bash
micromamba activate ml
bash run_cost_matrix_sweep.sh                                    # default config
bash run_cost_matrix_sweep.sh config/sweep_2class_bw_cost.json   # custom config
```

Run multiple sweep configs sequentially:
```bash
bash matrix_sweep_loop.sh
```

### Configuration File Structure

The sweep configuration is defined in a JSON file:

```json
{
    "cost_range": {
        "min": 0.0,
        "max": 10.0,
        "step": 0.5
    },
    "matrix_cell": {
        "row": 1,
        "col": 2,
        "description": "Matrix cell [row, col] to modify during sweep"
    },
    "experiment": {
        "output_dir": "results/cost_matrix_sweep_results",
        "base_config": "config/2classSpiders.json",
        "description": "Cost matrix sweep experiment description"
    },
    "analysis": {
        "update_graphs_each_iteration": true,
        "generate_final_report": true
    }
}
```

### Configuration Parameters

#### `cost_range`
- **`min`**: Minimum cost value (float)
- **`max`**: Maximum cost value (float)
- **`step`**: Step size between values (float)

#### `matrix_cell`
- **`row`**: Row index of the cost matrix cell to modify (0-based)
- **`col`**: Column index of the cost matrix cell to modify (0-based)
- **`description`**: Human-readable description

#### `experiment`
- **`output_dir`**: Directory path for results
- **`base_config`**: Path to the base training configuration JSON file
- **`description`**: Description of the experiment

#### `analysis`
- **`update_graphs_each_iteration`**: Update graphs after each training run (boolean)
- **`generate_final_report`**: Generate a final comprehensive report (boolean)

### Bash Sweep Output Structure

```
results/{output_dir}/
├── metrics_summary.csv          # Aggregated metrics from all runs
├── logs/                        # Training logs for each iteration
│   ├── training_cost_0.0.log
│   ├── training_cost_0.5.log
│   └── ...
└── graphs/                      # Analysis visualizations
    ├── overall_metrics.png
    ├── class{X}_metrics.png
    ├── confusion_cell_{row}_{col}_metrics.png
    └── comprehensive_dashboard_{row}_{col}.png
```

## Python/Optuna Sweep

### Quick Start

```bash
micromamba activate ml
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json
```

### CLI Options

```bash
python scripts/cost_matrix_sweep.py \
    --config config/2classSpiders.json \
    --min 0.0 \          # minimum cost value (default: 0.0)
    --max 10.0 \         # maximum cost value (default: 10.0)
    --step 0.5 \         # step size (default: 0.5)
    --row 0 \            # cost matrix row to sweep (default: 0)
    --col 1 \            # cost matrix col to sweep (default: 1)
    --output results/sweep_output   # output directory
```

### How It Works

1. Generates a grid of cost values from `min` to `max` with `step` increments
2. Uses Optuna `GridSampler` to iterate through each value
3. For each trial: updates the cost matrix cell, runs full training, collects metrics
4. Persists all results to a DuckDB database in the output directory
5. After all trials: generates interactive Plotly visualizations

### Python Sweep Output Structure

```
results/{output}/
├── sweep.duckdb                 # DuckDB database with all trial results
├── metrics_vs_cost.html         # Interactive accuracy/F1 vs cost chart
├── confusion_heatmaps.html      # Confusion matrix heatmaps per cost value
└── per_class_metrics.html       # Per-class precision/recall/F1 charts
```

### Resume Support

The Python sweep automatically resumes from where it left off. If interrupted, re-run the same command and completed trials will be skipped.

## Metrics Collected (Both Sweeps)

For each cost value:
- Overall accuracy, precision, recall, F1 score
- Per-class accuracy, precision, recall, F1
- Confusion matrix cell values (raw counts and rates)
- Training loss

## Tips

1. **Quick testing**: Use `--min 0 --max 1 --step 1` (just 2 values) to verify the pipeline works
2. **Bash sweep — faster iteration**: Set `update_graphs_each_iteration` to `false` to skip graph generation between runs
3. **Python sweep — resumability**: If a sweep is interrupted, just re-run the same command
4. **Different cells**: Experiment with different `row` and `col` values to explore the cost matrix space
