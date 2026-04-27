# NICME Artifact Manifest

This repository remains self-contained for now. Large artifacts are intentionally kept in the parent workspace and are not duplicated into `releases/`.

## Large Artifact Groups

| Path | Size | Purpose | Release-view policy |
|---|---:|---|---|
| `data/` | 732M | Local spider datasets | Reference by relative path only |
| `weights/` | 98M | Pretrained ResNet weights | Reference by relative path only |
| `checkpoints/` | 19G | Trainer checkpoints and HPO checkpoints | Do not copy into release views |
| `results/` | 284M | Metrics, plots, sweep CSVs, DuckDB outputs | Copy only compact summaries when needed |
| `playground/cost_sensitive_loss_classification/` | 5.4G | Archived comparison project and outputs | Reference as archived baseline material |

## Key Files And Checksums

| File | SHA256 |
|---|---|
| `weights/pytorch_model.bin` | `ff8163a1323333126706d649ce73ecd76e45d241b42d623dea6c723690cafe07` |
| `results/hpo_results/best_hyperparameters.json` | `18d1e6d3b5e49692b9d3e706ad7d86b10d2721f643c3b87aa102d79d89e0e23b` |
| `results/sweep_cost_0_1/sweep_results.csv` | `ebd08da0f6462daa05ce3ddf2c793993f058bf716d65335d6e59efc6394c0464` |
| `results/sweep_cost_0_1_reg/sweep_results.csv` | `60e57d97a66ddcf4bf3309b0bd37098c1e2f9af21571dc3b11131549f7c71ed9` |
| `results/sweep_cost_1_0/sweep_results.csv` | `1eb1a5610e00e680f872f629fb7e72a9efaf7d7b8d368aae16f394afe68e9646` |
| `results/sweep_cost_1_0_reg/sweep_results.csv` | `d07c28ae3a760d60607eff48e8ec9d7ed8b6b8fed85a4f581afdec426be8` |
| `results/sweep_comparison/comparison_summary.md` | `78e47c3e6716cbc5c0541dcbb95ca9a455af5f1933b1cd8b91633750b84f2d94` |
| `results/sweep_comparison_1_0/comparison_summary.md` | `bed23ffb7b008de2135b0ebb4947067719614eea2b0e9a2df60f7a5dfb4637e4` |

## Dataset Counts

| Dataset folder | Class | Count |
|---|---|---:|
| `data/2_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/2_class_black_widows` | `Steatoda_grossa` | 1499 |
| `data/3_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/3_class_black_widows` | `Steatoda_grossa` | 1500 |
| `data/3_class_black_widows` | `Steatoda_nobilis` | 1500 |

## Future External URL Fields

External artifact URLs are intentionally pending. See `docs/external_artifact_migration_plan.md`.

