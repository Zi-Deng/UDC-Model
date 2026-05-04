# NICME Artifact Manifest

Updated: 2026-05-04

This repository remains self-contained for now. Large artifacts are intentionally kept in the parent workspace and are not duplicated into `releases/`.

## Large Artifact Groups

| Path | Size | Purpose | Release-view policy |
|---|---:|---|---|
| `data/` | 145G | Local binary, multiclass, and PMI prepared datasets | Reference by relative path only |
| `weights/` | 98M | Pretrained ResNet weights | Reference by relative path only |
| `checkpoints/` | 327G | Trainer checkpoints, HPO checkpoints, and multiclass checkpoints | Do not copy into release views |
| `results/` | 666M | Metrics, plots, sweep CSVs, ledgers, and compact result summaries | Copy only compact summaries when needed |
| `docs/` | 820K | Live documentation and research memo PDFs | Include selected docs in releases as needed |
| `archive/results_docs_cleanup_20260504/` | 1.1M | Superseded probes, dry-runs, planning/preflight outputs, redundant comparison output, and stale memory | Historical archive only |
| `playground/cost_sensitive_loss_classification/` | 5.4G | Archived comparison project and outputs | Reference as archived baseline material |

## Key Files And Checksums

| File | SHA256 |
|---|---|
| `weights/pytorch_model.bin` | `ff8163a1323333126706d649ce73ecd76e45d241b42d623dea6c723690cafe07` |
| `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/grid_run_ledger.csv` | `547e5a7f178dbe45e6d2255e9c1370a90dbab7cf3942f4317b5e756b1235eb18` |
| `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/validation_ranked_table.csv` | `4b5fffb8bf0861d6b9e1564091cc8291ef79306998957be234b52068c9631c28` |
| `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/selected_config.json` | `471cf7a3757311a39185cc30f94c82c798e65a67c92d3ae3ede08d5a87817a41` |
| `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md` | `4a6b2dfe7ec33fe89bb16305bf2b4d2611e56afeda4f1d4b7e32ea906b2b7ab8` |
| `docs/nicme_v3_vs_csada_theory.pdf` | `e5f4797eafdf56e249c7fa50b9c3f8489c1ec8f92704f4ff298311b58de50ab3` |

## Current Canonical Artifact Roots

| Area | Root or file |
|---|---|
| Results index | `results/README.md` |
| PMI-10 current HPO | `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/` |
| PMI-10 post-HPO comparison | `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md` |
| PMI-10 prior combined baseline summary | `results/pmi10_sota_pretty_balanced_triple_lr_20260503/comparison_summary.md` |
| Binary final summaries | `results/stop3a_balanced_primary/`, `results/stop3b_imbalance_decoupling/`, `results/stop4a_backbone_ablation/`, `results/stop4b_cost_ratio_sensitivity/` |
| Multiclass active state | `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/`, `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/` |
| Theory memo | `docs/nicme_v3_vs_csada_theory.tex`, `docs/nicme_v3_vs_csada_theory.pdf` |

## Historical Artifact Roots

The old two-class HPO result `results/hpo_results/best_hyperparameters.json` moved to `archive/results_docs_cleanup_20260504/results/hpo_results/best_hyperparameters.json`. Historical cost-matrix sweeps remain under `results/sweep_cost_*` and `results/sweep_comparison*` for audit.

## Dataset Counts

| Dataset folder | Class | Count |
|---|---|---:|
| `data/2_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/2_class_black_widows` | `Steatoda_grossa` | 1499 |
| `data/3_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/3_class_black_widows` | `Steatoda_grossa` | 1500 |
| `data/3_class_black_widows` | `Steatoda_nobilis` | 1500 |
| `data/prepared/pmi_pills_10_no_cal/splits/balanced` | split counts | train 970, validation 310, test 320 |

## Future External URL Fields

External artifact URLs are intentionally pending. See `docs/external_artifact_migration_plan.md`.
