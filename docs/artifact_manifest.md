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
| `archive/paper_position_cleanup_20260504/` | 3.1M | Historical versioned runners/tests and superseded paper-facing PMI-10 comparison output | Historical archive only |
| `playground/cost_sensitive_loss_classification/` | 5.4G | Archived comparison project and outputs | Reference as archived baseline material |

## Key Files And Checksums

| File | SHA256 |
|---|---|
| `weights/pytorch_model.bin` | `ff8163a1323333126706d649ce73ecd76e45d241b42d623dea6c723690cafe07` |
| `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv` | `c16a6aef221e1cb4d493be270250f720330b0d2f42f8a6e446577d13eeb84358` |
| `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md` | `764b8e19663a085f7a4751abf0dfe22aefd82e2c653c72a27940ea5e0f631089` |
| `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/claim_audit.md` | `b27b4e1c58ed03287c08bbc399ce90cb187bd4fd07c3b0451e9625a53d08adf6` |
| `docs/nicme_vs_csada_theory.pdf` | `636d0612c41f24b1f9000d2b2098e4e072ca5b01134d9e1ad80fad5c018f64f8` |
| `docs/nicme_hyperparameters.pdf` | `dbdf44abe91b335f019bf55b9507c7b4118ee3b0d86de8e331b269bdf317b8b7` |

## Current Canonical Artifact Roots

| Area | Root or file |
|---|---|
| Results index | `results/README.md` |
| PMI-20 paper SOTA table | `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/` |
| PMI-20 source SOTA baseline suite | `results/pmi20_camera_ready_lr5e5_multiseed_20260504/` |
| PMI-20 source NICME candidate suite | `results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/` |
| PMI-10 supporting multi-seed suite | `results/pmi10_camera_ready_lr5e5_multiseed_20260504/` |
| PMI-10 supporting alpha/lambda suite | `results/pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504/` |
| Binary final summaries | `results/stop3a_balanced_primary/`, `results/stop3b_imbalance_decoupling/`, `results/stop4a_backbone_ablation/`, `results/stop4b_cost_ratio_sensitivity/` |
| Multiclass active state | `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/`, `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/` |
| Theory memo | `docs/nicme_vs_csada_theory.tex`, `docs/nicme_vs_csada_theory.pdf` |
| Hyperparameter memo | `docs/nicme_hyperparameters.tex`, `docs/nicme_hyperparameters.pdf` |

## Historical Artifact Roots

The old two-class HPO result `results/hpo_results/best_hyperparameters.json` moved to `archive/results_docs_cleanup_20260504/results/hpo_results/best_hyperparameters.json`. Historical cost-matrix sweeps remain under `results/sweep_cost_*` and `results/sweep_comparison*` for audit. Historical versioned PMI-10 generated roots remain provenance only and are indexed through [pmi10_hpo_sota_summary.md](pmi10_hpo_sota_summary.md).

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
