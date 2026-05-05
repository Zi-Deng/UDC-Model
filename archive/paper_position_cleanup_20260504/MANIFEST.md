# Paper Position Cleanup Manifest

Created: 2026-05-04

Purpose: consolidate the repository around versionless **NICME** and the PMI-20 paper-facing result with alpha `0.5`, lambda `0.1`. All moves were non-destructive. Raw metric trees, checkpoints, datasets, and completed source suites remain live unless listed below.

## Archived Paths

| Original path | Archived path | Reason | Canonical live replacement |
|---|---|---|---|
| `scripts/run_pmi10_no_cal_experiments.py` | `archive/paper_position_cleanup_20260504/scripts/run_pmi10_no_cal_experiments.py` | Historical no-calibration runner retained old versioned experiment language and previous NICME variants. | Current PMI-10 support runners: `scripts/run_pmi10_camera_ready_lr5e5.py`, `scripts/run_pmi10_nicme_top5_lr5e5.py` |
| `scripts/launch_pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_chain.sh` | `archive/paper_position_cleanup_20260504/scripts/launch_pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_chain.sh` | Historical single-seed PMI-10 HPO launcher; no longer current paper workflow. | `scripts/launch_pmi10_nicme_top5_lr5e5_chain.sh` for supporting PMI-10 sensitivity reruns |
| `scripts/launch_pmi10_v3_balanced_phase3_chain.sh` | `archive/paper_position_cleanup_20260504/scripts/launch_pmi10_v3_balanced_phase3_chain.sh` | Historical phase-3 launcher for earlier versioned comparisons. | Current paper sources are PMI-20 runners: `scripts/launch_pmi20_camera_ready_lr5e5_chain.sh`, `scripts/launch_pmi20_nicme_alpha_lambda6_lr5e5_chain.sh` |
| `tests/test_pmi10_no_cal_runner.py` | `archive/paper_position_cleanup_20260504/tests/test_pmi10_no_cal_runner.py` | Historical tests cover archived runner behavior and previous versioned method names. | Current tests: `tests/test_pmi10_camera_ready_runner.py`, `tests/test_pmi10_nicme_runner.py` |
| `results/pmi10_sota_pretty_balanced_lr1e5_20260503/` | `archive/paper_position_cleanup_20260504/results/pmi10_sota_pretty_balanced_lr1e5_20260503/` | Superseded single-seed PMI-10 LR-specific comparison output. | PMI-20 paper table: `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/` |
| `results/pmi10_sota_pretty_balanced_lr5e5_20260503/` | `archive/paper_position_cleanup_20260504/results/pmi10_sota_pretty_balanced_lr5e5_20260503/` | Superseded single-seed PMI-10 LR-specific comparison output. | PMI-10 multi-seed support table: `results/pmi10_camera_ready_lr5e5_multiseed_20260504/analysis/neurips_table.md` |
| `results/pmi10_sota_pretty_balanced_lr1e4_20260503/` | `archive/paper_position_cleanup_20260504/results/pmi10_sota_pretty_balanced_lr1e4_20260503/` | Superseded single-seed PMI-10 LR-specific comparison output. | PMI-20 paper table and PMI-10 multi-seed support table |
| `results/pmi10_sota_pretty_balanced_triple_lr_20260503/` | `archive/paper_position_cleanup_20260504/results/pmi10_sota_pretty_balanced_triple_lr_20260503/` | Superseded combined PMI-10 LR comparison summary. | `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md` |

## Renamed Live Paths

| Previous path | Current path | Reason |
|---|---|---|
| `docs/nicme_v3_hyperparameters.tex` / `.pdf` | `docs/nicme_hyperparameters.tex` / `.pdf` | Public memo should use the current versionless method name. |
| `docs/nicme_v3_vs_csada_theory.tex` / `.pdf` | `docs/nicme_vs_csada_theory.tex` / `.pdf` | Public theory memo should use the current versionless method name. |
| `scripts/run_pmi10_nicme_v3_top5_lr5e5.py` | `scripts/run_pmi10_nicme_top5_lr5e5.py` | Future runner should emit versionless NICME configs. |
| `scripts/run_pmi20_nicme_v3_alpha_lambda6_lr5e5.py` | `scripts/run_pmi20_nicme_alpha_lambda6_lr5e5.py` | Future runner should emit versionless NICME configs. |
| `results/pmi20_nicme_v3_alpha_lambda6_lr5e5_multiseed_20260504/` | `results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/` | Completed PMI-20 candidate source suite is retained live with a versionless result-root name. |
| `results/pmi10_nicme_v3_top5_alpha_lambda_lr5e5_multiseed_20260504/` | `results/pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504/` | Supporting PMI-10 alpha/lambda source suite is retained live with a versionless result-root name. |

## Notes

- The canonical paper-facing output is `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/`.
- Historical generated configs and ledgers may still contain old method strings where rewriting would harm provenance.
- Deprecated loss aliases remain tested in `tests/test_loss_functions.py` for old config compatibility.
