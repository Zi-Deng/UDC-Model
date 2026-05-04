# Current Status

Updated: 2026-05-04

## Repository

- Canonical root: `/mnt/storage/github/NICME`.
- Current branch tracks `origin/main`.
- Historical references to the previous sibling workspace path have been normalized to the canonical root in Markdown.
- The workspace migration record remains in [archive/migration_20260501/README.md](../archive/migration_20260501/README.md).

## Documentation State

- Live docs are intentionally compact and indexed from [docs/README.md](README.md).
- Live results are indexed from [results/README.md](../results/README.md).
- Historical plans, long memory files, dated session reports, and Markdown result summaries are archived under [archive/markdown_consolidation_20260501/](../archive/markdown_consolidation_20260501/).
- Superseded probes, dry-runs, launcher-only result folders, planning/preflight outputs, redundant PMI-10 dual-LR comparison output, and stale multiclass memory state are archived under [archive/results_docs_cleanup_20260504/](../archive/results_docs_cleanup_20260504/MANIFEST.md).
- Generated checkpoint READMEs, prepared-data audits, release-view docs, and playground-local docs remain in place because moving them would make those artifact trees less self-contained.

## Experiment State

- Binary Spider/BreaKHis Stop 3 and Stop 4 experiments are complete. See [binary_experiment_summary.md](binary_experiment_summary.md).
- The previous multiclass EyePACS DR and 20-class PMI Pills MC plans are shelved for now. MC3 remains paused after partial progress. See [multiclass_experiment_summary.md](multiclass_experiment_summary.md).
- The focused PMI-10 no-calibration track has completed the balanced LR 5e-5 NICME v3 alpha/lambda HPO. See [pmi10_hpo_sota_summary.md](pmi10_hpo_sota_summary.md).

## Current Source Pointers

- Results index: `results/README.md`
- Focused PMI-10 balanced HPO split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Completed PMI-10 NICME v3 HPO: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/README.md`
- Completed PMI-10 post-HPO SOTA/baseline comparison: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`
- PMI-10 pre-HPO combined baseline summary: `results/pmi10_sota_pretty_balanced_triple_lr_20260503/comparison_summary.md`
- NICME v3 vs CSADA theory memo: `docs/nicme_v3_vs_csada_theory.pdf`
- Current multiclass ledger: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`
- MC2 selection JSON: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.json`
- MC3 launch commands: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc3_launch_commands.sh`

## Historical Pointers

- Markdown archive index: [archive/markdown_consolidation_20260501/INDEX.md](../archive/markdown_consolidation_20260501/INDEX.md)
- Results/docs cleanup archive: [archive/results_docs_cleanup_20260504/MANIFEST.md](../archive/results_docs_cleanup_20260504/MANIFEST.md)
- Earlier PMI-10 no-calibration readiness, smoke, screen, and HPO planning remains in `results/pmi10_no_cal_20260501/` for audit only.
- The old two-class HPO output moved to `archive/results_docs_cleanup_20260504/results/hpo_results/` and is no longer current best-model guidance.

## Resume Guidance

The focused PMI-10 balanced HPO is complete; use the completed HPO guide, post-HPO comparison, and theory memo above before launching any further PMI-10 follow-up. Before resuming MC3 later, run a fresh health check for D-state user processes, GPU taint, and CUDA canary status. Completed MC3 rows should be skipped by the runner, and the paused EyePACS cost-sensitive CE seed should be rerun from a fresh attempt.
