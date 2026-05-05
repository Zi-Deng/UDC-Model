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
- Historical versioned runner/test files and superseded paper-facing comparison outputs are archived under [archive/paper_position_cleanup_20260504/](../archive/paper_position_cleanup_20260504/MANIFEST.md).
- Generated checkpoint READMEs, prepared-data audits, release-view docs, and playground-local docs remain in place because moving them would make those artifact trees less self-contained.

## Experiment State

- Binary Spider/BreaKHis Stop 3 and Stop 4 experiments are complete. See [binary_experiment_summary.md](binary_experiment_summary.md).
- The previous multiclass EyePACS DR and 20-class PMI Pills MC plans are shelved for now. MC3 remains paused after partial progress. See [multiclass_experiment_summary.md](multiclass_experiment_summary.md).
- The current paper-facing result is the PMI-20 balanced LR 5e-5 consolidated NICME SOTA table. NICME alpha `0.5`, lambda `0.1` ranks first under the recall-first cost-sensitive composite. See [paper_results_summary.md](paper_results_summary.md).
- The focused PMI-10 no-calibration track is complete and now serves as supporting sensitivity/robustness evidence. See [pmi10_hpo_sota_summary.md](pmi10_hpo_sota_summary.md).

## Current Source Pointers

- Results index: `results/README.md`
- Paper results summary: `docs/paper_results_summary.md`
- Canonical PMI-20 paper table: `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/pmi20_sota_table.md`
- Canonical PMI-20 claim audit: `results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/claim_audit.md`
- PMI-20 balanced split: `data/prepared/pmi_pills/splits/balanced`
- Supporting PMI-10 balanced split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Supporting PMI-10 HPO: retained as historical single-seed provenance; use [pmi10_hpo_sota_summary.md](pmi10_hpo_sota_summary.md) as the entry point.
- NICME vs CSADA theory memo: `docs/nicme_vs_csada_theory.pdf`
- Current multiclass ledger: `results/multiclass_mc3_selected_official_dinov3_lora_balanced_20260501/mc3/run_ledger.csv`
- MC2 selection JSON: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc2_selection.json`
- MC3 launch commands: `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/selection/mc3_launch_commands.sh`

## Historical Pointers

- Markdown archive index: [archive/markdown_consolidation_20260501/INDEX.md](../archive/markdown_consolidation_20260501/INDEX.md)
- Results/docs cleanup archive: [archive/results_docs_cleanup_20260504/MANIFEST.md](../archive/results_docs_cleanup_20260504/MANIFEST.md)
- Paper-position cleanup archive: [archive/paper_position_cleanup_20260504/MANIFEST.md](../archive/paper_position_cleanup_20260504/MANIFEST.md)
- Earlier PMI-10 no-calibration readiness, smoke, screen, and HPO planning remains in `results/pmi10_no_cal_20260501/` for audit only.
- The old two-class HPO output moved to `archive/results_docs_cleanup_20260504/results/hpo_results/` and is no longer current best-model guidance.

## Resume Guidance

Use [paper_results_summary.md](paper_results_summary.md) and the canonical PMI-20 consolidated table before writing or revising paper claims. Before resuming MC3 later, run a fresh health check for D-state user processes, GPU taint, and CUDA canary status. Completed MC3 rows should be skipped by the runner, and the paused EyePACS cost-sensitive CE seed should be rerun from a fresh attempt.
