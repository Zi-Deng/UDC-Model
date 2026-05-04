# Results And Docs Cleanup Archive - 2026-05-04

This archive contains small, superseded planning, probe, dry-run, launcher-only, and redundant comparison artifacts moved out of the live `results/` and `memory/` surfaces during the 2026-05-04 cleanup.

No experiment data was deleted. Paths are preserved under this archive so the original location is recoverable by removing the `archive/results_docs_cleanup_20260504/` prefix.

## Cleanup Policy

- Keep completed experiment ledgers, logs, and result trees live when they still support current summaries or audit trails.
- Archive small folders whose main purpose was planning, preflight validation, launch debugging, dry-run config generation, or superseded comparison display.
- Keep generated checkpoint READMEs, data-preparation audits, and release-view files in their artifact-local locations.
- Treat archived files as historical, not as current guidance.

## Moved Paths

| Original path | Archived path | Reason | Current replacement or status |
|---|---|---|---|
| `results/stop3_launch_probe/` | `archive/results_docs_cleanup_20260504/results/stop3_launch_probe/` | Stop 3 launch probe only. | Final Stop 3A artifacts remain in `results/stop3a_balanced_primary/`. |
| `results/stop3b_launch_check/` | `archive/results_docs_cleanup_20260504/results/stop3b_launch_check/` | Stop 3B launch check only. | Final Stop 3B artifacts remain in `results/stop3b_imbalance_decoupling/`. |
| `results/stop3_dry_run/` | `archive/results_docs_cleanup_20260504/results/stop3_dry_run/` | Dry-run config plan only. | Final Stop 3A artifacts remain in `results/stop3a_balanced_primary/`. |
| `results/stop3b_nohup_probe/` | `archive/results_docs_cleanup_20260504/results/stop3b_nohup_probe/` | Nohup launch probe only. | Final Stop 3B artifacts remain in `results/stop3b_imbalance_decoupling/`. |
| `results/stop3b_parse_probe/` | `archive/results_docs_cleanup_20260504/results/stop3b_parse_probe/` | Parse/config probe only. | Final Stop 3B artifacts remain in `results/stop3b_imbalance_decoupling/`. |
| `results/stop3b_stop4a_chain/` | `archive/results_docs_cleanup_20260504/results/stop3b_stop4a_chain/` | Launcher-only chain state after Stop 3B/4A completion. | Final Stop 3B and Stop 4A artifacts remain live. |
| `results/stop4b_dry_run/` | `archive/results_docs_cleanup_20260504/results/stop4b_dry_run/` | Dry-run config plan only. | Final Stop 4B artifacts remain in `results/stop4b_cost_ratio_sensitivity/`. |
| `results/multiclass_clean_plan/` | `archive/results_docs_cleanup_20260504/results/multiclass_clean_plan/` | Superseded MC1 planning-only output. | Current multiclass status is summarized in `docs/multiclass_experiment_summary.md`. |
| `results/multiclass_mc2_preflight_check_20260430/` | `archive/results_docs_cleanup_20260504/results/multiclass_mc2_preflight_check_20260430/` | Superseded MC2 preflight-only output. | Official MC2 artifacts remain in `results/multiclass_mc2_official_dinov3_lora_balanced_20260501/`. |
| `results/multiclass_mc2_fulltune_plan_20260501/` | `archive/results_docs_cleanup_20260504/results/multiclass_mc2_fulltune_plan_20260501/` | Superseded MC2 planning-only output. | Official MC2 artifacts remain live. |
| `results/multiclass_mc2_hardened_plan_check_20260430/` | `archive/results_docs_cleanup_20260504/results/multiclass_mc2_hardened_plan_check_20260430/` | Superseded MC2 planning/check output. | Official MC2 artifacts remain live. |
| `results/multiclass_mc_fb0_dinov3_storage_20260501/` | `archive/results_docs_cleanup_20260504/results/multiclass_mc_fb0_dinov3_storage_20260501/` | Storage-access audit superseded by live DINOv3 docs and MC-FB2/MC2 artifacts. | See `docs/huggingface_dinov3_access.md` and `docs/multiclass_experiment_summary.md`. |
| `results/multiclass_pause_summary_20260501/` | `archive/results_docs_cleanup_20260504/results/multiclass_pause_summary_20260501/` | Historical pause snapshot already consolidated into live docs. | See `docs/multiclass_experiment_summary.md`. |
| `results/multiclass_pmi_ready_20260430/` | `archive/results_docs_cleanup_20260504/results/multiclass_pmi_ready_20260430/` | Historical PMI readiness snapshot already consolidated into live docs. | See `docs/multiclass_experiment_summary.md`. |
| `results/hpo_results/` | `archive/results_docs_cleanup_20260504/results/hpo_results/` | Old two-class HPO output no longer represents current best work. | Historical binary HPO reference only. |
| `results/pmi10_sota_pretty_balanced_dual_lr_20260503/` | `archive/results_docs_cleanup_20260504/results/pmi10_sota_pretty_balanced_dual_lr_20260503/` | Redundant two-LR PMI-10 comparison superseded by triple-LR and post-HPO comparison. | Use `results/pmi10_sota_pretty_balanced_triple_lr_20260503/` and `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/post_hpo_sota_comparison.md`. |
| `memory/multiclass_2026_05_01/` | `archive/results_docs_cleanup_20260504/memory/multiclass_2026_05_01/` | Stale agent memory snapshot already represented in live docs and prior archive. | Live memory entrypoint is `memory/README.md`. |

## Live Canonical Entry Points

- Repository docs index: `docs/README.md`
- Current status: `docs/current_status.md`
- Results index: `results/README.md`
- PMI-10 HPO summary: `docs/pmi10_hpo_sota_summary.md`
- Completed PMI-10 HPO root: `results/pmi10_nicme_v3_pretty_alpha_lambda_lr5e5_20260503/`
- Binary summary: `docs/binary_experiment_summary.md`
- Multiclass summary: `docs/multiclass_experiment_summary.md`
- Theory memo: `docs/nicme_v3_vs_csada_theory.pdf`
