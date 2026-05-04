# Paused ConvNeXt-Base PMI-10 HPO Run

Paused at: `2026-05-02T02:16:01-07:00`

Reason: User requested pause to free GPU for a new task suite.

## Run State

- Result root: `/mnt/storage/github/NICME/results/pmi10_no_cal_convnext_base_20260501`
- Phase at pause: `hpo-nicme`
- NICME HPO completed: `16 / 36`
- Current in-progress trial config at pause: `/mnt/storage/github/NICME/results/pmi10_no_cal_convnext_base_20260501/hpo-nicme/configs/0017_91de691633d8a184.json`
- Current in-progress output directory at pause: `hpo-nicme_convnext_base_fb_in22k_ft_in1k_nicme_hybrid_s42_trial0016`
- CE HPO started: `False`
- Final evaluation started: `False`

## Best Completed NICME Trial So Far

- Best completed run ID: `13`
- Best config: `/mnt/storage/github/NICME/results/pmi10_no_cal_convnext_base_20260501/hpo-nicme/configs/0013_c20a8c3e21a2f617.json`
- Best metrics: `results/pmi10_convnext_base_fb_in22k_ft_in1k_test/hpo-nicme_convnext_base_fb_in22k_ft_in1k_nicme_hybrid_s42_trial0012_05-02_01-32/metrics_20260502_013242_nicme_hybrid.json`
- Target-min recall: `0.995`
- Target-macro recall: `0.99875`
- Normalized ATC: `0.0021168501270110076`
- Balanced accuracy: `0.9730851878439818`
- Critical-pair errors: `1`

## Snapshot Contents

- `pause_state.json`: machine-readable pause state.
- `process_snapshot.txt`: process table before stopping.
- `gpu_snapshot.txt`: GPU state before stopping.
- `snapshot_files/`: copied ledgers, chain status/log, preflight report, HPO configs, and HPO stdout/stderr logs.

## Continuation Notes

Completed trial metrics and run artifacts remain in their original result directories. The in-progress trial at pause time should be treated as interrupted unless a complete metrics JSON and ledger row exists. To continue later, use the completed rows in `hpo-nicme/hpo_trials.csv` as the authoritative completed-trial record, then decide whether to launch the remaining NICME trials or proceed from the best completed config.
