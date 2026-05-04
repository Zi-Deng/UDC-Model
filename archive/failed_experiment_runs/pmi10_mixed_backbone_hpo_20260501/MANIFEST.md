# Failed Mixed-Backbone PMI-10 HPO Archive

Archived as part of the ConvNeXt-base-only PMI-10 HPO recovery plan.

## Archived Paths

- `results/pmi10_no_cal_20260501/hpo-nicme` -> `archive/failed_experiment_runs/pmi10_mixed_backbone_hpo_20260501/results/pmi10_no_cal_20260501/hpo-nicme`
- `results/pmi10_no_cal_20260501/hpo-ce` -> `archive/failed_experiment_runs/pmi10_mixed_backbone_hpo_20260501/results/pmi10_no_cal_20260501/hpo-ce`

## Reason

The mixed-backbone NICME HPO was stopped by an Optuna dynamic categorical search-space error after trial 8. It is no longer part of the active experiment plan, which now resumes from Phase 2 using only `convnext_base.fb_in22k_ft_in1k` with the fixed `convnext_base_fast` HPO profile.

The completed smoke and screen artifacts remain in `results/pmi10_no_cal_20260501/` because they are still valid source-of-truth screening context.
