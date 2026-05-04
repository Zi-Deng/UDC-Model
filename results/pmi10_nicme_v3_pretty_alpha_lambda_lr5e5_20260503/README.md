# PMI-10 NICME V3 Pretty Alpha/Lambda HPO, LR 5e-5

Status: complete after direct resume on 2026-05-03.

This folder is the authoritative record for the completed NICME v3 alpha/lambda grid over the PMI-10 balanced no-calibration split. The original chain process failed at run 103 with a transient PyTorch module traversal error, but the grid was resumed from the ledger and completed.

## Completion Status

- Grid rows: `108/108` completed, `0` failed.
- Final resumed grid run: run `108`, ended `2026-05-03T17:38:27`.
- Analysis completed: `2026-05-03T17:38:36`.
- Split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Split counts: train `970`, validation `310`, test `320`.
- Cost matrix hash: `1160cbf20c4fac24dc7bb84cbbb229a16545259d585ae1ef1c066e0007082e08`.

Important note: `chain/status.jsonl` contains the original failed chain entries from `2026-05-03T08:45:53-07:00`. Use `grid_run_ledger.csv`, `validation_ranked_table.csv`, `selected_config.json`, and this README as the current post-resume state.

## Authoritative Files

- `grid_run_ledger.csv`: all completed grid rows and test metrics.
- `validation_ranked_table.csv`: validation-ranked table used for selection.
- `selected_config.json`: validation-selected configuration.
- `selected_result_summary.md`: validation-selected result summary.
- `alpha_lambda_heatmap_data.csv`: alpha/lambda heatmap data.
- `post_hpo_sota_comparison.md`: consolidated comparison against previous SOTA/baseline summaries.
- `post_hpo_top_rows.csv`: machine-readable top rows from the post-HPO comparison.

## Validation-Selected Config

This is the predeclared validation-based selection from `selected_config.json`.

| Run | Alpha | Lambda | LR | Val Target-Min | Val Norm. ATC | Val Bal. Acc. | Test Target-Min | Test Norm. ATC | Test Bal. Acc. | Critical Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 89 | 0.4 | 0.2 | 5e-05 | 1.0000 | 0.000000 | 1.0000 | 0.9062 | 0.005000 | 0.9781 | 1 |

## Best Test-Ranked HPO Row

Run 95 is the best test-ranked row when ordered by target-min recall, target-macro recall, normalized ATC, balanced accuracy, and macro F1. Run 69 ties run 95 on the main cost-sensitive test metrics and differs only by a very small macro-F1 margin.

| Run | Alpha | Lambda | LR | Test Target-Min | Test Target-Macro | Test Norm. ATC | Test ATC | Test Bal. Acc. | Test Macro F1 | Critical Errors | Val Target-Min | Val Norm. ATC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 95 | 0.5 | 0.07 | 5e-05 | 0.9688 | 0.9844 | 0.004688 | 0.046875 | 0.9812 | 0.981386 | 1 | 1.0000 | 0.000323 |
| 69 | 0.2 | 0.09 | 5e-05 | 0.9688 | 0.9844 | 0.004688 | 0.046875 | 0.9812 | 0.981293 | 1 | 1.0000 | 0.001935 |

## Interpretation Guardrail

For strict model-selection reporting, use run 89 because it was selected by validation metrics. For exploratory test-set comparison, run 95 is the best observed HPO test row and is rank 2 by validation. A paper-quality claim that run 95 is the final selected configuration should be supported by a prospective selection rule or a fresh multi-seed confirmation, because choosing it after seeing test metrics is test-set peeking.
