# Binary Cost Matrix Review Protocol

Updated: 2026-05-05

This protocol defines the public-source review used to construct binary cost matrices for the Spider and BreaKHis experiments. It uses Option 3: systematic public-source review plus sensitivity validation. No external expert panel is assumed.

The goal is not to prove that any single ratio is the uniquely true social or clinical cost. The goal is to make the matrix public-evidence-derived, decision-theoretically mapped, uncertainty-bounded, and empirically stress-tested.

## Decision Contexts

All matrices use `C[true][pred]` and zero cost for correct predictions.

Spider:

```text
class 0 = black_widow
class 1 = false_widow

C_spider = [
  [0, R_spider],
  [1, 0]
]
```

The high-cost error is `black_widow -> false_widow`, because a medically significant Latrodectus / black widow image should not be reassured as a false widow.

BreaKHis:

```text
class 0 = benign
class 1 = malignant

C_breakhis = [
  [0, 1],
  [R_breakhis, 0]
]
```

The high-cost error is `malignant -> benign`, because malignant breast histopathology should not be falsely supported as benign.

## Search Scope

Search date: 2026-05-04. Search sources include PubMed/NCBI, Google Scholar-style web search, Crossref/DOI lookups, official clinical/public health pages, poison-center/toxicology sources, extension/public health guidance, and dataset-primary sources.

Search strings used or predeclared:

- `Latrodectus envenomation National Poison Data System review`
- `black widow spider toxicity clinical management antivenom hospitalization`
- `Steatoda grossa envenomation false widow bite clinical severity`
- `black widow false widow medical significance extension guidance`
- `BreaKHis breast cancer histopathological database DOI 10.1109/TBME.2015.2496264`
- `breast biopsy false negative missed cancer diagnostic delay pathology`
- `breast cancer screening false positives false negatives interval cancers NCI`
- `breast biopsy diagnostic concordance pathologists JAMA`
- `USPSTF harms breast cancer screening false positive biopsy anxiety`

## Inclusion Criteria

A source is included for derivation when it satisfies at least one criterion:

- It defines the dataset or label semantics.
- It quantifies or clinically describes the harm of the high-cost false-negative error.
- It quantifies or clinically describes the harm of the lower-cost false-positive error.
- It provides decision-context evidence about urgency, reversibility, intervention burden, or uncertainty.
- It is an official clinical/public health/dataset source that directly applies to the task.

## Exclusion Criteria

A source is excluded from numeric derivation when:

- It is non-authoritative background and a higher-quality source covers the same point.
- It is a model-performance benchmark without clinical, toxicology, or cost evidence.
- It is social-media or anecdotal material.
- It is too old or too indirect to influence the primary ratio, although it may be retained as historical context.
- It is about a different organism, disease, screening modality, or procedure and does not transfer cleanly to the dataset decision context.

Excluded high-relevance sources must remain in the extraction CSV with an exclusion reason.

## Evidence Hierarchy

Priority order:

1. Systematic reviews, national registries, poison-center datasets, clinical guidelines, official national clinical evidence summaries.
2. Peer-reviewed cohorts, diagnostic accuracy studies, case series, or pathology/treatment studies.
3. Dataset-primary papers and official benchmark documentation.
4. Case reports, official extension guidance, or public health guidance.
5. Non-peer-reviewed sources, used only for background and excluded from numeric ratio derivation.

Quality weights used for harm scoring:

| Evidence type | Weight |
|---|---:|
| Registry, systematic review, guideline, official national evidence summary | 1.0 |
| Cohort, diagnostic accuracy study, case series | 0.8 |
| Dataset-primary paper or clinical review | 0.6 |
| Case report or official extension/public guidance | 0.4 |
| Excluded background source | 0.0 |

## Extraction Schema

Each evidence CSV row records:

- `source_id`
- dataset
- included/excluded status
- exclusion reason, if any
- priority tier
- source type and year
- citation and URL/DOI
- population or decision context
- evidence role
- whether the source supports high-cost false-negative harm, low-cost false-positive harm, or both
- extracted evidence
- uncertainty or limits
- quality weight

The live extraction files are:

- [search_log.csv](../data/cost_matrix_evidence/search_log.csv)
- [spider_sources.csv](../data/cost_matrix_evidence/spider_sources.csv)
- [breakhis_sources.csv](../data/cost_matrix_evidence/breakhis_sources.csv)
- [harm_scoring.csv](../data/cost_matrix_evidence/harm_scoring.csv)

## Cost Derivation

Use binary cost normalization:

```text
C[correct] = 0
lower-cost error = 1
higher-cost error = R
```

For a binary positive class where false negatives are higher cost:

```text
predict positive if p(y=positive | x) >= t
R = C_FN / C_FP = (1 - t) / t
t = C_FP / (C_FN + C_FP)
```

Two estimators are used.

Threshold estimator:

- Infer a plausible action threshold from public clinical/toxicology evidence.
- Convert the implied threshold into a ratio with `R = (1 - t) / t`; keep whole-number ratios rather than rounding to pretty values.
- If evidence supports low-threshold precaution but not an exact value, use an interval and mark the midpoint as uncertain.

Harm-index estimator:

- Score each error type across five dimensions on `0-4`:
  - morbidity/mortality risk
  - urgency/time sensitivity
  - downstream intervention burden
  - psychological/social/economic burden
  - irreversibility/correctability
- Compute evidence-weighted harm scores.
- Compute `R_harm = H_high_cost_error / H_low_cost_error`.

Primary ratio rule:

- If `R_threshold` and `R_harm` agree within a factor of 2, choose the nearest whole-number value to their geometric mean.
- If they disagree by more than a factor of 2, choose the conservative middle value and widen the sensitivity interval.
- If public evidence cannot narrow beyond broad asymmetry, use a conservative integer primary ratio and report the full sensitivity interval.

## Sensitivity Validation

Required broad sensitivity ratios:

```text
R = {1, 2, 5, 10, 20}
```

`R=1` is a symmetric-cost negative control. `R={2,5,10,20}` is the evidence-supported stress-test interval. Existing Stop 4B outputs already cover this set for both datasets:

- `results/stop4b_cost_ratio_sensitivity/spider_convnext/`
- `results/stop4b_cost_ratio_sensitivity/breakhis_dinov3_convnext_lora/`

The validation package is:

- [results/binary_cost_matrix_validation_20260504](../results/binary_cost_matrix_validation_20260504/)

The primary whole-number matrices are evaluated in the new camera-ready binary suite:

- `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/`

Claims must follow the sensitivity result:

- If NICME remains best or competitive across the evidence-supported interval, robustness may be claimed.
- If NICME wins only at the primary ratio, state “best under the primary evidence-derived matrix.”
- If conclusions flip across plausible ratios, report the flip and use it as evidence that cost-matrix specification matters.

## Review Audit

Review flow as of 2026-05-04:

| Dataset | Records identified | Records screened | Extracted into CSV | Included for derivation | Excluded from derivation |
|---|---:|---:|---:|---:|---:|
| Spider | 20 | 12 | 8 | 6 | 2 |
| BreaKHis | 24 | 15 | 11 | 8 | 3 |

Duplicate independent extraction was not available in this implementation. This is recorded as a limitation; the extraction tables are structured so a second reviewer can repeat scoring and resolve conflicts before submission.
