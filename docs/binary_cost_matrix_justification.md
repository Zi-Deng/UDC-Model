# Binary Cost Matrix Justification

Updated: 2026-05-05

This document justifies the binary cost matrices used for Spider and BreaKHis. Because no validated public cost matrices exist for these datasets, the matrices are evidence-derived and broad-sensitivity-audited, not asserted as uniquely true values.

Protocol: [binary_cost_matrix_review_protocol.md](binary_cost_matrix_review_protocol.md)  
Extraction tables: [data/cost_matrix_evidence](../data/cost_matrix_evidence/)  
Broad sensitivity package: [results/binary_cost_matrix_validation_20260504](../results/binary_cost_matrix_validation_20260504/)  
Primary integer-matrix experiment: `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/`

## Final Matrices

Spider, `C[true][pred]`, class 0 `black_widow`, class 1 `false_widow`:

```text
C_spider = [
  [0, 8],
  [1, 0]
]
```

BreaKHis, `C[true][pred]`, class 0 `benign`, class 1 `malignant`:

```text
C_breakhis = [
  [0, 1],
  [7, 0]
]
```

The primary ratios are Spider `R=8` and BreaKHis `R=7`, with broad stress-test interval `{2, 5, 10, 20}` and symmetric-cost negative control `R=1`.

## Why Not Claim A True Exact Ratio?

The public evidence supports strong asymmetry, but it does not identify a single externally validated numeric ratio for either dataset. Therefore, the defensible paper claim is:

> We constructed transparent, public-evidence-derived cost matrices and stress-tested conclusions across a plausible cost-ratio interval.

The paper should not say:

> The true cost ratio is exactly 8:1 for Spider or exactly 7:1 for BreaKHis.

## Spider Matrix

Decision context: image classifier distinguishes medically significant Latrodectus / black widow from visually similar Steatoda / false widow. The high-cost error is `black_widow -> false_widow`.

Primary evidence:

- Monte et al. analyzed symptomatic Latrodectus exposures in the US National Poison Data System and is the registry review cited in later clinical summaries: <https://pubmed.ncbi.nlm.nih.gov/22116992/>.
- StatPearls describes Latrodectus as responsible for most clinically significant spider envenomation in the US, summarizes systemic latrodectism, and cites the NPDS review for severity proportions: <https://www.ncbi.nlm.nih.gov/sites/books/NBK499987/>.
- Merck Manual Professional describes widow-bite systemic symptoms, moderate/severe latrodectism, and treatment with opioids, benzodiazepines, antivenom, and toxicology consultation when indicated: <https://www.merckmanuals.com/professional/injuries-poisoning/bites-and-stings/spider-bites>.
- Graudins et al. shows Steatoda grossa can produce clinically meaningful steatodism, so false-widow errors are not costless; however the evidence is case-level and does not support Latrodectus-level public-health severity: <https://pubmed.ncbi.nlm.nih.gov/12175614/>.
- UC IPM and MSU Extension support the image-confusion context: false widows are visually related/confusable, while true widows carry the medically significant public-health warning.

Harm-index scoring:

| Error | Harm score | Interpretation |
|---|---:|---|
| `black_widow -> false_widow` | 12 | potential delay in medical advice for clinically significant Latrodectus envenomation |
| `false_widow -> black_widow` | 2 | precaution, anxiety, or unnecessary evaluation; Steatoda harm not zero |

`R_harm = 12 / 2 = 6`. The threshold estimator is `R_threshold = 10`, reflecting low-threshold precaution for a medically significant venomous spider identification. The geometric mean is `sqrt(10 * 6) = 7.75`, whose nearest whole-number ratio is `8`. This avoids rounding to a prettier ratio and keeps the matrix directly tied to the recorded estimator values.

Final spider primary matrix:

```text
[[0, 8],
 [1, 0]]
```

## BreaKHis Matrix

Decision context: image classifier distinguishes benign vs malignant breast histopathology images as decision support. The high-cost error is `malignant -> benign`.

Primary evidence:

- The BreaKHis primary paper and official dataset page define the benign/malignant histopathology task. The official page states that malignant tumor is cancer and can invade, metastasize, and cause death, while benign tumors are relatively localized: <https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>.
- NCI Breast Cancer Screening PDQ summarizes false positives, biopsies, overdiagnosis, false negatives, and the worse prognosis of interval cancers as secondary proxy evidence: <https://www.cancer.gov/types/breast/hp/breast-screening-pdq>.
- NCI’s interval-cancer summary reports that interval cancers can show more aggressive clinical features than screen-detected cancers: <https://www.cancer.gov/types/breast/research/interval-breast-cancer>.
- The USPSTF harms review is used to bound false-positive workup burden: <https://pubmed.ncbi.nlm.nih.gov/26756737/>.
- Elmore et al. provides pathology-specific evidence of diagnostic variability in breast biopsy interpretation: <https://pubmed.ncbi.nlm.nih.gov/25781441/>.
- Breast biopsy false-negative studies provide pathology-adjacent evidence that missed malignant cases can delay diagnosis, while multidisciplinary review can mitigate some harm.

Harm-index scoring:

| Error | Harm score | Interpretation |
|---|---:|---|
| `malignant -> benign` | 18 | missed malignancy can delay treatment, permit progression, and is time-sensitive |
| `benign -> malignant` | 3 | anxiety and additional workup; in decision support, should trigger review rather than autonomous treatment |

`R_harm = 18 / 3 = 6`. For BreaKHis, the threshold estimator is set to the exact 10% action-threshold ratio, `R_threshold = (1 - 0.10) / 0.10 = 9`, reflecting low-threshold action for possible cancer in a decision-support setting. The geometric mean is `sqrt(9 * 6) = 7.35`, whose nearest whole-number ratio is `7`.

Final BreaKHis primary matrix:

```text
[[0, 1],
 [7, 0]]
```

## Sensitivity Validation Read

Existing Stop 4B results already cover broad sensitivity ratios `R={1,2,5,10,20}`. The validation summary is [analysis/sensitivity_summary.md](../results/binary_cost_matrix_validation_20260504/analysis/sensitivity_summary.md). The exact whole-number primary matrices are evaluated by the new fixed-protocol binary camera-ready suite at `results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/`.

Headline audit:

- Spider: historical Stop 4B shows NICME is the best composite row at `R=1,2,5`; at `R=10`, `nicme_hybrid / calibrated_threshold` remains an all-seed-floor row but is ranked behind CE calibrated cost-min rows by normalized ATC. At `R=20`, CE calibrated threshold is stronger under the all-seed-floor composite.
- BreaKHis: historical Stop 4B shows NICME is the best composite row at `R=1,2,5,10`; at `R=20`, CE calibrated cost-min is stronger and no row satisfies all strict floors across all seeds.

Therefore:

- It is defensible to use the whole-number primary ratios `R=8` for Spider and `R=7` for BreaKHis.
- Final performance claims under these primary binary matrices should come from the 2026-05-05 binary camera-ready suite, not the older Stop 4B ratio grid alone.
- It is not defensible to claim NICME uniformly dominates all baselines over every plausible ratio.
- The ratio sensitivity itself supports the paper’s broader methodological argument: cost-matrix specification materially changes cost-sensitive conclusions.

## Paper Wording

Recommended wording:

> Because no validated public cost matrices exist for Spider or BreaKHis, we constructed evidence-derived binary matrices from public toxicology, clinical, dataset, and public-health sources. The primary ratios are whole-number derivations, `R=8` for Spider and `R=7` for BreaKHis, and we audit robustness with broad ratio sensitivity plus a symmetric-cost negative control.

Recommended caveat:

> The primary matrices are transparent and reproducible, but they are not asserted to be uniquely correct clinical utilities. They are decision-context matrices whose conclusions are audited by sensitivity analysis.
