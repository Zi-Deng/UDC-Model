# NICME Paper Figures And Tables

This folder contains camera-ready table and figure assets generated from completed NICME result ledgers.

## Captions

**Table: Baselines and decision modes.** Methods differ in where cost information enters the pipeline: training loss, inference decision rule, class-frequency correction, adversarial inner loop, or output parameterization.

**Table: Main PMI-20 cost-sensitive comparison.** Mean +/- sample standard deviation over three seeds on the balanced PMI-20 split; rankings follow target-min recall, normalized ATC, balanced accuracy, and macro-F1.

**Table: NICME component ablation.** CE, expected-cost regularization, pairwise cost margins, and full NICME are compared under the same ConvNeXt-base LR 5e-5 protocol.

**Table: Binary cost-sensitive results.** Spider and BreaKHis comparisons use evidence-derived integer matrices, reporting both recall-first cost-sensitive metrics and clean balanced accuracy.

**Table: Computational and implementation cost.** Wall-time and storage estimates are taken from completed run ledgers and checkpoint directories; CSADA is the only method with an adversarial inner loop.

**Figure: PMI-20 recall-cost tradeoff.** Each point is a method; lower normalized ATC and higher target recall are preferred, and point size reflects total critical-pair errors.

**Figure: PMI-20 alpha-lambda sensitivity.** Six fixed NICME candidates show how expected cost and cared-class recall change with alpha/lambda.

**Figure: Critical-pair logit margin distributions.** ECDFs of raw margins `f_y(x)-f_k(x)` on critical pairs; right-shifted curves indicate larger clean margins against high-cost confusions.

## Provenance

- See `data/result_inventory.csv` for source paths.
- Metrics are repository SOTA/baseline comparisons, not external global SOTA claims.
