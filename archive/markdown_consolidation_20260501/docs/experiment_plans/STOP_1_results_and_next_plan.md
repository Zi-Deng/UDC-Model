# STOP 1 Results And Next Plan

Generated: 2026-04-28 05:02:10

## Stop Status

This artifact archives the plan used for Stop 1, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

### Data Audit: spider_smoke

- Path: `data/prepared/stop1_smoke/spider_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'black_widow': 48, 'false_widow': 48} | 96 | 0 | {} | {} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_smoke

- Path: `data/prepared/stop1_smoke/breakhis_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'benign': 48, 'malignant': 48} | 41 | 0 | {'40': 28, '100': 27, '200': 15, '400': 26} | {'A': 13, 'DC': 38, 'F': 16, 'LC': 1, 'MC': 8, 'PC': 1, 'PT': 10, 'TA': 9} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 7 | 0 | {'40': 7, '100': 4, '200': 8, '400': 5} | {'DC': 10, 'F': 12, 'PC': 2} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 8, '100': 4, '200': 5, '400': 7} | {'A': 3, 'DC': 6, 'F': 1, 'LC': 4, 'PC': 2, 'TA': 8} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 6, '100': 6, '200': 5, '400': 7} | {'DC': 8, 'LC': 1, 'MC': 1, 'PC': 2, 'PT': 10, 'TA': 2} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43

- Path: `results/convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43 | argmax | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43 | calibrated_cost_min | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43

- Path: `results/convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 | 2.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41

- Path: `results/convnext_test/stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41 | argmax | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 |  |
| stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41 | calibrated_cost_min | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 | 1.85 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42

- Path: `results/convnext_test/stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42 | argmax | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 |  |
| stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42 | calibrated_cost_min | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 | 2.05 |
### Metrics Summary: stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-01

- Path: `results/timm_dinov3_convnext_lora_test/stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-01`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-01 | argmax | 0.07916666666666666 | 0.9166666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.53125 | 0.24087109913428625 |  |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-01 | calibrated_cost_min | 0.07916666666666666 | 0.9166666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.53125 | 0.24087109913428625 | 0.95 |
### Metrics Summary: stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-01

- Path: `results/timm_dinov3_convnext_lora_test/stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-01`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-01 | argmax | 0.04166666666666667 | 1.0 | 0.5833333333333334 | 0.5833333333333334 | 0.49579831932773105 | 0.2732064500451088 |  |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-01 | calibrated_cost_min | 0.04166666666666667 | 1.0 | 0.5833333333333334 | 0.5833333333333334 | 0.49579831932773105 | 0.2732064500451088 | 2.2 |
### Metrics Summary: stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-00

- Path: `results/timm_dinov3_convnext_lora_test/stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-00`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-00 | argmax | 0.27083333333333337 | 0.5 | 0.5416666666666666 | 0.5416666666666667 | 0.5408695652173913 | 0.24035538484652838 |  |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_ce_04-28_05-00 | calibrated_cost_min | 0.27083333333333337 | 0.5 | 0.5416666666666666 | 0.5416666666666667 | 0.5408695652173913 | 0.24035538484652838 | 1.75 |
### Metrics Summary: stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-00

- Path: `results/timm_dinov3_convnext_lora_test/stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-00`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-00 | argmax | 0.1625 | 0.75 | 0.5 | 0.5 | 0.4666666666666667 | 0.3301990255713463 |  |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid_04-28_05-00 | calibrated_cost_min | 0.1625 | 0.75 | 0.5 | 0.5 | 0.4666666666666667 | 0.3301990255713463 | 4.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44 | argmax | 0.5 | 0.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.3770720462004344 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44 | calibrated_cost_min | 0.5 | 0.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.3770720462004344 | 10.0 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.47293688356876373 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.47293688356876373 | 10.0 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45 | argmax | 0.19166666666666668 | 0.6666666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.5804195804195804 | 0.20724892367919284 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45 | calibrated_cost_min | 0.19166666666666668 | 0.6666666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.5804195804195804 | 0.20724892367919284 | 2.0 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.4951525330543518 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.4951525330543518 | 10.0 |
### Metrics Summary: stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_05-00

- Path: `results/timm_dinov3_vit_lora_test/stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_05-00`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_05-00 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0013684978087743493 |  |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_05-00 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0013684978087743493 | 0.5 |
### Metrics Summary: stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_05-00

- Path: `results/timm_dinov3_vit_lora_test/stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_05-00`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_05-00 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.005752779543399811 |  |
| stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_05-00 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.005752779543399811 | 0.7 |
### Metrics Summary: stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_04-59

- Path: `results/timm_dinov3_vit_lora_test/stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_04-59`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_04-59 | argmax | 0.29166666666666663 | 0.4166666666666667 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.20739307502905535 |  |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_ce_04-28_04-59 | calibrated_cost_min | 0.29166666666666663 | 0.4166666666666667 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.20739307502905535 | 0.5 |
### Metrics Summary: stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_04-59

- Path: `results/timm_dinov3_vit_lora_test/stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_04-59`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_04-59 | argmax | 0.06666666666666667 | 0.9166666666666666 | 0.7083333333333334 | 0.7083333333333333 | 0.6950998185117967 | 0.20751207570234942 |  |
| stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid_04-28_04-59 | calibrated_cost_min | 0.06666666666666667 | 0.9166666666666666 | 0.7083333333333334 | 0.7083333333333333 | 0.6950998185117967 | 0.20751207570234942 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0012571662664413452 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0012571662664413452 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.006031756599744198 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.006031756599744198 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45 | argmax | 0.25 | 0.5 | 0.75 | 0.75 | 0.7333333333333334 | 0.24902446319659555 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45 | calibrated_cost_min | 0.25 | 0.5 | 0.75 | 0.75 | 0.7333333333333334 | 0.24902446319659555 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45 | argmax | 0.029166666666666667 | 1.0 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.2074486861626308 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45 | calibrated_cost_min | 0.029166666666666667 | 1.0 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.2074486861626308 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43

- Path: `results/vit_test/stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43 | argmax | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.23925547550121942 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43 | calibrated_cost_min | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.23925547550121942 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43

- Path: `results/vit_test/stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.12294590721527736 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.12294590721527736 | 2.05 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42

- Path: `results/vit_test/stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42 | argmax | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.22991500049829483 |  |
| stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42 | calibrated_cost_min | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.22991500049829483 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42

- Path: `results/vit_test/stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729129433632 |  |
| stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729129433632 | 10.0 |
### Run Log: stop1_smoke_timm_full_tuning

- Path: `results/stop1_smoke_timm/run_log.json`
- Exists: `True`
- Total runs: `16`
- Successful runs: `16`
- Failed runs: `0`
- Total elapsed seconds: `230.91`
- Mean successful-run elapsed seconds: `14.43`

| # | dataset | model | method | status | rc | seconds | stderr log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | spider_balanced_smoke | convnext | ce | ok | 0 | 15.29 | results/stop1_smoke_timm/logs/01_stop1_smoke_timm_spider_balanced_smoke_convnext_ce.stderr.log |
| 2 | spider_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 14.91 | results/stop1_smoke_timm/logs/02_stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 3 | spider_balanced_smoke | vit | ce | ok | 0 | 12.84 | results/stop1_smoke_timm/logs/03_stop1_smoke_timm_spider_balanced_smoke_vit_ce.stderr.log |
| 4 | spider_balanced_smoke | vit | nicme_hybrid | ok | 0 | 12.22 | results/stop1_smoke_timm/logs/04_stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 5 | spider_balanced_smoke | timm_dinov3_vit | ce | ok | 0 | 12.97 | results/stop1_smoke_timm/logs/05_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce.rerun.stderr.log |
| 6 | spider_balanced_smoke | timm_dinov3_vit | nicme_hybrid | ok | 0 | 13.07 | results/stop1_smoke_timm/logs/06_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid.rerun.stderr.log |
| 7 | spider_balanced_smoke | timm_dinov3_convnext | ce | ok | 0 | 14.28 | results/stop1_smoke_timm/logs/07_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce.rerun.stderr.log |
| 8 | spider_balanced_smoke | timm_dinov3_convnext | nicme_hybrid | ok | 0 | 13.68 | results/stop1_smoke_timm/logs/08_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid.rerun.stderr.log |
| 9 | breakhis_balanced_smoke | convnext | ce | ok | 0 | 14.81 | results/stop1_smoke_timm/logs/09_stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce.stderr.log |
| 10 | breakhis_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 14.48 | results/stop1_smoke_timm/logs/10_stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 11 | breakhis_balanced_smoke | vit | ce | ok | 0 | 13.94 | results/stop1_smoke_timm/logs/11_stop1_smoke_timm_breakhis_balanced_smoke_vit_ce.stderr.log |
| 12 | breakhis_balanced_smoke | vit | nicme_hybrid | ok | 0 | 13.76 | results/stop1_smoke_timm/logs/12_stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 13 | breakhis_balanced_smoke | timm_dinov3_vit | ce | ok | 0 | 17.01 | results/stop1_smoke_timm/logs/13_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce.stderr.log |
| 14 | breakhis_balanced_smoke | timm_dinov3_vit | nicme_hybrid | ok | 0 | 14.13 | results/stop1_smoke_timm/logs/14_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid.stderr.log |
| 15 | breakhis_balanced_smoke | timm_dinov3_convnext | ce | ok | 0 | 17.87 | results/stop1_smoke_timm/logs/15_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce.stderr.log |
| 16 | breakhis_balanced_smoke | timm_dinov3_convnext | nicme_hybrid | ok | 0 | 15.65 | results/stop1_smoke_timm/logs/16_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid.stderr.log |
### Run Log: stop1_smoke_timm_lora

- Path: `results/stop1_smoke_timm_lora/run_log.json`
- Exists: `True`
- Total runs: `8`
- Successful runs: `8`
- Failed runs: `0`
- Total elapsed seconds: `109.31`
- Mean successful-run elapsed seconds: `13.66`

| # | dataset | model | method | status | rc | seconds | stderr log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | spider_balanced_smoke | timm_dinov3_vit_lora | ce | ok | 0 | 12.72 | results/stop1_smoke_timm_lora/logs/01_stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_ce.stderr.log |
| 2 | spider_balanced_smoke | timm_dinov3_vit_lora | nicme_hybrid | ok | 0 | 12.54 | results/stop1_smoke_timm_lora/logs/02_stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid.stderr.log |
| 3 | spider_balanced_smoke | timm_dinov3_convnext_lora | ce | ok | 0 | 13.47 | results/stop1_smoke_timm_lora/logs/03_stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_ce.stderr.log |
| 4 | spider_balanced_smoke | timm_dinov3_convnext_lora | nicme_hybrid | ok | 0 | 14.12 | results/stop1_smoke_timm_lora/logs/04_stop1_smoke_timm_lora_spider_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid.stderr.log |
| 5 | breakhis_balanced_smoke | timm_dinov3_vit_lora | ce | ok | 0 | 13.59 | results/stop1_smoke_timm_lora/logs/05_stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_ce.stderr.log |
| 6 | breakhis_balanced_smoke | timm_dinov3_vit_lora | nicme_hybrid | ok | 0 | 13.41 | results/stop1_smoke_timm_lora/logs/06_stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_vit_lora_nicme_hybrid.stderr.log |
| 7 | breakhis_balanced_smoke | timm_dinov3_convnext_lora | ce | ok | 0 | 14.27 | results/stop1_smoke_timm_lora/logs/07_stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_ce.stderr.log |
| 8 | breakhis_balanced_smoke | timm_dinov3_convnext_lora | nicme_hybrid | ok | 0 | 15.19 | results/stop1_smoke_timm_lora/logs/08_stop1_smoke_timm_lora_breakhis_balanced_smoke_timm_dinov3_convnext_lora_nicme_hybrid.stderr.log |

## What Changed And Why

Stop 1 was revised again to make LoRA part of the smoke gate for non-gated timm DINOv3 models, because timm DINOv3 is now intended to remain in final paper comparisons rather than serving only as a temporary fallback.

The final revised Stop 1 gate consists of 24 successful rows: the prior 16-row accessible-DINOv3 smoke matrix plus 8 new timm DINOv3 LoRA rows. All rows completed successfully with metric export and calibrated-cost-min reports. The 24-row gate covers 2 datasets, CE and NICME hybrid, ConvNeXt, ViT, timm DINOv3 ViT full fine-tuning, timm DINOv3 ConvNeXt full fine-tuning, timm DINOv3 ViT LoRA, and timm DINOv3 ConvNeXt LoRA.

The LoRA targets differ by architecture and must be reported clearly. timm DINOv3 ViT uses fused attention projection modules named `qkv`, so `timm_dinov3_vit_lora` targets `qkv` and saves `model.head`. timm DINOv3 ConvNeXt is not an attention model; `timm_dinov3_convnext_lora` targets block MLP linear layers `mlp.fc1,mlp.fc2` and saves `model.head`, leaving depthwise convolutions frozen. Treat this as ConvNeXt MLP-LoRA/PEFT, not attention LoRA.

Official Meta `facebook/dinov3-*` presets remain implemented and should be rerun/appended after gated Hugging Face approval. Until then, Stop 2 should use timm DINOv3 ViT LoRA as the primary accessible DINOv3-family backbone and label all interim results as timm DINOv3 results.

## Next Plan: Stop 2 Prototype Runs

- Use the non-gated timm DINOv3 ViT-S equivalent for the interim Stop 2 prototype runs while Hugging Face approval for the official Meta DINOv3 repositories is pending.
- Preserve the official `facebook/dinov3-*` presets and rerun/append official DINOv3 results after gated access is approved.
- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and timm DINOv3-ViT LoRA.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1.
- Epochs: 5 for spider, 8 for BreaKHis, early stopping enabled.
- Target runtime: under 1 hour total on RTX 5090.
- Continue only if at least one cost-sensitive path improves ATC or cared-class recall without unacceptable accuracy collapse.


## User Checkpoint

Are the results satisfactory enough to continue, revise, or pause?

## Archived Previous Plan

# STOP 1 Results And Next Plan

Generated: 2026-04-28 04:47:06

## Stop Status

This artifact archives the plan used for Stop 1, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

### Data Audit: spider_smoke

- Path: `data/prepared/stop1_smoke/spider_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'black_widow': 48, 'false_widow': 48} | 96 | 0 | {} | {} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_smoke

- Path: `data/prepared/stop1_smoke/breakhis_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'benign': 48, 'malignant': 48} | 41 | 0 | {'40': 28, '100': 27, '200': 15, '400': 26} | {'A': 13, 'DC': 38, 'F': 16, 'LC': 1, 'MC': 8, 'PC': 1, 'PT': 10, 'TA': 9} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 7 | 0 | {'40': 7, '100': 4, '200': 8, '400': 5} | {'DC': 10, 'F': 12, 'PC': 2} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 8, '100': 4, '200': 5, '400': 7} | {'A': 3, 'DC': 6, 'F': 1, 'LC': 4, 'PC': 2, 'TA': 8} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 6, '100': 6, '200': 5, '400': 7} | {'DC': 8, 'LC': 1, 'MC': 1, 'PC': 2, 'PT': 10, 'TA': 2} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43

- Path: `results/convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43 | argmax | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce_04-28_04-43 | calibrated_cost_min | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43

- Path: `results/convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-43 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 | 2.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41

- Path: `results/convnext_test/stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41 | argmax | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 |  |
| stop1_smoke_timm_spider_balanced_smoke_convnext_ce_04-28_04-41 | calibrated_cost_min | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 | 1.85 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42

- Path: `results/convnext_test/stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42 | argmax | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 |  |
| stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-42 | calibrated_cost_min | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 | 2.05 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44 | argmax | 0.5 | 0.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.3770720462004344 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-44 | calibrated_cost_min | 0.5 | 0.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.3770720462004344 | 10.0 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.47293688356876373 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.47293688356876373 | 10.0 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45 | argmax | 0.19166666666666668 | 0.6666666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.5804195804195804 | 0.20724892367919284 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce_04-28_04-45 | calibrated_cost_min | 0.19166666666666668 | 0.6666666666666666 | 0.5833333333333334 | 0.5833333333333333 | 0.5804195804195804 | 0.20724892367919284 | 2.0 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46

- Path: `results/timm_dinov3_convnext_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.4951525330543518 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid_04-28_04-46 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.4951525330543518 | 10.0 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0012571662664413452 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.0012571662664413452 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.006031756599744198 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-44 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.006031756599744198 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45 | argmax | 0.25 | 0.5 | 0.75 | 0.75 | 0.7333333333333334 | 0.24902446319659555 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce_04-28_04-45 | calibrated_cost_min | 0.25 | 0.5 | 0.75 | 0.75 | 0.7333333333333334 | 0.24902446319659555 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45

- Path: `results/timm_dinov3_vit_test/stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45 | argmax | 0.029166666666666667 | 1.0 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.2074486861626308 |  |
| stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid_04-28_04-45 | calibrated_cost_min | 0.029166666666666667 | 1.0 | 0.7083333333333334 | 0.7083333333333334 | 0.6812144212523719 | 0.2074486861626308 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43

- Path: `results/vit_test/stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43 | argmax | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.23925547550121942 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_ce_04-28_04-43 | calibrated_cost_min | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.23925547550121942 | 0.5 |
### Metrics Summary: stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43

- Path: `results/vit_test/stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.12294590721527736 |  |
| stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-43 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.12294590721527736 | 2.05 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42

- Path: `results/vit_test/stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42 | argmax | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.22991500049829483 |  |
| stop1_smoke_timm_spider_balanced_smoke_vit_ce_04-28_04-42 | calibrated_cost_min | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.22991500049829483 | 0.5 |
### Metrics Summary: stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42

- Path: `results/vit_test/stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729129433632 |  |
| stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-42 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729129433632 | 10.0 |
### Run Log: stop1_smoke_timm_revised

- Path: `results/stop1_smoke_timm/run_log.json`
- Exists: `True`
- Total runs: `16`
- Successful runs: `16`
- Failed runs: `0`
- Total elapsed seconds: `230.91`
- Mean successful-run elapsed seconds: `14.43`

| # | dataset | model | method | status | rc | seconds | stderr log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | spider_balanced_smoke | convnext | ce | ok | 0 | 15.29 | results/stop1_smoke_timm/logs/01_stop1_smoke_timm_spider_balanced_smoke_convnext_ce.stderr.log |
| 2 | spider_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 14.91 | results/stop1_smoke_timm/logs/02_stop1_smoke_timm_spider_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 3 | spider_balanced_smoke | vit | ce | ok | 0 | 12.84 | results/stop1_smoke_timm/logs/03_stop1_smoke_timm_spider_balanced_smoke_vit_ce.stderr.log |
| 4 | spider_balanced_smoke | vit | nicme_hybrid | ok | 0 | 12.22 | results/stop1_smoke_timm/logs/04_stop1_smoke_timm_spider_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 5 | spider_balanced_smoke | timm_dinov3_vit | ce | ok | 0 | 12.97 | results/stop1_smoke_timm/logs/05_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_ce.rerun.stderr.log |
| 6 | spider_balanced_smoke | timm_dinov3_vit | nicme_hybrid | ok | 0 | 13.07 | results/stop1_smoke_timm/logs/06_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_vit_nicme_hybrid.rerun.stderr.log |
| 7 | spider_balanced_smoke | timm_dinov3_convnext | ce | ok | 0 | 14.28 | results/stop1_smoke_timm/logs/07_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_ce.rerun.stderr.log |
| 8 | spider_balanced_smoke | timm_dinov3_convnext | nicme_hybrid | ok | 0 | 13.68 | results/stop1_smoke_timm/logs/08_stop1_smoke_timm_spider_balanced_smoke_timm_dinov3_convnext_nicme_hybrid.rerun.stderr.log |
| 9 | breakhis_balanced_smoke | convnext | ce | ok | 0 | 14.81 | results/stop1_smoke_timm/logs/09_stop1_smoke_timm_breakhis_balanced_smoke_convnext_ce.stderr.log |
| 10 | breakhis_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 14.48 | results/stop1_smoke_timm/logs/10_stop1_smoke_timm_breakhis_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 11 | breakhis_balanced_smoke | vit | ce | ok | 0 | 13.94 | results/stop1_smoke_timm/logs/11_stop1_smoke_timm_breakhis_balanced_smoke_vit_ce.stderr.log |
| 12 | breakhis_balanced_smoke | vit | nicme_hybrid | ok | 0 | 13.76 | results/stop1_smoke_timm/logs/12_stop1_smoke_timm_breakhis_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 13 | breakhis_balanced_smoke | timm_dinov3_vit | ce | ok | 0 | 17.01 | results/stop1_smoke_timm/logs/13_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_ce.stderr.log |
| 14 | breakhis_balanced_smoke | timm_dinov3_vit | nicme_hybrid | ok | 0 | 14.13 | results/stop1_smoke_timm/logs/14_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_vit_nicme_hybrid.stderr.log |
| 15 | breakhis_balanced_smoke | timm_dinov3_convnext | ce | ok | 0 | 17.87 | results/stop1_smoke_timm/logs/15_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_ce.stderr.log |
| 16 | breakhis_balanced_smoke | timm_dinov3_convnext | nicme_hybrid | ok | 0 | 15.65 | results/stop1_smoke_timm/logs/16_stop1_smoke_timm_breakhis_balanced_smoke_timm_dinov3_convnext_nicme_hybrid.stderr.log |

## What Changed And Why

Stop 1 was revised while Hugging Face approval for the official Meta DINOv3 repositories is pending. The revised accessible-DINOv3 smoke gate uses timm DINOv3 equivalents: `timm/vit_small_patch16_dinov3.lvd1689m` and `timm/convnext_tiny.dinov3_lvd1689m`. The official `facebook/dinov3-*` presets remain implemented and should be rerun/appended after gated access is approved, but they no longer block interim prototyping.

The revised Stop 1 matrix ran 16 planned rows: 2 datasets × 4 model families × 2 methods. All 16 completed successfully with metric export and calibrated-cost-min reports. Model families were ConvNeXt, ViT, timm DINOv3 ViT, and timm DINOv3 ConvNeXt. Methods were `ce` and `nicme_hybrid`. The first pass exposed one implementation gap: config validation rejected the new `timm` backend before model construction. The validator was updated to accept `timm`, and the four affected spider timm rows were rerun successfully; the first-pass log is preserved at `results/stop1_smoke_timm/run_log_initial_with_validator_failures.json`.

The timm DINOv3 fallback uses full fine-tuning for smoke/prototype coverage, not LoRA. This is acceptable as an interim engineering gate, but paper-facing claims should distinguish timm interim runs from the later official Meta DINOv3 + LoRA runs after Hugging Face approval. The smoke subsets intentionally remain tiny and balanced: 48 training images per class and 12 images per class for validation/calibration/test.

## Next Plan: Stop 2 Prototype Runs

- Use the non-gated timm DINOv3 ViT-S equivalent for the interim Stop 2 prototype runs while Hugging Face approval for the official Meta DINOv3 repositories is pending.
- Preserve the official `facebook/dinov3-*` presets and rerun/append official DINOv3 results after gated access is approved.
- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and timm DINOv3-ViT.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1.
- Epochs: 5 for spider, 8 for BreaKHis, early stopping enabled.
- Target runtime: under 1 hour total on RTX 5090.
- Continue only if at least one cost-sensitive path improves ATC or cared-class recall without unacceptable accuracy collapse.


## User Checkpoint

Are the results satisfactory enough to continue, revise, or pause?

## Archived Previous Plan

# STOP 1 Results And Next Plan

Generated: 2026-04-28 04:13:54

## Stop Status

This artifact archives the plan used for Stop 1, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

### Data Audit: spider_smoke

- Path: `data/prepared/stop1_smoke/spider_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'black_widow': 48, 'false_widow': 48} | 96 | 0 | {} | {} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'black_widow': 12, 'false_widow': 12} | 24 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_smoke

- Path: `data/prepared/stop1_smoke/breakhis_balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 96 | 57.14% | 70.00% | {'0': 48, '1': 48} | {'0': 0.5, '1': 0.5} | {'benign': 48, 'malignant': 48} | 41 | 0 | {'40': 28, '100': 27, '200': 15, '400': 26} | {'A': 13, 'DC': 38, 'F': 16, 'LC': 1, 'MC': 8, 'PC': 1, 'PT': 10, 'TA': 9} |
| validation | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 7 | 0 | {'40': 7, '100': 4, '200': 8, '400': 5} | {'DC': 10, 'F': 12, 'PC': 2} |
| calibration | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 8, '100': 4, '200': 5, '400': 7} | {'A': 3, 'DC': 6, 'F': 1, 'LC': 4, 'PC': 2, 'TA': 8} |
| test | 24 | 14.29% | 10.00% | {'0': 12, '1': 12} | {'0': 0.5, '1': 0.5} | {'benign': 12, 'malignant': 12} | 8 | 0 | {'40': 6, '100': 6, '200': 5, '400': 7} | {'DC': 8, 'LC': 1, 'MC': 1, 'PC': 2, 'PT': 10, 'TA': 2} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Metrics Summary: spider_convnext_ce

- Path: `results/convnext_test/stop1_smoke_spider_balanced_smoke_convnext_ce_04-28_04-07`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_spider_balanced_smoke_convnext_ce_04-28_04-07 | argmax | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 |  |
| stop1_smoke_spider_balanced_smoke_convnext_ce_04-28_04-07 | calibrated_cost_min | 0.33333333333333337 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 | 0.625 | 0.17804197221994403 | 1.85 |
### Metrics Summary: spider_convnext_nicme

- Path: `results/convnext_test/stop1_smoke_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-08`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-08 | argmax | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 |  |
| stop1_smoke_spider_balanced_smoke_convnext_nicme_hybrid_04-28_04-08 | calibrated_cost_min | 0.07083333333333333 | 0.9166666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6444444444444445 | 0.1377050851782163 | 2.05 |
### Metrics Summary: spider_vit_ce

- Path: `results/vit_test/stop1_smoke_spider_balanced_smoke_vit_ce_04-28_04-08`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_spider_balanced_smoke_vit_ce_04-28_04-08 | argmax | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.2299149930477143 |  |
| stop1_smoke_spider_balanced_smoke_vit_ce_04-28_04-08 | calibrated_cost_min | 0.2125 | 0.5833333333333334 | 0.75 | 0.75 | 0.7428571428571429 | 0.2299149930477143 | 0.5 |
### Metrics Summary: spider_vit_nicme

- Path: `results/vit_test/stop1_smoke_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-08`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-08 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729030092557 |  |
| stop1_smoke_spider_balanced_smoke_vit_nicme_hybrid_04-28_04-08 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.2118729030092557 | 10.0 |
### Metrics Summary: breakhis_convnext_ce

- Path: `results/convnext_test/stop1_smoke_breakhis_balanced_smoke_convnext_ce_04-28_04-09`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_breakhis_balanced_smoke_convnext_ce_04-28_04-09 | argmax | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 |  |
| stop1_smoke_breakhis_balanced_smoke_convnext_ce_04-28_04-09 | calibrated_cost_min | 0.15416666666666667 | 0.75 | 0.5833333333333334 | 0.5833333333333334 | 0.5714285714285714 | 0.09360351413488391 | 0.5 |
### Metrics Summary: breakhis_convnext_nicme

- Path: `results/convnext_test/stop1_smoke_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-09`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-09 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 |  |
| stop1_smoke_breakhis_balanced_smoke_convnext_nicme_hybrid_04-28_04-09 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.17041444778442383 | 2.5 |
### Metrics Summary: breakhis_vit_ce

- Path: `results/vit_test/stop1_smoke_breakhis_balanced_smoke_vit_ce_04-28_04-09`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_breakhis_balanced_smoke_vit_ce_04-28_04-09 | argmax | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.2392554531494776 |  |
| stop1_smoke_breakhis_balanced_smoke_vit_ce_04-28_04-09 | calibrated_cost_min | 0.020833333333333336 | 1.0 | 0.7916666666666666 | 0.7916666666666667 | 0.7822141560798548 | 0.2392554531494776 | 0.5 |
### Metrics Summary: breakhis_vit_nicme

- Path: `results/vit_test/stop1_smoke_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-09`
- Exists: `True`
- Metrics files: `1`

| run | mode | normalized ATC | target recall | accuracy | balanced accuracy | macro-F1 | ECE | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop1_smoke_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-09 | argmax | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.1229459096988042 |  |
| stop1_smoke_breakhis_balanced_smoke_vit_nicme_hybrid_04-28_04-09 | calibrated_cost_min | 0.05 | 1.0 | 0.5 | 0.5 | 0.3333333333333333 | 0.1229459096988042 | 2.05 |
### Run Log: stop1_smoke

- Path: `results/stop1_smoke/run_log.json`
- Exists: `True`
- Total runs: `12`
- Successful runs: `8`
- Failed runs: `4`
- Total elapsed seconds: `132.44`
- Mean successful-run elapsed seconds: `15.00`

| # | dataset | model | method | status | rc | seconds | stderr log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | spider_balanced_smoke | convnext | ce | ok | 0 | 18.38 | results/stop1_smoke/logs/01_stop1_smoke_spider_balanced_smoke_convnext_ce.stderr.log |
| 2 | spider_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 13.85 | results/stop1_smoke/logs/02_stop1_smoke_spider_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 3 | spider_balanced_smoke | vit | ce | ok | 0 | 16.95 | results/stop1_smoke/logs/03_stop1_smoke_spider_balanced_smoke_vit_ce.stderr.log |
| 4 | spider_balanced_smoke | vit | nicme_hybrid | ok | 0 | 13.07 | results/stop1_smoke/logs/04_stop1_smoke_spider_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 5 | spider_balanced_smoke | dinov3_vit | ce | failed | 1 | 3.07 | results/stop1_smoke/logs/05_stop1_smoke_spider_balanced_smoke_dinov3_vit_ce.stderr.log |
| 6 | spider_balanced_smoke | dinov3_vit | nicme_hybrid | failed | 1 | 3.15 | results/stop1_smoke/logs/06_stop1_smoke_spider_balanced_smoke_dinov3_vit_nicme_hybrid.stderr.log |
| 7 | breakhis_balanced_smoke | convnext | ce | ok | 0 | 15.29 | results/stop1_smoke/logs/07_stop1_smoke_breakhis_balanced_smoke_convnext_ce.stderr.log |
| 8 | breakhis_balanced_smoke | convnext | nicme_hybrid | ok | 0 | 15.21 | results/stop1_smoke/logs/08_stop1_smoke_breakhis_balanced_smoke_convnext_nicme_hybrid.stderr.log |
| 9 | breakhis_balanced_smoke | vit | ce | ok | 0 | 13.85 | results/stop1_smoke/logs/09_stop1_smoke_breakhis_balanced_smoke_vit_ce.stderr.log |
| 10 | breakhis_balanced_smoke | vit | nicme_hybrid | ok | 0 | 13.43 | results/stop1_smoke/logs/10_stop1_smoke_breakhis_balanced_smoke_vit_nicme_hybrid.stderr.log |
| 11 | breakhis_balanced_smoke | dinov3_vit | ce | failed | 1 | 3.05 | results/stop1_smoke/logs/11_stop1_smoke_breakhis_balanced_smoke_dinov3_vit_ce.stderr.log |
| 12 | breakhis_balanced_smoke | dinov3_vit | nicme_hybrid | failed | 1 | 3.15 | results/stop1_smoke/logs/12_stop1_smoke_breakhis_balanced_smoke_dinov3_vit_nicme_hybrid.stderr.log |

## What Changed And Why

Stop 1 smoke matrix ran 12 planned rows on the Linux RTX 5090: 8 completed and 4 failed. ConvNeXt and ViT completed CE and NICME hybrid on both balanced smoke datasets with metrics and calibrated-cost-min reports exported. The smoke subsets intentionally cap each split at 48 training images per class and 12 images per class for validation/calibration/test, so their displayed ratios are 96/24/24/24 rather than the full 70/10/10/10 Stop 0 ratios; this was done only to keep Stop 1 small while preserving split roles and class balance. All DINOv3-ViT rows failed before training because `facebook/dinov3-vits16-pretrain-lvd1689m` is a gated Hugging Face repository and this environment is not authenticated for it; this is a Stop 2 blocker until access/authentication is resolved or an explicitly documented fallback is approved. The training path uses `train.csv` for training, `validation.csv` for checkpoint selection/eval, `calibration.csv` only inside post-training temperature fitting for calibrated-cost-min, and `test.csv` only for final evaluation, so the smoke run preserved calibration/test separation. Successful ConvNeXt/ViT smoke rows took roughly 13-18 seconds each on the 5090 with 96 training images, while DINOv3 failed in about 3 seconds at model download/config access. The toy one-epoch metrics are useful only for pipeline validation, not efficacy claims. The Stop 2 plan is therefore revised to resolve DINOv3 access before prototype results are considered complete.

## Next Plan: Stop 2 Prototype Runs

- Resolve the DINOv3 access blocker before counting Stop 2 as complete: either authenticate this machine for gated Hugging Face DINOv3 checkpoints after access approval, or explicitly approve a documented accessible fallback for prototype timing only.
- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and DINOv3-ViT.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1.
- Epochs: 5 for spider, 8 for BreaKHis, early stopping enabled.
- Target runtime: under 1 hour total on RTX 5090.
- Continue only if at least one cost-sensitive path improves ATC or cared-class recall without unacceptable accuracy collapse.


## User Checkpoint

Are the results satisfactory enough to continue, revise, or pause?

## Archived Previous Plan

# STOP 0 Results And Next Plan

Generated: 2026-04-28 03:59:36

## Stop Status

This artifact archives the plan used for Stop 0, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

### Data Audit: spider_natural

- Path: `data/prepared/spider/splits/natural`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2099 | 69.99% | 70.00% | {'0': 1050, '1': 1049} | {'0': 0.5002382086707956, '1': 0.4997617913292044} | {'black_widow': 1050, 'false_widow': 1049} | 2099 | 0 | {} | {} |
| validation | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| calibration | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| test | 300 | 10.00% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_balanced

- Path: `data/prepared/spider/splits/balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2098 | 69.98% | 70.00% | {'0': 1049, '1': 1049} | {'0': 0.5, '1': 0.5} | {'black_widow': 1049, 'false_widow': 1049} | 2098 | 0 | {} | {} |
| validation | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| calibration | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |
| test | 300 | 10.01% | 10.00% | {'0': 150, '1': 150} | {'0': 0.5, '1': 0.5} | {'black_widow': 150, 'false_widow': 150} | 300 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_target_minority

- Path: `data/prepared/spider/splits/target_minority`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1398 | 69.97% | 70.00% | {'0': 349, '1': 1049} | {'0': 0.24964234620886983, '1': 0.7503576537911302} | {'black_widow': 349, 'false_widow': 1049} | 1398 | 0 | {} | {} |
| validation | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |
| calibration | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |
| test | 200 | 10.01% | 10.00% | {'0': 50, '1': 150} | {'0': 0.25, '1': 0.75} | {'black_widow': 50, 'false_widow': 150} | 200 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: spider_target_majority

- Path: `data/prepared/spider/splits/target_majority`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1400 | 70.00% | 70.00% | {'0': 1050, '1': 350} | {'0': 0.75, '1': 0.25} | {'black_widow': 1050, 'false_widow': 350} | 1400 | 0 | {} | {} |
| validation | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |
| calibration | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |
| test | 200 | 10.00% | 10.00% | {'0': 150, '1': 50} | {'0': 0.75, '1': 0.25} | {'black_widow': 150, 'false_widow': 50} | 200 | 0 | {} | {} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_natural

- Path: `data/prepared/breakhis/splits/natural`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5616 | 71.01% | 70.00% | {'0': 1724, '1': 3892} | {'0': 0.306980056980057, '1': 0.6930199430199431} | {'benign': 1724, 'malignant': 3892} | 56 | 0 | {'40': 1435, '100': 1504, '200': 1416, '400': 1261} | {'A': 383, 'DC': 2587, 'F': 667, 'LC': 351, 'MC': 737, 'PC': 217, 'PT': 235, 'TA': 439} |
| validation | 768 | 9.71% | 10.00% | {'0': 205, '1': 563} | {'0': 0.2669270833333333, '1': 0.7330729166666666} | {'benign': 205, 'malignant': 563} | 8 | 0 | {'40': 201, '100': 191, '200': 194, '400': 182} | {'DC': 489, 'F': 205, 'PC': 74} |
| calibration | 815 | 10.30% | 10.00% | {'0': 269, '1': 546} | {'0': 0.3300613496932515, '1': 0.6699386503067485} | {'benign': 269, 'malignant': 546} | 8 | 0 | {'40': 203, '100': 207, '200': 210, '400': 195} | {'A': 61, 'DC': 203, 'F': 142, 'LC': 201, 'PC': 142, 'TA': 66} |
| test | 710 | 8.98% | 10.00% | {'0': 282, '1': 428} | {'0': 0.3971830985915493, '1': 0.6028169014084507} | {'benign': 282, 'malignant': 428} | 9 | 0 | {'40': 156, '100': 179, '200': 193, '400': 182} | {'DC': 172, 'LC': 74, 'MC': 55, 'PC': 127, 'PT': 218, 'TA': 64} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`
### Data Audit: breakhis_balanced

- Path: `data/prepared/breakhis/splits/balanced`
- Exists: `True`

| split | rows | observed ratio | target ratio | label counts | label prevalence | label-name counts | unique patients | missing images | magnification counts | tumor-type counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3448 | 69.52% | 70.00% | {'0': 1724, '1': 1724} | {'0': 0.5, '1': 0.5} | {'benign': 1724, 'malignant': 1724} | 56 | 0 | {'40': 887, '100': 925, '200': 847, '400': 789} | {'A': 383, 'DC': 1134, 'F': 667, 'LC': 158, 'MC': 331, 'PC': 101, 'PT': 235, 'TA': 439} |
| validation | 410 | 8.27% | 10.00% | {'0': 205, '1': 205} | {'0': 0.5, '1': 0.5} | {'benign': 205, 'malignant': 205} | 8 | 0 | {'40': 109, '100': 104, '200': 108, '400': 89} | {'DC': 179, 'F': 205, 'PC': 26} |
| calibration | 538 | 10.85% | 10.00% | {'0': 269, '1': 269} | {'0': 0.5, '1': 0.5} | {'benign': 269, 'malignant': 269} | 8 | 0 | {'40': 134, '100': 129, '200': 144, '400': 131} | {'A': 61, 'DC': 97, 'F': 142, 'LC': 110, 'PC': 62, 'TA': 66} |
| test | 564 | 11.37% | 10.00% | {'0': 282, '1': 282} | {'0': 0.5, '1': 0.5} | {'benign': 282, 'malignant': 282} | 9 | 0 | {'40': 121, '100': 142, '200': 150, '400': 151} | {'DC': 110, 'LC': 47, 'MC': 33, 'PC': 92, 'PT': 218, 'TA': 64} |

- Patient overlap: `{'calibration__test': 0, 'calibration__train': 0, 'calibration__validation': 0, 'test__train': 0, 'test__validation': 0, 'train__validation': 0}`

## What Changed And Why

Stop 0 data prep completed. During audit, the first balanced split generation exposed a pandas groupby behavior that dropped the numeric label column from balanced CSVs; this was fixed in nicme/data_prep.py, covered by a regression test, and all spider/BreaKHis splits were regenerated. Spider produced natural, balanced, target-minority, and target-majority splits with no missing images, numeric label columns present, no split overlap, and ratios within tolerance. BreaKHis downloaded the official 4,273,561,758-byte archive, extracted 7,909 PNG images, and produced natural and balanced patient-level splits with no missing images, numeric label columns present, no patient overlap, and ratios within 5 percentage points of 70/10/10/10. BreaKHis manifest matches the expected image and label counts: 2,480 benign and 5,429 malignant. The parser produced 81 patient groups rather than the commonly cited 82 because one numeric patient code spans two malignant tumor types; the split keeps that code grouped together conservatively to avoid leakage.

## Next Plan: Stop 1 Smoke Runs

- Use only audited, approved data splits.
- Run balanced spider and a balanced BreaKHis subset.
- Models: ConvNeXt, ViT, DINOv3-ViT.
- Methods: `ce`, `nicme_hybrid`.
- Seeds: 1; epochs: 1.
- Confirm no crashes, no NaNs, valid metric export, and no calibration/test leakage.
- Record runtime per run and projected Tier 1 runtime.


## User Checkpoint

Are the results satisfactory enough to continue, revise, or pause?

## Archived Previous Plan

# NICME Stop-Gated Experimental Master Plan

Date: 2026-04-28

This is the archived master plan for NICME's staged binary experiments. Each completed stop must generate a new stop report before any later stop begins.

## Summary

Run NICME experiments as a staged, NeurIPS-quality evaluation of cared-class recall and ATC/normalized ATC under explicit user-defined cost matrices, while separating cost sensitivity from class imbalance.

At every stop, Codex must:

- Summarize results from the just-completed stage.
- Archive the plan that governed that stage into both `docs/` and `memory/`.
- Construct a revised plan for the next stage using observed results.
- Ask the user to inspect results and explicitly continue before any next-stage work.

Stop rule:

- Do not begin Stop `N+1` execution until Stop `N` has a saved summary/next-plan artifact and the user has approved continuing.

## Required Stop Artifacts

For each completed stop, create:

- `docs/experiment_plans/STOP_N_results_and_next_plan.md`
- `memory/experiment_plan_stop_N.md`

Each stop artifact must include:

- Observed results summary.
- What changed and why.
- Decision-complete next plan.
- User checkpoint with the exact question: `Are the results satisfactory enough to continue, revise, or pause?`
- Archived previous plan.

Use the command:

```bash
nicme-experiment-stop --stop N --archived-plan PATH_TO_PRIOR_PLAN --split-dir name=PATH --results-dir name=PATH --notes "What changed and why"
```

## Stop 0: Data Audit

- Prepare spider and BreaKHis splits.
- Report split sizes, class prevalence, patient overlap, BreaKHis magnification/tumor-type distributions, and image-load failures.
- Verify 70/10/10/10 train/validation/calibration/test and patient-disjoint BreaKHis splits.
- Archive this master plan, then write the Stop 1 plan based on observed data quality.
- User checkpoint: approve data validity before model training.

## Stop 1: Smoke Runs

- Datasets: balanced spider and balanced BreaKHis subset.
- Models: ConvNeXt, ViT, DINOv3-ViT.
- Methods: `ce`, `nicme_hybrid`.
- Seeds: 1; epochs: 1.
- Success: no crashes, no NaNs, metrics export correctly, calibration split is not used for training.
- Archive Stop 1 plan/results, then write Stop 2 plan with measured runtime and failure fixes.
- User checkpoint: approve moving to prototype runs.

## Stop 2: Prototype Runs

- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and DINOv3-ViT.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1; epochs: 5 spider, 8 BreaKHis, early stopping enabled.
- Target: under 1 hour total on RTX 5090.
- Success: at least one cost-sensitive method improves ATC or cared-class recall over CE without unacceptable accuracy collapse.
- Archive Stop 2 plan/results, then write Stop 3 plan with selected runtime, seed count, and any method/config revisions.
- User checkpoint: approve main paper-scale runs.

## Stop 3: Main Binary Paper Runs

- Datasets: balanced spider, spider target-minority, spider target-majority, balanced BreaKHis, natural BreaKHis.
- Model: DINOv3-ViT-S + LoRA.
- Methods: all six implemented methods.
- Seeds: 3 for all methods; extend to 5 seeds if Stop 2 runtime predicts completion under 20 hours per dataset, otherwise 5 seeds only for `ce`, `ce_calibrated_cost_min`, and `nicme_hybrid`.
- Epoch cap: 20; early stopping patience 3.
- Success: NICME hybrid reduces normalized ATC or improves cared-class recall on balanced datasets while maintaining configured accuracy floors.
- Archive Stop 3 plan/results, then write Stop 4 ablation plan based on strongest and weakest findings.
- User checkpoint: approve ablations or request reframing.

## Stop 4: Ablations And Robustness

- Backbone ablation: balanced spider and balanced BreaKHis; ConvNeXt, ViT, DINOv3-ConvNeXt, DINOv3-ViT; methods `ce_calibrated_cost_min` and `nicme_hybrid`; 3 seeds.
- Cost-ratio sensitivity: 1:1, 2:1, 5:1, 10:1, 20:1 on balanced spider and BreaKHis.
- Calibration ablation: argmax vs calibrated-cost-min for all trained models.
- Component ablation: CE, CS regularizer only, NICME logit only, NICME hybrid.
- Archive Stop 4 plan/results, then write final paper experiment summary and unresolved-risk memo.
- User checkpoint: decide whether to run extra experiments or freeze results.

## Reporting And Acceptance Criteria

Every stop report must include, when available:

- ATC, normalized ATC, CRR.
- Cared-class recall, FNR, precision, FPR.
- Accuracy, balanced accuracy, macro-F1.
- AUROC/AUPRC for binary tasks.
- NLL, Brier, ECE, fitted temperature.
- Confusion matrix.
- Class prevalence for every split and result row.
- Runtime per run and projected runtime for the next stop.

Paper-level acceptance:

- Balanced-dataset gains are required before claiming cost sensitivity is decoupled from class imbalance.
- Natural/imbalanced gains alone are not enough for the central claim.
- `menon_logit_adjusted` must be interpreted as an imbalance baseline.
- `ce_calibrated_cost_min` must be interpreted as the post-hoc minimum expected cost baseline.
- If NICME hybrid fails to beat post-hoc cost-min on ATC, report that honestly and reframe the contribution around when training-time costs help.

Statistical reporting:

- Mean, standard deviation, and 95% CI across seeds.
- Paired comparisons for `nicme_hybrid` vs `ce` and `nicme_hybrid` vs `ce_calibrated_cost_min`.
- With only 3 seeds, mark significance claims as exploratory.

## Assumptions And Sources

Defaults:

- Hardware: one Linux RTX 5090.
- Primary backbone: `facebook/dinov3-vits16-pretrain-lvd1689m` with LoRA.
- Cost convention: `C[true_label][predicted_label]`.
- Spider target: `black_widow`; BreaKHis target: `malignant`.
- EyePACS/QWK and multiclass experiments stay out of scope.

Source anchors:

- Temperature scaling: Guo et al., ICML 2017, https://proceedings.mlr.press/v70/guo17a.html
- MetaCost / cost-sensitive wrapper baseline: Domingos 1999, https://aiweb.cs.washington.edu/ai/metacost.html
- Logit adjustment baseline: Menon et al., ICLR 2021, https://openreview.net/forum?id=37nvvqkCo5
- CSADA and ViT nuance: https://pubsonline.informs.org/doi/10.1287/ijds.2022.0033
- DINOv3: https://huggingface.co/docs/transformers/model_doc/dinov3
- PEFT LoRA image classification: https://huggingface.co/docs/peft/main/en/task_guides/image_classification_lora
