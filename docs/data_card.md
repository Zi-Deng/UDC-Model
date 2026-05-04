# NICME Data Card

Updated: 2026-05-04

## Dataset Families

The current NICME workspace uses local image data for:

- Binary spider classification.
- BreaKHis binary pathology experiments.
- EyePACS diabetic retinopathy multiclass experiments.
- PMI Pills multiclass experiments.
- Focused PMI-10 no-calibration pill experiments.

## Binary Spider Data

Local folders:

- `data/2_class_black_widows`
- `data/3_class_black_widows`

Paper-facing binary spider classes:

- Class 0: `Latrodectus_hesperus` / Black Widow
- Class 1: `Steatoda_grossa` / False Widow

| Split source | Class | Count |
|---|---|---:|
| `data/2_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/2_class_black_widows` | `Steatoda_grossa` | 1499 |
| `data/3_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/3_class_black_widows` | `Steatoda_grossa` | 1500 |
| `data/3_class_black_widows` | `Steatoda_nobilis` | 1500 |

Parent binary training creates stratified train/validation/test splits at runtime using `random_state=42`.

## Current PMI-10 Data

The completed PMI-10 HPO uses:

- Split root: `data/prepared/pmi_pills_10_no_cal/splits/balanced`
- Split counts: train `970`, validation `310`, test `320`
- Number of classes: `10`
- Cared classes: `50111-0434`, `53489-0156`, `53746-0544`, `68382-0227`
- Cost matrix hash: `1160cbf20c4fac24dc7bb84cbbb229a16545259d585ae1ef1c066e0007082e08`

## Current Multiclass Data

Current balanced paper-run splits:

| Dataset | Split root | Counts | Cared classes |
|---|---|---|---|
| EyePACS DR | `data/prepared/eyepacs_dr/splits/balanced` | train 2485, validation 335, calibration 350, test 370 | `DR4` |
| PMI Pills | `data/prepared/pmi_pills/splits/balanced` | train 1940, validation 320, calibration 300, test 640 | `50111-0434`, `53489-0156`, `53746-0544`, `68382-0227` |

Both balanced audits reported zero missing images and zero patient overlap across checked split pairs.

## Preprocessing

Images are loaded with PIL, converted to RGB, transformed through the configured image processor, and normalized with model-family-specific ImageNet-style statistics where applicable. PMI-10 HPO uses 224px inputs with random resized crop and random quarter-turn augmentation.

## Limitations To Document Before Submission

- Original data source and collection procedure for each dataset family.
- Image licensing and redistribution terms.
- Deduplication procedure.
- Species, disease-grade, and pill-label verification processes.
- Known class, geography, image-quality, and acquisition biases.
- Patient-level or source-level leakage controls for each split.
