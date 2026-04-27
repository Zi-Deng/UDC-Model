# NICME Data Card

## Dataset

The current NICME workspace uses local spider image folders:

- `data/2_class_black_widows`
- `data/3_class_black_widows`

The paper-facing binary task uses:

- Class 0: `Latrodectus_hesperus` / Black Widow
- Class 1: `Steatoda_grossa` / False Widow

## Local Counts

| Split source | Class | Count |
|---|---|---:|
| `data/2_class_black_widows` | `Latrodectus_hesperus` | 1500 |
| `data/2_class_black_widows` | `Steatoda_grossa` | 1499 |

Parent training creates stratified train/validation/test splits at runtime using `random_state=42`.

## Preprocessing

Images are loaded with PIL, converted to RGB, transformed through `CustomImageProcessor`, and normalized with model-family-specific ImageNet-style statistics.

## Limitations To Document Before Submission

- Original data source and collection procedure.
- Image licensing and redistribution terms.
- Deduplication procedure.
- Species-label verification process.
- Known class, geography, image-quality, and acquisition biases.

