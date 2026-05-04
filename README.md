# NICME

NICME (Non-identical Cost Matrix Embeddings) is a PyTorch research codebase for cost-sensitive image classification. It studies how fixed, asymmetric cost matrices can shape model training and deployment decisions when different error types have different consequences.

Canonical repository root:

```bash
/mnt/storage/github/NICME
```

## Current Documentation

Start with [docs/README.md](docs/README.md). The current live summaries are:

- [Current status](docs/current_status.md)
- [Binary experiment summary](docs/binary_experiment_summary.md)
- [Multiclass experiment summary](docs/multiclass_experiment_summary.md)

Historical Markdown from `docs/`, `memory/`, and `results/` has been consolidated into [archive/markdown_consolidation_20260501/](archive/markdown_consolidation_20260501/). The archive index records where each moved file came from and what current document replaces it.

## Quick Start

Use the micromamba environment as the source of truth:

```bash
micromamba env update -n ml -f environment.yml
micromamba activate ml
```

Binary spider training:

```bash
python scripts/train.py --config config/nicme_2class_spiders.json
python scripts/train_reg.py --config config/nicme_2class_spiders_regularized.json
```

Package entry points mirror the scripts when the project is installed:

```bash
nicme-train --config config/nicme_2class_spiders.json
nicme-train-reg --config config/nicme_2class_spiders_regularized.json
nicme-sweep --config config/nicme_2class_spiders_regularized.json --row 0 --col 1 --values "1,2,3"
nicme-hpo --config config/nicme_2class_spiders.json
nicme-compare-sweeps
```

## Data And Experiments

Prepare binary data:

```bash
nicme-prepare-data --dataset spider --input-dir data/2_class_black_widows --output-dir data/prepared/spider
nicme-prepare-data --dataset breakhis --download --extract --raw-dir data/raw/breakhis --output-dir data/prepared/breakhis
```

Prepare or check multiclass data:

```bash
nicme-prepare-data --dataset eyepacs_dr --raw-dir data/raw/eyepacs --output-dir data/prepared/eyepacs_dr
nicme-prepare-data --dataset pmi_pills --raw-dir data/raw/pmi --output-dir data/prepared/pmi_pills
nicme-check-multiclass-readiness --datasets eyepacs_dr,pmi_pills --variants balanced,natural
```

Prepare and run the focused PMI-10 no-calibration track:

```bash
nicme-prepare-data --dataset pmi_pills_10_no_cal --input-dir data/raw/pmi --output-dir data/prepared/pmi_pills_10_no_cal
nicme-check-multiclass-readiness --datasets pmi_pills_10_no_cal --variants natural --output-dir results/pmi10_no_cal_20260501/readiness
nicme-run-pmi10-no-cal-experiments --phase smoke --execute
nicme-run-pmi10-no-cal-experiments --phase screen --execute
nicme-run-pmi10-no-cal-experiments --phase select
nicme-run-pmi10-no-cal-experiments --phase hpo-nicme --trials 100 --execute
nicme-run-pmi10-no-cal-experiments --phase hpo-ce --trials 30 --execute
nicme-run-pmi10-no-cal-experiments --phase final --execute
```

Run multiclass experiment stops:

```bash
nicme-run-multiclass-experiments --stop mc2 --datasets eyepacs_dr,pmi_pills --variants balanced --execute
```

Cost matrices use `C[true_label][predicted_label]` throughout the active code and docs.

## Repository Layout

```text
config/        Canonical and legacy JSON configs
model/         Custom ResNet and ConvNeXt implementations
nicme/         Package namespace, CLI entry points, data prep, costs, and modeling helpers
scripts/       Backward-compatible runnable entry points and experiment runners
utils/         Training, preprocessing, loss, metrics, and analysis utilities
docs/          Current live documentation and reference cards
archive/       Historical docs, migration notes, and consolidated Markdown
examples/      Legacy sweep and utility examples
results/       Local generated experiment outputs
checkpoints/   Local model checkpoints
data/          Local datasets and generated split audits
releases/      Curated anonymous and camera-ready release views
```

## Validation

```bash
micromamba run -n ml ruff check .
micromamba run -n ml pytest -q
micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py
micromamba run -n ml python scripts/validate_release_views.py
```

## License

No open-source license has been selected yet. See [docs/license_pending.md](docs/license_pending.md).
