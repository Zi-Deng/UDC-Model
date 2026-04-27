# Commands And Environment

## Environment

Use the micromamba environment named `ml`, defined in `environment.yml`.

```bash
micromamba activate ml
```

The project docs consistently say to run parent scripts from the repository root:

```bash
cd <repo-root>
```

Prefer `python` inside the environment. Avoid assuming `python3` is the intended interpreter.

## Install Dependencies

The primary environment source is `environment.yml`. A root `requirements.txt` is also present as a convenience mirror for pip-based environments, and `examples/requirements.txt` is retained for legacy examples.

Preferred:

```bash
micromamba env update -n ml -f environment.yml
```

Fallback/reference only:

```bash
pip install -r requirements.txt
```

## Common Parent Commands

Train with the canonical NICME local spider config:

```bash
nicme-train --config config/nicme_2class_spiders.json
```

Train the canonical NICME regularized model:

```bash
nicme-train-reg --config config/nicme_2class_spiders_regularized.json
```

Legacy wrappers remain available:

```bash
python scripts/train.py --config config/2classSpiders.json
python scripts/train_reg.py --config config/2classSpiders_reg.json
```

Run the Python/Optuna cost-matrix sweep:

```bash
nicme-sweep --config config/nicme_2class_spiders_regularized.json --row 0 --col 1 --values "1,2,3"
```

Legacy sweep commands:

```bash
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json
python scripts/cost_matrix_sweep.py --config config/2classSpiders.json --row 0 --col 1 --min 0 --max 10 --step 0.5
python scripts/cost_matrix_sweep.py --config config/2classSpiders_reg.json --row 0 --col 1 --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" --output-dir results/sweep_cost_0_1_reg
```

Run HPO:

```bash
nicme-hpo --config config/nicme_2class_spiders.json
```

Compare sweep outputs:

```bash
nicme-compare-sweeps
python scripts/compare_sweeps.py --output-dir results/sweep_comparison_1_0 --parent results/sweep_cost_1_0/sweep_results.csv --playground playground/cost_sensitive_loss_classification/results/sweep_cost_1_0/sweep_results.csv --hybrid results/sweep_cost_1_0_reg/sweep_results.csv
```

Run the bash sweep from `examples/`:

```bash
bash examples/run_cost_matrix_sweep.sh config/nicme_sweep_2class_bw_cost.json
```

Important: the sweep script is under `examples/run_cost_matrix_sweep.sh`, not at the repository root.

## Playground Commands

The playground project expects to be run from:

```bash
cd <repo-root>/playground/cost_sensitive_loss_classification
```

Spider training:

```bash
python train_spiders.py
python train_spiders.py --n_epochs 3 --batch_size 4 --save_path smoke_test
python train_spiders.py --cost_ratio 5.0 --lambd 20 --base_loss gls
```

Spider cost sweep:

```bash
python cost_ratio_sweep.py
python cost_ratio_sweep.py --values "1,5,10,50,100"
python cost_ratio_sweep.py --row 1 --col 0 --values "1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100" --output-dir results/sweep_cost_1_0
```

Legacy diabetic-retinopathy training:

```bash
python train.py --csv_train train.csv --model_name resnext50 --base_loss gls --lambd 10 --exp 2 --save_path my_experiment
```

## Linting And Formatting

Ruff config lives in `pyproject.toml`:

- line length: 120
- target: `py312`
- selected rules: `E`, `F`, `W`, `I`, `UP`, `B`
- ignored rule: `E501`
- formatting quote style: double
- first-party imports: `nicme`, `model`, `utils`, `scripts`

Commands:

```bash
ruff check .
ruff check --fix .
ruff format .
ruff format --check .
```

Be conservative with broad formatting because this repository includes tracked generated/cache files and a nested legacy playground. Prefer linting/formatting files you actually touch unless asked for a larger cleanup.

## Tests

There is no conventional parent test suite, no `pytest.ini`, no CI config, and no pre-commit config found. Files named `test_*.py` exist only in the playground and are legacy runnable scripts, not a coherent test suite.

For parent code changes, verification is usually one or more of:

- `python -m py_compile <changed .py files>`
- `ruff check <changed .py files>`
- A small training/sweep smoke command only if runtime and GPU availability make sense.
