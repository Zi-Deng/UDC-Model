# Maintenance Notes

## Agent Behavior

Always honor the standing user instruction from `memory/README.md`: be correct, thoroughly investigated, current for 2026, non-hallucinatory, source-backed when needed, and unbiased.

For claims about this repository, prefer local source files, configs, generated metrics, and docs as sources. For claims about external libraries or APIs, verify against official documentation when the detail is version-sensitive.

## Coding Conventions

- Run parent scripts from repository root.
- Run playground scripts from `playground/cost_sensitive_loss_classification/`.
- Use `python`, not `python3`, inside the `ml` environment.
- Prefer Polars for new dataframe work.
- Prefer DuckDB for new SQL/persistent tabular outputs.
- Preserve existing patterns in `scripts/`: insert the repository root into `sys.path` before project imports.
- Keep changes scoped. This repo has large generated outputs and historical experiment files.

## Known Pitfalls

- `environment.yml` is the canonical environment source. Root `requirements.txt` is a convenience mirror, and `examples/requirements.txt` is retained for legacy examples.
- The bash sweep script is at `examples/run_cost_matrix_sweep.sh`.
- `.gitignore` ignores heavyweight artifact directories, but ignored paths can still contain historical tracked files. Do not assume ignored means untracked.
- `weights/` and `playground/` may contain important local artifacts despite being ignored by default.
- Parent cost-matrix row/column semantics vary by loss method. Check the exact loss implementation before interpreting a cost cell.
- Parent `scripts/cost_matrix_sweep.py` documents/presents resumability in surrounding docs, but current code does not persist the Optuna study. DuckDB export happens after trials complete.
- `utils/analyze_cost_matrix_results.py` and `utils/extract_metrics.py` use pandas/CSV patterns because they support the older bash sweep. New dataframe code should still prefer Polars unless compatibility requires otherwise.
- Parent `perform_comprehensive_evaluation()` writes under `results/<model_type>_test/`.
- Canonical NICME configs use JSON booleans. Legacy configs with string booleans are normalized for compatibility.
- `cost_matrix` is serialized to JSON text temporarily for `HfArgumentParser`, then converted back to a Python list.
- `cost_matrix_cross_entropy` will fail if `self.cost_matrix is None`; it indexes `self.cost_matrix` directly.
- The playground has its own import/run assumptions and no package `__init__.py` files in `models/` or `utils/`. Run scripts from the playground directory.
- Playground legacy metrics are hardcoded for five diabetic-retinopathy classes in some functions; `train_spiders.py` implements separate binary metrics.
- Playground optional `focal_loss` needs `kornia`; image preprocessing scripts mention `skimage` dependencies that are not in the parent `environment.yml`.

## Verification Strategy

For documentation-only changes:

```bash
find memory -maxdepth 1 -type f -name "*.md" -print
```

For Python changes:

```bash
ruff check <changed files>
python -m py_compile <changed files>
```

For training pipeline changes, use the smallest realistic smoke test. Full training and sweeps can be GPU/time expensive and should be run intentionally.

## Good Source Anchors

Use these files as source anchors for future explanations:

- `README.md`: high-level parent description and intended usage.
- `SWEEP_README.md`: sweep concepts, with caveats noted above.
- `pyproject.toml`: Python version and ruff config.
- `environment.yml`: dependency environment.
- `scripts/train.py`: actual parent training behavior.
- `utils/utils.py`: config schema, dataset preprocessing, metrics, output paths.
- `utils/loss_functions.py`: authoritative loss dispatch and cost-matrix semantics.
- `scripts/cost_matrix_sweep.py`: actual Python sweep behavior.
- `scripts/hpo_search.py`: actual HPO behavior.
- `scripts/compare_sweeps.py`: actual comparison behavior.
- `playground/cost_sensitive_loss_classification/train_spiders.py`: binary playground spider training behavior.
- `playground/cost_sensitive_loss_classification/cost_ratio_sweep.py`: playground spider sweep behavior.
