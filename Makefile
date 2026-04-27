.PHONY: setup lint compile validate-release smoke-help

setup:
	micromamba env update -n ml -f environment.yml
	micromamba run -n ml pip install -e .

lint:
	micromamba run -n ml ruff check .

compile:
	micromamba run -n ml python -m py_compile nicme/*.py scripts/*.py utils/*.py model/*.py

validate-release:
	micromamba run -n ml python scripts/validate_release_views.py

smoke-help:
	micromamba run -n ml nicme-train --help
	micromamba run -n ml nicme-train-reg --help
	micromamba run -n ml nicme-sweep --help
	micromamba run -n ml nicme-hpo --help
	micromamba run -n ml nicme-compare-sweeps --help
