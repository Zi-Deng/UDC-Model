"""Command-line entry points for NICME.

The historical ``scripts/*.py`` files remain supported. These wrappers provide
stable package-level commands for release and paper reproduction workflows.
"""

from __future__ import annotations

from transformers import set_seed
from transformers.utils import logging


def train() -> str:
    """Run the standard NICME training pipeline."""
    from scripts.train import main as train_main
    from utils.utils import parse_HF_args

    set_seed(42)
    logging.get_logger(__name__)
    return train_main(parse_HF_args())


def train_reg() -> str:
    """Run the regularized NICME training pipeline."""
    from scripts.train_reg import main

    return main()


def sweep() -> None:
    """Run a cost-matrix sweep."""
    from scripts.cost_matrix_sweep import main

    main()


def hpo() -> None:
    """Run hyperparameter optimization."""
    from scripts.hpo_search import main

    main()


def compare_sweeps() -> None:
    """Compare parent, playground, and hybrid sweep outputs."""
    from scripts.compare_sweeps import main

    main()

