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


def prepare_data() -> None:
    """Prepare NICME datasets such as BreaKHis and spider manifests."""
    from nicme.data_prep import main

    main()


def run_binary_experiments() -> None:
    """Create or execute binary experiment matrices."""
    from scripts.run_binary_experiments import main

    main()


def run_multiclass_experiments() -> None:
    """Create or execute multiclass experiment matrices."""
    from scripts.run_multiclass_experiments import main

    main()


def run_pmi10_no_cal_experiments() -> None:
    """Create or execute focused PMI-10 no-calibration experiment matrices."""
    from scripts.run_pmi10_no_cal_experiments import main

    main()


def check_multiclass_readiness() -> None:
    """Check MC0 readiness for multiclass datasets."""
    from scripts.check_multiclass_readiness import main

    main()


def check_dinov3_storage() -> None:
    """Check storage-safe Hugging Face access for official DINOv3 models."""
    from scripts.check_dinov3_storage_access import main

    main()


def experiment_stop() -> None:
    """Generate stop-gated experiment report and next-plan artifacts."""
    from nicme.experiment_stops import main

    main()
