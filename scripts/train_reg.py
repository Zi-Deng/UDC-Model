"""Cost-sensitive regularized training script.

Combines logit adjustment with cost-sensitive regularization for robust
cost-aware classification. Uses the HuggingFace Trainer pipeline with
the ``logit_adjustment_regularized`` loss function.

Usage:
    micromamba activate ml
    python scripts/train_reg.py --config config/2classSpiders_reg.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from train import main as train_main
from transformers import set_seed

from utils.utils import parse_HF_args


def main():
    set_seed(42)
    logger = logging.getLogger(__name__)
    args = parse_HF_args()

    # Validate regularization config
    if args.loss_function != "logit_adjustment_regularized":
        logger.warning(
            "train_reg.py is designed for 'logit_adjustment_regularized' loss, got '%s'. Proceeding anyway.",
            args.loss_function,
        )

    if args.cost_matrix is None:
        raise ValueError(
            "logit_adjustment_regularized requires a cost_matrix in config. "
            'Add e.g. "cost_matrix": [[0.0, 1.0], [0.0, 0.0]] to your JSON.'
        )

    cs_lambda = getattr(args, "cs_lambda", 0.0)
    cs_warmup = getattr(args, "cs_warmup_epochs", 0)

    print("=" * 50)
    print("Cost-Sensitive Regularized Training")
    print("=" * 50)
    print(f"Loss function:     {args.loss_function}")
    print(f"Cost matrix:       {args.cost_matrix}")
    print(f"CS lambda:         {cs_lambda}")
    print(f"CS warmup epochs:  {cs_warmup}")
    print(f"Num epochs:        {args.num_train_epochs}")
    print(f"Early stop:        {args.early_stopping_patience}")
    print(f"Frozen stages:     {args.num_frozen_stages}")
    print(f"Weight decay:      {args.weight_decay}")
    print("=" * 50)

    results_dir = train_main(args)
    print(f"\nResults saved to: {results_dir}")
    return results_dir


if __name__ == "__main__":
    main()
