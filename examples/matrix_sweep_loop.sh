#!/bin/bash
# Run multiple cost matrix sweeps sequentially.
#
# Each line runs run_cost_matrix_sweep.sh with a different sweep config.
# Add or remove lines as needed for your experiment.
#
# Usage:
#   micromamba activate ml
#   bash matrix_sweep_loop.sh

set -euo pipefail

bash run_cost_matrix_sweep.sh config/sweep_2class_bw_cost.json

echo "All sweeps complete."
