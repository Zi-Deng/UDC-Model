"""Utility package for UDC-Model.

Submodules:
    utils           – Core training helpers: arg parsing, dataset loading, metrics, evaluation.
    loss_functions  – Loss function dispatch (CE, cost-matrix CE, seesaw, logit adjustment).
    image_processor – CustomImageProcessor replacing HuggingFace AutoImageProcessor.
    extract_metrics – CLI tool for extracting metrics from results (used by bash sweep).
    analyze_cost_matrix_results – Graph generation for bash sweep analysis.
"""
