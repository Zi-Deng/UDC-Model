import csv
from pathlib import Path

import pytest
import torch

from scripts import export_pmi20_margin_distributions as margins


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_compute_pair_margins_filters_to_configured_critical_pairs():
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.0],
            [0.5, 2.5, 0.0],
            [1.0, 0.0, 4.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    pairs = [
        (0, 1, 10.0, "class_0", "class_1"),
        (1, 0, 8.0, "class_1", "class_0"),
    ]

    rows = margins.compute_pair_margins(logits, labels, pairs, start_index=5)

    assert len(rows) == 2
    assert rows[0]["sample_index"] == 5
    assert rows[0]["true_id"] == 0
    assert rows[0]["pred_id"] == 1
    assert rows[0]["margin"] == pytest.approx(2.0)
    assert rows[1]["sample_index"] == 6
    assert rows[1]["true_id"] == 1
    assert rows[1]["pred_id"] == 0
    assert rows[1]["margin"] == pytest.approx(2.0)


def test_selected_metric_rows_requires_all_methods_for_each_seed(tmp_path):
    pmi20 = tmp_path / "pmi20_per_seed.csv"
    alpha = tmp_path / "alpha_per_seed.csv"
    fields = ["seed", "method", "config_path", "checkpoint_path", "metric_path"]
    rows = []
    for seed in (42, 43):
        rows.extend(
            [
                {"seed": seed, "method": "ce_anchor_pmi20", "config_path": "a", "checkpoint_path": "b", "metric_path": "c"},
                {
                    "seed": seed,
                    "method": "cost_sensitive_regularized_ce",
                    "config_path": "a",
                    "checkpoint_path": "b",
                    "metric_path": "c",
                },
                {"seed": seed, "method": "csada", "config_path": "a", "checkpoint_path": "b", "metric_path": "c"},
            ]
        )
    write_csv(pmi20, rows, fields)
    write_csv(
        alpha,
        [
            {
                "seed": seed,
                "method": "nicme_extra_a0p5_l0p1_pmi20",
                "config_path": "a",
                "checkpoint_path": "b",
                "metric_path": "c",
            }
            for seed in (42, 43)
        ],
        fields,
    )

    selected = margins.selected_metric_rows(pmi20, alpha, (42, 43))

    assert len(selected) == 8
    assert {(int(row["seed"]), row["method"]) for row in selected} == {
        (seed, method) for seed in (42, 43) for method in margins.SOURCE_METHODS
    }


def test_selected_metric_rows_raises_when_a_required_method_is_missing(tmp_path):
    pmi20 = tmp_path / "pmi20_per_seed.csv"
    alpha = tmp_path / "alpha_per_seed.csv"
    fields = ["seed", "method", "config_path", "checkpoint_path", "metric_path"]
    write_csv(
        pmi20,
        [
            {"seed": 42, "method": "ce_anchor_pmi20", "config_path": "a", "checkpoint_path": "b", "metric_path": "c"},
            {"seed": 42, "method": "cost_sensitive_regularized_ce", "config_path": "a", "checkpoint_path": "b", "metric_path": "c"},
        ],
        fields,
    )
    write_csv(
        alpha,
        [
            {
                "seed": 42,
                "method": "nicme_extra_a0p5_l0p1_pmi20",
                "config_path": "a",
                "checkpoint_path": "b",
                "metric_path": "c",
            }
        ],
        fields,
    )

    with pytest.raises(RuntimeError, match="Missing metric rows"):
        margins.selected_metric_rows(pmi20, alpha, (42,))


def test_resolve_checkpoint_path_falls_back_to_config_output_dir():
    row = {"method": "ce_anchor_pmi20"}
    config = {"checkpoint_root": "checkpoints", "output_dir": "pmi20_run_ce_seed42"}

    assert margins.resolve_checkpoint_path(row, config) == "checkpoints/pmi20_run_ce_seed42"


def test_resolve_checkpoint_path_prefers_explicit_row_value():
    row = {"method": "ce_anchor_pmi20", "checkpoint_path": "explicit/checkpoint"}
    config = {"checkpoint_root": "checkpoints", "output_dir": "pmi20_run_ce_seed42"}

    assert margins.resolve_checkpoint_path(row, config) == "explicit/checkpoint"
