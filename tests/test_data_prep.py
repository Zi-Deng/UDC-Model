from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from nicme.data_prep import (  # noqa: E402
    _group_split,
    _stratified_split,
    balance_split,
    controlled_prevalence_split,
    parse_breakhis_filename,
)


def test_parse_breakhis_filename_extracts_label_patient_and_magnification():
    parsed = parse_breakhis_filename("SOB_M_DC-14-10926-100-001.png")

    assert parsed["label_name"] == "malignant"
    assert parsed["patient_id"] == "10926"
    assert parsed["tumor_type"] == "DC"
    assert parsed["magnification"] == "100"


def test_parse_breakhis_filename_rejects_unknown_names():
    with pytest.raises(ValueError):
        parse_breakhis_filename(Path("not_breakhis.png"))


def test_controlled_prevalence_split_downsamples_to_target_prevalence():
    df = pandas.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(100)],
            "label": [0] * 50 + [1] * 50,
        }
    )

    minority = controlled_prevalence_split(df, target_label=0, target_prevalence=0.25, seed=7)
    majority = controlled_prevalence_split(df, target_label=0, target_prevalence=0.75, seed=7)

    assert abs((minority["label"] == 0).mean() - 0.25) <= 0.03
    assert abs((majority["label"] == 0).mean() - 0.75) <= 0.03


def test_balance_split_preserves_numeric_label_column():
    df = pandas.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(7)],
            "label": [0, 0, 0, 0, 1, 1, 1],
            "label_name": ["a"] * 4 + ["b"] * 3,
        }
    )

    balanced = balance_split(df, seed=3)

    assert "label" in balanced.columns
    assert balanced["label"].value_counts().to_dict() == {0: 3, 1: 3}


def test_prepared_splits_are_70_10_10_10_and_patient_disjoint():
    df = pandas.DataFrame(
        {
            "image_path": [f"img_{idx}.png" for idx in range(100)],
            "label": [0] * 50 + [1] * 50,
            "patient_id": [f"patient_{idx:03d}" for idx in range(100)],
        }
    )

    stratified = _stratified_split(df, seed=11)
    grouped = _group_split(df, seed=11)

    assert {name: len(split) for name, split in stratified.items()} == {
        "train": 70,
        "validation": 10,
        "calibration": 10,
        "test": 10,
    }
    assert {name: len(split) for name, split in grouped.items()} == {
        "train": 70,
        "validation": 10,
        "calibration": 10,
        "test": 10,
    }
    patient_sets = [set(split["patient_id"]) for split in grouped.values()]
    assert sum(len(patients) for patients in patient_sets) == len(set().union(*patient_sets))
