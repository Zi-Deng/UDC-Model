import argparse
from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from nicme.data_prep import (  # noqa: E402
    _group_split,
    _stratified_split,
    balance_split,
    build_eyepacs_manifest,
    build_pmi_split_frames,
    controlled_prevalence_split,
    parse_breakhis_filename,
    prepare_pmi_pills,
    prepare_pmi_pills_10_no_cal,
)
from nicme.dataset_profiles import PMI_PILLS_10_NO_CAL_PROFILE, PMI_PILLS_PROFILE  # noqa: E402


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


def test_build_eyepacs_manifest_extracts_patient_id_and_dr_labels(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for image_id in ("10_left", "10_right", "11_left"):
        (train_dir / f"{image_id}.jpeg").write_bytes(b"fake")
    pandas.DataFrame(
        {
            "image": ["10_left", "10_right", "11_left"],
            "level": [0, 4, 2],
        }
    ).to_csv(tmp_path / "trainLabels.csv", index=False)

    manifest = build_eyepacs_manifest(tmp_path, tmp_path / "manifest.csv")

    assert manifest["label_name"].tolist() == ["DR0", "DR4", "DR2"]
    assert manifest["patient_id"].tolist() == ["10", "10", "11"]
    assert manifest["eye"].tolist() == ["left", "right", "left"]


def test_build_pmi_split_frames_validates_official_classes(tmp_path):
    for split in ("Train", "Valid", "Test"):
        for class_name in PMI_PILLS_PROFILE.class_names:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            (class_dir / f"{split}_{class_name}.jpg").write_bytes(b"fake")

    frames = build_pmi_split_frames(tmp_path)

    assert set(frames) == {"train", "valid", "test"}
    assert len(frames["train"]) == PMI_PILLS_PROFILE.num_classes
    assert set(frames["train"]["label_name"]) == set(PMI_PILLS_PROFILE.class_names)


def test_prepare_pmi_pills_splits_valid_into_validation_and_calibration(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    for split in ("Train", "Valid", "Test"):
        for class_name in PMI_PILLS_PROFILE.class_names:
            class_dir = input_dir / split / class_name
            class_dir.mkdir(parents=True)
            for idx in range(2):
                (class_dir / f"{split}_{class_name}_{idx}.jpg").write_bytes(b"fake")

    args = argparse.Namespace(input_dir=str(input_dir), output_dir=str(output_dir), seed=3, extract=False)
    prepare_pmi_pills(args)

    validation = pandas.read_csv(output_dir / "splits" / "natural" / "validation.csv")
    calibration = pandas.read_csv(output_dir / "splits" / "natural" / "calibration.csv")
    audit = output_dir / "audit_natural.json"

    assert len(validation) == PMI_PILLS_PROFILE.num_classes
    assert len(calibration) == PMI_PILLS_PROFILE.num_classes
    assert audit.exists()


def test_prepare_pmi_pills_10_no_cal_uses_official_train_valid_test_without_calibration(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "prepared"
    for split in ("Train", "Valid", "Test"):
        for class_name in PMI_PILLS_PROFILE.class_names:
            class_dir = input_dir / "NLM20" / split / class_name
            class_dir.mkdir(parents=True)
            for idx in range(2):
                (class_dir / f"{split}_{class_name}_{idx}.jpg").write_bytes(b"fake")

    args = argparse.Namespace(input_dir=str(input_dir), output_dir=str(output_dir), seed=3, extract=False)
    prepare_pmi_pills_10_no_cal(args)

    split_dir = output_dir / "splits" / "natural"
    balanced_dir = output_dir / "splits" / "balanced"
    train = pandas.read_csv(split_dir / "train.csv")
    validation = pandas.read_csv(split_dir / "validation.csv")
    test = pandas.read_csv(split_dir / "test.csv")
    balanced_train = pandas.read_csv(balanced_dir / "train.csv")
    balanced_validation = pandas.read_csv(balanced_dir / "validation.csv")
    balanced_test = pandas.read_csv(balanced_dir / "test.csv")

    assert not (split_dir / "calibration.csv").exists()
    assert not (balanced_dir / "calibration.csv").exists()
    assert len(train) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert len(validation) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert len(test) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert len(balanced_train) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert len(balanced_validation) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert len(balanced_test) == PMI_PILLS_10_NO_CAL_PROFILE.num_classes * 2
    assert balanced_train["label"].value_counts().nunique() == 1
    assert balanced_validation["label"].value_counts().nunique() == 1
    assert balanced_test["label"].value_counts().nunique() == 1
    assert set(train["label_name"]) == set(PMI_PILLS_10_NO_CAL_PROFILE.class_names)
    assert train["label"].max() == PMI_PILLS_10_NO_CAL_PROFILE.num_classes - 1
    assert (output_dir / "profile_natural.json").exists()
    assert (output_dir / "profile_balanced.json").exists()
    assert (output_dir / "audit_natural.json").exists()
    assert (output_dir / "audit_balanced.json").exists()
