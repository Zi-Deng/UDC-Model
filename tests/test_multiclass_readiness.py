import pandas as pd

from scripts.check_multiclass_readiness import prepared_readiness


def test_pmi10_no_cal_readiness_does_not_require_calibration(tmp_path):
    split_dir = tmp_path / "pmi_pills_10_no_cal" / "splits" / "natural"
    split_dir.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        image_path = tmp_path / f"{split}.jpg"
        image_path.write_bytes(b"fake")
        frame = pd.DataFrame(
            {
                "image_path": [str(image_path)],
                "label": [0],
                "label_name": ["00378-0208"],
                "patient_id": [split],
            }
        )
        frame.to_csv(split_dir / f"{split}.csv", index=False)

    report = prepared_readiness("pmi_pills_10_no_cal", tmp_path, "natural")

    assert report["required_split_names"] == ["train", "validation", "test"]
    assert report["split_files_present"] is True
    assert report["ready"] is True
