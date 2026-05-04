"""MC0 readiness checks for NICME multiclass datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nicme.data_prep import audit_split_dir
from nicme.dataset_profiles import get_profile


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _has_any(root: Path, names: tuple[str, ...]) -> bool:
    return any((root / name).exists() for name in names)


def _has_split_dir_recursive(root: Path, names: tuple[str, ...]) -> bool:
    lowered = {name.lower() for name in names}
    return any(path.is_dir() and path.name.lower() in lowered for path in [root, *root.rglob("*")])


def _required_split_names(dataset: str) -> tuple[str, ...]:
    profile = get_profile(dataset)
    if "no_calibration" in profile.split_policy:
        return ("train", "validation", "test")
    return ("train", "validation", "calibration", "test")


def raw_readiness(dataset: str, raw_root: Path) -> dict[str, Any]:
    if dataset == "eyepacs_dr":
        root = raw_root / "eyepacs"
        return {
            "path": str(root),
            "exists": root.exists(),
            "labels_present": _has_any(root, ("trainLabels.csv", "trainLabels.csv.zip")),
            "images_present": _has_any(root, ("train", "train.zip", "resized_train")),
            "kaggle_credentials_present": (Path.home() / ".kaggle" / "kaggle.json").exists(),
            "download_note": "Requires Kaggle competition terms acceptance and authenticated Kaggle API.",
        }
    if dataset in {"pmi_pills", "pmi_pills_10_no_cal"}:
        root = raw_root / "pmi"
        split_dirs_present = all(
            _has_split_dir_recursive(root, names)
            for names in (("Train", "train"), ("Valid", "Validation", "valid", "validation", "val"), ("Test", "test"))
        )
        return {
            "path": str(root),
            "exists": root.exists(),
            "archive_present": (root / "medication_images.zip").exists(),
            "split_dirs_present": split_dirs_present,
            "download_note": "Deep Blue direct CLI download may return Cloudflare 403; browser or Globus is expected.",
        }
    return {"path": str(raw_root / dataset), "exists": (raw_root / dataset).exists()}


def prepared_readiness(dataset: str, prepared_root: Path, variant: str) -> dict[str, Any]:
    profile = get_profile(dataset)
    split_dir = prepared_root / dataset / "splits" / variant
    profile_json = prepared_root / dataset / f"profile_{variant}.json"
    audit = audit_split_dir(split_dir, dataset)
    required_split_names = _required_split_names(dataset)
    split_files_present = all((split_dir / f"{name}.csv").exists() for name in required_split_names)
    missing_images = sum(split.get("missing_images", 0) for split in audit.get("splits", {}).values())
    patient_overlap = sum(audit.get("patient_overlap", {}).values())
    ready = split_files_present and missing_images == 0 and patient_overlap == 0
    return {
        "variant": variant,
        "profile_json": str(profile_json),
        "profile_json_present": profile_json.exists(),
        "split_dir": str(split_dir),
        "required_split_names": list(required_split_names),
        "split_files_present": split_files_present,
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "missing_images": missing_images,
        "patient_overlap": patient_overlap,
        "ready": ready,
        "audit": audit,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    datasets = parse_csv(args.datasets)
    variants = parse_csv(args.variants)
    report = {"datasets": {}, "all_ready": True}
    for dataset in datasets:
        get_profile(dataset)
        dataset_report = {
            "raw": raw_readiness(dataset, Path(args.raw_root)),
            "prepared": {},
        }
        for variant in variants:
            dataset_report["prepared"][variant] = prepared_readiness(dataset, Path(args.prepared_root), variant)
        dataset_ready = all(item["ready"] for item in dataset_report["prepared"].values())
        dataset_report["ready"] = dataset_ready
        report["datasets"][dataset] = dataset_report
        report["all_ready"] = report["all_ready"] and dataset_ready
    return report


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mc0_readiness.json"
    md_path = output_dir / "mc0_readiness.md"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write("# MC0 Multiclass Readiness\n\n")
        f.write(f"- All ready: `{report['all_ready']}`\n\n")
        for dataset, dataset_report in report["datasets"].items():
            f.write(f"## {dataset}\n\n")
            raw = dataset_report["raw"]
            f.write(f"- Raw path: `{raw['path']}`\n")
            f.write(f"- Raw exists: `{raw['exists']}`\n")
            f.write(f"- Ready: `{dataset_report['ready']}`\n")
            f.write(f"- Note: {raw.get('download_note', '')}\n\n")
            f.write("| Variant | Split Files | Missing Images | Patient Overlap | Ready |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for variant, prepared in dataset_report["prepared"].items():
                f.write(
                    f"| {variant} | {prepared['split_files_present']} | {prepared['missing_images']} | "
                    f"{prepared['patient_overlap']} | {prepared['ready']} |\n"
                )
            f.write("\n")
    print(f"Wrote readiness reports to {json_path} and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check multiclass MC0 dataset readiness")
    parser.add_argument("--datasets", default="eyepacs_dr,pmi_pills")
    parser.add_argument("--variants", default="balanced")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--prepared-root", default="data/prepared")
    parser.add_argument("--output-dir", default="results/multiclass/mc0")
    args = parser.parse_args()
    write_report(build_report(args), Path(args.output_dir))


if __name__ == "__main__":
    main()
