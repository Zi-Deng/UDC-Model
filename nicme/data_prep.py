"""Data preparation utilities for NICME binary experiments."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nicme.dataset_profiles import BREAKHIS_PROFILE, SPIDER_PROFILE

BREAKHIS_URL = "https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
BREAKHIS_EXPECTED_BYTES = 4_273_561_758
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


BREAKHIS_RE = re.compile(
    r"^SOB_(?P<class>[BM])_(?P<tumor_type>[A-Z]+)-(?P<year>\d+)-(?P<patient_id>[A-Z0-9]+)-(?P<magnification>\d+)-(?P<seq>\d+)\.png$",
    re.IGNORECASE,
)


def parse_breakhis_filename(filename: str) -> dict[str, str]:
    match = BREAKHIS_RE.match(Path(filename).name)
    if not match:
        raise ValueError(f"Not a BreaKHis image filename: {filename}")
    data = match.groupdict()
    tumor_class = data.pop("class").upper()
    data["label_name"] = "benign" if tumor_class == "B" else "malignant"
    data["patient_id"] = data["patient_id"].upper()
    data["tumor_type"] = data["tumor_type"].upper()
    data["magnification"] = data["magnification"].lower().removesuffix("x")
    return data


def download_with_resume(url: str, destination: Path, expected_bytes: int | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_bytes is None or destination.stat().st_size == expected_bytes:
            return destination
        raise RuntimeError(
            f"{destination} exists but has {destination.stat().st_size} bytes; expected {expected_bytes}. "
            "Delete it or move it aside before rerunning."
        )
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request) as response, open(tmp_path, mode) as out:
        if existing and getattr(response, "status", None) == 200:
            out.close()
            tmp_path.unlink()
            return download_with_resume(url, destination, expected_bytes)
        shutil.copyfileobj(response, out)
    if expected_bytes is not None and tmp_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Downloaded {tmp_path.stat().st_size} bytes for {url}, expected {expected_bytes}. "
            "Keep the .part file and rerun to resume."
        )
    tmp_path.replace(destination)
    return destination


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        root = output_dir.resolve()
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(output_dir)


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_breakhis_manifest(root: Path, output_csv: Path) -> pd.DataFrame:
    records = []
    label2id = BREAKHIS_PROFILE.label2id
    for image_path in iter_images(root):
        try:
            parsed = parse_breakhis_filename(image_path.name)
        except ValueError:
            continue
        label_name = parsed["label_name"]
        records.append(
            {
                "image_path": str(image_path),
                "label": label2id[label_name],
                "label_name": label_name,
                "patient_id": parsed["patient_id"],
                "magnification": parsed["magnification"],
                "tumor_type": parsed["tumor_type"],
            }
        )
    if not records:
        raise RuntimeError(f"No BreaKHis images found under {root}")
    df = pd.DataFrame(records).sort_values(["patient_id", "image_path"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_spider_manifest(root: Path, output_csv: Path) -> pd.DataFrame:
    mapping = {
        "Latrodectus_hesperus": "black_widow",
        "Steatoda_grossa": "false_widow",
        "black_widow": "black_widow",
        "false_widow": "false_widow",
    }
    label2id = SPIDER_PROFILE.label2id
    records = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label_name = mapping.get(class_dir.name, class_dir.name)
        if label_name not in label2id:
            continue
        for image_path in iter_images(class_dir):
            records.append(
                {
                    "image_path": str(image_path),
                    "label": label2id[label_name],
                    "label_name": label_name,
                    "patient_id": image_path.stem,
                    "magnification": "",
                    "tumor_type": "",
                }
            )
    if not records:
        raise RuntimeError(f"No spider images found under {root}")
    df = pd.DataFrame(records).sort_values(["label", "image_path"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _split_group_table(groups: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = groups["label"] if groups["label"].value_counts().min() >= 2 else None
    try:
        left, right = train_test_split(groups, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        left, right = train_test_split(groups, test_size=test_size, random_state=seed)
    return left.reset_index(drop=True), right.reset_index(drop=True)


def _rows_for_groups(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    return df[df["patient_id"].isin(groups["patient_id"])].reset_index(drop=True)


def _group_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    group_labels = (
        df.groupby("patient_id")["label"]
        .agg(lambda labels: labels.mode().iloc[0])
        .reset_index()
        .sort_values("patient_id")
    )
    train_val_cal_groups, test_groups = _split_group_table(group_labels, test_size=0.10, seed=seed)
    train_groups, val_cal_groups = _split_group_table(train_val_cal_groups, test_size=2 / 9, seed=seed + 1)
    val_groups, cal_groups = _split_group_table(val_cal_groups, test_size=0.50, seed=seed + 2)
    val_cal = _rows_for_groups(df, val_cal_groups)
    return {
        "train": _rows_for_groups(df, train_groups),
        "validation": _rows_for_groups(val_cal, val_groups),
        "calibration": _rows_for_groups(val_cal, cal_groups),
        "test": _rows_for_groups(df, test_groups),
    }


def _stratified_split(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    train_val_cal, test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df["label"])
    train, val_cal = train_test_split(
        train_val_cal,
        test_size=2 / 9,
        random_state=seed + 1,
        stratify=train_val_cal["label"],
    )
    validation, calibration = train_test_split(
        val_cal, test_size=0.50, random_state=seed + 2, stratify=val_cal["label"]
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def balance_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    min_count = int(df["label"].value_counts().min())
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min_count, random_state=seed))
        .sort_values("image_path")
        .reset_index(drop=True)
    )


def controlled_prevalence_split(
    df: pd.DataFrame,
    target_label: int,
    target_prevalence: float,
    seed: int,
) -> pd.DataFrame:
    """Downsample a binary split to a requested target-class prevalence."""
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must be between 0 and 1")
    counts = df["label"].value_counts()
    labels = sorted(counts.index.tolist())
    if len(labels) != 2 or target_label not in labels:
        raise ValueError("controlled_prevalence_split requires binary labels including target_label")

    other_label = labels[0] if labels[1] == target_label else labels[1]
    target_available = int(counts[target_label])
    other_available = int(counts[other_label])

    target_from_other = int(other_available * target_prevalence / (1.0 - target_prevalence))
    other_from_target = int(target_available * (1.0 - target_prevalence) / target_prevalence)
    target_n = min(target_available, max(1, target_from_other))
    other_n = min(other_available, max(1, other_from_target))

    actual_prevalence = target_n / (target_n + other_n)
    if abs(actual_prevalence - target_prevalence) > 0.03:
        if target_available / max(target_available + other_available, 1) > target_prevalence:
            other_n = min(other_available, max(1, int(round(target_n * (1.0 - target_prevalence) / target_prevalence))))
        else:
            target_n = min(target_available, max(1, int(round(other_n * target_prevalence / (1.0 - target_prevalence)))))

    sampled = [
        df[df["label"] == target_label].sample(n=target_n, random_state=seed),
        df[df["label"] == other_label].sample(n=other_n, random_state=seed + 1),
    ]
    return pd.concat(sampled).sort_values("image_path").reset_index(drop=True)


def make_splits(
    manifest: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
    group_by_patient: bool = False,
    make_balanced: bool = True,
    controlled_imbalance_target: int | None = None,
) -> None:
    split_fn = _group_split if group_by_patient else _stratified_split
    splits = split_fn(manifest, seed)
    natural_dir = output_dir / "natural"
    natural_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(natural_dir / f"{name}.csv", index=False)
    if make_balanced:
        balanced_dir = output_dir / "balanced"
        balanced_dir.mkdir(parents=True, exist_ok=True)
        for name, split in splits.items():
            balance_split(split, seed).to_csv(balanced_dir / f"{name}.csv", index=False)
    if controlled_imbalance_target is not None:
        for variant, prevalence in (("target_minority", 0.25), ("target_majority", 0.75)):
            variant_dir = output_dir / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            for name, split in splits.items():
                controlled_prevalence_split(
                    split,
                    target_label=controlled_imbalance_target,
                    target_prevalence=prevalence,
                    seed=seed,
                ).to_csv(variant_dir / f"{name}.csv", index=False)


def write_profile_json(path: Path, profile_name: str, variant: str, split_dir: Path) -> None:
    import json

    from nicme.dataset_profiles import get_profile

    profile = get_profile(profile_name)
    defaults = profile.balanced_selection if variant == "balanced" else profile.natural_selection
    data = {
        "dataset_profile": profile.name,
        "variant": variant,
        "split_dir": str(split_dir),
        "class_names": list(profile.class_names),
        "label2id": profile.label2id,
        "id2label": {str(k): v for k, v in profile.id2label.items()},
        "target_recall_class": profile.target_recall_class,
        "cost_matrix": [list(row) for row in profile.cost_matrix],
        "selection": {
            "target_recall_floor": defaults.target_recall_floor,
            "accuracy_floor": defaults.accuracy_floor,
            "accuracy_metric": defaults.accuracy_metric,
        },
        "selection_accuracy_metric": defaults.accuracy_metric,
        "cost_matrix_convention": "C[true_label][predicted_label]",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def prepare_breakhis(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    archive = raw_dir / "BreaKHis_v1.tar.gz"
    if args.download:
        download_with_resume(BREAKHIS_URL, archive, BREAKHIS_EXPECTED_BYTES)
    if args.extract:
        safe_extract_tar(archive, raw_dir)
    manifest = build_breakhis_manifest(raw_dir, Path(args.output_dir) / "manifest.csv")
    make_splits(manifest, Path(args.output_dir) / "splits", args.seed, group_by_patient=True)
    for variant in ("natural", "balanced"):
        write_profile_json(
            Path(args.output_dir) / f"profile_{variant}.json",
            "breakhis",
            variant,
            Path(args.output_dir) / "splits" / variant,
        )


def prepare_spider(args: argparse.Namespace) -> None:
    manifest = build_spider_manifest(Path(args.input_dir), Path(args.output_dir) / "manifest.csv")
    make_splits(
        manifest,
        Path(args.output_dir) / "splits",
        args.seed,
        group_by_patient=False,
        controlled_imbalance_target=SPIDER_PROFILE.label2id[SPIDER_PROFILE.target_recall_class],
    )
    for variant in ("natural", "balanced", "target_minority", "target_majority"):
        write_profile_json(
            Path(args.output_dir) / f"profile_{variant}.json",
            "spider",
            variant,
            Path(args.output_dir) / "splits" / variant,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare NICME datasets")
    parser.add_argument("--dataset", choices=["breakhis", "spider"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download raw data when supported")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded raw data when supported")
    parser.add_argument("--raw-dir", default="data/raw/breakhis")
    parser.add_argument("--input-dir", default="data/2_class_black_widows")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"data/prepared/{args.dataset}"
    if args.dataset == "breakhis":
        prepare_breakhis(args)
    else:
        prepare_spider(args)


__all__ = [
    "BREAKHIS_URL",
    "BREAKHIS_EXPECTED_BYTES",
    "parse_breakhis_filename",
    "build_breakhis_manifest",
    "build_spider_manifest",
    "make_splits",
    "prepare_breakhis",
    "prepare_spider",
]
