"""Data preparation utilities for NICME image-classification experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nicme.dataset_profiles import BREAKHIS_PROFILE, EYE_PACS_DR_PROFILE, PMI_PILLS_PROFILE, SPIDER_PROFILE, get_profile

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


def safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if os.path.commonpath([str(root), str(target)]) != str(root):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        archive.extractall(output_dir)


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


def _find_first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    lowered = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.name.lower() in lowered:
            return path
    return None


def _eyepacs_patient_id(image_id: str) -> str:
    return str(image_id).split("_", 1)[0]


def build_eyepacs_manifest(root: Path, output_csv: Path) -> pd.DataFrame:
    labels_path = _find_first_existing(root, ("trainLabels.csv", "trainLabels.csv.zip"))
    if labels_path is None:
        raise RuntimeError(
            f"Could not find EyePACS trainLabels.csv under {root}. "
            "Download the Kaggle competition data and extract trainLabels.csv.zip first."
        )
    if labels_path.suffix.lower() == ".zip":
        safe_extract_zip(labels_path, labels_path.parent)
        labels_path = labels_path.with_suffix("")

    labels = pd.read_csv(labels_path)
    if not {"image", "level"}.issubset(labels.columns):
        raise ValueError(f"{labels_path} must contain columns 'image' and 'level'")

    image_root = _find_first_existing(root, ("train", "resized_train", "images"))
    if image_root is None:
        image_root = root
    path_by_stem = {path.stem: path for path in iter_images(image_root)}

    label2id = EYE_PACS_DR_PROFILE.label2id
    records = []
    for _, row in labels.iterrows():
        image_id = str(row["image"])
        level = int(row["level"])
        label_name = f"DR{level}"
        if label_name not in label2id:
            raise ValueError(f"Unexpected EyePACS DR grade {level!r} for image {image_id!r}")
        image_path = path_by_stem.get(image_id, image_root / f"{image_id}.jpeg")
        records.append(
            {
                "image_path": str(image_path),
                "label": label2id[label_name],
                "label_name": label_name,
                "patient_id": _eyepacs_patient_id(image_id),
                "eye": image_id.split("_", 1)[1] if "_" in image_id else "",
                "image_id": image_id,
            }
        )
    df = pd.DataFrame(records).sort_values(["patient_id", "image_id"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _find_split_dir(root: Path, names: tuple[str, ...]) -> Path:
    lowered = {name.lower() for name in names}
    candidates = []
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and path.name.lower() in lowered:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"Could not find any of split directories {names} under {root}")
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def _build_pmi_split_frame(split_dir: Path, profile_name: str = "pmi_pills") -> pd.DataFrame:
    profile = get_profile(profile_name)
    label2id = profile.label2id
    records = []
    unexpected_classes = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label_name = class_dir.name
        if label_name not in label2id:
            if profile_name == PMI_PILLS_PROFILE.name:
                unexpected_classes.append(label_name)
            continue
        for image_path in iter_images(class_dir):
            records.append(
                {
                    "image_path": str(image_path),
                    "label": label2id[label_name],
                    "label_name": label_name,
                    "patient_id": image_path.stem,
                    "ndc": label_name,
                    "source_split": split_dir.name,
                }
            )
    if unexpected_classes:
        raise ValueError(
            f"PMI split {split_dir} contains unexpected class folders: {unexpected_classes}. "
            f"Expected NDC classes: {list(profile.class_names)}"
        )
    if not records:
        raise RuntimeError(f"No PMI pill images found in expected class folders under {split_dir}")
    frame = pd.DataFrame(records).sort_values(["label", "image_path"]).reset_index(drop=True)
    found = set(frame["label_name"].unique())
    missing = [name for name in profile.class_names if name not in found]
    if missing:
        raise ValueError(f"PMI split {split_dir} is missing expected NDC class folders: {missing}")
    return frame


def build_pmi_split_frames(root: Path, profile_name: str = "pmi_pills") -> dict[str, pd.DataFrame]:
    split_dirs = {
        "train": _find_split_dir(root, ("Train", "train")),
        "valid": _find_split_dir(root, ("Valid", "Validation", "valid", "validation", "val")),
        "test": _find_split_dir(root, ("Test", "test")),
    }
    return {name: _build_pmi_split_frame(path, profile_name=profile_name) for name, path in split_dirs.items()}


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


def _group_split(df: pd.DataFrame, seed: int, group_label_strategy: str = "mode") -> dict[str, pd.DataFrame]:
    if group_label_strategy == "max":
        label_agg = "max"
    elif group_label_strategy == "mode":
        def label_agg(labels):
            return labels.mode().iloc[0]
    else:
        raise ValueError("group_label_strategy must be 'mode' or 'max'")
    group_labels = (
        df.groupby("patient_id")["label"]
        .agg(label_agg)
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
    sampled = [group.sample(n=min_count, random_state=seed) for _, group in df.groupby("label", sort=True)]
    return pd.concat(sampled).sort_values("image_path").reset_index(drop=True)


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
    group_label_strategy: str = "mode",
) -> None:
    if group_by_patient:
        splits = _group_split(manifest, seed, group_label_strategy=group_label_strategy)
    else:
        splits = _stratified_split(manifest, seed)
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
        "target_recall_classes": list(profile.cared_classes),
        "secondary_target_recall_classes": list(profile.secondary_target_recall_classes),
        "cost_matrix": [list(row) for row in profile.cost_matrix],
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "cost_matrix_source": profile.cost_matrix_source,
        "metric_profile": profile.metric_profile,
        "split_policy": profile.split_policy,
        "critical_pairs": [
            {"true": true_name, "pred": pred_name, "cost": cost}
            for true_name, pred_name, cost in profile.critical_pairs
        ],
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


def audit_split_dir(split_dir: Path, profile_name: str) -> dict:
    profile = get_profile(profile_name)
    split_dir = Path(split_dir)
    audit = {
        "dataset_profile": profile.name,
        "split_dir": str(split_dir),
        "exists": split_dir.exists(),
        "cost_matrix_convention": "C[true_label][predicted_label]",
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "num_classes": profile.num_classes,
        "class_names": list(profile.class_names),
        "cared_classes": list(profile.cared_classes),
        "critical_pairs": [
            {"true": true_name, "pred": pred_name, "cost": cost}
            for true_name, pred_name, cost in profile.critical_pairs
        ],
        "splits": {},
        "patient_overlap": {},
        "duplicate_image_paths": {},
    }
    if not split_dir.exists():
        return audit

    split_frames = {}
    image_path_owners = {}
    for split_name in ("train", "validation", "calibration", "test"):
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            audit["splits"][split_name] = {"exists": False}
            continue
        frame = pd.read_csv(csv_path)
        split_frames[split_name] = frame
        row_count = int(len(frame))
        label_counts = frame["label"].value_counts().sort_index() if "label" in frame.columns else pd.Series(dtype=int)
        label_name_counts = (
            frame["label_name"].value_counts().sort_index() if "label_name" in frame.columns else pd.Series(dtype=int)
        )
        missing_images = 0
        if "image_path" in frame.columns:
            missing_images = int(sum(not Path(str(path)).exists() for path in frame["image_path"]))
            for image_path in frame["image_path"].astype(str):
                image_path_owners.setdefault(image_path, set()).add(split_name)
        cared_prevalence = {}
        for cared in profile.cared_classes:
            class_id = profile.label2id[cared]
            cared_prevalence[cared] = float((frame["label"] == class_id).mean()) if row_count and "label" in frame else 0.0
        audit["splits"][split_name] = {
            "exists": True,
            "rows": row_count,
            "label_counts": {str(key): int(value) for key, value in label_counts.items()},
            "label_name_counts": {str(key): int(value) for key, value in label_name_counts.items()},
            "missing_images": missing_images,
            "cared_class_prevalence": cared_prevalence,
            "unique_patients": int(frame["patient_id"].nunique()) if "patient_id" in frame.columns else None,
        }

    for left in split_frames:
        for right in split_frames:
            if left >= right:
                continue
            if "patient_id" in split_frames[left].columns and "patient_id" in split_frames[right].columns:
                left_ids = set(split_frames[left]["patient_id"].astype(str))
                right_ids = set(split_frames[right]["patient_id"].astype(str))
                audit["patient_overlap"][f"{left}__{right}"] = len(left_ids & right_ids)
    audit["duplicate_image_paths"] = {
        image_path: sorted(owners) for image_path, owners in image_path_owners.items() if len(owners) > 1
    }
    return audit


def write_audit_reports(output_dir: Path, profile_name: str, variants: tuple[str, ...]) -> None:
    output_dir = Path(output_dir)
    profile = get_profile(profile_name)
    for variant in variants:
        split_dir = output_dir / "splits" / variant
        audit = audit_split_dir(split_dir, profile_name)
        json_path = output_dir / f"audit_{variant}.json"
        md_path = output_dir / f"audit_{variant}.md"
        with open(json_path, "w") as f:
            json.dump(audit, f, indent=2)
        with open(md_path, "w") as f:
            f.write(f"# {profile.name} {variant} Data Audit\n\n")
            f.write(f"- Split dir: `{split_dir}`\n")
            f.write(f"- Cost matrix SHA256: `{profile.cost_matrix_sha256}`\n")
            f.write(f"- Cared classes: {', '.join(profile.cared_classes)}\n")
            f.write("- Cost convention: `C[true_label][predicted_label]`\n\n")
            f.write("| Split | Rows | Missing Images | Label Counts |\n")
            f.write("|---|---:|---:|---|\n")
            for split_name in ("train", "validation", "calibration", "test"):
                split = audit["splits"].get(split_name, {})
                if not split.get("exists"):
                    f.write(f"| {split_name} | missing | missing | missing |\n")
                    continue
                f.write(
                    f"| {split_name} | {split['rows']} | {split['missing_images']} | "
                    f"`{split['label_name_counts'] or split['label_counts']}` |\n"
                )
            if audit["patient_overlap"]:
                f.write("\n## Patient Overlap\n\n")
                for key, value in audit["patient_overlap"].items():
                    f.write(f"- `{key}`: {value}\n")


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


def prepare_eyepacs_dr(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if args.extract:
        for archive_name in ("trainLabels.csv.zip", "train.zip"):
            archive = input_dir / archive_name
            if archive.exists():
                safe_extract_zip(archive, input_dir)
    manifest = build_eyepacs_manifest(input_dir, Path(args.output_dir) / "manifest.csv")
    make_splits(
        manifest,
        Path(args.output_dir) / "splits",
        args.seed,
        group_by_patient=True,
        group_label_strategy="max",
    )
    for variant in ("natural", "balanced"):
        write_profile_json(
            Path(args.output_dir) / f"profile_{variant}.json",
            "eyepacs_dr",
            variant,
            Path(args.output_dir) / "splits" / variant,
        )
    write_audit_reports(Path(args.output_dir), "eyepacs_dr", ("natural", "balanced"))


def prepare_pmi_pills(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    archive = input_dir / "medication_images.zip"
    if args.extract and archive.exists():
        safe_extract_zip(archive, input_dir)
    frames = build_pmi_split_frames(input_dir)
    train = frames["train"].reset_index(drop=True)
    validation, calibration = train_test_split(
        frames["valid"],
        test_size=0.50,
        random_state=args.seed,
        stratify=frames["valid"]["label"],
    )
    splits = {
        "train": train,
        "validation": validation.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "test": frames["test"].reset_index(drop=True),
    }
    manifest = pd.concat(
        [frame.assign(prepared_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(args.output_dir) / "manifest.csv", index=False)

    natural_dir = Path(args.output_dir) / "splits" / "natural"
    natural_dir.mkdir(parents=True, exist_ok=True)
    balanced_dir = Path(args.output_dir) / "splits" / "balanced"
    balanced_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.to_csv(natural_dir / f"{split_name}.csv", index=False)
        balance_split(frame, args.seed).to_csv(balanced_dir / f"{split_name}.csv", index=False)
    for variant in ("natural", "balanced"):
        write_profile_json(
            Path(args.output_dir) / f"profile_{variant}.json",
            "pmi_pills",
            variant,
            Path(args.output_dir) / "splits" / variant,
        )
    write_audit_reports(Path(args.output_dir), "pmi_pills", ("natural", "balanced"))


def prepare_pmi_pills_10_no_cal(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    archive = input_dir / "medication_images.zip"
    if args.extract and archive.exists():
        safe_extract_zip(archive, input_dir)
    frames = build_pmi_split_frames(input_dir, profile_name="pmi_pills_10_no_cal")
    splits = {
        "train": frames["train"].reset_index(drop=True),
        "validation": frames["valid"].reset_index(drop=True),
        "test": frames["test"].reset_index(drop=True),
    }
    manifest = pd.concat(
        [frame.assign(prepared_split=name) for name, frame in splits.items()],
        ignore_index=True,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest.to_csv(Path(args.output_dir) / "manifest.csv", index=False)

    natural_dir = Path(args.output_dir) / "splits" / "natural"
    natural_dir.mkdir(parents=True, exist_ok=True)
    balanced_dir = Path(args.output_dir) / "splits" / "balanced"
    balanced_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame in splits.items():
        frame.to_csv(natural_dir / f"{split_name}.csv", index=False)
        balance_split(frame, args.seed).to_csv(balanced_dir / f"{split_name}.csv", index=False)
    write_profile_json(
        Path(args.output_dir) / "profile_natural.json",
        "pmi_pills_10_no_cal",
        "natural",
        natural_dir,
    )
    write_profile_json(
        Path(args.output_dir) / "profile_balanced.json",
        "pmi_pills_10_no_cal",
        "balanced",
        balanced_dir,
    )
    write_audit_reports(Path(args.output_dir), "pmi_pills_10_no_cal", ("natural", "balanced"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare NICME datasets")
    parser.add_argument(
        "--dataset",
        choices=["breakhis", "spider", "eyepacs_dr", "pmi_pills", "pmi_pills_10_no_cal"],
        required=True,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download raw data when supported")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded raw data when supported")
    parser.add_argument("--raw-dir", default="data/raw/breakhis")
    parser.add_argument("--input-dir", default="data/2_class_black_widows")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"data/prepared/{args.dataset}"
    if args.dataset in {"pmi_pills", "pmi_pills_10_no_cal"} and args.input_dir == "data/2_class_black_widows":
        args.input_dir = args.raw_dir if args.raw_dir != "data/raw/breakhis" else "data/raw/pmi"
    if args.dataset == "breakhis":
        prepare_breakhis(args)
    elif args.dataset == "spider":
        prepare_spider(args)
    elif args.dataset == "eyepacs_dr":
        prepare_eyepacs_dr(args)
    elif args.dataset == "pmi_pills":
        prepare_pmi_pills(args)
    else:
        prepare_pmi_pills_10_no_cal(args)


__all__ = [
    "BREAKHIS_URL",
    "BREAKHIS_EXPECTED_BYTES",
    "parse_breakhis_filename",
    "build_breakhis_manifest",
    "build_spider_manifest",
    "build_eyepacs_manifest",
    "build_pmi_split_frames",
    "make_splits",
    "prepare_breakhis",
    "prepare_spider",
    "prepare_eyepacs_dr",
    "prepare_pmi_pills",
    "prepare_pmi_pills_10_no_cal",
]


if __name__ == "__main__":
    main()
