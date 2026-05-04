"""Create NICME format CSV files from existing playground CSV files.

Converts the playground's (image_id, dr) CSV format to the NICME's
(image_path, binary_label) CSV format for both EyePACS and Messidor-2 datasets.

The playground CSVs use paths relative to playground/cost_sensitive_loss_classification/,
while NICME CSVs use paths relative to the local_folder_path config value.

Usage:
  python scripts/create_dr_csvs.py
  python scripts/create_dr_csvs.py --playground_dir playground/cost_sensitive_loss_classification
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import polars as pl


def create_eyepacs_csvs(playground_dir: Path, output_dir: Path):
    """Create NICME CSVs for EyePACS dataset.

    Combines playground train.csv + val.csv -> train_dataset.csv
    (the NICME pipeline does its own 90/10 train/val split internally).
    Uses playground test_eyepacs.csv -> test_dataset.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load playground CSVs
    train_df = pl.read_csv(str(playground_dir / "data" / "train.csv"))
    val_df = pl.read_csv(str(playground_dir / "data" / "val.csv"))
    test_df = pl.read_csv(str(playground_dir / "data" / "test_eyepacs.csv"))

    # Combine train + val for NICME (it does its own split)
    combined_train = pl.concat([train_df, val_df])

    # Convert paths: "data/images/X.jpg" -> "images/X.jpg"
    # (relative to local_folder_path="data/eyepacs", accessed via symlink data/eyepacs/images/)
    udc_train = combined_train.select(
        [
            pl.col("image_id").str.replace("data/images/", "images/").alias("image_path"),
            pl.col("dr").alias("binary_label"),
        ]
    )

    # Convert paths: "data/test_eyepacs/X.jpg" -> "test_images/X.jpg"
    # (accessed via symlink data/eyepacs/test_images/)
    udc_test = test_df.select(
        [
            pl.col("image_id").str.replace("data/test_eyepacs/", "test_images/").alias("image_path"),
            pl.col("dr").alias("binary_label"),
        ]
    )

    train_path = output_dir / "train_dataset.csv"
    test_path = output_dir / "test_dataset.csv"
    udc_train.write_csv(str(train_path))
    udc_test.write_csv(str(test_path))

    print(f"EyePACS train: {len(udc_train)} rows -> {train_path}")
    print(f"  Labels: {sorted(udc_train['binary_label'].unique().to_list())}")
    print(f"EyePACS test:  {len(udc_test)} rows -> {test_path}")
    print(f"  Labels: {sorted(udc_test['binary_label'].unique().to_list())}")


def create_messidor2_csvs(playground_dir: Path, output_dir: Path, eyepacs_dir: Path):
    """Create NICME CSVs for Messidor-2 dataset.

    Messidor-2 is test-only. For train_dataset.csv, we reuse the EyePACS training data
    (the standard approach: train on EyePACS, evaluate on Messidor-2 as external test).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Messidor-2 test CSV from playground
    messidor_df = pl.read_csv(str(playground_dir / "data" / "test_messidor_2.csv"))

    # Convert paths: "data/test_messidor_2/X.jpg" -> "images/X.jpg"
    # (accessed via symlink data/messidor2/images/)
    udc_test = messidor_df.select(
        [
            pl.col("image_id").str.replace("data/test_messidor_2/", "images/").alias("image_path"),
            pl.col("dr").alias("binary_label"),
        ]
    )

    test_path = output_dir / "test_dataset.csv"
    udc_test.write_csv(str(test_path))

    # For train_dataset.csv, reuse EyePACS training data with adjusted paths
    # The images are accessed via symlinks:
    #   data/messidor2/train_images/ -> playground/.../data/images/
    eyepacs_train = pl.read_csv(str(eyepacs_dir / "train_dataset.csv"))
    messidor_train = eyepacs_train.with_columns(
        pl.col("image_path").str.replace("images/", "train_images/"),
    )
    train_path = output_dir / "train_dataset.csv"
    messidor_train.write_csv(str(train_path))

    print(f"Messidor-2 test:  {len(udc_test)} rows -> {test_path}")
    print(f"  Labels: {sorted(udc_test['binary_label'].unique().to_list())}")
    print(f"Messidor-2 train (from EyePACS): {len(messidor_train)} rows -> {train_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create NICME CSV files from playground CSVs")
    parser.add_argument(
        "--playground_dir",
        type=str,
        default="playground/cost_sensitive_loss_classification",
        help="Path to playground directory (default: playground/cost_sensitive_loss_classification)",
    )
    parser.add_argument(
        "--eyepacs_dir",
        type=str,
        default="data/eyepacs",
        help="Output directory for EyePACS CSVs (default: data/eyepacs)",
    )
    parser.add_argument(
        "--messidor2_dir",
        type=str,
        default="data/messidor2",
        help="Output directory for Messidor-2 CSVs (default: data/messidor2)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    playground_dir = Path(args.playground_dir)
    eyepacs_dir = Path(args.eyepacs_dir)
    messidor2_dir = Path(args.messidor2_dir)

    if not playground_dir.exists():
        print(f"ERROR: Playground directory does not exist: {playground_dir}")
        sys.exit(1)

    # Check that playground CSVs exist
    required_csvs = ["data/train.csv", "data/val.csv", "data/test_eyepacs.csv", "data/test_messidor_2.csv"]
    for csv in required_csvs:
        if not (playground_dir / csv).exists():
            print(f"ERROR: Required CSV not found: {playground_dir / csv}")
            sys.exit(1)

    print("=== Creating EyePACS CSVs ===")
    create_eyepacs_csvs(playground_dir, eyepacs_dir)

    print("\n=== Creating Messidor-2 CSVs ===")
    create_messidor2_csvs(playground_dir, messidor2_dir, eyepacs_dir)

    print("\nDone. CSVs created successfully.")


if __name__ == "__main__":
    main()
