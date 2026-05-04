#!/usr/bin/env bash
# setup_dr_data.sh -- Download and preprocess EyePACS + Messidor-2 datasets
#
# Prepares data for both the main NICME pipeline (scripts/train.py)
# and the playground pipeline (playground/cost_sensitive_loss_classification/train.py).
#
# Prerequisites:
#   1. micromamba activate ml
#   2. pip install kaggle scikit-image
#   3. ~/.kaggle/kaggle.json configured (https://www.kaggle.com/settings -> API -> Create New Token)
#   4. Accept EyePACS competition rules at https://www.kaggle.com/c/diabetic-retinopathy-detection/rules
#
# Usage:
#   bash scripts/setup_dr_data.sh
#   bash scripts/setup_dr_data.sh --skip-download
#   bash scripts/setup_dr_data.sh --skip-download --skip-preprocess
#   bash scripts/setup_dr_data.sh --n-jobs 12
#
# Paper: Galdran et al., "Cost-Sensitive Regularization for Diabetic Retinopathy
#        Grading from Eye Fundus Images", MICCAI 2020 (arxiv 2010.00291)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults
SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
N_JOBS=6

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash scripts/setup_dr_data.sh [--skip-download] [--skip-preprocess] [--n-jobs N]"
            echo ""
            echo "Options:"
            echo "  --skip-download     Skip downloading raw datasets (use existing data/raw/)"
            echo "  --skip-preprocess   Skip image preprocessing (use existing preprocessed images)"
            echo "  --n-jobs N          Number of parallel workers for preprocessing (default: 6)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

PLAYGROUND_DIR="$REPO_ROOT/playground/cost_sensitive_loss_classification"

echo "============================================"
echo "  EyePACS + Messidor-2 Dataset Setup"
echo "============================================"
echo "Repo root: $REPO_ROOT"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Skip preprocess: $SKIP_PREPROCESS"
echo "Parallel jobs: $N_JOBS"
echo ""

# -----------------------------------------------
# Phase 0: Prerequisite checks
# -----------------------------------------------
echo "=== Phase 0: Checking prerequisites ==="

if ! command -v kaggle &>/dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi
echo "  kaggle CLI: $(kaggle --version)"

if ! python -c "import skimage" &>/dev/null; then
    echo "ERROR: scikit-image not found. Install with: pip install scikit-image"
    exit 1
fi
echo "  scikit-image: OK"

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "ERROR: Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    echo "  1. Go to https://www.kaggle.com/settings -> API -> Create New Token"
    echo "  2. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo "  3. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi
echo "  Kaggle credentials: OK"
echo ""

# -----------------------------------------------
# Phase 1: Download raw datasets
# -----------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=== Phase 1: Downloading raw datasets ==="
    echo ""
    echo "WARNING: EyePACS dataset is ~89 GB. Ensure you have sufficient disk space."
    echo "  Required: ~130 GB during download (raw + compressed archives)"
    echo "  After cleanup: ~89 GB (raw images only)"
    echo ""

    # --- EyePACS ---
    echo "--- Downloading EyePACS from Kaggle ---"
    mkdir -p data/raw/eyepacs
    cd data/raw/eyepacs

    # Training images (multipart zip)
    echo "  Downloading training images (~35 GB compressed)..."
    for i in 1 2 3 4 5; do
        if [ ! -f "train.zip.00$i" ]; then
            kaggle competitions download -c diabetic-retinopathy-detection -f "train.zip.00$i"
        else
            echo "    train.zip.00$i already exists, skipping"
        fi
    done

    if [ ! -d "train" ]; then
        echo "  Combining and extracting training images..."
        cat train.zip.* > train.zip
        unzip -q train.zip -d .
        rm -f train.zip
    else
        echo "  train/ directory already exists, skipping extraction"
    fi

    # Training labels
    if [ ! -f "trainLabels.csv" ]; then
        echo "  Downloading training labels..."
        kaggle competitions download -c diabetic-retinopathy-detection -f trainLabels.csv.zip
        unzip -q trainLabels.csv.zip
        rm -f trainLabels.csv.zip
    else
        echo "  trainLabels.csv already exists, skipping"
    fi

    # Test images (multipart zip)
    echo "  Downloading test images (~20 GB compressed)..."
    for part in $(kaggle competitions files -c diabetic-retinopathy-detection 2>/dev/null | grep 'test.zip' | awk '{print $1}'); do
        if [ ! -f "$part" ]; then
            kaggle competitions download -c diabetic-retinopathy-detection -f "$part"
        else
            echo "    $part already exists, skipping"
        fi
    done

    if [ ! -d "test" ]; then
        echo "  Combining and extracting test images..."
        if ls test.zip.* 1>/dev/null 2>&1; then
            cat test.zip.* > test.zip
            unzip -q test.zip -d .
            rm -f test.zip
        elif [ -f "test.zip" ]; then
            unzip -q test.zip -d .
            rm -f test.zip
        fi
    else
        echo "  test/ directory already exists, skipping extraction"
    fi

    # Test labels
    if [ ! -f "retinopathy_solution.csv" ]; then
        echo "  Downloading test labels..."
        kaggle competitions download -c diabetic-retinopathy-detection -f retinopathy_solution.csv.zip
        unzip -q retinopathy_solution.csv.zip
        rm -f retinopathy_solution.csv.zip
    else
        echo "  retinopathy_solution.csv already exists, skipping"
    fi

    cd "$REPO_ROOT"

    # --- Messidor-2 ---
    echo ""
    echo "--- Downloading Messidor-2 ---"
    mkdir -p data/raw/messidor2
    cd data/raw/messidor2

    # Preprocessed images (399 MB, already has black borders removed)
    if [ ! -d "preprocessed" ] && [ "$(ls -A . 2>/dev/null | grep -v '.zip' | head -1)" = "" ]; then
        echo "  Downloading preprocessed Messidor-2 images (399 MB)..."
        kaggle datasets download -d mariaherrerot/messidor2preprocess
        mkdir -p preprocessed
        unzip -q messidor2preprocess.zip -d preprocessed
        rm -f messidor2preprocess.zip
    else
        echo "  Messidor-2 images already present, skipping"
    fi

    # DR grades (7.5 KB)
    if [ ! -f "messidor_data.csv" ] && [ ! -f "messidor2_dr_grades.csv" ]; then
        echo "  Downloading Messidor-2 DR grades..."
        kaggle datasets download -d google-brain/messidor2-dr-grades
        unzip -q messidor2-dr-grades.zip
        rm -f messidor2-dr-grades.zip
    else
        echo "  Messidor-2 DR grades already present, skipping"
    fi

    cd "$REPO_ROOT"
    echo ""
    echo "Phase 1 complete."
else
    echo "=== Phase 1: Skipped (--skip-download) ==="
fi
echo ""

# -----------------------------------------------
# Phase 2: Preprocess images
# -----------------------------------------------
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "=== Phase 2: Preprocessing images (FOV crop + resize to 512x512) ==="
    echo ""

    # EyePACS training images
    EYEPACS_TRAIN_IN="data/raw/eyepacs/train"
    EYEPACS_TRAIN_OUT="$PLAYGROUND_DIR/data/images"
    if [ -d "$EYEPACS_TRAIN_IN" ]; then
        echo "--- Processing EyePACS training images ---"
        python scripts/preprocess_dr_data.py \
            --input_dir "$EYEPACS_TRAIN_IN" \
            --output_dir "$EYEPACS_TRAIN_OUT" \
            --n_jobs "$N_JOBS"
        echo ""
    else
        echo "  WARNING: $EYEPACS_TRAIN_IN not found. Skipping EyePACS train preprocessing."
    fi

    # EyePACS test images
    EYEPACS_TEST_IN="data/raw/eyepacs/test"
    EYEPACS_TEST_OUT="$PLAYGROUND_DIR/data/test_eyepacs"
    if [ -d "$EYEPACS_TEST_IN" ]; then
        echo "--- Processing EyePACS test images ---"
        python scripts/preprocess_dr_data.py \
            --input_dir "$EYEPACS_TEST_IN" \
            --output_dir "$EYEPACS_TEST_OUT" \
            --n_jobs "$N_JOBS"
        echo ""
    else
        echo "  WARNING: $EYEPACS_TEST_IN not found. Skipping EyePACS test preprocessing."
    fi

    # Messidor-2 images (resize only -- preprocessed version already has borders removed)
    MESSIDOR_IN="data/raw/messidor2/preprocessed"
    MESSIDOR_OUT="$PLAYGROUND_DIR/data/test_messidor_2"
    if [ -d "$MESSIDOR_IN" ]; then
        echo "--- Processing Messidor-2 images (resize only) ---"
        python scripts/preprocess_dr_data.py \
            --input_dir "$MESSIDOR_IN" \
            --output_dir "$MESSIDOR_OUT" \
            --n_jobs "$N_JOBS" \
            --resize_only
        echo ""
    else
        echo "  WARNING: $MESSIDOR_IN not found. Skipping Messidor-2 preprocessing."
    fi

    echo "Phase 2 complete."
else
    echo "=== Phase 2: Skipped (--skip-preprocess) ==="
fi
echo ""

# -----------------------------------------------
# Phase 3: Create symlinks and NICME CSVs
# -----------------------------------------------
echo "=== Phase 3: Setting up NICME directory structure ==="

# Create symlinks for EyePACS
echo "  Creating symlinks for EyePACS..."
mkdir -p data/eyepacs
if [ ! -L "data/eyepacs/images" ] && [ ! -d "data/eyepacs/images" ]; then
    ln -s "../../playground/cost_sensitive_loss_classification/data/images" "data/eyepacs/images"
    echo "    data/eyepacs/images -> playground/.../data/images"
fi
if [ ! -L "data/eyepacs/test_images" ] && [ ! -d "data/eyepacs/test_images" ]; then
    ln -s "../../playground/cost_sensitive_loss_classification/data/test_eyepacs" "data/eyepacs/test_images"
    echo "    data/eyepacs/test_images -> playground/.../data/test_eyepacs"
fi

# Create symlinks for Messidor-2
echo "  Creating symlinks for Messidor-2..."
mkdir -p data/messidor2
if [ ! -L "data/messidor2/images" ] && [ ! -d "data/messidor2/images" ]; then
    ln -s "../../playground/cost_sensitive_loss_classification/data/test_messidor_2" "data/messidor2/images"
    echo "    data/messidor2/images -> playground/.../data/test_messidor_2"
fi
if [ ! -L "data/messidor2/train_images" ] && [ ! -d "data/messidor2/train_images" ]; then
    ln -s "../../playground/cost_sensitive_loss_classification/data/images" "data/messidor2/train_images"
    echo "    data/messidor2/train_images -> playground/.../data/images"
fi

# Create NICME format CSVs
echo "  Creating NICME format CSV files..."
python scripts/create_dr_csvs.py

echo ""
echo "Phase 3 complete."
echo ""

# -----------------------------------------------
# Phase 4: Verification
# -----------------------------------------------
echo "=== Phase 4: Verification ==="
echo ""

# Check playground image directories
echo "--- Playground image counts ---"
for dir in "$PLAYGROUND_DIR/data/images" "$PLAYGROUND_DIR/data/test_eyepacs" "$PLAYGROUND_DIR/data/test_messidor_2"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
        echo "  $dir: $count images"
    else
        echo "  $dir: NOT FOUND"
    fi
done

echo ""
echo "--- NICME CSV files ---"
for csv in data/eyepacs/train_dataset.csv data/eyepacs/test_dataset.csv data/messidor2/train_dataset.csv data/messidor2/test_dataset.csv; do
    if [ -f "$csv" ]; then
        lines=$(($(wc -l < "$csv") - 1))
        echo "  $csv: $lines rows"
    else
        echo "  $csv: NOT FOUND"
    fi
done

echo ""
echo "--- Symlink verification ---"
for link in data/eyepacs/images data/eyepacs/test_images data/messidor2/images data/messidor2/train_images; do
    if [ -L "$link" ]; then
        target=$(readlink "$link")
        echo "  $link -> $target ($([ -e "$link" ] && echo 'OK' || echo 'BROKEN'))"
    else
        echo "  $link: NOT A SYMLINK"
    fi
done

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify image counts match expected values:"
echo "     - playground data/images:        ~35,126 (EyePACS train+val)"
echo "     - playground data/test_eyepacs:  ~53,576 (EyePACS test)"
echo "     - playground data/test_messidor_2: ~1,744 (Messidor-2)"
echo ""
echo "  2. Run playground smoke test:"
echo "     cd playground/cost_sensitive_loss_classification"
echo "     python train.py --csv_train train.csv --model_name resnet50 --base_loss gls --lambd 10 --exp 2 --n_epochs 1 --batch_size 4 --patience 1 --save_path smoke_test"
echo ""
echo "  3. Run NICME smoke test (set num_train_epochs to 1 first):"
echo "     python scripts/train.py --config config/eyepacs_5class.json"
