"""Preprocess retinal fundus images for diabetic retinopathy classification.

Replicates the FOV (field-of-view) crop + resize pipeline from:
  Galdran et al., "Cost-Sensitive Regularization for Diabetic Retinopathy Grading
  from Eye Fundus Images", MICCAI 2020 (arxiv 2010.00291)

Algorithm:
  1. Threshold the green channel using Li's method
  2. Find the largest connected component (the retinal FOV)
  3. Crop to the bounding box of the FOV
  4. Resize to 512x512
  5. Save as JPEG

Reference implementation:
  playground/cost_sensitive_loss_classification/prepare_training_data.py (lines 20-42)

Usage:
  python scripts/preprocess_dr_data.py --input_dir data/raw/eyepacs/train --output_dir playground/cost_sensitive_loss_classification/data/images --n_jobs 6
  python scripts/preprocess_dr_data.py --input_dir data/raw/messidor2/preprocessed --output_dir playground/cost_sensitive_loss_classification/data/test_messidor_2 --n_jobs 6 --resize_only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage import io
from skimage.filters import threshold_li
from skimage.measure import label, regionprops
from torchvision import transforms as tr
from tqdm import tqdm

TG_SIZE = (512, 512)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
RSZ = tr.Resize(TG_SIZE)


def get_largest_cc(segmentation):
    """Return a binary mask of the largest connected component."""
    labels = label(segmentation)
    largest_cc = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largest_cc


def process_image_fov(input_path, output_path):
    """Crop to field-of-view bounding box and resize to 512x512.

    Uses Li's threshold on the green channel to detect the retinal FOV,
    then crops to the bounding box of the largest connected component.
    """
    try:
        im = io.imread(str(input_path))

        # Handle grayscale images
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]

        # Li's threshold on green channel to find FOV
        t = threshold_li(im[:, :, 1])
        binary = im[:, :, 1] > t
        binary = get_largest_cc(binary)

        # Find bounding box of largest connected component
        label_img = label(binary)
        regions = regionprops(label_img)
        if not regions:
            # Fallback: no regions found, resize the whole image
            cropped = RSZ(Image.fromarray(im))
            cropped.save(str(output_path))
            return True

        areas = [r.area for r in regions]
        largest_cc_idx = np.argmax(areas)
        fov = regions[largest_cc_idx]

        # Crop to FOV bounding box and resize
        cropped = im[fov.bbox[0] : fov.bbox[2], fov.bbox[1] : fov.bbox[3], :]
        cropped = RSZ(Image.fromarray(cropped))
        cropped.save(str(output_path))
        return True
    except Exception as e:
        print(f"  ERROR processing {input_path}: {e}")
        return False


def process_image_resize_only(input_path, output_path):
    """Resize image to 512x512 without FOV cropping (for pre-cropped data)."""
    try:
        im = Image.open(str(input_path)).convert("RGB")
        im = RSZ(im)
        im.save(str(output_path))
        return True
    except Exception as e:
        print(f"  ERROR processing {input_path}: {e}")
        return False


def process_one(input_path, output_dir, resize_only=False):
    """Process a single image: FOV crop + resize, or resize only.

    Output filename is always {stem}.jpg regardless of input extension,
    matching the path format in the existing playground CSVs.
    """
    output_path = output_dir / (input_path.stem + ".jpg")

    # Resume support: skip if output already exists
    if output_path.exists():
        return True

    if resize_only:
        return process_image_resize_only(input_path, output_path)
    else:
        return process_image_fov(input_path, output_path)


def collect_images(input_dir):
    """Collect all image files from the input directory."""
    images = []
    for f in sorted(input_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess retinal fundus images (FOV crop + resize to 512x512)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for preprocessed images")
    parser.add_argument("--n_jobs", type=int, default=6, help="Number of parallel workers (default: 6)")
    parser.add_argument(
        "--resize_only",
        action="store_true",
        help="Only resize to 512x512, skip FOV cropping (for pre-cropped datasets like Messidor-2 preprocessed)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    print(f"Found {len(images)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'resize only' if args.resize_only else 'FOV crop + resize'}")
    print(f"Workers: {args.n_jobs}")

    if not images:
        print("No images found. Exiting.")
        sys.exit(1)

    # Process in parallel with progress bar
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_one)(img, output_dir, args.resize_only) for img in tqdm(images, desc="Processing")
    )

    success = sum(1 for r in results if r)
    failed = len(results) - success
    print(f"\nDone: {success} succeeded, {failed} failed out of {len(images)} total")

    if failed > 0:
        print("WARNING: Some images failed to process. Check the error messages above.")


if __name__ == "__main__":
    main()
