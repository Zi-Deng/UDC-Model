import pandas as pd
import sqlite3
import os
import sys
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, parent_dir)

import subprocess
import argparse
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset
import shutil
from huggingface_hub import login
from utils.utils import preprocess_local_csv_dataset

@dataclass
class script_args:
    def __init__(self):
        self.numSpecies = 2
        self.numImages = 9000
        self.datasetName = f'examples/breast_cancer_dataset'
        self.hf_token = ""
        self.repo_name = "zkdeng/cbis-ddsm-breast-cancerz"

def convert_paths_to_images(example):
    """
    Convert image_path to actual image data for a single example.
    This function will be used with dataset.map() for memory-efficient processing.
    """
    try:
        # Load image and convert to RGB
        image = Image.open(example['image_path']).convert('RGB')
        example['image'] = image
        # Remove the image_path since we now have the actual image
        del example['image_path']
        return example
    except Exception as e:
        print(f"Warning: Could not load image {example['image_path']}: {e}")
        # Return None to filter out corrupted images
        return None

def load_images_for_dataset_efficient(dataset, batch_size=100):
    """
    Convert a dataset with image_path column to one with actual loaded images.
    Uses memory-efficient batch processing.
    
    Args:
        dataset: HuggingFace Dataset with 'image_path' and 'label' columns
        batch_size: Number of images to process at once
        
    Returns:
        Dataset with 'image' and 'label' columns (images loaded as PIL Images)
    """
    print(f"Converting {len(dataset)} image paths to actual images...")
    print("Using memory-efficient batch processing...")
    
    # Use dataset.map() for memory-efficient processing
    # This processes data in batches and doesn't load everything into memory at once
    dataset_with_images = dataset.map(
        convert_paths_to_images,
        batch_size=batch_size,
        remove_columns=['image_path'],  # Remove image_path column
        desc="Loading images"
    )
    
    # Filter out any None entries (corrupted images)
    dataset_with_images = dataset_with_images.filter(lambda x: x is not None)
    
    print(f"Successfully converted dataset with {len(dataset_with_images)} images")
    return dataset_with_images

def create_imagefolder_structure(dataset, output_dir="temp_imagefolder"):
    """
    Create an imagefolder structure for more efficient upload to Hugging Face.
    This avoids loading all images into memory simultaneously.
    
    Args:
        dataset: Dataset with image_path, label, and split columns
        output_dir: Directory to create the imagefolder structure
        
    Returns:
        path to the created imagefolder directory
    """
    print(f"Creating imagefolder structure in {output_dir}...")
    
    # Remove existing directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create directory structure: output_dir/split/label/images
    os.makedirs(output_dir, exist_ok=True)
    
    # Track copied files
    total_copied = 0
    
    for example in dataset:
        split = example['split']
        label = example['label']
        image_path = example['image_path']
        
        # Create split/label directory
        target_dir = os.path.join(output_dir, split, str(label))
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy image to new location
        image_filename = os.path.basename(image_path)
        target_path = os.path.join(target_dir, image_filename)
        
        # Handle duplicate filenames by adding a counter
        counter = 1
        original_target = target_path
        while os.path.exists(target_path):
            name, ext = os.path.splitext(original_target)
            target_path = f"{name}_{counter}{ext}"
            counter += 1
        
        try:
            shutil.copy2(image_path, target_path)
            total_copied += 1
            
            if total_copied % 100 == 0:
                print(f"Copied {total_copied} images...")
                
        except Exception as e:
            print(f"Warning: Could not copy image {image_path}: {e}")
    
    print(f"Successfully created imagefolder structure with {total_copied} images")
    return output_dir

def main(args: script_args):
    args = script_args()
    numSpecies = args.numSpecies
    numImages = args.numImages
    datasetName = args.datasetName
    hf_token = args.hf_token
    repo_name = args.repo_name

    login(token=hf_token)

    # Get the three datasets from preprocessing (these contain image_path, not actual images)
    train_dataset, val_dataset, test_dataset, class_names = preprocess_local_csv_dataset(datasetName, 'convnext')
    
    print(f"Original datasets loaded:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class names: {class_names}")
    
    # Add a 'split' column to identify which split each sample belongs to
    train_dataset = train_dataset.add_column('split', ['train'] * len(train_dataset))
    val_dataset = val_dataset.add_column('split', ['validation'] * len(val_dataset))
    test_dataset = test_dataset.add_column('split', ['test'] * len(test_dataset))
    
    # Combine all three datasets into one (still with image_path)
    combined_dataset_paths = concatenate_datasets([train_dataset, val_dataset, test_dataset])
    
    print(f"\nCombined dataset with paths created: {len(combined_dataset_paths)} samples")
    
    # Choose upload method based on dataset size
    dataset_size = len(combined_dataset_paths)
    
    if dataset_size > 1000:
        print(f"\nDataset is large ({dataset_size} samples). Using imagefolder approach for memory efficiency...")
        
        # Method 1: Create imagefolder structure (most memory efficient)
        imagefolder_dir = create_imagefolder_structure(combined_dataset_paths)
        
        # Load dataset from imagefolder structure
        final_dataset = load_dataset("imagefolder", data_dir=imagefolder_dir)
        
        print(f"\nFinal dataset ready for upload!")
        print(f"Dataset splits: {list(final_dataset.keys())}")
        
        # Upload the imagefolder dataset
        print(f"\nUploading dataset to: {repo_name}")
        print("Using imagefolder format for efficient upload...")
        final_dataset.push_to_hub(repo_name)
        
        # Clean up temporary directory
        print(f"Cleaning up temporary directory: {imagefolder_dir}")
        shutil.rmtree(imagefolder_dir)
        
    else:
        print(f"\nDataset is small ({dataset_size} samples). Loading images directly...")
        
        # Method 2: Load images directly (for smaller datasets)
        combined_dataset_with_images = load_images_for_dataset_efficient(combined_dataset_paths, batch_size=50)
        
        print(f"\nFinal dataset ready for upload!")
        print(f"Total samples: {len(combined_dataset_with_images)}")
        print(f"Dataset features: {combined_dataset_with_images.features}")
        print(f"Sample entry keys: {list(combined_dataset_with_images[0].keys())}")
        
        # Upload the combined dataset with actual images to Hugging Face
        print(f"\nUploading dataset to: {repo_name}")
        print("This may take a while as we're uploading actual images...")
        combined_dataset_with_images.push_to_hub(repo_name)

if __name__ == "__main__":
    args = script_args()
    main(args)

