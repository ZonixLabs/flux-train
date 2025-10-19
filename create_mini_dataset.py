#!/usr/bin/env python3
"""
Script to create a mini dataset by randomly sampling from the main dataset.
Copies sample folders and creates a new metadata.json file.
"""

import json
import random
import shutil
import csv
from pathlib import Path
from typing import List, Dict

def load_metadata(metadata_path: Path) -> List[Dict]:
    """Load metadata.json."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_series_tracking(csv_path: Path) -> Dict[str, Dict]:
    """Load series tracking and return sample -> data mapping."""
    sample_data = {}
    
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_data[row['sample']] = {
                    'series': row['series'],
                    'video': row['video']
                }
    
    return sample_data

def copy_sample_folder(src_data_dir: Path, dst_data_dir: Path, sample_name: str):
    """Copy a sample folder from source to destination."""
    src_folder = src_data_dir / sample_name
    dst_folder = dst_data_dir / sample_name
    
    if src_folder.exists():
        shutil.copytree(src_folder, dst_folder)
        return True
    else:
        print(f"Warning: Source folder not found: {src_folder}")
        return False

def create_mini_dataset(
    source_data_dir: str = "data",
    output_data_dir: str = "data_mini",
    num_samples: int = 10,
    seed: int = None
):
    """Create a mini dataset by randomly sampling from the main dataset."""
    
    source_data_path = Path(source_data_dir)
    output_data_path = Path(output_data_dir)
    
    # Load metadata
    metadata_path = source_data_path / "metadata.json"
    series_tracking_path = Path("series_tracking.csv")
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = load_metadata(metadata_path)
    print(f"Found {len(metadata)} samples in main dataset")
    
    # Load series tracking if it exists
    series_data = load_series_tracking(series_tracking_path)
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Randomly sample
    if num_samples > len(metadata):
        print(f"Warning: Requested {num_samples} samples but only {len(metadata)} available")
        num_samples = len(metadata)
    
    print(f"Randomly sampling {num_samples} samples...")
    sampled_metadata = random.sample(metadata, num_samples)
    
    # Create output directory
    output_data_path.mkdir(parents=True, exist_ok=True)
    
    # Copy sample folders and collect series tracking data
    print(f"Copying sample folders to {output_data_path}...")
    copied_count = 0
    mini_series_tracking = []
    
    for entry in sampled_metadata:
        # Extract sample name from image path
        sample_name = entry['image'].split('/')[0]
        
        # Copy folder
        success = copy_sample_folder(source_data_path, output_data_path, sample_name)
        if success:
            copied_count += 1
        
        # Collect series tracking data
        if sample_name in series_data:
            mini_series_tracking.append({
                'sample': sample_name,
                'series': series_data[sample_name]['series'],
                'video': series_data[sample_name]['video']
            })
    
    print(f"Successfully copied {copied_count}/{num_samples} sample folders")
    
    # Write mini metadata.json
    mini_metadata_path = output_data_path / "metadata.json"
    print(f"Writing mini metadata to {mini_metadata_path}...")
    with open(mini_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_metadata, f, indent=2, ensure_ascii=False)
    
    # Write mini series_tracking.csv if we have data
    if mini_series_tracking:
        mini_series_tracking_path = Path(f"series_tracking_mini.csv")
        print(f"Writing mini series tracking to {mini_series_tracking_path}...")
        with open(mini_series_tracking_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample', 'series', 'video'])
            writer.writeheader()
            writer.writerows(mini_series_tracking)
    
    print(f"\n✓ Mini dataset created!")
    print(f"  Location: {output_data_path}")
    print(f"  Samples: {num_samples}")
    print(f"  Metadata: {mini_metadata_path}")
    if mini_series_tracking:
        print(f"  Series tracking: series_tracking_mini.csv")

def main():
    # Configuration
    SOURCE_DATA_DIR = "dataset"           # Main dataset directory
    OUTPUT_DATA_DIR = "data_mini"      # Output mini dataset directory
    NUM_SAMPLES = 10                   # Number of samples to randomly select
    RANDOM_SEED = 42                   # Set to None for different results each time
    
    create_mini_dataset(
        source_data_dir=SOURCE_DATA_DIR,
        output_data_dir=OUTPUT_DATA_DIR,
        num_samples=NUM_SAMPLES,
        seed=RANDOM_SEED
    )

if __name__ == "__main__":
    main()