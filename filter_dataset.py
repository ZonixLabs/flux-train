#!/usr/bin/env python3
"""
Filter dataset to remove samples that don't have input images.
This script:
1. Reads metadata.json
2. Checks if 1.jpg, 2.jpg, 3.jpg, or 4.jpg exists for each sample
3. Removes sample directories that don't have any input images
4. Creates a cleaned metadata.json
Usage:
    python filter_dataset.py [--data-dir data] [--dry-run]
"""
import json
import shutil
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("filter_dataset")


def filter_dataset(data_dir: Path, dry_run: bool = False):
    """Filter out samples without input images."""
    
    metadata_path = data_dir / "metadata.json"
    
    if not metadata_path.exists():
        LOG.error(f"metadata.json not found in {data_dir}")
        return
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    LOG.info(f"Loaded {len(metadata)} samples from metadata.json")
    
    # Filter samples
    cleaned_metadata = []
    removed_count = 0
    
    for entry in metadata:
        # Get the sample directory from the image path
        image_path = entry.get('image', '')
        sample_dir = data_dir / Path(image_path).parent
        
        # Check if any of 1.jpg, 2.jpg, 3.jpg, or 4.jpg exists
        input_images = [sample_dir / f"{i}.jpg" for i in [1, 2, 3, 4]]
        has_input = any(img.exists() for img in input_images)
        
        if has_input:
            cleaned_metadata.append(entry)
        else:
            removed_count += 1
            LOG.warning(f"Removing {sample_dir.name}: no input images (1.jpg, 2.jpg, 3.jpg, or 4.jpg) found")
            
            if not dry_run:
                # Remove the sample directory
                if sample_dir.exists():
                    try:
                        shutil.rmtree(sample_dir)
                        LOG.info(f"Deleted directory: {sample_dir}")
                    except Exception as e:
                        LOG.error(f"Failed to delete {sample_dir}: {e}")
    
    # Summary
    LOG.info(f"Samples with input images: {len(cleaned_metadata)}")
    LOG.info(f"Samples without input images (removed): {removed_count}")
    
    if dry_run:
        LOG.info("DRY RUN - No changes made")
        return
    
    # Backup original metadata
    backup_path = data_dir / "metadata.json.backup"
    shutil.copy(metadata_path, backup_path)
    LOG.info(f"Backed up original metadata to: {backup_path}")
    
    # Write cleaned metadata
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metadata, f, indent=2, ensure_ascii=False)
    
    LOG.info(f"Wrote cleaned metadata to: {metadata_path}")
    LOG.info("✓ Filtering complete!")


def main():
    parser = argparse.ArgumentParser(description="Filter dataset to remove samples without input images")
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Path to the data directory (default: data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually deleting anything'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        LOG.error(f"Data directory not found: {args.data_dir}")
        return
    
    filter_dataset(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()