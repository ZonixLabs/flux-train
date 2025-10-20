#!/usr/bin/env python3
"""
Script to transform dataset by stitching context frames vertically.
Takes 2.jpg, 3.jpg, 4.jpg from each sample and stitches them into a single 2.jpg.
Writes to a new directory to preserve the original.
"""

import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from typing import List

def stitch_images_vertically(images: List[np.ndarray]) -> np.ndarray:
    """Stitch images vertically."""
    return np.vstack(images)

def process_sample(source_dir: Path, dest_dir: Path) -> int:
    """
    Process a single sample directory.
    Copies files to dest_dir and stitches context frames.
    Returns number of context frames stitched (0 if none).
    """
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy 1.jpg (characters) and out.jpg as-is
    for filename in ['1.jpg', 'out.jpg']:
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, dest_dir / filename)
    
    # Check for context frames (2.jpg, 3.jpg, 4.jpg)
    context_frames = []
    
    for i in range(2, 5):  # 2, 3, 4
        frame_path = source_dir / f"{i}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                context_frames.append(img)
    
    # If no context frames, nothing to do
    if len(context_frames) == 0:
        return 0
    
    # If only one context frame, just copy it
    if len(context_frames) == 1:
        shutil.copy2(source_dir / "2.jpg", dest_dir / "2.jpg")
        return 1
    
    # Stitch vertically
    stitched = stitch_images_vertically(context_frames)
    
    # Save as 2.jpg in destination
    output_path = dest_dir / "2.jpg"
    cv2.imwrite(str(output_path), stitched, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return len(context_frames)

def update_metadata(metadata_path: Path):
    """Update metadata.json to reflect new structure."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    updated_count = 0
    
    for entry in metadata:
        edit_images = entry.get('edit_image', [])
        
        # Filter out 3.jpg and 4.jpg references
        new_edit_images = []
        for img_path in edit_images:
            # Keep 1.jpg and 2.jpg, remove 3.jpg and 4.jpg
            if not (img_path.endswith('/3.jpg') or img_path.endswith('/4.jpg')):
                new_edit_images.append(img_path)
        
        if len(new_edit_images) != len(edit_images):
            entry['edit_image'] = new_edit_images
            updated_count += 1
    
    # Write back
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return updated_count

def main():
    input_dir = Path("dataset")
    output_dir = Path("data_stitched")
    input_metadata_path = input_dir / "metadata.json"
    output_metadata_path = output_dir / "metadata.json"
    
    if not input_dir.exists():
        print(f"Error: {input_dir} not found")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find all sample directories
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    print(f"Found {len(sample_dirs)} sample directories")
    print("Processing samples...")
    
    total_stitched = 0
    samples_modified = 0
    
    for sample_dir in sample_dirs:
        dest_sample_dir = output_dir / sample_dir.name
        num_frames = process_sample(sample_dir, dest_sample_dir)
        if num_frames > 0:
            total_stitched += num_frames
            samples_modified += 1
            if samples_modified % 100 == 0:
                print(f"  Processed {samples_modified} samples...")
    
    print(f"\n✓ Stitched {total_stitched} total context frames across {samples_modified} samples")
    
    # Update and copy metadata
    if input_metadata_path.exists():
        print("\nUpdating metadata.json...")
        updated = update_metadata(input_metadata_path, output_metadata_path)
        print(f"✓ Updated {updated} entries in metadata.json")
    else:
        print(f"\nWarning: {input_metadata_path} not found, skipping metadata update")
    
    print(f"\n✓ Dataset transformation complete!")
    print(f"Original dataset preserved in: {input_dir}")
    print(f"New dataset created in: {output_dir}")

if __name__ == "__main__":
    main()