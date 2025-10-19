#!/usr/bin/env python3
"""
Script to append series name to each prompt in the training dataset.
Adds ". {series name} anime style" to the end of each prompt.
"""

import json
import csv
from pathlib import Path

def load_series_tracking(csv_path: str) -> dict:
    """Load series tracking CSV and return sample -> series mapping."""
    sample_to_series = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = row['sample']
            series = row['series']
            sample_to_series[sample] = series
    
    return sample_to_series

def format_series_name(series: str) -> str:
    """Format series name by replacing underscores with spaces."""
    return series.replace('_', ' ')

def append_series_to_prompts(metadata_path: str, series_tracking_path: str, output_path: str = None):
    """Append series name to each prompt in metadata.json."""
    
    # Load series tracking
    print(f"Loading series tracking from {series_tracking_path}...")
    sample_to_series = load_series_tracking(series_tracking_path)
    print(f"Loaded {len(sample_to_series)} sample mappings")
    
    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} samples")
    
    # Update prompts
    updated_count = 0
    missing_count = 0
    
    for entry in metadata:
        # Extract sample name from image path (e.g., "sample_001/out.jpg" -> "sample_001")
        image_path = entry['image']
        sample_name = image_path.split('/')[0]
        
        if sample_name in sample_to_series:
            series = sample_to_series[sample_name]
            series_formatted = format_series_name(series)
            
            # Append series to prompt
            original_prompt = entry['prompt']
            entry['prompt'] = f"{original_prompt}. Style: Anime ({series_formatted} series)"
            
            updated_count += 1
        else:
            print(f"Warning: No series found for {sample_name}")
            missing_count += 1
    
    # Write updated metadata
    if output_path is None:
        output_path = metadata_path
    
    print(f"Writing updated metadata to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Complete!")
    print(f"  Updated: {updated_count} samples")
    print(f"  Missing: {missing_count} samples")

def main():
    metadata_path = "dataset/metadata.json"
    series_tracking_path = "series_tracking.csv"
    
    # You can optionally specify a different output path to preserve the original
    # output_path = "data/metadata_with_series.json"
    output_path = None # "dataset/metadata_with_series.json"
    
    append_series_to_prompts(metadata_path, series_tracking_path, output_path)

if __name__ == "__main__":
    main()