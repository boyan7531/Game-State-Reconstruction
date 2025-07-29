#!/usr/bin/env python3
"""
Script to create proper YOLO format dataset from SoccerNet GSR data.
Creates separate train/val/test splits in YOLO format.
"""

from dataset import SoccerNetGSRDataset, YOLODataset
import os
from pathlib import Path
from PIL import Image
import numpy as np

def create_yolo_files_split(dataset: YOLODataset, output_dir: str, split_name: str):
    """Create YOLO format files for a specific split."""
    output_path = Path(output_dir)
    split_dir = output_path / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {split_name} split: {len(dataset)} frames")
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Save image
        img_filename = f"{data['sequence']}_{data['frame_idx']:06d}.jpg"
        img_path = images_dir / img_filename
        
        image_pil = Image.fromarray(data["image"])
        image_pil.save(img_path)
        
        # Save labels
        label_filename = f"{data['sequence']}_{data['frame_idx']:06d}.txt"
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for label in data["labels"]:
                # Format: class_id (int) x_center y_center width height (floats)
                class_id = int(label[0])
                coords = label[1:]
                f.write(f"{class_id} {' '.join(map(str, coords))}\n")
        
        if i % 100 == 0:
            print(f"  Processed {i}/{len(dataset)} frames")
    
    print(f"  {split_name} split completed: {len(dataset)} frames")
    return len(dataset)

def create_dataset_yaml(output_dir: str):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# SoccerNet GSR YOLO Dataset Configuration
# Generated automatically by create_yolo_dataset.py

path: {Path(output_dir).absolute()}  # Dataset root dir
train: train/images  # Train images (relative to 'path')
val: val/images      # Val images (relative to 'path')
test: test/images    # Test images (relative to 'path')

# Classes
nc: 4  # Number of classes
names:
  0: player
  1: goalkeeper
  2: referee
  3: ball
"""
    
    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset configuration saved to {yaml_path}")
    return yaml_path

def main():
    """Main function to create complete YOLO dataset with train/val/test splits."""
    # Configuration
    data_root = "SoccerNet/SN-GSR-2025"
    output_dir = "yolo_dataset_proper"
    
    # Define splits to process
    splits = ["train", "valid", "test"]
    split_mapping = {
        "train": "train",
        "valid": "val",  # Rename 'valid' to 'val' for YOLO convention
        "test": "test"
    }
    
    print("Creating proper YOLO dataset structure...")
    print(f"Output directory: {output_dir}")
    print(f"Processing splits: {splits}")
    print("="*50)
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_frames = 0
    split_stats = {}
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        try:
            # Load base dataset for this split
            base_dataset = SoccerNetGSRDataset(
                data_root=data_root,
                split=split,
                load_images=True
            )
            
            print(f"Loaded {len(base_dataset.sequences)} sequences, {len(base_dataset)} frames")
            
            # Create YOLO dataset wrapper
            yolo_dataset = YOLODataset(base_dataset)
            
            # Get output split name (valid -> val)
            output_split = split_mapping[split]
            
            # Create YOLO files for this split
            frames_count = create_yolo_files_split(yolo_dataset, output_dir, output_split)
            
            split_stats[output_split] = {
                'sequences': len(base_dataset.sequences),
                'frames': frames_count
            }
            total_frames += frames_count
            
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            print(f"Skipping {split} split...")
            continue
    
    # Create dataset.yaml configuration
    print("\nCreating dataset configuration...")
    yaml_path = create_dataset_yaml(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("YOLO DATASET CREATION COMPLETED")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Total frames processed: {total_frames}")
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/     # {split_stats.get('train', {}).get('frames', 0)} files")
    print(f"  │   └── labels/     # {split_stats.get('train', {}).get('frames', 0)} files")
    print(f"  ├── val/")
    print(f"  │   ├── images/     # {split_stats.get('val', {}).get('frames', 0)} files")
    print(f"  │   └── labels/     # {split_stats.get('val', {}).get('frames', 0)} files")
    print(f"  ├── test/")
    print(f"  │   ├── images/     # {split_stats.get('test', {}).get('frames', 0)} files")
    print(f"  │   └── labels/     # {split_stats.get('test', {}).get('frames', 0)} files")
    print(f"  └── dataset.yaml")
    
    print("\nSplit statistics:")
    for split_name, stats in split_stats.items():
        print(f"  {split_name}: {stats['sequences']} sequences, {stats['frames']} frames")
    
    print("\nClass mapping:")
    print("  0: player")
    print("  1: goalkeeper")
    print("  2: referee")
    print("  3: ball")
    
    print("\nNext steps:")
    print(f"1. Update detection.py to use: {output_dir}/dataset.yaml")
    print("2. Run training with proper train/val separation")
    print("3. Use test split for final evaluation")
    
if __name__ == "__main__":
    main()