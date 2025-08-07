#!/usr/bin/env python3
"""
Combined script to download, unzip, and create YOLO dataset from SoccerNet GSR data.
Only processes train and valid splits (no test or challenge).
"""

import os
import zipfile
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from dataset import SoccerNetGSRDataset, YOLODataset

def download_dataset():
    """Download only train and valid splits from SoccerNet GSR dataset."""
    print("Step 1: Downloading SoccerNet GSR dataset (train and valid only)...")
    
    # Only download the files we need
    allow_patterns = [
        "train.zip",
        "valid.zip", 
        "README.md",
        ".gitattributes"
    ]
    
    snapshot_download(
        repo_id="SoccerNet/SN-GSR-2025",
        repo_type="dataset", 
        revision="main",
        local_dir="SoccerNet/SN-GSR-2025",
        allow_patterns=allow_patterns
    )
    print("✓ Dataset downloaded successfully to SoccerNet/SN-GSR-2025")

def unzip_files():
    """Unzip all zip files in the dataset directory."""
    print("\nStep 2: Extracting zip files...")
    base_dir = "SoccerNet/SN-GSR-2025"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        return False
    
    zip_files = [f for f in os.listdir(base_dir) if f.endswith(".zip")]
    if not zip_files:
        print("No zip files found to extract.")
        return True
    
    for filename in zip_files:
        zip_path = os.path.join(base_dir, filename)
        extract_dir = os.path.join(base_dir, filename.replace(".zip", ""))
        print(f"  Extracting {filename}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    print("✓ All zip files extracted successfully")
    return True

def create_yolo_files_split(dataset: YOLODataset, output_dir: str, split_name: str):
    """Create YOLO format files for a specific split."""
    output_path = Path(output_dir)
    split_dir = output_path / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Processing {split_name} split: {len(dataset)} frames")
    
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
            print(f"    Processed {i}/{len(dataset)} frames")
    
    print(f"  ✓ {split_name} split completed: {len(dataset)} frames")
    return len(dataset)

def create_dataset_yaml(output_dir: str):
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# SoccerNet GSR YOLO Dataset Configuration
# Generated automatically by setup_dataset.py

path: {Path(output_dir).absolute()}  # Dataset root dir
train: train/images  # Train images (relative to 'path')
val: val/images      # Val images (relative to 'path')

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
    
    print(f"✓ Dataset configuration saved to {yaml_path}")
    return yaml_path

def create_yolo_dataset():
    """Create YOLO dataset from the downloaded and extracted data."""
    print("\nStep 3: Creating YOLO dataset...")
    
    # Configuration
    data_root = "SoccerNet/SN-GSR-2025"
    output_dir = "yolo_dataset_proper"
    
    # Only process train and valid splits
    splits = ["train", "valid"]
    split_mapping = {
        "train": "train",
        "valid": "val"  # Rename 'valid' to 'val' for YOLO convention
    }
    
    print(f"  Output directory: {output_dir}")
    print(f"  Processing splits: {splits}")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_frames = 0
    split_stats = {}
    
    # Process each split
    for split in splits:
        print(f"\n  Processing {split} split...")
        
        try:
            # Load base dataset for this split
            base_dataset = SoccerNetGSRDataset(
                data_root=data_root,
                split=split,
                load_images=True
            )
            
            print(f"    Loaded {len(base_dataset.sequences)} sequences, {len(base_dataset)} frames")
            
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
            print(f"    Error processing {split} split: {e}")
            print(f"    Skipping {split} split...")
            continue
    
    # Create dataset.yaml configuration
    print("\n  Creating dataset configuration...")
    yaml_path = create_dataset_yaml(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SETUP COMPLETED SUCCESSFULLY!")
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
    print(f"  └── dataset.yaml")
    
    print("\nSplit statistics:")
    for split_name, stats in split_stats.items():
        print(f"  {split_name}: {stats['sequences']} sequences, {stats['frames']} frames")
    
    print("\nClass mapping:")
    print("  0: player")
    print("  1: goalkeeper") 
    print("  2: referee")
    print("  3: ball")
    
    print(f"\nReady to train! Use: {output_dir}/dataset.yaml")

def main():
    """Main function to run the complete dataset setup process."""
    print("SoccerNet GSR Dataset Setup")
    print("Processing only train and valid splits")
    print("="*50)
    
    try:
        # Step 1: Download dataset
        download_dataset()
        
        # Step 2: Unzip files
        if not unzip_files():
            print("Failed to extract files. Exiting.")
            return
        
        # Step 3: Create YOLO dataset
        create_yolo_dataset()
        
        print("\n🎉 All steps completed successfully!")
        print("Your dataset is ready for training.")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()