#!/usr/bin/env python3
"""
Script to analyze the SoccerNet GSR dataset for potential issues that could lead to false predictions.
"""

import os
import yaml
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm


def analyze_dataset_distribution(data_yaml: str):
    """
    Analyze the class distribution and potential issues in the dataset.
    
    Args:
        data_yaml: Path to dataset YAML configuration
    """
    print("📊 Analyzing dataset distribution...")
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Class names
    class_names = {v: k for k, v in data_config['names'].items()}
    print(f"Classes: {class_names}")
    
    # Analyze train and val splits
    for split in ['train', 'val']:
        print(f"\n📁 Analyzing {split} split...")
        split_path = os.path.join(data_config['path'], data_config[split])
        label_path = split_path.replace('images', 'labels')
        
        # Counters for analysis
        class_counts = Counter()
        bbox_sizes = defaultdict(list)  # per class
        bbox_positions = defaultdict(list)  # per class
        aspect_ratios = defaultdict(list)  # per class
        
        # Get label files
        label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
        
        print(f"Processing {len(label_files)} samples...")
        for label_file in tqdm(label_files, desc=f"Analyzing {split}"):
            label_path_full = os.path.join(label_path, label_file)
            
            # Read labels
            with open(label_path_full, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Update counters
                        class_counts[class_id] += 1
                        
                        # Store bbox properties
                        bbox_sizes[class_id].append(width * height)  # relative area
                        bbox_positions[class_id].append((x_center, y_center))
                        if height > 0:
                            aspect_ratios[class_id].append(width / height)
        
        # Print results
        print(f"\n{split.upper()} Split Statistics:")
        print("-" * 30)
        total_objects = sum(class_counts.values())
        for class_id, count in class_counts.items():
            percentage = (count / total_objects) * 100
            print(f"  {class_names[class_id]} ({class_id}): {count} ({percentage:.1f}%)")
        
        # Size and position analysis
        print(f"\nObject Size Analysis:")
        for class_id, sizes in bbox_sizes.items():
            if sizes:
                avg_size = np.mean(sizes)
                std_size = np.std(sizes)
                print(f"  {class_names[class_id]}: avg={avg_size:.4f}, std={std_size:.4f}")
        
        print(f"\nAspect Ratio Analysis:")
        for class_id, ratios in aspect_ratios.items():
            if ratios:
                avg_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)
                print(f"  {class_names[class_id]}: avg={avg_ratio:.2f}, std={std_ratio:.2f}")


def find_potential_mislabeled_examples(data_yaml: str, sample_size: int = 1000):
    """
    Find potential mislabeled examples by analyzing unusual patterns.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        sample_size: Number of samples to analyze
    """
    print("\n🔍 Searching for potential mislabeled examples...")
    
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Class names
    class_names = {v: k for k, v in data_config['names'].items()}
    
    # Analyze train split
    split = 'train'
    split_path = os.path.join(data_config['path'], data_config[split])
    label_path = split_path.replace('images', 'labels')
    
    # Suspicious patterns to track
    suspicious_cases = {
        'ball_in_crowd': [],  # Balls in areas with many people
        'goalkeeper_like_objects': [],  # Objects mislabeled as goalkeepers
        'extreme_sizes': [],  # Objects with unusual sizes
        'wrong_aspect_ratios': [],  # Objects with wrong aspect ratios
    }
    
    # Get label files
    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    sample_files = label_files[:sample_size] if len(label_files) > sample_size else label_files
    
    print(f"Analyzing {len(sample_files)} samples for suspicious patterns...")
    for label_file in tqdm(sample_files, desc="Finding suspicious cases"):
        label_path_full = os.path.join(label_path, label_file)
        
        # Read labels
        labels = []
        with open(label_path_full, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append((class_id, x_center, y_center, width, height))
        
        # Analyze for suspicious patterns
        for i, (class_id, x_center, y_center, width, height) in enumerate(labels):
            # Check for extreme sizes
            area = width * height
            if class_id == 3 and area < 0.0001:  # Very small ball
                suspicious_cases['extreme_sizes'].append((label_file, 'ball', area))
            elif class_id == 3 and area > 0.1:  # Very large ball
                suspicious_cases['extreme_sizes'].append((label_file, 'ball', area))
            
            # Check for wrong aspect ratios
            if height > 0:
                ratio = width / height
                if class_id == 3 and (ratio < 0.5 or ratio > 2.0):  # Balls should be roughly circular
                    suspicious_cases['wrong_aspect_ratios'].append((label_file, 'ball', ratio))
            
            # Check if ball is in crowded area
            if class_id == 3:
                nearby_objects = 0
                for j, (other_class_id, other_x, other_y, other_w, other_h) in enumerate(labels):
                    if i != j:  # Don't compare with itself
                        # Calculate distance between centers
                        distance = np.sqrt((x_center - other_x)**2 + (y_center - other_y)**2)
                        # If another object is relatively close
                        if distance < max(width, height) * 2:
                            nearby_objects += 1
                
                if nearby_objects > 5:  # Arbitrarily defined "crowded"
                    suspicious_cases['ball_in_crowd'].append((label_file, nearby_objects))
            
            # Check for goalkeeper-like positioning (behind goal)
            if class_id == 1:  # goalkeeper
                # Goalkeepers are usually near the edges of the frame
                if x_center > 0.1 and x_center < 0.9 and y_center > 0.1 and y_center < 0.9:
                    # Not at edge - might be mislabeled
                    suspicious_cases['goalkeeper_like_objects'].append((label_file, x_center, y_center))
    
    # Print results
    print("\nSuspicious Cases Found:")
    print("-" * 25)
    for case_type, cases in suspicious_cases.items():
        print(f"{case_type}: {len(cases)} cases")
        # Show some examples
        if cases:
            print("  Examples:")
            for case in cases[:5]:  # Show first 5
                print(f"    {case}")
            if len(cases) > 5:
                print(f"    ... and {len(cases) - 5} more")


def main():
    """Main function to run dataset analysis."""
    data_dir = "yolo_dataset_proper"
    dataset_yaml = os.path.join(data_dir, "dataset.yaml")
    
    if not os.path.exists(dataset_yaml):
        print(f"❌ Dataset configuration not found: {dataset_yaml}")
        print("Please run create_yolo_dataset.py first.")
        return
    
    # Analyze dataset distribution
    analyze_dataset_distribution(dataset_yaml)
    
    # Find potential mislabeled examples
    find_potential_mislabeled_examples(dataset_yaml)


if __name__ == "__main__":
    main()