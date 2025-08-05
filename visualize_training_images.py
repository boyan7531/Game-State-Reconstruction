#!/usr/bin/env python3
"""
Training Image Visualization Script for SoccerNet GSR Dataset
Shows how original 1920x1080 images are transformed to 1280x1280 for training.
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_yolo_annotations(label_path):
    """Load YOLO format annotations from a label file."""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def yolo_to_bbox(annotation, img_width, img_height):
    """Convert YOLO format to bounding box coordinates."""
    class_id, x_center, y_center, width, height = annotation
    
    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate top-left and bottom-right coordinates
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return class_id, (x1, y1, x2, y2)

def draw_bounding_boxes(image, annotations):
    """Draw bounding boxes on the image."""
    # Class names and colors
    class_names = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
    colors = {
        0: (0, 255, 0),    # Green for player
        1: (255, 0, 0),    # Blue for goalkeeper  
        2: (0, 255, 255),  # Yellow for referee
        3: (255, 255, 0)   # Cyan for ball
    }
    
    img_height, img_width = image.shape[:2]
    
    for annotation in annotations:
        class_id, bbox = yolo_to_bbox(annotation, img_width, img_height)
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        color = colors.get(class_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = class_names.get(class_id, f'class_{class_id}')
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Background for text
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def resize_16_9(image, target_width=1280, target_height=720):
    """
    Resize image to 16:9 aspect ratio (1280x720) - no padding needed!
    Direct resize since original is already 16:9.
    """
    h, w = image.shape[:2]
    
    # For 1920x1080 -> 1280x720, this is a simple scale down
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate scale factors for annotation transformation
    scale_x = target_width / w
    scale_y = target_height / h
    
    return resized, scale_x, scale_y

def letterbox_resize(image, target_size=1280):
    """
    Resize image to target_size x target_size with letterboxing (original YOLO preprocessing).
    Maintains aspect ratio and pads with gray.
    """
    h, w = image.shape[:2]
    
    # Calculate scale factor
    scale = min(target_size / w, target_size / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # Gray padding
    
    # Calculate padding offsets
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    # Place resized image in center
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded, scale, (pad_x, pad_y)

def visualize_training_samples(dataset_path="yolo_dataset_proper", num_samples=4):
    """Visualize how original images are transformed for training - comparing square vs 16:9."""
    
    train_images_path = os.path.join(dataset_path, "train", "images")
    train_labels_path = os.path.join(dataset_path, "train", "labels")
    
    if not os.path.exists(train_images_path):
        print(f"❌ Training images directory not found: {train_images_path}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(train_images_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"❌ No image files found in {train_images_path}")
        return
    
    print(f"📊 Found {len(image_files)} training images")
    print(f"🎯 Showing {min(num_samples, len(image_files))} random samples")
    print(f"📐 Comparing: 1280x1280 (square + padding) vs 1280x720 (16:9, no padding)")
    
    # Select random samples
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create subplot grid - 3 columns (original, square, 16:9)
    fig, axes = plt.subplots(len(sample_files), 3, figsize=(20, 4 * len(sample_files)))
    
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_file in enumerate(sample_files):
        # Load image
        image_path = os.path.join(train_images_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Could not load image: {image_file}")
            continue
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape
        
        print(f"📸 {image_file}: {original_shape[1]}x{original_shape[0]} pixels")
        
        # Load corresponding annotations
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(train_labels_path, label_file)
        annotations = load_yolo_annotations(label_path)
        
        # Count objects by class
        class_counts = {}
        for ann in annotations:
            class_id = ann[0]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        class_names = {0: 'players', 1: 'goalkeepers', 2: 'referees', 3: 'balls'}
        count_str = ", ".join([f"{class_counts.get(i, 0)} {class_names[i]}" for i in range(4)])
        print(f"   🏷️  {count_str}")
        
        # Draw bounding boxes on original
        if annotations:
            original_with_boxes = draw_bounding_boxes(image_rgb.copy(), annotations)
        else:
            original_with_boxes = image_rgb
        
        # Method 1: Square with letterboxing (1280x1280)
        square_image, scale, (pad_x, pad_y) = letterbox_resize(image_rgb, 1280)
        square_with_boxes = square_image.copy()
        if annotations:
            # Transform annotations for square image
            square_annotations = []
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                
                # Convert to pixel coordinates in original image
                orig_x_center = x_center * original_shape[1]
                orig_y_center = y_center * original_shape[0]
                orig_width = width * original_shape[1]
                orig_height = height * original_shape[0]
                
                # Apply scale and padding
                new_x_center = orig_x_center * scale + pad_x
                new_y_center = orig_y_center * scale + pad_y
                new_width = orig_width * scale
                new_height = orig_height * scale
                
                # Convert back to normalized coordinates for the 1280x1280 image
                norm_x_center = new_x_center / 1280
                norm_y_center = new_y_center / 1280
                norm_width = new_width / 1280
                norm_height = new_height / 1280
                
                square_annotations.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))
            
            square_with_boxes = draw_bounding_boxes(square_with_boxes, square_annotations)
        
        # Method 2: 16:9 resize (1280x720) - no padding!
        ratio_image, scale_x, scale_y = resize_16_9(image_rgb, 1280, 720)
        ratio_with_boxes = ratio_image.copy()
        if annotations:
            # Transform annotations for 16:9 image (simple scaling)
            ratio_annotations = []
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                # No coordinate transformation needed - just use original normalized coords!
                ratio_annotations.append((class_id, x_center, y_center, width, height))
            
            ratio_with_boxes = draw_bounding_boxes(ratio_with_boxes, ratio_annotations)
        
        # Display original image
        axes[idx, 0].imshow(original_with_boxes)
        axes[idx, 0].set_title(f"Original: {original_shape[1]}x{original_shape[0]}\n{len(annotations)} objects", 
                              fontsize=10)
        axes[idx, 0].axis('off')
        
        # Display square processed image
        axes[idx, 1].imshow(square_with_boxes)
        axes[idx, 1].set_title(f"Square: 1280x1280 (letterboxed)\nScale: {scale:.3f}, Padding: ({pad_x}, {pad_y})", 
                              fontsize=10)
        axes[idx, 1].axis('off')
        
        # Display 16:9 processed image
        axes[idx, 2].imshow(ratio_with_boxes)
        axes[idx, 2].set_title(f"16:9: 1280x720 (no padding)\nScale: {scale_x:.3f}x, {scale_y:.3f}y", 
                              fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle("SoccerNet GSR: Square vs 16:9 Training Resolution Comparison", 
                fontsize=16, y=0.98)
    
    # Save the visualization
    output_path = "square_vs_16_9_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_path}")
    
    plt.show()

def show_dataset_stats(dataset_path="yolo_dataset_proper"):
    """Show dataset statistics."""
    print("\n📊 DATASET STATISTICS")
    print("=" * 50)
    
    splits = ['train', 'val', 'test']
    total_images = 0
    total_annotations = 0
    
    class_names = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
    global_class_counts = {i: 0 for i in range(4)}
    
    for split in splits:
        images_path = os.path.join(dataset_path, split, "images")
        labels_path = os.path.join(dataset_path, split, "labels")
        
        if not os.path.exists(images_path):
            continue
        
        image_files = [f for f in os.listdir(images_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        split_annotations = 0
        split_class_counts = {i: 0 for i in range(4)}
        
        for image_file in image_files:
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_path, label_file)
            annotations = load_yolo_annotations(label_path)
            
            split_annotations += len(annotations)
            
            for ann in annotations:
                class_id = ann[0]
                split_class_counts[class_id] += 1
                global_class_counts[class_id] += 1
        
        print(f"{split.upper():>5}: {len(image_files):>6} images, {split_annotations:>7} annotations")
        
        # Show class distribution for this split
        for class_id, count in split_class_counts.items():
            if count > 0:
                print(f"       {class_names[class_id]:>10}: {count:>6}")
        
        total_images += len(image_files)
        total_annotations += split_annotations
    
    print("-" * 50)
    print(f"{'TOTAL':>5}: {total_images:>6} images, {total_annotations:>7} annotations")
    print(f"\nGlobal class distribution:")
    for class_id, count in global_class_counts.items():
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"  {class_names[class_id]:>10}: {count:>6} ({percentage:>5.1f}%)")

def main():
    """Main visualization function."""
    print("🏈 SoccerNet GSR Training Image Visualizer")
    print("=" * 50)
    print("📐 Comparing square (1280x1280) vs 16:9 (1280x720) training formats")
    
    # Show dataset statistics
    show_dataset_stats()
    
    # Visualize sample images
    print(f"\n🖼️  SQUARE vs 16:9 COMPARISON")
    print("=" * 50)
    visualize_training_samples(num_samples=3)
    
    print("\n✅ Visualization complete!")
    print("\nLegend:")
    print("  🟢 Green boxes: Players")
    print("  🔵 Blue boxes: Goalkeepers") 
    print("  🟡 Yellow boxes: Referees")
    print("  🔵 Cyan boxes: Ball")
    print("\nKey insights:")
    print("  • Original: 1920x1080 (16:9 aspect ratio)")
    print("  • Square: 1280x1280 with gray padding (wastes ~22% of pixels)")
    print("  • 16:9: 1280x720 with no padding (uses 100% of pixels)")
    print("  • 16:9 format eliminates gray bars and matches natural soccer footage")
    print("  • For soccer-only datasets, 16:9 is likely more efficient!")

if __name__ == "__main__":
    main()