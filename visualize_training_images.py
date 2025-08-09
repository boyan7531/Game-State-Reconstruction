#!/usr/bin/env python3
"""
Training Image Visualization Script for SoccerNet GSR Dataset
Shows how original 1920x1080 images are transformed during actual YOLO training.
Includes real YOLO preprocessing and augmentations.
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics.data.augment import Compose, LetterBox, RandomHSV, RandomFlip, RandomPerspective
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER

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

def apply_copy_paste_augmentation(image, annotations, imgsz=1280):
    """
    Apply copy-paste augmentation that copies objects from one part of the image to another.
    This is a powerful data augmentation technique used in YOLO training.
    """
    augmented_image = image.copy()
    augmented_annotations = annotations.copy() if annotations else []
    
    # Only apply copy-paste if we have annotations
    if not annotations or len(annotations) < 2:
        return augmented_image, augmented_annotations, "Copy-Paste (No objects to copy)"
    
    # Apply copy-paste with 100% probability for demonstration
    if random.random() < 1.0:
        h, w = image.shape[:2]
        
        # Prefer to copy ball objects (class_id 3) for demonstration
        ball_annotations = [ann for ann in annotations if ann[0] == 3]  # class_id 3 is ball
        if ball_annotations:
            source_ann = random.choice(ball_annotations)
        else:
            source_ann = random.choice(annotations)
        class_id, x_center, y_center, width, height = source_ann
        
        # Convert to pixel coordinates
        x_center_px = int(x_center * w)
        y_center_px = int(y_center * h)
        width_px = int(width * w)
        height_px = int(height * h)
        
        # Calculate bounding box
        x1 = max(0, x_center_px - width_px // 2)
        y1 = max(0, y_center_px - height_px // 2)
        x2 = min(w, x_center_px + width_px // 2)
        y2 = min(h, y_center_px + height_px // 2)
        
        # Extract the object region
        if x2 > x1 and y2 > y1:
            object_region = image[y1:y2, x1:x2].copy()
            
            # Find a random location to paste (avoid overlapping too much)
            max_attempts = 10
            for _ in range(max_attempts):
                # Random paste location
                paste_x = random.randint(width_px // 2, w - width_px // 2)
                paste_y = random.randint(height_px // 2, h - height_px // 2)
                
                # Calculate paste bounding box
                paste_x1 = paste_x - width_px // 2
                paste_y1 = paste_y - height_px // 2
                paste_x2 = paste_x1 + (x2 - x1)
                paste_y2 = paste_y1 + (y2 - y1)
                
                # Ensure paste location is within image bounds
                if paste_x2 <= w and paste_y2 <= h and paste_x1 >= 0 and paste_y1 >= 0:
                    # Paste the object
                    augmented_image[paste_y1:paste_y2, paste_x1:paste_x2] = object_region
                    
                    # Add new annotation for the pasted object
                    new_x_center = paste_x / w
                    new_y_center = paste_y / h
                    augmented_annotations.append((class_id, new_x_center, new_y_center, width, height))
                    break
    
    return augmented_image, augmented_annotations, "Copy-Paste Augmentation"

def apply_manual_augmentations(image, annotations, imgsz=1280):
    """
    Apply manual augmentations that simulate YOLO training transformations.
    This shows how images really look during training.
    """
    # Start with letterbox resize
    letterboxed, scale, (pad_x, pad_y) = letterbox_resize(image, imgsz)
    
    # Transform annotations for letterboxed image
    letterbox_annotations = []
    if annotations:
        h, w = image.shape[:2]
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Convert to pixel coordinates in original image
            orig_x_center = x_center * w
            orig_y_center = y_center * h
            orig_width = width * w
            orig_height = height * h
            
            # Apply scale and padding
            new_x_center = orig_x_center * scale + pad_x
            new_y_center = orig_y_center * scale + pad_y
            new_width = orig_width * scale
            new_height = orig_height * scale
            
            # Convert back to normalized coordinates
            norm_x_center = new_x_center / imgsz
            norm_y_center = new_y_center / imgsz
            norm_width = new_width / imgsz
            norm_height = new_height / imgsz
            
            letterbox_annotations.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))
    
    # Apply copy-paste augmentation first (before letterboxing)
    copy_paste_image, copy_paste_annotations, _ = apply_copy_paste_augmentation(image, annotations, imgsz)
    
    # Re-apply letterbox to copy-paste result
    letterboxed, scale, (pad_x, pad_y) = letterbox_resize(copy_paste_image, imgsz)
    
    # Transform copy-paste annotations for letterboxed image
    letterbox_annotations = []
    if copy_paste_annotations:
        h, w = copy_paste_image.shape[:2]
        for ann in copy_paste_annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Convert to pixel coordinates in original image
            orig_x_center = x_center * w
            orig_y_center = y_center * h
            orig_width = width * w
            orig_height = height * h
            
            # Apply scale and padding
            new_x_center = orig_x_center * scale + pad_x
            new_y_center = orig_y_center * scale + pad_y
            new_width = orig_width * scale
            new_height = orig_height * scale
            
            # Convert back to normalized coordinates
            norm_x_center = new_x_center / imgsz
            norm_y_center = new_y_center / imgsz
            norm_width = new_width / imgsz
            norm_height = new_height / imgsz
            
            letterbox_annotations.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))
    
    # Apply random augmentations manually
    augmented_image = letterboxed.copy()
    augmented_annotations = letterbox_annotations.copy()
    
    # Random HSV adjustments (matching training settings)
    if random.random() < 0.8:  # Apply HSV augmentation 80% of the time
        hsv = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Hue shift (±1.5%)
        h_gain = random.uniform(-0.015, 0.015)
        hsv[:, :, 0] = (hsv[:, :, 0] * (1 + h_gain)) % 180
        
        # Saturation adjustment (±50%)
        s_gain = random.uniform(-0.5, 0.5)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + s_gain)
        
        # Value/brightness adjustment (±30%)
        v_gain = random.uniform(-0.3, 0.3)
        hsv[:, :, 2] = hsv[:, :, 2] * (1 + v_gain)
        
        # Clip values and convert back
        hsv = np.clip(hsv, 0, 255)
        augmented_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Random horizontal flip (50% chance)
    if random.random() < 0.5:
        augmented_image = cv2.flip(augmented_image, 1)  # Horizontal flip
        
        # Flip annotations too
        flipped_annotations = []
        for ann in augmented_annotations:
            class_id, x_center, y_center, width, height = ann
            # Flip x coordinate
            flipped_x_center = 1.0 - x_center
            flipped_annotations.append((class_id, flipped_x_center, y_center, width, height))
        augmented_annotations = flipped_annotations
    
    aug_method = "Copy-Paste + HSV + Flip Augmentation"
    return augmented_image, augmented_annotations, aug_method

def apply_yolo_training_augmentations(image, annotations, imgsz=1280):
    """
    Apply training augmentations - fallback to manual implementation.
    """
    return apply_manual_augmentations(image, annotations, imgsz)

def visualize_training_samples(dataset_path="yolo_dataset_proper", num_samples=4):
    """Visualize how original images are transformed during actual YOLO training."""
    
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
    print(f"🔄 Showing: Original → Letterbox → YOLO Training Augmented")
    
    # Select random samples
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create subplot grid - 3 columns (original, letterbox, training augmented)
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
        
        # Method 1: Basic letterbox (what YOLO does first)
        letterbox_image, scale, (pad_x, pad_y) = letterbox_resize(image_rgb, 1280)
        letterbox_with_boxes = letterbox_image.copy()
        letterbox_annotations = []
        if annotations:
            # Transform annotations for letterbox image
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
                
                letterbox_annotations.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height))
            
            letterbox_with_boxes = draw_bounding_boxes(letterbox_with_boxes, letterbox_annotations)
        
        # Method 2: YOLO training augmentations (how images actually look during training)
        augmented_image, augmented_annotations, aug_method = apply_yolo_training_augmentations(image_rgb, annotations, 1280)
        augmented_with_boxes = augmented_image.copy()
        if augmented_annotations:
            augmented_with_boxes = draw_bounding_boxes(augmented_with_boxes, augmented_annotations)
        
        # Display original image
        axes[idx, 0].imshow(original_with_boxes)
        axes[idx, 0].set_title(f"Original: {original_shape[1]}x{original_shape[0]}\n{len(annotations)} objects", 
                              fontsize=10)
        axes[idx, 0].axis('off')
        
        # Display letterbox processed image
        axes[idx, 1].imshow(letterbox_with_boxes)
        axes[idx, 1].set_title(f"Letterbox: 1280x1280\nScale: {scale:.3f}, Padding: ({pad_x}, {pad_y})", 
                              fontsize=10)
        axes[idx, 1].axis('off')
        
        # Display training augmented image
        axes[idx, 2].imshow(augmented_with_boxes)
        axes[idx, 2].set_title(f"Training Augmented: 1280x1280\n{aug_method}\n{len(augmented_annotations)} objects", 
                              fontsize=9)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle("SoccerNet GSR: How Images Look During YOLO Training", 
                fontsize=16, y=0.98)
    
    # Save the visualization
    output_path = "yolo_training_visualization.png"
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

def visualize_augmentation_variations(dataset_path="yolo_dataset_proper", num_variations=5):
    """Show multiple augmentation variations of the same image to see training diversity."""
    
    train_images_path = os.path.join(dataset_path, "train", "images")
    train_labels_path = os.path.join(dataset_path, "train", "labels")
    
    if not os.path.exists(train_images_path):
        print(f"❌ Training images directory not found: {train_images_path}")
        return
    
    # Get a random image
    image_files = [f for f in os.listdir(train_images_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"❌ No image files found in {train_images_path}")
        return
    
    # Pick one image to show variations
    image_file = random.choice(image_files)
    image_path = os.path.join(train_images_path, image_file)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load annotations
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(train_labels_path, label_file)
    annotations = load_yolo_annotations(label_path)
    
    print(f"\n🎲 Showing {num_variations} augmentation variations of: {image_file}")
    
    # Create subplot grid
    fig, axes = plt.subplots(1, num_variations + 1, figsize=(4 * (num_variations + 1), 4))
    
    # Show original
    original_with_boxes = draw_bounding_boxes(image_rgb.copy(), annotations) if annotations else image_rgb
    axes[0].imshow(original_with_boxes)
    axes[0].set_title(f"Original\n{len(annotations)} objects", fontsize=10)
    axes[0].axis('off')
    
    # Show variations
    for i in range(num_variations):
        aug_image, aug_annotations, aug_method = apply_yolo_training_augmentations(image_rgb, annotations, 1280)
        aug_with_boxes = draw_bounding_boxes(aug_image.copy(), aug_annotations) if aug_annotations else aug_image
        
        axes[i + 1].imshow(aug_with_boxes)
        axes[i + 1].set_title(f"Variation {i+1}\n{len(aug_annotations)} objects", fontsize=10)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Training Augmentation Variations: {image_file}", fontsize=14, y=0.98)
    
    # Save the visualization
    output_path = "augmentation_variations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Augmentation variations saved to: {output_path}")
    
    plt.show()

def main():
    """Main visualization function."""
    print("🏈 SoccerNet GSR Training Image Visualizer")
    print("=" * 50)
    print("🔄 Showing how images are transformed during actual YOLO training")
    
    # Show dataset statistics
    show_dataset_stats()
    
    # Visualize sample images with training transformations
    print(f"\n🖼️  YOLO TRAINING PIPELINE VISUALIZATION")
    print("=" * 50)
    visualize_training_samples(num_samples=3)
    
    # Show augmentation variations
    print(f"\n🎲 AUGMENTATION VARIATIONS")
    print("=" * 50)
    visualize_augmentation_variations(num_variations=4)
    
    print("\n✅ Visualization complete!")
    print("\nLegend:")
    print("  🟢 Green boxes: Players")
    print("  🔵 Blue boxes: Goalkeepers") 
    print("  🟡 Yellow boxes: Referees")
    print("  🔵 Cyan boxes: Ball")
    print("\nKey insights:")
    print("  • Original: 1920x1080 (16:9 aspect ratio)")
    print("  • Letterbox: 1280x1280 with gray padding (YOLO standard)")
    print("  • Training: Same size but with HSV, flip, and other augmentations")
    print("  • Augmentations help the model generalize to different lighting/conditions")
    print("  • Each training epoch sees slightly different versions of the same image!")

if __name__ == "__main__":
    main()