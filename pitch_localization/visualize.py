#!/usr/bin/env python3
"""
Visualization script for pitch localization training data.
Shows original images alongside their ground truth masks with different line thicknesses.
"""

import argparse
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataset import create_train_dataset
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_training_samples(
    data_root: str,
    output_dir: str = "visualization_output",
    num_samples: int = 5,
    line_thicknesses: list = [2, 5, 8],
    target_size: tuple = (512, 512),
    enable_augmentation: bool = False
):
    """
    Visualize training samples with different line thicknesses.
    
    Args:
        data_root: Path to training data root
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
        line_thicknesses: List of line thicknesses to compare
        target_size: Target image size for visualization
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"Saving visualizations to {output_path}")
    
    # Create datasets with different line thicknesses (NO AUGMENTATION for clear visualization)
    datasets = {}
    for thickness in line_thicknesses:
        logger.info(f"Creating dataset with line thickness {thickness}")
        datasets[thickness] = create_train_dataset(
            data_root,
            target_size=target_size,
            line_thickness=thickness,
            cache_masks=False,
            rotation_range=0.0 if not enable_augmentation else 10.0,  # Conditional rotation
            horizontal_flip_prob=0.0 if not enable_augmentation else 0.5,  # Conditional flips
            brightness_range=0.0 if not enable_augmentation else 0.15,  # Conditional color augmentation
            contrast_range=0.0 if not enable_augmentation else 0.15,
            saturation_range=0.0 if not enable_augmentation else 0.15
        )
    
    # Get the same samples from each dataset
    sample_indices = np.random.choice(len(datasets[line_thicknesses[0]]), 
                                    min(num_samples, len(datasets[line_thicknesses[0]])), 
                                    replace=False)
    
    logger.info(f"Visualizing {len(sample_indices)} samples with line thicknesses: {line_thicknesses}")
    
    # Create comparison visualizations
    for i, sample_idx in enumerate(sample_indices):
        fig, axes = plt.subplots(2, len(line_thicknesses) + 1, figsize=(20, 10))
        fig.suptitle(f'Training Sample {i+1} - Line Thickness Comparison', fontsize=16)
        
        # Get original image (same across all thicknesses)
        sample = datasets[line_thicknesses[0]][sample_idx]
        original_image = sample['image']
        
        # Convert tensor to numpy for visualization
        if torch.is_tensor(original_image):
            # Denormalize the image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original_image = original_image * std + mean
            original_image = torch.clamp(original_image, 0, 1)
            original_image = original_image.permute(1, 2, 0).numpy()
        
        # Show original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(original_image)
        axes[1, 0].set_title('Original Image', fontsize=12) 
        axes[1, 0].axis('off')
        
        # Show masks with different line thicknesses
        for j, thickness in enumerate(line_thicknesses):
            sample = datasets[thickness][sample_idx]
            mask = sample['mask']
            
            # Convert tensor to numpy
            if torch.is_tensor(mask):
                mask = mask.squeeze().numpy()
            
            # Show mask alone
            axes[0, j + 1].imshow(mask, cmap='gray')
            axes[0, j + 1].set_title(f'Mask (thickness={thickness}px)', fontsize=12)
            axes[0, j + 1].axis('off')
            
            # Show overlay
            overlay = original_image.copy()
            mask_colored = np.zeros_like(original_image)
            mask_colored[:, :, 1] = mask  # Green channel for mask
            overlay = cv2.addWeighted((overlay * 255).astype(np.uint8), 0.7, 
                                    (mask_colored * 255).astype(np.uint8), 0.3, 0)
            overlay = overlay.astype(np.float32) / 255.0
            
            axes[1, j + 1].imshow(overlay)
            axes[1, j + 1].set_title(f'Overlay (thickness={thickness}px)', fontsize=12)
            axes[1, j + 1].axis('off')
        
        plt.tight_layout()
        
        # Save the comparison
        output_file = output_path / f'training_sample_{i+1}_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_file}")
        plt.close()
    
    # Create a detailed analysis of a single sample
    logger.info("Creating detailed analysis of line thickness effects...")
    create_detailed_analysis(datasets, line_thicknesses, output_path, sample_indices[0])
    
    logger.info(f"Visualization complete! Check {output_path} for results.")


def create_detailed_analysis(datasets, line_thicknesses, output_path, sample_idx):
    """Create detailed analysis showing pixel-level differences in line thickness."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Line Thickness Analysis - Pixel Level Comparison', fontsize=16)
    
    # Get sample data
    sample = datasets[line_thicknesses[0]][sample_idx]
    original_image = sample['image']
    
    # Denormalize for display
    if torch.is_tensor(original_image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original_image = original_image * std + mean
        original_image = torch.clamp(original_image, 0, 1)
        original_image = original_image.permute(1, 2, 0).numpy()
    
    # Show original
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Show individual masks
    masks = []
    for i, thickness in enumerate(line_thicknesses):
        sample = datasets[thickness][sample_idx]
        mask = sample['mask']
        
        if torch.is_tensor(mask):
            mask = mask.squeeze().numpy()
        
        masks.append(mask)
        
        if i < 2:  # Show first two masks individually
            axes[0, i + 1].imshow(mask, cmap='hot')
            axes[0, i + 1].set_title(f'Thickness {thickness}px\n({np.sum(mask > 0)} pixels)', fontsize=12)
            axes[0, i + 1].axis('off')
    
    # Create difference visualization
    diff_2_8 = masks[2] - masks[0]  # thickness 8 - thickness 2
    axes[1, 0].imshow(diff_2_8, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Difference: {line_thicknesses[2]}px - {line_thicknesses[0]}px\nRed = Thicker only', fontsize=12)
    axes[1, 0].axis('off')
    
    # Show combined overlay with all thicknesses
    overlay = original_image.copy()
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue
    
    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_colored = np.zeros_like(original_image)
        for c in range(3):
            mask_colored[:, :, c] = mask * color[c]
        overlay = overlay + mask_colored * 0.3
    
    overlay = np.clip(overlay, 0, 1)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Combined Overlay\nRed: 2px, Green: 5px, Blue: 8px', fontsize=12)
    axes[1, 1].axis('off')
    
    # Statistics comparison
    stats_text = "Line Thickness Statistics:\n\n"
    for thickness, mask in zip(line_thicknesses, masks):
        total_pixels = np.sum(mask > 0)
        percentage = (total_pixels / mask.size) * 100
        stats_text += f"{thickness}px thickness:\n"
        stats_text += f"  • {total_pixels:,} pixels\n"
        stats_text += f"  • {percentage:.3f}% of image\n\n"
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Pixel Statistics', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save detailed analysis
    output_file = output_path / 'detailed_line_thickness_analysis.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    logger.info(f"Saved detailed analysis: {output_file}")
    plt.close()


def create_goal_post_focus_analysis(data_root, output_path, line_thicknesses):
    """Create focused analysis on goal post regions."""
    logger.info("Creating goal post focused analysis...")
    
    # This would require identifying frames with visible goal posts
    # For now, we'll create a general analysis
    pass


def main():
    parser = argparse.ArgumentParser(description='Visualize pitch localization training data')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to training data root directory')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Directory to save visualization images')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of training samples to visualize')
    parser.add_argument('--line-thicknesses', type=int, nargs='+', default=[2, 5, 8],
                       help='Line thicknesses to compare')
    parser.add_argument('--target-size', type=int, nargs=2, default=[512, 512],
                       help='Target image size for visualization')
    parser.add_argument('--enable-augmentation', action='store_true',
                       help='Enable augmentation to see rotation effects (may cause misalignment)')
    
    args = parser.parse_args()
    
    # Validate data root
    if not os.path.exists(args.data_root):
        raise ValueError(f"Data root does not exist: {args.data_root}")
    
    visualize_training_samples(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        line_thicknesses=args.line_thicknesses,
        target_size=tuple(args.target_size),
        enable_augmentation=args.enable_augmentation
    )


if __name__ == '__main__':
    main()