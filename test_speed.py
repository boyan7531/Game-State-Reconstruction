#!/usr/bin/env python3
"""
Quick test to compare binary vs multiclass training speed on MacBook.
"""
import time
import torch
import sys
import os

# Change to the pitch_localization directory to import correctly
original_dir = os.getcwd()
os.chdir('/Users/boyan531/Documents/football/pitch_localization')
sys.path.insert(0, '/Users/boyan531/Documents/football/pitch_localization')

from train import SegmentationMetrics
from dataset import NUM_CLASSES

def test_metrics_speed():
    """Test the speed difference between binary and multiclass metrics."""
    
    # Simulate batch data (smaller for MacBook)
    batch_size = 4
    height, width = 256, 256
    num_classes = NUM_CLASSES
    
    print(f"Testing with batch_size={batch_size}, image_size={height}x{width}")
    print(f"Total pixels per batch: {batch_size * height * width:,}")
    
    # Create fake predictions and targets
    pred_multiclass = torch.randn(batch_size, num_classes, height, width)
    pred_binary = torch.sigmoid(torch.randn(batch_size, 1, height, width))
    target_multiclass = torch.randint(0, num_classes, (batch_size, height, width))
    target_binary = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test multiclass metrics (the fixed version)
    print("\n=== Testing Multiclass Metrics ===")
    multiclass_metrics = SegmentationMetrics(task_type='multiclass', num_classes=num_classes)
    
    start_time = time.time()
    for i in range(5):  # 5 batches
        multiclass_metrics.update(pred_multiclass, target_multiclass)
    multiclass_time = time.time() - start_time
    
    print(f"Multiclass: {multiclass_time:.3f} seconds for 5 batches")
    print(f"Per batch: {multiclass_time/5:.3f} seconds")
    
    # Test binary metrics
    print("\n=== Testing Binary Metrics ===")
    binary_metrics = SegmentationMetrics(task_type='binary')
    
    start_time = time.time()
    for i in range(5):  # 5 batches
        binary_metrics.update(pred_binary, target_binary)
    binary_time = time.time() - start_time
    
    print(f"Binary: {binary_time:.3f} seconds for 5 batches")
    print(f"Per batch: {binary_time/5:.3f} seconds")
    
    # Compare
    print(f"\n=== Comparison ===")
    if binary_time > 0:
        ratio = multiclass_time / binary_time
        print(f"Multiclass is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than binary")
    
    print(f"Multiclass per-pixel time: {(multiclass_time/5)/(batch_size*height*width)*1000:.4f} ms per 1000 pixels")

if __name__ == "__main__":
    test_metrics_speed()