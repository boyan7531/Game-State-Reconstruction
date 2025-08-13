"""
Configuration file for pitch localization training
"""

import os
from pathlib import Path

# Data configuration
DATA_CONFIG = {
    'data_root': 'SoccerNet/SN-GSR-2025/train',
    'target_size': (512, 512),
    'line_thickness': 2,
    'cache_masks': False,
    'train_split': 0.8,
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 10.0,
    'brightness_range': 0.15,
    'contrast_range': 0.15,
    'saturation_range': 0.15,
    'hue_range': 0.05,
    'horizontal_flip_prob': 0.5,
    'vertical_flip_prob': 0.0,
}

# Model configuration
MODEL_CONFIG = {
    'architecture': 'deeplabv3',  # 'deeplabv3' or 'unet'
    'backbone': 'resnet50',       # 'resnet50', 'resnet101', 'resnet34'
    'num_classes': 1,
    'pretrained': True,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'pin_memory': True,
    'gradient_clip_max_norm': 1.0,
}

# Loss function configuration
LOSS_CONFIG = {
    'loss_type': 'combined',  # 'bce', 'dice', 'combined'
    'bce_weight': 0.5,
    'dice_weight': 0.5,
    'focal_alpha': 0.25,      # For focal loss
    'focal_gamma': 2.0,       # For focal loss
}

# Optimizer and scheduler configuration
OPTIMIZER_CONFIG = {
    'optimizer': 'adamw',     # 'adam', 'adamw', 'sgd'
    'scheduler': 'cosine',    # 'cosine', 'step', 'plateau', 'none'
    'step_size': 10,          # For step scheduler
    'gamma': 0.1,             # For step scheduler
    'patience': 5,            # For plateau scheduler
    'factor': 0.5,            # For plateau scheduler
}

# Evaluation configuration
EVAL_CONFIG = {
    'eval_frequency': 1,      # Evaluate every N epochs
    'save_frequency': 5,      # Save checkpoint every N epochs
    'metric_threshold': 0.5,  # Threshold for binary metrics
    'early_stopping': True,
    'early_stopping_patience': 10,
    'early_stopping_metric': 'iou',
}

# Logging configuration
LOGGING_CONFIG = {
    'log_dir': 'runs',
    'checkpoint_dir': 'checkpoints',
    'log_level': 'INFO',
    'save_samples': True,     # Save sample predictions during training
    'num_sample_images': 4,   # Number of sample images to save
}

# Hardware configuration
HARDWARE_CONFIG = {
    'device': 'cuda',         # 'cuda', 'cpu', 'auto'
    'mixed_precision': True,  # Use automatic mixed precision
    'compile_model': False,   # Use torch.compile (PyTorch 2.0+)
}

# Default configuration combining all sections
DEFAULT_CONFIG = {
    'data': DATA_CONFIG,
    'augmentation': AUGMENTATION_CONFIG,
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'loss': LOSS_CONFIG,
    'optimizer': OPTIMIZER_CONFIG,
    'eval': EVAL_CONFIG,
    'logging': LOGGING_CONFIG,
    'hardware': HARDWARE_CONFIG,
}


def get_config(config_name: str = 'default'):
    """Get configuration dictionary."""
    if config_name == 'default':
        return DEFAULT_CONFIG
    elif config_name == 'fast':
        # Fast training configuration for testing
        config = DEFAULT_CONFIG.copy()
        config['training']['num_epochs'] = 5
        config['training']['batch_size'] = 4
        config['data']['target_size'] = (256, 256)
        return config
    elif config_name == 'high_quality':
        # High quality training configuration
        config = DEFAULT_CONFIG.copy()
        config['training']['num_epochs'] = 100
        config['training']['batch_size'] = 4  # Smaller batch for higher resolution
        config['data']['target_size'] = (768, 768)
        config['training']['learning_rate'] = 5e-4
        return config
    else:
        raise ValueError(f"Unknown config name: {config_name}")


def print_config(config: dict, indent: int = 0):
    """Pretty print configuration."""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == '__main__':
    # Print default configuration
    print("Default Configuration:")
    print_config(DEFAULT_CONFIG)