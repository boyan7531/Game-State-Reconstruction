# Pitch Localization Training Pipeline

A complete PyTorch implementation for training pitch localization models on SoccerNet GSR dataset. This pipeline provides semantic segmentation of soccer pitch lines using deep learning.

## Features

- **Dataset Support**: Full integration with SoccerNet GSR dataset
- **Model Architectures**: DeepLabV3 (ResNet50/101) and U-Net (ResNet34)
- **Data Augmentation**: Comprehensive augmentation pipeline with image-mask consistency
- **Loss Functions**: Combined BCE + Dice loss for better segmentation
- **Metrics**: IoU, F1-score, precision, recall tracking
- **Visualization**: TensorBoard logging and inference visualization
- **Checkpointing**: Automatic model saving and resuming

## Installation

```bash
# Clone the repository (if applicable)
cd pitch_localization/

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Ensure your SoccerNet GSR data is organized as follows:
```
SoccerNet/SN-GSR-2025/train/
├── SNGS-001/
│   ├── img1/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── Labels-GameState.json
├── SNGS-002/
└── ...
```

## Quick Start

### 1. Test Dataset Loading

```bash
python dataset.py
```

This will:
- Discover all frame-annotation pairs
- Create train/val datasets with augmentations
- Test sample loading and visualization
- Save debug outputs to `debug_output/`

### 2. Train Model

```bash
# Basic training with default parameters
python train.py --data-root SoccerNet/SN-GSR-2025/train

# Custom training configuration
python train.py \
    --data-root SoccerNet/SN-GSR-2025/train \
    --model deeplabv3 \
    --backbone resnet50 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-3 \
    --target-size 512 512
```

### 3. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir runs/
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- IoU, F1, precision, recall metrics
- Sample predictions

### 4. Run Inference

```bash
# Evaluate on validation set
python inference.py \
    --model-path checkpoints/pitch_segmentation_YYYYMMDD_HHMMSS_best.pth \
    --data-root SoccerNet/SN-GSR-2025/train \
    --evaluate \
    --output-dir results/

# Single image inference
python inference.py \
    --model-path checkpoints/pitch_segmentation_YYYYMMDD_HHMMSS_best.pth \
    --image-path path/to/your/image.jpg \
    --output-dir single_inference/
```

## Configuration

The pipeline supports flexible configuration through `config.py`:

```python
from config import get_config

# Default configuration
config = get_config('default')

# Fast training for testing
config = get_config('fast')

# High quality training
config = get_config('high_quality')
```

## Command Line Arguments

### Training (`train.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `SoccerNet/SN-GSR-2025/train` | Path to dataset |
| `--model` | `deeplabv3` | Model architecture (`deeplabv3`, `unet`) |
| `--backbone` | `resnet50` | Model backbone |
| `--batch-size` | `8` | Batch size |
| `--epochs` | `50` | Number of epochs |
| `--lr` | `1e-3` | Learning rate |
| `--target-size` | `512 512` | Input image size |
| `--device` | `cuda` | Device to use |

### Inference (`inference.py`)

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to trained model checkpoint |
| `--data-root` | Path to dataset (for evaluation) |
| `--image-path` | Single image path (for single inference) |
| `--evaluate` | Run full evaluation on validation set |
| `--threshold` | Binary threshold for masks (default: 0.5) |

## Model Architecture

### DeepLabV3 (Default)
- **Backbone**: ResNet50/101 with atrous convolution
- **Decoder**: ASPP (Atrous Spatial Pyramid Pooling)
- **Output**: 1-channel binary segmentation mask

### U-Net Alternative
- **Encoder**: ResNet34 backbone
- **Decoder**: Skip connections with upsampling
- **Output**: 1-channel binary segmentation mask

## Loss Function

Combined loss for better segmentation:
```python
Loss = α * BCE_Loss + β * Dice_Loss
```

- **BCE Loss**: Pixel-wise binary cross-entropy
- **Dice Loss**: Overlap-based loss for better boundary detection
- **Weights**: α=0.5, β=0.5 (configurable)

## Metrics

- **IoU (Intersection over Union)**: Primary metric
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

## Data Augmentation

**Geometric Augmentations** (applied to both image and mask):
- Random rotation (±10°)
- Horizontal flip (50% probability)
- Resize to target resolution

**Photometric Augmentations** (applied to image only):
- Brightness adjustment (±15%)
- Contrast adjustment (±15%)
- Saturation adjustment (±15%)
- Hue adjustment (±5%)

## File Structure

```
pitch_localization/
├── dataset.py           # Dataset implementation
├── train.py            # Training script
├── inference.py        # Inference and evaluation
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
├── README.md          # This file
├── runs/              # TensorBoard logs
├── checkpoints/       # Model checkpoints
└── debug_output/      # Debug visualizations
```

## Expected Performance

On SoccerNet GSR dataset:
- **IoU**: 0.7-0.8 (depending on data quality)
- **F1-Score**: 0.8-0.9
- **Training Time**: ~2-4 hours on GPU (50 epochs)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch-size` or `--target-size`
2. **No samples found**: Check dataset path and structure
3. **Low performance**: 
   - Increase training epochs
   - Try different learning rates
   - Check data augmentation settings

### Debug Dataset

```bash
python dataset.py
```

This will show:
- Number of discovered samples
- Sample loading tests
- Debug visualizations

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

## Citation

If you use this code in your research, please cite the SoccerNet GSR dataset:

```bibtex
@article{soccernet_gsr,
    title={SoccerNet Game State Reconstruction},
    author={...},
    journal={...},
    year={2025}
}
```