#!/usr/bin/env python3
"""
YOLO Training Script for SoccerNet Game State Reconstruction
Trains YOLOv8 model on the converted SoccerNet GSR dataset.
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm

class TrainingProgressCallback:
    """Custom callback to show batch-level training progress with tqdm."""
    
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_pbar = None
        self.batch_pbar = None
        self.current_epoch = 0
        self.total_batches = 0
        self.current_batch = 0
    
    def on_train_start(self, trainer):
        """Called when training starts."""
        print(f"🚀 Starting training for {self.epochs} epochs...")
        # Get total number of batches from trainer
        if hasattr(trainer, 'train_loader') and trainer.train_loader:
            self.total_batches = len(trainer.train_loader)
        else:
            self.total_batches = 100  # fallback estimate
        
        self.epoch_pbar = tqdm(total=self.epochs, desc="Epochs", position=0, leave=True)
    
    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch."""
        self.current_epoch += 1
        self.current_batch = 0
        
        # Create batch progress bar for this epoch
        if self.batch_pbar:
            self.batch_pbar.close()
        
        self.batch_pbar = tqdm(
            total=self.total_batches,
            desc=f"Epoch {self.current_epoch}/{self.epochs} - Batches",
            position=1,
            leave=False,
            unit="batch"
        )
    
    def on_train_batch_end(self, trainer):
        """Called at the end of each training batch."""
        self.current_batch += 1
        if self.batch_pbar:
            # Get current loss values if available
            loss_info = ""
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                try:
                    losses = trainer.loss_items
                    if len(losses) >= 3:
                        box_loss = losses[0] if losses[0] is not None else 0
                        cls_loss = losses[1] if losses[1] is not None else 0
                        dfl_loss = losses[2] if losses[2] is not None else 0
                        loss_info = f"Box: {box_loss:.3f} | Cls: {cls_loss:.3f} | DFL: {dfl_loss:.3f}"
                except (IndexError, TypeError):
                    pass
            
            self.batch_pbar.set_postfix_str(loss_info)
            self.batch_pbar.update(1)
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        if self.epoch_pbar:
            # Get epoch metrics
            metrics_info = ""
            if hasattr(trainer, 'metrics') and trainer.metrics:
                try:
                    metrics = trainer.metrics
                    if 'train/box_loss' in metrics and 'train/cls_loss' in metrics:
                        box_loss = metrics['train/box_loss']
                        cls_loss = metrics['train/cls_loss']
                        metrics_info = f"Box: {box_loss:.4f} | Cls: {cls_loss:.4f}"
                except (KeyError, TypeError):
                    pass
            
            self.epoch_pbar.set_postfix_str(metrics_info)
            self.epoch_pbar.update(1)
    
    def on_train_end(self, trainer):
        """Called when training ends."""
        if self.batch_pbar:
            self.batch_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()
        print("\n✅ Training completed!")

def create_dataset_yaml(data_dir: str, output_path: str = "dataset.yaml"):
    """
    Create YOLO dataset configuration file.
    
    Args:
        data_dir: Path to the YOLO dataset directory
        output_path: Path to save the dataset.yaml file
    """
    data_config = {
        'path': str(Path(data_dir).absolute()),  # Dataset root dir
        'train': 'train/images',  # Train images (relative to 'path')
        'val': 'val/images',      # Val images (relative to 'path') 
        'test': 'test/images',    # Test images (relative to 'path')
        
        # Classes
        'nc': 4,  # Number of classes
        'names': {
            0: 'player',
            1: 'goalkeeper', 
            2: 'referee',
            3: 'ball'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    return output_path

def train_yolo_model(
    data_yaml: str,
    model_size: str = 'yolo12l.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 52,
    device: str = 'auto',
    project: str = 'runs/detect',
    name: str = 'soccernet_gsr',
    save_period: int = 10,
    patience: int = 50,
    dataset_size: int = None,
    **kwargs
):
    """
    Train YOLO model on SoccerNet GSR dataset.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        model_size: YOLO model size ('yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt', 'yolo12l.pt', 'yolo12x.pt')
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        device: Device to use ('auto', 'cpu', '0', '1', etc.)
        project: Project directory for saving results
        name: Experiment name
        save_period: Save checkpoint every N epochs
        patience: Early stopping patience
        dataset_size: Number of training images (auto-detected if None)
        **kwargs: Additional training arguments
    """
    
    # Enable TF32 for faster training on RTX 4090
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled for faster training")
    
    # Load model
    print(f"Loading YOLO model: {model_size}")
    model = YOLO(model_size)
    
    # Check device - prioritize MPS for Apple Silicon, then CUDA, then CPU
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
            print(f"✅ Using Apple Silicon GPU (MPS): {device}")
        elif torch.cuda.is_available():
            device = 'cuda'
            print(f"✅ Using NVIDIA GPU (CUDA): {device}")
        else:
            device = 'cpu'
            print(f"⚠️  Using CPU (no GPU acceleration available): {device}")
    else:
        print(f"Using specified device: {device}")
    
    # Verify device availability
    try:
        if device == 'mps' and not torch.backends.mps.is_available():
            print("⚠️  MPS not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
    except Exception as e:
        print(f"⚠️  Device check failed: {e}, using CPU")
        device = 'cpu'
    
    # Adaptive learning rate based on dataset size
    if dataset_size is None:
        # Try to estimate dataset size from YAML
        try:
            with open(data_yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)
            train_path = os.path.join(yaml_data['path'], yaml_data['train'])
            if os.path.exists(train_path):
                dataset_size = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"📊 Estimated dataset size: {dataset_size} images")
        except:
            dataset_size = 5000  # fallback estimate
            print(f"⚠️  Could not estimate dataset size, using fallback: {dataset_size}")
    
    
    lr0 = 0.1  # Suggested by LR Finder analysis
    
    print(f"🎯 Using LR Finder suggested learning rate: {lr0:.6f} (dataset size: {dataset_size})")
    
    # Training arguments with optimizations
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'save_period': save_period,
        'patience': patience,
        'workers': 8,  # More data loading threads
        'amp': True,  # Explicit AMP (already default but explicit)
        'optimizer': 'AdamW',
        'lr0': lr0,  # Adaptive learning rate
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'nbs': 64,
        'plots': True,  # Enable confusion matrix and other plots during validation
        # Data augmentation disabled - using raw data only
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,
        'degrees': 0.0,
        'translate': 0.0,
        'scale': 0.0,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.0,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        **kwargs
    }
    
    print("📋 Training configuration:")
    key_params = ['epochs', 'batch', 'imgsz', 'device', 'optimizer', 'lr0']
    for key in key_params:
        if key in train_args:
            print(f"  {key}: {train_args[key]}")
    print(f"  dataset_size: {dataset_size}")
    print(f"  ... and {len(train_args) - len(key_params)} more parameters")
    
    print("\n🔄 Initializing training...")
    
    # Create progress callback
    progress_callback = TrainingProgressCallback(epochs)
    
    # Add callbacks to the model for batch-level progress tracking
    model.add_callback('on_train_start', progress_callback.on_train_start)
    model.add_callback('on_train_epoch_start', progress_callback.on_train_epoch_start)
    model.add_callback('on_train_batch_end', progress_callback.on_train_batch_end)
    model.add_callback('on_train_epoch_end', progress_callback.on_train_epoch_end)
    model.add_callback('on_train_end', progress_callback.on_train_end)
    
    # Start training with progress tracking
    try:
        results = model.train(**train_args)
        
        print(f"\n📊 Training Results:")
        print(f"  📁 Results saved to: {project}/{name}")
        print(f"  🏆 Best model: {project}/{name}/weights/best.pt")
        print(f"  📝 Last model: {project}/{name}/weights/last.pt")
        
        return results, model
        
    except Exception as e:
        # Clean up progress bars on error
        if hasattr(progress_callback, 'batch_pbar') and progress_callback.batch_pbar:
            progress_callback.batch_pbar.close()
        if hasattr(progress_callback, 'epoch_pbar') and progress_callback.epoch_pbar:
            progress_callback.epoch_pbar.close()
        print(f"\n❌ Training failed: {e}")
        raise

def validate_model(model_path: str, data_yaml: str, **kwargs):
    """
    Validate trained model on test/validation set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML configuration
        **kwargs: Additional validation arguments
    """
    print(f"🔍 Validating model: {model_path}")
    
    with tqdm(total=1, desc="Validation", unit="phase") as pbar:
        model = YOLO(model_path)
        pbar.set_postfix_str("Loading model...")
        pbar.update(0.3)
        
        pbar.set_postfix_str("Running validation...")
        results = model.val(data=data_yaml, **kwargs)
        pbar.update(0.7)
        
        pbar.set_postfix_str("Completed")
    
    print("✅ Validation completed!")
    return results

def main():
    """
    Main training pipeline with enhanced progress tracking.
    """
    # Configuration
    data_dir = "yolo_dataset_proper"
    dataset_yaml = "soccernet_dataset.yaml"
    
    print("🏈 SoccerNet Game State Reconstruction - YOLO Training Pipeline")
    print("=" * 65)
    
    # Check if proper dataset structure exists
    print("\n📂 Checking dataset structure...")
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory '{data_dir}' not found!")
        print("Please run create_yolo_dataset.py first to generate the proper YOLO dataset.")
        return
    
    # Check for proper split structure
    required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
    missing_dirs = []
    
    with tqdm(total=len(required_dirs), desc="Verifying directories", unit="dir") as pbar:
        for req_dir in required_dirs:
            if not os.path.exists(os.path.join(data_dir, req_dir)):
                missing_dirs.append(req_dir)
            pbar.update(1)
    
    if missing_dirs:
        print(f"❌ Error: Missing required directories in {data_dir}:")
        for missing in missing_dirs:
            print(f"  - {missing}")
        print("Please run create_yolo_dataset.py to create the proper dataset structure.")
        return
    
    print("✅ Dataset structure verified!")
    
    # Use existing dataset.yaml if available, otherwise create one
    print("\n📝 Preparing configuration...")
    existing_yaml = os.path.join(data_dir, "dataset.yaml")
    if os.path.exists(existing_yaml):
        print(f"Using existing dataset configuration: {existing_yaml}")
        dataset_yaml = existing_yaml
    else:
        print("Creating dataset configuration...")
        create_dataset_yaml(data_dir, dataset_yaml)
    
    # Training configuration (optimized for RTX 4090)
    training_config = {
        'model_size': 'yolo12l.pt',  # YOLOv12 Large model for better performance
        'epochs': 100,
        'imgsz': 1280,
        'batch_size': 8,  # Optimized for RTX 4090 (24GB VRAM)
        'device': 'auto',
        'project': 'runs/detect',
        'name': 'soccernet_gsr_v12_optimized',
        'save_period': 10,
        'patience': 20,
    }
    
    print("\n🚀 Starting YOLO training for SoccerNet GSR...")
    
    try:
        # Train model
        results, model = train_yolo_model(dataset_yaml, **training_config)
        
        # Validate model
        best_model_path = f"{training_config['project']}/{training_config['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print("\n🔍 Running validation on best model...")
            val_results = validate_model(best_model_path, dataset_yaml)
        
        print("\n" + "="*50)
        print("🎉 TRAINING SUMMARY")
        print("="*50)
        print(f"📊 Model: {training_config['model_size']}")
        print(f"⏱️  Epochs: {training_config['epochs']}")
        print(f"📁 Dataset: {data_dir}")
        print(f"🏆 Best weights: {best_model_path}")
        print("\n📋 Next steps:")
        print("  1. 📊 Check training results in runs/detect/soccernet_gsr_v1/")
        print("  2. 🔮 Use best.pt for inference")
        print("  3. 🎯 Implement tracking and other components")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("  1. Check GPU memory (reduce batch_size if needed)")
        print("  2. Verify dataset format and paths")
        print("  3. Check CUDA/PyTorch installation")
        raise

if __name__ == "__main__":
    main()