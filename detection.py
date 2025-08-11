#!/usr/bin/env python3
"""
YOLO Training Script for SoccerNet Game State Reconstruction
Trains YOLOv8 model on the converted SoccerNet GSR dataset.
"""

import os
import yaml
import argparse
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
        self.class_names = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
    
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
    
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each epoch (both training and validation)."""
        # Show per-class validation metrics if available
        if hasattr(trainer, 'validator') and trainer.validator and hasattr(trainer.validator, 'metrics'):
            metrics = trainer.validator.metrics
            if hasattr(metrics, 'box') and hasattr(metrics.box, 'ap50'):
                # Get per-class AP@0.5 and AP@0.5:0.95
                ap50_per_class = metrics.box.ap50
                ap_per_class = metrics.box.maps  # Fixed: use .maps for per-class AP@0.5:0.95
                
                if ap50_per_class is not None and len(ap50_per_class) > 0:
                    print(f"\n📊 Epoch {self.current_epoch} - Per-Class Validation Metrics:")
                    print("-" * 55)
                    for i, (class_id, class_name) in enumerate(self.class_names.items()):
                        if i < len(ap50_per_class):
                            ap50 = ap50_per_class[i] if ap50_per_class[i] is not None else 0.0
                            ap = ap_per_class[i] if ap_per_class is not None and i < len(ap_per_class) and ap_per_class[i] is not None else 0.0
                            print(f"  {class_name:12} | AP@0.5: {ap50:.4f} | AP@0.5:0.95: {ap:.4f}")
    
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
    use_augmentations: bool = True,
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
        use_augmentations: Whether to use data augmentations (True) or not (False)
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

    lr0 = 0.001  

    # Training arguments with optimizations
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': 1280,  # Base image size
        'batch': batch_size,
        'device': device,
        'project': project,
        'name': name,
        'save_period': save_period,
        'patience': patience,
        'workers': 8, 
        'amp': True,  
        'optimizer': 'AdamW',
        'lr0': lr0,
        'lrf': 0.1, 
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3, 
        'cos_lr': True,  
        'box': 6.0,
        'cls': 1.2,
        'dfl': 1.4,
        'label_smoothing': 0.0,
        'nbs': 64,
        **kwargs
    }
    
    # Add augmentations conditionally based on flag
    if use_augmentations:
        print("🎨 Using enhanced data augmentations")
        augmentation_args = {
            # Enhanced data augmentation tuned for soccer footage (without harmful mosaic)
            'hsv_h': 0.015,
            'hsv_s': 0.5,       
            'hsv_v': 0.3,       
            'degrees': 5.0,     
            'translate': 0.1,   
            'scale': 0.2,       
            'shear': 0.1,       
            'perspective': 0.0, 
            'flipud': 0.0,      
            'fliplr': 0.5,      
            'mosaic': 0.0,      
            'mixup': 0.05,      
            'copy_paste': 0.10, 
            'multi_scale': True,
        }
        train_args.update(augmentation_args)
    else:
        print("🚫 Training without data augmentations")
        # Disable all augmentations
        no_augmentation_args = {
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
            'multi_scale': False,
        }
        train_args.update(no_augmentation_args)
    
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
    model.add_callback('on_fit_epoch_end', progress_callback.on_fit_epoch_end)
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

def validate_model(model_path: str, data_yaml: str, device: str = 'auto', batch_size: int = 32, **kwargs):
    """
    Validate trained model on test/validation set.
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML configuration
        device: Device to use for validation ('auto', 'cpu', '0', '1', etc.)
        batch_size: Batch size for validation (larger = faster)
        **kwargs: Additional validation arguments
    """
    print(f"🔍 Validating model: {model_path}")
    
    # Set device based on availability (matching training logic)
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Optimized validation arguments
    val_args = {
        'data': data_yaml,
        'device': device,
        'batch': batch_size,  # Larger batch size for faster validation
        'imgsz': 1280,        # Use middle value for consistent validation
        'workers': 8,         # Parallel data loading
        'verbose': True,      # Show detailed progress
        **kwargs
    }
    
    # More informative progress tracking
    with tqdm(total=3, desc="Validation", unit="phase") as pbar:
        pbar.set_postfix_str("Loading model...")
        model = YOLO(model_path)
        pbar.update(1)
        
        pbar.set_postfix_str("Running validation...")
        results = model.val(**val_args)
        pbar.update(1)
        
        # Display key metrics
        if hasattr(results, 'box') and hasattr(results.box, 'map'):
            metrics_str = f"mAP@0.5: {results.box.map50:.4f} | mAP@0.5:0.95: {results.box.map:.4f}"
            pbar.set_postfix_str(metrics_str)
        pbar.update(1)
    
    # Display per-class metrics
    if hasattr(results, 'box') and hasattr(results.box, 'ap50'):
        print("\n📊 Per-Class Validation Metrics:")
        print("-" * 50)
        
        # Get class names from the dataset config
        class_names = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
        
        # Get per-class AP@0.5 and AP@0.5:0.95
        ap50_per_class = results.box.ap50
        ap_per_class = results.box.maps
        
        if ap50_per_class is not None and len(ap50_per_class) > 0:
            for i, (class_id, class_name) in enumerate(class_names.items()):
                if i < len(ap50_per_class):
                    ap50 = ap50_per_class[i] if ap50_per_class[i] is not None else 0.0
                    ap = ap_per_class[i] if ap_per_class is not None and i < len(ap_per_class) and ap_per_class[i] is not None else 0.0
                    print(f"  {class_name:12} | AP@0.5: {ap50:.4f} | AP@0.5:0.95: {ap:.4f}")
        else:
            print("  Per-class metrics not available")
    
    print("✅ Validation completed!")
    return results

def main():
    """
    Main training pipeline with enhanced progress tracking.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Training Script for SoccerNet Game State Reconstruction')
    parser.add_argument('--no-augmentations', action='store_true', 
                       help='Disable data augmentations during training')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training (default: 8)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model size (default: yolov8n.pt)')
    parser.add_argument('--name', type=str, default='soccernet_gsr_v8_improved',
                       help='Experiment name (default: soccernet_gsr_v8_improved)')
    
    args = parser.parse_args()
    use_augmentations = not args.no_augmentations
    
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
    
    # Training configuration (optimized for better object classification)
    experiment_name = f"{args.name}{'_no_aug' if not use_augmentations else '_with_aug'}"
    training_config = {
        'model_size': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': 'auto',
        'project': 'runs/detect',
        'name': experiment_name,
        'save_period': 1,
        'patience': 5,
        'use_augmentations': use_augmentations,
    }
    
    augmentation_status = "with augmentations" if use_augmentations else "without augmentations"
    print(f"\n🚀 Starting YOLO training for SoccerNet GSR ({augmentation_status})...")
    
    try:
        # Train model
        results, _ = train_yolo_model(dataset_yaml, **training_config)
        
        # Display some key training results
        if results and hasattr(results, 'results_dict'):
            print(f"\n📈 Key Training Metrics:")
            results_dict = results.results_dict
            if 'train/box_loss' in results_dict:
                print(f"  Final Box Loss: {results_dict['train/box_loss']:.4f}")
            if 'train/cls_loss' in results_dict:
                print(f"  Final Class Loss: {results_dict['train/cls_loss']:.4f}")
            if 'metrics/mAP50(B)' in results_dict:
                print(f"  Best mAP@0.5: {results_dict['metrics/mAP50(B)']:.4f}")
        
        # Validate model
        best_model_path = f"{training_config['project']}/{training_config['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print("\n🔍 Running validation on best model...")
            val_results = validate_model(
                best_model_path, 
                dataset_yaml,
                device=training_config['device'],
                batch_size=min(training_config['batch_size'] * 2, 64)  # Larger batch for validation
            )
            
            # Display validation results
            if val_results:
                print(f"\n📈 Validation Results:")
                if hasattr(val_results, 'box'):
                    box = val_results.box
                    if hasattr(box, 'map50') and hasattr(box, 'map'):
                        print(f"  mAP@0.5: {box.map50:.4f}")
                        print(f"  mAP@0.5:0.95: {box.map:.4f}")
                    
                    # Show per-class metrics
                    class_names = {0: 'player', 1: 'goalkeeper', 2: 'referee', 3: 'ball'}
                    ap50_per_class = box.ap50
                    ap_per_class = box.maps
                    
                    if ap50_per_class is not None and len(ap50_per_class) > 0:
                        print(f"\n📊 Per-Class Validation Metrics:")
                        print("-" * 40)
                        for i, (class_id, class_name) in enumerate(class_names.items()):
                            if i < len(ap50_per_class):
                                ap50 = ap50_per_class[i] if ap50_per_class[i] is not None else 0.0
                                ap = ap_per_class[i] if ap_per_class is not None and i < len(ap_per_class) and ap_per_class[i] is not None else 0.0
                                print(f"  {class_name:12} | AP@0.5: {ap50:.4f} | AP@0.5:0.95: {ap:.4f}")
                if hasattr(val_results, 'speed') and val_results.speed:
                    print(f"  Inference Speed: {val_results.speed['inference']:.2f}ms per image")
            
        
        print("\n" + "="*50)
        print("🎉 TRAINING SUMMARY")
        print("="*50)
        print(f"📊 Model: {training_config['model_size']}")
        print(f"⏱️  Epochs: {training_config['epochs']}")
        print(f"📁 Dataset: {data_dir}")
        print(f"🏆 Best weights: {best_model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except RuntimeError as e:
        if "MPS backend out of memory" in str(e):
            print("\n❌ Training failed due to insufficient GPU memory.")
            print("🔧 Troubleshooting tips:")
            print("  1. Reduce batch_size in training_config (currently {})".format(training_config['batch_size']))
            print("  2. Try a smaller model (yolov8m.pt or yolov8s.pt instead of yolov8l.pt)")
            print("  3. Use CPU instead of MPS by setting device='cpu' in training_config")
        else:
            print(f"\n❌ Training failed with error: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("  1. Check GPU memory (reduce batch_size if needed)")
            print("  2. Verify dataset format and paths")
            print("  3. Check CUDA/PyTorch installation")
        raise
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("  1. Check GPU memory (reduce batch_size if needed)")
        print("  2. Verify dataset format and paths")
        print("  3. Check CUDA/PyTorch installation")
        raise

if __name__ == "__main__":
    main()