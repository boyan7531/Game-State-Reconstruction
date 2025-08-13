import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models.segmentation as seg_models
import numpy as np
from tqdm import tqdm

from dataset import create_train_dataset, create_val_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, backbone: str = 'resnet34'):
        super(UNet, self).__init__()
        
        # Use pretrained backbone
        if backbone == 'resnet34':
            from torchvision.models import resnet34
            backbone_model = resnet34(pretrained=True)
            
            # Encoder layers
            self.encoder1 = nn.Sequential(
                backbone_model.conv1,
                backbone_model.bn1,
                backbone_model.relu
            )
            self.encoder2 = backbone_model.layer1  # 64 channels
            self.encoder3 = backbone_model.layer2  # 128 channels
            self.encoder4 = backbone_model.layer3  # 256 channels
            self.encoder5 = backbone_model.layer4  # 512 channels
            
            # Decoder with skip connections
            self.decoder4 = self._make_decoder_block(512, 256, 256)
            self.decoder3 = self._make_decoder_block(256 + 256, 128, 128)
            self.decoder2 = self._make_decoder_block(128 + 128, 64, 64)
            self.decoder1 = self._make_decoder_block(64 + 64, 64, 32)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Final classifier
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def _make_decoder_block(self, in_channels: int, skip_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)  # 64 x H/2 x W/2
        e2 = self.encoder2(self.pool(e1))  # 64 x H/4 x W/4
        e3 = self.encoder3(e2)  # 128 x H/8 x W/8
        e4 = self.encoder4(e3)  # 256 x H/16 x W/16
        e5 = self.encoder5(e4)  # 512 x H/32 x W/32
        
        # Decoder path with skip connections
        d4 = self.decoder4(e5)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Skip connection
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Skip connection
        
        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        
        # Final output
        out = self.final_conv(d1)
        return self.sigmoid(out)


def create_model(model_name: str = 'deeplabv3', backbone: str = 'resnet50', num_classes: int = 1) -> nn.Module:
    """Create segmentation model."""
    if model_name == 'deeplabv3':
        if backbone == 'resnet50':
            model = seg_models.deeplabv3_resnet50(pretrained=True)
        elif backbone == 'resnet101':
            model = seg_models.deeplabv3_resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone {backbone} for DeepLabV3")
        
        # Modify classifier for our task
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        # Initialize the new layer properly
        nn.init.kaiming_normal_(model.classifier[4].weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(model.classifier[4].bias, 0)
        model.aux_classifier = None  # Remove auxiliary classifier
        
    elif model_name == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes, backbone='resnet34')
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        pred_sum = pred.sum()
        target_sum = target.sum()
        
        # Correct Dice coefficient formula
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Clamp to ensure dice is in [0, 1] range
        dice = torch.clamp(dice, 0., 1.)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure predictions are in valid range [0, 1]
        pred = torch.clamp(pred, 0., 1.)
        target = torch.clamp(target, 0., 1.)
        
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Validate that individual losses are positive
        if bce < 0 or dice < 0:
            logger.warning(f"Invalid loss values: BCE={bce.item():.4f}, Dice={dice.item():.4f}")
        
        combined = self.bce_weight * bce + self.dice_weight * dice
        return combined


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_binary = (pred > self.threshold).float()
        target_binary = target.float()
        
        self.tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
        self.fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
        self.tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()
        self.fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()
    
    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + eps)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy
        }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        log_dir: str = 'runs',
        checkpoint_dir: str = 'checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Setup logging and checkpointing
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create unique experiment name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"pitch_segmentation_{timestamp}"
        self.writer = SummaryWriter(self.log_dir / self.experiment_name)
        
        # Training state
        self.epoch = 0
        self.best_val_iou = 0.0
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        logger.info(f"Trainer initialized. Experiment: {self.experiment_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle different model outputs
            if isinstance(outputs, dict):  # DeepLabV3 returns dict
                outputs = outputs['out']
            
            # Apply sigmoid to get probabilities for DeepLabV3
            # UNet already has sigmoid in forward pass
            if hasattr(self.model, 'classifier'):  # DeepLabV3 model
                outputs = torch.sigmoid(outputs)
            
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        
        # Log metrics
        self.writer.add_scalar('Train/Loss', avg_loss, self.epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Train/{metric_name.capitalize()}', metric_value, self.epoch)
        
        return {'loss': avg_loss, **metrics}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle different model outputs
                if isinstance(outputs, dict):  # DeepLabV3 returns dict
                    outputs = outputs['out']
                
                # Apply sigmoid to get probabilities for DeepLabV3
                # UNet already has sigmoid in forward pass
                if hasattr(self.model, 'classifier'):  # DeepLabV3 model
                    outputs = torch.sigmoid(outputs)
                
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                self.val_metrics.update(outputs, masks)
        
        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        
        # Log metrics
        self.writer.add_scalar('Val/Loss', avg_loss, self.epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name.capitalize()}', metric_value, self.epoch)
        
        return {'loss': avg_loss, **metrics}
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_iou': self.best_val_iou
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.experiment_name}_epoch_{self.epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / f"{self.experiment_name}_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: IoU = {self.best_val_iou:.4f}")
    
    def train(self, num_epochs: int, save_freq: int = 5):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Check for best model
            val_iou = val_metrics['iou']
            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train IoU: {train_metrics['iou']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val IoU: {val_metrics['iou']:.4f}"
            )
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        logger.info(f"Training completed. Best Val IoU: {self.best_val_iou:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train pitch localization model')
    parser.add_argument('--data-root', type=str, default='SoccerNet/SN-GSR-2025',
                       help='Path to SoccerNet GSR dataset root (containing train and valid folders)')
    parser.add_argument('--model', type=str, default='deeplabv3', choices=['deeplabv3', 'unet'],
                       help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'resnet34'],
                       help='Model backbone')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--target-size', type=int, nargs=2, default=[512, 512], help='Target image size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device to use')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Set device with MPS support
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create datasets from separate train and valid folders
    logger.info("Creating datasets...")
    train_data_root = str(Path(args.data_root) / "train")
    val_data_root = str(Path(args.data_root) / "valid")
    
    train_dataset = create_train_dataset(
        train_data_root,
        target_size=tuple(args.target_size),
        cache_masks=False
    )
    
    val_dataset = create_val_dataset(
        val_data_root,
        target_size=tuple(args.target_size),
        cache_masks=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating {args.model} model with {args.backbone} backbone...")
    model = create_model(args.model, args.backbone, num_classes=1)
    
    # Create loss function and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()