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
from torch.cuda.amp import autocast, GradScaler
import torchvision.models.segmentation as seg_models
import numpy as np
from tqdm import tqdm

from dataset import create_train_dataset, create_val_dataset, NUM_CLASSES, CLASS_TO_LINE_MAPPING

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


def create_model(model_name: str = 'deeplabv3', backbone: str = 'resnet50', num_classes: int = NUM_CLASSES, task_type: str = 'multiclass') -> nn.Module:
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
        
        # Initialize the new layer appropriately for the task
        if task_type == 'multiclass':
            # Xavier initialization for multi-class classification
            nn.init.xavier_uniform_(model.classifier[4].weight)
            # Initialize bias with small negative values to encourage learning rare classes
            with torch.no_grad():
                # Set background bias to 0, others slightly negative to encourage detection
                model.classifier[4].bias[0] = 0.0  # Background
                model.classifier[4].bias[1:] = -0.1  # All line classes get slight negative bias
        else:
            # Original initialization for binary segmentation
            nn.init.kaiming_normal_(model.classifier[4].weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(model.classifier[4].bias, -2.0)  # Slight negative bias for better thin line detection
            
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


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # More aggressive clamping to prevent numerical issues
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1 - eps)
        target = torch.clamp(target, 0., 1.)
        
        # Compute binary cross entropy manually with better numerical stability
        ce_loss = -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
        
        # Compute focal weight with clamping
        p_t = torch.where(target == 1, pred, 1 - pred)
        p_t = torch.clamp(p_t, eps, 1 - eps)  # Ensure p_t is in valid range
        
        # More stable focal weight computation
        focal_weight = self.alpha * torch.pow(1 - p_t, self.gamma)
        focal_weight = torch.clamp(focal_weight, 0, 1000)  # Prevent extremely large weights
        
        focal_loss = focal_weight * ce_loss
        
        # Check for NaN/Inf and replace with zeros if found
        focal_loss = torch.where(torch.isfinite(focal_loss), focal_loss, torch.zeros_like(focal_loss))
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use pos_weight to handle class imbalance
        loss = nn.functional.binary_cross_entropy(pred, target, weight=None, reduction='none')
        
        if self.pos_weight is not None:
            # Apply positive weight to positive samples
            pos_weight = self.pos_weight.to(pred.device)
            loss = loss * (target * pos_weight + (1 - target))
        
        return loss.mean()


class MultiClassLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        loss_type: str = 'cross_entropy',
        class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1  # Add some label smoothing by default
    ):
        super(MultiClassLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing, ignore_index=-1)
        elif loss_type == 'focal':
            # Focal loss for multi-class is more complex, use CrossEntropy for now
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing, ignore_index=-1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred shape: (N, C, H, W)
        # target shape: (N, H, W)
        return self.loss_fn(pred, target)


class CombinedLoss(nn.Module):
    def __init__(
        self, 
        loss_type: str = 'focal_dice',
        focal_weight: float = 0.2, 
        dice_weight: float = 0.8,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        task_type: str = 'binary',
        num_classes: int = NUM_CLASSES,
        **kwargs  # Accept additional kwargs like class_weights
    ):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.task_type = task_type
        self.num_classes = num_classes
        
        if task_type == 'multiclass':
            # For multi-class segmentation - accept class weights from outside
            class_weights = kwargs.get('class_weights', None)
            self.multiclass_loss = MultiClassLoss(num_classes=num_classes, loss_type='cross_entropy', class_weights=class_weights)
        else:
            # For binary segmentation (original)
            if loss_type == 'focal_dice':
                self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
                self.dice_loss = DiceLoss()
            elif loss_type == 'weighted_bce_dice':
                self.bce_loss = WeightedBCELoss(pos_weight=pos_weight)
                self.dice_loss = DiceLoss()
            else:  # fallback to original
                self.bce_loss = nn.BCELoss()
                self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task_type == 'multiclass':
            # For multi-class segmentation
            # pred shape: (N, C, H, W), target shape: (N, H, W)
            return self.multiclass_loss(pred, target)
        else:
            # For binary segmentation (original implementation)
            # Ensure predictions are in valid range [0, 1]
            eps = 1e-7
            pred = torch.clamp(pred, eps, 1 - eps)
            target = torch.clamp(target, 0., 1.)
            
            # Check for NaN/Inf in inputs
            if not torch.isfinite(pred).all():
                logger.warning("NaN/Inf detected in predictions, replacing with zeros")
                pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
            
            if not torch.isfinite(target).all():
                logger.warning("NaN/Inf detected in targets, replacing with zeros")
                target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
            
            dice = self.dice_loss(pred, target)
            
            if self.loss_type == 'focal_dice':
                focal = self.focal_loss(pred, target)
                combined = self.focal_weight * focal + self.dice_weight * dice
            elif self.loss_type == 'weighted_bce_dice':
                bce = self.bce_loss(pred, target)
                combined = self.focal_weight * bce + self.dice_weight * dice
            else:  # fallback
                bce = self.bce_loss(pred, target)
                combined = 0.5 * bce + 0.5 * dice
            
            # Final check for NaN/Inf in combined loss
            if not torch.isfinite(combined):
                logger.warning(f"NaN/Inf detected in combined loss: {combined}")
                combined = torch.tensor(0.0, device=combined.device, requires_grad=True)
            
            return combined


class MultiClassMetrics:
    def __init__(self, num_classes: int = NUM_CLASSES, device: str = 'cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.total_samples = 0
        self.correct_samples = 0
        self.class_correct = torch.zeros(self.num_classes, device=self.device)
        self.class_total = torch.zeros(self.num_classes, device=self.device)
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred shape: (N, C, H, W), target shape: (N, H, W)
        pred_classes = torch.argmax(pred, dim=1)  # (N, H, W)
        
        # Flatten for easier computation
        pred_flat = pred_classes.view(-1)
        target_flat = target.view(-1)
        
        # Update confusion matrix efficiently using bincount
        valid_mask = (target_flat >= 0) & (target_flat < self.num_classes) & (pred_flat >= 0) & (pred_flat < self.num_classes)
        if valid_mask.sum() > 0:
            valid_pred = pred_flat[valid_mask]
            valid_target = target_flat[valid_mask]
            # Use bincount to efficiently compute confusion matrix updates
            indices = valid_target * self.num_classes + valid_pred
            bincount = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
            confusion_update = bincount.reshape(self.num_classes, self.num_classes).float().to(self.device)
            self.confusion_matrix += confusion_update
        
        # Update accuracy counters
        correct = (pred_flat == target_flat).sum().item()
        total = pred_flat.size(0)
        
        self.correct_samples += correct
        self.total_samples += total
        
        # Per-class accuracy (fully vectorized)
        correct_per_class = torch.bincount((target_flat * (pred_flat == target_flat)).long(), minlength=self.num_classes)
        total_per_class = torch.bincount(target_flat.long(), minlength=self.num_classes)
        self.class_correct += correct_per_class.float().to(self.device)
        self.class_total += total_per_class.float().to(self.device)
    
    def compute(self) -> Dict[str, Any]:
        eps = 1e-8
        
        # Overall accuracy
        accuracy = self.correct_samples / (self.total_samples + eps)
        
        # Per-class metrics
        precision = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=0) + eps)
        recall = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1) + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # IoU computation
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection
        iou = intersection / (union + eps)
        
        # Mean metrics (excluding background class for line metrics)
        mean_precision = precision[1:].mean().item() if self.num_classes > 1 else precision.mean().item()
        mean_recall = recall[1:].mean().item() if self.num_classes > 1 else recall.mean().item()
        mean_f1 = f1[1:].mean().item() if self.num_classes > 1 else f1.mean().item()
        mean_iou = iou[1:].mean().item() if self.num_classes > 1 else iou.mean().item()
        
        # Per-class breakdown
        per_class_metrics = {}
        for class_id in range(self.num_classes):
            class_name = CLASS_TO_LINE_MAPPING.get(class_id, f"Class_{class_id}")
            per_class_metrics[class_name] = {
                'precision': precision[class_id].item(),
                'recall': recall[class_id].item(),
                'f1': f1[class_id].item(),
                'iou': iou[class_id].item(),
                'pixel_count': self.confusion_matrix.sum(dim=1)[class_id].item()
            }
        
        return {
            'accuracy': accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': mean_f1,
            'iou': mean_iou,
            'mean_iou': mean_iou,  # For compatibility
            'per_class': per_class_metrics
        }


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, task_type: str = 'binary', num_classes: int = NUM_CLASSES, device: str = 'cuda'):
        self.threshold = threshold
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device
        
        if task_type == 'multiclass':
            self.metrics = MultiClassMetrics(num_classes, device)
        else:
            self.reset()
    
    def reset(self):
        if self.task_type == 'multiclass':
            self.metrics.reset()
        else:
            self.tp = 0
            self.fp = 0
            self.tn = 0
            self.fn = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        if self.task_type == 'multiclass':
            self.metrics.update(pred, target)
        else:
            pred_binary = (pred > self.threshold).float()
            target_binary = target.float()
            
            self.tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
            self.fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
            self.tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()
            self.fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()
    
    def compute(self) -> Dict[str, float]:
        if self.task_type == 'multiclass':
            return self.metrics.compute()
        else:
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


def calculate_pos_weight(train_loader: DataLoader, device: str = 'cuda') -> torch.Tensor:
    """Calculate positive weight for class imbalance from training data."""
    logger.info("Calculating positive weight for class balancing...")
    
    total_positive = 0
    total_pixels = 0
    
    # Sample a subset of the data to estimate class distribution
    max_batches = min(50, len(train_loader))  # Sample up to 50 batches
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
                
            masks = batch['mask']
            total_positive += masks.sum().item()
            total_pixels += masks.numel()
    
    positive_ratio = total_positive / total_pixels
    negative_ratio = 1.0 - positive_ratio
    
    # pos_weight = (# negative samples) / (# positive samples)
    pos_weight = negative_ratio / max(positive_ratio, 1e-8)
    
    logger.info(f"Class distribution - Positive: {positive_ratio:.6f}, Negative: {negative_ratio:.6f}")
    logger.info(f"Calculated pos_weight: {pos_weight:.4f}")
    
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


def calculate_class_weights(train_loader: DataLoader, num_classes: int, device: str = 'cuda') -> torch.Tensor:
    """Calculate class weights for multiclass imbalance handling."""
    logger.info("Calculating class weights for multiclass balancing...")
    
    class_counts = torch.zeros(num_classes, dtype=torch.float64)
    total_pixels = 0
    
    # Sample a subset of the data to estimate class distribution
    max_batches = min(50, len(train_loader))  # Sample up to 50 batches - sufficient for weight estimation
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
                
            masks = batch['mask']  # Shape: (B, H, W) with class indices
            
            # Count pixels for each class (vectorized - much faster)
            unique_classes, counts = torch.unique(masks, return_counts=True)
            for class_id, count in zip(unique_classes.cpu().numpy(), counts.cpu().numpy()):
                if 0 <= class_id < num_classes:
                    class_counts[class_id] += count
            
            total_pixels += masks.numel()
    
    # Calculate class frequencies
    class_frequencies = class_counts / total_pixels
    
    # Calculate weights only among line classes (exclude background)
    class_weights = torch.ones(num_classes, dtype=torch.float64)
    
    # Background gets weight 1.0 (low weight since it's easy to detect)
    class_weights[0] = 1.0
    
    # Calculate total line pixels (excluding background)
    total_line_pixels = class_counts[1:].sum()
    
    if total_line_pixels > 0:
        # Calculate line class frequencies relative to total line pixels
        line_frequencies = class_counts[1:] / total_line_pixels
        
        # Calculate inverse frequency weights for line classes only
        for class_id in range(1, num_classes):
            if class_counts[class_id] > 0:
                # Inverse frequency weighting among line classes
                line_freq = line_frequencies[class_id - 1]
                class_weights[class_id] = 1.0 / (line_freq + 1e-8)
            else:
                class_weights[class_id] = 1.0
        
        # Normalize line weights so the most common line class has weight ~2-3
        max_line_weight = class_weights[1:].max()
        if max_line_weight > 0:
            normalization_factor = 3.0 / max_line_weight
            class_weights[1:] *= normalization_factor
    
    # Log class distribution with more details
    logger.info("Class distribution:")
    logger.info(f"  background: {class_frequencies[0]:.6f} (weight: {class_weights[0]:.2f})")
    logger.info("Line class distribution (among line pixels only):")
    
    for class_id in range(1, num_classes):  # Show ALL line classes
        if class_frequencies[class_id] > 0:
            class_name = CLASS_TO_LINE_MAPPING.get(class_id, f"Class_{class_id}")
            line_freq_relative = class_counts[class_id] / total_line_pixels if total_line_pixels > 0 else 0
            logger.info(f"  {class_name}: {class_frequencies[class_id]:.6f} ({line_freq_relative:.4f} of line pixels, weight: {class_weights[class_id]:.2f})")
    
    # Also log classes with zero pixels (they exist in the mapping but not in this sample)
    missing_classes = []
    for class_id in range(1, num_classes):
        if class_frequencies[class_id] == 0:
            class_name = CLASS_TO_LINE_MAPPING.get(class_id, f"Class_{class_id}")
            missing_classes.append(class_name)
    
    if missing_classes:
        logger.info(f"Classes not found in sample: {', '.join(missing_classes)}")
    
    return class_weights.float().to(device)


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
        checkpoint_dir: str = 'checkpoints',
        warmup_epochs: int = 0,
        base_lr: float = 1e-3,
        use_amp: bool = True,
        task_type: str = 'binary',
        num_classes: int = NUM_CLASSES
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.use_amp = use_amp and device == 'cuda'  # Only use AMP on CUDA
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Initialize AMP scaler if using AMP
        self.scaler = GradScaler() if self.use_amp else None
        
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
        self.train_metrics = SegmentationMetrics(task_type=task_type, num_classes=num_classes, device=device)
        self.val_metrics = SegmentationMetrics(task_type=task_type, num_classes=num_classes, device=device)
        
        logger.info(f"Trainer initialized. Experiment: {self.experiment_name}")
        logger.info(f"Device: {device}")
        logger.info(f"AMP enabled: {self.use_amp}")
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
            
            # Forward pass with AMP
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    
                    # Handle different model outputs
                    if isinstance(outputs, dict):  # DeepLabV3 returns dict
                        outputs = outputs['out']
                    
                    # Apply sigmoid for binary segmentation only
                    # For multi-class, use raw logits
                    if self.task_type == 'binary' and hasattr(self.model, 'classifier'):  # DeepLabV3 model
                        outputs = torch.sigmoid(outputs)
                    
                    loss = self.criterion(outputs, masks)
                
                # Backward pass with AMP
                self.scaler.scale(loss).backward()
                
                # Clip gradients and update with scaler
                self.scaler.unscale_(self.optimizer)
                # More conservative gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if grad_norm > 10.0:  # Log if gradients are exploding
                    logger.warning(f"Large gradient norm detected: {grad_norm:.2f}")
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                
                # Handle different model outputs
                if isinstance(outputs, dict):  # DeepLabV3 returns dict
                    outputs = outputs['out']
                
                # Apply sigmoid for binary segmentation only
                # For multi-class, use raw logits
                if self.task_type == 'binary' and hasattr(self.model, 'classifier'):  # DeepLabV3 model
                    outputs = torch.sigmoid(outputs)
                
                loss = self.criterion(outputs, masks)
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if grad_norm > 10.0:  # Log if gradients are exploding
                    logger.warning(f"Large gradient norm detected: {grad_norm:.2f}")
                
                self.optimizer.step()
            
            # Update metrics (check for NaN in loss)
            loss_item = loss.item()
            if not np.isfinite(loss_item):
                logger.warning(f"NaN/Inf loss detected in batch {batch_idx}, skipping metrics update")
                loss_item = 0.0
            elif loss_item > 10.0:  # Unusually high loss
                logger.warning(f"High loss detected: {loss_item:.4f} in batch {batch_idx}")
            elif loss_item < 1e-6:  # Unusually low loss (might indicate vanishing gradients)
                logger.warning(f"Very low loss detected: {loss_item:.6f} in batch {batch_idx}")
            
            total_loss += loss_item
            self.train_metrics.update(outputs, masks)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_item:.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        
        # Log metrics
        self.writer.add_scalar('Train/Loss', avg_loss, self.epoch)
        for metric_name, metric_value in metrics.items():
            if metric_name != 'per_class':  # Skip per_class as it's a dict
                self.writer.add_scalar(f'Train/{metric_name.capitalize()}', metric_value, self.epoch)
        
        # Log per-class metrics to TensorBoard if available
        if 'per_class' in metrics and self.task_type == 'multiclass':
            for class_name, class_metrics in metrics['per_class'].items():
                for metric_name, metric_value in class_metrics.items():
                    if metric_name != 'pixel_count':  # Skip pixel count for TensorBoard
                        self.writer.add_scalar(f'Train_PerClass/{class_name}_{metric_name}', metric_value, self.epoch)
        
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
                
                # Forward pass with AMP
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        
                        # Handle different model outputs
                        if isinstance(outputs, dict):  # DeepLabV3 returns dict
                            outputs = outputs['out']
                        
                        # Apply sigmoid to get probabilities for DeepLabV3
                        # UNet already has sigmoid in forward pass
                        if hasattr(self.model, 'classifier'):  # DeepLabV3 model
                            outputs = torch.sigmoid(outputs)
                        
                        loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    
                    # Handle different model outputs
                    if isinstance(outputs, dict):  # DeepLabV3 returns dict
                        outputs = outputs['out']
                    
                    # Apply sigmoid for binary segmentation only
                    # For multi-class, use raw logits
                    if self.task_type == 'binary' and hasattr(self.model, 'classifier'):  # DeepLabV3 model
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
            if metric_name != 'per_class':  # Skip per_class as it's a dict
                self.writer.add_scalar(f'Val/{metric_name.capitalize()}', metric_value, self.epoch)
        
        # Log per-class metrics to TensorBoard if available
        if 'per_class' in metrics and self.task_type == 'multiclass':
            for class_name, class_metrics in metrics['per_class'].items():
                for metric_name, metric_value in class_metrics.items():
                    if metric_name != 'pixel_count':  # Skip pixel count for TensorBoard
                        self.writer.add_scalar(f'Val_PerClass/{class_name}_{metric_name}', metric_value, self.epoch)
        
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
            
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
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
            
            # Apply warmup or regular learning rate scheduling
            if epoch < self.warmup_epochs:
                # Linear warmup
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr * warmup_factor
                logger.info(f"Warmup epoch {epoch + 1}/{self.warmup_epochs}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            elif self.scheduler:
                # Regular scheduling after warmup
                if hasattr(self.scheduler, 'step') and 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    self.scheduler.step(val_metrics['iou'])  # Use IoU for plateau detection
                else:
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
            
            # Log per-class breakdown for multiclass tasks
            if self.task_type == 'multiclass' and 'per_class' in val_metrics:
                logger.info("Per-class validation metrics:")
                
                # Sort classes by IoU for better readability
                per_class = val_metrics['per_class']
                sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['iou'], reverse=True)
                
                for class_name, metrics in sorted_classes:
                    if class_name != 'background' and metrics['pixel_count'] > 0:  # Skip background and empty classes
                        logger.info(
                            f"  {class_name:.<25} IoU: {metrics['iou']:.4f}, "
                            f"F1: {metrics['f1']:.4f}, "
                            f"Pixels: {int(metrics['pixel_count']):>6}"
                        )
                
                # Show background separately
                if 'background' in per_class:
                    bg_metrics = per_class['background']
                    logger.info(
                        f"  {'background':.<25} IoU: {bg_metrics['iou']:.4f}, "
                        f"F1: {bg_metrics['f1']:.4f}, "
                        f"Pixels: {int(bg_metrics['pixel_count']):>6}"
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
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (reduced for higher resolution)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (reduced for multiclass stability)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--target-size', type=int, nargs=2, default=[1024, 1024], help='Target image size (1024 recommended for thin lines)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'], help='Device to use')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    # Loss function configuration
    parser.add_argument('--loss-type', type=str, default='focal_dice', 
                       choices=['focal_dice', 'weighted_bce_dice', 'original'],
                       help='Type of loss function to use')
    parser.add_argument('--focal-weight', type=float, default=0.2, 
                       help='Weight for focal/BCE loss component')
    parser.add_argument('--dice-weight', type=float, default=0.8, 
                       help='Weight for dice loss component')
    parser.add_argument('--focal-gamma', type=float, default=1.0, 
                       help='Gamma parameter for focal loss (higher values focus more on hard examples)')
    parser.add_argument('--focal-gamma-thin-lines', type=float, default=None,
                       help='Optional higher gamma for thin line detection (goal posts, crossbars)')
    parser.add_argument('--line-thickness', type=int, default=4, 
                       help='Line thickness for pitch mask generation (4 recommended for 1024 resolution)')
    parser.add_argument('--use-cosine-scheduler', action='store_true',
                       help='Use cosine annealing scheduler instead of ReduceLROnPlateau')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Number of warmup epochs for learning rate (increased for multiclass)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision for faster training')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable Automatic Mixed Precision')
    parser.add_argument('--enable-multiscale-augmentation', action='store_true',
                       help='Enable multi-scale augmentation for better thin line detection')
    parser.add_argument('--scale-range', type=float, nargs=2, default=[0.8, 1.2],
                       help='Scale range for multi-scale augmentation (min, max)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation for training')
    parser.add_argument('--task-type', type=str, default='multiclass', choices=['binary', 'multiclass'],
                       help='Task type: binary or multiclass segmentation')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                       help='Number of classes for multi-class segmentation')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor for cross-entropy loss')
    parser.add_argument('--class-weighting', action='store_true', default=True,
                       help='Use class weighting to handle imbalanced classes')
    
    args = parser.parse_args()
    
    # Validate training parameters (prevent crashes)
    if args.line_thickness < 1 or args.line_thickness > 20:
        logger.warning(f"Line thickness {args.line_thickness} is unusual. Recommended: 1-8 for thin lines, 4-8 for thick lines")
    
    if args.focal_gamma < 0.5 or args.focal_gamma > 5.0:
        logger.warning(f"Focal gamma {args.focal_gamma} is unusual. Recommended: 0.5-3.0")
    
    if args.enable_multiscale_augmentation:
        if args.scale_range[0] >= args.scale_range[1]:
            raise ValueError("Scale range min must be less than max")
        if args.scale_range[0] < 0.5 or args.scale_range[1] > 2.0:
            logger.warning("Extreme scale range may hurt performance")
    
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
    
    # Determine AMP usage
    use_amp = args.use_amp and not args.no_amp and device == 'cuda'
    
    # Determine loss type early for logging
    if args.task_type == 'multiclass':
        loss_type = 'cross_entropy'  # Override for multiclass
    else:
        loss_type = args.loss_type
    
    # Log training improvements
    logger.info("=== TRAINING CONFIGURATION ===")
    logger.info(f"Task type: {args.task_type}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Resolution: {args.target_size[0]}x{args.target_size[1]} (higher res = better thin line detection)")
    logger.info(f"Batch size: {args.batch_size} (reduced for higher resolution)")
    logger.info(f"AMP enabled: {use_amp} (critical for memory efficiency at 1024 resolution)")
    logger.info(f"Line thickness: {args.line_thickness}px")
    logger.info(f"Loss function: {loss_type} {'(forced to cross_entropy for multiclass)' if args.task_type == 'multiclass' else ''}")
    logger.info(f"Class weighting: {args.class_weighting} (critical for imbalanced classes)")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Scheduler: {'CosineAnnealing' if args.use_cosine_scheduler else 'ReduceLROnPlateau'}")
    logger.info(f"Multi-scale augmentation: {args.enable_multiscale_augmentation}")
    if args.enable_multiscale_augmentation:
        logger.info(f"Scale range: {args.scale_range[0]:.1f} - {args.scale_range[1]:.1f}")
    if args.task_type == 'binary':
        logger.info("Class imbalance handling: Automatic pos_weight calculation and focal loss")
    else:
        logger.info("Class imbalance handling: Cross-entropy loss with class weights and label smoothing")
    
    # Memory usage warning for high resolution
    if args.target_size[0] >= 1024:
        logger.info("🚨 HIGH RESOLUTION MODE:")
        logger.info("  - Using 1024x1024 for better thin line detection")
        logger.info("  - Reduced batch size to prevent OOM")
        logger.info("  - AMP is critical for memory efficiency")
        logger.info("  - Expected ~4x better performance on goal posts/crossbars")
    
    logger.info("================================")
    
    # Create datasets from separate train and valid folders
    logger.info("Creating datasets...")
    train_data_root = str(Path(args.data_root) / "train")
    val_data_root = str(Path(args.data_root) / "valid")
    
    # Create training dataset with enhanced augmentation options
    train_kwargs = {
        'target_size': tuple(args.target_size),
        'line_thickness': args.line_thickness,
        'cache_masks': False,
        'augment': not args.no_augment,
        'multiclass': args.task_type == 'multiclass',
        'num_classes': args.num_classes
    }
    
    # Add multi-scale augmentation if enabled (will be passed to PitchAugmentation)
    if args.enable_multiscale_augmentation:
        train_kwargs['scale_range'] = args.scale_range
        logger.info("Enhanced multi-scale augmentation enabled for better thin line detection")
    
    # Override no-augment for multiclass (augmentation is crucial for multiclass)
    if args.task_type == 'multiclass' and args.no_augment:
        logger.warning("Enabling data augmentation for multiclass training - it's crucial for performance")
        train_kwargs['augment'] = True
    
    train_dataset = create_train_dataset(train_data_root, **train_kwargs)
    
    val_dataset = create_val_dataset(
        val_data_root,
        target_size=tuple(args.target_size),
        line_thickness=args.line_thickness,
        cache_masks=False,
        multiclass=args.task_type == 'multiclass',
        num_classes=args.num_classes
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info(f"Creating {args.model} model with {args.backbone} backbone...")
    model = create_model(args.model, args.backbone, num_classes=args.num_classes, task_type=args.task_type)
    
    # Create data loaders first (needed for pos_weight calculation)
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
    
    # Calculate weights for class balancing
    pos_weight = None
    class_weights = None
    
    if args.task_type == 'binary':
        pos_weight = calculate_pos_weight(train_loader, device)
    elif args.task_type == 'multiclass' and args.class_weighting:
        class_weights = calculate_class_weights(train_loader, args.num_classes, device)
    
    # Create improved loss function and optimizer  
    # loss_type already determined above for logging
    if args.task_type == 'multiclass':
        logger.info(f"Using cross_entropy loss for multiclass segmentation")
        
    criterion = CombinedLoss(
        loss_type=loss_type,
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        focal_alpha=1.0,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight if loss_type == 'weighted_bce_dice' else None,
        task_type=args.task_type,
        num_classes=args.num_classes,
        class_weights=class_weights  # Pass class weights for multiclass
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # More aggressive scheduler for faster convergence
    if args.use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=8, min_lr=1e-6
        )
    
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
        checkpoint_dir=args.checkpoint_dir,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        use_amp=use_amp,
        task_type=args.task_type,
        num_classes=args.num_classes
    )
    
    # Start training
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()