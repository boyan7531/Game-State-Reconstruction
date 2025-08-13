import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from train import create_model
from dataset import PitchLocalizationDataset, create_val_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchInference:
    def __init__(self, model_path: str, device: str = 'auto'):
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = 'mps'
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (you may need to adjust this based on your saved model info)
        self.model = create_model('deeplabv3', 'resnet50', num_classes=1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict mask for a single image tensor."""
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
            output = self.model(image)
            
            # Handle different model outputs
            if isinstance(output, dict):
                output = output['out']
            
            # Apply sigmoid and remove batch dimension
            pred_mask = torch.sigmoid(output).squeeze(0).cpu()
            
            return pred_mask
    
    def predict_image_path(self, image_path: str, target_size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mask for image file path."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Resize and normalize
        image_resized = image.resize(target_size)
        image_tensor = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        normalize = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - normalize) / std
        
        # Predict
        pred_mask = self.predict(image_tensor)
        
        # Resize prediction back to original size
        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0),
            size=original_size[::-1],  # PIL uses (W, H), torch uses (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return np.array(image), pred_mask_resized
    
    def batch_predict_dataset(self, dataset: PitchLocalizationDataset, threshold: float = 0.5) -> List[Dict]:
        """Predict masks for entire dataset."""
        results = []
        
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            sample = dataset[i]
            image_tensor = sample['image']
            gt_mask = sample['mask'].squeeze().numpy()
            metadata = sample['metadata']
            
            # Predict
            pred_mask = self.predict(image_tensor).squeeze().numpy()
            
            # Apply threshold
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            
            # Calculate metrics
            intersection = (pred_binary & gt_binary).sum()
            union = (pred_binary | gt_binary).sum()
            iou = intersection / (union + 1e-8)
            
            tp = (pred_binary & gt_binary).sum()
            fp = (pred_binary & ~gt_binary).sum()
            fn = (~pred_binary & gt_binary).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            results.append({
                'index': i,
                'frame_path': metadata['frame_path'],
                'sequence': metadata['sequence'],
                'frame_name': metadata['frame_name'],
                'pred_mask': pred_mask,
                'gt_mask': gt_mask,
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        return results


def visualize_predictions(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    save_path: Optional[str] = None
):
    """Visualize prediction results."""
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(pred_mask, cmap='hot', alpha=0.7)
    axes[1].imshow(image, alpha=0.3)
    axes[1].set_title(f'Predicted Mask (threshold={threshold})')
    axes[1].axis('off')
    
    # Ground truth mask (if available)
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap='hot', alpha=0.7)
        axes[2].imshow(image, alpha=0.3)
        axes[2].set_title('Ground Truth Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_comparison_grid(results: List[Dict], num_samples: int = 9, save_path: str = 'comparison_grid.png'):
    """Create a grid comparison of predictions vs ground truth."""
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    for i in range(min(num_samples, len(results))):
        result = results[i]
        
        # Load original image
        try:
            image = np.array(Image.open(result['frame_path']).convert('RGB'))
        except:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        pred_mask = result['pred_mask']
        gt_mask = result['gt_mask']
        
        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"IoU: {result['iou']:.3f}")
        axes[0, i].axis('off')
        
        # Predicted mask overlay
        axes[1, i].imshow(image)
        axes[1, i].imshow(pred_mask, cmap='Reds', alpha=0.6)
        axes[1, i].set_title('Prediction')
        axes[1, i].axis('off')
        
        # Ground truth mask overlay
        axes[2, i].imshow(image)
        axes[2, i].imshow(gt_mask, cmap='Greens', alpha=0.6)
        axes[2, i].set_title('Ground Truth')
        axes[2, i].axis('off')
    
    # Remove empty subplots
    for i in range(num_samples, len(axes[0])):
        for row in axes:
            row[i].remove()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison grid saved to {save_path}")


def evaluate_model(model_path: str, data_root: str, output_dir: str = 'inference_results'):
    """Comprehensive model evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize inference
    inference = PitchInference(model_path)
    
    # Create validation dataset
    val_dataset = create_val_dataset(data_root, target_size=(512, 512))
    logger.info(f"Evaluating on {len(val_dataset)} samples")
    
    # Run predictions
    results = inference.batch_predict_dataset(val_dataset)
    
    # Calculate overall metrics
    ious = [r['iou'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    overall_metrics = {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1s),
        'num_samples': len(results)
    }
    
    # Log metrics
    logger.info("Evaluation Results:")
    logger.info(f"Mean IoU: {overall_metrics['mean_iou']:.4f} ± {overall_metrics['std_iou']:.4f}")
    logger.info(f"Mean Precision: {overall_metrics['mean_precision']:.4f}")
    logger.info(f"Mean Recall: {overall_metrics['mean_recall']:.4f}")
    logger.info(f"Mean F1: {overall_metrics['mean_f1']:.4f}")
    
    # Save metrics
    with open(output_dir / 'evaluation_metrics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {k: float(v) if isinstance(v, np.number) else v 
                       for k, v in overall_metrics.items()}
        json.dump(json_metrics, f, indent=2)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Sort results by IoU for better visualization
    results_sorted = sorted(results, key=lambda x: x['iou'], reverse=True)
    
    # Best and worst predictions
    best_results = results_sorted[:9]
    worst_results = results_sorted[-9:]
    
    create_comparison_grid(best_results, save_path=output_dir / 'best_predictions.png')
    create_comparison_grid(worst_results, save_path=output_dir / 'worst_predictions.png')
    
    # IoU histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title(f'IoU Distribution (Mean: {overall_metrics["mean_iou"]:.3f})')
    plt.axvline(overall_metrics['mean_iou'], color='red', linestyle='--', 
                label=f'Mean: {overall_metrics["mean_iou"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'iou_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics by sequence
    sequence_metrics = {}
    for result in results:
        seq = result['sequence']
        if seq not in sequence_metrics:
            sequence_metrics[seq] = []
        sequence_metrics[seq].append(result['iou'])
    
    seq_names = []
    seq_mean_ious = []
    for seq, ious in sequence_metrics.items():
        seq_names.append(seq)
        seq_mean_ious.append(np.mean(ious))
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(seq_names)), seq_mean_ious)
    plt.xlabel('Sequence')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU by Sequence')
    plt.xticks(range(len(seq_names)), seq_names, rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'iou_by_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")
    
    return overall_metrics, results


def main():
    parser = argparse.ArgumentParser(description='Inference for pitch localization model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data-root', type=str, default='SoccerNet/SN-GSR-2025/train',
                       help='Path to SoccerNet GSR dataset')
    parser.add_argument('--image-path', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run full evaluation on validation set')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    
    args = parser.parse_args()
    
    if args.image_path:
        # Single image inference
        inference = PitchInference(args.model_path)
        image, pred_mask = inference.predict_image_path(args.image_path)
        
        # Apply threshold
        pred_binary = (pred_mask > args.threshold).astype(np.float32)
        
        # Visualize
        output_path = Path(args.output_dir) / f"prediction_{Path(args.image_path).stem}.png"
        output_path.parent.mkdir(exist_ok=True)
        
        visualize_predictions(image, pred_binary, save_path=str(output_path))
        logger.info(f"Prediction saved to {output_path}")
        
    elif args.evaluate:
        # Full evaluation
        evaluate_model(args.model_path, args.data_root, args.output_dir)
    
    else:
        print("Please specify either --image-path for single inference or --evaluate for full evaluation")


if __name__ == '__main__':
    main()