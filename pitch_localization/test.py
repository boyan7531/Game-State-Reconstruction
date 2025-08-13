#!/usr/bin/env python3
"""
Test script for pitch localization model.
Loads a trained model and creates a video with predicted pitch lines overlaid.
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from train import create_model
from dataset import PitchLocalizationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchVideoProcessor:
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the video processor with a trained model."""
        # Set device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Image preprocessing parameters
        self.target_size = (512, 512)
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (adjust architecture if needed based on checkpoint)
        model = create_model('deeplabv3', 'resnet50', num_classes=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for model input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and convert to tensor
        frame_pil = Image.fromarray(frame_rgb)
        frame_resized = frame_pil.resize(self.target_size)
        frame_tensor = torch.from_numpy(np.array(frame_resized)).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        frame_tensor = (frame_tensor - self.normalize_mean) / self.normalize_std
        
        return frame_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
    
    def predict_mask(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict pitch lines mask for a frame."""
        with torch.no_grad():
            # Preprocess
            frame_tensor = self.preprocess_frame(frame)
            
            # Predict
            output = self.model(frame_tensor)
            if isinstance(output, dict):
                output = output['out']
            
            # Apply sigmoid and threshold
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            
            # Resize back to original frame size
            h, w = frame.shape[:2]
            pred_resized = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            
            return pred_resized
    
    def overlay_mask_on_frame(self, frame: np.ndarray, mask: np.ndarray, 
                             color: tuple = (0, 255, 0), alpha: float = 0.7) -> np.ndarray:
        """Overlay predicted mask on the original frame."""
        overlay = frame.copy()
        
        # Create colored mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = color
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
        
        # Add mask contours for better visibility
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    
    def _get_gt_mask(self, annotations: dict, frame_name: str, width: int, height: int) -> np.ndarray:
        """Generate ground truth mask from annotations."""
        # Find the specific image
        target_image = None
        for img in annotations.get("images", []):
            if img.get("file_name") == frame_name:
                target_image = img
                break
        
        if target_image is None or not target_image.get("has_labeled_pitch", False):
            return None
        
        # Find annotations for this image
        pitch_annotations = []
        image_id = target_image.get("image_id")
        
        for ann in annotations.get("annotations", []):
            if (ann.get("image_id") == image_id and 
                ann.get("supercategory") == "pitch"):
                pitch_annotations.append(ann)
        
        if not pitch_annotations:
            return None
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in pitch_annotations:
            lines = ann.get("lines", {})
            
            for line_name, line_coords in lines.items():
                if not isinstance(line_coords, list) or len(line_coords) == 0:
                    continue
                    
                # Convert normalized coordinates to pixels
                points = []
                for coord in line_coords:
                    try:
                        if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                            x_norm, y_norm = float(coord['x']), float(coord['y'])
                        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            x_norm, y_norm = float(coord[0]), float(coord[1])
                        else:
                            continue
                        
                        x_px = int(x_norm * width)
                        y_px = int(y_norm * height)
                        points.append((x_px, y_px))
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if len(points) >= 2:
                    points_array = np.array(points, dtype=np.int32)
                    
                    # Draw lines based on type
                    if line_name in ['center_circle', 'left_penalty_arc', 'right_penalty_arc']:
                        cv2.polylines(mask, [points_array], isClosed=True, color=255, thickness=2)
                    elif line_name in ['penalty_area_left', 'penalty_area_right', 'goal_area_left', 'goal_area_right']:
                        cv2.polylines(mask, [points_array], isClosed=True, color=255, thickness=2)
                    else:
                        cv2.polylines(mask, [points_array], isClosed=False, color=255, thickness=2)
        
        return mask
    
    def process_sequence_to_video(self, sequence_dir: str, output_path: str, 
                                 fps: int = 10, max_frames: int = None,
                                 threshold: float = 0.5, show_comparison: bool = True):
        """Process an entire sequence and create a video."""
        sequence_dir = Path(sequence_dir)
        img_dir = sequence_dir / "img1"
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        
        # Get all frame files
        frame_files = sorted(list(img_dir.glob("*.jpg")))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        logger.info(f"Processing {len(frame_files)} frames from {sequence_dir.name}")
        
        # Initialize video writer
        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        
        if show_comparison:
            # Side-by-side: original | predicted
            video_width = w * 2
            video_height = h
        else:
            # Only overlay
            video_width = w
            video_height = h
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
        
        # Load ground truth annotations only if needed for comparison
        gt_annotations = None
        if show_comparison:
            try:
                import json
                json_path = sequence_dir / "Labels-GameState.json"
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        gt_annotations = json.load(f)
                    logger.info("Ground truth annotations loaded for comparison")
                else:
                    logger.warning(f"No ground truth file found: {json_path}")
                    show_comparison = False  # Disable comparison if no GT available
            except Exception as e:
                logger.warning(f"Could not load ground truth: {e}")
                show_comparison = False
        
        # Process frames
        for i, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            # Predict mask
            pred_mask = self.predict_mask(frame, threshold)
            
            # Create overlay
            overlay_frame = self.overlay_mask_on_frame(frame, pred_mask, color=(0, 255, 0), alpha=0.6)
            
            if show_comparison:
                # Try to get ground truth
                gt_overlay = frame.copy()
                if gt_annotations:
                    try:
                        frame_name = frame_file.name
                        gt_mask = self._get_gt_mask(gt_annotations, frame_name, w, h)
                        if gt_mask is not None:
                            # Create GT overlay
                            gt_overlay = self.overlay_mask_on_frame(
                                frame, gt_mask, color=(0, 0, 255), alpha=0.6
                            )
                    except Exception as e:
                        logger.debug(f"Could not get ground truth for frame {i}: {e}")
                
                # Create side-by-side comparison
                comparison = np.hstack([gt_overlay, overlay_frame])
                
                # Add labels
                cv2.putText(comparison, "Ground Truth", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(comparison, "Prediction", (w + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(comparison)
            else:
                # Add frame info
                cv2.putText(overlay_frame, f"Frame {i+1}/{len(frame_files)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(overlay_frame, f"Threshold: {threshold}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(overlay_frame)
        
        out.release()
        logger.info(f"Video saved to {output_path}")
    
    def process_single_frame(self, frame_path: str, output_path: str, threshold: float = 0.5):
        """Process a single frame and save the result."""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load image: {frame_path}")
        
        # Predict mask
        pred_mask = self.predict_mask(frame, threshold)
        
        # Create overlay
        overlay_frame = self.overlay_mask_on_frame(frame, pred_mask, color=(0, 255, 0), alpha=0.6)
        
        # Save result
        cv2.imwrite(output_path, overlay_frame)
        logger.info(f"Processed frame saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test pitch localization model on video sequences')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--sequence-dir', type=str, required=True,
                       help='Path to sequence directory (e.g., SoccerNet/SN-GSR-2025/train/SNGS-062)')
    parser.add_argument('--output-path', type=str, default='output_video.mp4',
                       help='Output video path')
    parser.add_argument('--frame-path', type=str, default=None,
                       help='Process single frame instead of video')
    parser.add_argument('--fps', type=int, default=10,
                       help='Output video FPS')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum number of frames to process (None for all)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Do not show ground truth comparison')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PitchVideoProcessor(args.model_path, args.device)
    
    if args.frame_path:
        # Process single frame
        processor.process_single_frame(args.frame_path, args.output_path, args.threshold)
    else:
        # Process sequence to video
        processor.process_sequence_to_video(
            args.sequence_dir,
            args.output_path,
            fps=args.fps,
            max_frames=args.max_frames,
            threshold=args.threshold,
            show_comparison=not args.no_comparison
        )


if __name__ == '__main__':
    main()