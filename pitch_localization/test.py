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
from sklearn.metrics import confusion_matrix, classification_report
import json

from train import create_model
from dataset import PitchLocalizationDataset, NUM_CLASSES, LINE_CLASS_MAPPING, CLASS_TO_LINE_MAPPING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchVideoProcessor:
    def __init__(self, model_path: str, device: str = 'auto', task_type: str = 'auto', num_classes: int = None):
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
        
        # Load model and determine task type
        self.model, self.task_type, self.num_classes = self._load_model(model_path, task_type, num_classes)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Task type: {self.task_type}, Number of classes: {self.num_classes}")
        
        # Image preprocessing parameters
        self.target_size = (512, 512)
        self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def _load_model(self, model_path: str, task_type: str = 'auto', num_classes: int = None):
        """Load the trained model and determine its configuration."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to determine task type and num_classes from checkpoint or infer from model
        if task_type == 'auto' or num_classes is None:
            # First, try to create a test model to inspect the output shape
            try:
                # Try multiclass first (most likely scenario)
                test_model = create_model('deeplabv3', 'resnet50', num_classes=NUM_CLASSES, task_type='multiclass')
                test_model.load_state_dict(checkpoint['model_state_dict'])
                detected_task_type = 'multiclass'
                detected_num_classes = NUM_CLASSES
                logger.info("Detected multiclass model")
            except Exception:
                try:
                    # Try binary as fallback
                    test_model = create_model('deeplabv3', 'resnet50', num_classes=1, task_type='binary')
                    test_model.load_state_dict(checkpoint['model_state_dict'])
                    detected_task_type = 'binary'
                    detected_num_classes = 1
                    logger.info("Detected binary model")
                except Exception as e:
                    logger.error(f"Could not determine model type: {e}")
                    # Default to multiclass
                    detected_task_type = 'multiclass'
                    detected_num_classes = NUM_CLASSES
        else:
            detected_task_type = task_type
            detected_num_classes = num_classes if num_classes is not None else (NUM_CLASSES if task_type == 'multiclass' else 1)
        
        # Create the actual model
        model = create_model('deeplabv3', 'resnet50', num_classes=detected_num_classes, task_type=detected_task_type)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, detected_task_type, detected_num_classes
    
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
    
    def predict_mask(self, frame: np.ndarray, threshold: float = 0.5, enhance_goal_posts: bool = True, multiscale: bool = False, adaptive_threshold: bool = False) -> np.ndarray:
        """Predict pitch lines mask for a frame with optional goal post enhancement and multi-scale inference."""
        with torch.no_grad():
            if multiscale:
                return self._predict_multiscale(frame, threshold, enhance_goal_posts)
            
            # Standard single-scale prediction
            # Preprocess
            frame_tensor = self.preprocess_frame(frame)
            
            # Predict
            output = self.model(frame_tensor)
            if isinstance(output, dict):
                output = output['out']
            
            if self.task_type == 'multiclass':
                # For multiclass: use softmax and argmax
                pred_probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()  # (C, H, W)
                pred_classes = np.argmax(pred_probs, axis=0).astype(np.uint8)  # (H, W)
                
                # Resize back to original frame size
                h, w = frame.shape[:2]
                pred_resized = cv2.resize(pred_classes, (w, h), interpolation=cv2.INTER_NEAREST)
                
                return pred_resized
            else:
                # For binary: use sigmoid and threshold
                pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                
                if adaptive_threshold:
                    pred_binary = self._apply_adaptive_threshold(pred_mask, threshold)
                else:
                    pred_binary = (pred_mask > threshold).astype(np.uint8)
                
                # Resize back to original frame size
                h, w = frame.shape[:2]
                pred_resized = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Goal post enhancement post-processing
                if enhance_goal_posts:
                    pred_resized = self._enhance_goal_posts(pred_resized)
                
                return pred_resized
    
    def _predict_multiscale(self, frame: np.ndarray, threshold: float = 0.5, enhance_goal_posts: bool = True) -> np.ndarray:
        """Multi-scale inference for better thin line detection."""
        h, w = frame.shape[:2]
        scales = [0.8, 1.0, 1.2]  # Different scales to capture various line thicknesses
        
        if self.task_type == 'multiclass':
            # For multiclass: collect probability distributions
            predictions = []
            
            for scale in scales:
                # Resize frame
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h))
                
                # Preprocess scaled frame
                frame_tensor = self.preprocess_frame(scaled_frame)
                
                # Predict
                output = self.model(frame_tensor)
                if isinstance(output, dict):
                    output = output['out']
                
                # Get probability distribution
                pred_probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()  # (C, H, W)
                
                # Resize back to original size
                pred_probs_resized = np.zeros((self.num_classes, h, w), dtype=np.float32)
                for c in range(self.num_classes):
                    pred_probs_resized[c] = cv2.resize(pred_probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
                
                predictions.append(pred_probs_resized)
            
            # Average predictions from different scales
            avg_probs = np.mean(predictions, axis=0)  # (C, H, W)
            
            # Get final class predictions
            pred_classes = np.argmax(avg_probs, axis=0).astype(np.uint8)  # (H, W)
            
            return pred_classes
        else:
            # For binary: original implementation
            predictions = []
            
            for scale in scales:
                # Resize frame
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h))
                
                # Preprocess scaled frame
                frame_tensor = self.preprocess_frame(scaled_frame)
                
                # Predict
                output = self.model(frame_tensor)
                if isinstance(output, dict):
                    output = output['out']
                
                # Get probability map
                pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Resize back to original size
                pred_prob_resized = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)
                predictions.append(pred_prob_resized)
            
            # Average predictions from different scales
            avg_pred = np.mean(predictions, axis=0)
            
            # Apply threshold
            pred_binary = (avg_pred > threshold).astype(np.uint8)
            
            # Goal post enhancement post-processing
            if enhance_goal_posts:
                pred_binary = self._enhance_goal_posts(pred_binary)
            
            return pred_binary
    
    def _apply_adaptive_threshold(self, pred_prob: np.ndarray, base_threshold: float) -> np.ndarray:
        """Apply adaptive thresholding based on local statistics to better detect thin structures."""
        # Use multiple thresholds for different confidence regions
        high_conf = pred_prob > (base_threshold + 0.2)  # Very confident predictions
        medium_conf = pred_prob > base_threshold  # Standard threshold
        low_conf = pred_prob > (base_threshold - 0.2)  # Lower threshold for thin structures
        
        # Create binary mask
        result = high_conf.astype(np.uint8)
        
        # Add medium confidence pixels that are connected to high confidence ones
        kernel = np.ones((3, 3), np.uint8)
        dilated_high = cv2.dilate(result, kernel, iterations=1)
        connected_medium = cv2.bitwise_and(medium_conf.astype(np.uint8), dilated_high)
        result = cv2.bitwise_or(result, connected_medium)
        
        # Add low confidence pixels that form thin lines (likely goal posts)
        # Focus on pixels that have linear structure
        thin_structure = self._detect_thin_structures(low_conf.astype(np.uint8))
        result = cv2.bitwise_or(result, thin_structure)
        
        return result
    
    def _detect_thin_structures(self, mask: np.ndarray) -> np.ndarray:
        """Detect thin linear structures that might be goal posts."""
        # Morphological operations to detect thin lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Detect horizontal lines  
        horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Combine thin line structures
        thin_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
        
        return thin_lines
            
    def _enhance_goal_posts(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to enhance thin vertical/horizontal structures (goal posts)."""
        enhanced_mask = mask.copy()
        
        # Create multiple kernel sizes for different goal post thicknesses
        # Small kernels for thin goal posts
        vertical_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        horizontal_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        
        # Larger kernels for thicker structures
        vertical_kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        horizontal_kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        
        # Multi-scale enhancement for vertical structures
        vertical_small = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel_small)
        vertical_small = cv2.morphologyEx(vertical_small, cv2.MORPH_OPEN, vertical_kernel_small)
        
        vertical_large = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vertical_kernel_large)
        vertical_large = cv2.morphologyEx(vertical_large, cv2.MORPH_OPEN, vertical_kernel_large)
        
        # Multi-scale enhancement for horizontal structures  
        horizontal_small = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel_small)
        horizontal_small = cv2.morphologyEx(horizontal_small, cv2.MORPH_OPEN, horizontal_kernel_small)
        
        horizontal_large = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel_large)
        horizontal_large = cv2.morphologyEx(horizontal_large, cv2.MORPH_OPEN, horizontal_kernel_large)
        
        # Combine all enhanced structures
        vertical_combined = cv2.bitwise_or(vertical_small, vertical_large)
        horizontal_combined = cv2.bitwise_or(horizontal_small, horizontal_large)
        goal_posts = cv2.bitwise_or(vertical_combined, horizontal_combined)
        
        # Add Hough line detection for straight goal post segments
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=10, minLineLength=15, maxLineGap=3)
        hough_mask = np.zeros_like(mask)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is primarily vertical or horizontal
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                if dy > 3 * dx or dx > 3 * dy:  # Vertical or horizontal line
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 1)
        
        # Combine all enhancements
        all_enhancements = cv2.bitwise_or(goal_posts, hough_mask)
        
        # More aggressive approach: add detected goal post pixels
        enhanced_mask = cv2.bitwise_or(mask, all_enhancements)
        
        return enhanced_mask
    
    def _get_class_colors(self):
        """Get color map for different pitch line classes."""
        # Define distinct colors for each class
        colors = {
            0: (0, 0, 0),        # Background - black
            1: (255, 0, 0),      # Side line top - red
            2: (0, 255, 0),      # Circle central - green
            3: (0, 0, 255),      # Big rect. right main - blue
            4: (255, 255, 0),    # Big rect. right top - cyan
            5: (255, 0, 255),    # Big rect. left main - magenta
            6: (0, 255, 255),    # Big rect. left top - yellow
            7: (128, 0, 128),    # Circle right - purple
            8: (255, 165, 0),    # Middle line - orange
            9: (255, 192, 203),  # Side line right - pink
            10: (0, 128, 0),     # Circle left - dark green
            11: (128, 128, 0),   # Side line bottom - olive
            12: (0, 0, 128),     # Side line left - navy
            13: (128, 0, 0),     # Small rect. right main - maroon
            14: (0, 128, 128),   # Small rect. right top - teal
            15: (128, 128, 128), # Small rect. left main - gray
            16: (255, 228, 196), # Small rect. left top - bisque
            17: (255, 20, 147),  # Goal right post left - deep pink
            18: (0, 191, 255),   # Goal right crossbar - deep sky blue
            19: (50, 205, 50),   # Goal left post right - lime green
            20: (255, 69, 0),    # Goal left crossbar - red orange
            21: (138, 43, 226),  # Goal right post right - blue violet
            22: (255, 140, 0),   # Small rect. right bottom - dark orange
            23: (32, 178, 170),  # Big rect. right bottom - light sea green
            24: (220, 20, 60),   # Goal left post left - crimson
            25: (75, 0, 130),    # Small rect. left bottom - indigo
            26: (255, 215, 0),   # Big rect. left bottom - gold
        }
        return colors
    
    def _add_class_legend(self, image: np.ndarray, colors: dict) -> np.ndarray:
        """Add a legend showing class colors and names."""
        h, w = image.shape[:2]
        legend_width = 300
        legend_height = min(h, 600)
        
        # Create legend area
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(legend, "Pitch Lines", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add class entries (only for classes 1-26, skip background)
        y_offset = 50
        line_height = 20
        
        for class_id in range(1, min(self.num_classes, 27)):  # Limit to visible area
            if class_id in CLASS_TO_LINE_MAPPING and class_id in colors:
                class_name = CLASS_TO_LINE_MAPPING[class_id]
                color = colors[class_id]
                
                # Draw colored rectangle
                cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
                
                # Add class name (truncated if too long)
                name_text = class_name[:25] + '...' if len(class_name) > 25 else class_name
                cv2.putText(legend, name_text, (35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                y_offset += line_height
                
                if y_offset > legend_height - 20:
                    break
        
        # Combine image with legend
        if w + legend_width <= 1920:  # Only add legend if it fits
            result = np.hstack([image, legend])
        else:
            result = image
        
        return result
    
    def overlay_mask_on_frame(self, frame: np.ndarray, mask: np.ndarray, 
                             color: tuple = (0, 255, 0), alpha: float = 0.7, show_legend: bool = False) -> np.ndarray:
        """Overlay predicted mask on the original frame."""
        overlay = frame.copy()
        
        if self.task_type == 'multiclass':
            # Create color map for different classes
            colors = self._get_class_colors()
            
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            for class_id in range(1, self.num_classes):  # Skip background (0)
                class_mask = (mask == class_id)
                if np.any(class_mask):
                    colored_mask[class_mask] = colors[class_id]
            
            # Blend with original frame
            result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
            
            # Add contours for each class
            for class_id in range(1, self.num_classes):  # Skip background (0)
                class_mask = (mask == class_id).astype(np.uint8) * 255
                if np.any(class_mask):
                    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(result, contours, -1, colors[class_id], 2)
            
            # Add legend if requested
            if show_legend:
                result = self._add_class_legend(result, colors)
                
        else:
            # Binary mode (original implementation)
            # Create colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            
            # Blend with original frame
            result = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
            
            # Add mask contours for better visibility
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    
    def _get_gt_masks_by_line_type(self, annotations: dict, frame_name: str, width: int, height: int) -> dict:
        """Generate ground truth masks by line type from annotations."""
        # Find the specific image
        target_image = None
        for img in annotations.get("images", []):
            if img.get("file_name") == frame_name:
                target_image = img
                break
        
        if target_image is None or not target_image.get("has_labeled_pitch", False):
            return {}
        
        # Find annotations for this image
        pitch_annotations = []
        image_id = target_image.get("image_id")
        
        for ann in annotations.get("annotations", []):
            if (ann.get("image_id") == image_id and 
                ann.get("supercategory") == "pitch"):
                pitch_annotations.append(ann)
        
        if not pitch_annotations:
            return {}
        
        # Create separate masks for each line type
        line_masks = {}
        
        for ann in pitch_annotations:
            lines = ann.get("lines", {})
            
            for line_name, line_coords in lines.items():
                if not isinstance(line_coords, list) or len(line_coords) == 0:
                    continue
                
                # Initialize mask for this line type if not exists
                if line_name not in line_masks:
                    line_masks[line_name] = np.zeros((height, width), dtype=np.uint8)
                    
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
                    if 'Circle' in line_name or 'circle' in line_name.lower():
                        cv2.polylines(line_masks[line_name], [points_array], isClosed=True, color=255, thickness=2)
                    elif 'rect' in line_name.lower() or 'Rect' in line_name:
                        cv2.polylines(line_masks[line_name], [points_array], isClosed=True, color=255, thickness=2)
                    else:
                        cv2.polylines(line_masks[line_name], [points_array], isClosed=False, color=255, thickness=2)
        
        return line_masks
    
    def _get_gt_mask(self, annotations: dict, frame_name: str, width: int, height: int) -> np.ndarray:
        """Generate ground truth mask from annotations."""
        if self.task_type == 'multiclass':
            return self._get_gt_multiclass_mask(annotations, frame_name, width, height)
        else:
            # Binary mode (original implementation)
            line_masks = self._get_gt_masks_by_line_type(annotations, frame_name, width, height)
            if not line_masks:
                return None
            
            # Combine all line masks
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in line_masks.values():
                combined_mask = np.maximum(combined_mask, mask)
            
            return combined_mask
    
    def _get_gt_multiclass_mask(self, annotations: dict, frame_name: str, width: int, height: int) -> np.ndarray:
        """Generate multiclass ground truth mask from annotations."""
        from dataset import create_pitch_mask
        
        # Parse from the annotations directly
        pitch_lines = self._parse_pitch_geometry_from_annotations(annotations, frame_name, width, height)
        
        if pitch_lines is None:
            return None
        
        # Create multiclass mask using dataset function
        mask = create_pitch_mask(pitch_lines, width, height, line_thickness=2, multiclass=True)
        
        return mask
    
    def _parse_pitch_geometry_from_annotations(self, annotations: dict, frame_name: str, width: int, height: int) -> dict:
        """Parse pitch geometry directly from loaded annotations."""
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
        
        # Extract and denormalize line coordinates
        pitch_lines = {}
        
        for ann in pitch_annotations:
            lines = ann.get("lines", {})
            
            for line_name, line_coords in lines.items():
                if not isinstance(line_coords, list) or len(line_coords) == 0:
                    continue
                    
                denormalized_coords = []
                
                for coord in line_coords:
                    try:
                        if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                            x_norm, y_norm = float(coord['x']), float(coord['y'])
                        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            x_norm, y_norm = float(coord[0]), float(coord[1])
                        else:
                            continue
                        
                        x_px = x_norm * width
                        y_px = y_norm * height
                        
                        x_px = max(0, min(x_px, width))
                        y_px = max(0, min(y_px, height))
                        
                        denormalized_coords.append((x_px, y_px))
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if denormalized_coords:
                    pitch_lines[line_name] = denormalized_coords
        
        return pitch_lines if pitch_lines else None
    
    def evaluate_on_sequence_by_line_type(self, sequence_dir: str, threshold: float = 0.5, max_frames: int = None, 
                                         multiscale: bool = False, enhance_goal_posts: bool = True, adaptive_threshold: bool = False):
        """Evaluate model on a single sequence and compute per-line-type metrics."""
        sequence_dir = Path(sequence_dir)
        img_dir = sequence_dir / "img1"
        json_path = sequence_dir / "Labels-GameState.json"
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not json_path.exists():
            raise ValueError(f"Annotation file not found: {json_path}")
        
        # Load annotations
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load annotations: {e}")
        
        # Get frame files
        frame_files = sorted(list(img_dir.glob("*.jpg")))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        logger.info(f"Evaluating on sequence: {sequence_dir.name} ({len(frame_files)} frames)")
        
        # Track metrics per line type
        line_type_metrics = {}
        total_frames = 0
        overall_correct_pixels = 0
        overall_total_pixels = 0
        
        for frame_file in tqdm(frame_files, desc="Evaluating frames"):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            # Get ground truth masks by line type
            gt_masks_by_type = self._get_gt_masks_by_line_type(annotations, frame_file.name, w, h)
            if not gt_masks_by_type:
                continue
            
            # Get prediction mask
            pred_mask = self.predict_mask(frame, threshold, enhance_goal_posts=enhance_goal_posts, 
                                        multiscale=multiscale, adaptive_threshold=adaptive_threshold)
            
            if self.task_type == 'multiclass':
                # For multiclass: evaluate per-class performance
                gt_multiclass_mask = self._get_gt_multiclass_mask(annotations, frame_file.name, w, h)
                if gt_multiclass_mask is None:
                    continue
                
                # Calculate metrics for each class that appears in either GT or prediction
                unique_gt_classes = np.unique(gt_multiclass_mask)
                unique_pred_classes = np.unique(pred_mask)
                all_classes = np.unique(np.concatenate([unique_gt_classes, unique_pred_classes]))
                
                for class_id in all_classes:
                    if class_id == 0:  # Skip background
                        continue
                    
                    class_name = CLASS_TO_LINE_MAPPING.get(class_id, f"Class_{class_id}")
                    
                    if class_name not in line_type_metrics:
                        line_type_metrics[class_name] = {
                            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                            'total_gt_pixels': 0, 'total_pred_pixels': 0,
                            'frames_with_gt': 0
                        }
                    
                    # Create binary masks for this class
                    gt_binary = (gt_multiclass_mask == class_id).astype(np.uint8)
                    pred_binary = (pred_mask == class_id).astype(np.uint8)
                    
                    # Flatten for metrics
                    gt_flat = gt_binary.flatten()
                    pred_flat = pred_binary.flatten()
                    
                    # Update confusion matrix for this class
                    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
                    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
                    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
                    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
                    
                    line_type_metrics[class_name]['tp'] += tp
                    line_type_metrics[class_name]['fp'] += fp
                    line_type_metrics[class_name]['tn'] += tn
                    line_type_metrics[class_name]['fn'] += fn
                    line_type_metrics[class_name]['total_gt_pixels'] += np.sum(gt_flat)
                    line_type_metrics[class_name]['total_pred_pixels'] += np.sum(pred_flat)
                    
                    if np.sum(gt_flat) > 0:  # Only count frames where this class exists in GT
                        line_type_metrics[class_name]['frames_with_gt'] += 1
                
                # Overall pixel accuracy
                overall_correct_pixels += np.sum(gt_multiclass_mask.flatten() == pred_mask.flatten())
                overall_total_pixels += gt_multiclass_mask.size
                
            else:
                # Binary mode (original implementation)
                pred_binary = (pred_mask > 0).astype(np.uint8)
                
                # Calculate metrics for each line type
                for line_type, gt_mask in gt_masks_by_type.items():
                    if line_type not in line_type_metrics:
                        line_type_metrics[line_type] = {
                            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                            'total_gt_pixels': 0, 'total_pred_pixels': 0,
                            'frames_with_gt': 0
                        }
                    
                    gt_binary = (gt_mask > 0).astype(np.uint8)
                    
                    # Flatten for metrics
                    gt_flat = gt_binary.flatten()
                    pred_flat = pred_binary.flatten()
                    
                    # Update confusion matrix for this line type
                    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
                    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
                    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
                    fn = np.sum((gt_flat == 1) & (pred_flat == 0))
                    
                    line_type_metrics[line_type]['tp'] += tp
                    line_type_metrics[line_type]['fp'] += fp
                    line_type_metrics[line_type]['tn'] += tn
                    line_type_metrics[line_type]['fn'] += fn
                    line_type_metrics[line_type]['total_gt_pixels'] += np.sum(gt_flat)
                    line_type_metrics[line_type]['total_pred_pixels'] += np.sum(pred_flat)
                    line_type_metrics[line_type]['frames_with_gt'] += 1
                    
                    # Overall metrics
                    overall_correct_pixels += np.sum(gt_flat == pred_flat)
                    overall_total_pixels += len(gt_flat)
            
            total_frames += 1
        
        if total_frames == 0:
            logger.error("No valid frames found for evaluation")
            return
        
        # Calculate and display results
        print("\n" + "="*80)
        mode_str = "MULTICLASS" if self.task_type == 'multiclass' else "BINARY PER-LINE-TYPE"
        print(f"{mode_str} EVALUATION RESULTS - {sequence_dir.name}")
        print("="*80)
        print(f"Total frames evaluated: {total_frames}")
        print(f"Task type: {self.task_type}")
        if self.task_type == 'binary':
            print(f"Threshold: {threshold}")
        
        # Sort line types for consistent output
        sorted_line_types = sorted(line_type_metrics.keys())
        
        class_or_line = "Class" if self.task_type == 'multiclass' else "Line Type"
        print(f"\n{class_or_line:<25} {'Precision':<10} {'Recall':<10} {'F1':<8} {'IoU':<8} {'GT Pixels':<12} {'Frames':<8}")
        print("-" * 80)
        
        results = {}
        
        for line_type in sorted_line_types:
            metrics = line_type_metrics[line_type]
            
            # Calculate metrics
            tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            
            results[line_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn,
                'gt_pixels': metrics['total_gt_pixels'],
                'frames_with_gt': metrics['frames_with_gt']
            }
            
            print(f"{line_type:<25} {precision:<10.3f} {recall:<10.3f} {f1:<8.3f} {iou:<8.3f} {metrics['total_gt_pixels']:<12,} {metrics['frames_with_gt']:<8}")
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Calculate average metrics (weighted by GT pixels)
        total_gt_pixels = sum(r['gt_pixels'] for r in results.values())
        if total_gt_pixels > 0:
            weighted_precision = sum(r['precision'] * r['gt_pixels'] for r in results.values()) / total_gt_pixels
            weighted_recall = sum(r['recall'] * r['gt_pixels'] for r in results.values()) / total_gt_pixels
            weighted_f1 = sum(r['f1'] * r['gt_pixels'] for r in results.values()) / total_gt_pixels
            weighted_iou = sum(r['iou'] * r['gt_pixels'] for r in results.values()) / total_gt_pixels
            
            print(f"Weighted Average Precision: {weighted_precision:.3f}")
            print(f"Weighted Average Recall: {weighted_recall:.3f}")
            print(f"Weighted Average F1: {weighted_f1:.3f}")
            print(f"Weighted Average IoU: {weighted_iou:.3f}")
        
        # Macro averages
        if results:
            macro_precision = np.mean([r['precision'] for r in results.values()])
            macro_recall = np.mean([r['recall'] for r in results.values()])
            macro_f1 = np.mean([r['f1'] for r in results.values()])
            macro_iou = np.mean([r['iou'] for r in results.values()])
            
            print(f"Macro Average Precision: {macro_precision:.3f}")
            print(f"Macro Average Recall: {macro_recall:.3f}")
            print(f"Macro Average F1: {macro_f1:.3f}")
            print(f"Macro Average IoU: {macro_iou:.3f}")
        
        print("="*80)
        
        return results
    
    def process_sequence_to_video(self, sequence_dir: str, output_path: str, 
                                 fps: int = 10, max_frames: int = None,
                                 threshold: float = 0.5, show_comparison: bool = True, show_legend: bool = False):
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
            overlay_frame = self.overlay_mask_on_frame(frame, pred_mask, color=(0, 255, 0), alpha=0.6, show_legend=show_legend)
            
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
    
    def evaluate_on_dataset(self, dataset_path: str, threshold: float = 0.5):
        """Evaluate model on a dataset and compute per-class metrics."""
        dataset_path = Path(dataset_path)
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        total_frames = 0
        correct_pixels = 0
        total_pixels = 0
        
        # Metrics for binary classification
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        # Find all sequence directories
        sequence_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for seq_dir in tqdm(sequence_dirs, desc="Processing sequences"):
            img_dir = seq_dir / "img1"
            json_path = seq_dir / "Labels-GameState.json"
            
            if not img_dir.exists() or not json_path.exists():
                continue
                
            # Load annotations
            try:
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load annotations for {seq_dir}: {e}")
                continue
            
            # Process each frame
            frame_files = sorted(list(img_dir.glob("*.jpg")))
            
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is None:
                    continue
                
                h, w = frame.shape[:2]
                
                # Get ground truth
                gt_mask = self._get_gt_mask(annotations, frame_file.name, w, h)
                if gt_mask is None:
                    continue
                
                # Get prediction
                pred_mask = self.predict_mask(frame, threshold)
                
                # Convert to binary
                gt_binary = (gt_mask > 0).astype(np.uint8)
                pred_binary = (pred_mask > 0).astype(np.uint8)
                
                # Flatten for metrics
                gt_flat = gt_binary.flatten()
                pred_flat = pred_binary.flatten()
                
                # Collect for overall statistics
                all_ground_truths.extend(gt_flat)
                all_predictions.extend(pred_flat)
                
                # Update pixel-wise accuracy
                correct_pixels += np.sum(gt_flat == pred_flat)
                total_pixels += len(gt_flat)
                
                # Update confusion matrix components
                tp = np.sum((gt_flat == 1) & (pred_flat == 1))
                fp = np.sum((gt_flat == 0) & (pred_flat == 1))
                tn = np.sum((gt_flat == 0) & (pred_flat == 0))
                fn = np.sum((gt_flat == 1) & (pred_flat == 0))
                
                true_positives += tp
                false_positives += fp
                true_negatives += tn
                false_negatives += fn
                
                total_frames += 1
        
        if total_frames == 0:
            logger.error("No valid frames found for evaluation")
            return
        
        # Calculate metrics
        overall_accuracy = correct_pixels / total_pixels
        
        # Per-class metrics
        background_accuracy = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        pitch_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Additional metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU for pitch class
        iou_pitch = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total frames evaluated: {total_frames}")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Threshold: {threshold}")
        print("\nOVERALL METRICS:")
        print(f"  Pixel Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        print("\nPER-CLASS ACCURACY:")
        print(f"  Background (Class 0): {background_accuracy:.4f} ({background_accuracy*100:.2f}%)")
        print(f"  Pitch Lines (Class 1): {pitch_accuracy:.4f} ({pitch_accuracy*100:.2f}%)")
        
        print("\nPITCH LINES METRICS:")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  IoU: {iou_pitch:.4f}")
        
        print("\nCONFUSION MATRIX:")
        print(f"  True Negatives (BG→BG):  {true_negatives:,}")
        print(f"  False Positives (BG→PL): {false_positives:,}")
        print(f"  False Negatives (PL→BG): {false_negatives:,}")
        print(f"  True Positives (PL→PL):  {true_positives:,}")
        
        # Class distribution
        total_bg_pixels = true_negatives + false_positives
        total_pl_pixels = true_positives + false_negatives
        print(f"\nCLASS DISTRIBUTION:")
        print(f"  Background pixels: {total_bg_pixels:,} ({total_bg_pixels/total_pixels*100:.2f}%)")
        print(f"  Pitch line pixels: {total_pl_pixels:,} ({total_pl_pixels/total_pixels*100:.2f}%)")
        print("="*60)
        
        return {
            'overall_accuracy': overall_accuracy,
            'background_accuracy': background_accuracy,
            'pitch_accuracy': pitch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou_pitch': iou_pitch,
            'total_frames': total_frames,
            'confusion_matrix': {
                'tn': true_negatives,
                'fp': false_positives,
                'fn': false_negatives,
                'tp': true_positives
            }
        }
    
    def evaluate_on_sequence(self, sequence_dir: str, threshold: float = 0.5, max_frames: int = None):
        """Evaluate model on a single sequence and compute per-class metrics."""
        sequence_dir = Path(sequence_dir)
        img_dir = sequence_dir / "img1"
        json_path = sequence_dir / "Labels-GameState.json"
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not json_path.exists():
            raise ValueError(f"Annotation file not found: {json_path}")
        
        # Load annotations
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            raise ValueError(f"Could not load annotations: {e}")
        
        # Get frame files
        frame_files = sorted(list(img_dir.glob("*.jpg")))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        logger.info(f"Evaluating on sequence: {sequence_dir.name} ({len(frame_files)} frames)")
        
        # Metrics tracking
        total_frames = 0
        correct_pixels = 0
        total_pixels = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for frame_file in tqdm(frame_files, desc="Evaluating frames"):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            h, w = frame.shape[:2]
            
            # Get ground truth
            gt_mask = self._get_gt_mask(annotations, frame_file.name, w, h)
            if gt_mask is None:
                continue
            
            # Get prediction
            pred_mask = self.predict_mask(frame, threshold)
            
            # Convert to binary
            gt_binary = (gt_mask > 0).astype(np.uint8)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            
            # Flatten for metrics
            gt_flat = gt_binary.flatten()
            pred_flat = pred_binary.flatten()
            
            # Update pixel-wise accuracy
            correct_pixels += np.sum(gt_flat == pred_flat)
            total_pixels += len(gt_flat)
            
            # Update confusion matrix components
            tp = np.sum((gt_flat == 1) & (pred_flat == 1))
            fp = np.sum((gt_flat == 0) & (pred_flat == 1))
            tn = np.sum((gt_flat == 0) & (pred_flat == 0))
            fn = np.sum((gt_flat == 1) & (pred_flat == 0))
            
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
            
            total_frames += 1
        
        if total_frames == 0:
            logger.error("No valid frames found for evaluation")
            return
        
        # Calculate metrics
        overall_accuracy = correct_pixels / total_pixels
        
        # Per-class metrics
        background_accuracy = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        pitch_accuracy = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Additional metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU for pitch class
        iou_pitch = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        
        # Print results
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS - {sequence_dir.name}")
        print("="*60)
        print(f"Total frames evaluated: {total_frames}")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Threshold: {threshold}")
        print("\nOVERALL METRICS:")
        print(f"  Pixel Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        print("\nPER-CLASS ACCURACY:")
        print(f"  Background (Class 0): {background_accuracy:.4f} ({background_accuracy*100:.2f}%)")
        print(f"  Pitch Lines (Class 1): {pitch_accuracy:.4f} ({pitch_accuracy*100:.2f}%)")
        
        print("\nPITCH LINES METRICS:")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  IoU: {iou_pitch:.4f}")
        
        print("\nCONFUSION MATRIX:")
        print(f"  True Negatives (BG→BG):  {true_negatives:,}")
        print(f"  False Positives (BG→PL): {false_positives:,}")
        print(f"  False Negatives (PL→BG): {false_negatives:,}")
        print(f"  True Positives (PL→PL):  {true_positives:,}")
        
        # Class distribution
        total_bg_pixels = true_negatives + false_positives
        total_pl_pixels = true_positives + false_negatives
        print(f"\nCLASS DISTRIBUTION:")
        print(f"  Background pixels: {total_bg_pixels:,} ({total_bg_pixels/total_pixels*100:.2f}%)")
        print(f"  Pitch line pixels: {total_pl_pixels:,} ({total_pl_pixels/total_pixels*100:.2f}%)")
        print("="*60)
        
        return {
            'overall_accuracy': overall_accuracy,
            'background_accuracy': background_accuracy,
            'pitch_accuracy': pitch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou_pitch': iou_pitch,
            'total_frames': total_frames,
            'confusion_matrix': {
                'tn': true_negatives,
                'fp': false_positives,
                'fn': false_negatives,
                'tp': true_positives
            }
        }
    
    def process_single_frame(self, frame_path: str, output_path: str, threshold: float = 0.5):
        """Process a single frame and save the result."""
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load image: {frame_path}")
        
        # Predict mask
        pred_mask = self.predict_mask(frame, threshold)
        
        # Create overlay
        overlay_frame = self.overlay_mask_on_frame(frame, pred_mask, color=(0, 255, 0), alpha=0.6, show_legend=False)
        
        # Save result
        cv2.imwrite(output_path, overlay_frame)
        logger.info(f"Processed frame saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test pitch localization model on video sequences')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--sequence-dir', type=str,
                       help='Path to sequence directory (e.g., SoccerNet/SN-GSR-2025/train/SNGS-062)')
    parser.add_argument('--dataset-path', type=str,
                       help='Path to dataset directory for evaluation (e.g., SoccerNet/SN-GSR-2025/train)')
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
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation mode to compute per-class accuracy metrics')
    parser.add_argument('--evaluate-by-line-type', action='store_true',
                       help='Run evaluation mode to compute per-line-type accuracy metrics')
    parser.add_argument('--multiscale', action='store_true',
                       help='Use multi-scale inference for better thin line detection')
    parser.add_argument('--no-goal-post-enhancement', action='store_true',
                       help='Disable goal post enhancement post-processing')
    parser.add_argument('--adaptive-threshold', action='store_true',
                       help='Use adaptive thresholding for better thin line detection')
    parser.add_argument('--task-type', type=str, default='auto', choices=['auto', 'binary', 'multiclass'],
                       help='Task type: auto-detect, binary, or multiclass')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Number of classes (auto-detect if not specified)')
    parser.add_argument('--show-legend', action='store_true',
                       help='Show class legend in multiclass visualization')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PitchVideoProcessor(args.model_path, args.device, args.task_type, args.num_classes)
    
    if args.evaluate:
        # Run evaluation mode
        if args.dataset_path:
            processor.evaluate_on_dataset(args.dataset_path, args.threshold)
        elif args.sequence_dir:
            processor.evaluate_on_sequence(args.sequence_dir, args.threshold, args.max_frames)
        else:
            parser.error("Either --dataset-path or --sequence-dir is required when using --evaluate")
    elif args.evaluate_by_line_type:
        # Run per-line-type evaluation mode
        if not args.sequence_dir:
            parser.error("--sequence-dir is required when using --evaluate-by-line-type")
        processor.evaluate_on_sequence_by_line_type(
            args.sequence_dir, 
            args.threshold, 
            args.max_frames, 
            multiscale=args.multiscale,
            enhance_goal_posts=not args.no_goal_post_enhancement,
            adaptive_threshold=args.adaptive_threshold
        )
    elif args.frame_path:
        # Process single frame
        processor.process_single_frame(args.frame_path, args.output_path, args.threshold)
    else:
        # Process sequence to video
        if not args.sequence_dir:
            parser.error("--sequence-dir is required when not using --evaluate or --frame-path")
        processor.process_sequence_to_video(
            args.sequence_dir,
            args.output_path,
            fps=args.fps,
            max_frames=args.max_frames,
            threshold=args.threshold,
            show_comparison=not args.no_comparison,
            show_legend=args.show_legend
        )


if __name__ == '__main__':
    main()