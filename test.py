#!/usr/bin/env python3
"""
YOLO Model Testing Script for SoccerNet Game State Reconstruction
Loads the trained model and runs inference on test sequences.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from tqdm import tqdm
import argparse
import time
import torch
from collections import defaultdict

class SoccerNetTester:
    """Test trained YOLO model on SoccerNet GSR sequences."""
    
    def __init__(self, model_path: str, confidence: float = 0.25, iou_threshold: float = 0.45, device: str = 'auto', legacy_mapping: bool = False):
        """
        Initialize the tester.
        
        Args:
            model_path: Path to the trained YOLO model (.pt file)
            confidence: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('auto', 'cpu', 'mps', 'cuda', or specific device)
            legacy_mapping: Use legacy class mapping (ball=0, goalkeeper=1, player=2, referee=3)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Set up device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"🖥️  Using device: {self.device}")
        
        # Class names mapping - support both legacy and new mappings
        if legacy_mapping:
            self.class_names = {
                0: 'ball',
                1: 'goalkeeper',
                2: 'player',
                3: 'referee'
            }
            print("🔄 Using legacy class mapping: ball=0, goalkeeper=1, player=2, referee=3")
        else:
            self.class_names = {
                0: 'player',
                1: 'goalkeeper',
                2: 'referee', 
                3: 'ball'
            }
            print("🔄 Using standard class mapping: player=0, goalkeeper=1, referee=2, ball=3")
        
        # Enhanced colors for visualization (BGR format) with gradients
        self.colors = {
            0: (50, 205, 50),     # player - lime green
            1: (255, 140, 0),     # goalkeeper - dark orange
            2: (220, 20, 60),     # referee - crimson
            3: (0, 215, 255)      # ball - gold
        }
        
        # Secondary colors for gradients and effects
        self.secondary_colors = {
            0: (144, 238, 144),   # light green
            1: (255, 165, 0),     # orange
            2: (255, 69, 0),      # red orange
            3: (255, 255, 0)      # yellow
        }
        
        # Sporty accent colors for enhanced visual effects
        self.accent_colors = {
            0: (0, 255, 127),     # spring green
            1: (255, 69, 0),      # red orange
            2: (255, 20, 147),    # deep pink
            3: (255, 215, 0)      # gold
        }
        
        # Ball trail tracking for motion effects
        self.ball_trail = []  # List of (center_x, center_y, timestamp) tuples
        self.trail_max_length = 15  # Maximum trail points
        self.trail_fade_time = 3.0  # Trail fades over 3 seconds
        
        print(f"🏈 Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Move model to specified device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        print("✅ Model loaded successfully!")
        
        # Accuracy calculation settings
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Category mapping from SoccerNet to model classes
        if legacy_mapping:
            self.category_mapping = {
                1: 2,  # player -> 2 (legacy)
                2: 1,  # goalkeeper -> 1 (legacy)
                3: 3,  # referee -> 3 (legacy)
                4: 0   # ball -> 0 (legacy)
            }
        else:
            self.category_mapping = {
                1: 0,  # player -> 0 (standard)
                2: 1,  # goalkeeper -> 1 (standard)
                3: 2,  # referee -> 2 (standard)
                4: 3   # ball -> 3 (standard)
            }
    
    def predict_single_image(self, image_path: str, save_path: str = None, visualize: bool = True):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotated image (optional)
            visualize: Whether to draw bounding boxes
            
        Returns:
            dict: Detection results
        """
        # Run inference
        results = self.model(image_path, conf=self.confidence, iou=self.iou_threshold, device=self.device, verbose=False)

        # Capture original image shape if available
        orig_shape = None
        try:
            if len(results) > 0 and hasattr(results[0], 'orig_shape'):
                orig_shape = results[0].orig_shape
        except Exception:
            orig_shape = None

        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = boxes.conf[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, str(class_id))
                }
                detections.append(detection)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections),
            'orig_shape': orig_shape
        }
    
    def _draw_detections(self, image_path: str, detections: list, save_path: str):
        """
        Draw sporty enhanced bounding boxes on image and save.
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            save_path: Path to save annotated image
        """
        annotated = self._get_annotated_image(image_path, detections)
        if annotated is None:
            print(f"❌ Could not load image: {image_path}")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, annotated)
    
    def _update_ball_trail(self, detections: list, current_time: float):
        """
        Update ball trail with current ball detections.
        
        Args:
            detections: List of detection dictionaries
            current_time: Current timestamp
        """
        # Always prune old trail points
        self.ball_trail = [
            (x, y, t) for x, y, t in self.ball_trail
            if current_time - t <= self.trail_fade_time
        ]

        # Add current ball position if detected
        for det in detections:
            if det['class_id'] == 3:  # ball
                x1, y1, x2, y2 = det['bbox']
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.ball_trail.append((center_x, center_y, current_time))
                break

        # Limit trail length
        if len(self.ball_trail) > self.trail_max_length:
            self.ball_trail = self.ball_trail[-self.trail_max_length:]
    
    def _draw_ball_trail(self, image, current_time: float):
        """
        Draw fading ball trail effect.
        
        Args:
            image: OpenCV image to draw on
            current_time: Current timestamp
        """
        if len(self.ball_trail) < 2:
            return
        
        # Draw trail as connected circles with fading effect
        for i in range(len(self.ball_trail) - 1):
            x1, y1, t1 = self.ball_trail[i]
            x2, y2, t2 = self.ball_trail[i + 1]
            
            # Calculate fade factor based on age
            age = current_time - t1
            fade_factor = max(0, 1 - (age / self.trail_fade_time))
            
            # Trail color with fade effect
            trail_color = (int(255 * fade_factor), int(215 * fade_factor), 0)  # Fading gold
            
            # Draw trail segment
            radius = max(2, int(8 * fade_factor))
            cv2.circle(image, (x1, y1), radius, trail_color, -1)
            
            # Connect with line
            if fade_factor > 0.1:
                cv2.line(image, (x1, y1), (x2, y2), trail_color, max(1, int(3 * fade_factor)))
    
    def _draw_sporty_ball(self, image, x1: int, y1: int, x2: int, y2: int, 
                         primary_color, secondary_color, accent_color, confidence: float, class_name: str):
        """
        Draw sporty circular ball detection.
        
        Args:
            image: OpenCV image to draw on
            x1, y1, x2, y2: Bounding box coordinates
            primary_color, secondary_color, accent_color: Colors for styling
            confidence: Detection confidence
            class_name: Class name
        """
        # Calculate circle parameters
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        radius = int(max((x2 - x1), (y2 - y1)) / 2)
        
        # Draw shadow circle
        shadow_offset = 4
        cv2.circle(image, (center_x + shadow_offset, center_y + shadow_offset), 
                  radius + 2, (0, 0, 0), 3)
        
        # Draw outer glow effect
        for i in range(3, 0, -1):
            glow_alpha = 0.3 * i
            glow_color = tuple(int(c * glow_alpha) for c in accent_color)
            cv2.circle(image, (center_x, center_y), radius + i * 3, glow_color, 2)
        
        # Draw main circle with gradient effect
        cv2.circle(image, (center_x, center_y), radius + 2, secondary_color, 4)
        cv2.circle(image, (center_x, center_y), radius, primary_color, 3)
        
        # Add soccer ball pattern
        pattern_radius = radius // 3
        cv2.circle(image, (center_x, center_y), pattern_radius, (255, 255, 255), 2)
        
        # Add motion lines for dynamic effect
        for angle in [0, 60, 120, 180, 240, 300]:
            angle_rad = np.radians(angle)
            start_x = center_x + int((radius - 5) * np.cos(angle_rad))
            start_y = center_y + int((radius - 5) * np.sin(angle_rad))
            end_x = center_x + int((radius + 8) * np.cos(angle_rad))
            end_y = center_y + int((radius + 8) * np.sin(angle_rad))
            cv2.line(image, (start_x, start_y), (end_x, end_y), accent_color, 2)
        
        # Label removed - showing only bounding boxes
    
    def _draw_sporty_bbox(self, image, x1: int, y1: int, x2: int, y2: int,
                         primary_color, secondary_color, accent_color, confidence: float, 
                         class_name: str, class_id: int):
        """
        Draw sporty rectangular bounding box for players, goalkeepers, and referees.
        
        Args:
            image: OpenCV image to draw on
            x1, y1, x2, y2: Bounding box coordinates
            primary_color, secondary_color, accent_color: Colors for styling
            confidence: Detection confidence
            class_name: Class name
            class_id: Class ID for specific styling
        """
        # Draw shadow effect
        shadow_offset = 4
        cv2.rectangle(image, (x1 + shadow_offset, y1 + shadow_offset), 
                     (x2 + shadow_offset, y2 + shadow_offset), (0, 0, 0), 3)
        
        # Draw outer glow for dynamic effect
        for i in range(2, 0, -1):
            glow_color = tuple(int(c * 0.4) for c in accent_color)
            cv2.rectangle(image, (x1 - i, y1 - i), (x2 + i, y2 + i), glow_color, 2)
        
        # Main bounding box with double border
        cv2.rectangle(image, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), secondary_color, 4)
        cv2.rectangle(image, (x1, y1), (x2, y2), primary_color, 3)
        
        # Add sporty corner brackets
        bracket_length = min(25, (x2 - x1) // 3, (y2 - y1) // 3)
        bracket_thickness = 5
        
        # Top-left bracket
        cv2.line(image, (x1 - 5, y1), (x1 + bracket_length, y1), accent_color, bracket_thickness)
        cv2.line(image, (x1, y1 - 5), (x1, y1 + bracket_length), accent_color, bracket_thickness)
        
        # Top-right bracket
        cv2.line(image, (x2 + 5, y1), (x2 - bracket_length, y1), accent_color, bracket_thickness)
        cv2.line(image, (x2, y1 - 5), (x2, y1 + bracket_length), accent_color, bracket_thickness)
        
        # Bottom-left bracket
        cv2.line(image, (x1 - 5, y2), (x1 + bracket_length, y2), accent_color, bracket_thickness)
        cv2.line(image, (x1, y2 + 5), (x1, y2 - bracket_length), accent_color, bracket_thickness)
        
        # Bottom-right bracket
        cv2.line(image, (x2 + 5, y2), (x2 - bracket_length, y2), accent_color, bracket_thickness)
        cv2.line(image, (x2, y2 + 5), (x2, y2 - bracket_length), accent_color, bracket_thickness)
        
        # Add class-specific decorations
        if class_id == 1:  # goalkeeper - add glove icons
            glove_size = 8
            cv2.circle(image, (x1 + 10, y1 + 10), glove_size, accent_color, -1)
            cv2.circle(image, (x2 - 10, y1 + 10), glove_size, accent_color, -1)
        elif class_id == 2:  # referee - add whistle icon
            whistle_center = (x1 + 15, y1 + 15)
            cv2.circle(image, whistle_center, 6, accent_color, -1)
            cv2.circle(image, whistle_center, 6, (255, 255, 255), 2)
        
        # Label removed - showing only bounding boxes
    
    def _draw_sporty_label(self, image, x: int, y: int, class_name: str, confidence: float,
                          primary_color, secondary_color):
        """
        Draw sporty styled label with modern design.
        
        Args:
            image: OpenCV image to draw on
            x, y: Label position
            class_name: Class name to display
            confidence: Detection confidence
            primary_color, secondary_color: Colors for styling
        """
        label = f"{class_name.upper()}"
        confidence_label = f"{confidence:.1%}"
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text dimensions
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        (conf_w, conf_h), _ = cv2.getTextSize(confidence_label, font, font_scale - 0.2, font_thickness - 1)
        
        # Calculate background dimensions
        padding = 10
        bg_w = max(label_w, conf_w) + 2 * padding
        bg_h = label_h + conf_h + 3 * padding
        
        # Ensure label stays within image bounds
        if y < 0:
            y = 10
        if x + bg_w > image.shape[1]:
            x = image.shape[1] - bg_w - 10
        
        # Draw label background with sporty styling
        # Shadow
        cv2.rectangle(image, (x + 3, y + 3), (x + bg_w + 3, y + bg_h + 3), (0, 0, 0), -1)
        
        # Main background with gradient effect
        cv2.rectangle(image, (x, y), (x + bg_w, y + bg_h), primary_color, -1)
        cv2.rectangle(image, (x, y), (x + bg_w, y + bg_h), secondary_color, 3)
        
        # Add corner accents
        accent_size = 8
        cv2.rectangle(image, (x, y), (x + accent_size, y + accent_size), (255, 255, 255), -1)
        cv2.rectangle(image, (x + bg_w - accent_size, y), (x + bg_w, y + accent_size), (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(image, label, (x + padding, y + label_h + padding),
                   font, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(image, confidence_label, (x + padding, y + label_h + conf_h + 2 * padding),
                   font, font_scale - 0.2, (220, 220, 220), font_thickness - 1)
    
    def _get_annotated_image(self, image_path: str, detections: list, enable_ball_trail: bool = True):
        """
        Get sporty annotated image for video creation.
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            enable_ball_trail: Whether to enable ball trail effects (for videos)
            
        Returns:
            numpy.ndarray: Sporty annotated image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Update ball trail with current detections (only for videos)
        if enable_ball_trail:
            current_time = time.time()
            self._update_ball_trail(detections, current_time)
            
            # Draw ball trail first (so it appears behind the ball)
            self._draw_ball_trail(image, current_time)
        
        # Draw detections with sporty styling
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Get colors for this class
            primary_color = self.colors.get(class_id, (128, 128, 128))
            secondary_color = self.secondary_colors.get(class_id, (200, 200, 200))
            accent_color = self.accent_colors.get(class_id, (255, 255, 255))
            
            # Convert to int coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Special handling for ball - draw as circle
            if class_id == 3:  # ball
                self._draw_sporty_ball(image, x1, y1, x2, y2, primary_color, secondary_color, accent_color, confidence, class_name)
            else:
                self._draw_sporty_bbox(image, x1, y1, x2, y2, primary_color, secondary_color, accent_color, confidence, class_name, class_id)
            
        
        return image
    
    def _load_ground_truth(self, sequence_path: str):
        """
        Load ground truth annotations from Labels-GameState.json.
        
        Args:
            sequence_path: Path to sequence directory
            
        Returns:
            dict: Ground truth annotations indexed by frame number
        """
        labels_file = os.path.join(sequence_path, "Labels-GameState.json")
        
        if not os.path.exists(labels_file):
            print(f"⚠️  No ground truth found: {labels_file}")
            return {}
        
        with open(labels_file, 'r') as f:
            data = json.load(f)
        
        # Parse annotations by frame
        gt_by_frame = defaultdict(list)
        
        for annotation in data.get('annotations', []):
            # Extract frame number from image_id or file_name
            image_id = annotation.get('image_id', '')
            # Assuming image_id format like "3117000001" where last 6 digits are frame number
            frame_num = int(image_id[-6:]) if len(image_id) >= 6 else 0
            
            # Convert bbox format from x,y,w,h to x1,y1,x2,y2
            bbox_img = annotation.get('bbox_image', {})
            if bbox_img:
                x, y, w, h = bbox_img['x'], bbox_img['y'], bbox_img['w'], bbox_img['h']
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Map category_id to our class system
                category_id = annotation.get('category_id', 0)
                class_id = self.category_mapping.get(category_id, 0)
                
                gt_annotation = {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'category_id': category_id,
                    'track_id': annotation.get('track_id', -1)
                }
                
                gt_by_frame[frame_num].append(gt_annotation)
        
        return dict(gt_by_frame)
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2] format
            
        Returns:
            float: IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_ap(self, predictions, ground_truths, iou_threshold=0.5, class_id=None):
        """
        Calculate Average Precision (AP) for a specific class and IoU threshold.
        
        Args:
            predictions: List of prediction dicts with 'bbox', 'confidence', 'class_id'
            ground_truths: List of ground truth dicts with 'bbox', 'class_id'
            iou_threshold: IoU threshold for positive detection
            class_id: Specific class to calculate AP for (None for all)
            
        Returns:
            float: Average Precision value
        """
        if class_id is not None:
            predictions = [p for p in predictions if p['class_id'] == class_id]
            ground_truths = [gt for gt in ground_truths if gt['class_id'] == class_id]
        
        if len(ground_truths) == 0:
            # Return 0.0 to avoid inflating mAP for classes with no GT
            return 0.0
        
        if len(predictions) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(ground_truths)
        
        true_positives = []
        false_positives = []
        
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue

                # Restrict matching to the same frame if frame numbers are available
                if ('frame_number' in pred and 'frame_number' in gt
                        and pred['frame_number'] != gt['frame_number']):
                    continue

                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives.append(1)
                false_positives.append(0)
                gt_matched[best_gt_idx] = True
            else:
                true_positives.append(0)
                false_positives.append(1)
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def _calculate_accuracy_metrics(self, all_predictions, all_ground_truths):
        """
        Calculate comprehensive accuracy metrics.
        
        Args:
            all_predictions: List of all predictions across frames
            all_ground_truths: List of all ground truths across frames
            
        Returns:
            dict: Accuracy metrics including mAP, per-class AP, etc.
        """
        metrics = {
            'mAP': {},  # mAP at different IoU thresholds
            'mAP_50': 0.0,  # mAP at IoU 0.5
            'mAP_75': 0.0,  # mAP at IoU 0.75
            'mAP_50_95': 0.0,  # mAP averaged over IoU 0.5:0.95
            'class_AP': {},  # AP per class
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
        
        if not all_ground_truths:
            print("⚠️  No ground truth annotations found for accuracy calculation")
            return metrics
        
        # Calculate mAP for different IoU thresholds
        for iou_thresh in self.iou_thresholds:
            class_aps = []
            
            for class_id in range(len(self.class_names)):
                ap = self._calculate_ap(all_predictions, all_ground_truths, iou_thresh, class_id)
                class_aps.append(ap)
                
                if iou_thresh == 0.5:
                    metrics['class_AP'][self.class_names[class_id]] = ap
            
            mean_ap = np.mean(class_aps)
            metrics['mAP'][f'IoU_{iou_thresh:.2f}'] = mean_ap
            
            if iou_thresh == 0.5:
                metrics['mAP_50'] = mean_ap
            elif iou_thresh == 0.75:
                metrics['mAP_75'] = mean_ap
        
        # Calculate mAP_50_95 (average over IoU 0.5:0.95)
        metrics['mAP_50_95'] = np.mean([metrics['mAP'][f'IoU_{t:.2f}'] for t in self.iou_thresholds])
        
        # Calculate precision, recall, F1 at IoU 0.5
        iou_thresh = 0.5
        total_tp = 0
        total_fp = 0
        total_gt = len(all_ground_truths)
        
        if all_predictions:
            predictions_sorted = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
            gt_matched = [False] * len(all_ground_truths)
            
            for pred in predictions_sorted:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(all_ground_truths):
                    if gt_matched[gt_idx] or pred['class_id'] != gt['class_id']:
                        continue

                    # Restrict matching to the same frame if frame numbers are available
                    if ('frame_number' in pred and 'frame_number' in gt
                            and pred['frame_number'] != gt['frame_number']):
                        continue

                    iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1
        
        # Calculate final metrics
        if total_tp + total_fp > 0:
            metrics['precision'] = total_tp / (total_tp + total_fp)
        
        if total_gt > 0:
            metrics['recall'] = total_tp / total_gt
        
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        return metrics
    
    def _resolve_yolo_dirs(self, base_path: str):
        """
        Resolve YOLO split directories (images/ and labels/) from a base path.
        Supports paths pointing to the split dir itself, to its images dir,
        and 'valid' vs 'val' naming.
        """
        base_path = os.path.abspath(base_path)
        # Case 1: base has images/labels
        images_dir = os.path.join(base_path, "images")
        labels_dir = os.path.join(base_path, "labels")
        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            return images_dir, labels_dir, os.path.basename(base_path)
        # Case 2: base is images folder
        if os.path.basename(base_path) == "images":
            sibling_labels = os.path.join(os.path.dirname(base_path), "labels")
            if os.path.isdir(sibling_labels):
                return base_path, sibling_labels, os.path.basename(os.path.dirname(base_path))
        # Case 3: handle 'valid' -> 'val'
        if os.path.basename(base_path) == "valid":
            alt = os.path.join(os.path.dirname(base_path), "val")
            images_dir = os.path.join(alt, "images")
            labels_dir = os.path.join(alt, "labels")
            if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
                return images_dir, labels_dir, "val"
        return None, None, None

    def _yolo_to_xyxy(self, x: float, y: float, w: float, h: float, img_w: int, img_h: int):
        """Convert normalized YOLO bbox (xc,yc,w,h) to pixel xyxy."""
        x1 = (x - w / 2.0) * img_w
        y1 = (y - h / 2.0) * img_h
        x2 = (x + w / 2.0) * img_w
        y2 = (y + h / 2.0) * img_h
        return [float(x1), float(y1), float(x2), float(y2)]

    def _parse_yolo_filename(self, filename: str):
        """Parse '<sequence>_<frame>.jpg' into (sequence, frame_number)."""
        stem = os.path.splitext(filename)[0]
        if "_" in stem:
            seq, frame_str = stem.rsplit("_", 1)
            try:
                frame_num = int(frame_str)
            except Exception:
                frame_num = -1
            return seq, frame_num
        return "", -1

    def test_yolo_split(self, split_path: str, output_dir: str = "test_results",
                        save_images: bool = True, save_json: bool = True,
                        save_video: bool = True, max_frames: int = None,
                        frame_step: int = 1, calculate_accuracy: bool = True,
                        batch_size: int = 16):
        """
        Test model on a YOLO-format dataset split directory containing
        'images/' and 'labels/' subdirectories.

        Args:
            split_path: Path to split directory (e.g., yolo_dataset_proper/valid or val)
            batch_size: Batch size for inference (default: 8)
        """
        images_dir, labels_dir, split_name = self._resolve_yolo_dirs(split_path)
        if images_dir is None:
            print(f"❌ Not a YOLO split directory: {split_path}")
            return None

        # Collect image files
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Apply frame step and max_frames
        image_files = image_files[::frame_step]
        if max_frames:
            image_files = image_files[:max_frames]

        print(f"\n🎯 Testing YOLO split: {split_name}")
        print(f"📁 Processing {len(image_files)} images...")
        print(f"⚡ Using batch size: {batch_size}")
        
        # Optimization: Skip visualization entirely if not needed
        need_visualization = save_images or save_video
        if not need_visualization:
            print("🚀 Visualization disabled - running in fast inference mode")
        
        # Pre-load all ground truth data for batch processing
        ground_truth_cache = {}
        image_dimensions_cache = {}
        if calculate_accuracy:
            print("📊 Pre-loading ground truth annotations...")
            for image_file in tqdm(image_files, desc="Loading GT", leave=False):
                label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
                if os.path.exists(label_file):
                    with open(label_file, 'r') as lf:
                        ground_truth_cache[image_file] = [line.strip().split() for line in lf if line.strip()]

        # Output directories
        sequence_output_dir = os.path.join(output_dir, f"yolo_{split_name}")
        if save_images:
            images_output_dir = os.path.join(sequence_output_dir, "annotated_images")
            os.makedirs(images_output_dir, exist_ok=True)

        video_writer = None
        video_path = None
        if save_video:
            video_path = os.path.join(sequence_output_dir, f"{split_name}_detections.mp4")
            os.makedirs(sequence_output_dir, exist_ok=True)

        # Aggregations
        all_results = []
        all_predictions = []
        all_ground_truths = []

        stats = {
            'total_frames': len(image_files),
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'avg_detections_per_frame': 0,
            'processing_time': 0,
            'accuracy_metrics': None
        }

        start_time = time.time()
        
        # Process images in batches for faster inference
        for i in tqdm(range(0, len(image_files), batch_size), 
                      desc=f"Processing {split_name} (batch)", 
                      unit="batch"):
            batch_files = image_files[i:i+batch_size]
            batch_paths = [os.path.join(images_dir, f) for f in batch_files]
            
            # Run batch inference
            try:
                batch_results = self.model(
                    batch_paths, 
                    conf=self.confidence, 
                    iou=self.iou_threshold, 
                    device=self.device, 
                    verbose=False,
                    batch=batch_size  # Explicitly set batch size
                )
            except Exception as e:
                print(f"⚠️  Batch inference failed, falling back to individual inference: {e}")
                # Fallback to individual inference for this batch
                batch_results = []
                for path in batch_paths:
                    try:
                        single_result = self.model(
                            path, 
                            conf=self.confidence, 
                            iou=self.iou_threshold, 
                            device=self.device, 
                            verbose=False
                        )
                        batch_results.append(single_result[0] if isinstance(single_result, list) else single_result)
                    except Exception as ie:
                        print(f"❌ Individual inference also failed for {path}: {ie}")
                        # Create a dummy result for failed inference
                        dummy_result = type('DummyResult', (), {
                            'boxes': None,
                            'orig_shape': None
                        })()
                        batch_results.append(dummy_result)
            
            # Optimize: Batch process all tensor operations
            batch_detections = []
            for j, (image_file, result) in enumerate(zip(batch_files, batch_results)):
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    # Keep tensors on GPU and batch convert to CPU
                    xyxy_batch = boxes.xyxy.cpu().numpy()
                    conf_batch = boxes.conf.cpu().numpy()
                    cls_batch = boxes.cls.cpu().numpy().astype(int)
                    
                    for k in range(len(boxes)):
                        detection = {
                            'bbox': [float(xyxy_batch[k][0]), float(xyxy_batch[k][1]), 
                                   float(xyxy_batch[k][2]), float(xyxy_batch[k][3])],
                            'confidence': float(conf_batch[k]),
                            'class_id': int(cls_batch[k]),
                            'class_name': self.class_names.get(int(cls_batch[k]), str(int(cls_batch[k])))
                        }
                        detections.append(detection)
                batch_detections.append((image_file, result, detections))
            
            # Process each result in the batch
            for j, (image_file, result, detections) in enumerate(batch_detections):
                image_path = os.path.join(images_dir, image_file)
                
                # Create result dict
                result_dict = {
                    'image_path': image_path,
                    'detections': detections,
                    'num_detections': len(detections),
                    'orig_shape': result.orig_shape if hasattr(result, 'orig_shape') else None
                }
                
                # For metrics bookkeeping
                sequence_name, frame_number = self._parse_yolo_filename(image_file)
                result_dict['frame_number'] = frame_number
                result_dict['sequence'] = sequence_name
                result_dict['file_name'] = image_file
                all_results.append(result_dict)

                # Accuracy: add predictions
                if calculate_accuracy:
                    for det in result_dict['detections']:
                        all_predictions.append({
                            'bbox': det['bbox'],
                            'confidence': det['confidence'],
                            'class_id': det['class_id'],
                            'frame_number': frame_number
                        })

                    # Use pre-loaded ground truth data
                    if image_file in ground_truth_cache:
                        # Get image dimensions from cache or inference result
                        img_h, img_w = None, None
                        if image_file in image_dimensions_cache:
                            img_h, img_w = image_dimensions_cache[image_file]
                        elif isinstance(result_dict.get('orig_shape'), (tuple, list)) and len(result_dict['orig_shape']) >= 2:
                            img_h, img_w = int(result_dict['orig_shape'][0]), int(result_dict['orig_shape'][1])
                            image_dimensions_cache[image_file] = (img_h, img_w)
                        else:
                            # Only load image if absolutely necessary
                            img_for_shape = cv2.imread(image_path)
                            if img_for_shape is not None:
                                img_h, img_w = img_for_shape.shape[:2]
                                image_dimensions_cache[image_file] = (img_h, img_w)
                        
                        if img_h is not None and img_w is not None:
                            for parts in ground_truth_cache[image_file]:
                                if len(parts) >= 5:
                                    try:
                                        cls_id = int(float(parts[0]))
                                        xc, yc, bw, bh = map(float, parts[1:5])
                                        bbox_xyxy = self._yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h)
                                        all_ground_truths.append({
                                            'bbox': bbox_xyxy,
                                            'class_id': cls_id,
                                            'frame_number': frame_number
                                        })
                                    except Exception:
                                        continue

                # Create annotated image only if visualization is needed
                if need_visualization:
                    # Enable ball trail only for videos, not for static images
                    enable_ball_trail = save_video
                    annotated_image = self._get_annotated_image(image_path, result_dict['detections'], enable_ball_trail)

                    # Save annotated image
                    if save_images and annotated_image is not None:
                        save_path = os.path.join(images_output_dir, f"annotated_{image_file}")
                        cv2.imwrite(save_path, annotated_image)

                    # Video frame
                    if save_video and annotated_image is not None:
                        if video_writer is None:
                            height, width = annotated_image.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))
                            print(f"\n🎬 Creating video: {video_path} ({width}x{height})")
                        video_writer.write(annotated_image)

                # Update stats
                stats['total_detections'] += result_dict['num_detections']
                for det in result_dict['detections']:
                    stats['class_counts'][det['class_name']] += 1

        if video_writer is not None:
            video_writer.release()
            print(f"\n🎥 Video saved: {video_path}")

        processing_time = time.time() - start_time
        stats['processing_time'] = processing_time
        stats['avg_detections_per_frame'] = stats['total_detections'] / max(1, len(image_files))
        stats['fps'] = len(image_files) / processing_time if processing_time > 0 else 0.0

        if calculate_accuracy and all_ground_truths:
            print("\n📊 Calculating accuracy metrics...")
            accuracy_metrics = self._calculate_accuracy_metrics(all_predictions, all_ground_truths)
            stats['accuracy_metrics'] = accuracy_metrics
            print(f"✅ Accuracy calculation complete:")
            print(f"   📈 mAP@0.5: {accuracy_metrics['mAP_50']:.3f}")
            print(f"   📈 mAP@0.75: {accuracy_metrics['mAP_75']:.3f}")
            print(f"   📈 mAP@0.5:0.95: {accuracy_metrics['mAP_50_95']:.3f}")
            print(f"   🎯 Precision: {accuracy_metrics['precision']:.3f}")
            print(f"   🎯 Recall: {accuracy_metrics['recall']:.3f}")
            print(f"   🎯 F1-Score: {accuracy_metrics['f1_score']:.3f}")
        elif calculate_accuracy:
            print("⚠️  No ground truth annotations found for accuracy calculation")

        # Save results as JSON
        if save_json:
            results_file = os.path.join(sequence_output_dir, "detection_results.json")
            os.makedirs(sequence_output_dir, exist_ok=True)
            output_data = {
                'split_info': {
                    'name': split_name,
                    'total_images': len(image_files),
                    'frame_step': frame_step,
                    'dataset_format': 'YOLO'
                },
                'model_info': {
                    'model_path': self.model_path,
                    'confidence_threshold': self.confidence,
                    'iou_threshold': self.iou_threshold,
                    'batch_size': batch_size
                },
                'statistics': stats,
                'detections': all_results
            }
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")

        # Print summary
        print(f"\n📊 SUMMARY for YOLO split {split_name}:")
        print(f"  🖼️  Images processed: {stats['total_frames']}")
        print(f"  🎯 Total detections: {stats['total_detections']}")
        print(f"  📈 Avg detections/image: {stats['avg_detections_per_frame']:.2f}")
        print(f"  ⚡ Processing speed: {stats['fps']:.2f} FPS")
        print(f"  ⏱️  Total time: {stats['processing_time']:.2f}s")
        if stats['accuracy_metrics']:
            acc = stats['accuracy_metrics']
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"  📊 mAP@0.5: {acc['mAP_50']:.3f}")
            print(f"  📊 mAP@0.75: {acc['mAP_75']:.3f}")
            print(f"  📊 mAP@0.5:0.95: {acc['mAP_50_95']:.3f}")
            print(f"  🎯 Precision: {acc['precision']:.3f}")
            print(f"  🎯 Recall: {acc['recall']:.3f}")
            print(f"  🎯 F1-Score: {acc['f1_score']:.3f}")
            print(f"\n📋 Per-class AP@0.5:")
            for class_name, ap in acc['class_AP'].items():
                print(f"    {class_name}: {ap:.3f}")
        print(f"\n🏷️  Class breakdown:")
        for class_name, count in stats['class_counts'].items():
            print(f"    {class_name}: {count}")

        return stats

    def test_sequence(self, sequence_path: str, output_dir: str = "test_results", 
                     save_images: bool = True, save_json: bool = True, 
                     save_video: bool = True, max_frames: int = None, frame_step: int = 1,
                     calculate_accuracy: bool = True):
        """
        Test model on a complete sequence.
        
        Args:
            sequence_path: Path to sequence directory (e.g., SNGS-116)
            output_dir: Directory to save results
            save_images: Whether to save annotated images
            save_json: Whether to save detection results as JSON
            save_video: Whether to create output video
            max_frames: Maximum number of frames to process (None for all)
            frame_step: Process every Nth frame (1 for all frames)
            calculate_accuracy: Whether to calculate accuracy metrics using ground truth
            
        Returns:
            dict: Summary statistics including accuracy metrics if available
        """
        sequence_name = os.path.basename(sequence_path)
        images_dir = os.path.join(sequence_path, "img1")
        
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return None
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        # Apply frame step and max_frames
        image_files = image_files[::frame_step]
        if max_frames:
            image_files = image_files[:max_frames]
        
        print(f"\n🎯 Testing sequence: {sequence_name}")
        print(f"📁 Processing {len(image_files)} frames...")
        
        # Optimization: Skip visualization entirely if not needed
        need_visualization = save_images or save_video
        if not need_visualization:
            print("🚀 Visualization disabled - running in fast inference mode")
        
        # Load ground truth annotations for accuracy calculation
        ground_truth = {}
        if calculate_accuracy:
            print("📋 Loading ground truth annotations...")
            ground_truth = self._load_ground_truth(sequence_path)
            if ground_truth:
                print(f"✅ Loaded ground truth for {len(ground_truth)} frames")
            else:
                print("⚠️  No ground truth found - accuracy calculation disabled")
                calculate_accuracy = False
        
        # Create output directories
        sequence_output_dir = os.path.join(output_dir, sequence_name)
        if save_images:
            images_output_dir = os.path.join(sequence_output_dir, "annotated_images")
            os.makedirs(images_output_dir, exist_ok=True)
        
        # Initialize video writer if needed
        video_writer = None
        video_path = None
        if save_video:
            video_path = os.path.join(sequence_output_dir, f"{sequence_name}_detections.mp4")
            os.makedirs(sequence_output_dir, exist_ok=True)
        
        # Process frames
        all_results = []
        all_predictions = []  # For accuracy calculation
        all_ground_truths = []  # For accuracy calculation
        
        stats = {
            'total_frames': len(image_files),
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names.values()},
            'avg_detections_per_frame': 0,
            'processing_time': 0,
            'accuracy_metrics': None  # Will be populated if calculate_accuracy is True
        }
        
        start_time = time.time()
        
        with tqdm(total=len(image_files), desc=f"Processing {sequence_name}", unit="frame") as pbar:
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(images_dir, image_file)
                
                # Prepare save path for annotated image
                save_path = None
                if save_images:
                    save_path = os.path.join(images_output_dir, f"annotated_{image_file}")
                
                # Run inference (do not draw here to avoid double trail updates)
                result = self.predict_single_image(image_path, save_path=None, visualize=False)
                
                # Load and annotate once if needed for image saving and/or video
                annotated_image = None
                if need_visualization:
                    # Enable ball trail only for videos, not for static images
                    enable_ball_trail = save_video
                    annotated_image = self._get_annotated_image(image_path, result['detections'], enable_ball_trail)

                    # Save annotated image
                    if save_images and annotated_image is not None:
                        cv2.imwrite(save_path, annotated_image)

                # Add frame to video if needed
                if save_video and annotated_image is not None:
                    # Initialize video writer on first frame
                    if video_writer is None:
                        height, width = annotated_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (width, height))
                        print(f"\n🎬 Creating video: {video_path} ({width}x{height})")
                    
                    # Write frame to video
                    video_writer.write(annotated_image)
                
                # Add frame info
                frame_number = int(os.path.splitext(image_file)[0])
                result['frame_number'] = frame_number
                result['sequence'] = sequence_name
                
                all_results.append(result)
                
                # Collect predictions and ground truths for accuracy calculation
                if calculate_accuracy:
                    # Add predictions
                    for det in result['detections']:
                        all_predictions.append({
                            'bbox': det['bbox'],
                            'confidence': det['confidence'],
                            'class_id': det['class_id'],
                            'frame_number': frame_number
                        })
                    
                    # Add ground truths for this frame
                    if frame_number in ground_truth:
                        for gt in ground_truth[frame_number]:
                            gt_copy = gt.copy()
                            gt_copy['frame_number'] = frame_number
                            all_ground_truths.append(gt_copy)
                
                # Update statistics
                stats['total_detections'] += result['num_detections']
                for det in result['detections']:
                    stats['class_counts'][det['class_name']] += 1
                
                # Update progress
                pbar.set_postfix({
                    'Detections': result['num_detections'],
                    'Total': stats['total_detections']
                })
                pbar.update(1)
        
        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print(f"\n🎥 Video saved: {video_path}")
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        stats['processing_time'] = processing_time
        stats['avg_detections_per_frame'] = stats['total_detections'] / max(1, len(image_files))
        stats['fps'] = len(image_files) / processing_time if processing_time > 0 else 0.0
        
        # Calculate accuracy metrics if ground truth is available
        if calculate_accuracy and all_ground_truths:
            print("\n📊 Calculating accuracy metrics...")
            accuracy_metrics = self._calculate_accuracy_metrics(all_predictions, all_ground_truths)
            stats['accuracy_metrics'] = accuracy_metrics
            
            print(f"✅ Accuracy calculation complete:")
            print(f"   📈 mAP@0.5: {accuracy_metrics['mAP_50']:.3f}")
            print(f"   📈 mAP@0.75: {accuracy_metrics['mAP_75']:.3f}")
            print(f"   📈 mAP@0.5:0.95: {accuracy_metrics['mAP_50_95']:.3f}")
            print(f"   🎯 Precision: {accuracy_metrics['precision']:.3f}")
            print(f"   🎯 Recall: {accuracy_metrics['recall']:.3f}")
            print(f"   🎯 F1-Score: {accuracy_metrics['f1_score']:.3f}")
        elif calculate_accuracy:
            print("⚠️  No ground truth annotations found for accuracy calculation")
        
        # Save results as JSON
        if save_json:
            results_file = os.path.join(sequence_output_dir, "detection_results.json")
            os.makedirs(sequence_output_dir, exist_ok=True)
            
            output_data = {
                'sequence_info': {
                    'name': sequence_name,
                    'total_frames': len(image_files),
                    'frame_step': frame_step
                },
                'model_info': {
                    'model_path': self.model_path,
                    'confidence_threshold': self.confidence,
                    'iou_threshold': self.iou_threshold
                },
                'statistics': stats,
                'detections': all_results
            }
            
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
        
        # Print summary
        print(f"\n📊 SUMMARY for {sequence_name}:")
        print(f"  🎬 Frames processed: {stats['total_frames']}")
        print(f"  🎯 Total detections: {stats['total_detections']}")
        print(f"  📈 Avg detections/frame: {stats['avg_detections_per_frame']:.2f}")
        print(f"  ⚡ Processing speed: {stats['fps']:.2f} FPS")
        print(f"  ⏱️  Total time: {stats['processing_time']:.2f}s")
        
        # Print accuracy metrics if available
        if stats['accuracy_metrics']:
            acc = stats['accuracy_metrics']
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"  📊 mAP@0.5: {acc['mAP_50']:.3f}")
            print(f"  📊 mAP@0.75: {acc['mAP_75']:.3f}")
            print(f"  📊 mAP@0.5:0.95: {acc['mAP_50_95']:.3f}")
            print(f"  🎯 Precision: {acc['precision']:.3f}")
            print(f"  🎯 Recall: {acc['recall']:.3f}")
            print(f"  🎯 F1-Score: {acc['f1_score']:.3f}")
            print(f"\n📋 Per-class AP@0.5:")
            for class_name, ap in acc['class_AP'].items():
                print(f"    {class_name}: {ap:.3f}")
        
        print(f"\n🏷️  Class breakdown:")
        for class_name, count in stats['class_counts'].items():
            print(f"    {class_name}: {count}")
        
        return stats

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Test YOLO model on SoccerNet GSR sequences")
    
    # Model arguments
    parser.add_argument("--model", "-m", type=str, 
                       default="best_new.pt",
                       help="Path to trained YOLO model")
    parser.add_argument("--confidence", "-c", type=float, default=0.25,
                       help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    
    # Data arguments
    parser.add_argument("--sequence", "-s", type=str, default="SNGS-021",
                       help="Sequence name to test (SoccerNet mode). Ignored in YOLO split mode.")
    parser.add_argument("--data_dir", type=str, 
                       default="SoccerNet/SN-GSR-2025/valid",
                       help="Path to test data directory. Supports YOLO split (images/labels) or SoccerNet sequences.")
    parser.add_argument("--all_sequences", action="store_true",
                       help="Test all valid sequences in data_dir (SoccerNet mode only)")
    
    # Output arguments
    parser.add_argument("--output", "-o", type=str, default="test_results",
                       help="Output directory for results")
    parser.add_argument("--no_images", action="store_true",
                       help="Don't save annotated images")
    parser.add_argument("--no_json", action="store_true",
                       help="Don't save JSON results")
    parser.add_argument("--no_video", action="store_true",
                       help="Don't create output video")
    
    # Processing arguments
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--frame_step", type=int, default=1,
                       help="Process every Nth frame")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference (default: 16, optimized for M4 Pro)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'auto', 'cpu', 'mps', 'cuda' (default: auto)")
    parser.add_argument("--no_accuracy", action="store_true",
                       help="Skip accuracy calculation (faster processing)")
    parser.add_argument("--legacy_mapping", action="store_true",
                       help="Use legacy class mapping: ball=0, goalkeeper=1, player=2, referee=3")
    parser.add_argument("--use_native_val", action="store_true",
                       help="Use YOLO's native validation (fastest, metrics only, no custom outputs)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("\n💡 Available model paths to try:")
        print("  - runs/detect/soccernet_gsr_v1_optimized/weights/best.pt")
        print("  - runs/detect/soccernet_gsr_v1_optimized/weights/last.pt")
        return
    
    print("🏈 SoccerNet GSR - YOLO Model Testing")
    print("=" * 40)
    print(f"📦 Model: {args.model}")
    print(f"🎯 Sequence: {args.sequence}")
    print(f"📊 Confidence: {args.confidence}")
    print(f"🔄 IoU threshold: {args.iou}")
    print(f"🖥️  Device: {args.device}")
    print(f"📈 Accuracy calculation: {'Disabled' if args.no_accuracy else 'Enabled'}")
    print(f"🔄 Class mapping: {'Legacy' if args.legacy_mapping else 'Standard'}")
    print(f"📦 Batch size: {args.batch_size}")
    if args.all_sequences:
        print(f"🔄 Testing mode: All sequences")
    else:
        print(f"🔄 Testing mode: Single sequence")
    
    # Initialize tester
    tester = SoccerNetTester(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou,
        device=args.device,
        legacy_mapping=args.legacy_mapping
    )
    
    # Decide mode: YOLO split vs SoccerNet sequence
    images_dir, labels_dir, split_name = tester._resolve_yolo_dirs(args.data_dir)
    is_yolo_split = images_dir is not None
    if is_yolo_split:
        print(f"🧭 Detected YOLO split at: {os.path.abspath(args.data_dir)} (split: {split_name})")
    else:
        print(f"🧭 Using SoccerNet sequence mode from: {os.path.abspath(args.data_dir)}")

    # Run testing
    try:
        # Use YOLO's native validation for maximum speed (metrics only)
        if args.use_native_val and is_yolo_split:
            print("\n🚀 Using YOLO native validation (fastest mode)")
            print("📊 This will only output metrics, no custom visualizations or JSON")
            
            # Construct path to dataset.yaml if it exists
            dataset_yaml = None
            possible_yaml_paths = [
                os.path.join(os.path.dirname(args.data_dir), "dataset.yaml"),
                os.path.join(os.path.dirname(args.data_dir), "data.yaml"),
                os.path.join(args.data_dir, "dataset.yaml"),
                os.path.join(args.data_dir, "data.yaml")
            ]
            
            for yaml_path in possible_yaml_paths:
                if os.path.exists(yaml_path):
                    dataset_yaml = yaml_path
                    break
            
            if dataset_yaml:
                print(f"📋 Using dataset config: {dataset_yaml}")
                start_time = time.time()
                
                # Run native validation
                # Fix device selection for native validation
                device_for_val = tester.device if tester.device != 'auto' else 'mps'
                metrics = tester.model.val(
                    data=dataset_yaml,
                    split=split_name,
                    batch=args.batch_size,
                    conf=args.confidence,
                    iou=args.iou,
                    device=device_for_val,
                    verbose=True
                )
                
                end_time = time.time()
                print(f"\n✅ Native validation completed in {end_time - start_time:.2f}s")
                print(f"📊 Results: mAP50={metrics.box.map50:.3f}, mAP50-95={metrics.box.map:.3f}")
                return
            else:
                print("⚠️  No dataset.yaml found, falling back to custom validation")
        
        if is_yolo_split:
            stats = tester.test_yolo_split(
                split_path=args.data_dir,
                output_dir=args.output,
                save_images=not args.no_images,
                save_json=not args.no_json,
                save_video=not args.no_video,
                max_frames=args.max_frames,
                frame_step=args.frame_step,
                calculate_accuracy=not args.no_accuracy,
                batch_size=args.batch_size
            )
        else:
            # Handle all sequences mode
            if args.all_sequences:
                # Get all subdirectories that look like valid sequences
                sequence_dirs = []
                for item in os.listdir(args.data_dir):
                    item_path = os.path.join(args.data_dir, item)
                    if os.path.isdir(item_path):
                        img1_path = os.path.join(item_path, "img1")
                        labels_path = os.path.join(item_path, "Labels-GameState.json")
                        if os.path.exists(img1_path) and os.path.exists(labels_path):
                            sequence_dirs.append(item)
                
                if not sequence_dirs:
                    print(f"❌ No valid sequences found in: {args.data_dir}")
                    return
                
                print(f"🎯 Found {len(sequence_dirs)} valid sequences to test: {sequence_dirs}")
                
                # Test each sequence
                all_stats = []
                for seq_name in sequence_dirs:
                    print(f"\n{'='*50}")
                    print(f"Testing sequence: {seq_name}")
                    print(f"{'='*50}")
                    
                    sequence_path = os.path.join(args.data_dir, seq_name)
                    stats = tester.test_sequence(
                        sequence_path=sequence_path,
                        output_dir=args.output,
                        save_images=not args.no_images,
                        save_json=not args.no_json,
                        save_video=not args.no_video,
                        max_frames=args.max_frames,
                        frame_step=args.frame_step,
                        calculate_accuracy=not args.no_accuracy
                    )
                    
                    if stats:
                        stats['sequence_name'] = seq_name
                        all_stats.append(stats)
                
                # Print summary of all sequences
                if all_stats:
                    print(f"\n{'='*60}")
                    print(f"AGGREGATED RESULTS FOR ALL SEQUENCES")
                    print(f"{'='*60}")
                    total_frames = sum(s['total_frames'] for s in all_stats)
                    total_detections = sum(s['total_detections'] for s in all_stats)
                    avg_fps = sum(s['fps'] for s in all_stats) / len(all_stats)
                    total_time = sum(s['processing_time'] for s in all_stats)
                    
                    print(f"  🎬 Total sequences tested: {len(all_stats)}")
                    print(f"  🖼️  Total frames processed: {total_frames}")
                    print(f"  🎯 Total detections: {total_detections}")
                    print(f"  ⚡ Avg processing speed: {avg_fps:.2f} FPS")
                    print(f"  ⏱️  Total time: {total_time:.2f}s")
                    
                    # Print per-sequence breakdown
                    print(f"\n📋 Per-sequence breakdown:")
                    for stats in all_stats:
                        print(f"    {stats['sequence_name']}: {stats['total_frames']} frames, "
                              f"{stats['fps']:.2f} FPS, {stats['processing_time']:.2f}s")
                    
                    print(f"\n✅ All sequences testing completed successfully!")
                    print(f"📁 Results saved in: {args.output}")
            else:
                # Build sequence path for SoccerNet mode
                sequence_path = os.path.join(args.data_dir, args.sequence)
                if not os.path.exists(sequence_path):
                    print(f"❌ Sequence not found: {sequence_path}")
                    return
                stats = tester.test_sequence(
                    sequence_path=sequence_path,
                    output_dir=args.output,
                    save_images=not args.no_images,
                    save_json=not args.no_json,
                    save_video=not args.no_video,
                    max_frames=args.max_frames,
                    frame_step=args.frame_step,
                    calculate_accuracy=not args.no_accuracy
                )
        
        if not args.all_sequences and not is_yolo_split:
            if stats:
                print("\n✅ Testing completed successfully!")
                print(f"📁 Results saved in: {args.output}/{args.sequence}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()