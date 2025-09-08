import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


# Line type to class ID mapping (exact names from annotations)
LINE_CLASS_MAPPING = {
    'background': 0,  # No line/background
    'Side line top': 1,
    'Circle central': 2,
    'Big rect. right main': 3,
    'Big rect. right top': 4,
    'Big rect. left main': 5,
    'Big rect. left top': 6,
    'Circle right': 7,
    'Middle line': 8,
    'Side line right': 9,
    'Circle left': 10,
    'Side line bottom': 11,
    'Side line left': 12,
    'Small rect. right main': 13,
    'Small rect. right top': 14,
    'Small rect. left main': 15,
    'Small rect. left top': 16,
    'Goal right post left': 17,
    'Goal right crossbar': 18,
    'Goal left post right': 19,
    'Goal left crossbar': 20,
    'Goal right post right': 21,
    'Small rect. right bottom': 22,
    'Big rect. right bottom': 23,
    'Goal left post left': 24,
    'Small rect. left bottom': 25,
    'Big rect. left bottom': 26,
}

# Reverse mapping for visualization/debugging
CLASS_TO_LINE_MAPPING = {v: k for k, v in LINE_CLASS_MAPPING.items()}

# Total number of classes
NUM_CLASSES = len(LINE_CLASS_MAPPING)


def discover_frame_annotation_pairs(
    data_root: str,
    output_csv: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Discover and pair frame images with their corresponding annotations.
    
    Args:
        data_root: Root directory containing SoccerNet GSR data (e.g., "SoccerNet/SN-GSR-2025/train")
        output_csv: Optional path to save pairs as CSV
        
    Returns:
        List of (frame_path, json_path) tuples
    """
    data_root = Path(data_root)
    valid_pairs = []
    skipped_dirs = []
    
    # Find all sequence directories (e.g., SNGS-075)
    sequence_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith('SNGS-')]
    logger.info(f"Found {len(sequence_dirs)} sequence directories in {data_root}")
    
    for seq_dir in sequence_dirs:
        labels_path = seq_dir / "Labels-GameState.json"
        
        if not labels_path.exists():
            skipped_dirs.append((str(seq_dir), "Missing JSON file"))
            continue
            
        # Validate JSON contains pitch labels
        try:
            with open(labels_path, 'r') as f:
                annotation = json.load(f)
                
            # Check if any image has pitch labels
            has_pitch_labels = False
            if "images" in annotation:
                for img in annotation["images"]:
                    if img.get("has_labeled_pitch", False):
                        has_pitch_labels = True
                        break
                        
            if not has_pitch_labels:
                skipped_dirs.append((str(seq_dir), "No pitch labels in JSON"))
                continue
            
            # Find all frames in this directory
            img_dir = seq_dir / "img1"
            if img_dir.exists():
                frame_files = list(img_dir.glob("*.jpg"))
                logger.info(f"Found {len(frame_files)} frames in {seq_dir.name}")
                
                for frame_path in frame_files:
                    valid_pairs.append((str(frame_path), str(labels_path)))
            else:
                skipped_dirs.append((str(seq_dir), "Missing img1 directory"))
                
        except (json.JSONDecodeError, KeyError) as e:
            skipped_dirs.append((str(seq_dir), f"JSON error: {e}"))
            continue
    
    logger.info(f"Valid pairs: {len(valid_pairs)}, Skipped directories: {len(skipped_dirs)}")
    
    # Log skipped directories for review
    if skipped_dirs:
        logger.warning("Skipped directories:")
        for dir_path, reason in skipped_dirs:
            logger.warning(f"  {dir_path}: {reason}")
    
    # Save to CSV if requested (optional for debugging)
    if output_csv and valid_pairs:
        df = pd.DataFrame(valid_pairs, columns=['frame_path', 'json_path'])
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(valid_pairs)} pairs to {output_csv}")
    
    return valid_pairs


def parse_pitch_geometry(
    json_path: str,
    frame_name: str,
    img_width: int,
    img_height: int
) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """
    Parse pitch geometry from JSON annotation for a specific frame.
    
    Args:
        json_path: Path to the Labels-GameState.json file
        frame_name: Name of the frame (e.g., "000001.jpg")
        img_width: Width of the image in pixels
        img_height: Height of the image in pixels
        
    Returns:
        Dictionary mapping line names to lists of (x_px, y_px) coordinates,
        or None if no pitch geometry found
    """
    try:
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        # Find the specific image
        if "images" not in annotation:
            logger.warning(f"No 'images' key in {json_path}")
            return None
            
        target_image = None
        for img in annotation["images"]:
            if img.get("file_name") == frame_name:
                target_image = img
                break
        
        # If exact match not found, try to find by frame number
        if target_image is None:
            try:
                # Extract frame number from filename (e.g., "000608.jpg" -> 608)
                frame_num = int(Path(frame_name).stem)
                
                # Look for image with matching frame number (1-indexed in JSON)
                for img in annotation["images"]:
                    img_name = img.get("file_name", "")
                    if img_name:
                        try:
                            img_frame_num = int(Path(img_name).stem)
                            if img_frame_num == frame_num:
                                target_image = img
                                break
                        except ValueError:
                            continue
            except ValueError:
                pass
        
        if target_image is None:
            logger.warning(f"Frame {frame_name} not found in {json_path}")
            return None
            
        if not target_image.get("has_labeled_pitch", False):
            return None
            
        # Find annotations for this image
        if "annotations" not in annotation:
            logger.warning(f"No 'annotations' key in {json_path}")
            return None
            
        pitch_annotations = []
        image_id = target_image.get("image_id")  # Changed from "id" to "image_id"
        
        for ann in annotation["annotations"]:
            if (ann.get("image_id") == image_id and 
                ann.get("supercategory") == "pitch"):
                pitch_annotations.append(ann)
        
        if not pitch_annotations:
            logger.debug(f"No pitch annotations found for {frame_name}")
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
                        # Handle dictionary format: {'x': 0.5, 'y': 0.3}
                        if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                            x_norm, y_norm = float(coord['x']), float(coord['y'])
                        # Handle list/tuple format: [0.5, 0.3]
                        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            x_norm, y_norm = float(coord[0]), float(coord[1])
                        else:
                            logger.warning(f"Invalid coordinate format: {coord}")
                            continue
                        
                        # Denormalize coordinates
                        x_px = x_norm * img_width
                        y_px = y_norm * img_height
                        
                        # Sanity check - clamp to image bounds
                        x_px = max(0, min(x_px, img_width))
                        y_px = max(0, min(y_px, img_height))
                        
                        denormalized_coords.append((x_px, y_px))
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Invalid coordinate format: {coord}, error: {e}")
                        continue
                
                if denormalized_coords:
                    pitch_lines[line_name] = denormalized_coords
        
        return pitch_lines if pitch_lines else None
        
    except (json.JSONDecodeError, KeyError, IOError) as e:
        logger.error(f"Error parsing pitch geometry from {json_path}: {e}")
        return None


def create_pitch_mask(
    pitch_lines: Dict[str, List[Tuple[float, float]]],
    img_width: int,
    img_height: int,
    line_thickness: int = 8,
    multiclass: bool = True
) -> np.ndarray:
    """
    Create multi-class or binary mask from pitch line coordinates.
    
    Args:
        pitch_lines: Dictionary mapping line names to lists of (x_px, y_px) coordinates
        img_width: Width of the output mask in pixels
        img_height: Height of the output mask in pixels
        line_thickness: Thickness of the drawn lines in pixels
        multiclass: If True, create multi-class mask; if False, create binary mask
        
    Returns:
        Multi-class mask as numpy array (height, width) with dtype uint8
        Background is 0, each line type has its own class ID
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for line_name, coords in pitch_lines.items():
        if not coords or len(coords) < 2:
            continue
            
        coords_array = np.array(coords, dtype=np.int32)
        
        # Get class ID for this line type
        if multiclass:
            class_id = LINE_CLASS_MAPPING.get(line_name, 0)  # Default to background if unknown
            if class_id == 0:
                logger.warning(f"Unknown line type: '{line_name}', treating as background")
                continue
        else:
            class_id = 255  # Binary mask: all lines are white
        
        # Determine if line should be closed based on line type
        is_closed = line_name in [
            'Circle central', 'Circle left', 'Circle right',
            'Big rect. left main', 'Big rect. right main',
            'Small rect. left main', 'Small rect. right main',
            'Big rect. left top', 'Big rect. right top',
            'Small rect. left top', 'Small rect. right top',
            'Big rect. left bottom', 'Big rect. right bottom',
            'Small rect. left bottom', 'Small rect. right bottom'
        ]
        
        cv2.polylines(mask, [coords_array], isClosed=is_closed, color=class_id, thickness=line_thickness)
    
    return mask


class PitchLocalizationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        line_thickness: int = 8,
        cache_masks: bool = False,
        subset: Optional[str] = None,
        multiclass: bool = True,
        num_classes: int = NUM_CLASSES
    ):
        """
        PyTorch Dataset for pitch localization training.
        
        Args:
            data_root: Root directory containing SoccerNet GSR data
            transform: Optional transforms to apply to images and masks
            target_size: Target size for resizing (width, height)
            line_thickness: Thickness for drawing pitch lines in masks
            cache_masks: Whether to cache generated masks for faster training
            subset: Optional subset name for debugging (e.g., 'train', 'val')
            multiclass: Whether to use multi-class or binary segmentation
            num_classes: Number of classes for multi-class segmentation
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.target_size = target_size
        self.line_thickness = line_thickness
        self.cache_masks = cache_masks
        self.subset = subset
        self.multiclass = multiclass
        self.num_classes = num_classes
        
        # Load sequences and annotations (efficient approach)
        logger.info(f"Loading sequences from {data_root}")
        self.sequences = self._load_sequences()
        logger.info(f"Loading annotations for {len(self.sequences)} sequences")
        self.annotations = self._load_annotations()
        
        # Build frame index for efficient access
        logger.info("Building frame index")
        self.frame_index = self._build_frame_index()
        logger.info(f"Found {len(self.frame_index)} frames total")
        
        # Cache for masks if enabled
        self.mask_cache = {} if cache_masks else None
        
        # Default image normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _load_sequences(self) -> List[str]:
        """Load list of sequence directories."""
        sequences = []
        for seq_dir in self.data_root.iterdir():
            if seq_dir.is_dir() and seq_dir.name.startswith('SNGS-'):
                sequences.append(seq_dir.name)
        return sorted(sequences)
    
    def _load_annotations(self) -> Dict:
        """Load all annotations for the sequences."""
        annotations = {}
        for seq_name in self.sequences:
            json_path = self.data_root / seq_name / "Labels-GameState.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    seq_data = json.load(f)
                annotations[seq_name] = seq_data
            else:
                logger.warning(f"No annotations found for {seq_name}")
        return annotations
    
    def _build_frame_index(self) -> List[Tuple[str, str, int]]:
        """Build index of (sequence, frame_path, frame_idx) for efficient access."""
        frame_index = []
        
        for seq_name in self.sequences:
            if seq_name not in self.annotations:
                continue
                
            seq_data = self.annotations[seq_name]
            img_dir = self.data_root / seq_name / "img1"
            
            if not img_dir.exists():
                continue
            
            # Only add frames that have pitch annotations
            images_with_pitch = {
                img.get("file_name"): img for img in seq_data.get("images", []) 
                if img.get("has_labeled_pitch", False)
            }
            
            if not images_with_pitch:
                logger.debug(f"Skipping {seq_name}: no pitch labels")
                continue
            
            # Add only frames with pitch annotations
            for img_file in img_dir.glob("*.jpg"):
                frame_name = img_file.name
                if frame_name in images_with_pitch:
                    frame_idx = int(img_file.stem)
                    frame_index.append((seq_name, str(img_file), frame_idx))
            
            logger.debug(f"Added {len([f for f in frame_index if f[0] == seq_name])} frames from {seq_name}")
        
        return frame_index
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'image': Processed image tensor (C, H, W)
            - 'mask': Binary mask tensor (1, H, W)
            - 'metadata': Additional information about the sample
        """
        if idx >= len(self.frame_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.frame_index)}")
        
        seq_name, frame_path, frame_idx = self.frame_index[idx]
        frame_name = f"{frame_idx:06d}.jpg"
        
        try:
            # Load image
            image = self._load_image(frame_path)
            original_size = image.size  # (width, height)
            
            # Generate or retrieve cached mask
            mask = self._get_mask(seq_name, frame_name, original_size)
            
            # This should not happen since we pre-filtered frames
            if mask is None:
                logger.error(f"No pitch geometry found for {frame_name} in {seq_name} - this should not happen!")
                return self._get_dummy_sample()
            
            # Apply transforms
            if self.transform:
                image, mask = self._apply_transforms(image, mask)
            else:
                image = self._default_transform(image)
                mask = self._resize_mask(mask, self.target_size)
            
            # Convert mask to tensor
            if self.multiclass:
                # For multi-class segmentation, keep integer class labels
                mask_tensor = torch.from_numpy(mask.astype(np.int64))  # (H, W) with class indices
            else:
                # For binary segmentation, normalize to [0, 1] range
                mask = mask.astype(np.float32) / 255.0  # Normalize from [0, 255] to [0, 1]
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
            
            # Prepare metadata
            metadata = {
                'frame_path': frame_path,
                'frame_name': frame_name,
                'sequence': seq_name,
                'original_size': original_size,
                'target_size': self.target_size
            }
            
            return {
                'image': image,
                'mask': mask_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({frame_path}): {e}")
            # Return a dummy sample to avoid breaking training
            return self._get_dummy_sample()
    
    def _load_image(self, frame_path: str) -> Image.Image:
        """Load image from file path."""
        image = Image.open(frame_path).convert('RGB')
        return image
    
    def _get_mask(self, seq_name: str, frame_name: str, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Generate or retrieve cached mask."""
        cache_key = f"{seq_name}:{frame_name}"
        
        if self.mask_cache is not None and cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        # Parse pitch geometry using pre-loaded annotations
        pitch_lines = self._parse_pitch_geometry(seq_name, frame_name, image_size[0], image_size[1])
        
        if pitch_lines is None:
            return None  # No pitch geometry found
        
        # Create mask
        mask = create_pitch_mask(
            pitch_lines, image_size[0], image_size[1], self.line_thickness, self.multiclass
        )
        
        # Cache if enabled
        if self.mask_cache is not None:
            self.mask_cache[cache_key] = mask
        
        return mask
    
    def _parse_pitch_geometry(self, seq_name: str, frame_name: str, img_width: int, img_height: int) -> Optional[Dict[str, List[Tuple[float, float]]]]:
        """Parse pitch geometry from pre-loaded annotations."""
        if seq_name not in self.annotations:
            return None
        
        seq_data = self.annotations[seq_name]
        
        # Find the specific image
        target_image = None
        for img in seq_data.get("images", []):
            if img.get("file_name") == frame_name:
                target_image = img
                break
        
        if target_image is None or not target_image.get("has_labeled_pitch", False):
            return None
        
        # Find annotations for this image
        pitch_annotations = []
        image_id = target_image.get("image_id")
        
        for ann in seq_data.get("annotations", []):
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
                        # Handle dictionary format: {'x': 0.5, 'y': 0.3}
                        if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                            x_norm, y_norm = float(coord['x']), float(coord['y'])
                        # Handle list/tuple format: [0.5, 0.3]
                        elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            x_norm, y_norm = float(coord[0]), float(coord[1])
                        else:
                            continue
                        
                        # Denormalize coordinates
                        x_px = x_norm * img_width
                        y_px = y_norm * img_height
                        
                        # Sanity check - clamp to image bounds
                        x_px = max(0, min(x_px, img_width))
                        y_px = max(0, min(y_px, img_height))
                        
                        denormalized_coords.append((x_px, y_px))
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if denormalized_coords:
                    pitch_lines[line_name] = denormalized_coords
        
        return pitch_lines if pitch_lines else None
    
    def _default_transform(self, image: Image.Image) -> torch.Tensor:
        """Apply default image transforms."""
        transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            self.normalize
        ])
        return transform(image)
    
    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize mask using nearest neighbor interpolation."""
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask_resized
    
    def _apply_transforms(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply transforms to both image and mask consistently."""
        if hasattr(self.transform, 'apply_to_mask'):
            # Custom transform that handles both image and mask
            image_tensor, mask_transformed = self.transform(image, mask)
        else:
            # Standard transforms - apply to image only, resize mask separately
            image_tensor = self.transform(image)
            mask_transformed = self._resize_mask(mask, self.target_size)
        
        return image_tensor, mask_transformed
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return a dummy sample in case of errors."""
        dummy_image = torch.zeros(3, self.target_size[1], self.target_size[0])
        
        if self.multiclass:
            dummy_mask = torch.zeros(self.target_size[1], self.target_size[0], dtype=torch.int64)
        else:
            dummy_mask = torch.zeros(1, self.target_size[1], self.target_size[0])
            
        dummy_metadata = {
            'frame_path': 'dummy',
            'frame_name': 'dummy.jpg',
            'sequence': 'dummy',
            'original_size': (1920, 1080),
            'target_size': self.target_size
        }
        
        return {
            'image': dummy_image,
            'mask': dummy_mask,
            'metadata': dummy_metadata
        }
    
    def get_sample_for_debugging(self, idx: int, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Get a sample with additional debugging information and optionally save visualizations."""
        sample = self[idx]
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)
            
            # Denormalize and save image
            image_denorm = self._denormalize_image(sample['image'])
            image_pil = transforms.ToPILImage()(image_denorm)
            image_pil.save(save_path / f"debug_image_{idx}.jpg")
            
            # Save mask
            if self.multiclass:
                # For multi-class, convert class indices to colors for visualization
                mask_np = sample['mask'].numpy().astype(np.uint8)
                # Scale class indices to visible range (0-255)
                mask_vis = (mask_np * (255 // self.num_classes)).astype(np.uint8)
                cv2.imwrite(str(save_path / f"debug_mask_{idx}.png"), mask_vis)
            else:
                # For binary, convert back to 0-255 range
                mask_np = (sample['mask'].squeeze().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(save_path / f"debug_mask_{idx}.png"), mask_np)
            
            logger.info(f"Saved debug visualizations to {save_path}")
        
        return sample
    
    def _denormalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image tensor for visualization."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return image_tensor * std + mean
    
    def split_dataset(self, train_ratio: float = 0.8) -> Tuple['PitchLocalizationDataset', 'PitchLocalizationDataset']:
        """Split dataset into train and validation sets."""
        total_samples = len(self.frame_index)
        train_size = int(total_samples * train_ratio)
        
        # Create indices
        indices = list(range(total_samples))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = PitchLocalizationDataset.__new__(PitchLocalizationDataset)
        train_dataset.__dict__.update(self.__dict__)
        train_dataset.frame_index = [self.frame_index[i] for i in train_indices]
        train_dataset.subset = 'train'
        
        val_dataset = PitchLocalizationDataset.__new__(PitchLocalizationDataset)
        val_dataset.__dict__.update(self.__dict__)
        val_dataset.frame_index = [self.frame_index[i] for i in val_indices]
        val_dataset.subset = 'val'
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_dataset, val_dataset


class PitchAugmentation:
    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        rotation_range: float = 15.0,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
        hue_range: float = 0.1,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.0,
        normalize: bool = True
    ):
        """
        Custom augmentation class that applies consistent transforms to images and masks.
        
        Args:
            target_size: Target size for resizing (width, height)
            rotation_range: Range of rotation in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment  
            saturation_range: Range for saturation adjustment
            hue_range: Range for hue adjustment
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            normalize: Whether to apply ImageNet normalization
        """
        self.target_size = target_size
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.normalize = normalize
        
        if normalize:
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Apply augmentations to image and mask consistently."""
        # Convert mask to PIL for consistent transformations
        mask_pil = Image.fromarray(mask).convert('L')
        
        # Apply geometric transformations
        if np.random.random() < self.horizontal_flip_prob:
            image = transforms.functional.hflip(image)
            mask_pil = transforms.functional.hflip(mask_pil)
        
        if np.random.random() < self.vertical_flip_prob:
            image = transforms.functional.vflip(image)
            mask_pil = transforms.functional.vflip(mask_pil)
        
        # Apply rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = transforms.functional.rotate(image, angle, fill=0)
            mask_pil = transforms.functional.rotate(mask_pil, angle, fill=0)
        
        # Resize both image and mask
        image = transforms.functional.resize(image, self.target_size)
        mask_pil = transforms.functional.resize(mask_pil, self.target_size, interpolation=Image.NEAREST)
        
        # Apply photometric transformations to image only
        if self.brightness_range > 0:
            brightness_factor = 1.0 + np.random.uniform(-self.brightness_range, self.brightness_range)
            image = transforms.functional.adjust_brightness(image, brightness_factor)
        
        if self.contrast_range > 0:
            contrast_factor = 1.0 + np.random.uniform(-self.contrast_range, self.contrast_range)
            image = transforms.functional.adjust_contrast(image, contrast_factor)
        
        if self.saturation_range > 0:
            saturation_factor = 1.0 + np.random.uniform(-self.saturation_range, self.saturation_range)
            image = transforms.functional.adjust_saturation(image, saturation_factor)
        
        if self.hue_range > 0:
            hue_factor = np.random.uniform(-self.hue_range, self.hue_range)
            image = transforms.functional.adjust_hue(image, hue_factor)
        
        # Convert to tensors
        image_tensor = transforms.functional.to_tensor(image)
        
        if self.normalize:
            image_tensor = self.normalize_transform(image_tensor)
        
        # Convert mask back to numpy
        mask_transformed = np.array(mask_pil)
        
        return image_tensor, mask_transformed
    
    def apply_to_mask(self):
        """Marker method to indicate this transform handles masks."""
        return True


def create_train_dataset(data_root: str, **kwargs) -> PitchLocalizationDataset:
    """Create training dataset with optional augmentations."""
    augment = kwargs.get('augment', True)
    multiclass = kwargs.get('multiclass', True)
    
    transform = None
    if augment:
        transform = PitchAugmentation(
            target_size=kwargs.get('target_size', (512, 512)),
            rotation_range=kwargs.get('rotation_range', 10.0),
            brightness_range=kwargs.get('brightness_range', 0.15),
            contrast_range=kwargs.get('contrast_range', 0.15),
            saturation_range=kwargs.get('saturation_range', 0.15),
            horizontal_flip_prob=kwargs.get('horizontal_flip_prob', 0.5)
        )
    
    return PitchLocalizationDataset(
        data_root=data_root,
        transform=transform,
        **{k: v for k, v in kwargs.items() if k not in [
            'rotation_range', 'brightness_range', 'contrast_range', 
            'saturation_range', 'horizontal_flip_prob', 'augment'
        ]}
    )


def create_val_dataset(data_root: str, **kwargs) -> PitchLocalizationDataset:
    """Create validation dataset without augmentations."""
    return PitchLocalizationDataset(
        data_root=data_root,
        transform=None,  # No augmentations for validation
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Adjust path as needed
    data_root = "/Users/boyan531/Documents/football/SoccerNet/SN-GSR-2025/train"
    
    print("=== Testing Dataset Creation ===")
    try:
        # Create dataset with augmentations
        train_dataset = create_train_dataset(
            data_root,
            target_size=(512, 512),
            cache_masks=False,
            rotation_range=10.0,
            horizontal_flip_prob=0.5
        )
        
        print(f"Created training dataset with {len(train_dataset)} samples")
        
        # Create validation dataset
        val_dataset = create_val_dataset(
            data_root,
            target_size=(512, 512),
            cache_masks=False
        )
        
        print(f"Created validation dataset with {len(val_dataset)} samples")
        
        # Test loading samples
        if len(train_dataset) > 0:
            print("\n=== Testing Sample Loading ===")
            
            # Load first few samples
            for i in range(min(3, len(train_dataset))):
                print(f"\nLoading sample {i}...")
                sample = train_dataset[i]
                
                print(f"  Image shape: {sample['image'].shape}")
                print(f"  Mask shape: {sample['mask'].shape}")
                print(f"  Sequence: {sample['metadata']['sequence']}")
                print(f"  Frame: {sample['metadata']['frame_name']}")
                print(f"  Original size: {sample['metadata']['original_size']}")
                print(f"  Target size: {sample['metadata']['target_size']}")
                if train_dataset.multiclass:
                    print(f"  Mask stats: min={sample['mask'].min().item()}, max={sample['mask'].max().item()}")
                    print(f"  Unique classes: {torch.unique(sample['mask']).tolist()}")
                    print(f"  Non-zero pixels: {torch.count_nonzero(sample['mask']).item()}")
                else:
                    print(f"  Mask stats: min={sample['mask'].min():.3f}, max={sample['mask'].max():.3f}")
                    print(f"  Non-zero pixels: {torch.count_nonzero(sample['mask']).item()}")
            
            # Test debugging functionality
            print("\n=== Testing Debug Functionality ===")
            debug_sample = train_dataset.get_sample_for_debugging(0, save_path="debug_output")
            print("Saved debug visualizations to debug_output/")
            
            # Test dataset splitting
            print("\n=== Testing Dataset Splitting ===")
            train_split, val_split = train_dataset.split_dataset(train_ratio=0.8)
            print(f"Split dataset: {len(train_split)} train, {len(val_split)} val samples")
            
        else:
            print("No valid samples found in dataset")
            
    except Exception as e:
        print(f"Error during dataset testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Original functionality for backward compatibility
    print("\n=== Testing Original Functions ===")
    pairs = discover_frame_annotation_pairs(data_root)
    
    print(f"Discovered {len(pairs)} valid frame-annotation pairs")
    
    # Test pitch geometry parsing - use frames that actually exist
    if pairs:
        print("\nTesting pitch geometry parsing:")
        
        # Get the first JSON file to find images with labeled pitch
        _, first_json = pairs[0]
        print(f"Using JSON from: {Path(first_json).parent.name}")
        try:
            with open(first_json, 'r') as f:
                annotation = json.load(f)
            
            # Find images with labeled pitch
            labeled_imgs = [img for img in annotation['images'] if img.get('has_labeled_pitch')][:2]
            
            if labeled_imgs:
                for i, img in enumerate(labeled_imgs):
                    frame_name = img.get('file_name')
                    img_width = img.get('width', 1920)
                    img_height = img.get('height', 1080)
                    
                    print(f"\nTest {i+1}: {frame_name}")
                    print(f"  Dimensions: {img_width}x{img_height}")
                    
                    pitch_lines = parse_pitch_geometry(first_json, frame_name, img_width, img_height)
                    
                    if pitch_lines:
                        print(f"  Found {len(pitch_lines)} line types:")
                        for line_name, coords in pitch_lines.items():
                            print(f"    {line_name}: {len(coords)} points")
                        
                        # Generate and save binary mask
                        mask = create_pitch_mask(pitch_lines, img_width, img_height)
                        mask_filename = f"test_mask_{frame_name.replace('.jpg', '.png')}"
                        cv2.imwrite(mask_filename, mask)
                        print(f"  Generated mask: {mask_filename} (shape: {mask.shape})")
                        print(f"  Mask stats: min={mask.min()}, max={mask.max()}, non-zero pixels={np.count_nonzero(mask)}")
                    else:
                        print("  No pitch geometry found")
            else:
                print("  No images with labeled pitch found for testing")
                
        except Exception as e:
            print(f"  Error during testing: {e}")