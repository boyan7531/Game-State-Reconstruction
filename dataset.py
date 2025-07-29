"""
SoccerNet GSR Dataset classes for loading and processing annotations.
Uses hybrid approach: base dataset + task-specific wrappers.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image


class SoccerNetGSRDataset(Dataset):
    """
    Base dataset class for SoccerNet Game State Reconstruction challenge.
    Handles core data loading functionality.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        load_images: bool = True,
        cache_annotations: bool = True
    ):
        """
        Args:
            data_root: Path to SoccerNet data directory
            split: Dataset split ('train', 'valid', 'test', 'challenge')
            load_images: Whether to load actual images
            cache_annotations: Whether to cache parsed annotations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.load_images = load_images
        self.cache_annotations = cache_annotations
        
        # Load sequences and annotations
        self.sequences = self._load_sequences()
        self.annotations = self._load_annotations()
        
        # Build frame index for efficient access
        self.frame_index = self._build_frame_index()
        
    def _load_sequences(self) -> List[str]:
        """Load list of sequence directories for the split."""
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        sequences = []
        for seq_dir in split_dir.iterdir():
            if seq_dir.is_dir() and seq_dir.name.startswith("SNGS-"):
                sequences.append(seq_dir.name)
        
        return sorted(sequences)
    
    def _load_annotations(self) -> Dict:
        """Load all annotations for the sequences."""
        annotations = {}
        
        for seq_name in self.sequences:
            json_path = self.data_root / self.split / seq_name / "Labels-GameState.json"
            
            if json_path.exists():
                with open(json_path, 'r') as f:
                    seq_data = json.load(f)
                annotations[seq_name] = seq_data
            else:
                print(f"Warning: No annotations found for {seq_name}")
                
        return annotations
    
    def _build_frame_index(self) -> List[Tuple[str, int, str]]:
        """Build index of (sequence, frame_idx, image_id) for efficient access."""
        frame_index = []
        
        for seq_name in self.sequences:
            if seq_name not in self.annotations:
                continue
                
            seq_data = self.annotations[seq_name]
            
            for img_info in seq_data["images"]:
                frame_idx = int(img_info["file_name"].split('.')[0])
                image_id = img_info["image_id"]
                frame_index.append((seq_name, frame_idx, image_id))
        
        return frame_index
    
    def get_sequence_info(self, seq_name: str) -> Dict:
        """Get metadata for a specific sequence."""
        if seq_name not in self.annotations:
            raise ValueError(f"Sequence {seq_name} not found")
            
        return self.annotations[seq_name]["info"]
    
    def get_frame_annotations(self, seq_name: str, frame_idx: int) -> Dict:
        """Get annotations for a specific frame."""
        if seq_name not in self.annotations:
            raise ValueError(f"Sequence {seq_name} not found")
            
        seq_data = self.annotations[seq_name]
        
        # Find image info
        image_info = None
        for img in seq_data["images"]:
            if img["file_name"] == f"{frame_idx:06d}.jpg":
                image_info = img
                break
                
        if image_info is None:
            raise ValueError(f"Frame {frame_idx} not found in {seq_name}")
        
        # Get annotations for this frame
        frame_annotations = []
        for ann in seq_data["annotations"]:
            if ann["image_id"] == image_info["image_id"]:
                frame_annotations.append(ann)
                
        return {
            "image_info": image_info,
            "annotations": frame_annotations,
            "categories": seq_data["categories"]
        }
    
    def load_image(self, seq_name: str, frame_idx: int) -> np.ndarray:
        """Load image for a specific frame."""
        img_path = (self.data_root / self.split / seq_name / "img1" / 
                   f"{frame_idx:06d}.jpg")
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single frame with annotations."""
        seq_name, frame_idx, image_id = self.frame_index[idx]
        
        # Get annotations
        frame_data = self.get_frame_annotations(seq_name, frame_idx)
        
        result = {
            "sequence": seq_name,
            "frame_idx": frame_idx,
            "image_id": image_id,
            "annotations": frame_data["annotations"],
            "image_info": frame_data["image_info"]
        }
        
        # Load image if requested
        if self.load_images:
            image = self.load_image(seq_name, frame_idx)
            result["image"] = image
                
        return result


class YOLODataset(Dataset):
    """
    YOLO format dataset wrapper.
    Converts SoccerNet annotations to YOLO training format.
    """
    
    def __init__(
        self,
        base_dataset: SoccerNetGSRDataset,
        class_mapping: Optional[Dict[int, int]] = None,
        transform=None
    ):
        """
        Args:
            base_dataset: SoccerNet base dataset instance
            class_mapping: Mapping from SoccerNet category_id to YOLO class_id
            transform: Optional transforms to apply
        """
        self.base_dataset = base_dataset
        self.transform = transform
        
        # Default class mapping
        if class_mapping is None:
            self.class_mapping = {
                1: 0,  # player -> 0
                2: 1,  # goalkeeper -> 1  
                3: 2,  # referee -> 2
                4: 3,  # ball -> 3
            }
        else:
            self.class_mapping = class_mapping
    
    def _convert_bbox_to_yolo(self, bbox: Dict, img_width: int, img_height: int) -> List[float]:
        """Convert bounding box to YOLO format (normalized xywh)."""
        x = bbox["x"]
        y = bbox["y"] 
        w = bbox["w"]
        h = bbox["h"]
        
        # Convert to center coordinates and normalize
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        width = w / img_width
        height = h / img_height
        
        return [x_center, y_center, width, height]
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get frame in YOLO format."""
        # Get base data
        data = self.base_dataset[idx]
        
        if "image" not in data:
            raise ValueError("Base dataset must load images for YOLO format")
        
        image = data["image"]
        img_height, img_width = image.shape[:2]
        
        # Convert annotations to YOLO format
        yolo_labels = []
        for ann in data["annotations"]:
            if ann["category_id"] in self.class_mapping:
                class_id = self.class_mapping[ann["category_id"]]
                bbox_yolo = self._convert_bbox_to_yolo(
                    ann["bbox_image"], img_width, img_height
                )
                yolo_labels.append([class_id] + bbox_yolo)
        
        result = {
            "image": image,
            "labels": np.array(yolo_labels) if yolo_labels else np.empty((0, 5)),
            "sequence": data["sequence"],
            "frame_idx": data["frame_idx"]
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result


class TeamClassificationDataset(Dataset):
    """
    Team classification dataset wrapper.
    Extracts player crops with team labels.
    """
    
    def __init__(
        self,
        base_dataset: SoccerNetGSRDataset,
        crop_size: Tuple[int, int] = (224, 224),
        padding: float = 0.1,
        transform=None
    ):
        """
        Args:
            base_dataset: SoccerNet base dataset instance
            crop_size: Target size for player crops
            padding: Padding around bounding box (as fraction of bbox size)
            transform: Optional transforms to apply
        """
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.padding = padding
        self.transform = transform
        
        # Build player index
        self.player_index = self._build_player_index()
    
    def _build_player_index(self) -> List[Tuple[int, int]]:
        """Build index of (frame_idx, annotation_idx) for players."""
        player_index = []
        
        # Temporarily disable image loading for indexing
        original_load_images = self.base_dataset.load_images
        self.base_dataset.load_images = False
        
        for frame_idx in range(len(self.base_dataset)):
            data = self.base_dataset[frame_idx]  # Get without loading image
            
            for ann_idx, ann in enumerate(data["annotations"]):
                # Only include players and goalkeepers with team info
                if (ann["category_id"] in [1, 2] and 
                    "attributes" in ann and 
                    "team" in ann["attributes"] and
                    ann["attributes"]["team"] in ["left", "right"]):
                    player_index.append((frame_idx, ann_idx))
        
        # Restore original setting
        self.base_dataset.load_images = original_load_images
        
        return player_index
    
    def _extract_player_crop(self, image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Extract and resize player crop from image."""
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        # Add padding
        pad_w, pad_h = int(w * self.padding), int(h * self.padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        
        # Resize to target size
        if crop.size > 0:
            crop_pil = Image.fromarray(crop)
            crop_pil = crop_pil.resize(self.crop_size, Image.LANCZOS)
            crop = np.array(crop_pil)
        
        return crop
    
    def __len__(self) -> int:
        return len(self.player_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get player crop with team label."""
        frame_idx, ann_idx = self.player_index[idx]
        
        # Get frame data
        data = self.base_dataset[frame_idx]
        image = data["image"]
        annotation = data["annotations"][ann_idx]
        
        # Extract player crop
        crop = self._extract_player_crop(image, annotation["bbox_image"])
        
        # Get team label
        team = annotation["attributes"]["team"]
        team_label = 0 if team == "left" else 1  # left=0, right=1
        
        result = {
            "image": crop,
            "team": team_label,
            "team_name": team,
            "sequence": data["sequence"],
            "frame_idx": data["frame_idx"],
            "track_id": annotation.get("track_id", -1),
            "category": annotation["category_id"]  # 1=player, 2=goalkeeper
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result


class JerseyOCRDataset(Dataset):
    """
    Jersey number recognition dataset wrapper.
    Extracts jersey regions with number labels.
    """
    
    def __init__(
        self,
        base_dataset: SoccerNetGSRDataset,
        crop_size: Tuple[int, int] = (128, 64),
        jersey_region: str = "upper",  # "upper", "full", "center"
        transform=None
    ):
        """
        Args:
            base_dataset: SoccerNet base dataset instance
            crop_size: Target size for jersey crops
            jersey_region: Which part of player to focus on
            transform: Optional transforms to apply
        """
        self.base_dataset = base_dataset
        self.crop_size = crop_size
        self.jersey_region = jersey_region
        self.transform = transform
        
        # Build jersey index
        self.jersey_index = self._build_jersey_index()
    
    def _build_jersey_index(self) -> List[Tuple[int, int]]:
        """Build index of (frame_idx, annotation_idx) for players with jersey numbers."""
        jersey_index = []
        
        for frame_idx in range(len(self.base_dataset)):
            data = self.base_dataset[frame_idx]  # Get without loading image
            
            for ann_idx, ann in enumerate(data["annotations"]):
                # Only include players/goalkeepers with jersey numbers
                if (ann["category_id"] in [1, 2] and 
                    "attributes" in ann and 
                    "jersey" in ann["attributes"] and
                    ann["attributes"]["jersey"] is not None and
                    ann["attributes"]["jersey"].isdigit()):
                    jersey_index.append((frame_idx, ann_idx))
        
        return jersey_index
    
    def _extract_jersey_region(self, image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Extract jersey region from player bounding box."""
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        
        if self.jersey_region == "upper":
            # Focus on upper 60% of player
            y2 = y + int(h * 0.6)
            crop = image[y:y2, x:x+w]
        elif self.jersey_region == "center":
            # Focus on center region
            margin_h, margin_w = int(h * 0.2), int(w * 0.1)
            crop = image[y+margin_h:y+h-margin_h, x+margin_w:x+w-margin_w]
        else:  # full
            crop = image[y:y+h, x:x+w]
        
        # Resize to target size
        if crop.size > 0:
            crop_pil = Image.fromarray(crop)
            crop_pil = crop_pil.resize(self.crop_size, Image.LANCZOS)
            crop = np.array(crop_pil)
        
        return crop
    
    def __len__(self) -> int:
        return len(self.jersey_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get jersey crop with number label."""
        frame_idx, ann_idx = self.jersey_index[idx]
        
        # Get frame data
        data = self.base_dataset[frame_idx]
        image = data["image"]
        annotation = data["annotations"][ann_idx]
        
        # Extract jersey region
        crop = self._extract_jersey_region(image, annotation["bbox_image"])
        
        # Get jersey number
        jersey_number = annotation["attributes"]["jersey"]
        
        result = {
            "image": crop,
            "jersey_number": int(jersey_number),
            "jersey_text": jersey_number,
            "sequence": data["sequence"],
            "frame_idx": data["frame_idx"],
            "track_id": annotation.get("track_id", -1),
            "team": annotation["attributes"].get("team", "unknown")
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result


class PitchLocalizationDataset(Dataset):
    """
    Pitch localization dataset wrapper.
    Provides full frames with pitch line annotations.
    """
    
    def __init__(
        self,
        base_dataset: SoccerNetGSRDataset,
        target_size: Optional[Tuple[int, int]] = None,
        transform=None
    ):
        """
        Args:
            base_dataset: SoccerNet base dataset instance
            target_size: Optional target size for images
            transform: Optional transforms to apply
        """
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.transform = transform
        
        # Build pitch index (frames with pitch annotations)
        self.pitch_index = self._build_pitch_index()
    
    def _build_pitch_index(self) -> List[int]:
        """Build index of frame indices with pitch annotations."""
        pitch_index = []
        
        for frame_idx in range(len(self.base_dataset)):
            data = self.base_dataset[frame_idx]  # Get without loading image
            
            # Check if frame has pitch annotations
            has_pitch = any(ann["category_id"] == 5 for ann in data["annotations"])
            if has_pitch and data["image_info"].get("has_labeled_pitch", False):
                pitch_index.append(frame_idx)
        
        return pitch_index
    
    def __len__(self) -> int:
        return len(self.pitch_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get frame with pitch annotations."""
        frame_idx = self.pitch_index[idx]
        
        # Get frame data
        data = self.base_dataset[frame_idx]
        image = data["image"]
        
        # Extract pitch annotations
        pitch_annotations = [ann for ann in data["annotations"] if ann["category_id"] == 5]
        
        # Resize image if needed
        if self.target_size:
            h, w = image.shape[:2]
            target_w, target_h = self.target_size
            
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(self.target_size, Image.LANCZOS)
            image = np.array(image_pil)
            
            # Scale factors for annotations
            scale_x = target_w / w
            scale_y = target_h / h
        else:
            scale_x = scale_y = 1.0
        
        result = {
            "image": image,
            "pitch_annotations": pitch_annotations,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "sequence": data["sequence"],
            "frame_idx": data["frame_idx"],
            "original_size": (data["image_info"]["width"], data["image_info"]["height"])
        }
        
        if self.transform:
            result = self.transform(result)
            
        return result


# Utility functions
def create_yolo_files(dataset: YOLODataset, output_dir: str):
    """Create YOLO format files from dataset."""
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Save image
        img_filename = f"{data['sequence']}_{data['frame_idx']:06d}.jpg"
        img_path = images_dir / img_filename
        
        image_pil = Image.fromarray(data["image"])
        image_pil.save(img_path)
        
        # Save labels
        label_filename = f"{data['sequence']}_{data['frame_idx']:06d}.txt"
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for label in data["labels"]:
                # Format: class_id (int) x_center y_center width height (floats)
                class_id = int(label[0])
                coords = label[1:]
                f.write(f"{class_id} {' '.join(map(str, coords))}\n")
        
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} frames")


if __name__ == "__main__":
    # Example usage
    print("Loading SoccerNet GSR dataset...")
    
    # Base dataset
    base_dataset = SoccerNetGSRDataset("SoccerNet/SN-GSR-2025", split="train")
    print(f"Loaded {len(base_dataset.sequences)} sequences, {len(base_dataset)} frames")
    
    # YOLO dataset
    yolo_dataset = YOLODataset(base_dataset)
    print(f"YOLO dataset: {len(yolo_dataset)} frames")
    
    # Team classification dataset
    team_dataset = TeamClassificationDataset(base_dataset)
    print(f"Team dataset: {len(team_dataset)} player crops")
    
    # Jersey OCR dataset
    jersey_dataset = JerseyOCRDataset(base_dataset)
    print(f"Jersey dataset: {len(jersey_dataset)} jersey crops")
    
    # Pitch localization dataset
    pitch_dataset = PitchLocalizationDataset(base_dataset)
    print(f"Pitch dataset: {len(pitch_dataset)} frames with pitch annotations")
    
    # Test loading a sample
    if len(yolo_dataset) > 0:
        sample = yolo_dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")