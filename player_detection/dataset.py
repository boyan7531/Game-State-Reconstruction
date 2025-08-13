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
        
        # Determine image dimensions without requiring the image to be loaded
        image = data.get("image")
        if image is not None:
            img_height, img_width = image.shape[:2]
        else:
            # Use metadata when image is not loaded (faster path)
            img_width = int(data["image_info"]["width"])  # type: ignore[index]
            img_height = int(data["image_info"]["height"])  # type: ignore[index]
        
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
            "labels": np.array(yolo_labels) if yolo_labels else np.empty((0, 5)),
            "sequence": data["sequence"],
            "frame_idx": data["frame_idx"],
            "image_info": data["image_info"],
        }
        
        # Only include the image array when it was actually loaded
        if image is not None:
            result["image"] = image
        
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
    print("Loading SoccerNet GSR dataset for detection...")
    
    # Base dataset
    base_dataset = SoccerNetGSRDataset("SoccerNet/SN-GSR-2025", split="train")
    print(f"Loaded {len(base_dataset.sequences)} sequences, {len(base_dataset)} frames")
    
    # YOLO dataset for detection
    yolo_dataset = YOLODataset(base_dataset)
    print(f"YOLO dataset: {len(yolo_dataset)} frames")
    
    # Test loading a sample
    if len(yolo_dataset) > 0:
        sample = yolo_dataset[0]
        print(f"Sample image shape: {sample['image'].shape}")
        print(f"Sample labels shape: {sample['labels'].shape}")