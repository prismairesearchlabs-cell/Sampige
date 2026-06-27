"""
KITTI Dataset Implementation
Handles loading images and YOLO-format labels for object detection
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import random

from .config import config
from .transforms import get_transforms, get_augmentations
from .collate import collate_fn


class KITTIDataset(Dataset):
    """
    KITTI Dataset for object detection with YOLO format labels
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO format label files
        class_file: JSON file containing class names
        image_size: Target image size
        augment: Whether to apply augmentations
        transform: Custom transforms
        split: Dataset split ('train', 'val', 'test')
    """
    
    def __init__(
        self,
        image_dir: str = config.path.IMAGE_DIR,
        label_dir: str = config.path.LABEL_DIR,
        class_file: str = config.path.CLASS_FILE,
        image_size: int = config.data.IMAGE_SIZE,
        augment: bool = True,
        transform: Optional[transforms.Compose] = None,
        split: str = "train"
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_file = class_file
        self.image_size = image_size
        self.augment = augment
        self.split = split
        
        # Load class names
        self.classes = self._load_classes()
        self.num_classes = len(self.classes)
        
        # Get image and label files
        self.image_files, self.label_files = self._get_file_lists()
        
        # Set up transforms
        if transform is None:
            self.transform = get_transforms(
                image_size=image_size,
                augment=augment,
                split=split
            )
        else:
            self.transform = transform
        
        # Additional augmentations
        self.augmentations = get_augmentations() if augment else None
        
        print(f"KITTI Dataset initialized: {len(self.image_files)} images, {self.num_classes} classes")
    
    def _load_classes(self) -> List[str]:
        """Load class names from JSON file"""
        try:
            with open(self.class_file, 'r') as f:
                class_data = json.load(f)
                return class_data.get('names', [])
        except FileNotFoundError:
            # Default KITTI classes
            return [
                'Car', 'Van', 'Truck', 'Pedestrian', 
                'Person_sitting', 'Cyclist', 'Tram', 'Misc'
            ]
    
    def _get_file_lists(self) -> Tuple[List[str], List[str]]:
        """Get lists of image and label files"""
        image_files = []
        label_files = []
        
        # Get all image files
        if os.path.exists(self.image_dir):
            for filename in sorted(os.listdir(self.image_dir)):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(self.image_dir, filename)
                    image_files.append(image_path)
                    
                    # Find corresponding label file
                    label_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
                    label_path = os.path.join(self.label_dir, label_filename)
                    
                    if os.path.exists(label_path):
                        label_files.append(label_path)
                    else:
                        label_files.append("")  # Empty label
        
        return image_files, label_files
    
    def _load_image(self, path: str) -> Image.Image:
        """Load image from path"""
        try:
            image = Image.open(path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return blank image
            return Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
    
    def _load_labels(self, path: str) -> np.ndarray:
        """Load YOLO format labels from file"""
        if not path or not os.path.exists(path):
            return np.zeros((0, 6))  # No labels
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            
            labels = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x_center, y_center, width, height = map(float, parts[:5])
                    labels.append([class_id, x_center, y_center, width, height])
            
            return np.array(labels, dtype=np.float32)
        except Exception as e:
            print(f"Error loading labels {path}: {e}")
            return np.zeros((0, 6))
    
    def _convert_to_xyxy(self, labels: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        """Convert YOLO format (cx, cy, w, h) to xyxy format"""
        if len(labels) == 0:
            return np.zeros((0, 5))
        
        converted = np.zeros((len(labels), 5))
        for i, label in enumerate(labels):
            class_id, cx, cy, w, h = label
            
            # Convert from normalized [0,1] to absolute coordinates
            x_center = cx * image_width
            y_center = cy * image_height
            width = w * image_width
            height = h * image_height
            
            # Convert to xyxy format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            converted[i] = [x1, y1, x2, y2, class_id]
        
        return converted
    
    def _apply_augmentations(self, image: Image.Image, labels: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Apply augmentations to image and labels"""
        if not self.augment or self.augmentations is None:
            return image, labels
        
        # Random horizontal flip
        if random.random() < 0.5 and config.data.HORIZONTAL_FLIP:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(labels) > 0:
                # Flip bounding boxes
                image_width = image.width
                labels[:, 0] = image_width - labels[:, 2]  # x1
                labels[:, 2] = image_width - labels[:, 0]  # x2
                # Swap x1 and x2
                labels[:, [0, 2]] = labels[:, [2, 0]]
        
        # Color augmentations
        if random.random() < config.data.COLOR_JITTER_PROB:
            # Brightness
            brightness_factor = random.uniform(
                max(0, 1 - config.data.BRIGHTNESS), 
                1 + config.data.BRIGHTNESS
            )
            image = transforms.functional.adjust_brightness(image, brightness_factor)
            
            # Contrast
            contrast_factor = random.uniform(
                max(0, 1 - config.data.CONTRAST), 
                1 + config.data.CONTRAST
            )
            image = transforms.functional.adjust_contrast(image, contrast_factor)
            
            # Saturation
            saturation_factor = random.uniform(
                max(0, 1 - config.data.SATURATION), 
                1 + config.data.SATURATION
            )
            image = transforms.functional.adjust_saturation(image, saturation_factor)
            
            # Hue
            hue_factor = random.uniform(-config.data.HUE, config.data.HUE)
            image = transforms.functional.adjust_hue(image, hue_factor)
        
        return image, labels
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item
        
        Returns:
            Dictionary containing:
                - image: Transformed image tensor (C, H, W)
                - labels: Bounding boxes in xyxy format with class ids (N, 5)
                - image_path: Original image path
                - image_id: Index
                - original_size: Original image dimensions
        """
        # Load image
        image_path = self.image_files[idx]
        image = self._load_image(image_path)
        
        # Store original dimensions
        original_width, original_height = image.size
        
        # Load and convert labels
        label_path = self.label_files[idx]
        yolo_labels = self._load_labels(label_path)
        xyxy_labels = self._convert_to_xyxy(yolo_labels, original_width, original_height)
        
        # Apply augmentations
        image, xyxy_labels = self._apply_augmentations(image, xyxy_labels)
        
        # Apply transforms
        if self.transform:
            # Convert to tensor and normalize
            image_tensor = self.transform(image)
        else:
            # Default transform
            resize = transforms.Resize((self.image_size, self.image_size))
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(
                mean=config.data.MEAN, 
                std=config.data.STD
            )
            image_tensor = normalize(to_tensor(resize(image)))
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(xyxy_labels, dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'labels': labels_tensor,
            'image_path': image_path,
            'image_id': idx,
            'original_size': torch.tensor([original_width, original_height], dtype=torch.float32),
            'scale_factor': torch.tensor([
                self.image_size / original_width, 
                self.image_size / original_height
            ], dtype=torch.float32)
        }


class KITTIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for KITTI dataset
    
    Handles train/val/test splits and data loaders
    """
    
    def __init__(
        self,
        batch_size: int = config.training.BATCH_SIZE,
        val_batch_size: int = config.validation.VAL_BATCH_SIZE,
        image_size: int = config.data.IMAGE_SIZE,
        num_workers: int = config.training.NUM_WORKERS,
        train_split: float = config.data.TRAIN_SPLIT,
        val_split: float = config.data.VAL_SPLIT,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        
        # Dataset instances
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets"""
        # Create full dataset
        full_dataset = KITTIDataset(
            image_dir=config.path.IMAGE_DIR,
            label_dir=config.path.LABEL_DIR,
            class_file=config.path.CLASS_FILE,
            image_size=self.image_size,
            augment=False,  # Will be handled in train/val
            split="train"
        )
        
        # Split dataset
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.train_split)
        val_size = int(dataset_size * self.val_split)
        test_size = dataset_size - train_size - val_size
        
        # Create splits
        train_dataset = torch.utils.data.Subset(
            full_dataset, 
            range(0, train_size)
        )
        val_dataset = torch.utils.data.Subset(
            full_dataset,
            range(train_size, train_size + val_size)
        )
        test_dataset = torch.utils.data.Subset(
            full_dataset,
            range(train_size + val_size, dataset_size)
        )
        
        # Create dataset instances with appropriate augmentations
        self.train_dataset = KITTIDataset(
            image_dir=config.path.IMAGE_DIR,
            label_dir=config.path.LABEL_DIR,
            class_file=config.path.CLASS_FILE,
            image_size=self.image_size,
            augment=True,
            split="train"
        )
        
        # Filter to only include train split indices
        self.train_dataset.image_files = [
            full_dataset.image_files[i] for i in range(train_size)
        ]
        self.train_dataset.label_files = [
            full_dataset.label_files[i] for i in range(train_size)
        ]
        
        self.val_dataset = KITTIDataset(
            image_dir=config.path.IMAGE_DIR,
            label_dir=config.path.LABEL_DIR,
            class_file=config.path.CLASS_FILE,
            image_size=self.image_size,
            augment=False,
            split="val"
        )
        
        # Filter to only include val split indices
        self.val_dataset.image_files = [
            full_dataset.image_files[i] for i in range(train_size, train_size + val_size)
        ]
        self.val_dataset.label_files = [
            full_dataset.label_files[i] for i in range(train_size, train_size + val_size)
        ]
        
        self.test_dataset = KITTIDataset(
            image_dir=config.path.IMAGE_DIR,
            label_dir=config.path.LABEL_DIR,
            class_file=config.path.CLASS_FILE,
            image_size=self.image_size,
            augment=False,
            split="test"
        )
        
        # Filter to only include test split indices
        self.test_dataset.image_files = [
            full_dataset.image_files[i] for i in range(train_size + val_size, dataset_size)
        ]
        self.test_dataset.label_files = [
            full_dataset.label_files[i] for i in range(train_size + val_size, dataset_size)
        ]
        
        print(f"Dataset setup complete:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self):
        """Create training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=config.data.PIN_MEMORY,
            persistent_workers=config.data.PERSISTENT_WORKERS,
            drop_last=True
        )
    
    def val_dataloader(self):
        """Create validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=config.data.PIN_MEMORY,
            persistent_workers=config.data.PERSISTENT_WORKERS,
            drop_last=False
        )
    
    def test_dataloader(self):
        """Create test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=config.data.PIN_MEMORY,
            persistent_workers=config.data.PERSISTENT_WORKERS,
            drop_last=False
        )
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return self.train_dataset.classes if self.train_dataset else []
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return self.train_dataset.num_classes if self.train_dataset else 0


if __name__ == "__main__":
    # Test the dataset
    print("Testing KITTI Dataset...")
    
    # Create dataset
    dataset = KITTIDataset(
        image_dir=config.path.LOCAL_IMAGES if os.path.exists(config.path.LOCAL_IMAGES) else config.path.IMAGE_DIR,
        label_dir=config.path.LOCAL_LABELS if os.path.exists(config.path.LOCAL_LABELS) else config.path.LABEL_DIR,
        image_size=256,  # Smaller for testing
        augment=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first item
        item = dataset[0]
        print(f"Image shape: {item['image'].shape}")
        print(f"Labels shape: {item['labels'].shape}")
        print(f"Image path: {item['image_path']}")
        print(f"Classes: {dataset.classes}")
    
    # Test DataModule
    print("\nTesting KITTI DataModule...")
    datamodule = KITTIDataModule(
        batch_size=2,
        val_batch_size=2,
        image_size=256,
        num_workers=0
    )
    
    # Setup with minimal data
    datamodule.setup()
    
    # Test data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"Train loader length: {len(train_loader)}")
    print(f"Val loader length: {len(val_loader)}")
    
    # Test batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch labels: {len(batch['labels'])}")
