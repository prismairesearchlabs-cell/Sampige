"""
Custom Collate Function for KITTI Dataset
Handles variable-length bounding boxes and batching
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from config import config


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for KITTI dataset
    Handles variable-length bounding boxes by padding
    
    Args:
        batch: List of dictionary items from dataset
    
    Returns:
        Batched dictionary with stacked tensors
    """
    # Separate batch items
    images = []
    labels_list = []
    image_paths = []
    image_ids = []
    original_sizes = []
    scale_factors = []
    
    for item in batch:
        images.append(item['image'])
        labels_list.append(item['labels'])
        image_paths.append(item['image_path'])
        image_ids.append(item['image_id'])
        original_sizes.append(item['original_size'])
        scale_factors.append(item['scale_factor'])
    
    # Stack images (all same size)
    batched_images = torch.stack(images, dim=0)
    
    # Handle variable-length labels
    # Find max number of labels in batch
    max_labels = max(len(labels) for labels in labels_list) if labels_list else 0
    
    # Create padded labels tensor
    if max_labels > 0:
        padded_labels = torch.zeros((len(batch), max_labels, 5), dtype=torch.float32)
        
        for i, labels in enumerate(labels_list):
            num_labels = len(labels)
            if num_labels > 0:
                padded_labels[i, :num_labels] = labels
            # Fill remaining with -1 to indicate padding
            if num_labels < max_labels:
                padded_labels[i, num_labels:, 0] = -1  # Mark as invalid
    else:
        padded_labels = torch.zeros((len(batch), 0, 5), dtype=torch.float32)
    
    # Stack other tensors
    batched_original_sizes = torch.stack(original_sizes, dim=0)
    batched_scale_factors = torch.stack(scale_factors, dim=0)
    
    # Create label mask to indicate valid labels
    label_mask = torch.zeros((len(batch), max_labels), dtype=torch.bool)
    for i, labels in enumerate(labels_list):
        num_labels = len(labels)
        label_mask[i, :num_labels] = True
    
    return {
        'image': batched_images,
        'labels': padded_labels,
        'label_mask': label_mask,
        'image_paths': image_paths,
        'image_ids': torch.tensor(image_ids, dtype=torch.long),
        'original_sizes': batched_original_sizes,
        'scale_factors': batched_scale_factors,
        'batch_size': len(batch)
    }


class CustomCollate:
    """
    Custom collate class for more control over batching
    """
    
    def __init__(
        self,
        max_labels: int = 100,
        pad_value: float = -1.0
    ):
        self.max_labels = max_labels
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function with configurable parameters
        """
        # Separate batch items
        images = []
        labels_list = []
        image_paths = []
        image_ids = []
        original_sizes = []
        scale_factors = []
        
        for item in batch:
            images.append(item['image'])
            labels_list.append(item['labels'])
            image_paths.append(item['image_path'])
            image_ids.append(item['image_id'])
            original_sizes.append(item['original_size'])
            scale_factors.append(item['scale_factor'])
        
        # Stack images
        batched_images = torch.stack(images, dim=0)
        
        # Handle labels with fixed max
        use_max = min(self.max_labels, max(len(labels) for labels in labels_list)) if labels_list else 0
        padded_labels = torch.full((len(batch), use_max, 5), self.pad_value, dtype=torch.float32)
        
        label_mask = torch.zeros((len(batch), use_max), dtype=torch.bool)
        
        for i, labels in enumerate(labels_list):
            num_labels = min(len(labels), use_max)
            if num_labels > 0:
                padded_labels[i, :num_labels] = labels[:num_labels]
                label_mask[i, :num_labels] = True
        
        # Stack other tensors
        batched_original_sizes = torch.stack(original_sizes, dim=0)
        batched_scale_factors = torch.stack(scale_factors, dim=0)
        
        return {
            'image': batched_images,
            'labels': padded_labels,
            'label_mask': label_mask,
            'image_paths': image_paths,
            'image_ids': torch.tensor(image_ids, dtype=torch.long),
            'original_sizes': batched_original_sizes,
            'scale_factors': batched_scale_factors,
            'batch_size': len(batch)
        }


def collate_fn_teacher(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for teacher model inputs
    Only handles images, no labels needed
    """
    images = [item['image'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    batched_images = torch.stack(images, dim=0)
    
    return {
        'image': batched_images,
        'image_paths': image_paths,
        'batch_size': len(batch)
    }


def collate_fn_detection(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function optimized for detection tasks
    """
    return collate_fn(batch)


class BatchProcessor:
    """
    Utility class for processing batches
    """
    
    @staticmethod
    def filter_valid_labels(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out invalid/padded labels from batch
        """
        labels = batch['labels']
        label_mask = batch['label_mask']
        
        valid_labels = []
        for i in range(labels.shape[0]):
            # Get valid labels for this image
            valid_idx = label_mask[i]
            valid = labels[i][valid_idx]
            valid_labels.append(valid)
        
        batch['valid_labels'] = valid_labels
        return batch
    
    @staticmethod
    def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move batch tensors to device
        """
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, list):
                # Handle list of tensors
                batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        return batch
    
    @staticmethod
    def get_max_labels(batch: Dict[str, Any]) -> int:
        """
        Get maximum number of labels in batch
        """
        if 'labels' not in batch:
            return 0
        
        label_mask = batch.get('label_mask', None)
        if label_mask is not None:
            return label_mask.sum(dim=1).max().item()
        else:
            return batch['labels'].shape[1]


if __name__ == "__main__":
    # Test collate function
    print("Testing collate function...")
    
    # Create dummy batch
    batch = [
        {
            'image': torch.randn(3, 256, 256),
            'labels': torch.tensor([[10, 20, 100, 150, 0], [50, 60, 120, 180, 1]], dtype=torch.float32),
            'image_path': 'image1.jpg',
            'image_id': 0,
            'original_size': torch.tensor([640, 480], dtype=torch.float32),
            'scale_factor': torch.tensor([0.4, 0.533], dtype=torch.float32)
        },
        {
            'image': torch.randn(3, 256, 256),
            'labels': torch.tensor([[30, 40, 80, 120, 2]], dtype=torch.float32),
            'image_path': 'image2.jpg',
            'image_id': 1,
            'original_size': torch.tensor([640, 480], dtype=torch.float32),
            'scale_factor': torch.tensor([0.4, 0.533], dtype=torch.float32)
        },
        {
            'image': torch.randn(3, 256, 256),
            'labels': torch.zeros((0, 5), dtype=torch.float32),  # No labels
            'image_path': 'image3.jpg',
            'image_id': 2,
            'original_size': torch.tensor([640, 480], dtype=torch.float32),
            'scale_factor': torch.tensor([0.4, 0.533], dtype=torch.float32)
        }
    ]
    
    # Test collate
    collated = collate_fn(batch)
    
    print(f"Collated image shape: {collated['image'].shape}")
    print(f"Collated labels shape: {collated['labels'].shape}")
    print(f"Label mask shape: {collated['label_mask'].shape}")
    print(f"Label mask:\n{collated['label_mask']}")
    print(f"Batch size: {collated['batch_size']}")
    
    # Test CustomCollate
    custom_collate = CustomCollate(max_labels=5, pad_value=-1.0)
    custom_collated = custom_collate(batch)
    
    print(f"\nCustom collated labels shape: {custom_collated['labels'].shape}")
    print(f"Custom label mask:\n{custom_collated['label_mask']}")
    
    print("Collate function test completed!")
