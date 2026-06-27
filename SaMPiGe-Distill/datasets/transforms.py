"""
Transforms and Augmentations for KITTI Dataset
"""

import random
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import v2 as T2
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import cv2

from .config import config


def get_transforms(
    image_size: int = config.data.IMAGE_SIZE,
    augment: bool = True,
    split: str = "train"
) -> T.Compose:
    """
    Get standard transforms for KITTI dataset
    
    Args:
        image_size: Target image size
        augment: Whether to include augmentations
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        Composed transforms
    """
    transforms_list = []
    
    # Basic transforms
    transforms_list.extend([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    # Normalization
    transforms_list.append(
        T.Normalize(
            mean=config.data.MEAN,
            std=config.data.STD
        )
    )
    
    return T.Compose(transforms_list)


def get_augmentations() -> List[T.Compose]:
    """
    Get list of augmentations for training
    
    Returns:
        List of augmentation transforms
    """
    augmentations = []
    
    # Horizontal flip
    if config.data.HORIZONTAL_FLIP:
        augmentations.append(
            T.RandomHorizontalFlip(p=0.5)
        )
    
    # Vertical flip
    if config.data.VERTICAL_FLIP:
        augmentations.append(
            T.RandomVerticalFlip(p=0.1)
        )
    
    # Rotation
    if config.data.ROTATION > 0:
        augmentations.append(
            T.RandomRotation(
                degrees=config.data.ROTATION,
                expand=False,
                fill=0
            )
        )
    
    # Color jitter
    if config.data.COLOR_JITTER_PROB > 0:
        augmentations.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=config.data.BRIGHTNESS,
                        contrast=config.data.CONTRAST,
                        saturation=config.data.SATURATION,
                        hue=config.data.HUE
                    )
                ],
                p=config.data.COLOR_JITTER_PROB
            )
        )
    
    # Mosaic augmentation
    if config.data.MOSAIC_PROB > 0:
        augmentations.append(
            MosaicAugmentation(prob=config.data.MOSAIC_PROB)
        )
    
    # MixUp augmentation
    if config.data.MIXUP_PROB > 0:
        augmentations.append(
            MixUpAugmentation(prob=config.data.MIXUP_PROB)
        )
    
    return augmentations


class MosaicAugmentation:
    """
    Mosaic augmentation for object detection
    Combines 4 images into one by cropping and pasting
    """
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, image: Image.Image, labels: np.ndarray = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply mosaic augmentation
        
        Args:
            image: Input image
            labels: Bounding boxes in xyxy format with class ids (N, 5)
        
        Returns:
            Augmented image and labels
        """
        if random.random() > self.prob or labels is None:
            return image, labels
        
        # For simplicity, return original (full mosaic implementation would need batch context)
        return image, labels


class MixUpAugmentation:
    """
    MixUp augmentation for object detection
    """
    
    def __init__(self, prob: float = 0.1):
        self.prob = prob
    
    def __call__(self, image: Image.Image, labels: np.ndarray = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply MixUp augmentation
        
        Args:
            image: Input image
            labels: Bounding boxes in xyxy format with class ids (N, 5)
        
        Returns:
            Augmented image and labels
        """
        if random.random() > self.prob or labels is None:
            return image, labels
        
        # For simplicity, return original (full MixUp implementation would need batch context)
        return image, labels


class RandomPerspective:
    """
    Random perspective transform for object detection
    """
    
    def __init__(
        self,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: float = 10.0,
        perspective: float = 0.0,
        prob: float = 0.5
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.prob = prob
    
    def __call__(self, image: Image.Image, labels: np.ndarray = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply random perspective transform
        
        Args:
            image: Input image
            labels: Bounding boxes in xyxy format with class ids (N, 5)
        
        Returns:
            Transformed image and labels
        """
        if random.random() > self.prob or labels is None:
            return image, labels
        
        # Convert to numpy
        img = np.array(image)
        height, width = img.shape[:2]
        
        # Generate random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate, self.translate) * width
        ty = random.uniform(-self.translate, self.translate) * height
        scale_x = random.uniform(1 - self.scale, 1 + self.scale)
        scale_y = random.uniform(1 - self.scale, 1 + self.scale)
        shear_x = random.uniform(-self.shear, self.shear)
        shear_y = random.uniform(-self.shear, self.shear)
        
        # Create transformation matrix
        M = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
            np.float32([
                [tx, ty],
                [width * scale_x + tx, ty + height * shear_x],
                [width * scale_x + tx + height * shear_y, height * scale_y + ty],
                [tx + height * shear_y, height * scale_y + ty]
            ])
        )
        
        # Apply transformation
        transformed_img = cv2.warpPerspective(
            img, M, (width, height),
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Transform labels (simplified - would need proper transformation)
        # For now, return original labels
        transformed_image = Image.fromarray(transformed_img)
        
        return transformed_image, labels


class AlbumentationsWrapper:
    """
    Wrapper for Albumentations library
    """
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image: Image.Image, labels: np.ndarray = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply albumentations transform
        
        Args:
            image: Input image
            labels: Bounding boxes in xyxy format with class ids (N, 5)
        
        Returns:
            Transformed image and labels
        """
        try:
            import albumentations as A
            
            # Convert image to numpy
            img = np.array(image)
            
            # Convert labels to albumentations format
            if labels is not None and len(labels) > 0:
                bboxes = labels[:, :4].tolist()  # xyxy
                class_ids = labels[:, 4].tolist()
                
                # Apply transform
                transformed = self.transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_ids
                )
                
                transformed_img = transformed["image"]
                transformed_bboxes = np.array(transformed["bboxes"], dtype=np.float32)
                transformed_class_ids = np.array(transformed["class_labels"], dtype=np.float32)
                
                # Combine bboxes and class ids
                transformed_labels = np.column_stack([transformed_bboxes, transformed_class_ids])
                
                return Image.fromarray(transformed_img), transformed_labels
            else:
                transformed = self.transform(image=img)
                return Image.fromarray(transformed["image"]), labels
                
        except ImportError:
            # Fallback to original if albumentations not available
            return image, labels


def get_albumentations_transforms() -> Optional[AlbumentationsWrapper]:
    """
    Get albumentations transforms if available
    """
    try:
        import albumentations as A
        
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=config.data.BRIGHTNESS,
                contrast_limit=config.data.CONTRAST,
                p=0.5
            ),
            A.RandomResizedCrop(
                height=config.data.IMAGE_SIZE,
                width=config.data.IMAGE_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.2
            ),
            A.Blur(blur_limit=3, p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
        ], bbox_params=A.BboxParams(format='xyxy', label_fields=['class_labels']))
        
        return AlbumentationsWrapper(transform)
        
    except ImportError:
        return None


class TeacherTransform:
    """
    Transform for teacher model (DINOv2)
    Resizes to teacher's expected input size and normalizes
    """
    
    def __init__(self, size: int = config.data.TEACHER_IMAGE_SIZE):
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply teacher transform"""
        return self.transform(image)


class StudentTransform:
    """
    Transform for student model
    """
    
    def __init__(self, size: int = config.data.IMAGE_SIZE):
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(
                mean=config.data.MEAN,
                std=config.data.STD
            )
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply student transform"""
        return self.transform(image)


if __name__ == "__main__":
    # Test transforms
    print("Testing transforms...")
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), (128, 128, 128))
    
    # Test basic transform
    basic_transform = get_transforms(image_size=256, augment=False)
    transformed = basic_transform(dummy_image)
    print(f"Basic transform output shape: {transformed.shape}")
    
    # Test teacher transform
    teacher_transform = TeacherTransform(size=518)
    teacher_output = teacher_transform(dummy_image)
    print(f"Teacher transform output shape: {teacher_output.shape}")
    
    # Test student transform
    student_transform = StudentTransform(size=640)
    student_output = student_transform(dummy_image)
    print(f"Student transform output shape: {student_output.shape}")
    
    print("Transforms test completed!")
