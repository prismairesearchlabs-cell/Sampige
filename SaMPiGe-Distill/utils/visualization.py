"""
Visualization Utilities
For visualizing predictions, feature maps, and training progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import os
from PIL import Image, ImageDraw, ImageFont
import warnings

from .config import config


class Visualizer:
    """
    Base class for visualization
    """
    
    def __init__(self):
        self.colors = self._generate_colors()
        self.class_names = []
    
    def _generate_colors(self, n: int = 100) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(n):
            # Use HSV color space for better distinction
            hue = i / n
            saturation = 0.7
            value = 0.9
            
            # Convert HSV to RGB
            rgb = hsv_to_rgb([hue, saturation, value])
            colors.append((int(rb * 255) for rb in rgb))
        
        return colors
    
    def set_class_names(self, class_names: List[str]):
        """Set class names for visualization"""
        self.class_names = class_names


class DetectionVisualizer(Visualizer):
    """
    Visualizer for object detection results
    
    Args:
        class_names: List of class names
        colors: Custom colors for classes
        line_width: Width of bounding box lines
        font_size: Font size for labels
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        line_width: int = 2,
        font_size: int = 12
    ):
        super().__init__()
        
        if class_names:
            self.class_names = class_names
        if colors:
            self.colors = colors
        
        self.line_width = line_width
        self.font_size = font_size
        
        # Try to load a better font
        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except:
            self.font = ImageFont.load_default()
    
    def visualize(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
        confidence_threshold: float = 0.25,
        show_scores: bool = True,
        show_targets: bool = True,
        return_image: bool = True,
        save_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Visualize detection results on an image
        
        Args:
            image: Input image (H, W, 3) or (C, H, W)
            predictions: Predicted boxes (N, 5) in xyxy format with class
            targets: Ground truth boxes (M, 5) in xyxy format with class
            scores: Confidence scores for predictions
            confidence_threshold: Minimum confidence to show
            show_scores: Whether to show confidence scores
            show_targets: Whether to show ground truth boxes
            return_image: Whether to return the image
            save_path: Path to save the image
        
        Returns:
            Visualized image (if return_image=True)
        """
        # Convert image to PIL
        if isinstance(image, torch.Tensor):
            image = self._tensor_to_numpy(image)
        elif isinstance(image, np.ndarray):
            if image.shape[0] == 3:  # CHW format
                image = image.transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
            image = Image.fromarray(image)
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Draw ground truth boxes
        if show_targets and targets is not None:
            self._draw_boxes(
                draw, targets, 
                color=(0, 255, 0),  # Green
                label_prefix="GT: ",
                line_width=self.line_width
            )
        
        # Draw predictions
        if predictions is not None:
            # Convert to numpy if needed
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            
            # Filter by confidence
            if scores is not None:
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                
                # Combine predictions with scores
                if predictions.shape[1] == 5:  # xyxy + class
                    # Add scores as last column
                    predictions = np.column_stack([predictions, scores])
                
                # Filter
                mask = predictions[:, -1] >= confidence_threshold
                predictions = predictions[mask]
            
            # Draw filtered predictions
            self._draw_boxes(
                draw, predictions,
                color=None,  # Use class colors
                label_prefix="",
                show_scores=show_scores,
                line_width=self.line_width
            )
        
        # Save or return
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
        
        if return_image:
            return image
        
        return None
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        # Denormalize
        mean = np.array(config.data.MEAN)
        std = np.array(config.data.STD)
        
        tensor = tensor.detach().cpu()
        if tensor.shape[0] == 3:  # CHW format
            tensor = tensor.permute(1, 2, 0)
        
        # Convert to numpy
        image = tensor.numpy()
        
        # Denormalize
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def _draw_boxes(
        self,
        draw: ImageDraw.Draw,
        boxes: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        label_prefix: str = "",
        show_scores: bool = True,
        line_width: int = 2
    ):
        """Draw bounding boxes on image"""
        for i, box in enumerate(boxes):
            # Parse box
            if box.shape[0] == 5:  # xyxy + class
                x1, y1, x2, y2, class_id = box[:5]
                score = 1.0
            elif box.shape[0] == 6:  # xyxy + class + score
                x1, y1, x2, y2, class_id, score = box[:6]
            else:
                continue
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(class_id)
            
            # Get color
            if color is None:
                color = self.colors[class_id % len(self.colors)]
            
            # Draw rectangle
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=line_width
            )
            
            # Draw label
            label = label_prefix
            if class_id < len(self.class_names):
                label += self.class_names[class_id]
            else:
                label += f"class_{class_id}"
            
            if show_scores and box.shape[0] >= 6:
                label += f" {score:.2f}"
            
            # Draw text background
            text_width, text_height = draw.textsize(label, font=self.font)
            draw.rectangle(
                [(x1, y1 - text_height - 5), (x1 + text_width + 5, y1)],
                fill=color
            )
            
            # Draw text
            draw.text(
                (x1 + 2, y1 - text_height - 3),
                label,
                fill=(255, 255, 255),
                font=self.font
            )
    
    def visualize_batch(
        self,
        images: List[Union[np.ndarray, torch.Tensor, Image.Image]],
        predictions: Optional[List[np.ndarray]] = None,
        targets: Optional[List[np.ndarray]] = None,
        num_images: int = 4,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize a batch of images with predictions
        
        Args:
            images: List of images
            predictions: List of predictions for each image
            targets: List of targets for each image
            num_images: Number of images to visualize
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        num_images = min(num_images, len(images))
        
        fig, axes = plt.subplots(1, num_images, figsize=figsize)
        if num_images == 1:
            axes = [axes]
        
        for i in range(num_images):
            ax = axes[i]
            
            # Visualize single image
            image = images[i]
            pred = predictions[i] if predictions else None
            target = targets[i] if targets else None
            
            # Convert to PIL if needed
            if isinstance(image, (np.ndarray, torch.Tensor)):
                if isinstance(image, torch.Tensor):
                    image = self._tensor_to_numpy(image)
                if isinstance(image, np.ndarray):
                    if image.shape[0] == 3:
                        image = image.transpose(1, 2, 0)
                    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
                    image = Image.fromarray(image)
            
            # Draw on image
            self._draw_on_axes(ax, image, pred, target)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def _draw_on_axes(
        self,
        ax: plt.Axes,
        image: Image.Image,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None
    ):
        """Draw image and boxes on matplotlib axes"""
        # Display image
        ax.imshow(image)
        
        # Draw boxes
        if predictions is not None:
            self._draw_boxes_matplotlib(ax, predictions, color=None, linestyle='-')
        
        if targets is not None:
            self._draw_boxes_matplotlib(ax, targets, color='g', linestyle='--')
    
    def _draw_boxes_matplotlib(
        self,
        ax: plt.Axes,
        boxes: np.ndarray,
        color: Optional[str] = None,
        linestyle: str = '-'
    ):
        """Draw boxes on matplotlib axes"""
        for i, box in enumerate(boxes):
            if box.shape[0] == 5:  # xyxy + class
                x1, y1, x2, y2, class_id = box[:5]
            elif box.shape[0] == 6:  # xyxy + class + score
                x1, y1, x2, y2, class_id, _ = box[:6]
            else:
                continue
            
            class_id = int(class_id)
            
            # Get color
            if color is None:
                rgb = [c / 255 for c in self.colors[class_id % len(self.colors)]]
            else:
                rgb = plt.get_color(color)
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=self.line_width,
                edgecolor=rgb,
                facecolor='none',
                linestyle=linestyle
            )
            ax.add_patch(rect)
            
            # Add label
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"{class_id}"
            ax.text(
                x1, y1 - 2,
                label,
                color=rgb,
                fontsize=self.font_size,
                bbox=dict(facecolor=rgb, alpha=0.5, edgecolor='none')
            )


class FeatureMapVisualizer(Visualizer):
    """
    Visualizer for feature maps
    """
    
    def __init__(self):
        super().__init__()
    
    def visualize_feature_maps(
        self,
        feature_maps: Dict[str, torch.Tensor],
        num_maps: int = 16,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize feature maps from different scales
        
        Args:
            feature_maps: Dictionary of feature maps (scale -> tensor)
            num_maps: Number of feature maps to visualize per scale
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        num_scales = len(feature_maps)
        
        # Create subplots
        fig, axes = plt.subplots(
            num_scales, num_maps, 
            figsize=figsize,
            squeeze=False
        )
        
        for i, (scale, features) in enumerate(feature_maps.items()):
            # Convert to numpy
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu().numpy()
            
            # Normalize
            features = self._normalize_features(features)
            
            # Select first num_maps channels
            num_channels = min(num_maps, features.shape[1])
            
            for j in range(num_channels):
                ax = axes[i, j]
                
                # Display feature map
                ax.imshow(features[0, j], cmap='viridis')
                ax.set_title(f"{scale}-{j}")
                ax.axis('off')
            
            # Hide unused subplots
            for j in range(num_channels, num_maps):
                axes[i, j].axis('off')
        
        # Hide unused rows
        for i in range(num_scales, len(axes)):
            for j in range(num_maps):
                axes[i, j].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature maps for visualization"""
        # Normalize each channel independently
        for i in range(features.shape[1]):
            channel = features[:, i]
            channel_min = channel.min()
            channel_max = channel.max()
            
            if channel_max - channel_min > 0:
                features[:, i] = (channel - channel_min) / (channel_max - channel_min)
        
        return features
    
    def visualize_attention_maps(
        self,
        attention_maps: torch.Tensor,
        image: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_maps: int = 8,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualize attention maps
        
        Args:
            attention_maps: Attention maps (B, N, H, W) or (N, H, W)
            image: Optional background image
            num_maps: Number of attention maps to visualize
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        if isinstance(attention_maps, torch.Tensor):
            attention_maps = attention_maps.detach().cpu().numpy()
        
        # Handle batch dimension
        if attention_maps.ndim == 4:
            attention_maps = attention_maps[0]  # Take first in batch
        
        # Normalize
        attention_maps = self._normalize_features(attention_maps)
        
        # Create subplots
        num_maps = min(num_maps, attention_maps.shape[0])
        cols = min(4, num_maps)
        rows = int(np.ceil(num_maps / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if num_maps == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        for i in range(num_maps):
            ax = axes[i]
            
            # Display attention map
            if image is not None:
                # Convert image to numpy
                if isinstance(image, torch.Tensor):
                    image_np = self._tensor_to_numpy(image)
                else:
                    image_np = image
                
                # Resize attention map to match image
                attention_resized = cv2.resize(
                    attention_maps[i],
                    (image_np.shape[1], image_np.shape[0])
                )
                
                # Overlay on image
                overlay = self._overlay_attention(image_np, attention_resized)
                ax.imshow(overlay)
            else:
                ax.imshow(attention_maps[i], cmap='hot')
            
            ax.set_title(f"Attention {i}")
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_maps, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _overlay_attention(
        self,
        image: np.ndarray,
        attention: np.ndarray
    ) -> np.ndarray:
        """Overlay attention map on image"""
        # Normalize image
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Resize attention to match image
        if attention.shape != image.shape[:2]:
            attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return overlay


def plot_predictions(
    image: Union[np.ndarray, torch.Tensor, Image.Image],
    predictions: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
    confidence_threshold: float = 0.25,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predictions on an image
    
    Args:
        image: Input image
        predictions: Predicted boxes (N, 5) or (N, 6) in xyxy format
        class_names: List of class names
        confidence_threshold: Minimum confidence to show
        save_path: Path to save the image
        show: Whether to show the plot
    """
    visualizer = DetectionVisualizer(class_names)
    
    # Convert predictions to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Filter by confidence if scores are included
    if predictions.shape[1] >= 6:
        mask = predictions[:, 5] >= confidence_threshold
        predictions = predictions[mask]
    
    # Visualize
    fig = visualizer.visualize(
        image, predictions, return_image=False
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def plot_feature_maps(
    feature_maps: Dict[str, torch.Tensor],
    num_maps: int = 16,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot feature maps from different scales
    
    Args:
        feature_maps: Dictionary of feature maps
        num_maps: Number of maps to show per scale
        save_path: Path to save the figure
        show: Whether to show the plot
    """
    visualizer = FeatureMapVisualizer()
    fig = visualizer.visualize_feature_maps(feature_maps, num_maps)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig


def create_visualization_callback(
    datamodule,
    class_names: List[str],
    num_images: int = 4,
    interval: int = 5,
    output_dir: str = config.path.OUTPUT_DIR
) -> callable:
    """
    Create a callback for visualizing validation results
    
    Args:
        datamodule: DataModule for getting class names
        class_names: List of class names
        num_images: Number of images to visualize
        interval: Visualize every N validation steps
        output_dir: Directory to save visualizations
    
    Returns:
        Callback function for PyTorch Lightning
    """
    visualizer = DetectionVisualizer(class_names)
    step_count = 0
    
    def visualization_callback(pl_module, batch, batch_idx):
        nonlocal step_count
        
        step_count += 1
        
        # Only visualize at intervals
        if step_count % interval != 0:
            return
        
        # Get predictions
        with torch.no_grad():
            images = batch['image']
            
            # Get model predictions
            if hasattr(pl_module, 'student'):
                outputs = pl_module.student(images)
                predictions = outputs.get('detections', None)
            else:
                predictions = pl_module(images)
            
            # Convert predictions to bounding box format
            if predictions is not None:
                # This is simplified - actual conversion depends on model output format
                predictions = [self._convert_to_bbox_format(pred) for pred in predictions]
            
            # Get targets
            targets = batch.get('labels', None)
            if targets is not None:
                targets = [self._convert_to_bbox_format(target) for target in targets]
        
        # Visualize
        fig = visualizer.visualize_batch(
            images=images,
            predictions=predictions,
            targets=targets,
            num_images=num_images
        )
        
        # Save
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"val_step_{step_count}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
    
    def _convert_to_bbox_format(tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to bbox format"""
        if tensor is None:
            return np.zeros((0, 5))
        
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Simplified conversion - actual implementation depends on model
        if tensor.ndim == 2 and tensor.shape[1] >= 5:
            return tensor
        elif tensor.ndim == 3:
            # (H, W, num_anchors * (5 + num_classes))
            # This would need proper decoding
            return np.zeros((0, 5))
        else:
            return np.zeros((0, 5))
    
    return visualization_callback


if __name__ == "__main__":
    # Test visualization
    print("Testing Visualization...")
    
    # Create dummy data
    class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']
    
    # Create dummy image
    dummy_image = np.random.rand(256, 256, 3) * 255
    dummy_image = dummy_image.astype(np.uint8)
    
    # Create dummy predictions
    dummy_preds = np.array([
        [50, 50, 150, 150, 0, 0.95],  # Car with high confidence
        [200, 200, 250, 250, 1, 0.85],  # Van with medium confidence
        [100, 100, 120, 120, 2, 0.30]   # Truck with low confidence (will be filtered)
    ])
    
    # Create dummy targets
    dummy_targets = np.array([
        [55, 55, 155, 155, 0],  # Car
        [205, 205, 255, 255, 1]   # Van
    ])
    
    # Test DetectionVisualizer
    print("\nTesting DetectionVisualizer...")
    visualizer = DetectionVisualizer(class_names)
    
    # Visualize single image
    fig = visualizer.visualize(
        dummy_image,
        predictions=dummy_preds,
        targets=dummy_targets,
        confidence_threshold=0.5,
        show_scores=True
    )
    
    print("Visualization created successfully!")
    
    # Test FeatureMapVisualizer
    print("\nTesting FeatureMapVisualizer...")
    
    # Create dummy feature maps
    dummy_features = {
        "P3": torch.randn(1, 16, 64, 64),
        "P4": torch.randn(1, 32, 32, 32),
        "P5": torch.randn(1, 64, 16, 16)
    }
    
    feature_visualizer = FeatureMapVisualizer()
    fig = feature_visualizer.visualize_feature_maps(dummy_features, num_maps=8)
    
    print("Feature map visualization created successfully!")
    
    print("Visualization test completed!")
