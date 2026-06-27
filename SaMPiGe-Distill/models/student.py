"""
Student Network Implementation
Custom YOLO-based student backbone with feature hooks for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional, Tuple, List
import warnings

from ...config import config
from .hooks import FeatureHooks, HookManager


class StudentBackbone(nn.Module):
    """
    Custom student backbone based on YOLO architecture
    
    Provides multi-scale feature maps (P3, P4, P5) for object detection
    and a global embedding for knowledge distillation
    
    Args:
        backbone_type: Type of backbone ('yolov8s', 'yolov8n', 'yolov8m', 'custom')
        pretrained: Whether to load pretrained weights
        feature_dims: Dictionary of feature dimensions for each scale
    """
    
    def __init__(
        self,
        backbone_type: str = config.model.STUDENT_BACKBONE,
        pretrained: bool = config.model.STUDENT_PRETRAINED,
        feature_dims: Dict[str, int] = config.model.STUDENT_FEATURE_DIMS
    ):
        super().__init__()
        
        self.backbone_type = backbone_type
        self.pretrained = pretrained
        self.feature_dims = feature_dims
        
        # Initialize backbone
        self._init_backbone()
        
        # Feature hooks for extracting intermediate features
        self.hooks = FeatureHooks(self.backbone, feature_dims)
        
        # Global pooling for embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_proj = nn.Linear(
            sum(feature_dims.values()), 
            config.model.PROJECTION_DIM
        )
        
        print(f"Student Backbone initialized: {backbone_type}")
        print(f"Feature dimensions: {feature_dims}")
    
    def _init_backbone(self):
        """Initialize the backbone network"""
        if self.backbone_type.startswith('yolov8'):
            self._init_yolov8_backbone()
        else:
            self._init_custom_backbone()
    
    def _init_yolov8_backbone(self):
        """Initialize YOLOv8-based backbone"""
        try:
            from ultralytics import YOLO
            
            # Load YOLOv8 model
            model = YOLO(f"{self.backbone_type}.pt")
            self.backbone = model.model
            
            # Remove detection head if present
            if hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()
            
            # Update feature dimensions based on actual backbone
            if hasattr(self.backbone, 'backbone'):
                backbone = self.backbone.backbone
                if hasattr(backbone, 'stages'):
                    # Update feature dimensions
                    for i, stage in enumerate(backbone.stages):
                        out_channels = stage.conv2.conv.out_channels
                        scale_name = f"P{i+3}"
                        if scale_name in self.feature_dims:
                            self.feature_dims[scale_name] = out_channels
        
        except ImportError:
            warnings.warn("Ultralytics not available, using custom backbone")
            self._init_custom_backbone()
    
    def _init_custom_backbone(self):
        """Initialize custom backbone based on ResNet or similar"""
        # Use ResNet50 as base
        base_model = models.resnet50(pretrained=self.pretrained)
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Update feature dimensions
        self.feature_dims = {
            "P3": 512,  # layer2 output
            "P4": 1024, # layer3 output
            "P5": 2048  # layer4 output
        }
        
        # Add feature pyramid network
        self.fpn = FeaturePyramidNetwork(
            in_channels=[512, 1024, 2048],
            out_channels=256
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through student backbone
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Dictionary containing:
                - features: Dictionary of multi-scale features (P3, P4, P5)
                - global_embedding: Global embedding vector (B, embed_dim)
                - backbone_features: Raw backbone outputs
        """
        # Forward through backbone
        if hasattr(self.backbone, 'forward'):
            backbone_features = self.backbone(x)
        else:
            # Handle sequential backbone
            backbone_features = x
            for layer in self.backbone:
                backbone_features = layer(backbone_features)
        
        # Extract features using hooks
        features = self.hooks(backbone_features)
        
        # If using custom backbone with FPN
        if hasattr(self, 'fpn'):
            features = self.fpn(features)
        
        # Compute global embedding
        global_embedding = self._compute_global_embedding(features)
        
        return {
            'features': features,
            'global_embedding': global_embedding,
            'backbone_features': backbone_features
        }
    
    def _compute_global_embedding(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute global embedding from multi-scale features
        
        Args:
            features: Dictionary of multi-scale features
        
        Returns:
            Global embedding tensor (B, embed_dim)
        """
        # Concatenate features from different scales
        concatenated = []
        for scale in ["P3", "P4", "P5"]:
            if scale in features:
                # Global average pooling
                pooled = self.global_pool(features[scale])
                pooled = pooled.view(pooled.size(0), -1)  # Flatten
                concatenated.append(pooled)
        
        if concatenated:
            concatenated_features = torch.cat(concatenated, dim=1)
            global_embedding = self.embedding_proj(concatenated_features)
        else:
            # Fallback: use P5 features
            p5_features = features.get("P5", features[list(features.keys())[0]])
            pooled = self.global_pool(p5_features)
            pooled = pooled.view(pooled.size(0), -1)
            global_embedding = self.embedding_proj(pooled)
        
        return global_embedding
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps for detection head"""
        outputs = self.forward(x)
        return outputs['features']
    
    def get_global_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get global embedding for knowledge distillation"""
        outputs = self.forward(x)
        return outputs['global_embedding']


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )
        
        # Top-down connections
        self.top_down_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.top_down_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FPN
        
        Args:
            features: Dictionary of input features (P3, P4, P5)
        
        Returns:
            Dictionary of output features
        """
        # Get features in order (P5, P4, P3)
        p5 = features.get("P5", features.get("layer4", None))
        p4 = features.get("P4", features.get("layer3", None))
        p3 = features.get("P3", features.get("layer2", None))
        
        if p5 is None or p4 is None or p3 is None:
            return features
        
        # Lateral connections
        p5_lateral = self.lateral_convs[0](p5)
        p4_lateral = self.lateral_convs[1](p4)
        p3_lateral = self.lateral_convs[2](p3)
        
        # Top-down
        p4_out = self.top_down_convs[0](p5_lateral + self.upsample(p4_lateral))
        p3_out = self.top_down_convs[1](self.upsample(p4_out) + p3_lateral)
        
        return {
            "P3": p3_out,
            "P4": p4_out,
            "P5": p5_lateral
        }


class YOLOStudent(nn.Module):
    """
    Complete YOLO student model with detection head
    
    Combines student backbone with YOLO detection head
    """
    
    def __init__(
        self,
        backbone: Optional[StudentBackbone] = None,
        num_classes: int = config.model.NUM_CLASSES,
        detection_head: str = config.model.DETECTION_HEAD
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.detection_head = detection_head
        
        # Initialize backbone
        if backbone is None:
            self.backbone = StudentBackbone()
        else:
            self.backbone = backbone
        
        # Initialize detection head
        self._init_detection_head()
        
        print(f"YOLO Student initialized with {num_classes} classes")
    
    def _init_detection_head(self):
        """Initialize detection head"""
        if self.detection_head == "yolov8":
            self._init_yolov8_head()
        else:
            self._init_custom_head()
    
    def _init_yolov8_head(self):
        """Initialize YOLOv8-style detection head"""
        try:
            from ultralytics.nn.modules import Detect
            
            # Create detection head
            self.head = Detect(
                nc=self.num_classes,
                ch=[256, 256, 256],  # Input channels from FPN
                anchors=3,  # Number of anchors per scale
                stride=[8, 16, 32]  # Strides for each scale
            )
        
        except ImportError:
            warnings.warn("Ultralytics not available, using custom detection head")
            self._init_custom_head()
    
    def _init_custom_head(self):
        """Initialize custom detection head"""
        # Simple detection head for testing
        self.head = CustomDetectionHead(
            in_channels=256,
            num_classes=self.num_classes,
            num_anchors=3
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete student model
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Dictionary containing:
                - detections: Detection outputs
                - features: Multi-scale features
                - global_embedding: Global embedding
        """
        # Get backbone outputs
        backbone_outputs = self.backbone(x)
        features = backbone_outputs['features']
        global_embedding = backbone_outputs['global_embedding']
        
        # Get detection outputs
        if hasattr(self.head, 'forward'):
            # Prepare inputs for detection head
            head_inputs = [
                features.get("P3", torch.zeros_like(features["P5"])),
                features.get("P4", torch.zeros_like(features["P5"])),
                features.get("P5", torch.zeros_like(features["P5"]))
            ]
            detections = self.head(head_inputs)
        else:
            # Fallback
            detections = self.head(features["P5"])
        
        return {
            'detections': detections,
            'features': features,
            'global_embedding': global_embedding
        }
    
    def get_detections(self, x: torch.Tensor) -> torch.Tensor:
        """Get detection outputs"""
        outputs = self.forward(x)
        return outputs['detections']
    
    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get feature maps"""
        outputs = self.forward(x)
        return outputs['features']
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get global embedding"""
        outputs = self.forward(x)
        return outputs['global_embedding']


class CustomDetectionHead(nn.Module):
    """
    Custom detection head for testing
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 8,
        num_anchors: int = 3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)  # xywh
        )
        
        # Objectness head
        self.obj_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            Detection outputs (B, num_anchors * (5 + num_classes), H, W)
        """
        cls_output = self.cls_head(x)
        reg_output = self.reg_head(x)
        obj_output = self.obj_head(x)
        
        # Concatenate outputs
        output = torch.cat([reg_output, obj_output, cls_output], dim=1)
        
        return output


if __name__ == "__main__":
    # Test student model
    print("Testing Student Backbone...")
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 640, 640)
    
    # Test StudentBackbone
    backbone = StudentBackbone(
        backbone_type="custom",
        pretrained=False
    )
    
    outputs = backbone(dummy_input)
    
    print(f"Features keys: {list(outputs['features'].keys())}")
    for key, value in outputs['features'].items():
        print(f"  {key}: {value.shape}")
    print(f"Global embedding shape: {outputs['global_embedding'].shape}")
    
    # Test YOLOStudent
    print("\nTesting YOLO Student...")
    student = YOLOStudent(
        backbone=backbone,
        num_classes=8,
        detection_head="custom"
    )
    
    student_outputs = student(dummy_input)
    
    print(f"Detections shape: {student_outputs['detections'].shape}")
    print(f"Features keys: {list(student_outputs['features'].keys())}")
    print(f"Global embedding shape: {student_outputs['global_embedding'].shape}")
    
    print("Student model test completed!")
