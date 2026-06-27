"""
Projection Head Implementation
Projects student features to teacher's embedding space for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from ...config import config


class ProjectionHead(nn.Module):
    """
    Projection head for mapping student features to teacher's embedding space
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension (should match teacher)
        use_bn: Whether to use batch normalization
        activation: Activation function ('relu', 'leaky_relu', 'gelu', 'swish')
        num_layers: Number of projection layers
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = config.model.TEACHER_EMBED_DIM,
        use_bn: bool = config.model.PROJECTION_USE_BN,
        activation: str = config.model.PROJECTION_ACTIVATION,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bn = use_bn
        self.activation = activation
        self.num_layers = num_layers
        
        # Build projection layers
        self.projection = self._build_projection()
        
        print(f"Projection Head: {input_dim} -> {output_dim}, layers={num_layers}, activation={activation}")
    
    def _build_projection(self) -> nn.Sequential:
        """Build the projection network"""
        layers = []
        
        # Input dimension
        in_dim = self.input_dim
        
        # Hidden dimensions (geometric progression)
        hidden_dims = []
        for i in range(self.num_layers - 1):
            hidden_dim = int(in_dim * (2 ** (i + 1)))
            hidden_dims.append(hidden_dim)
        
        # Add layers
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=not self.use_bn))
            
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_dim))
            
            layers.append(self._get_activation())
            
            in_dim = out_dim
        
        # Final layer
        layers.append(nn.Conv2d(in_dim, self.output_dim, kernel_size=1, bias=True))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """Get activation function"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif self.activation == 'gelu':
            return nn.GELU()
        elif self.activation == 'swish':
            return nn.SiLU()  # Swish is alias for SiLU
        else:
            return nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head
        
        Args:
            x: Input feature tensor (B, C, H, W)
        
        Returns:
            Projected embedding tensor (B, output_dim, H, W)
        """
        return self.projection(x)


class FeatureProjection(nn.Module):
    """
    Feature projection for multi-scale features
    
    Projects features from multiple scales to a common embedding space
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int] = config.model.STUDENT_FEATURE_DIMS,
        output_dim: int = config.model.TEACHER_EMBED_DIM
    ):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Create projection head for each scale
        self.projections = nn.ModuleDict()
        for scale, dim in feature_dims.items():
            self.projections[scale] = ProjectionHead(
                input_dim=dim,
                output_dim=output_dim,
                num_layers=1  # Single layer for efficiency
            )
        
        print(f"Feature Projection initialized for scales: {list(feature_dims.keys())}")
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Project multi-scale features
        
        Args:
            features: Dictionary of multi-scale features
        
        Returns:
            Dictionary of projected features
        """
        projected = {}
        
        for scale, feature in features.items():
            if scale in self.projections:
                projected[scale] = self.projections[scale](feature)
            else:
                # Use closest scale
                for available_scale in self.projections.keys():
                    if available_scale in feature_dims:
                        projected[scale] = self.projections[available_scale](feature)
                        break
        
        return projected
    
    def get_global_projection(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get global projection by pooling and projecting
        
        Args:
            features: Dictionary of multi-scale features
        
        Returns:
            Global projected embedding (B, output_dim)
        """
        # Project each scale
        projected_features = self.forward(features)
        
        # Global average pooling for each scale
        pooled_features = []
        for scale, feature in projected_features.items():
            pooled = F.adaptive_avg_pool2d(feature, (1, 1))
            pooled = pooled.view(pooled.size(0), -1)  # (B, output_dim)
            pooled_features.append(pooled)
        
        # Average across scales
        if pooled_features:
            global_embedding = torch.stack(pooled_features, dim=1).mean(dim=1)
        else:
            global_embedding = torch.zeros(
                (features[list(features.keys())[0]].size(0), self.output_dim),
                device=features[list(features.keys())[0]].device
            )
        
        return global_embedding


class MultiHeadProjection(nn.Module):
    """
    Multi-head projection for different types of features
    
    Provides separate projections for:
    - Global features (CLS token)
    - Patch features
    - Spatial features
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = config.model.TEACHER_EMBED_DIM,
        num_heads: int = 3
    ):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Create multiple projection heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(
                ProjectionHead(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_layers=2
                )
            )
        
        print(f"MultiHead Projection: {num_heads} heads, {input_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all heads
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            List of projected tensors from each head
        """
        outputs = []
        for head in self.heads:
            outputs.append(head(x))
        
        return outputs


class AdaptiveProjection(nn.Module):
    """
    Adaptive projection that learns to weight different feature scales
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int] = config.model.STUDENT_FEATURE_DIMS,
        output_dim: int = config.model.TEACHER_EMBED_DIM
    ):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Projection for each scale
        self.projections = nn.ModuleDict()
        for scale, dim in feature_dims.items():
            self.projections[scale] = nn.Sequential(
                nn.Conv2d(dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU()
            )
        
        # Attention weights for each scale
        self.attention = nn.Sequential(
            nn.Linear(len(feature_dims), len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Adaptive projection with learned weighting
        
        Args:
            features: Dictionary of multi-scale features
        
        Returns:
            Global projected embedding (B, output_dim)
        """
        # Project each scale
        projected = {}
        for scale, feature in features.items():
            if scale in self.projections:
                projected[scale] = self.projections[scale](feature)
        
        # Compute attention weights
        B = list(projected.values())[0].size(0)
        
        # Get spatial features for attention
        spatial_features = []
        for scale, feature in projected.items():
            pooled = self.global_pool(feature).view(B, -1)
            spatial_features.append(pooled)
        
        spatial_features = torch.stack(spatial_features, dim=1)  # (B, num_scales, output_dim)
        
        # Compute attention weights
        attention_weights = self.attention(spatial_features.mean(dim=-1))  # (B, num_scales)
        
        # Weighted sum of projected features
        weighted_features = []
        for i, (scale, feature) in enumerate(projected.items()):
            weight = attention_weights[:, i:i+1, None, None]  # (B, 1, 1, 1)
            weighted = feature * weight
            weighted_features.append(weighted)
        
        # Sum weighted features
        combined = torch.stack(weighted_features, dim=1).sum(dim=1)
        
        # Global pooling
        global_embedding = self.global_pool(combined).view(B, -1)
        
        return global_embedding


if __name__ == "__main__":
    # Test projection heads
    print("Testing Projection Heads...")
    
    # Create dummy features
    batch_size = 2
    feature_dims = {"P3": 256, "P4": 512, "P5": 1024}
    
    dummy_features = {
        "P3": torch.randn(batch_size, 256, 32, 32),
        "P4": torch.randn(batch_size, 512, 16, 16),
        "P5": torch.randn(batch_size, 1024, 8, 8)
    }
    
    # Test ProjectionHead
    print("\nTesting ProjectionHead...")
    proj_head = ProjectionHead(input_dim=256, output_dim=768)
    output = proj_head(dummy_features["P3"])
    print(f"ProjectionHead output shape: {output.shape}")
    
    # Test FeatureProjection
    print("\nTesting FeatureProjection...")
    feature_proj = FeatureProjection(feature_dims, output_dim=768)
    projected_features = feature_proj(dummy_features)
    
    for scale, feature in projected_features.items():
        print(f"  {scale}: {feature.shape}")
    
    global_embedding = feature_proj.get_global_projection(dummy_features)
    print(f"Global embedding shape: {global_embedding.shape}")
    
    # Test MultiHeadProjection
    print("\nTesting MultiHeadProjection...")
    multi_head = MultiHeadProjection(input_dim=256, output_dim=768, num_heads=3)
    outputs = multi_head(dummy_features["P3"])
    
    for i, output in enumerate(outputs):
        print(f"  Head {i}: {output.shape}")
    
    # Test AdaptiveProjection
    print("\nTesting AdaptiveProjection...")
    adaptive_proj = AdaptiveProjection(feature_dims, output_dim=768)
    adaptive_embedding = adaptive_proj(dummy_features)
    print(f"Adaptive projection output shape: {adaptive_embedding.shape}")
    
    print("Projection heads test completed!")
