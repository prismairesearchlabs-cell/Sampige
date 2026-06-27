"""
Feature Hooks Implementation
For extracting intermediate features from student backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
from collections import OrderedDict

from ...config import config


class FeatureHooks:
    """
    Feature hooks for extracting intermediate features from neural networks
    
    Args:
        model: The model to extract features from
        feature_dims: Dictionary of expected feature dimensions
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_dims: Dict[str, int] = config.model.STUDENT_FEATURE_DIMS
    ):
        self.model = model
        self.feature_dims = feature_dims
        self.hooks = OrderedDict()
        self.features = OrderedDict()
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on the model"""
        # Clear existing hooks
        self._clear_hooks()
        
        # Register hooks based on model type
        if hasattr(self.model, 'backbone'):
            self._register_yolo_hooks()
        else:
            self._register_generic_hooks()
    
    def _register_yolo_hooks(self):
        """Register hooks for YOLO-style backbone"""
        backbone = self.model.backbone
        
        # Try to find specific layers
        if hasattr(backbone, 'stages'):
            # YOLOv8 backbone
            for i, stage in enumerate(backbone.stages):
                scale_name = f"P{i+3}"
                if scale_name in self.feature_dims:
                    self._register_hook(stage, scale_name)
        
        elif hasattr(backbone, 'backbone'):
            # Nested backbone
            inner_backbone = backbone.backbone
            if hasattr(inner_backbone, 'stages'):
                for i, stage in enumerate(inner_backbone.stages):
                    scale_name = f"P{i+3}"
                    if scale_name in self.feature_dims:
                        self._register_hook(stage, scale_name)
    
    def _register_generic_hooks(self):
        """Register hooks for generic models"""
        # Try to find layers by name
        for name, module in self.model.named_modules():
            for scale_name in self.feature_dims.keys():
                if scale_name.lower() in name.lower():
                    self._register_hook(module, scale_name)
                    break
        
        # If no hooks registered, try to find conv layers
        if not self.hooks:
            conv_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    scale_name = f"P{conv_count + 3}"
                    if scale_name in self.feature_dims:
                        self._register_hook(module, scale_name)
                        conv_count += 1
    
    def _register_hook(self, module: nn.Module, name: str):
        """Register a hook on a specific module"""
        def hook_fn(module, input, output):
            self.features[name] = output
        
        handle = module.register_forward_hook(hook_fn)
        self.hooks[name] = handle
    
    def _clear_hooks(self):
        """Clear all registered hooks"""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
        self.features.clear()
    
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features by running forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of extracted features
        """
        # Clear previous features
        self.features.clear()
        
        # Run forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Return features
        return self.features.copy()
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get extracted features"""
        return self.features.copy()
    
    def remove_hooks(self):
        """Remove all hooks"""
        self._clear_hooks()


class HookManager:
    """
    Advanced hook manager for extracting features at specific layers
    
    Provides more control over feature extraction
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = OrderedDict()
        self.features = OrderedDict()
        self.hook_handles = []
    
    def register_hook(
        self,
        layer_name: str,
        hook_fn: Optional[Callable] = None,
        use_output: bool = True
    ) -> Callable:
        """
        Register a hook on a specific layer
        
        Args:
            layer_name: Name of the layer to hook
            hook_fn: Custom hook function
            use_output: Whether to use output or input
        
        Returns:
            Hook handle
        """
        # Find the layer
        layer = self._find_layer(layer_name)
        if layer is None:
            warnings.warn(f"Layer {layer_name} not found")
            return None
        
        # Create hook function
        if hook_fn is None:
            def default_hook(module, input, output):
                if use_output:
                    self.features[layer_name] = output
                else:
                    self.features[layer_name] = input[0] if isinstance(input, tuple) else input
        else:
            def custom_hook(module, input, output):
                hook_fn(module, input, output)
        
        # Register hook
        handle = layer.register_forward_hook(default_hook if hook_fn is None else custom_hook)
        self.hook_handles.append(handle)
        
        return handle
    
    def _find_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Find a layer by name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None
    
    def register_by_type(
        self,
        layer_type: type,
        prefix: str = "",
        max_count: int = 10
    ) -> List[str]:
        """
        Register hooks on all layers of a specific type
        
        Args:
            layer_type: Type of layer to hook (e.g., nn.Conv2d)
            prefix: Prefix for hook names
            max_count: Maximum number of hooks to register
        
        Returns:
            List of registered hook names
        """
        registered = []
        count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_type):
                hook_name = f"{prefix}{name}" if prefix else name
                self.register_hook(hook_name)
                registered.append(hook_name)
                count += 1
                
                if count >= max_count:
                    break
        
        return registered
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features by running forward pass
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of extracted features
        """
        self.features.clear()
        
        with torch.no_grad():
            _ = self.model(x)
        
        return self.features.copy()
    
    def clear_hooks(self):
        """Clear all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.features.clear()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.clear_hooks()


class MultiScaleFeatureExtractor:
    """
    Extract multi-scale features from a model
    
    Specifically designed for FPN-style architectures
    """
    
    def __init__(
        self,
        model: nn.Module,
        scales: List[str] = ["P3", "P4", "P5"]
    ):
        self.model = model
        self.scales = scales
        self.hook_manager = HookManager(model)
        
        # Register hooks for each scale
        for scale in scales:
            self.hook_manager.register_hook(scale)
    
    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features
        
        Args:
            x: Input tensor
        
        Returns:
            Dictionary of multi-scale features
        """
        features = self.hook_manager.extract_features(x)
        
        # Ensure all scales are present
        result = {}
        for scale in self.scales:
            if scale in features:
                result[scale] = features[scale]
            else:
                # Try to find similar scale
                for key in features.keys():
                    if scale.lower() in key.lower():
                        result[scale] = features[key]
                        break
        
        return result
    
    def clear(self):
        """Clear hooks"""
        self.hook_manager.clear_hooks()


class LayerOutputHook:
    """
    Simple hook to capture layer outputs
    """
    
    def __init__(self):
        self.outputs = {}
        self.hooks = []
    
    def add_hook(self, layer: nn.Module, name: str):
        """Add hook to a layer"""
        def hook_fn(module, input, output):
            self.outputs[name] = output
        
        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def clear(self):
        """Clear all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.outputs.clear()


if __name__ == "__main__":
    # Test feature hooks
    print("Testing Feature Hooks...")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            return x
    
    model = SimpleModel()
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Test FeatureHooks
    feature_dims = {"P3": 64, "P4": 128, "P5": 256}
    hooks = FeatureHooks(model, feature_dims)
    
    # Manually register hooks for testing
    hooks._register_hook(model.conv1, "P3")
    hooks._register_hook(model.conv2, "P4")
    hooks._register_hook(model.conv3, "P5")
    
    features = hooks(dummy_input)
    
    print(f"Extracted features: {list(features.keys())}")
    for key, value in features.items():
        print(f"  {key}: {value.shape}")
    
    # Test HookManager
    print("\nTesting HookManager...")
    hook_manager = HookManager(model)
    hook_manager.register_hook("conv1", use_output=True)
    hook_manager.register_hook("conv2", use_output=True)
    hook_manager.register_hook("conv3", use_output=True)
    
    extracted_features = hook_manager.extract_features(dummy_input)
    
    print(f"HookManager features: {list(extracted_features.keys())}")
    for key, value in extracted_features.items():
        print(f"  {key}: {value.shape}")
    
    # Clean up
    hooks.remove_hooks()
    hook_manager.clear_hooks()
    
    print("Feature hooks test completed!")
