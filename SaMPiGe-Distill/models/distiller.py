"""
Knowledge Distiller Implementation
Main module that combines all distillation components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import warnings

from config import config
from .teacher import DINOv2Teacher, TeacherWrapper
from .student import YOLOStudent, StudentBackbone
from .projection import ProjectionHead, FeatureProjection
from .prototype import PrototypeModule, SemanticPrototypeLoss
from .hooks import FeatureHooks


class KnowledgeDistiller(nn.Module):
    """
    Knowledge Distiller for multi-task learning
    
    Combines:
    - Feature distillation (global embeddings)
    - Patch distillation (local features)
    - Attention distillation (spatial information)
    - Relation distillation (feature relationships)
    - Prototype distillation (semantic concepts)
    - Detection loss (object detection)
    
    Args:
        teacher: Teacher model (DINOv2)
        student: Student model (YOLO-based)
        loss_weights: Dictionary of loss weights
    """
    
    def __init__(
        self,
        teacher: Optional[DINOv2Teacher] = None,
        student: Optional[YOLOStudent] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        # Initialize teacher
        if teacher is None:
            self.teacher = TeacherWrapper()
        else:
            self.teacher = TeacherWrapper(teacher)
        
        # Initialize student
        if student is None:
            self.student = YOLOStudent()
        else:
            self.student = student
        
        # Loss weights
        if loss_weights is None:
            self.loss_weights = {
                'detection': config.training.DETECTION_WEIGHT,
                'feature': config.training.FEATURE_WEIGHT,
                'attention': config.training.ATTENTION_WEIGHT,
                'relation': config.training.RELATION_WEIGHT,
                'prototype': config.training.PROTOTYPE_WEIGHT,
                'patch': config.training.PATCH_WEIGHT,
                'consistency': config.training.CONSISTENCY_WEIGHT
            }
        else:
            self.loss_weights = loss_weights
        
        # Projection head for student features
        self.projection = FeatureProjection(
            feature_dims=config.model.STUDENT_FEATURE_DIMS,
            output_dim=config.model.TEACHER_EMBED_DIM
        )
        
        # Prototype module
        self.prototype_module = PrototypeModule(
            num_prototypes=config.model.NUM_PROTOTYPES,
            prototype_dim=config.model.TEACHER_EMBED_DIM
        )
        
        # Loss functions
        self._init_loss_functions()
        
        print("Knowledge Distiller initialized")
        print(f"Loss weights: {self.loss_weights}")
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        # Detection loss (will be set by training module)
        self.detection_loss_fn = None
        
        # Feature distillation loss
        self.feature_loss_fn = nn.MSELoss()
        
        # Attention distillation loss
        self.attention_loss_fn = nn.MSELoss()
        
        # Relation distillation loss
        self.relation_loss_fn = nn.MSELoss()
        
        # Patch distillation loss
        self.patch_loss_fn = nn.MSELoss()
        
        # Prototype loss
        self.prototype_loss_fn = SemanticPrototypeLoss(
            num_prototypes=config.model.NUM_PROTOTYPES,
            prototype_dim=config.model.TEACHER_EMBED_DIM,
            loss_type="mse"
        )
        
        # Consistency loss
        self.consistency_loss_fn = nn.MSELoss()
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through distiller
        
        Args:
            x: Input tensor (B, C, H, W)
            labels: Ground truth labels (optional)
        
        Returns:
            Dictionary containing all outputs and losses
        """
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
        
        # Student forward pass
        student_outputs = self.student(x)
        
        # Compute losses
        losses = self.compute_losses(
            teacher_outputs=teacher_outputs,
            student_outputs=student_outputs,
            labels=labels
        )
        
        return {
            **teacher_outputs,
            **student_outputs,
            **losses
        }
    
    def compute_losses(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all distillation and detection losses
        
        Args:
            teacher_outputs: Teacher model outputs
            student_outputs: Student model outputs
            labels: Ground truth labels
        
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Feature distillation loss
        losses['feature_loss'] = self._compute_feature_loss(
            teacher_outputs, student_outputs
        )
        
        # Patch distillation loss
        losses['patch_loss'] = self._compute_patch_loss(
            teacher_outputs, student_outputs
        )
        
        # Attention distillation loss
        losses['attention_loss'] = self._compute_attention_loss(
            teacher_outputs, student_outputs
        )
        
        # Relation distillation loss
        losses['relation_loss'] = self._compute_relation_loss(
            teacher_outputs, student_outputs
        )
        
        # Prototype distillation loss
        losses['prototype_loss'] = self._compute_prototype_loss(
            teacher_outputs, student_outputs
        )
        
        # Detection loss
        if labels is not None and self.detection_loss_fn is not None:
            losses['detection_loss'] = self._compute_detection_loss(
                student_outputs, labels
            )
        else:
            losses['detection_loss'] = torch.tensor(0.0, device=x.device)
        
        # Consistency loss (requires augmented input)
        losses['consistency_loss'] = torch.tensor(0.0, device=x.device)
        
        # Total loss
        losses['total_loss'] = self._compute_total_loss(losses)
        
        return losses
    
    def _compute_feature_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute feature distillation loss (global embeddings)"""
        # Get teacher CLS token
        teacher_cls = teacher_outputs.get('cls_token', teacher_outputs.get('cls_token_norm'))
        
        # Get student global embedding
        student_embedding = student_outputs.get('global_embedding')
        
        if teacher_cls is None or student_embedding is None:
            return torch.tensor(0.0, device=teacher_cls.device if teacher_cls is not None else student_embedding.device)
        
        # Project student embedding to teacher space
        # For now, assume student embedding is already in correct space
        # In practice, we'd use the projection head
        
        # Normalize both
        teacher_cls_norm = F.normalize(teacher_cls, p=2, dim=-1)
        student_embedding_norm = F.normalize(student_embedding, p=2, dim=-1)
        
        # MSE loss between normalized embeddings
        loss = self.feature_loss_fn(teacher_cls_norm, student_embedding_norm)
        
        return loss
    
    def _compute_patch_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute patch-level distillation loss"""
        # Get teacher patch tokens
        teacher_patches = teacher_outputs.get('patch_tokens')
        
        # Get student features
        student_features = student_outputs.get('features', {})
        
        if teacher_patches is None or not student_features:
            return torch.tensor(0.0, device=teacher_patches.device if teacher_patches is not None else list(student_features.values())[0].device)
        
        # For simplicity, use P5 features (highest level)
        student_p5 = student_features.get('P5', list(student_features.values())[0])
        
        # Reshape teacher patches to match student spatial dimensions
        B, N, D = teacher_patches.shape
        
        # Project student features to teacher dimension
        projected_student = self.projection({"P5": student_p5})["P5"]
        
        # Reshape for comparison
        # This is simplified - in practice, we'd need proper spatial alignment
        student_flat = projected_student.view(B, -1, D)
        
        # Take first N patches from student
        if student_flat.shape[1] > N:
            student_flat = student_flat[:, :N]
        elif student_flat.shape[1] < N:
            # Pad with zeros
            padding = torch.zeros(
                (B, N - student_flat.shape[1], D),
                device=student_flat.device,
                dtype=student_flat.dtype
            )
            student_flat = torch.cat([student_flat, padding], dim=1)
        
        # MSE loss
        loss = self.patch_loss_fn(teacher_patches, student_flat)
        
        return loss
    
    def _compute_attention_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention distillation loss"""
        # Get teacher attention maps (if available)
        teacher_attention = teacher_outputs.get('attention_features')
        
        # Get student features
        student_features = student_outputs.get('features', {})
        
        if teacher_attention is None or not student_features:
            return torch.tensor(0.0, device=teacher_attention.device if teacher_attention is not None else list(student_features.values())[0].device)
        
        # Compute student attention maps
        # This would require extracting attention from student model
        # For now, use a simplified approach
        
        # Use P5 features for attention computation
        student_p5 = student_features.get('P5', list(student_features.values())[0])
        
        # Compute self-attention for student
        B, C, H, W = student_p5.shape
        student_flat = student_p5.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        student_norm = F.normalize(student_flat, p=2, dim=-1)
        student_attention = torch.bmm(student_norm, student_norm.transpose(1, 2))
        
        # Reshape teacher attention if needed
        if teacher_attention.dim() == 4:
            teacher_attention = teacher_attention.view(B, -1, teacher_attention.shape[-1])
        
        # Ensure same shape
        if student_attention.shape != teacher_attention.shape:
            # Resize teacher attention to match student
            teacher_attention = F.interpolate(
                teacher_attention.view(B, 1, *teacher_attention.shape[1:]),
                size=student_attention.shape[1:],
                mode='bilinear'
            ).view_as(teacher_attention)
        
        # MSE loss
        loss = self.attention_loss_fn(student_attention, teacher_attention)
        
        return loss
    
    def _compute_relation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute relation distillation loss (Gram matrix)"""
        # Get teacher features
        teacher_features = teacher_outputs.get('features')
        
        # Get student features
        student_features = student_outputs.get('features', {})
        
        if teacher_features is None or not student_features:
            return torch.tensor(0.0, device=teacher_features.device if teacher_features is not None else list(student_features.values())[0].device)
        
        # Compute Gram matrices
        # Teacher Gram matrix
        teacher_gram = self._compute_gram_matrix(teacher_features)
        
        # Student Gram matrix (use P5 features)
        student_p5 = student_features.get('P5', list(student_features.values())[0])
        student_gram = self._compute_gram_matrix(student_p5)
        
        # Reshape for comparison
        if teacher_gram.shape != student_gram.shape:
            # Resize student Gram matrix
            student_gram = F.interpolate(
                student_gram.unsqueeze(1),
                size=teacher_gram.shape[1:],
                mode='bilinear'
            ).squeeze(1)
        
        # MSE loss
        loss = self.relation_loss_fn(teacher_gram, student_gram)
        
        return loss
    
    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix from features"""
        if features.dim() == 3:
            # (B, N, D) -> (B, D, D)
            gram = torch.bmm(features, features.transpose(1, 2))
            # Normalize
            gram = gram / (features.shape[1] ** 0.5)
        elif features.dim() == 4:
            # (B, C, H, W) -> (B, C, C)
            B, C, H, W = features.shape
            features_flat = features.view(B, C, -1)  # (B, C, H*W)
            gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
            # Normalize
            gram = gram / (H * W ** 0.5)
        else:
            # Fallback
            gram = torch.zeros(
                (features.shape[0], features.shape[1], features.shape[1]),
                device=features.device,
                dtype=features.dtype
            )
        
        return gram
    
    def _compute_prototype_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute prototype distillation loss"""
        # Get teacher patch tokens
        teacher_patches = teacher_outputs.get('patch_tokens')
        
        # Get student features
        student_features = student_outputs.get('features', {})
        
        if teacher_patches is None or not student_features:
            return torch.tensor(0.0, device=teacher_patches.device if teacher_patches is not None else list(student_features.values())[0].device)
        
        # Use P5 features for student
        student_p5 = student_features.get('P5', list(student_features.values())[0])
        
        # Project student features to teacher dimension
        projected_student = self.projection({"P5": student_p5})["P5"]
        
        # Reshape for prototype module
        B, C, H, W = projected_student.shape
        student_flat = projected_student.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        # Teacher patches: (B, N, D) -> (B, N, D)
        teacher_flat = teacher_patches
        
        # Compute prototype loss
        try:
            loss = self.prototype_loss_fn(student_flat, teacher_flat)
        except Exception as e:
            warnings.warn(f"Prototype loss computation failed: {e}")
            loss = torch.tensor(0.0, device=student_flat.device)
        
        return loss
    
    def _compute_detection_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute detection loss"""
        if self.detection_loss_fn is None:
            return torch.tensor(0.0, device=labels.device)
        
        detections = student_outputs.get('detections')
        
        if detections is None:
            return torch.tensor(0.0, device=labels.device)
        
        try:
            loss = self.detection_loss_fn(detections, labels)
        except Exception as e:
            warnings.warn(f"Detection loss computation failed: {e}")
            loss = torch.tensor(0.0, device=labels.device)
        
        return loss
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss"""
        total_loss = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        for loss_name, weight in self.loss_weights.items():
            if loss_name in losses:
                total_loss += weight * losses[loss_name]
        
        return total_loss
    
    def set_detection_loss_fn(self, loss_fn: nn.Module):
        """Set detection loss function"""
        self.detection_loss_fn = loss_fn
    
    def update_loss_weights(self, weights: Dict[str, float]):
        """Update loss weights"""
        for key, value in weights.items():
            if key in self.loss_weights:
                self.loss_weights[key] = value
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.loss_weights.copy()


class MultiTaskDistiller(nn.Module):
    """
    Multi-task distiller with dynamic loss weighting
    
    Extends KnowledgeDistiller with:
    - Dynamic loss weight scheduling
    - Consistency regularization
    - EMA updates
    """
    
    def __init__(
        self,
        teacher: Optional[DINOv2Teacher] = None,
        student: Optional[YOLOStudent] = None,
        scheduler: Optional[nn.Module] = None
    ):
        super().__init__()
        
        # Initialize base distiller
        super(MultiTaskDistiller, self).__init__()
        
        self.distiller = KnowledgeDistiller(teacher, student)
        self.scheduler = scheduler
        
        # EMA model
        self.ema_model = None
        self.ema_alpha = 0.99
        
        # Consistency augmentation
        self.consistency_aug = None
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic weighting
        
        Args:
            x: Input tensor
            labels: Ground truth labels
            epoch: Current epoch for scheduling
        
        Returns:
            Dictionary of outputs and losses
        """
        # Update loss weights based on scheduler
        if self.scheduler is not None:
            new_weights = self.scheduler(epoch)
            self.distiller.update_loss_weights(new_weights)
        
        # Forward pass
        outputs = self.distiller(x, labels)
        
        # Add consistency loss if augmentation is available
        if self.consistency_aug is not None:
            outputs['consistency_loss'] = self._compute_consistency_loss(x)
            outputs['total_loss'] = self.distiller._compute_total_loss(outputs)
        
        return outputs
    
    def _compute_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss using augmented inputs"""
        if self.consistency_aug is None:
            return torch.tensor(0.0, device=x.device)
        
        # Create augmented version
        x_aug = self.consistency_aug(x)
        
        # Get student outputs for both
        with torch.no_grad():
            outputs1 = self.distiller.student(x)
            outputs2 = self.distiller.student(x_aug)
        
        # Compare global embeddings
        emb1 = outputs1.get('global_embedding', outputs1.get('features', {}).get('P5'))
        emb2 = outputs2.get('global_embedding', outputs2.get('features', {}).get('P5'))
        
        if emb1 is None or emb2 is None:
            return torch.tensor(0.0, device=x.device)
        
        # MSE between embeddings
        loss = F.mse_loss(emb1, emb2)
        
        return loss
    
    def update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            # Initialize EMA model
            self.ema_model = self._create_ema_model()
            self._copy_weights(self.distiller.student, self.ema_model)
        else:
            # Update EMA
            self._update_ema_weights()
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of student model"""
        # Create a copy of the student model
        ema_model = type(self.distiller.student)()
        
        # Copy state dict
        ema_model.load_state_dict(self.distiller.student.state_dict())
        
        # Set to eval mode
        ema_model.eval()
        
        return ema_model
    
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target"""
        target.load_state_dict(source.state_dict())
    
    def _update_ema_weights(self):
        """Update EMA weights"""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.distiller.student.parameters()
            ):
                ema_param.data.mul_(self.ema_alpha).add_(
                    param.data, alpha=1 - self.ema_alpha
                )
    
    def get_ema_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from EMA model"""
        if self.ema_model is None:
            return self.distiller.student(x)
        
        with torch.no_grad():
            return self.ema_model(x)


if __name__ == "__main__":
    # Test knowledge distiller
    print("Testing Knowledge Distiller...")
    
    # Create dummy inputs
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    dummy_labels = torch.randn(batch_size, 10, 5)  # 10 bounding boxes per image
    
    # Create mock teacher (since DINOv2 might not be available)
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.cls_token = nn.Parameter(torch.randn(1, 768))
            self.patch_proj = nn.Linear(3 * 16 * 16, 768)
            
        def forward(self, x):
            B = x.shape[0]
            cls_token = self.cls_token.expand(B, -1)
            
            # Simplified patch extraction
            patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
            patches = patches.contiguous().view(B, 3, -1, 16, 16)
            patches = patches.permute(0, 2, 3, 4, 1).contiguous()
            patches = patches.view(B, -1, 3 * 16 * 16)
            
            patch_tokens = self.patch_proj(patches)
            features = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
            
            return {
                'cls_token': cls_token,
                'patch_tokens': patch_tokens,
                'features': features,
                'embed_dim': self.embed_dim
            }
    
    # Create mock student
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.detection_head = nn.Linear(256, 10 * 6)  # 10 boxes, 6 values each
            
        def forward(self, x):
            features = self.backbone(x)
            global_embedding = features.view(features.size(0), -1)
            
            # Expand for detection
            detections = self.detection_head(global_embedding)
            detections = detections.view(x.size(0), 10, 6)
            
            return {
                'features': {"P5": features},
                'global_embedding': global_embedding,
                'detections': detections
            }
    
    # Test with mock models
    print("\nTesting with mock models...")
    
    mock_teacher = MockTeacher()
    mock_student = MockStudent()
    
    distiller = KnowledgeDistiller(
        teacher=mock_teacher,
        student=mock_student
    )
    
    # Set a simple detection loss
    def simple_detection_loss(pred, target):
        return F.mse_loss(pred, target)
    
    distiller.set_detection_loss_fn(simple_detection_loss)
    
    # Forward pass
    outputs = distiller(dummy_input, dummy_labels)
    
    print(f"Outputs keys: {list(outputs.keys())}")
    print(f"Total loss: {outputs['total_loss'].item()}")
    print(f"Feature loss: {outputs['feature_loss'].item()}")
    print(f"Detection loss: {outputs['detection_loss'].item()}")
    
    print("Knowledge Distiller test completed!")
