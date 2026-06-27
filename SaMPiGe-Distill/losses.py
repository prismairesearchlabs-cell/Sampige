"""
Loss Functions for SaMPiGe-Distill
Custom loss functions for detection and knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math

from config import config


class DetectionLoss(nn.Module):
    """
    Combined detection loss for object detection
    
    Combines:
    - Classification loss
    - Localization loss (box regression)
    - Objectness loss
    
    Args:
        num_classes: Number of classes
        box_loss: Type of box loss ('ciou', 'diou', 'siou', 'mse')
        cls_loss: Type of classification loss ('bce', 'ce')
        dfl_loss: Whether to use Distribution Focal Loss
        alpha: Weight for classification loss
        beta: Weight for box regression loss
        gamma: Weight for objectness loss
    """
    
    def __init__(
        self,
        num_classes: int = config.model.NUM_CLASSES,
        box_loss: str = config.training.BOX_LOSS,
        cls_loss: str = config.training.CLS_LOSS,
        dfl_loss: bool = config.training.DFL_LOSS,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.box_loss = box_loss
        self.cls_loss = cls_loss
        self.dfl_loss = dfl_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize loss functions
        self._init_loss_functions()
        
        print(f"DetectionLoss: box={box_loss}, cls={cls_loss}, dfl={dfl_loss}")
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        if self.cls_loss == 'bce':
            self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        else:  # 'ce'
            self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        # Box loss function
        if self.box_loss == 'mse':
            self.box_loss_fn = nn.MSELoss(reduction='none')
        else:
            self.box_loss_fn = self._get_box_loss_fn()
        
        # Objectness loss
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # DFL loss
        if self.dfl_loss:
            self.dfl_loss_fn = DistributionFocalLoss()
    
    def _get_box_loss_fn(self) -> callable:
        """Get box loss function"""
        if self.box_loss == 'ciou':
            return CIoULoss()
        elif self.box_loss == 'diou':
            return DIoULoss()
        elif self.box_loss == 'siou':
            return SIoULoss()
        else:
            return nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute detection loss
        
        Args:
            predictions: Model predictions (B, num_anchors * (5 + num_classes), H, W)
                         or (B, num_anchors, 5 + num_classes)
            targets: Ground truth targets (B, num_targets, 5) in xyxy format with class
            label_mask: Mask indicating valid labels (B, num_targets)
        
        Returns:
            Total detection loss
        """
        # Parse predictions and targets
        device = predictions.device
        
        # Check prediction format
        if predictions.dim() == 4:
            # (B, C, H, W) format
            B, C, H, W = predictions.shape
            num_anchors = C // (5 + self.num_classes)
            predictions = predictions.view(B, num_anchors, 5 + self.num_classes, H, W)
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
            predictions = predictions.view(B, -1, 5 + self.num_classes)
        
        # predictions: (B, num_preds, 5 + num_classes)
        # targets: (B, num_targets, 5)
        
        B, num_preds, _ = predictions.shape
        num_targets = targets.shape[1] if targets.dim() == 3 else 1
        
        # Split predictions
        pred_boxes = predictions[..., :4]  # xyxy
        pred_scores = predictions[..., 4:5]  # objectness
        pred_classes = predictions[..., 5:]  # class probabilities
        
        # Split targets
        target_boxes = targets[..., :4]  # xyxy
        target_classes = targets[..., 4:5].long()  # class indices
        
        # Compute box loss
        box_loss = self._compute_box_loss(pred_boxes, target_boxes, label_mask)
        
        # Compute classification loss
        cls_loss = self._compute_cls_loss(pred_classes, target_classes, label_mask)
        
        # Compute objectness loss
        obj_loss = self._compute_obj_loss(pred_scores, target_classes, label_mask)
        
        # Compute DFL loss if enabled
        dfl_loss = torch.tensor(0.0, device=device)
        if self.dfl_loss and self.dfl_loss_fn is not None:
            dfl_loss = self.dfl_loss_fn(pred_scores, target_classes, label_mask)
        
        # Total loss
        total_loss = self.alpha * cls_loss + self.beta * box_loss + self.gamma * obj_loss + dfl_loss
        
        return total_loss
    
    def _compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute box regression loss"""
        if label_mask is None:
            label_mask = torch.ones_like(target_boxes[..., 0], dtype=torch.bool)
        
        # Expand label mask for all box coordinates
        box_mask = label_mask.unsqueeze(-1).expand_as(target_boxes)
        
        # Compute loss only for valid targets
        if box_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Use box loss function
        if isinstance(self.box_loss_fn, nn.MSELoss):
            loss = self.box_loss_fn(pred_boxes, target_boxes)
        else:
            # For IoU-based losses, we need to match predictions to targets
            # This is simplified - in practice, we'd use proper matching
            loss = torch.zeros_like(pred_boxes[..., 0])
            for b in range(pred_boxes.shape[0]):
                for i in range(pred_boxes.shape[1]):
                    if i < target_boxes.shape[1] and label_mask[b, i]:
                        loss[b, i] = self.box_loss_fn(
                            pred_boxes[b, i:i+1],
                            target_boxes[b, i:i+1]
                        )
        
        # Apply mask and mean
        masked_loss = loss[box_mask]
        if masked_loss.numel() > 0:
            return masked_loss.mean()
        else:
            return torch.tensor(0.0, device=pred_boxes.device)
    
    def _compute_cls_loss(
        self,
        pred_classes: torch.Tensor,
        target_classes: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute classification loss"""
        if label_mask is None:
            label_mask = torch.ones_like(target_classes, dtype=torch.bool)
        
        # For BCE loss, create one-hot targets
        if self.cls_loss == 'bce':
            targets_one_hot = torch.zeros_like(pred_classes)
            targets_one_hot.scatter_(-1, target_classes.unsqueeze(-1), 1)
            
            loss = self.cls_loss_fn(pred_classes, targets_one_hot)
        else:  # Cross entropy
            # Reshape for CE loss
            pred_flat = pred_classes.view(-1, self.num_classes)
            target_flat = target_classes.view(-1)
            
            loss = self.cls_loss_fn(pred_flat, target_flat)
            loss = loss.view_as(target_classes)
        
        # Apply mask
        masked_loss = loss[label_mask]
        if masked_loss.numel() > 0:
            return masked_loss.mean()
        else:
            return torch.tensor(0.0, device=pred_classes.device)
    
    def _compute_obj_loss(
        self,
        pred_scores: torch.Tensor,
        target_classes: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute objectness loss"""
        if label_mask is None:
            label_mask = torch.ones_like(target_classes, dtype=torch.bool)
        
        # Create objectness targets (1 for valid objects, 0 otherwise)
        obj_targets = torch.zeros_like(pred_scores)
        obj_targets[label_mask] = 1.0
        
        loss = self.obj_loss_fn(pred_scores, obj_targets)
        
        return loss.mean()


class CIoULoss(nn.Module):
    """
    Complete IoU Loss
    """
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CIoU loss
        
        Args:
            pred: Predicted boxes (N, 4) in xyxy format
            target: Target boxes (N, 4) in xyxy format
        
        Returns:
            CIoU loss (N,)
        """
        # Compute IoU
        iou = self._box_iou(pred, target)
        
        # Compute distance between centers
        pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
        pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
        target_ctr_x = (target[..., 0] + target[..., 2]) / 2
        target_ctr_y = (target[..., 1] + target[..., 3]) / 2
        
        ctr_dist = ((pred_ctr_x - target_ctr_x) ** 2 + 
                   (pred_ctr_y - target_ctr_y) ** 2) ** 0.5
        
        # Compute diagonal of bounding box that covers both boxes
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        c_square = (pred_w ** 2 + pred_h ** 2 + 
                   target_w ** 2 + target_h ** 2) / 4 + self.epsilon
        
        # Compute CIoU
        u = ctr_dist ** 2 / c_square + self.epsilon
        cious = iou - u
        
        # Loss
        loss = 1 - cious.clamp(min=-1.0, max=1.0)
        
        return loss
    
    def _box_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes"""
        # Intersection
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:], target[..., 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # Union
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = pred_area + target_area - inter + self.epsilon
        
        # IoU
        iou = inter / union
        
        return iou


class DIoULoss(nn.Module):
    """
    Distance IoU Loss
    """
    
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DIoU loss"""
        # Compute IoU
        iou = self._box_iou(pred, target)
        
        # Compute distance between centers
        pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
        pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
        target_ctr_x = (target[..., 0] + target[..., 2]) / 2
        target_ctr_y = (target[..., 1] + target[..., 3]) / 2
        
        ctr_dist = ((pred_ctr_x - target_ctr_x) ** 2 + 
                   (pred_ctr_y - target_ctr_y) ** 2) ** 0.5
        
        # Compute diagonal of bounding box that covers both boxes
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        c_square = (pred_w ** 2 + pred_h ** 2 + 
                   target_w ** 2 + target_h ** 2) / 4 + self.epsilon
        
        # Compute DIoU
        u = ctr_dist ** 2 / c_square + self.epsilon
        dious = iou - u
        
        # Loss
        loss = 1 - dious.clamp(min=-1.0, max=1.0)
        
        return loss
    
    def _box_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes"""
        # Intersection
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:], target[..., 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # Union
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = pred_area + target_area - inter + self.epsilon
        
        # IoU
        iou = inter / union
        
        return iou


class SIoULoss(nn.Module):
    """
    Scaled IoU Loss
    """
    
    def __init__(self, epsilon: float = 1e-7, alpha: float = 0.5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SIoU loss"""
        # Compute IoU
        iou = self._box_iou(pred, target)
        
        # Compute distance between centers
        pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
        pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
        target_ctr_x = (target[..., 0] + target[..., 2]) / 2
        target_ctr_y = (target[..., 1] + target[..., 3]) / 2
        
        ctr_dist = ((pred_ctr_x - target_ctr_x) ** 2 + 
                   (pred_ctr_y - target_ctr_y) ** 2) ** 0.5
        
        # Compute diagonal of bounding box that covers both boxes
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        c_square = (pred_w ** 2 + pred_h ** 2 + 
                   target_w ** 2 + target_h ** 2) / 4 + self.epsilon
        
        # Compute SIoU
        u = ctr_dist ** 2 / c_square + self.epsilon
        
        # Angle consistency
        pred_angle = torch.atan2(pred_h, pred_w)
        target_angle = torch.atan2(target_h, target_w)
        angle_diff = (pred_angle - target_angle) * (math.pi / 180)
        angle_consistency = (torch.cos(angle_diff) * 4 / (math.pi ** 2)) - 1
        
        # Scale consistency
        scale_consistency = (pred_w - target_w) ** 2 / (pred_w ** 2 + target_w ** 2 + self.epsilon) + \
                           (pred_h - target_h) ** 2 / (pred_h ** 2 + target_h ** 2 + self.epsilon)
        
        # Aspect ratio consistency
        pred_ratio = pred_w / (pred_h + self.epsilon)
        target_ratio = target_w / (target_h + self.epsilon)
        ratio_diff = (pred_ratio - target_ratio) ** 2
        
        # Combine
        sigma = 2 - iou
        alpha = self.alpha * sigma
        
        sious = iou - (u + alpha * angle_consistency + scale_consistency + ratio_diff) / 4
        
        # Loss
        loss = 1 - sious.clamp(min=-1.0, max=1.0)
        
        return loss
    
    def _box_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU between boxes"""
        # Intersection
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:], target[..., 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # Union
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = pred_area + target_area - inter + self.epsilon
        
        # IoU
        iou = inter / union
        
        return iou


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss for object detection
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Distribution Focal Loss
        
        Args:
            pred: Predicted scores (N,)
            target: Target scores (N,) - 1 for positive, 0 for negative
            label_mask: Mask for valid labels
        
        Returns:
            DFL loss
        """
        # Compute BCE loss
        bce_loss = self.bce_loss(pred, target)
        
        # Compute focal weight
        if label_mask is not None:
            # For positive samples
            pos_mask = (target == 1) & label_mask
            neg_mask = (target == 0) & label_mask
            
            pos_loss = bce_loss[pos_mask]
            neg_loss = bce_loss[neg_mask]
            
            # Focal weight for positive samples
            pos_weight = self.alpha * (1 - torch.sigmoid(pred[pos_mask])) ** self.gamma
            pos_loss = pos_loss * pos_weight
            
            # Focal weight for negative samples
            neg_weight = (1 - self.alpha) * (torch.sigmoid(pred[neg_mask])) ** self.gamma
            neg_loss = neg_loss * neg_weight
            
            # Combine
            loss = torch.cat([pos_loss, neg_loss]).mean()
        else:
            # Simple focal loss
            p = torch.sigmoid(pred)
            focal_weight = self.alpha * (1 - p) ** self.gamma * target + \
                          (1 - self.alpha) * p ** self.gamma * (1 - target)
            loss = (bce_loss * focal_weight).mean()
        
        return loss


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for feature matching
    
    Args:
        loss_type: Type of KD loss ('mse', 'cosine', 'kl')
        temperature: Temperature for softening distributions
    """
    
    def __init__(
        self,
        loss_type: str = config.training.KD_LOSS,
        temperature: float = config.training.KD_TEMPERATURE
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.temperature = temperature
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = self._cosine_loss
        elif loss_type == 'kl':
            self.loss_fn = self._kl_loss
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_features: Student features
            teacher_features: Teacher features
        
        Returns:
            KD loss
        """
        return self.loss_fn(student_features, teacher_features)
    
    def _cosine_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss"""
        # Normalize
        student_norm = F.normalize(student, p=2, dim=-1)
        teacher_norm = F.normalize(teacher, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.sum(student_norm * teacher_norm, dim=-1)
        
        # Loss: minimize distance (1 - similarity)
        loss = (1 - similarity).mean()
        
        return loss
    
    def _kl_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """KL divergence loss"""
        # Apply temperature
        student_logits = student / self.temperature
        teacher_logits = teacher / self.temperature
        
        # Softmax
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # KL divergence
        kl_div = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-8) - torch.log(student_probs + 1e-8)),
            dim=-1
        )
        
        # Scale by temperature^2
        loss = (self.temperature ** 2) * kl_div.mean()
        
        return loss


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Create dummy data
    batch_size = 2
    num_preds = 10
    num_classes = 8
    
    # Predictions: (B, num_preds, 5 + num_classes)
    dummy_preds = torch.randn(batch_size, num_preds, 5 + num_classes)
    
    # Targets: (B, num_targets, 5)
    dummy_targets = torch.randn(batch_size, 5, 5)  # 5 targets per image
    dummy_targets[..., 4] = torch.randint(0, num_classes, (batch_size, 5))  # class indices
    
    # Label mask
    dummy_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    
    # Test DetectionLoss
    print("\nTesting DetectionLoss...")
    detection_loss = DetectionLoss(
        num_classes=num_classes,
        box_loss="ciou",
        cls_loss="bce",
        dfl_loss=False
    )
    
    loss = detection_loss(dummy_preds, dummy_targets, dummy_mask)
    print(f"Detection loss: {loss.item()}")
    
    # Test individual box losses
    print("\nTesting Box Losses...")
    
    pred_boxes = torch.randn(10, 4)  # (N, 4)
    target_boxes = torch.randn(10, 4)
    
    ciou_loss = CIoULoss()
    diou_loss = DIoULoss()
    siou_loss = SIoULoss()
    
    print(f"CIoU loss: {ciou_loss(pred_boxes, target_boxes).mean().item()}")
    print(f"DIoU loss: {diou_loss(pred_boxes, target_boxes).mean().item()}")
    print(f"SIoU loss: {siou_loss(pred_boxes, target_boxes).mean().item()}")
    
    # Test Knowledge Distillation Loss
    print("\nTesting Knowledge Distillation Loss...")
    
    student_features = torch.randn(10, 768)
    teacher_features = torch.randn(10, 768)
    
    kd_loss_mse = KnowledgeDistillationLoss(loss_type="mse")
    kd_loss_cosine = KnowledgeDistillationLoss(loss_type="cosine")
    kd_loss_kl = KnowledgeDistillationLoss(loss_type="kl", temperature=1.0)
    
    print(f"KD MSE loss: {kd_loss_mse(student_features, teacher_features).item()}")
    print(f"KD Cosine loss: {kd_loss_cosine(student_features, teacher_features).item()}")
    print(f"KD KL loss: {kd_loss_kl(student_features, teacher_features).item()}")
    
    print("Loss functions test completed!")
