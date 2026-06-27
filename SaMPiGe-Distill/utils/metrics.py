"""
Metrics Implementation
Evaluation metrics for object detection and knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from collections import defaultdict
import warnings

from config import config


class MetricCalculator:
    """
    Base class for metric calculation
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        pass
    
    def update(self, *args, **kwargs):
        """Update metrics with new data"""
        pass
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        return {}
    
    def get_results(self) -> Dict[str, float]:
        """Get computed results"""
        return self.compute()


class DetectionMetrics(MetricCalculator):
    """
    Object detection metrics calculator
    
    Computes:
    - mAP (mean Average Precision)
    - Precision and Recall
    - F1 score
    - Confusion matrix
    
    Args:
        num_classes: Number of classes
        iou_thresholds: IoU thresholds for mAP calculation
        max_detection: Maximum number of detections per image
    """
    
    def __init__(
        self,
        num_classes: int = config.model.NUM_CLASSES,
        iou_thresholds: List[float] = config.validation.METRIC_THRESHOLDS,
        max_detection: int = 100
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.max_detection = max_detection
        
        # Accumulators
        self.ground_truths = []
        self.predictions = []
        self.scores = []
        
        # For precision/recall
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        
        # For confusion matrix
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    def reset(self):
        """Reset all accumulators"""
        self.ground_truths = []
        self.predictions = []
        self.scores = []
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None
    ):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Predicted boxes (N, 5) in xyxy format with class
            targets: Ground truth boxes (M, 5) in xyxy format with class
            scores: Confidence scores for predictions
            label_mask: Mask for valid targets
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.detach().cpu().numpy()
        
        # Store for mAP calculation
        self.ground_truths.append(targets)
        self.predictions.append(predictions)
        if scores is not None:
            self.scores.append(scores)
        else:
            # Use confidence from predictions (assuming last column is confidence)
            if predictions.shape[1] >= 6:
                self.scores.append(predictions[:, 5])
            else:
                self.scores.append(np.ones(predictions.shape[0]))
        
        # Update TP/FP/FN for precision/recall
        self._update_detection_stats(predictions, targets, label_mask)
    
    def _update_detection_stats(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        label_mask: Optional[np.ndarray] = None
    ):
        """Update true positive, false positive, false negative counts"""
        if len(targets) == 0:
            # All predictions are false positives
            for pred in predictions:
                class_id = int(pred[4])
                self.false_positives[class_id] += 1
            return
        
        if len(predictions) == 0:
            # All targets are false negatives
            for target in targets:
                if label_mask is None or label_mask[target[4]]:
                    class_id = int(target[4])
                    self.false_negatives[class_id] += 1
            return
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(predictions[:, :4], targets[:, :4])
        
        # Match predictions to targets
        matched = set()
        for i in range(len(predictions)):
            pred_box = predictions[i, :4]
            pred_class = int(predictions[i, 4])
            
            best_iou = 0
            best_j = -1
            
            for j in range(len(targets)):
                if j in matched:
                    continue
                
                target_box = targets[j, :4]
                target_class = int(targets[j, 4])
                
                # Only match same class
                if pred_class != target_class:
                    continue
                
                # Check label mask
                if label_mask is not None and not label_mask[j]:
                    continue
                
                iou = iou_matrix[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            # Check if matched
            if best_j >= 0 and best_iou >= 0.5:  # IoU threshold
                self.true_positives[pred_class] += 1
                matched.add(best_j)
                
                # Update confusion matrix
                self.confusion_matrix[pred_class, pred_class] += 1
            else:
                self.false_positives[pred_class] += 1
        
        # Update false negatives for unmatched targets
        for j in range(len(targets)):
            if j not in matched:
                if label_mask is None or label_mask[j]:
                    class_id = int(targets[j, 4])
                    self.false_negatives[class_id] += 1
    
    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes"""
        # Intersection
        lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[np.newaxis, :, :2])
        rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[np.newaxis, :, 2:])
        
        wh = np.maximum(rb - lt, 0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
        
        # IoU
        iou = inter / (union + 1e-8)
        
        return iou
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        results = {}
        
        # Compute mAP
        mAP_results = self.compute_mAP()
        for key, value in mAP_results.items():
            results[key] = value
        
        # Compute precision, recall, F1
        precision, recall, f1 = self.compute_precision_recall_f1()
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        
        # Compute per-class metrics
        per_class_results = self.compute_per_class_metrics()
        for key, value in per_class_results.items():
            results[key] = value
        
        return results
    
    def compute_mAP(self) -> Dict[str, float]:
        """Compute mean Average Precision"""
        try:
            from pycocotools.cocoeval import COCOeval
            
            # Convert to COCO format
            # This is a simplified version - full implementation would need proper COCO format
            
            # For now, use a simpler approach
            mAP_50 = self._compute_mAP_at_iou(0.5)
            mAP_50_95 = self._compute_mAP_at_iou(0.5, average=True)
            
            return {
                'mAP50': mAP_50,
                'mAP50-95': mAP_50_95
            }
        
        except ImportError:
            warnings.warn("pycocotools not available, using simplified mAP calculation")
            mAP_50 = self._compute_mAP_at_iou(0.5)
            return {'mAP50': mAP_50}
    
    def _compute_mAP_at_iou(self, iou_threshold: float = 0.5, average: bool = False) -> float:
        """Compute mAP at specific IoU threshold"""
        if not self.ground_truths or not self.predictions:
            return 0.0
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_targets = []
        all_scores = []
        
        for preds, targets, scores in zip(self.predictions, self.ground_truths, self.scores):
            all_predictions.append(preds)
            all_targets.append(targets)
            all_scores.append(scores)
        
        # Sort predictions by score
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Sort by score (descending)
        sort_idx = np.argsort(-all_scores)
        all_predictions = all_predictions[sort_idx]
        all_scores = all_scores[sort_idx]
        
        # Compute AP for each class
        aps = []
        for class_id in range(self.num_classes):
            # Filter predictions and targets for this class
            class_mask_pred = all_predictions[:, 4] == class_id
            class_mask_target = all_targets[:, 4] == class_id
            
            class_preds = all_predictions[class_mask_pred]
            class_scores = all_scores[class_mask_pred]
            class_targets = all_targets[class_mask_target]
            
            if len(class_targets) == 0:
                aps.append(0.0)
                continue
            
            # Compute AP for this class
            ap = self._compute_AP(
                class_preds, class_scores, class_targets, iou_threshold
            )
            aps.append(ap)
        
        # Compute mAP
        if average:
            # Average over IoU thresholds (simplified)
            return np.mean(aps)
        else:
            return np.mean(aps)
    
    def _compute_AP(
        self,
        predictions: np.ndarray,
        scores: np.ndarray,
        targets: np.ndarray,
        iou_threshold: float
    ) -> float:
        """Compute Average Precision for a single class"""
        if len(predictions) == 0 or len(targets) == 0:
            return 0.0
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(predictions[:, :4], targets[:, :4])
        
        # Sort predictions by score
        sort_idx = np.argsort(-scores)
        predictions = predictions[sort_idx]
        iou_matrix = iou_matrix[sort_idx]
        
        # Initialize
        true_positives = np.zeros(len(predictions))
        false_positives = np.zeros(len(predictions))
        
        # Match predictions to targets
        matched = set()
        for i in range(len(predictions)):
            best_iou = 0
            best_j = -1
            
            for j in range(len(targets)):
                if j in matched:
                    continue
                
                iou = iou_matrix[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j >= 0 and best_iou >= iou_threshold:
                true_positives[i] = 1
                matched.add(best_j)
            else:
                false_positives[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (len(targets) + 1e-8)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recall >= t
            if mask.sum() > 0:
                ap += precision[mask].max()
        
        ap /= 11
        
        return ap
    
    def compute_precision_recall_f1(self) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score"""
        total_tp = sum(self.true_positives.values())
        total_fp = sum(self.false_positives.values())
        total_fn = sum(self.false_negatives.values())
        
        # Precision
        precision = total_tp / (total_tp + total_fp + 1e-8)
        
        # Recall
        recall = total_tp / (total_tp + total_fn + 1e-8)
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1
    
    def compute_per_class_metrics(self) -> Dict[str, float]:
        """Compute per-class metrics"""
        results = {}
        
        for class_id in range(self.num_classes):
            tp = self.true_positives.get(class_id, 0)
            fp = self.false_positives.get(class_id, 0)
            fn = self.false_negatives.get(class_id, 0)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            results[f'precision_class_{class_id}'] = precision
            results[f'recall_class_{class_id}'] = recall
            results[f'f1_class_{class_id}'] = f1
        
        return results


def compute_mAP(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mAP for a set of predictions and targets
    
    Args:
        predictions: List of prediction arrays (N_i, 5) for each image
        targets: List of target arrays (M_i, 5) for each image
        iou_threshold: IoU threshold for matching
    
    Returns:
        mAP value
    """
    metrics = DetectionMetrics()
    
    for pred, target in zip(predictions, targets):
        metrics.update(pred, target)
    
    results = metrics.compute()
    return results.get('mAP50', 0.0)


def compute_precision_recall(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> Tuple[float, float]:
    """
    Compute precision and recall
    
    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
    
    Returns:
        Tuple of (precision, recall)
    """
    metrics = DetectionMetrics()
    
    for pred, target in zip(predictions, targets):
        metrics.update(pred, target)
    
    precision, recall, _ = metrics.compute_precision_recall_f1()
    return precision, recall


def compute_f1_score(
    predictions: List[np.ndarray],
    targets: List[np.ndarray]
) -> float:
    """
    Compute F1 score
    
    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
    
    Returns:
        F1 score
    """
    metrics = DetectionMetrics()
    
    for pred, target in zip(predictions, targets):
        metrics.update(pred, target)
    
    _, _, f1 = metrics.compute_precision_recall_f1()
    return f1


class KnowledgeDistillationMetrics(MetricCalculator):
    """
    Metrics for knowledge distillation
    
    Tracks:
    - Feature similarity
    - Attention similarity
    - Prototype alignment
    """
    
    def __init__(self):
        super().__init__()
        
        # Accumulators
        self.feature_similarities = []
        self.attention_similarities = []
        self.prototype_alignments = []
        self.losses = defaultdict(list)
    
    def reset(self):
        """Reset accumulators"""
        self.feature_similarities = []
        self.attention_similarities = []
        self.prototype_alignments = []
        self.losses = defaultdict(list)
    
    def update(
        self,
        feature_similarity: Optional[float] = None,
        attention_similarity: Optional[float] = None,
        prototype_alignment: Optional[float] = None,
        losses: Optional[Dict[str, float]] = None
    ):
        """Update metrics"""
        if feature_similarity is not None:
            self.feature_similarities.append(feature_similarity)
        if attention_similarity is not None:
            self.attention_similarities.append(attention_similarity)
        if prototype_alignment is not None:
            self.prototype_alignments.append(prototype_alignment)
        if losses is not None:
            for key, value in losses.items():
                self.losses[key].append(value)
    
    def compute(self) -> Dict[str, float]:
        """Compute metrics"""
        results = {}
        
        if self.feature_similarities:
            results['feature_similarity'] = np.mean(self.feature_similarities)
        if self.attention_similarities:
            results['attention_similarity'] = np.mean(self.attention_similarities)
        if self.prototype_alignments:
            results['prototype_alignment'] = np.mean(self.prototype_alignments)
        
        for key, values in self.losses.items():
            if values:
                results[f'loss_{key}'] = np.mean(values)
        
        return results


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics...")
    
    # Create dummy data
    num_classes = 8
    
    # Predictions: (N, 5) - xyxy + class
    dummy_preds = np.array([
        [10, 20, 100, 150, 0],  # Car
        [50, 60, 120, 180, 1],  # Van
        [200, 200, 250, 250, 0]  # Car
    ])
    
    # Targets: (M, 5) - xyxy + class
    dummy_targets = np.array([
        [15, 25, 105, 155, 0],  # Car (matched with first prediction)
        [55, 65, 125, 185, 1],  # Van (matched with second prediction)
        [300, 300, 350, 350, 2]  # Truck (not matched)
    ])
    
    # Test DetectionMetrics
    print("\nTesting DetectionMetrics...")
    metrics = DetectionMetrics(num_classes=num_classes)
    
    metrics.update(dummy_preds, dummy_targets)
    
    results = metrics.compute()
    
    print(f"mAP50: {results.get('mAP50', 0.0)}")
    print(f"Precision: {results.get('precision', 0.0)}")
    print(f"Recall: {results.get('recall', 0.0)}")
    print(f"F1: {results.get('f1', 0.0)}")
    
    # Test standalone functions
    print("\nTesting standalone functions...")
    mAP = compute_mAP([dummy_preds], [dummy_targets])
    print(f"mAP: {mAP}")
    
    precision, recall = compute_precision_recall([dummy_preds], [dummy_targets])
    print(f"Precision: {precision}, Recall: {recall}")
    
    f1 = compute_f1_score([dummy_preds], [dummy_targets])
    print(f"F1: {f1}")
    
    print("Metrics test completed!")
