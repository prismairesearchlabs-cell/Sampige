"""
Dynamic Loss Weight Scheduler
Adaptively adjusts loss weights during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Callable
import math
import numpy as np

from config import config


class DynamicWeightScheduler:
    """
    Dynamic weight scheduler for multi-task learning
    
    Adjusts loss weights based on:
    - Training progress (epoch)
    - Loss values
    - Gradient magnitudes
    - Custom schedules
    
    Args:
        initial_weights: Initial loss weights
        schedule_type: Type of schedule ('linear', 'cosine', 'step', 'adaptive')
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        cooldown_epochs: Number of cooldown epochs
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        schedule_type: str = config.training.SCHEDULER,
        total_epochs: int = config.training.EPOCHS,
        warmup_epochs: int = config.training.WARMUP_EPOCHS,
        cooldown_epochs: int = config.training.COOLDOWN_EPOCHS
    ):
        self.initial_weights = initial_weights or {
            'detection': config.training.DETECTION_WEIGHT,
            'feature': config.training.FEATURE_WEIGHT,
            'attention': config.training.ATTENTION_WEIGHT,
            'relation': config.training.RELATION_WEIGHT,
            'prototype': config.training.PROTOTYPE_WEIGHT,
            'patch': config.training.PATCH_WEIGHT,
            'consistency': config.training.CONSISTENCY_WEIGHT
        }
        
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        
        # Current epoch
        self.current_epoch = 0
        
        # Loss history for adaptive scheduling
        self.loss_history = {key: [] for key in self.initial_weights.keys()}
        
        # Gradient history
        self.grad_history = {key: [] for key in self.initial_weights.keys()}
        
        # Initialize schedule
        self._init_schedule()
        
        print(f"DynamicWeightScheduler: {schedule_type}, epochs={total_epochs}")
        print(f"Initial weights: {self.initial_weights}")
    
    def _init_schedule(self):
        """Initialize the scheduling strategy"""
        if self.schedule_type == 'linear':
            self.schedule_fn = self._linear_schedule
        elif self.schedule_type == 'cosine':
            self.schedule_fn = self._cosine_schedule
        elif self.schedule_type == 'step':
            self.schedule_fn = self._step_schedule
        elif self.schedule_type == 'adaptive':
            self.schedule_fn = self._adaptive_schedule
        else:
            self.schedule_fn = self._default_schedule
    
    def __call__(self, epoch: int) -> Dict[str, float]:
        """
        Get weights for current epoch
        
        Args:
            epoch: Current epoch
        
        Returns:
            Dictionary of updated weights
        """
        self.current_epoch = epoch
        return self.schedule_fn(epoch)
    
    def _default_schedule(self, epoch: int) -> Dict[str, float]:
        """Default schedule - return initial weights"""
        return self.initial_weights.copy()
    
    def _linear_schedule(self, epoch: int) -> Dict[str, float]:
        """
        Linear schedule
        
        Detection weight increases, distillation weights decrease
        """
        weights = self.initial_weights.copy()
        
        # Normalized epoch (0 to 1)
        norm_epoch = min(epoch / self.total_epochs, 1.0)
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            warmup_factor = min(epoch / self.warmup_epochs, 1.0)
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * warmup_factor
        
        # Cooldown phase
        elif epoch > self.total_epochs - self.cooldown_epochs:
            cooldown_factor = max(
                0, 
                (self.total_epochs - epoch) / self.cooldown_epochs
            )
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * cooldown_factor
        
        # Main phase
        else:
            # Detection weight increases
            detection_factor = 0.5 + 0.5 * norm_epoch
            weights['detection'] = self.initial_weights['detection'] * detection_factor
            
            # Distillation weights decrease
            distill_factor = 1.0 - 0.5 * norm_epoch
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * distill_factor
        
        return weights
    
    def _cosine_schedule(self, epoch: int) -> Dict[str, float]:
        """
        Cosine schedule
        
        Smooth transition between phases
        """
        weights = self.initial_weights.copy()
        
        # Normalized epoch (0 to 1)
        norm_epoch = min(epoch / self.total_epochs, 1.0)
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            warmup_factor = 0.5 * (1 + math.cos(math.pi * 
                (1 - min(epoch / self.warmup_epochs, 1.0))))
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * warmup_factor
        
        # Cooldown phase
        elif epoch > self.total_epochs - self.cooldown_epochs:
            cooldown_factor = 0.5 * (1 + math.cos(math.pi * 
                min((epoch - (self.total_epochs - self.cooldown_epochs)) / 
                   self.cooldown_epochs, 1.0)))
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * (1 - cooldown_factor)
        
        # Main phase
        else:
            # Detection weight follows cosine from 0.5 to 1.0
            detection_factor = 0.5 * (1 + math.cos(math.pi * (1 - norm_epoch)))
            weights['detection'] = self.initial_weights['detection'] * detection_factor
            
            # Distillation weights follow cosine from 1.0 to 0.5
            distill_factor = 0.5 * (1 + math.cos(math.pi * norm_epoch))
            for key in weights:
                if key != 'detection':
                    weights[key] = self.initial_weights[key] * distill_factor
        
        return weights
    
    def _step_schedule(self, epoch: int) -> Dict[str, float]:
        """
        Step schedule with discrete changes
        """
        weights = self.initial_weights.copy()
        
        # Define step points
        step_points = [
            (0, 1.0),  # Start
            (self.warmup_epochs, 1.0),  # End warmup
            (self.total_epochs // 2, 0.5),  # Midpoint
            (self.total_epochs - self.cooldown_epochs, 0.25),  # Start cooldown
            (self.total_epochs, 0.1)  # End
        ]
        
        # Find current step
        current_factor = 1.0
        for step_epoch, factor in step_points:
            if epoch >= step_epoch:
                current_factor = factor
            else:
                break
        
        # Apply factor
        for key in weights:
            if key != 'detection':
                weights[key] = self.initial_weights[key] * current_factor
        
        # Detection weight is inverse
        detection_factor = 2.0 - current_factor
        weights['detection'] = self.initial_weights['detection'] * detection_factor
        
        return weights
    
    def _adaptive_schedule(self, epoch: int) -> Dict[str, float]:
        """
        Adaptive schedule based on loss values
        
        Increases weights for losses that are decreasing slowly
        """
        weights = self.initial_weights.copy()
        
        # If we don't have enough history, use default
        if len(self.loss_history[list(self.loss_history.keys())[0]]) < 10:
            return weights
        
        # Compute loss trends
        trends = {}
        for key, history in self.loss_history.items():
            if len(history) >= 10:
                # Compute slope of last 10 losses
                x = np.arange(len(history))
                y = np.array(history)
                A = np.vstack([x, np.ones(len(x))]).T
                m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                trends[key] = m
            else:
                trends[key] = 0
        
        # Adjust weights based on trends
        # If loss is decreasing slowly (small negative slope), increase weight
        for key in weights:
            if key in trends:
                # Normalize trend
                norm_trend = trends[key] / (abs(trends[key]) + 1e-8)
                
                # Adjust weight
                adjustment = 0.5 * (1 - norm_trend)  # 0 to 1
                weights[key] = self.initial_weights[key] * (1 + adjustment)
        
        # Normalize weights to maintain similar scale
        total_weight = sum(weights.values())
        target_total = sum(self.initial_weights.values())
        scale = target_total / (total_weight + 1e-8)
        
        for key in weights:
            weights[key] = weights[key] * scale
        
        return weights
    
    def update_loss_history(self, losses: Dict[str, float]):
        """
        Update loss history for adaptive scheduling
        
        Args:
            losses: Dictionary of current losses
        """
        for key, value in losses.items():
            if key in self.loss_history:
                self.loss_history[key].append(value)
                # Keep only last 100 values
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]
    
    def update_gradient_history(self, gradients: Dict[str, float]):
        """
        Update gradient history for adaptive scheduling
        
        Args:
            gradients: Dictionary of gradient magnitudes
        """
        for key, value in gradients.items():
            if key in self.grad_history:
                self.grad_history[key].append(value)
                if len(self.grad_history[key]) > 100:
                    self.grad_history[key] = self.grad_history[key][-100:]
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current weights for current epoch"""
        return self(self.current_epoch)
    
    def reset(self):
        """Reset scheduler"""
        self.current_epoch = 0
        self.loss_history = {key: [] for key in self.initial_weights.keys()}
        self.grad_history = {key: [] for key in self.initial_weights.keys()}


class LambdaScheduler:
    """
    Lambda-based scheduler for custom weight functions
    
    Args:
        initial_weights: Initial weights
        lambda_functions: Dictionary of lambda functions for each weight
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        lambda_functions: Optional[Dict[str, Callable[[int], float]]] = None
    ):
        self.initial_weights = initial_weights
        self.lambda_functions = lambda_functions or {}
        self.current_epoch = 0
    
    def __call__(self, epoch: int) -> Dict[str, float]:
        """Get weights for current epoch"""
        self.current_epoch = epoch
        weights = {}
        
        for key, initial_weight in self.initial_weights.items():
            if key in self.lambda_functions:
                lambda_val = self.lambda_functions[key](epoch)
                weights[key] = initial_weight * lambda_val
            else:
                weights[key] = initial_weight
        
        return weights
    
    def add_lambda(self, key: str, lambda_fn: Callable[[int], float]):
        """Add lambda function for a specific weight"""
        self.lambda_functions[key] = lambda_fn


class LossBalancingScheduler:
    """
    Loss balancing scheduler based on gradient magnitudes
    
    Adjusts weights to balance gradient magnitudes across tasks
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        target_grad_ratio: float = 1.0,
        smoothing: float = 0.9
    ):
        self.initial_weights = initial_weights
        self.target_grad_ratio = target_grad_ratio
        self.smoothing = smoothing
        
        # Gradient statistics
        self.grad_stats = {key: 0.0 for key in initial_weights.keys()}
        self.current_epoch = 0
    
    def __call__(self, epoch: int) -> Dict[str, float]:
        """Get balanced weights"""
        self.current_epoch = epoch
        
        if epoch == 0:
            return self.initial_weights.copy()
        
        # Compute current gradient ratios
        grad_values = list(self.grad_stats.values())
        if not grad_values or all(g == 0 for g in grad_values):
            return self.initial_weights.copy()
        
        mean_grad = sum(grad_values) / len(grad_values)
        
        # Compute adjustment factors
        weights = {}
        for key, initial_weight in self.initial_weights.items():
            if self.grad_stats[key] > 0:
                # If gradient is higher than average, reduce weight
                ratio = self.grad_stats[key] / (mean_grad + 1e-8)
                adjustment = 1.0 / (ratio ** 0.5)  # Square root for smoother adjustment
                weights[key] = initial_weight * adjustment
            else:
                weights[key] = initial_weight
        
        return weights
    
    def update_gradients(self, gradients: Dict[str, float]):
        """
        Update gradient statistics
        
        Args:
            gradients: Dictionary of gradient magnitudes
        """
        for key, value in gradients.items():
            if key in self.grad_stats:
                # Exponential moving average
                self.grad_stats[key] = (
                    self.smoothing * self.grad_stats[key] + 
                    (1 - self.smoothing) * value
                )


class WarmupScheduler:
    """
    Warmup scheduler for learning rate and loss weights
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        warmup_epochs: int = config.training.WARMUP_EPOCHS,
        warmup_type: str = 'linear'  # or 'cosine', 'exponential'
    ):
        self.initial_weights = initial_weights
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.current_epoch = 0
    
    def __call__(self, epoch: int) -> Dict[str, float]:
        """Get weights with warmup"""
        self.current_epoch = epoch
        
        if epoch >= self.warmup_epochs:
            return self.initial_weights.copy()
        
        # Compute warmup factor
        if self.warmup_type == 'linear':
            factor = min(epoch / self.warmup_epochs, 1.0)
        elif self.warmup_type == 'cosine':
            factor = 0.5 * (1 + math.cos(math.pi * (1 - min(epoch / self.warmup_epochs, 1.0))))
        elif self.warmup_type == 'exponential':
            factor = 2 ** (-5 * (1 - min(epoch / self.warmup_epochs, 1.0)))
        else:
            factor = min(epoch / self.warmup_epochs, 1.0)
        
        # Apply warmup factor to all weights
        weights = {key: value * factor for key, value in self.initial_weights.items()}
        
        return weights


class CompositeScheduler:
    """
    Composite scheduler combining multiple scheduling strategies
    """
    
    def __init__(
        self,
        schedulers: List[Callable[[int], Dict[str, float]]],
        weights: List[float] = None
    ):
        self.schedulers = schedulers
        self.weights = weights or [1.0 / len(schedulers) for _ in schedulers]
        
        assert len(self.schedulers) == len(self.weights), "Schedulers and weights must have same length"
    
    def __call__(self, epoch: int) -> Dict[str, float]:
        """Combine weights from multiple schedulers"""
        # Get weights from all schedulers
        all_weights = []
        for scheduler in self.schedulers:
            all_weights.append(scheduler(epoch))
        
        # Combine weights (weighted average)
        combined_weights = {}
        for key in all_weights[0].keys():
            combined_weights[key] = 0.0
            for i, weights in enumerate(all_weights):
                combined_weights[key] += weights.get(key, 0.0) * self.weights[i]
        
        return combined_weights
    
    def add_scheduler(self, scheduler: Callable[[int], Dict[str, float]], weight: float = 1.0):
        """Add a new scheduler"""
        self.schedulers.append(scheduler)
        self.weights.append(weight)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]


if __name__ == "__main__":
    # Test schedulers
    print("Testing Dynamic Weight Schedulers...")
    
    initial_weights = {
        'detection': 1.0,
        'feature': 0.5,
        'attention': 0.3,
        'relation': 0.2,
        'prototype': 0.2,
        'patch': 0.1,
        'consistency': 0.1
    }
    
    total_epochs = 100
    
    # Test Linear Scheduler
    print("\nTesting Linear Scheduler...")
    linear_scheduler = DynamicWeightScheduler(
        initial_weights=initial_weights,
        schedule_type='linear',
        total_epochs=total_epochs,
        warmup_epochs=5,
        cooldown_epochs=10
    )
    
    for epoch in [0, 5, 25, 50, 75, 95, 99]:
        weights = linear_scheduler(epoch)
        print(f"Epoch {epoch}: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")
    
    # Test Cosine Scheduler
    print("\nTesting Cosine Scheduler...")
    cosine_scheduler = DynamicWeightScheduler(
        initial_weights=initial_weights,
        schedule_type='cosine',
        total_epochs=total_epochs,
        warmup_epochs=5,
        cooldown_epochs=10
    )
    
    for epoch in [0, 5, 25, 50, 75, 95, 99]:
        weights = cosine_scheduler(epoch)
        print(f"Epoch {epoch}: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")
    
    # Test Lambda Scheduler
    print("\nTesting Lambda Scheduler...")
    
    # Define lambda functions
    def detection_lambda(epoch):
        return min(epoch / 20, 1.0)
    
    def feature_lambda(epoch):
        return max(0, 1.0 - epoch / 50)
    
    lambda_scheduler = LambdaScheduler(
        initial_weights=initial_weights,
        lambda_functions={
            'detection': detection_lambda,
            'feature': feature_lambda
        }
    )
    
    for epoch in [0, 10, 20, 30, 40, 50]:
        weights = lambda_scheduler(epoch)
        print(f"Epoch {epoch}: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")
    
    # Test Warmup Scheduler
    print("\nTesting Warmup Scheduler...")
    warmup_scheduler = WarmupScheduler(
        initial_weights=initial_weights,
        warmup_epochs=10,
        warmup_type='cosine'
    )
    
    for epoch in [0, 2, 5, 8, 10, 15]:
        weights = warmup_scheduler(epoch)
        print(f"Epoch {epoch}: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")
    
    print("Schedulers test completed!")
