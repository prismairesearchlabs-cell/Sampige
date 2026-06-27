"""
SaMPiGe-Distill Package
Multi-Task Knowledge Distillation for Object Detection
"""

__version__ = "0.1.0"

# Import configuration
from .config import config

# Import datasets
from .datasets import KITTIDataset, KITTIDataModule, collate_fn

# Import models
from .models import (
    DINOv2Teacher, TeacherWrapper,
    YOLOStudent, StudentBackbone,
    ProjectionHead, FeatureProjection,
    PrototypeModule, PrototypeMemoryBank,
    FeatureHooks, HookManager,
    KnowledgeDistiller, MultiTaskDistiller
)

# Import losses
from .losses import (
    DetectionLoss, CIoULoss, DIoULoss, SIoULoss,
    DistributionFocalLoss, KnowledgeDistillationLoss
)

# Import schedulers
from .scheduler import (
    DynamicWeightScheduler, LambdaScheduler,
    LossBalancingScheduler, WarmupScheduler, CompositeScheduler
)

# Import utils
from .utils import (
    DetectionMetrics, KnowledgeDistillationMetrics,
    DetectionVisualizer, FeatureMapVisualizer,
    CheckpointManager, LogManager, CSVLogger, JSONLogger
)

__all__ = [
    # Config
    'config',
    
    # Datasets
    'KITTIDataset', 'KITTIDataModule', 'collate_fn',
    
    # Models
    'DINOv2Teacher', 'TeacherWrapper',
    'YOLOStudent', 'StudentBackbone',
    'ProjectionHead', 'FeatureProjection',
    'PrototypeModule', 'PrototypeMemoryBank',
    'FeatureHooks', 'HookManager',
    'KnowledgeDistiller', 'MultiTaskDistiller',
    
    # Losses
    'DetectionLoss', 'CIoULoss', 'DIoULoss', 'SIoULoss',
    'DistributionFocalLoss', 'KnowledgeDistillationLoss',
    
    # Schedulers
    'DynamicWeightScheduler', 'LambdaScheduler',
    'LossBalancingScheduler', 'WarmupScheduler', 'CompositeScheduler',
    
    # Utils
    'DetectionMetrics', 'KnowledgeDistillationMetrics',
    'DetectionVisualizer', 'FeatureMapVisualizer',
    'CheckpointManager', 'LogManager', 'CSVLogger', 'JSONLogger'
]
