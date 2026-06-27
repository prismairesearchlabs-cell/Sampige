"""
Utils module for SaMPiGe-Distill
"""

import sys
import os

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .metrics import (
    MetricCalculator,
    DetectionMetrics,
    compute_mAP,
    compute_precision_recall,
    compute_f1_score
)
from .visualization import (
    Visualizer,
    DetectionVisualizer,
    plot_predictions,
    plot_feature_maps,
    create_visualization_callback
)
from .checkpoint import (
    CheckpointManager,
    ModelCheckpoint,
    save_checkpoint,
    load_checkpoint
)
from .logger import (
    Logger,
    TensorBoardLogger,
    WandBLogger,
    CSVLogger,
    LogManager
)

__all__ = [
    # Metrics
    "MetricCalculator",
    "DetectionMetrics",
    "compute_mAP",
    "compute_precision_recall",
    "compute_f1_score",
    
    # Visualization
    "Visualizer",
    "DetectionVisualizer",
    "plot_predictions",
    "plot_feature_maps",
    "create_visualization_callback",
    
    # Checkpoint
    "CheckpointManager",
    "ModelCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    
    # Logger
    "Logger",
    "TensorBoardLogger",
    "WandBLogger",
    "CSVLogger",
    "LogManager"
]
