"""
Utils module for SaMPiGe-Distill
"""

import sys
import os

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import metrics (doesn't require OpenCV)
from .metrics import (
    MetricCalculator,
    DetectionMetrics,
    compute_mAP,
    compute_precision_recall,
    compute_f1_score
)

# Import checkpoint (doesn't require OpenCV)
from .checkpoint import (
    CheckpointManager,
    ModelCheckpoint,
    save_checkpoint,
    load_checkpoint
)

# Import logger (doesn't require OpenCV)
from .logger import (
    Logger,
    TensorBoardLogger,
    WandBLogger,
    CSVLogger,
    LogManager
)

# Try to import visualization (requires OpenCV)
try:
    from .visualization import (
        Visualizer,
        DetectionVisualizer,
        plot_predictions,
        plot_feature_maps,
        create_visualization_callback
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    Visualizer = None
    DetectionVisualizer = None
    plot_predictions = None
    plot_feature_maps = None
    create_visualization_callback = None

__all__ = [
    # Metrics
    "MetricCalculator",
    "DetectionMetrics",
    "compute_mAP",
    "compute_precision_recall",
    "compute_f1_score",
    
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
    "LogManager",
    
    # Visualization (optional)
    "VISUALIZATION_AVAILABLE",
    "Visualizer",
    "DetectionVisualizer",
    "plot_predictions",
    "plot_feature_maps",
    "create_visualization_callback"
]
