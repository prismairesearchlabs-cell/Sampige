"""
SaMPiGe-Distill Configuration Module
Centralized configuration for the entire pipeline
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import numpy as np


@dataclass
class PathConfig:
    """Path configurations for KITTI dataset and outputs"""
    # Base directories
    BASE_DIR: str = "/kaggle/input/kitti-dataset"
    
    # KITTI paths
    IMAGE_DIR: str = os.path.join(BASE_DIR, "training/image_2")
    LABEL_DIR: str = "/kaggle/input/kitti-dataset-yolo-format/labels"
    CLASS_FILE: str = "/kaggle/input/kitti-dataset-yolo-format/classes.json"
    
    # Working directories
    TRAIN_DIR: str = "/kaggle/working/train"
    VAL_DIR: str = "/kaggle/working/valid"
    CHECKPOINT_DIR: str = "/kaggle/working/checkpoints"
    OUTPUT_DIR: str = "/kaggle/working/output"
    
    # Local fallback paths for development
    LOCAL_BASE: str = "./data/kitti"
    LOCAL_IMAGES: str = os.path.join(LOCAL_BASE, "training/image_2")
    LOCAL_LABELS: str = os.path.join(LOCAL_BASE, "labels")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        os.makedirs(self.TRAIN_DIR, exist_ok=True)
        os.makedirs(self.VAL_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configurations"""
    # Teacher model (DINOv2)
    TEACHER_MODEL: str = "dinov2_vitb14"
    TEACHER_PRETRAINED: str = "facebook/dinov2-base"
    TEACHER_FREEZE: bool = True
    TEACHER_EMBED_DIM: int = 768
    
    # Student model
    STUDENT_BACKBONE: str = "yolov8s"  # or "yolov8n", "yolov8m"
    STUDENT_PRETRAINED: bool = True
    STUDENT_FEATURE_DIMS: Dict[str, int] = field(default_factory=lambda: {
        "P3": 256, "P4": 512, "P5": 1024
    })
    
    # Projection head
    PROJECTION_DIM: int = 768  # Match teacher embedding dimension
    PROJECTION_USE_BN: bool = True
    PROJECTION_ACTIVATION: str = "relu"
    
    # Prototype module
    NUM_PROTOTYPES: int = 100
    PROTOTYPE_DIM: int = 768
    PROTOTYPE_INIT: str = "kmeans"  # or "random"
    PROTOTYPE_MEMORY_BANK_SIZE: int = 1000
    
    # Detection head
    NUM_CLASSES: int = 8  # KITTI classes
    DETECTION_HEAD: str = "yolov8"  # or "custom"
    
    # Feature hooks
    HOOK_LAYERS: List[str] = field(default_factory=lambda: ["P3", "P4", "P5"])


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Batch and epochs
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 4
    EPOCHS: int = 100
    
    # Optimizer
    OPTIMIZER: str = "adamw"
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    MOMENTUM: float = 0.9
    
    # Scheduler
    SCHEDULER: str = "cosine"  # or "step", "linear"
    WARMUP_EPOCHS: int = 5
    COOLDOWN_EPOCHS: int = 10
    
    # Loss weights (dynamic scheduler will override these)
    DETECTION_WEIGHT: float = 1.0
    FEATURE_WEIGHT: float = 0.5
    ATTENTION_WEIGHT: float = 0.3
    RELATION_WEIGHT: float = 0.2
    PROTOTYPE_WEIGHT: float = 0.2
    PATCH_WEIGHT: float = 0.1
    CONSISTENCY_WEIGHT: float = 0.1
    
    # Detection loss
    BOX_LOSS: str = "ciou"  # or "diou", "siou", "mse"
    CLS_LOSS: str = "bce"  # or "ce"
    DFL_LOSS: bool = True  # Distribution Focal Loss
    
    # Knowledge distillation
    KD_TEMPERATURE: float = 1.0
    KD_LOSS: str = "mse"  # or "cosine", "kl"
    
    # Regularization
    DROPOUT: float = 0.1
    LABEL_SMOOTHING: float = 0.1
    
    # Gradient clipping
    GRAD_CLIP: float = 1.0
    GRAD_CLIP_MODE: str = "norm"  # or "value"


@dataclass
class DataConfig:
    """Data loading and augmentation configurations"""
    # Image parameters
    IMAGE_SIZE: int = 640
    TEACHER_IMAGE_SIZE: int = 518
    
    # Augmentation
    HORIZONTAL_FLIP: bool = True
    VERTICAL_FLIP: bool = False
    ROTATION: float = 0.0  # degrees
    BRIGHTNESS: float = 0.2
    CONTRAST: float = 0.2
    SATURATION: float = 0.2
    HUE: float = 0.1
    
    # Color jitter
    COLOR_JITTER_PROB: float = 0.5
    
    # Mosaic augmentation
    MOSAIC_PROB: float = 0.5
    MIXUP_PROB: float = 0.1
    
    # Normalization
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Dataset split
    TRAIN_SPLIT: float = 0.9
    VAL_SPLIT: float = 0.1
    
    # Batching
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True


@dataclass
class ValidationConfig:
    """Validation and testing configurations"""
    VAL_BATCH_SIZE: int = 8
    VAL_INTERVAL: int = 1  # Validate every N epochs
    
    # Metrics
    METRIC_THRESHOLDS: List[float] = field(default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    MAP_TYPE: str = "coco"  # or "voc"
    
    # NMS
    NMS_THRESHOLD: float = 0.45
    CONFIDENCE_THRESHOLD: float = 0.25
    
    # Visualization
    VISUALIZE: bool = True
    VISUALIZE_INTERVAL: int = 5  # Every N validation steps
    VISUALIZE_NUM_IMAGES: int = 4


@dataclass
class LoggingConfig:
    """Logging and checkpointing configurations"""
    LOG_DIR: str = "./logs"
    LOG_INTERVAL: int = 10  # Log every N batches
    
    # TensorBoard
    TENSORBOARD: bool = True
    TENSORBOARD_PORT: int = 6006
    
    # Weights & Biases
    WANDB: bool = False
    WANDB_PROJECT: str = "sampige-distill"
    WANDB_ENTITY: str = ""
    
    # Checkpointing
    SAVE_CHECKPOINT: bool = True
    SAVE_INTERVAL: int = 1  # Save every N epochs
    KEEP_LAST_N: int = 5
    SAVE_BEST: bool = True
    BEST_METRIC: str = "mAP50"  # or "mAP50-95", "val_loss"
    
    # Model saving
    SAVE_FORMAT: str = "pytorch"  # or "onnx", "torchscript"
    
    # Verbosity
    VERBOSE: bool = True


@dataclass
class SystemConfig:
    """System and hardware configurations"""
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    ACCELERATOR: str = "gpu" if torch.cuda.is_available() else "cpu"
    NUM_DEVICES: int = torch.cuda.device_count() if torch.cuda.is_available() else 1
    PRECISION: str = "16-mixed"  # or "32", "bf16"
    
    # Distributed training
    STRATEGY: str = "ddp" if NUM_DEVICES > 1 else "auto"
    SYNC_BATCHNORM: bool = True if NUM_DEVICES > 1 else False
    
    # Memory
    MAX_MEMORY: Dict[str, Any] = field(default_factory=lambda: {
        "gpu": "16GB", "cpu": "32GB"
    })
    
    # Seeds
    SEED: int = 42
    DETERMINISTIC: bool = True


@dataclass
class Config:
    """Main configuration class combining all configurations"""
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Initialize and validate configurations"""
        # Set device
        self.system.DEVICE = torch.device(self.system.DEVICE)
        
        # Create log directory
        os.makedirs(self.logging.LOG_DIR, exist_ok=True)
        
        # Validate configurations
        self._validate_configs()
    
    def _validate_configs(self):
        """Validate configuration parameters"""
        # Check if paths exist or create them
        if not os.path.exists(self.path.CHECKPOINT_DIR):
            os.makedirs(self.path.CHECKPOINT_DIR, exist_ok=True)
        
        if not os.path.exists(self.path.OUTPUT_DIR):
            os.makedirs(self.path.OUTPUT_DIR, exist_ok=True)
        
        # Validate image sizes
        assert self.data.IMAGE_SIZE > 0, "Image size must be positive"
        assert self.data.TEACHER_IMAGE_SIZE > 0, "Teacher image size must be positive"
        
        # Validate batch sizes
        assert self.training.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.validation.VAL_BATCH_SIZE > 0, "Validation batch size must be positive"
        
        # Validate learning rate
        assert self.training.LEARNING_RATE > 0, "Learning rate must be positive"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "path": self.path.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "validation": self.validation.__dict__,
            "logging": self.logging.__dict__,
            "system": self.system.__dict__
        }
    
    def save(self, path: str = "./config.yaml"):
        """Save configuration to YAML file"""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str = "./config.yaml") -> 'Config':
        """Load configuration from YAML file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclasses
        config_dict = {k: v for k, v in config_dict.items()}
        return cls(**config_dict)


# Global configuration instance
config = Config()

if __name__ == "__main__":
    # Test configuration
    print("Configuration loaded successfully!")
    print(f"Device: {config.system.DEVICE}")
    print(f"Model: {config.model.TEACHER_MODEL}")
    print(f"Batch size: {config.training.BATCH_SIZE}")
    print(f"Image size: {config.data.IMAGE_SIZE}")
