"""
Main Training Script for SaMPiGe-Distill
PyTorch Lightning-based training pipeline
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import warnings
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from datasets import KITTIDataModule, collate_fn
from models import (
    DINOv2Teacher, TeacherWrapper,
    YOLOStudent, StudentBackbone,
    KnowledgeDistiller, MultiTaskDistiller
)
from losses import DetectionLoss, KnowledgeDistillationLoss
from scheduler import DynamicWeightScheduler, LambdaScheduler
from utils import (
    DetectionMetrics, KnowledgeDistillationMetrics,
    DetectionVisualizer, create_visualization_callback,
    CheckpointManager, LogManager
)


class SaMPiGeDistillModule(pl.LightningModule):
    """
    PyTorch Lightning module for SaMPiGe-Distill
    
    Combines:
    - Knowledge distillation from DINOv2 teacher
    - YOLO-based object detection
    - Multi-task learning with dynamic loss weighting
    """
    
    def __init__(
        self,
        teacher: Optional[DINOv2Teacher] = None,
        student: Optional[YOLOStudent] = None,
        num_classes: int = config.model.NUM_CLASSES,
        loss_weights: Optional[Dict[str, float]] = None,
        scheduler_type: str = config.training.SCHEDULER
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize teacher
        if teacher is None:
            try:
                self.teacher = TeacherWrapper()
            except Exception as e:
                warnings.warn(f"Failed to initialize DINOv2 teacher: {e}")
                self.teacher = None
        else:
            self.teacher = TeacherWrapper(teacher)
        
        # Initialize student
        if student is None:
            self.student = YOLOStudent(num_classes=num_classes)
        else:
            self.student = student
        
        # Initialize distiller
        self.distiller = KnowledgeDistiller(
            teacher=self.teacher.model if self.teacher else None,
            student=self.student,
            loss_weights=loss_weights
        )
        
        # Set detection loss function
        self.detection_loss_fn = DetectionLoss(
            num_classes=num_classes,
            box_loss=config.training.BOX_LOSS,
            cls_loss=config.training.CLS_LOSS,
            dfl_loss=config.training.DFL_LOSS
        )
        self.distiller.set_detection_loss_fn(self.detection_loss_fn)
        
        # Initialize scheduler
        self._init_scheduler(scheduler_type)
        
        # Metrics
        self.train_metrics = DetectionMetrics(num_classes=num_classes)
        self.val_metrics = DetectionMetrics(num_classes=num_classes)
        self.kd_metrics = KnowledgeDistillationMetrics()
        
        # Visualization
        self.visualizer = None
        
        # Loss tracking
        self.loss_history = {
            'train': [],
            'val': []
        }
        
        print("SaMPiGeDistillModule initialized")
        print(f"Teacher: {'Available' if self.teacher else 'Not available'}")
        print(f"Student: {type(self.student).__name__}")
        print(f"Detection loss: {config.training.BOX_LOSS}/{config.training.CLS_LOSS}")
    
    def _init_scheduler(self, scheduler_type: str):
        """Initialize loss weight scheduler"""
        initial_weights = self.distiller.get_loss_weights()
        
        if scheduler_type == 'dynamic':
            self.scheduler = DynamicWeightScheduler(
                initial_weights=initial_weights,
                schedule_type='cosine',
                total_epochs=config.training.EPOCHS,
                warmup_epochs=config.training.WARMUP_EPOCHS,
                cooldown_epochs=config.training.COOLDOWN_EPOCHS
            )
        elif scheduler_type == 'lambda':
            # Define lambda functions for each weight
            lambda_functions = {
                'detection': lambda epoch: min(epoch / 20, 1.0),
                'feature': lambda epoch: max(0, 1.0 - epoch / 50),
                'attention': lambda epoch: max(0, 1.0 - epoch / 60),
                'relation': lambda epoch: max(0, 1.0 - epoch / 70),
                'prototype': lambda epoch: max(0, 1.0 - epoch / 80),
                'patch': lambda epoch: max(0, 1.0 - epoch / 90),
                'consistency': lambda epoch: 0.1
            }
            self.scheduler = LambdaScheduler(
                initial_weights=initial_weights,
                lambda_functions=lambda_functions
            )
        else:
            # Static scheduler
            self.scheduler = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        if self.teacher:
            # Teacher-student forward
            outputs = self.distiller(x)
        else:
            # Student-only forward
            outputs = self.student(x)
        
        return outputs
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step"""
        images = batch['image']
        labels = batch['labels']
        label_mask = batch.get('label_mask', None)
        
        # Forward pass
        if self.teacher:
            outputs = self.distiller(images, labels)
        else:
            outputs = self.student(images)
            # Compute detection loss manually
            detections = outputs.get('detections')
            detection_loss = self.detection_loss_fn(detections, labels, label_mask)
            outputs['detection_loss'] = detection_loss
            outputs['total_loss'] = detection_loss
        
        # Update loss weights based on scheduler
        if self.scheduler:
            weights = self.scheduler(self.current_epoch)
            self.distiller.update_loss_weights(weights)
        
        # Log metrics
        self._log_metrics(outputs, 'train')
        
        # Update metrics
        if 'detections' in outputs and labels is not None:
            # Convert outputs to numpy for metrics
            detections = outputs['detections'].detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            
            # Filter valid labels
            if label_mask is not None:
                valid_mask = label_mask.detach().cpu().numpy()
                targets = [t[valid_mask[i]] for i, t in enumerate(targets)]
            
            self.train_metrics.update(detections, targets)
        
        # Return loss for backpropagation
        return {
            'loss': outputs['total_loss'],
            **{f'loss_{k}': v for k, v in outputs.items() if 'loss' in k}
        }
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step"""
        images = batch['image']
        labels = batch['labels']
        label_mask = batch.get('label_mask', None)
        
        # Forward pass (no gradient)
        with torch.no_grad():
            if self.teacher:
                outputs = self.distiller(images, labels)
            else:
                outputs = self.student(images)
                detections = outputs.get('detections')
                detection_loss = self.detection_loss_fn(detections, labels, label_mask)
                outputs['detection_loss'] = detection_loss
                outputs['total_loss'] = detection_loss
        
        # Log metrics
        self._log_metrics(outputs, 'val')
        
        # Update metrics
        if 'detections' in outputs and labels is not None:
            detections = outputs['detections'].detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            
            if label_mask is not None:
                valid_mask = label_mask.detach().cpu().numpy()
                targets = [t[valid_mask[i]] for i, t in enumerate(targets)]
            
            self.val_metrics.update(detections, targets)
        
        return outputs
    
    def on_validation_epoch_end(self):
        """Called at end of validation epoch"""
        # Compute and log metrics
        train_results = self.train_metrics.compute()
        val_results = self.val_metrics.compute()
        
        # Log metrics
        for key, value in train_results.items():
            self.log(f"train_{key}", value, prog_bar=False)
        
        for key, value in val_results.items():
            self.log(f"val_{key}", value, prog_bar=True)
        
        # Reset metrics
        self.train_metrics.reset()
        self.val_metrics.reset()
        
        # Visualize some predictions
        if config.validation.VISUALIZE and self.visualizer:
            self._visualize_predictions()
    
    def _log_metrics(self, outputs: Dict[str, torch.Tensor], prefix: str = 'train'):
        """Log metrics from outputs"""
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(f"{prefix}_{key}", value.item(), prog_bar=(key == 'total_loss'))
            elif isinstance(value, (int, float)):
                self.log(f"{prefix}_{key}", value, prog_bar=(key == 'total_loss'))
    
    def _visualize_predictions(self):
        """Visualize predictions on validation images"""
        # This would be implemented with actual data
        pass
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Optimizer
        if config.training.OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=config.training.LEARNING_RATE,
                weight_decay=config.training.WEIGHT_DECAY
            )
        elif config.training.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=config.training.LEARNING_RATE,
                momentum=config.training.MOMENTUM,
                weight_decay=config.training.WEIGHT_DECAY
            )
        else:
            optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=config.training.LEARNING_RATE,
                weight_decay=config.training.WEIGHT_DECAY
            )
        
        # Learning rate scheduler
        if config.training.SCHEDULER == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.EPOCHS,
                eta_min=1e-6
            )
        elif config.training.SCHEDULER == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.EPOCHS // 3,
                gamma=0.1
            )
        elif config.training.SCHEDULER == 'linear':
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=config.training.EPOCHS
            )
        else:
            lr_scheduler = None
        
        # Return optimizer and scheduler
        if lr_scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        else:
            return optimizer


class SaMPiGeDistillTrainer:
    """
    Trainer class for SaMPiGe-Distill
    
    Handles:
    - Data loading
    - Model initialization
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        config: Optional[config.Config] = None
    ):
        self.config = config or config.config
        
        # Initialize components
        self.datamodule = None
        self.model = None
        self.trainer = None
        self.logger = None
        self.checkpoint_callback = None
        self.visualization_callback = None
        
        # Setup
        self._setup()
    
    def _setup(self):
        """Setup training components"""
        print("Setting up SaMPiGe-Distill Trainer...")
        
        # Set device
        self.device = torch.device(self.config.system.DEVICE)
        print(f"Device: {self.device}")
        
        # Initialize data module
        self._init_datamodule()
        
        # Initialize model
        self._init_model()
        
        # Initialize callbacks
        self._init_callbacks()
        
        # Initialize logger
        self._init_logger()
        
        # Initialize trainer
        self._init_trainer()
        
        print("Setup complete!")
    
    def _init_datamodule(self):
        """Initialize data module"""
        print("Initializing DataModule...")
        
        self.datamodule = KITTIDataModule(
            batch_size=self.config.training.BATCH_SIZE,
            val_batch_size=self.config.validation.VAL_BATCH_SIZE,
            image_size=self.config.data.IMAGE_SIZE,
            num_workers=self.config.training.NUM_WORKERS
        )
        
        # Setup data
        self.datamodule.setup()
        
        print(f"DataModule initialized: {len(self.datamodule.train_dataset)} train samples")
    
    def _init_model(self):
        """Initialize model"""
        print("Initializing Model...")
        
        # Get class names from datamodule
        class_names = self.datamodule.get_class_names()
        num_classes = self.datamodule.get_num_classes()
        
        # Initialize teacher (if available)
        teacher = None
        try:
            teacher = DINOv2Teacher(
                model_name=self.config.model.TEACHER_MODEL,
                pretrained=True,
                freeze=True
            )
        except Exception as e:
            warnings.warn(f"Teacher initialization failed: {e}")
        
        # Initialize student
        student = YOLOStudent(
            num_classes=num_classes,
            detection_head=self.config.model.DETECTION_HEAD
        )
        
        # Initialize module
        self.model = SaMPiGeDistillModule(
            teacher=teacher,
            student=student,
            num_classes=num_classes,
            scheduler_type=self.config.training.SCHEDULER
        )
        
        print("Model initialized")
    
    def _init_callbacks(self):
        """Initialize training callbacks"""
        print("Initializing Callbacks...")
        
        # Model checkpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.path.CHECKPOINT_DIR,
            filename='sampige-{epoch:04d}-{val_mAP50:.4f}',
            save_top_k=self.config.logging.KEEP_LAST_N,
            monitor='val_mAP50' if self.config.logging.BEST_METRIC == 'mAP50' else 'val_loss',
            mode='max' if 'mAP' in self.config.logging.BEST_METRIC else 'min',
            save_last=True,
            save_weights_only=False
        )
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_mAP50' if self.config.logging.BEST_METRIC == 'mAP50' else 'val_loss',
            patience=10,
            mode='max' if 'mAP' in self.config.logging.BEST_METRIC else 'min'
        )
        
        self.callbacks = [
            self.checkpoint_callback,
            lr_monitor,
            early_stopping
        ]
        
        # Visualization callback
        if self.config.validation.VISUALIZE:
            class_names = self.datamodule.get_class_names()
            self.visualization_callback = create_visualization_callback(
                self.datamodule,
                class_names,
                num_images=self.config.validation.VISUALIZE_NUM_IMAGES,
                interval=self.config.validation.VISUALIZE_INTERVAL,
                output_dir=self.config.path.OUTPUT_DIR
            )
            self.callbacks.append(self.visualization_callback)
        
        print("Callbacks initialized")
    
    def _init_logger(self):
        """Initialize logger"""
        print("Initializing Logger...")
        
        # TensorBoard logger
        if self.config.logging.TENSORBOARD:
            tensorboard_logger = TensorBoardLogger(
                self.config.logging.LOG_DIR,
                name='sampige_distill'
            )
        else:
            tensorboard_logger = None
        
        # CSV logger
        csv_logger = CSVLogger(
            self.config.logging.LOG_DIR,
            name='sampige_distill'
        )
        
        self.logger = [tensorboard_logger, csv_logger] if tensorboard_logger else [csv_logger]
        
        print("Logger initialized")
    
    def _init_trainer(self):
        """Initialize PyTorch Lightning trainer"""
        print("Initializing Trainer...")
        
        self.trainer = pl.Trainer(
            accelerator=self.config.system.ACCELERATOR,
            devices=self.config.system.NUM_DEVICES,
            strategy=self.config.system.STRATEGY,
            max_epochs=self.config.training.EPOCHS,
            precision=self.config.system.PRECISION,
            callbacks=self.callbacks,
            logger=self.logger[0] if len(self.logger) == 1 else self.logger,
            enable_progress_bar=True,
            enable_model_summary=True,
            gradient_clip_val=self.config.training.GRAD_CLIP,
            gradient_clip_algorithm=self.config.training.GRAD_CLIP_MODE
        )
        
        print("Trainer initialized")
    
    def train(self):
        """Start training"""
        print("Starting training...")
        
        # Train the model
        self.trainer.fit(
            self.model,
            datamodule=self.datamodule
        )
        
        print("Training completed!")
    
    def validate(self):
        """Run validation"""
        print("Running validation...")
        
        results = self.trainer.validate(
            self.model,
            datamodule=self.datamodule
        )
        
        print("Validation results:")
        for key, value in results[0].items():
            print(f"  {key}: {value}")
        
        return results
    
    def test(self):
        """Run testing"""
        print("Running testing...")
        
        results = self.trainer.test(
            self.model,
            datamodule=self.datamodule
        )
        
        print("Test results:")
        for key, value in results[0].items():
            print(f"  {key}: {value}")
        
        return results
    
    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions on new images"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
        
        return outputs


def main():
    """Main training function"""
    print("=" * 60)
    print("SaMPiGe-Distill Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = SaMPiGeDistillTrainer()
    
    # Train
    trainer.train()
    
    # Validate
    trainer.validate()
    
    # Test
    trainer.test()
    
    print("Training pipeline completed!")


if __name__ == "__main__":
    main()
