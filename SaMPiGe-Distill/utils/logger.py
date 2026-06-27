"""
Logging Utilities
For tracking training progress and metrics
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import os
import json
import csv
from datetime import datetime
import warnings

from config import config


class Logger:
    """
    Base logger class
    """
    
    def __init__(self):
        self.logs = []
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics"""
        log_entry = {'step': step, **metrics}
        self.logs.append(log_entry)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs"""
        return self.logs.copy()
    
    def save(self, path: str):
        """Save logs to file"""
        pass
    
    def close(self):
        """Close logger"""
        pass


class TensorBoardLogger(Logger):
    """
    TensorBoard logger
    """
    
    def __init__(self, log_dir: str = config.logging.LOG_DIR):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            warnings.warn("TensorBoard not available")
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics to TensorBoard"""
        super().log(metrics, step)
        
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int = 0):
        """Log image to TensorBoard"""
        if self.writer:
            self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag: str, figure, step: int = 0):
        """Log matplotlib figure to TensorBoard"""
        if self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class WandBLogger(Logger):
    """
    Weights & Biases logger
    """
    
    def __init__(
        self,
        project: str = config.logging.WANDB_PROJECT,
        entity: str = config.logging.WANDB_ENTITY,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.run = None
        
        if config.logging.WANDB:
            try:
                import wandb
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    config=self.config,
                    reinit=True
                )
            except ImportError:
                warnings.warn("Weights & Biases not available")
            except Exception as e:
                warnings.warn(f"WandB initialization failed: {e}")
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics to WandB"""
        super().log(metrics, step)
        
        if self.run:
            self.run.log(metrics, step=step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int = 0):
        """Log image to WandB"""
        if self.run:
            self.run.log({tag: self.run.Image(image)}, step=step)
    
    def log_figure(self, tag: str, figure, step: int = 0):
        """Log matplotlib figure to WandB"""
        if self.run:
            self.run.log({tag: figure}, step=step)
    
    def close(self):
        """Close WandB run"""
        if self.run:
            self.run.finish()


class CSVLogger(Logger):
    """
    CSV logger for simple logging
    """
    
    def __init__(self, log_dir: str = config.logging.LOG_DIR, filename: str = "metrics.csv"):
        super().__init__()
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = ['step', 'timestamp']
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file and write header
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics to CSV"""
        super().log(metrics, step)
        
        # Update fieldnames if new metrics
        for key in metrics.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        
        # Write to file
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            
            row = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                **metrics
            }
            writer.writerow(row)
    
    def close(self):
        """Close CSV logger"""
        pass


class JSONLogger(Logger):
    """
    JSON logger for structured logging
    """
    
    def __init__(self, log_dir: str = config.logging.LOG_DIR, filename: str = "metrics.json"):
        super().__init__()
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(log_dir, filename)
        
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics to JSON"""
        super().log(metrics, step)
        
        # Load existing logs
        logs = []
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                logs = json.load(f)
        
        # Add new log
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        logs.append(log_entry)
        
        # Save back to file
        with open(self.filepath, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def close(self):
        """Close JSON logger"""
        pass


class LogManager:
    """
    Manages multiple loggers
    
    Args:
        loggers: List of logger instances
    """
    
    def __init__(self, loggers: Optional[List[Logger]] = None):
        self.loggers = loggers or []
        
        # Add default loggers if none provided
        if not self.loggers:
            if config.logging.TENSORBOARD:
                self.loggers.append(TensorBoardLogger())
            if config.logging.WANDB:
                self.loggers.append(WandBLogger())
            self.loggers.append(CSVLogger())
    
    def add_logger(self, logger: Logger):
        """Add a logger"""
        self.loggers.append(logger)
    
    def log(self, metrics: Dict[str, float], step: int = 0):
        """Log metrics to all loggers"""
        for logger in self.loggers:
            logger.log(metrics, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int = 0):
        """Log image to all loggers that support it"""
        for logger in self.loggers:
            if hasattr(logger, 'log_image'):
                logger.log_image(tag, image, step)
    
    def log_figure(self, tag: str, figure, step: int = 0):
        """Log figure to all loggers that support it"""
        for logger in self.loggers:
            if hasattr(logger, 'log_figure'):
                logger.log_figure(tag, figure, step)
    
    def close(self):
        """Close all loggers"""
        for logger in self.loggers:
            logger.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class TrainingLogger:
    """
    Logger specifically for training progress
    """
    
    def __init__(self, log_dir: str = config.logging.LOG_DIR):
        self.log_dir = log_dir
        self.log_manager = LogManager()
        self.training_logs = []
        self.validation_logs = []
        
        # Create subdirectories
        self.train_dir = os.path.join(log_dir, "train")
        self.val_dir = os.path.join(log_dir, "val")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
    
    def log_training(self, metrics: Dict[str, float], step: int = 0, epoch: int = 0):
        """Log training metrics"""
        # Add epoch to metrics
        metrics_with_epoch = {
            f"train/{key}": value for key, value in metrics.items()
        }
        metrics_with_epoch['epoch'] = epoch
        
        self.log_manager.log(metrics_with_epoch, step)
        self.training_logs.append(metrics_with_epoch)
    
    def log_validation(self, metrics: Dict[str, float], step: int = 0, epoch: int = 0):
        """Log validation metrics"""
        # Add epoch to metrics
        metrics_with_epoch = {
            f"val/{key}": value for key, value in metrics.items()
        }
        metrics_with_epoch['epoch'] = epoch
        
        self.log_manager.log(metrics_with_epoch, step)
        self.validation_logs.append(metrics_with_epoch)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int = 0):
        """Log images"""
        self.log_manager.log_image(tag, images, step)
    
    def get_best_metric(self, metric_name: str = "mAP50") -> float:
        """Get best validation metric"""
        best_value = 0.0
        
        for log in self.validation_logs:
            if metric_name in log:
                value = log[metric_name]
                if value > best_value:
                    best_value = value
        
        return best_value
    
    def close(self):
        """Close logger"""
        self.log_manager.close()


class PyTorchLightningLogger:
    """
    Logger for PyTorch Lightning
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_manager = LogManager()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at end of training batch"""
        # Log training metrics
        if hasattr(trainer, 'current_epoch'):
            epoch = trainer.current_epoch
        else:
            epoch = 0
        
        step = trainer.global_step
        
        # Get metrics from outputs
        metrics = self._extract_metrics(outputs, 'train')
        
        self.log_manager.log(metrics, step)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called at end of validation batch"""
        if hasattr(trainer, 'current_epoch'):
            epoch = trainer.current_epoch
        else:
            epoch = 0
        
        step = trainer.global_step
        
        # Get metrics from outputs
        metrics = self._extract_metrics(outputs, 'val')
        
        self.log_manager.log(metrics, step)
    
    def _extract_metrics(self, outputs: Dict[str, Any], prefix: str = '') -> Dict[str, float]:
        """Extract metrics from outputs"""
        metrics = {}
        
        for key, value in outputs.items():
            if isinstance(value, (int, float)):
                metric_name = f"{prefix}/{key}" if prefix else key
                metrics[metric_name] = float(value)
        
        return metrics
    
    def close(self):
        """Close logger"""
        self.log_manager.close()


if __name__ == "__main__":
    # Test loggers
    print("Testing Loggers...")
    
    # Create dummy metrics
    metrics = {
        'loss': 0.5,
        'accuracy': 0.9,
        'mAP50': 0.75
    }
    
    # Test CSVLogger
    print("\nTesting CSVLogger...")
    csv_logger = CSVLogger(log_dir="./test_logs", filename="test.csv")
    csv_logger.log(metrics, step=0)
    csv_logger.log({**metrics, 'loss': 0.4}, step=1)
    csv_logger.close()
    
    # Test JSONLogger
    print("\nTesting JSONLogger...")
    json_logger = JSONLogger(log_dir="./test_logs", filename="test.json")
    json_logger.log(metrics, step=0)
    json_logger.log({**metrics, 'loss': 0.4}, step=1)
    json_logger.close()
    
    # Test LogManager
    print("\nTesting LogManager...")
    log_manager = LogManager([
        CSVLogger(log_dir="./test_logs", filename="manager.csv")
    ])
    log_manager.log(metrics, step=0)
    log_manager.close()
    
    # Clean up
    import shutil
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
    
    print("Loggers test completed!")
