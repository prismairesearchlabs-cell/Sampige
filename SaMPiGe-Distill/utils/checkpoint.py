"""
Checkpoint Management
For saving and loading model checkpoints
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import os
import json
import glob
from datetime import datetime
import warnings

from config import config


class CheckpointManager:
    """
    Manages model checkpoints during training
    
    Handles:
    - Saving checkpoints at intervals
    - Loading checkpoints
    - Keeping best N checkpoints
    - Resuming training
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N epochs
        keep_last_n: Keep last N checkpoints
        save_best: Whether to save best checkpoint
        best_metric: Metric to use for best checkpoint
        prefix: Prefix for checkpoint filenames
    """
    
    def __init__(
        self,
        checkpoint_dir: str = config.path.CHECKPOINT_DIR,
        save_interval: int = config.logging.SAVE_INTERVAL,
        keep_last_n: int = config.logging.KEEP_LAST_N,
        save_best: bool = config.logging.SAVE_BEST,
        best_metric: str = config.logging.BEST_METRIC,
        prefix: str = "model"
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_metric = best_metric
        self.prefix = prefix
        
        # Create directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # State
        self.best_metric_value = float('-inf') if 'mAP' in best_metric else float('inf')
        self.epoch = 0
        self.checkpoints = []
        
        print(f"CheckpointManager: dir={checkpoint_dir}, interval={save_interval}, keep={keep_last_n}")
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        Save a checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
        """
        self.epoch = epoch
        
        # Create checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Create filename
        filename = self._create_filename(epoch, is_best)
        
        # Save checkpoint
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint_data, path)
        
        # Add to checkpoints list
        self.checkpoints.append({
            'path': path,
            'epoch': epoch,
            'metrics': metrics or {},
            'is_best': is_best
        })
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        # Update best metric
        if is_best and metrics and self.best_metric in metrics:
            self.best_metric_value = metrics[self.best_metric]
        
        print(f"Checkpoint saved: {filename}")
        return path
    
    def _create_filename(self, epoch: int, is_best: bool = False) -> str:
        """Create checkpoint filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            return f"{self.prefix}_best_{self.best_metric}_{self.best_metric_value:.4f}.pt"
        else:
            return f"{self.prefix}_epoch_{epoch:04d}_{timestamp}.pt"
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to maintain keep_last_n"""
        if len(self.checkpoints) <= self.keep_last_n:
            return
        
        # Sort by epoch
        self.checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            oldest = self.checkpoints.pop(0)
            if os.path.exists(oldest['path']):
                os.remove(oldest['path'])
                print(f"Removed old checkpoint: {oldest['path']}")
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        path: Optional[str] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            path: Path to checkpoint file (None for latest)
            strict: Whether to strictly enforce matching keys
        
        Returns:
            Checkpoint data
        """
        if path is None:
            path = self.get_latest_checkpoint()
        
        if path is None:
            warnings.warn("No checkpoint found")
            return {}
        
        # Load checkpoint
        checkpoint_data = torch.load(path, map_location='cpu')
        
        # Load model
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
        
        # Load optimizer
        if optimizer and checkpoint_data.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler and checkpoint_data.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Update state
        self.epoch = checkpoint_data.get('epoch', 0)
        if self.best_metric in checkpoint_data.get('metrics', {}):
            self.best_metric_value = checkpoint_data['metrics'][self.best_metric]
        
        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch: {self.epoch}")
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoints = self._get_checkpoint_files()
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        checkpoints = self._get_checkpoint_files()
        if not checkpoints:
            return None
        
        # Find best checkpoint by metric
        best_path = None
        best_value = float('-inf') if 'mAP' in self.best_metric else float('inf')
        
        for path in checkpoints:
            try:
                checkpoint_data = torch.load(path, map_location='cpu')
                metrics = checkpoint_data.get('metrics', {})
                
                if self.best_metric in metrics:
                    value = metrics[self.best_metric]
                    
                    if ('mAP' in self.best_metric and value > best_value) or \
                       ('loss' in self.best_metric and value < best_value):
                        best_value = value
                        best_path = path
            except Exception as e:
                warnings.warn(f"Error loading checkpoint {path}: {e}")
                continue
        
        return best_path
    
    def _get_checkpoint_files(self) -> List[str]:
        """Get list of checkpoint files"""
        pattern = os.path.join(self.checkpoint_dir, f"{self.prefix}_*.pt")
        return glob.glob(pattern)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for path in self._get_checkpoint_files():
            try:
                checkpoint_data = torch.load(path, map_location='cpu')
                checkpoints.append({
                    'path': path,
                    'epoch': checkpoint_data.get('epoch', 0),
                    'timestamp': checkpoint_data.get('timestamp', ''),
                    'metrics': checkpoint_data.get('metrics', {}),
                    'size': os.path.getsize(path)
                })
            except Exception as e:
                warnings.warn(f"Error loading checkpoint info {path}: {e}")
                continue
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        
        return checkpoints
    
    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved at this epoch"""
        return (epoch + 1) % self.save_interval == 0 or epoch == 0
    
    def check_best(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics are the best so far"""
        if not self.save_best or self.best_metric not in metrics:
            return False
        
        current_value = metrics[self.best_metric]
        
        if 'mAP' in self.best_metric:
            return current_value > self.best_metric_value
        else:  # loss
            return current_value < self.best_metric_value


class ModelCheckpoint:
    """
    PyTorch Lightning compatible checkpoint callback
    """
    
    def __init__(
        self,
        checkpoint_dir: str = config.path.CHECKPOINT_DIR,
        save_interval: int = config.logging.SAVE_INTERVAL,
        keep_last_n: int = config.logging.KEEP_LAST_N,
        save_best: bool = config.logging.SAVE_BEST,
        best_metric: str = config.logging.BEST_METRIC,
        prefix: str = "model"
    ):
        self.manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
            keep_last_n=keep_last_n,
            save_best=save_best,
            best_metric=best_metric,
            prefix=prefix
        )
    
    def on_validation_end(self, trainer, pl_module):
        """Called at end of validation"""
        epoch = trainer.current_epoch
        
        # Check if should save
        if not self.manager.should_save(epoch):
            return
        
        # Get metrics
        metrics = trainer.callback_metrics
        
        # Check if best
        is_best = self.manager.check_best(metrics)
        
        # Save checkpoint
        self.manager.save_checkpoint(
            model=pl_module,
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            scheduler=trainer.lr_schedulers[0] if trainer.lr_schedulers else None,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
    
    def on_train_start(self, trainer, pl_module):
        """Called at start of training"""
        # Try to resume from checkpoint
        checkpoint_path = self.manager.get_latest_checkpoint()
        if checkpoint_path:
            self.manager.load_checkpoint(
                model=pl_module,
                optimizer=trainer.optimizers[0] if trainer.optimizers else None,
                scheduler=trainer.lr_schedulers[0] if trainer.lr_schedulers else None,
                path=checkpoint_path
            )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    path: str = config.path.CHECKPOINT_DIR,
    prefix: str = "model"
) -> str:
    """
    Save a checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Current metrics
        path: Directory to save checkpoint
        prefix: Prefix for filename
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(path, exist_ok=True)
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"{prefix}_epoch_{epoch:04d}.pt"
    filepath = os.path.join(path, filename)
    
    torch.save(checkpoint_data, filepath)
    
    print(f"Checkpoint saved: {filepath}")
    return filepath


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a checkpoint
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        strict: Whether to strictly enforce matching keys
    
    Returns:
        Checkpoint data
    """
    checkpoint_data = torch.load(path, map_location='cpu')
    
    # Load model
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
    
    # Load optimizer
    if optimizer and checkpoint_data.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    # Load scheduler
    if scheduler and checkpoint_data.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
    
    print(f"Checkpoint loaded: {path}")
    print(f"Epoch: {checkpoint_data.get('epoch', 0)}")
    print(f"Metrics: {checkpoint_data.get('metrics', {})}")
    
    return checkpoint_data


def save_model_config(
    model: nn.Module,
    path: str = config.path.CHECKPOINT_DIR,
    filename: str = "model_config.json"
):
    """
    Save model configuration
    
    Args:
        model: Model to save config for
        path: Directory to save config
        filename: Config filename
    """
    os.makedirs(path, exist_ok=True)
    
    config_data = {
        'class': model.__class__.__name__,
        'state_dict_keys': list(model.state_dict().keys()),
        'timestamp': datetime.now().isoformat()
    }
    
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Model config saved: {filepath}")


if __name__ == "__main__":
    # Test checkpoint management
    print("Testing Checkpoint Management...")
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            return self.fc(self.conv(x).mean(dim=[2, 3]))
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test CheckpointManager
    print("\nTesting CheckpointManager...")
    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        save_interval=5,
        keep_last_n=3,
        save_best=True,
        best_metric="mAP50"
    )
    
    # Save some checkpoints
    metrics1 = {"mAP50": 0.5, "loss": 0.5}
    manager.save_checkpoint(model, optimizer, epoch=0, metrics=metrics1)
    
    metrics2 = {"mAP50": 0.6, "loss": 0.4}
    manager.save_checkpoint(model, optimizer, epoch=5, metrics=metrics2, is_best=True)
    
    metrics3 = {"mAP50": 0.7, "loss": 0.3}
    manager.save_checkpoint(model, optimizer, epoch=10, metrics=metrics3, is_best=True)
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  {cp['path']}: epoch={cp['epoch']}, metrics={cp['metrics']}")
    
    # Load best checkpoint
    best_path = manager.get_best_checkpoint()
    print(f"Best checkpoint: {best_path}")
    
    # Clean up
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    
    print("Checkpoint management test completed!")
