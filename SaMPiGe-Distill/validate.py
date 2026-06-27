"""
Validation Script for SaMPiGe-Distill
Evaluates model performance on validation/test data
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from datasets import KITTIDataModule
from models import YOLOStudent, DINOv2Teacher, TeacherWrapper, KnowledgeDistiller
from losses import DetectionLoss
from utils import DetectionMetrics, KnowledgeDistillationMetrics


class ModelValidator:
    """
    Validates model performance on validation/test data
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[config.Config] = None
    ):
        self.config = config or config.config
        self.model_path = model_path
        self.model = None
        self.datamodule = None
        self.metrics = None
        self.device = torch.device(self.config.system.DEVICE)
        
        # Initialize components
        self._init_model()
        self._init_datamodule()
        self._init_metrics()
    
    def _init_model(self):
        """Initialize model from checkpoint"""
        print("Initializing model...")
        
        # Load checkpoint
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Initialize student model
            num_classes = self.config.model.NUM_CLASSES
            self.model = YOLOStudent(num_classes=num_classes)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Model loaded from: {self.model_path}")
        else:
            # Initialize new model
            num_classes = self.config.model.NUM_CLASSES
            self.model = YOLOStudent(num_classes=num_classes)
            
            if self.model_path:
                warnings.warn(f"Model path not found: {self.model_path}, using new model")
            else:
                warnings.warn("No model path provided, using new model")
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model initialized")
    
    def _init_datamodule(self):
        """Initialize data module"""
        print("Initializing DataModule...")
        
        self.datamodule = KITTIDataModule(
            batch_size=self.config.validation.VAL_BATCH_SIZE,
            val_batch_size=self.config.validation.VAL_BATCH_SIZE,
            image_size=self.config.data.IMAGE_SIZE,
            num_workers=self.config.training.NUM_WORKERS
        )
        
        self.datamodule.setup()
        
        print(f"DataModule initialized: {len(self.datamodule.val_dataset)} val samples")
    
    def _init_metrics(self):
        """Initialize metrics"""
        num_classes = self.config.model.NUM_CLASSES
        self.metrics = DetectionMetrics(num_classes=num_classes)
        
        print("Metrics initialized")
    
    def validate(self, split: str = 'val') -> Dict[str, float]:
        """
        Run validation on specified split
        
        Args:
            split: Data split to validate on ('val', 'test')
        
        Returns:
            Dictionary of metrics
        """
        print(f"Validating on {split} split...")
        
        # Reset metrics
        self.metrics.reset()
        
        # Get data loader
        if split == 'val':
            dataloader = self.datamodule.val_dataloader()
        else:
            dataloader = self.datamodule.test_dataloader()
        
        # Run validation
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(self.device)
            labels = batch['labels']
            label_mask = batch.get('label_mask', None)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(images)
            
            # Get detections
            detections = outputs.get('detections', None)
            
            if detections is not None and labels is not None:
                # Convert to numpy
                detections_np = detections.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                
                # Filter valid labels
                if label_mask is not None:
                    valid_mask = label_mask.detach().cpu().numpy()
                    labels_np = [labels_np[i][valid_mask[i]] for i in range(len(labels_np))]
                
                # Update metrics
                for i in range(len(detections_np)):
                    self.metrics.update(
                        detections_np[i:i+1],
                        labels_np[i] if i < len(labels_np) else np.zeros((0, 5))
                    )
            
            total_samples += images.shape[0]
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # Compute metrics
        results = self.metrics.compute()
        
        print(f"Validation complete: {total_samples} samples")
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def evaluate(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate model on specific images
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            Dictionary of predictions and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(images.to(self.device))
        
        return outputs
    
    def get_metrics_report(self) -> str:
        """Get formatted metrics report"""
        results = self.metrics.compute()
        
        report = "Validation Metrics Report\n"
        report += "=" * 40 + "\n"
        
        for key, value in results.items():
            report += f"{key:20s}: {value:.4f}\n"
        
        report += "=" * 40
        
        return report


def run_validation(
    model_path: Optional[str] = None,
    split: str = 'val',
    save_results: bool = True,
    output_dir: str = config.path.OUTPUT_DIR
) -> Dict[str, float]:
    """
    Run validation on a model
    
    Args:
        model_path: Path to model checkpoint
        split: Data split to validate on
        save_results: Whether to save results
        output_dir: Directory to save results
    
    Returns:
        Dictionary of validation metrics
    """
    print("Running validation...")
    
    # Initialize validator
    validator = ModelValidator(model_path=model_path)
    
    # Run validation
    results = validator.validate(split=split)
    
    # Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, f"metrics_{split}.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save report
        report = validator.get_metrics_report()
        report_path = os.path.join(output_dir, f"report_{split}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Results saved to: {output_dir}")
    
    return results


def compare_models(
    model_paths: List[str],
    split: str = 'val'
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on validation data
    
    Args:
        model_paths: List of model checkpoint paths
        split: Data split to validate on
    
    Returns:
        Dictionary of results for each model
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
        
        validator = ModelValidator(model_path=model_path)
        model_results = validator.validate(split=split)
        
        # Extract model name from path
        model_name = os.path.basename(model_path).replace('.pt', '')
        results[model_name] = model_results
    
    # Print comparison
    print("\nModel Comparison:")
    print("=" * 60)
    
    # Get all metric names
    metric_names = list(results[list(results.keys())[0]].keys())
    
    # Print header
    header = f"{'Model':<20}" + "".join(f"{name:>15}" for name in metric_names)
    print(header)
    print("-" * len(header))
    
    # Print results for each model
    for model_name, model_results in results.items():
        row = f"{model_name:<20}"
        for metric_name in metric_names:
            value = model_results.get(metric_name, 0)
            row += f"{value:>15.4f}"
        print(row)
    
    return results


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SaMPiGe-Distill Validation')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Data split to validate on')
    parser.add_argument('--save_results', action='store_true', help='Save validation results')
    parser.add_argument('--output_dir', type=str, default=config.path.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        results = compare_models(args.compare, split=args.split)
    else:
        # Run single validation
        results = run_validation(
            model_path=args.model_path,
            split=args.split,
            save_results=args.save_results,
            output_dir=args.output_dir
        )
    
    print("Validation completed!")


if __name__ == "__main__":
    main()
