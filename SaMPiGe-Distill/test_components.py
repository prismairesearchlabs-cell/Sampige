#!/usr/bin/env python3
"""
Comprehensive Test Script for SaMPiGe-Distill Components
Tests all major components to ensure they work correctly
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from datasets import KITTIDataset, KITTIDataModule, collate_fn
from models import (
    DINOv2Teacher, TeacherWrapper,
    YOLOStudent, StudentBackbone,
    ProjectionHead, FeatureProjection,
    PrototypeModule, PrototypeMemoryBank,
    FeatureHooks, HookManager,
    KnowledgeDistiller, MultiTaskDistiller
)
from losses import (
    DetectionLoss, CIoULoss, DIoULoss, SIoULoss,
    DistributionFocalLoss, KnowledgeDistillationLoss
)
from scheduler import (
    DynamicWeightScheduler, LambdaScheduler,
    LossBalancingScheduler, WarmupScheduler, CompositeScheduler
)
from utils import (
    DetectionMetrics, KnowledgeDistillationMetrics,
    DetectionVisualizer, FeatureMapVisualizer,
    CheckpointManager, LogManager, CSVLogger, JSONLogger
)


def test_config():
    """Test configuration module"""
    print("Testing Config...")
    
    # Test config instance
    assert config is not None
    assert hasattr(config, 'path')
    assert hasattr(config, 'model')
    assert hasattr(config, 'training')
    
    # Test path config
    assert config.path.IMAGE_DIR is not None
    assert config.path.LABEL_DIR is not None
    
    # Test model config
    assert config.model.TEACHER_MODEL == "dinov2_vitb14"
    assert config.model.NUM_CLASSES == 8
    
    print("✅ Config test passed")


def test_datasets():
    """Test dataset components"""
    print("\nTesting Datasets...")
    
    # Test collate function
    batch = [
        {
            'image': torch.randn(3, 256, 256),
            'labels': torch.tensor([[10, 20, 100, 150, 0], [50, 60, 120, 180, 1]], dtype=torch.float32),
            'image_path': 'image1.jpg',
            'image_id': 0,
            'original_size': torch.tensor([640, 480], dtype=torch.float32),
            'scale_factor': torch.tensor([0.4, 0.533], dtype=torch.float32)
        },
        {
            'image': torch.randn(3, 256, 256),
            'labels': torch.tensor([[30, 40, 80, 120, 2]], dtype=torch.float32),
            'image_path': 'image2.jpg',
            'image_id': 1,
            'original_size': torch.tensor([640, 480], dtype=torch.float32),
            'scale_factor': torch.tensor([0.4, 0.533], dtype=torch.float32)
        }
    ]
    
    collated = collate_fn(batch)
    assert collated['image'].shape == (2, 3, 256, 256)
    assert collated['labels'].shape[0] == 2
    assert collated['label_mask'].shape == (2, 2)
    
    print("✅ Dataset collate test passed")


def test_models():
    """Test model components"""
    print("\nTesting Models...")
    
    # Test StudentBackbone
    backbone = StudentBackbone(
        backbone_type="custom",
        pretrained=False
    )
    
    dummy_input = torch.randn(2, 3, 256, 256)
    outputs = backbone(dummy_input)
    
    assert 'features' in outputs
    assert 'global_embedding' in outputs
    assert outputs['global_embedding'].shape[0] == 2
    assert outputs['global_embedding'].shape[1] == config.model.PROJECTION_DIM
    
    print("✅ StudentBackbone test passed")
    
    # Test ProjectionHead
    proj_head = ProjectionHead(
        input_dim=256,
        output_dim=768
    )
    
    dummy_features = torch.randn(2, 256, 32, 32)
    projected = proj_head(dummy_features)
    assert projected.shape == (2, 768, 32, 32)
    
    print("✅ ProjectionHead test passed")
    
    # Test FeatureProjection
    feature_dims = {"P3": 256, "P4": 512, "P5": 1024}
    feature_proj = FeatureProjection(feature_dims, output_dim=768)
    
    dummy_features = {
        "P3": torch.randn(2, 256, 64, 64),
        "P4": torch.randn(2, 512, 32, 32),
        "P5": torch.randn(2, 1024, 16, 16)
    }
    
    projected_features = feature_proj(dummy_features)
    assert len(projected_features) == 3
    
    print("✅ FeatureProjection test passed")
    
    # Test PrototypeModule
    prototype_module = PrototypeModule(
        num_prototypes=50,
        prototype_dim=768,
        init_method="random"
    )
    
    dummy_features = torch.randn(2, 100, 768)  # (B, N, dim)
    results = prototype_module(dummy_features)
    
    assert 'assignments' in results
    assert 'distances' in results
    assert 'closest' in results
    assert results['assignments'].shape == (2, 100, 50)
    
    print("✅ PrototypeModule test passed")
    
    # Test PrototypeMemoryBank
    memory_bank = PrototypeMemoryBank(
        capacity=1000,
        feature_dim=768
    )
    
    dummy_features = torch.randn(100, 768)
    memory_bank.add(dummy_features)
    assert memory_bank.size == 100
    
    sampled = memory_bank.sample(10)
    assert sampled.shape == (10, 768)
    
    print("✅ PrototypeMemoryBank test passed")


def test_losses():
    """Test loss functions"""
    print("\nTesting Losses...")
    
    # Test DetectionLoss
    detection_loss = DetectionLoss(
        num_classes=8,
        box_loss="ciou",
        cls_loss="bce",
        dfl_loss=False
    )
    
    # Create dummy data
    batch_size = 2
    num_preds = 10
    num_classes = 8
    
    predictions = torch.randn(batch_size, num_preds, 5 + num_classes)
    targets = torch.randn(batch_size, 5, 5)
    targets[..., 4] = torch.randint(0, num_classes, (batch_size, 5))
    label_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    
    loss = detection_loss(predictions, targets, label_mask)
    assert loss.item() >= 0
    
    print("✅ DetectionLoss test passed")
    
    # Test individual box losses
    pred_boxes = torch.randn(10, 4)
    target_boxes = torch.randn(10, 4)
    
    ciou_loss = CIoULoss()
    diou_loss = DIoULoss()
    siou_loss = SIoULoss()
    
    ciou_val = ciou_loss(pred_boxes, target_boxes).mean()
    diou_val = diou_loss(pred_boxes, target_boxes).mean()
    siou_val = siou_loss(pred_boxes, target_boxes).mean()
    
    assert ciou_val.item() >= 0
    assert diou_val.item() >= 0
    assert siou_val.item() >= 0
    
    print("✅ Box loss tests passed")
    
    # Test KnowledgeDistillationLoss
    kd_loss_mse = KnowledgeDistillationLoss(loss_type="mse")
    kd_loss_cosine = KnowledgeDistillationLoss(loss_type="cosine")
    
    student_features = torch.randn(10, 768)
    teacher_features = torch.randn(10, 768)
    
    mse_loss = kd_loss_mse(student_features, teacher_features)
    cosine_loss = kd_loss_cosine(student_features, teacher_features)
    
    assert mse_loss.item() >= 0
    assert cosine_loss.item() >= 0
    
    print("✅ KnowledgeDistillationLoss test passed")


def test_schedulers():
    """Test scheduler components"""
    print("\nTesting Schedulers...")
    
    initial_weights = {
        'detection': 1.0,
        'feature': 0.5,
        'attention': 0.3
    }
    
    # Test DynamicWeightScheduler
    scheduler = DynamicWeightScheduler(
        initial_weights=initial_weights,
        schedule_type='linear',
        total_epochs=100,
        warmup_epochs=5,
        cooldown_epochs=10
    )
    
    weights_epoch_0 = scheduler(0)
    weights_epoch_50 = scheduler(50)
    weights_epoch_99 = scheduler(99)
    
    assert 'detection' in weights_epoch_0
    assert 'feature' in weights_epoch_0
    
    print("✅ DynamicWeightScheduler test passed")
    
    # Test LambdaScheduler
    lambda_scheduler = LambdaScheduler(
        initial_weights=initial_weights,
        lambda_functions={
            'detection': lambda epoch: min(epoch / 20, 1.0),
            'feature': lambda epoch: max(0, 1.0 - epoch / 50)
        }
    )
    
    weights = lambda_scheduler(10)
    assert 'detection' in weights
    assert 'feature' in weights
    
    print("✅ LambdaScheduler test passed")
    
    # Test WarmupScheduler
    warmup_scheduler = WarmupScheduler(
        initial_weights=initial_weights,
        warmup_epochs=10,
        warmup_type='linear'
    )
    
    weights = warmup_scheduler(5)
    assert 'detection' in weights
    
    print("✅ WarmupScheduler test passed")


def test_utils():
    """Test utility components"""
    print("\nTesting Utils...")
    
    # Test DetectionMetrics
    metrics = DetectionMetrics(num_classes=8)
    
    dummy_preds = np.array([
        [10, 20, 100, 150, 0],  # Car
        [50, 60, 120, 180, 1]   # Van
    ])
    
    dummy_targets = np.array([
        [15, 25, 105, 155, 0],  # Car
        [55, 65, 125, 185, 1]   # Van
    ])
    
    metrics.update(dummy_preds, dummy_targets)
    results = metrics.compute()
    
    assert 'precision' in results
    assert 'recall' in results
    assert 'f1' in results
    
    print("✅ DetectionMetrics test passed")
    
    # Test KnowledgeDistillationMetrics
    kd_metrics = KnowledgeDistillationMetrics()
    kd_metrics.update(
        feature_similarity=0.8,
        attention_similarity=0.7,
        losses={'feature_loss': 0.1, 'attention_loss': 0.2}
    )
    
    kd_results = kd_metrics.compute()
    assert 'feature_similarity' in kd_results
    
    print("✅ KnowledgeDistillationMetrics test passed")
    
    # Test DetectionVisualizer
    class_names = ['Car', 'Van', 'Truck']
    visualizer = DetectionVisualizer(class_names)
    
    dummy_image = np.random.rand(256, 256, 3) * 255
    dummy_image = dummy_image.astype(np.uint8)
    
    dummy_preds = np.array([
        [50, 50, 150, 150, 0, 0.95],
        [200, 200, 250, 250, 1, 0.85]
    ])
    
    fig = visualizer.visualize(
        dummy_image,
        predictions=dummy_preds,
        confidence_threshold=0.5
    )
    
    assert fig is not None
    
    print("✅ DetectionVisualizer test passed")
    
    # Test FeatureMapVisualizer
    feature_visualizer = FeatureMapVisualizer()
    
    dummy_features = {
        "P3": torch.randn(1, 16, 64, 64),
        "P4": torch.randn(1, 32, 32, 32),
        "P5": torch.randn(1, 64, 16, 16)
    }
    
    fig = feature_visualizer.visualize_feature_maps(dummy_features, num_maps=4)
    assert fig is not None
    
    print("✅ FeatureMapVisualizer test passed")
    
    # Test CheckpointManager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        save_interval=5,
        keep_last_n=3
    )
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3)
        
        def forward(self, x):
            return self.conv(x)
    
    dummy_model = DummyModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters())
    
    # Save checkpoint
    path = checkpoint_manager.save_checkpoint(
        model=dummy_model,
        optimizer=dummy_optimizer,
        epoch=0,
        metrics={'loss': 0.5}
    )
    
    assert os.path.exists(path)
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 1
    
    # Clean up
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    
    print("✅ CheckpointManager test passed")
    
    # Test Loggers
    csv_logger = CSVLogger(log_dir="./test_logs", filename="test.csv")
    csv_logger.log({'loss': 0.5, 'accuracy': 0.9}, step=0)
    
    json_logger = JSONLogger(log_dir="./test_logs", filename="test.json")
    json_logger.log({'loss': 0.5, 'accuracy': 0.9}, step=0)
    
    # Clean up
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")
    
    print("✅ Logger tests passed")


def test_integration():
    """Test integration of components"""
    print("\nTesting Integration...")
    
    # Test KnowledgeDistiller with mock models
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 768
            self.cls_token = nn.Parameter(torch.randn(1, 768))
            
        def forward(self, x):
            B = x.shape[0]
            cls_token = self.cls_token.expand(B, -1)
            return {
                'cls_token': cls_token,
                'patch_tokens': torch.randn(B, 100, 768),
                'features': torch.randn(B, 101, 768),
                'embed_dim': self.embed_dim
            }
    
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU()
            )
            self.detection_head = nn.Linear(64, 10 * 6)
            
        def forward(self, x):
            features = self.backbone(x)
            global_embedding = features.mean(dim=[2, 3])
            detections = self.detection_head(global_embedding)
            return {
                'features': {"P5": features},
                'global_embedding': global_embedding,
                'detections': detections.view(x.size(0), 10, 6)
            }
    
    mock_teacher = MockTeacher()
    mock_student = MockStudent()
    
    distiller = KnowledgeDistiller(
        teacher=mock_teacher,
        student=mock_student
    )
    
    # Set detection loss
    detection_loss = DetectionLoss(num_classes=8)
    distiller.set_detection_loss_fn(detection_loss)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_labels = torch.randn(2, 5, 5)
    
    outputs = distiller(dummy_input, dummy_labels)
    
    assert 'total_loss' in outputs
    assert 'detection_loss' in outputs
    assert 'feature_loss' in outputs
    
    print("✅ KnowledgeDistiller integration test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SaMPiGe-Distill Component Tests")
    print("=" * 60)
    
    try:
        test_config()
        test_datasets()
        test_models()
        test_losses()
        test_schedulers()
        test_utils()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
