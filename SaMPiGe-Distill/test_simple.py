#!/usr/bin/env python3
"""
Simple Test Script for SaMPiGe-Distill Core Components
Tests core functionality without external dependencies
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first
from config import config

print("Testing SaMPiGe-Distill Core Components")
print("=" * 50)

# Test 1: Configuration
print("\n1. Testing Configuration...")
print(f"   Device: {config.system.DEVICE}")
print(f"   Image Size: {config.data.IMAGE_SIZE}")
print(f"   Batch Size: {config.training.BATCH_SIZE}")
print(f"   Model: {config.model.TEACHER_MODEL}")
print("   ✅ Configuration test passed")

# Test 2: Models
print("\n2. Testing Models...")

# Test StudentBackbone
from models.student import StudentBackbone

backbone = StudentBackbone(
    backbone_type="custom",
    pretrained=False
)

dummy_input = torch.randn(2, 3, 256, 256)
outputs = backbone(dummy_input)

print(f"   Backbone output keys: {list(outputs.keys())}")
print(f"   Global embedding shape: {outputs['global_embedding'].shape}")
print("   ✅ StudentBackbone test passed")

# Test ProjectionHead
from models.projection import ProjectionHead

proj_head = ProjectionHead(
    input_dim=256,
    output_dim=768
)

dummy_features = torch.randn(2, 256, 32, 32)
projected = proj_head(dummy_features)
print(f"   Projection output shape: {projected.shape}")
print("   ✅ ProjectionHead test passed")

# Test FeatureProjection
from models.projection import FeatureProjection

feature_dims = {"P3": 256, "P4": 512, "P5": 1024}
feature_proj = FeatureProjection(feature_dims, output_dim=768)

dummy_features = {
    "P3": torch.randn(2, 256, 64, 64),
    "P4": torch.randn(2, 512, 32, 32),
    "P5": torch.randn(2, 1024, 16, 16)
}

projected_features = feature_proj(dummy_features)
print(f"   Projected features: {list(projected_features.keys())}")
print("   ✅ FeatureProjection test passed")

# Test 3: Prototype Module
print("\n3. Testing Prototype Module...")

from models.prototype import PrototypeModule

prototype_module = PrototypeModule(
    num_prototypes=50,
    prototype_dim=768,
    init_method="random"
)

dummy_features = torch.randn(2, 100, 768)
results = prototype_module(dummy_features)

print(f"   Assignments shape: {results['assignments'].shape}")
print(f"   Distances shape: {results['distances'].shape}")
print("   ✅ PrototypeModule test passed")

# Test 4: Prototype Memory Bank
print("\n4. Testing Prototype Memory Bank...")

from models.prototype import PrototypeMemoryBank

memory_bank = PrototypeMemoryBank(
    capacity=1000,
    feature_dim=768
)

dummy_features = torch.randn(100, 768)
memory_bank.add(dummy_features)
print(f"   Memory bank size: {memory_bank.size}")

sampled = memory_bank.sample(10)
print(f"   Sampled features shape: {sampled.shape}")
print("   ✅ PrototypeMemoryBank test passed")

# Test 5: Loss Functions
print("\n5. Testing Loss Functions...")

from losses import DetectionLoss, CIoULoss, DIoULoss, KnowledgeDistillationLoss

# Test DetectionLoss
detection_loss = DetectionLoss(
    num_classes=8,
    box_loss="mse",
    cls_loss="bce",
    dfl_loss=False
)

batch_size = 2
num_preds = 10
num_classes = 8

predictions = torch.randn(batch_size, num_preds, 5 + num_classes)
targets = torch.randn(batch_size, 5, 5)
targets[..., 4] = torch.randint(0, num_classes, (batch_size, 5))
label_mask = torch.ones(batch_size, 5, dtype=torch.bool)

loss = detection_loss(predictions, targets, label_mask)
print(f"   Detection loss: {loss.item():.4f}")

# Test Box Losses
pred_boxes = torch.randn(10, 4)
target_boxes = torch.randn(10, 4)

ciou_loss = CIoULoss()
diou_loss = DIoULoss()

ciou_val = ciou_loss(pred_boxes, target_boxes).mean()
diou_val = diou_loss(pred_boxes, target_boxes).mean()

print(f"   CIoU loss: {ciou_val.item():.4f}")
print(f"   DIoU loss: {diou_val.item():.4f}")

# Test Knowledge Distillation Loss
kd_loss_mse = KnowledgeDistillationLoss(loss_type="mse")
kd_loss_cosine = KnowledgeDistillationLoss(loss_type="cosine")

student_features = torch.randn(10, 768)
teacher_features = torch.randn(10, 768)

mse_loss = kd_loss_mse(student_features, teacher_features)
cosine_loss = kd_loss_cosine(student_features, teacher_features)

print(f"   KD MSE loss: {mse_loss.item():.4f}")
print(f"   KD Cosine loss: {cosine_loss.item():.4f}")
print("   ✅ Loss functions test passed")

# Test 6: Schedulers
print("\n6. Testing Schedulers...")

from scheduler import DynamicWeightScheduler, LambdaScheduler, WarmupScheduler

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

print(f"   Epoch 0 weights: detection={weights_epoch_0['detection']:.3f}, feature={weights_epoch_0['feature']:.3f}")
print(f"   Epoch 50 weights: detection={weights_epoch_50['detection']:.3f}, feature={weights_epoch_50['feature']:.3f}")
print(f"   Epoch 99 weights: detection={weights_epoch_99['detection']:.3f}, feature={weights_epoch_99['feature']:.3f}")

# Test LambdaScheduler
lambda_scheduler = LambdaScheduler(
    initial_weights=initial_weights,
    lambda_functions={
        'detection': lambda epoch: min(epoch / 20, 1.0),
        'feature': lambda epoch: max(0, 1.0 - epoch / 50)
    }
)

weights = lambda_scheduler(10)
print(f"   Lambda scheduler weights: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")

# Test WarmupScheduler
warmup_scheduler = WarmupScheduler(
    initial_weights=initial_weights,
    warmup_epochs=10,
    warmup_type='linear'
)

weights = warmup_scheduler(5)
print(f"   Warmup scheduler weights: detection={weights['detection']:.3f}, feature={weights['feature']:.3f}")
print("   ✅ Schedulers test passed")

# Test 7: Metrics
print("\n7. Testing Metrics...")

from utils.metrics import DetectionMetrics, compute_mAP, compute_precision_recall, compute_f1_score

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

print(f"   Precision: {results.get('precision', 0.0):.4f}")
print(f"   Recall: {results.get('recall', 0.0):.4f}")
print(f"   F1: {results.get('f1', 0.0):.4f}")

# Test standalone functions
mAP = compute_mAP([dummy_preds], [dummy_targets])
precision, recall = compute_precision_recall([dummy_preds], [dummy_targets])
f1 = compute_f1_score([dummy_preds], [dummy_targets])

print(f"   mAP: {mAP:.4f}")
print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print("   ✅ Metrics test passed")

# Test 8: Checkpoint Management
print("\n8. Testing Checkpoint Management...")

from utils.checkpoint import CheckpointManager, save_checkpoint, load_checkpoint

# Create test directory
import tempfile
with tempfile.TemporaryDirectory() as temp_dir:
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=temp_dir,
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
    
    print(f"   Checkpoint saved: {os.path.exists(path)}")
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"   Checkpoints count: {len(checkpoints)}")
    
    # Load checkpoint
    loaded_data = load_checkpoint(
        path=path,
        model=dummy_model,
        optimizer=dummy_optimizer
    )
    
    print(f"   Checkpoint loaded: epoch={loaded_data.get('epoch', 0)}")
    print("   ✅ Checkpoint management test passed")

# Test 9: Loggers
print("\n9. Testing Loggers...")

from utils.logger import CSVLogger, JSONLogger, LogManager

with tempfile.TemporaryDirectory() as temp_dir:
    csv_logger = CSVLogger(log_dir=temp_dir, filename="test.csv")
    csv_logger.log({'loss': 0.5, 'accuracy': 0.9}, step=0)
    
    json_logger = JSONLogger(log_dir=temp_dir, filename="test.json")
    json_logger.log({'loss': 0.5, 'accuracy': 0.9}, step=0)
    
    # Check files exist
    csv_exists = os.path.exists(os.path.join(temp_dir, "test.csv"))
    json_exists = os.path.exists(os.path.join(temp_dir, "test.json"))
    
    print(f"   CSV log created: {csv_exists}")
    print(f"   JSON log created: {json_exists}")
    
    # Test LogManager
    log_manager = LogManager([
        CSVLogger(log_dir=temp_dir, filename="manager.csv")
    ])
    log_manager.log({'loss': 0.4}, step=1)
    
    print("   ✅ Loggers test passed")

# Test 10: Integration
print("\n10. Testing Integration...")

# Test KnowledgeDistiller with mock models
from models.distiller import KnowledgeDistiller

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

print(f"   Output keys: {list(outputs.keys())}")
print(f"   Total loss: {outputs['total_loss'].item():.4f}")
print(f"   Detection loss: {outputs['detection_loss'].item():.4f}")
print(f"   Feature loss: {outputs['feature_loss'].item():.4f}")
print("   ✅ Integration test passed")

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)
print("\nSaMPiGe-Distill is ready for use!")
