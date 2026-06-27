#!/usr/bin/env python3
"""
Final Test Script for SaMPiGe-Distill
Tests the complete package structure
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing SaMPiGe-Distill Package")
print("=" * 50)

# Test 1: Import the package
print("\n1. Testing Package Import...")
try:
    import sampige_distill
    print("   ✅ Package imported successfully")
    print(f"   Version: {sampige_distill.__version__}")
except Exception as e:
    print(f"   ❌ Package import failed: {e}")
    sys.exit(1)

# Test 2: Import configuration
print("\n2. Testing Configuration...")
try:
    from .config import config
    print(f"   Device: {config.system.DEVICE}")
    print(f"   Image Size: {config.data.IMAGE_SIZE}")
    print(f"   Batch Size: {config.training.BATCH_SIZE}")
    print("   ✅ Configuration test passed")
except Exception as e:
    print(f"   ❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 3: Import and test models
print("\n3. Testing Models...")
try:
    import torch
    from .models.student import StudentBackbone
    
    backbone = StudentBackbone(
        backbone_type="custom",
        pretrained=False
    )
    
    dummy_input = torch.randn(2, 3, 256, 256)
    outputs = backbone(dummy_input)
    
    print(f"   Backbone output keys: {list(outputs.keys())}")
    print(f"   Global embedding shape: {outputs['global_embedding'].shape}")
    print("   ✅ StudentBackbone test passed")
except Exception as e:
    print(f"   ❌ StudentBackbone test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import and test losses
print("\n4. Testing Loss Functions...")
try:
    import torch
    from .losses import DetectionLoss, CIoULoss, KnowledgeDistillationLoss
    
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
    
    # Test Knowledge Distillation Loss
    kd_loss = KnowledgeDistillationLoss(loss_type="mse")
    student_features = torch.randn(10, 768)
    teacher_features = torch.randn(10, 768)
    kd_loss_val = kd_loss(student_features, teacher_features)
    print(f"   KD loss: {kd_loss_val.item():.4f}")
    
    print("   ✅ Loss functions test passed")
except Exception as e:
    print(f"   ❌ Loss functions test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import and test schedulers
print("\n5. Testing Schedulers...")
try:
    from .scheduler import DynamicWeightScheduler, LambdaScheduler
    
    initial_weights = {
        'detection': 1.0,
        'feature': 0.5,
        'attention': 0.3
    }
    
    scheduler = DynamicWeightScheduler(
        initial_weights=initial_weights,
        schedule_type='linear',
        total_epochs=100,
        warmup_epochs=5,
        cooldown_epochs=10
    )
    
    weights = scheduler(50)
    print(f"   Scheduler weights: {weights}")
    print("   ✅ Schedulers test passed")
except Exception as e:
    print(f"   ❌ Schedulers test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Import and test utils
print("\n6. Testing Utilities...")
try:
    import numpy as np
    from .utils.metrics import DetectionMetrics
    
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
    print("   ✅ Metrics test passed")
except Exception as e:
    print(f"   ❌ Metrics test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Integration test
print("\n7. Testing Integration...")
try:
    import torch
    import torch.nn as nn
    from .models.distiller import KnowledgeDistiller
    from .losses import DetectionLoss
    
    # Mock models
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
    print("   ✅ Integration test passed")
except Exception as e:
    print(f"   ❌ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED!")
print("=" * 50)
print("\n🎉 SaMPiGe-Distill is ready for use!")
print("\nYou can now:")
print("  - Run training: python train.py")
print("  - Run validation: python validate.py --model_path checkpoints/model.pt")
print("  - Run inference: python infer.py --model_path checkpoints/model.pt --input image.jpg")
