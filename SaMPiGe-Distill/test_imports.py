#!/usr/bin/env python3
"""
Test imports for SaMPiGe-Distill
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing SaMPiGe-Distill imports...")

# Test 1: Config
try:
    from config import config
    print("✅ Config imported successfully")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

# Test 2: Models
try:
    from models.student import StudentBackbone
    from models.projection import ProjectionHead
    from models.prototype import PrototypeModule
    from models.hooks import FeatureHooks
    print("✅ Models imported successfully")
except Exception as e:
    print(f"❌ Models import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Losses
try:
    from losses import DetectionLoss, CIoULoss, KnowledgeDistillationLoss
    print("✅ Losses imported successfully")
except Exception as e:
    print(f"❌ Losses import failed: {e}")
    sys.exit(1)

# Test 4: Scheduler
try:
    from scheduler import DynamicWeightScheduler, LambdaScheduler
    print("✅ Scheduler imported successfully")
except Exception as e:
    print(f"❌ Scheduler import failed: {e}")
    sys.exit(1)

# Test 5: Utils
try:
    from utils.metrics import DetectionMetrics
    from utils.checkpoint import CheckpointManager
    from utils.logger import CSVLogger, JSONLogger
    print("✅ Utils imported successfully")
except Exception as e:
    print(f"❌ Utils import failed: {e}")
    sys.exit(1)

# Test 6: Datasets
try:
    from datasets.collate import collate_fn
    print("✅ Datasets imported successfully")
except Exception as e:
    print(f"❌ Datasets import failed: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
print("SaMPiGe-Distill is ready to use.")
