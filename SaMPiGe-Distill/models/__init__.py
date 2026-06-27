"""
Models module for SaMPiGe-Distill
"""

import sys
import os

# Add parent directory to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .teacher import DINOv2Teacher, TeacherWrapper
from .student import YOLOStudent, StudentBackbone
from .projection import ProjectionHead, FeatureProjection
from .prototype import PrototypeModule, PrototypeMemoryBank
from .hooks import FeatureHooks, HookManager
from .distiller import KnowledgeDistiller, MultiTaskDistiller

__all__ = [
    "DINOv2Teacher",
    "TeacherWrapper",
    "YOLOStudent", 
    "StudentBackbone",
    "ProjectionHead",
    "FeatureProjection",
    "PrototypeModule",
    "PrototypeMemoryBank",
    "FeatureHooks",
    "HookManager",
    "KnowledgeDistiller",
    "MultiTaskDistiller"
]
