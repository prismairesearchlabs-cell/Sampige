"""
Models module for SaMPiGe-Distill
"""

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
