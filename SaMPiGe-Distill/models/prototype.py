"""
Prototype Module Implementation
Semantic prototype memory bank for knowledge distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.cluster import KMeans
import warnings

from config import config


class PrototypeModule(nn.Module):
    """
    Prototype module for semantic concept alignment
    
    Creates and maintains a memory bank of semantic prototypes
    extracted from teacher patch tokens
    
    Args:
        num_prototypes: Number of semantic prototypes
        prototype_dim: Dimension of prototype vectors
        init_method: Initialization method ('kmeans', 'random')
        memory_bank_size: Size of memory bank for prototype updates
    """
    
    def __init__(
        self,
        num_prototypes: int = config.model.NUM_PROTOTYPES,
        prototype_dim: int = config.model.PROTOTYPE_DIM,
        init_method: str = config.model.PROTOTYPE_INIT,
        memory_bank_size: int = config.model.PROTOTYPE_MEMORY_BANK_SIZE
    ):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.init_method = init_method
        self.memory_bank_size = memory_bank_size
        
        # Initialize prototypes
        self.prototypes = self._init_prototypes()
        
        # Memory bank for prototype updates
        self.memory_bank = torch.zeros(
            (memory_bank_size, prototype_dim),
            dtype=torch.float32
        )
        self.memory_index = 0
        
        # Prototype assignment softmax temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        print(f"Prototype Module: {num_prototypes} prototypes, dim={prototype_dim}")
        print(f"Memory bank size: {memory_bank_size}")
    
    def _init_prototypes(self) -> nn.Parameter:
        """Initialize prototypes"""
        if self.init_method == 'kmeans':
            # Initialize with K-means clustering (requires data)
            prototypes = nn.Parameter(
                torch.randn(self.num_prototypes, self.prototype_dim)
            )
            nn.init.normal_(prototypes, std=0.02)
        else:
            # Random initialization
            prototypes = nn.Parameter(
                torch.randn(self.num_prototypes, self.prototype_dim)
            )
            nn.init.normal_(prototypes, std=0.02)
        
        return prototypes
    
    def initialize_with_kmeans(self, features: torch.Tensor):
        """
        Initialize prototypes using K-means clustering
        
        Args:
            features: Teacher patch features (N, prototype_dim)
        """
        try:
            # Convert to numpy
            features_np = features.detach().cpu().numpy()
            
            # Apply K-means
            kmeans = KMeans(
                n_clusters=self.num_prototypes,
                random_state=42,
                n_init=10
            )
            kmeans.fit(features_np)
            
            # Set prototypes to cluster centers
            with torch.no_grad():
                self.prototypes.data = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=torch.float32,
                    device=self.prototypes.device
                )
            
            print(f"Prototypes initialized with K-means on {features.shape[0]} features")
            
        except ImportError:
            warnings.warn("scikit-learn not available, using random initialization")
            nn.init.normal_(self.prototypes, std=0.02)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: assign features to prototypes
        
        Args:
            features: Input features (B, N, prototype_dim) or (B, prototype_dim)
        
        Returns:
            Dictionary containing:
                - assignments: Soft assignments to prototypes (B, N, num_prototypes)
                - distances: Distances to prototypes (B, N, num_prototypes)
                - closest: Closest prototype indices (B, N)
        """
        # Ensure correct shape
        if features.dim() == 2:
            features = features.unsqueeze(1)  # (B, 1, dim)
        
        B, N, D = features.shape
        
        # Normalize features and prototypes
        features_norm = F.normalize(features, p=2, dim=-1)  # (B, N, dim)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)  # (num_prototypes, dim)
        
        # Compute similarity scores (cosine similarity)
        # (B, N, dim) @ (dim, num_prototypes) -> (B, N, num_prototypes)
        similarity = torch.bmm(features_norm, prototypes_norm.transpose(0, 1))
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Softmax to get assignments
        assignments = F.softmax(similarity, dim=-1)
        
        # Compute distances (1 - similarity)
        distances = 1 - similarity
        
        # Get closest prototype indices
        closest = torch.argmin(distances, dim=-1)
        
        return {
            'assignments': assignments,
            'distances': distances,
            'closest': closest,
            'similarity': similarity
        }
    
    def get_prototype_embeddings(self) -> torch.Tensor:
        """Get prototype embeddings"""
        return self.prototypes
    
    def update_memory_bank(self, features: torch.Tensor):
        """
        Update memory bank with new features
        
        Args:
            features: New features to add to memory bank (N, prototype_dim)
        """
        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=-1)
        
        # Update memory bank (circular buffer)
        end_idx = min(self.memory_index + features.shape[0], self.memory_bank_size)
        
        self.memory_bank[self.memory_index:end_idx] = features[:end_idx - self.memory_index]
        
        if end_idx == self.memory_bank_size:
            # Wrap around
            remaining = features[end_idx - self.memory_index:]
            if remaining.shape[0] > 0:
                self.memory_bank[:remaining.shape[0]] = remaining
            self.memory_index = remaining.shape[0]
        else:
            self.memory_index = end_idx
    
    def update_prototypes(self):
        """
        Update prototypes using memory bank (optional)
        Can be called periodically during training
        """
        if self.memory_index == 0:
            return  # No features in memory bank
        
        # Get valid features from memory bank
        valid_features = self.memory_bank[:self.memory_index]
        
        # Re-initialize prototypes with K-means on memory bank
        self.initialize_with_kmeans(valid_features)
    
    def get_prototype_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototype alignment loss
        
        Args:
            student_features: Student features (B, N, dim)
            teacher_features: Teacher features (B, N, dim)
        
        Returns:
            Prototype alignment loss
        """
        # Get assignments for both student and teacher
        student_results = self.forward(student_features)
        teacher_results = self.forward(teacher_features)
        
        # Use KL divergence between assignments
        student_assignments = student_results['assignments']
        teacher_assignments = teacher_results['assignments']
        
        # Add small epsilon for numerical stability
        epsilon = 1e-8
        student_assignments = torch.clamp(student_assignments, min=epsilon)
        teacher_assignments = torch.clamp(teacher_assignments, min=epsilon)
        
        # KL divergence: sum(teacher * log(teacher / student))
        kl_div = torch.sum(
            teacher_assignments * (
                torch.log(teacher_assignments) - torch.log(student_assignments)
            ),
            dim=-1
        )
        
        # Mean over batch and sequence
        loss = kl_div.mean()
        
        return loss


class PrototypeMemoryBank(nn.Module):
    """
    Memory bank for storing and retrieving prototype features
    
    Maintains a queue of features for prototype learning
    """
    
    def __init__(
        self,
        capacity: int = config.model.PROTOTYPE_MEMORY_BANK_SIZE,
        feature_dim: int = config.model.PROTOTYPE_DIM
    ):
        super().__init__()
        
        self.capacity = capacity
        self.feature_dim = feature_dim
        
        # Memory bank
        self.memory = torch.zeros((capacity, feature_dim), dtype=torch.float32)
        self.pointer = 0
        self.size = 0
        
        print(f"Prototype Memory Bank: capacity={capacity}, dim={feature_dim}")
    
    def add(self, features: torch.Tensor):
        """
        Add features to memory bank
        
        Args:
            features: Features to add (N, feature_dim)
        """
        N = features.shape[0]
        
        # Ensure we don't exceed capacity
        if self.pointer + N > self.capacity:
            # Wrap around
            remaining = self.capacity - self.pointer
            self.memory[self.pointer:] = features[:remaining]
            self.memory[:N - remaining] = features[remaining:]
            self.pointer = N - remaining
        else:
            self.memory[self.pointer:self.pointer + N] = features
            self.pointer += N
        
        # Update size (but not beyond capacity)
        self.size = min(self.size + N, self.capacity)
    
    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample features from memory bank
        
        Args:
            batch_size: Number of features to sample
        
        Returns:
            Sampled features (batch_size, feature_dim)
        """
        if self.size == 0:
            return torch.zeros((batch_size, self.feature_dim), dtype=torch.float32)
        
        # Random indices
        indices = torch.randint(0, self.size, (batch_size,))
        
        return self.memory[indices]
    
    def get_all(self) -> torch.Tensor:
        """Get all features in memory bank"""
        return self.memory[:self.size]
    
    def clear(self):
        """Clear memory bank"""
        self.memory.zero_()
        self.pointer = 0
        self.size = 0
    
    def is_full(self) -> bool:
        """Check if memory bank is full"""
        return self.size == self.capacity


class SemanticPrototypeLoss(nn.Module):
    """
    Semantic prototype alignment loss
    
    Aligns student features with semantic prototypes learned from teacher
    """
    
    def __init__(
        self,
        num_prototypes: int = config.model.NUM_PROTOTYPES,
        prototype_dim: int = config.model.PROTOTYPE_DIM,
        loss_type: str = "mse"  # or "cosine", "kl"
    ):
        super().__init__()
        
        self.num_prototypes = num_prototypes
        self.prototype_dim = prototype_dim
        self.loss_type = loss_type
        
        # Prototype module
        self.prototype_module = PrototypeModule(
            num_prototypes=num_prototypes,
            prototype_dim=prototype_dim
        )
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic prototype loss
        
        Args:
            student_features: Student features (B, N, dim)
            teacher_features: Teacher features (B, N, dim)
        
        Returns:
            Prototype alignment loss
        """
        if self.loss_type == "mse":
            return self._mse_loss(student_features, teacher_features)
        elif self.loss_type == "cosine":
            return self._cosine_loss(student_features, teacher_features)
        elif self.loss_type == "kl":
            return self._kl_loss(student_features, teacher_features)
        else:
            return self._mse_loss(student_features, teacher_features)
    
    def _mse_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss between student and teacher prototype assignments"""
        # Get prototype assignments
        student_results = self.prototype_module.forward(student_features)
        teacher_results = self.prototype_module.forward(teacher_features)
        
        # Use similarity scores
        student_sim = student_results['similarity']
        teacher_sim = teacher_results['similarity']
        
        # MSE loss
        loss = F.mse_loss(student_sim, teacher_sim)
        
        return loss
    
    def _cosine_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Cosine similarity loss"""
        # Normalize features
        student_norm = F.normalize(student_features, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        
        # Cosine similarity
        similarity = torch.bmm(student_norm, teacher_norm.transpose(1, 2))
        
        # Loss: minimize distance (1 - similarity)
        loss = (1 - similarity).mean()
        
        return loss
    
    def _kl_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence loss"""
        # Get prototype assignments
        student_results = self.prototype_module.forward(student_features)
        teacher_results = self.prototype_module.forward(teacher_features)
        
        student_assignments = student_results['assignments']
        teacher_assignments = teacher_results['assignments']
        
        # KL divergence
        epsilon = 1e-8
        student_assignments = torch.clamp(student_assignments, min=epsilon)
        teacher_assignments = torch.clamp(teacher_assignments, min=epsilon)
        
        kl_div = torch.sum(
            teacher_assignments * (
                torch.log(teacher_assignments) - torch.log(student_assignments)
            ),
            dim=-1
        )
        
        loss = kl_div.mean()
        
        return loss


if __name__ == "__main__":
    # Test prototype module
    print("Testing Prototype Module...")
    
    # Create dummy features
    batch_size = 2
    num_patches = 100
    feature_dim = 768
    
    dummy_student_features = torch.randn(batch_size, num_patches, feature_dim)
    dummy_teacher_features = torch.randn(batch_size, num_patches, feature_dim)
    
    # Test PrototypeModule
    print("\nTesting PrototypeModule...")
    prototype_module = PrototypeModule(
        num_prototypes=50,
        prototype_dim=feature_dim,
        init_method="random"
    )
    
    # Forward pass
    results = prototype_module(dummy_student_features)
    
    print(f"Assignments shape: {results['assignments'].shape}")
    print(f"Distances shape: {results['distances'].shape}")
    print(f"Closest shape: {results['closest'].shape}")
    print(f"Similarity shape: {results['similarity'].shape}")
    
    # Test prototype loss
    print("\nTesting Prototype Loss...")
    prototype_loss = SemanticPrototypeLoss(
        num_prototypes=50,
        prototype_dim=feature_dim,
        loss_type="mse"
    )
    
    loss = prototype_loss(dummy_student_features, dummy_teacher_features)
    print(f"Prototype loss: {loss.item()}")
    
    # Test PrototypeMemoryBank
    print("\nTesting PrototypeMemoryBank...")
    memory_bank = PrototypeMemoryBank(
        capacity=1000,
        feature_dim=feature_dim
    )
    
    # Add features
    memory_features = torch.randn(100, feature_dim)
    memory_bank.add(memory_features)
    
    print(f"Memory bank size: {memory_bank.size}")
    print(f"Memory bank capacity: {memory_bank.capacity}")
    
    # Sample features
    sampled = memory_bank.sample(10)
    print(f"Sampled features shape: {sampled.shape}")
    
    print("Prototype module test completed!")
