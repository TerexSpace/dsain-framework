#!/usr/bin/env python3
"""
Sophisticated Byzantine Attack Implementations
===============================================

This module implements state-of-the-art Byzantine attacks for evaluating
Byzantine-resilient federated learning algorithms.

References:
- Bit-flipping: Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust FL", 2020
- Label flipping: Tolpegin et al., "Data Poisoning Attacks Against FL Systems", 2020
- Optimization-based: Baruch et al., "A Little Is Enough", 2019
- Min-max: Shejwalkar et al., "Manipulating the Byzantine", 2021

Author: Almas Ospanov
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ByzantineAttack(ABC):
    """Base class for Byzantine attacks."""

    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        self.round_count = 0

    @abstractmethod
    def attack(self, delta: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Apply Byzantine attack to model update.

        Args:
            delta: Honest model update
            **kwargs: Additional attack-specific parameters

        Returns:
            Malicious model update
        """
        pass

    def increment_round(self):
        """Track rounds for adaptive attacks."""
        self.round_count += 1


class SignFlippingAttack(ByzantineAttack):
    """
    Sign-flipping attack: negate the gradient update.
    Simple but effective against naive aggregation.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__("SignFlipping")
        self.scale = scale

    def attack(self, delta: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Flip signs and optionally scale."""
        attacked = {}
        for name, tensor in delta.items():
            attacked[name] = -self.scale * tensor
        return attacked


class BitFlippingAttack(ByzantineAttack):
    """
    Bit-flipping attack: flip random bits in the gradient.
    More subtle than sign flipping.
    """

    def __init__(self, flip_prob: float = 0.1):
        super().__init__("BitFlipping")
        self.flip_prob = flip_prob

    def attack(self, delta: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Randomly flip bits with given probability."""
        attacked = {}
        for name, tensor in delta.items():
            # Create bit mask
            mask = torch.rand_like(tensor) < self.flip_prob
            # Flip sign where mask is True
            attacked[name] = torch.where(mask, -tensor, tensor)
        return attacked


class GaussianNoiseAttack(ByzantineAttack):
    """
    Add large Gaussian noise to gradients.
    """

    def __init__(self, noise_scale: float = 10.0):
        super().__init__("GaussianNoise")
        self.noise_scale = noise_scale

    def attack(self, delta: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Add scaled Gaussian noise."""
        attacked = {}
        for name, tensor in delta.items():
            noise = torch.randn_like(tensor) * self.noise_scale
            attacked[name] = tensor + noise
        return attacked


class LittleIsEnoughAttack(ByzantineAttack):
    """
    "A Little Is Enough" attack (Baruch et al., 2019).

    Optimization-based attack that crafts malicious updates to evade
    Byzantine-resilient aggregation while maximizing damage.

    Strategy: Malicious clients estimate the mean of honest updates,
    then send updates slightly outside the distribution but not too far
    to avoid detection.
    """

    def __init__(self, num_byzantine: int, num_honest: int, epsilon: float = 0.5):
        super().__init__("LittleIsEnough")
        self.num_byzantine = num_byzantine
        self.num_honest = num_honest
        self.epsilon = epsilon  # Perturbation magnitude

    def attack(
        self,
        delta: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Craft malicious update based on estimated honest mean.

        If honest_updates are provided (omniscient attacker), use them.
        Otherwise, use the current delta as estimate.
        """
        if honest_updates is None or len(honest_updates) == 0:
            # Fallback: just scale and flip
            attacked = {}
            for name, tensor in delta.items():
                attacked[name] = -(1 + self.epsilon) * tensor
            return attacked

        # Estimate honest mean
        mean_honest = {}
        for name in delta.keys():
            stacked = torch.stack([u[name] for u in honest_updates])
            mean_honest[name] = stacked.mean(dim=0)

        # Estimate standard deviation
        std_honest = {}
        for name in delta.keys():
            stacked = torch.stack([u[name] for u in honest_updates])
            std_honest[name] = stacked.std(dim=0) + 1e-8

        # Craft attack: move along negative gradient direction
        # but stay within reasonable bounds
        attacked = {}
        for name in delta.keys():
            # Direction: opposite of honest mean
            direction = -mean_honest[name]
            direction_norm = torch.norm(direction)

            if direction_norm > 1e-8:
                direction = direction / direction_norm

            # Magnitude: epsilon times the honest std
            magnitude = self.epsilon * torch.norm(mean_honest[name])

            # Malicious update
            attacked[name] = mean_honest[name] + magnitude * direction

        return attacked


class MinMaxAttack(ByzantineAttack):
    """
    Min-Max attack (Shejwalkar et al., 2021).

    Sophisticated attack that optimizes Byzantine updates to:
    1. Maximize distance to honest mean (damage)
    2. Minimize distance to each other (coordination)
    3. Stay within detection threshold
    """

    def __init__(self, num_byzantine: int, lambda_coord: float = 0.5):
        super().__init__("MinMax")
        self.num_byzantine = num_byzantine
        self.lambda_coord = lambda_coord  # Coordination weight
        self.byzantine_updates = []  # Track other Byzantine updates

    def attack(
        self,
        delta: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize Byzantine update via min-max objective.
        """
        if honest_updates is None or len(honest_updates) == 0:
            # Fallback
            return SignFlippingAttack(scale=5.0).attack(delta)

        # Estimate honest statistics
        mean_honest = {}
        for name in delta.keys():
            stacked = torch.stack([u[name] for u in honest_updates])
            mean_honest[name] = stacked.mean(dim=0)

        # Objective: maximize distance to honest mean
        # while coordinating with other Byzantine clients
        attacked = {}

        for name in delta.keys():
            # Start from opposite direction
            direction = -mean_honest[name]

            # If we have previous Byzantine updates, coordinate
            if len(self.byzantine_updates) > 0:
                byz_mean = torch.stack([b[name] for b in self.byzantine_updates]).mean(dim=0)
                # Blend: move away from honest, towards Byzantine consensus
                direction = (1 - self.lambda_coord) * direction + self.lambda_coord * (byz_mean - mean_honest[name])

            # Normalize and scale
            dir_norm = torch.norm(direction)
            if dir_norm > 1e-8:
                direction = direction / dir_norm

            # Scale by honest update magnitude
            scale = torch.norm(mean_honest[name]) * 3.0  # Aggressive but not too obvious

            attacked[name] = mean_honest[name] + scale * direction

        # Store this Byzantine update for coordination
        self.byzantine_updates.append(attacked)
        if len(self.byzantine_updates) > self.num_byzantine:
            self.byzantine_updates.pop(0)

        return attacked


class IPMAttack(ByzantineAttack):
    """
    Inner Product Manipulation (IPM) attack.

    Crafts updates that have negative inner product with honest updates,
    slowing down convergence.
    """

    def __init__(self, scale: float = 2.0):
        super().__init__("IPM")
        self.scale = scale

    def attack(
        self,
        delta: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Craft update orthogonal/opposite to honest updates."""
        if honest_updates is None or len(honest_updates) == 0:
            return SignFlippingAttack(scale=self.scale).attack(delta)

        # Compute mean honest update
        mean_honest = {}
        for name in delta.keys():
            stacked = torch.stack([u[name] for u in honest_updates])
            mean_honest[name] = stacked.mean(dim=0)

        # Send opposite direction with scaling
        attacked = {}
        for name in delta.keys():
            attacked[name] = -self.scale * mean_honest[name]

        return attacked


class AdaptiveAttack(ByzantineAttack):
    """
    Adaptive attack that changes strategy based on round number.

    Early rounds: Stealthy (small perturbations)
    Middle rounds: Aggressive (large perturbations)
    Late rounds: Strategic (targeted)
    """

    def __init__(self, total_rounds: int = 200):
        super().__init__("Adaptive")
        self.total_rounds = total_rounds
        self.phase_threshold_1 = int(0.3 * total_rounds)
        self.phase_threshold_2 = int(0.7 * total_rounds)

    def attack(
        self,
        delta: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Select attack strategy based on phase."""

        if self.round_count < self.phase_threshold_1:
            # Phase 1: Stealthy - small noise
            attack_impl = GaussianNoiseAttack(noise_scale=0.5)
        elif self.round_count < self.phase_threshold_2:
            # Phase 2: Aggressive - sign flipping
            attack_impl = SignFlippingAttack(scale=5.0)
        else:
            # Phase 3: Strategic - optimization-based
            attack_impl = LittleIsEnoughAttack(
                num_byzantine=kwargs.get('num_byzantine', 1),
                num_honest=kwargs.get('num_honest', 10),
                epsilon=1.0
            )

        return attack_impl.attack(delta, honest_updates=honest_updates)


class LabelFlippingAttack(ByzantineAttack):
    """
    Label flipping attack for data poisoning.

    Note: This modifies training data labels, not gradients.
    Should be applied at data loading time.
    """

    def __init__(self, flip_fraction: float = 0.3, num_classes: int = 10):
        super().__init__("LabelFlipping")
        self.flip_fraction = flip_fraction
        self.num_classes = num_classes

    def flip_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Flip labels randomly."""
        flipped = labels.clone()
        n = len(labels)
        flip_indices = torch.rand(n) < self.flip_fraction

        # For flipped labels, choose random wrong class
        for i in torch.where(flip_indices)[0]:
            # Pick a different class
            wrong_classes = list(range(self.num_classes))
            wrong_classes.remove(labels[i].item())
            flipped[i] = np.random.choice(wrong_classes)

        return flipped

    def attack(self, delta: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """For consistency with interface, but label flipping affects data not gradients."""
        logger.warning("LabelFlippingAttack should be applied to labels, not gradients.")
        return delta


class FallOfEmpiresAttack(ByzantineAttack):
    """
    Fall of Empires attack: Byzantine clients send the same malicious update
    to maximize impact on geometric median.
    """

    def __init__(self, scale: float = 10.0):
        super().__init__("FallOfEmpires")
        self.scale = scale
        self.shared_malicious_update = None

    def attack(
        self,
        delta: Dict[str, torch.Tensor],
        honest_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        byzantine_id: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """All Byzantine clients send identical malicious update."""

        # First Byzantine client computes the malicious update
        if byzantine_id == 0 or self.shared_malicious_update is None:
            if honest_updates and len(honest_updates) > 0:
                # Estimate honest mean and flip
                mean_honest = {}
                for name in delta.keys():
                    stacked = torch.stack([u[name] for u in honest_updates])
                    mean_honest[name] = stacked.mean(dim=0)

                self.shared_malicious_update = {}
                for name in delta.keys():
                    self.shared_malicious_update[name] = -self.scale * mean_honest[name]
            else:
                # Fallback
                self.shared_malicious_update = {}
                for name, tensor in delta.items():
                    self.shared_malicious_update[name] = -self.scale * tensor

        # All Byzantine clients return the same malicious update
        return {k: v.clone() for k, v in self.shared_malicious_update.items()}


# Factory function
def get_byzantine_attack(
    attack_type: str,
    num_byzantine: int = 1,
    num_honest: int = 10,
    **kwargs
) -> ByzantineAttack:
    """
    Factory function to create Byzantine attack instances.

    Args:
        attack_type: Name of attack
        num_byzantine: Number of Byzantine clients
        num_honest: Number of honest clients
        **kwargs: Additional attack-specific parameters

    Returns:
        ByzantineAttack instance
    """
    attacks = {
        'sign_flipping': lambda: SignFlippingAttack(scale=kwargs.get('scale', 5.0)),
        'bit_flipping': lambda: BitFlippingAttack(flip_prob=kwargs.get('flip_prob', 0.1)),
        'gaussian_noise': lambda: GaussianNoiseAttack(noise_scale=kwargs.get('noise_scale', 10.0)),
        'little_is_enough': lambda: LittleIsEnoughAttack(num_byzantine, num_honest, epsilon=kwargs.get('epsilon', 0.5)),
        'min_max': lambda: MinMaxAttack(num_byzantine, lambda_coord=kwargs.get('lambda_coord', 0.5)),
        'ipm': lambda: IPMAttack(scale=kwargs.get('scale', 2.0)),
        'adaptive': lambda: AdaptiveAttack(total_rounds=kwargs.get('total_rounds', 200)),
        'label_flipping': lambda: LabelFlippingAttack(
            flip_fraction=kwargs.get('flip_fraction', 0.3),
            num_classes=kwargs.get('num_classes', 10)
        ),
        'fall_of_empires': lambda: FallOfEmpiresAttack(scale=kwargs.get('scale', 10.0)),
    }

    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}. Available: {list(attacks.keys())}")

    return attacks[attack_type]()


# Utility functions for evaluation
def compute_attack_success_rate(
    clean_accuracy: float,
    attacked_accuracy: float,
    threshold: float = 0.1
) -> float:
    """
    Compute attack success rate.

    Attack is successful if accuracy drops by more than threshold.
    """
    accuracy_drop = clean_accuracy - attacked_accuracy
    return float(accuracy_drop > threshold)


def evaluate_attack_stealthiness(
    clean_gradients: List[torch.Tensor],
    attacked_gradients: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Evaluate how stealthy an attack is.

    Returns:
        Dictionary with stealthiness metrics
    """
    # Compute distances
    distances = []
    for clean, attacked in zip(clean_gradients, attacked_gradients):
        dist = torch.norm(clean - attacked).item()
        distances.append(dist)

    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances)
    }
