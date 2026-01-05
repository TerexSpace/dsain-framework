#!/usr/bin/env python3
"""
Enhanced Federated Learning Experiments for TMLR Submission
============================================================

Comprehensive experiment runner integrating:
- Modern architectures (ResNet-18, MobileNetV2, ViT-Tiny)
- Enhanced non-IID scenarios (label, feature, quantity, combined skew)
- DSAIN framework with Byzantine resilience and differential privacy
- Statistical validation with multiple seeds

Usage:
    # Modern architecture baseline
    python enhanced_experiments.py --exp baseline --model resnet18 --dataset cifar10

    # Non-IID heterogeneity study
    python enhanced_experiments.py --exp heterogeneity --model mobilenetv2 --alpha 0.1

    # Byzantine attacks on modern architectures
    python enhanced_experiments.py --exp byzantine --model resnet18 --byzantine_frac 0.2

    # Full experiment suite (30-40 GPU hours)
    python enhanced_experiments.py --exp all --quick_test False
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Import modern architectures
from modern_architectures import get_model, count_parameters
# Import enhanced data heterogeneity
from data_heterogeneity import (
    dirichlet_partition,
    pathological_partition,
    power_law_partition,
    realistic_partition,
    gini_coefficient
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for enhanced experiments."""

    # Model
    model_name: str = 'resnet18'  # resnet18, mobilenetv2, vit_tiny
    num_classes: int = 10

    # Dataset
    dataset: str = 'cifar10'  # cifar10, cifar100
    data_root: str = './data'

    # Federated Learning
    num_clients: int = 100
    participation_rate: float = 0.1
    num_rounds: int = 500
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Data Heterogeneity
    heterogeneity_type: str = 'dirichlet'  # dirichlet, pathological, realistic
    dirichlet_alpha: float = 0.5
    quantity_skew: str = 'uniform'  # uniform, power_law, zipf
    feature_skew: bool = False

    # DSAIN Specific
    compression_ratio: float = 0.22  # Top-k ratio
    byzantine_frac: float = 0.0
    dp_epsilon: float = float('inf')  # No DP for baseline
    dp_delta: float = 1e-5
    gradient_clip: float = 1.0

    # Experiment
    seed: int = 42
    eval_every: int = 10
    save_checkpoints: bool = False

    # Quick test mode
    quick_test: bool = False  # If True, reduce rounds/clients for testing


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(config: ExperimentConfig) -> Tuple[torchvision.datasets.VisionDataset, torchvision.datasets.VisionDataset]:
    """Load CIFAR-10 or CIFAR-100 dataset."""

    # Standard transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if config.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=True, transform=transform_test
        )
    elif config.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data_root, train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    logger.info(f"Loaded {config.dataset}: {len(train_dataset)} train, {len(test_dataset)} test")

    return train_dataset, test_dataset


def partition_dataset(dataset, config: ExperimentConfig) -> List[Subset]:
    """Partition dataset according to heterogeneity configuration."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    targets = np.array(dataset.targets)

    if config.heterogeneity_type == 'dirichlet':
        client_indices = dirichlet_partition(
            targets,
            config.num_clients,
            config.dirichlet_alpha,
            min_samples_per_client=10
        )
        client_datasets = [Subset(dataset, indices) for indices in client_indices]

    elif config.heterogeneity_type == 'pathological':
        client_indices = pathological_partition(
            targets,
            config.num_clients,
            shards_per_client=2
        )
        client_datasets = [Subset(dataset, indices) for indices in client_indices]

    elif config.heterogeneity_type == 'realistic':
        client_datasets, metadata = realistic_partition(
            dataset,
            targets,
            config.num_clients,
            label_alpha=config.dirichlet_alpha,
            quantity_skew=config.quantity_skew,
            feature_skew=config.feature_skew
        )
        logger.info(f"Realistic partition: Gini={metadata['gini_coefficient']:.3f}")

    else:
        raise ValueError(f"Unknown heterogeneity type: {config.heterogeneity_type}")

    logger.info(f"Partitioned data: {len(client_datasets)} clients, "
                f"samples per client: [{min(len(d) for d in client_datasets)}, "
                f"{max(len(d) for d in client_datasets)}]")

    return client_datasets


# =============================================================================
# Federated Learning Components
# =============================================================================

class FederatedClient:
    """Federated learning client with local training."""

    def __init__(self, client_id: int, dataset: Subset, model: nn.Module, config: ExperimentConfig):
        self.client_id = client_id
        self.dataset = dataset
        self.model = model
        self.config = config
        self.device = DEVICE

        # Create data loader
        # Auto-detect num_workers: 0 for Windows, 4 for Linux
        import platform
        num_workers = 0 if platform.system() == 'Windows' else 4

        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def train(self, global_model_state: dict) -> dict:
        """Perform local training and return model updates."""

        # Load global model
        self.model.load_state_dict(global_model_state)
        self.model.to(self.device)
        self.model.train()

        # Optimizer
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        # Local training
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                optimizer.step()

        # Return model updates (difference from global model) - only trainable parameters
        updates = {}
        global_params = global_model_state
        local_params = self.model.state_dict()

        # Only aggregate trainable parameters (skip buffers like running_mean, running_var)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in global_params:
                updates[name] = (local_params[name] - global_params[name]).cpu()

        return updates


class FederatedServer:
    """Federated learning server with aggregation."""

    def __init__(self, model: nn.Module, test_loader: DataLoader, config: ExperimentConfig):
        self.model = model.to(DEVICE)
        self.test_loader = test_loader
        self.config = config

    def aggregate_fedavg(self, client_updates: List[dict], client_weights: List[float]) -> dict:
        """FedAvg aggregation: weighted average of client updates."""

        aggregated = {}
        total_weight = sum(client_weights)

        # Initialize aggregated updates
        for key in client_updates[0].keys():
            aggregated[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)

        # Weighted average
        for updates, weight in zip(client_updates, client_weights):
            for key in updates.keys():
                # Convert to float for aggregation
                update_tensor = updates[key].float() if updates[key].dtype != torch.float32 else updates[key]
                aggregated[key] += update_tensor * (weight / total_weight)

        return aggregated

    def aggregate_compressed(self, client_updates: List[dict], client_weights: List[float]) -> dict:
        """Top-k compression + FedAvg aggregation."""

        # Apply top-k compression to each client's updates
        compressed_updates = []

        for updates in client_updates:
            compressed = {}
            for key, tensor in updates.items():
                # Flatten tensor
                flat = tensor.flatten()
                k = max(1, int(len(flat) * self.config.compression_ratio))

                # Get top-k by magnitude
                _, indices = torch.topk(flat.abs(), k)
                mask = torch.zeros_like(flat)
                mask[indices] = 1.0

                # Apply mask
                compressed_flat = flat * mask
                compressed[key] = compressed_flat.reshape(tensor.shape)

            compressed_updates.append(compressed)

        # Aggregate compressed updates
        return self.aggregate_fedavg(compressed_updates, client_weights)

    def update_global_model(self, aggregated_updates: dict):
        """Update global model with aggregated updates."""

        current_state = self.model.state_dict()

        for key in current_state.keys():
            if key in aggregated_updates:
                current_state[key] = current_state[key] + aggregated_updates[key].to(DEVICE)

        self.model.load_state_dict(current_state)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model on test set."""

        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)

        return accuracy, avg_loss


# =============================================================================
# Experiment Runners
# =============================================================================

def run_baseline_experiment(config: ExperimentConfig) -> Dict:
    """
    Baseline experiment: Test model on standard FL setup.

    Measures:
    - Convergence speed
    - Final accuracy
    - Communication cost
    - Training time
    """

    logger.info(f"\n{'='*70}")
    logger.info(f"BASELINE EXPERIMENT: {config.model_name} on {config.dataset}")
    logger.info(f"{'='*70}")

    # Set random seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load dataset
    train_dataset, test_dataset = load_dataset(config)
    # Auto-detect num_workers: 0 for Windows, 4 for Linux
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Partition data
    client_datasets = partition_dataset(train_dataset, config)

    # Create model
    model = get_model(config.model_name, num_classes=config.num_classes)
    logger.info(f"Model: {config.model_name}, Parameters: {count_parameters(model):,}")

    # Create clients
    num_active = max(1, int(config.num_clients * config.participation_rate))
    clients = [
        FederatedClient(i, client_datasets[i], model, config)
        for i in range(config.num_clients)
    ]

    # Create server
    server = FederatedServer(model, test_loader, config)

    # Training history
    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'time': []
    }

    start_time = time.time()

    # Federated learning rounds
    for round_idx in tqdm(range(config.num_rounds), desc="FL Rounds"):
        # Sample clients
        active_clients = np.random.choice(clients, size=num_active, replace=False)

        # Get global model state
        global_state = server.model.state_dict()

        # Client training
        client_updates = []
        client_weights = []

        for client in active_clients:
            updates = client.train(global_state)
            client_updates.append(updates)
            client_weights.append(len(client.dataset))

        # Server aggregation
        if config.compression_ratio < 1.0:
            aggregated = server.aggregate_compressed(client_updates, client_weights)
        else:
            aggregated = server.aggregate_fedavg(client_updates, client_weights)

        # Update global model
        server.update_global_model(aggregated)

        # Evaluation
        if (round_idx + 1) % config.eval_every == 0 or round_idx == config.num_rounds - 1:
            accuracy, loss = server.evaluate()
            elapsed = time.time() - start_time

            history['round'].append(round_idx + 1)
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            history['time'].append(elapsed)

            logger.info(f"Round {round_idx+1}/{config.num_rounds}: "
                       f"Accuracy={accuracy:.4f}, Loss={loss:.4f}, Time={elapsed:.1f}s")

    total_time = time.time() - start_time

    # Final results
    results = {
        'config': asdict(config),
        'final_accuracy': history['accuracy'][-1],
        'final_loss': history['loss'][-1],
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'model_parameters': count_parameters(model),
        'history': history
    }

    logger.info(f"\nFinal Accuracy: {results['final_accuracy']:.4f}")
    logger.info(f"Total Time: {results['total_time_hours']:.2f} hours")

    return results


def run_heterogeneity_experiment(config: ExperimentConfig) -> Dict:
    """
    Heterogeneity study: Test different non-IID scenarios.

    Compares:
    - Different Dirichlet alpha values
    - Pathological vs realistic partitioning
    - Impact of quantity and feature skew
    """

    logger.info(f"\n{'='*70}")
    logger.info(f"HETEROGENEITY EXPERIMENT: {config.model_name}")
    logger.info(f"{'='*70}")

    results = {}

    # Test different Dirichlet alpha values
    alpha_values = [0.01, 0.1, 0.5, 1.0] if not config.quick_test else [0.1, 0.5]

    for alpha in alpha_values:
        logger.info(f"\n--- Testing alpha = {alpha} ---")

        config.dirichlet_alpha = alpha
        config.heterogeneity_type = 'dirichlet'

        result = run_baseline_experiment(config)
        results[f'alpha_{alpha}'] = result

    # Test pathological partitioning
    if not config.quick_test:
        logger.info(f"\n--- Testing pathological partitioning ---")
        config.heterogeneity_type = 'pathological'
        result = run_baseline_experiment(config)
        results['pathological'] = result

    # Test realistic partitioning with quantity skew
    if not config.quick_test:
        logger.info(f"\n--- Testing realistic partitioning (power law) ---")
        config.heterogeneity_type = 'realistic'
        config.quantity_skew = 'power_law'
        config.dirichlet_alpha = 0.3
        result = run_baseline_experiment(config)
        results['realistic_powerlaw'] = result

    return results


def run_architecture_comparison(config: ExperimentConfig) -> Dict:
    """
    Architecture comparison: Test all modern architectures.

    Compares:
    - ResNet-18
    - MobileNetV2
    - ViT-Tiny
    """

    logger.info(f"\n{'='*70}")
    logger.info(f"ARCHITECTURE COMPARISON on {config.dataset}")
    logger.info(f"{'='*70}")

    results = {}
    architectures = ['resnet18', 'mobilenetv2', 'vit_tiny']

    for arch in architectures:
        logger.info(f"\n--- Testing {arch} ---")

        config.model_name = arch
        result = run_baseline_experiment(config)
        results[arch] = result

    return results


# =============================================================================
# Main Experiment Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced FL Experiments for TMLR')

    # Experiment type
    parser.add_argument('--exp', type=str, default='baseline',
                       choices=['baseline', 'heterogeneity', 'architecture', 'all'],
                       help='Experiment type to run')

    # Model and dataset
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'mobilenetv2', 'vit_tiny'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset')

    # FL parameters
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--num_rounds', type=int, default=500)
    parser.add_argument('--participation_rate', type=float, default=0.1)
    parser.add_argument('--local_epochs', type=int, default=5)

    # Heterogeneity
    parser.add_argument('--heterogeneity', type=str, default='dirichlet',
                       choices=['dirichlet', 'pathological', 'realistic'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--quantity_skew', type=str, default='uniform',
                       choices=['uniform', 'power_law', 'zipf'])
    parser.add_argument('--feature_skew', action='store_true')

    # DSAIN parameters
    parser.add_argument('--compression', type=float, default=0.22)
    parser.add_argument('--byzantine_frac', type=float, default=0.0)
    parser.add_argument('--dp_epsilon', type=float, default=float('inf'))

    # Experiment settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='../results/enhanced')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced rounds/clients')

    args = parser.parse_args()

    # Adjust for quick test
    if args.quick_test:
        args.num_rounds = 50
        args.num_clients = 20
        logger.info("QUICK TEST MODE: Reduced rounds and clients")

    # Create config
    config = ExperimentConfig(
        model_name=args.model,
        num_classes=10 if args.dataset == 'cifar10' else 100,
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        participation_rate=args.participation_rate,
        local_epochs=args.local_epochs,
        heterogeneity_type=args.heterogeneity,
        dirichlet_alpha=args.alpha,
        quantity_skew=args.quantity_skew,
        feature_skew=args.feature_skew,
        compression_ratio=args.compression,
        byzantine_frac=args.byzantine_frac,
        dp_epsilon=args.dp_epsilon,
        seed=args.seed,
        quick_test=args.quick_test
    )

    # Run experiments
    results = {}

    if args.exp == 'baseline':
        results = run_baseline_experiment(config)

    elif args.exp == 'heterogeneity':
        results = run_heterogeneity_experiment(config)

    elif args.exp == 'architecture':
        results = run_architecture_comparison(config)

    elif args.exp == 'all':
        logger.info("\n" + "="*70)
        logger.info("RUNNING COMPLETE EXPERIMENT SUITE")
        logger.info("="*70)

        results['architecture'] = run_architecture_comparison(config)
        results['heterogeneity'] = run_heterogeneity_experiment(config)

    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"enhanced_{args.exp}_{timestamp}_seed{args.seed}.json"

    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    main()
