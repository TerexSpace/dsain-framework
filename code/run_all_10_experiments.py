#!/usr/bin/env python3
"""
Run All 10 Critical Experiments for TMLR Publication
====================================================

Single script to run all experiments sequentially on cloud Pod (RunPod compatible).
Estimated time: 33 hours on A5000 GPU
Estimated cost: $9.24 on RunPod A5000 @ $0.28/hour

Usage:
    python run_all_10_experiments.py

This will run all 10 experiments and save results to results/final_experiments/
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
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

# Get script directory for relative imports
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Import required modules
from modern_architectures import get_model
from data_heterogeneity import dirichlet_partition
from byzantine_attacks import LabelFlippingAttack
from privacy_accounting import DPOptimizer

# Set up output directories before logging
RESULTS_DIR = SCRIPT_DIR.parent / 'results' / 'final_experiments'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SCRIPT_DIR.parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / 'experiment_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Auto-detect num_workers: 0 for Windows (multiprocessing issues), 4 for Linux
NUM_WORKERS = 0 if sys.platform == 'win32' else 4

# Enable CUDA deterministic mode for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger.info(f"Using device: {DEVICE}")
logger.info(f"Platform: {sys.platform}")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Results directory: {RESULTS_DIR}")
logger.info(f"Data directory: {DATA_DIR}")
logger.info(f"DataLoader num_workers: {NUM_WORKERS}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"CUDA version: {torch.version.cuda}")


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Experiment ID
    exp_id: str
    exp_name: str

    # Model
    model_name: str = 'resnet18'
    num_classes: int = 10

    # Dataset
    dataset: str = 'cifar10'
    data_root: str = str(DATA_DIR)

    # Federated Learning
    num_clients: int = 20
    participation_rate: float = 0.25
    num_rounds: int = 500
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Data Heterogeneity
    heterogeneity_type: str = 'dirichlet'
    dirichlet_alpha: float = 0.5

    # DSAIN vs FedAvg
    compression_ratio: float = 0.22  # 0.22 = DSAIN, 1.0 = FedAvg
    byzantine_frac: float = 0.0
    byzantine_defense: bool = True  # Always use ByzFed when Byzantine > 0

    # Privacy
    dp_epsilon: float = float('inf')  # inf = no DP
    dp_delta: float = 1e-5
    gradient_clip: float = 1.0

    # Experiment settings
    seed: int = 42
    eval_every: int = 25
    save_checkpoints: bool = False


# =============================================================================
# Top-k Compression (DSAIN Feature)
# =============================================================================

def topk_compress(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
    """Top-k gradient compression."""
    if ratio >= 1.0:
        return tensor

    k = max(1, int(tensor.numel() * ratio))
    flat = tensor.flatten()
    _, indices = torch.topk(flat.abs(), k)

    compressed = torch.zeros_like(flat)
    compressed[indices] = flat[indices]

    return compressed.reshape(tensor.shape)


# =============================================================================
# Byzantine-Resilient Aggregation (ByzFed)
# =============================================================================

def geometric_median_aggregate(gradients: List[torch.Tensor], max_iter: int = 10) -> torch.Tensor:
    """Geometric median aggregation for Byzantine resilience."""
    if len(gradients) == 0:
        raise ValueError("No gradients to aggregate")

    # Initialize with mean
    median = torch.mean(torch.stack(gradients), dim=0)

    for _ in range(max_iter):
        norms = torch.stack([torch.norm(g - median) for g in gradients])
        weights = 1.0 / (norms + 1e-8)
        weights = weights / weights.sum()

        median_new = sum(w * g for w, g in zip(weights, gradients))

        if torch.norm(median_new - median) < 1e-6:
            break
        median = median_new

    return median


# =============================================================================
# Federated Client
# =============================================================================

class FederatedClient:
    """Federated learning client."""

    def __init__(self, client_id: int, config: ExperimentConfig, train_loader: DataLoader):
        self.client_id = client_id
        self.config = config
        self.train_loader = train_loader

        # Model
        self.model = get_model(config.model_name, num_classes=config.num_classes).to(DEVICE)

        # Optimizer
        if config.dp_epsilon < float('inf'):
            self.optimizer = DPOptimizer(
                self.model.parameters(),
                lr=config.learning_rate,
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                clip_norm=config.gradient_clip
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )

        self.criterion = nn.CrossEntropyLoss()
        self.is_byzantine = False
        self.byzantine_attack = None

    def set_byzantine(self, is_byzantine: bool):
        """Mark client as Byzantine attacker."""
        self.is_byzantine = is_byzantine
        if is_byzantine:
            self.byzantine_attack = LabelFlippingAttack(num_classes=self.config.num_classes)

    def set_weights(self, global_weights: Dict):
        """Load global model weights."""
        self.model.load_state_dict(global_weights)

    def train_epoch(self) -> float:
        """Train for one local epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for data, target in self.train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Byzantine attack: flip labels
            if self.is_byzantine and self.byzantine_attack:
                target = self.byzantine_attack.apply(target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def get_gradient(self, global_weights: Dict) -> Dict:
        """Compute gradient (delta from global model)."""
        gradient = {}
        local_weights = self.model.state_dict()

        for key in global_weights.keys():
            gradient[key] = local_weights[key] - global_weights[key]

            # Apply compression if DSAIN
            if self.config.compression_ratio < 1.0:
                gradient[key] = topk_compress(gradient[key], self.config.compression_ratio)

        return gradient


# =============================================================================
# Federated Server
# =============================================================================

class FederatedServer:
    """Federated learning server."""

    def __init__(self, config: ExperimentConfig, test_loader: DataLoader):
        self.config = config
        self.test_loader = test_loader

        self.model = get_model(config.model_name, num_classes=config.num_classes).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate(self, gradients: List[Dict]) -> Dict:
        """Aggregate client gradients."""
        if len(gradients) == 0:
            return {}

        aggregated = {}

        for key in gradients[0].keys():
            grad_list = [g[key] for g in gradients]

            # Use geometric median if Byzantine defense enabled
            if self.config.byzantine_defense and self.config.byzantine_frac > 0:
                aggregated[key] = geometric_median_aggregate(grad_list)
            else:
                aggregated[key] = torch.mean(torch.stack(grad_list), dim=0)

        return aggregated

    def apply_update(self, aggregated_gradient: Dict):
        """Apply aggregated gradient to global model."""
        current_weights = self.model.state_dict()

        for key in current_weights.keys():
            if key in aggregated_gradient:
                current_weights[key] += aggregated_gradient[key]

        self.model.load_state_dict(current_weights)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)

        return accuracy, avg_loss

    def get_weights(self) -> Dict:
        """Get current global model weights."""
        return self.model.state_dict()


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single federated learning experiment."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Experiment: {config.exp_id} - {config.exp_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Config: {asdict(config)}")

    start_time = time.time()

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    # Load dataset
    logger.info("Loading dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    if config.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=config.data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

    # Partition data across clients
    logger.info(f"Partitioning data (Dirichlet α={config.dirichlet_alpha})...")
    # Convert targets to numpy array (CIFAR-10 returns list)
    targets_np = np.array(trainset.targets)
    client_indices = dirichlet_partition(
        targets_np,
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        min_samples_per_client=1  # Allow small client datasets for heterogeneous scenarios
    )

    # Create clients
    clients = []
    for i in range(config.num_clients):
        client_dataset = Subset(trainset, client_indices[i])
        client_loader = DataLoader(
            client_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        client = FederatedClient(i, config, client_loader)

        # Mark Byzantine clients
        if i < int(config.num_clients * config.byzantine_frac):
            client.set_byzantine(True)
            logger.info(f"Client {i} marked as Byzantine")

        clients.append(client)

    logger.info(f"Created {len(clients)} clients ({int(config.byzantine_frac * 100)}% Byzantine)")

    # Create server
    server = FederatedServer(config, test_loader)

    # Training loop
    logger.info(f"\nStarting {config.num_rounds} rounds of federated learning...")

    history = {
        'round': [],
        'accuracy': [],
        'loss': [],
        'time': []
    }

    for round_num in range(1, config.num_rounds + 1):
        round_start = time.time()

        # Select participating clients
        num_selected = max(1, int(config.num_clients * config.participation_rate))
        selected_indices = np.random.choice(len(clients), num_selected, replace=False)
        selected_clients = [clients[i] for i in selected_indices]

        # Get global weights
        global_weights = server.get_weights()

        # Client training
        gradients = []
        for client in selected_clients:
            client.set_weights(global_weights)

            # Train for local epochs
            for _ in range(config.local_epochs):
                client.train_epoch()

            # Get gradient
            gradient = client.get_gradient(global_weights)
            gradients.append(gradient)

        # Server aggregation
        aggregated_gradient = server.aggregate(gradients)
        server.apply_update(aggregated_gradient)

        # Evaluate periodically
        if round_num % config.eval_every == 0 or round_num == config.num_rounds:
            accuracy, loss = server.evaluate()
            elapsed = time.time() - start_time

            history['round'].append(round_num)
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            history['time'].append(elapsed)

            logger.info(
                f"Round {round_num:3d}/{config.num_rounds} | "
                f"Acc: {accuracy:.4f} | Loss: {loss:.4f} | "
                f"Time: {elapsed/60:.1f}m"
            )

    total_time = time.time() - start_time
    final_accuracy = history['accuracy'][-1]
    final_loss = history['loss'][-1]

    logger.info(f"\n{'='*80}")
    logger.info(f"Experiment Complete: {config.exp_id}")
    logger.info(f"Final Accuracy: {final_accuracy:.4f}")
    logger.info(f"Final Loss: {final_loss:.4f}")
    logger.info(f"Total Time: {total_time/3600:.2f} hours")
    logger.info(f"{'='*80}\n")

    # Prepare results
    results = {
        'config': asdict(config),
        'final_accuracy': final_accuracy,
        'final_loss': final_loss,
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    return results


# =============================================================================
# Define All 10 Experiments
# =============================================================================

def get_all_experiments() -> List[ExperimentConfig]:
    """Define all 10 critical experiments."""

    base_config = {
        'model_name': 'resnet18',
        'num_classes': 10,
        'dataset': 'cifar10',
        'num_clients': 20,
        'participation_rate': 0.25,
        'num_rounds': 500,
        'local_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.01,
        'seed': 42,
        'eval_every': 25
    }

    experiments = [
        # E1: DSAIN α=0.5, clean
        ExperimentConfig(
            exp_id='E1',
            exp_name='DSAIN_alpha0.5_clean',
            dirichlet_alpha=0.5,
            compression_ratio=0.22,
            byzantine_frac=0.0,
            byzantine_defense=True,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E2: FedAvg α=0.5, clean
        ExperimentConfig(
            exp_id='E2',
            exp_name='FedAvg_alpha0.5_clean',
            dirichlet_alpha=0.5,
            compression_ratio=1.0,  # No compression = FedAvg
            byzantine_frac=0.0,
            byzantine_defense=False,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E3: DSAIN α=0.5, 20% Byzantine
        ExperimentConfig(
            exp_id='E3',
            exp_name='DSAIN_alpha0.5_byz20',
            dirichlet_alpha=0.5,
            compression_ratio=0.22,
            byzantine_frac=0.2,
            byzantine_defense=True,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E4: FedAvg α=0.5, 20% Byzantine
        ExperimentConfig(
            exp_id='E4',
            exp_name='FedAvg_alpha0.5_byz20',
            dirichlet_alpha=0.5,
            compression_ratio=1.0,
            byzantine_frac=0.2,
            byzantine_defense=False,  # FedAvg has no defense
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E5: DSAIN α=1.0, clean
        ExperimentConfig(
            exp_id='E5',
            exp_name='DSAIN_alpha1.0_clean',
            dirichlet_alpha=1.0,
            compression_ratio=0.22,
            byzantine_frac=0.0,
            byzantine_defense=True,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E6: FedAvg α=1.0, clean
        ExperimentConfig(
            exp_id='E6',
            exp_name='FedAvg_alpha1.0_clean',
            dirichlet_alpha=1.0,
            compression_ratio=1.0,
            byzantine_frac=0.0,
            byzantine_defense=False,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E7: DSAIN α=0.1, clean
        ExperimentConfig(
            exp_id='E7',
            exp_name='DSAIN_alpha0.1_clean',
            dirichlet_alpha=0.1,
            compression_ratio=0.22,
            byzantine_frac=0.0,
            byzantine_defense=True,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E8: FedAvg α=0.1, clean
        ExperimentConfig(
            exp_id='E8',
            exp_name='FedAvg_alpha0.1_clean',
            dirichlet_alpha=0.1,
            compression_ratio=1.0,
            byzantine_frac=0.0,
            byzantine_defense=False,
            dp_epsilon=float('inf'),
            **base_config
        ),

        # E9: DSAIN α=0.5, ε=2.0 privacy
        ExperimentConfig(
            exp_id='E9',
            exp_name='DSAIN_alpha0.5_dp2.0',
            dirichlet_alpha=0.5,
            compression_ratio=0.22,
            byzantine_frac=0.0,
            byzantine_defense=True,
            dp_epsilon=2.0,
            dp_delta=1e-5,
            gradient_clip=1.0,
            **base_config
        ),

        # E10: DSAIN α=0.5, 10% Byzantine
        ExperimentConfig(
            exp_id='E10',
            exp_name='DSAIN_alpha0.5_byz10',
            dirichlet_alpha=0.5,
            compression_ratio=0.22,
            byzantine_frac=0.1,
            byzantine_defense=True,
            dp_epsilon=float('inf'),
            **base_config
        ),
    ]

    return experiments


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all 10 experiments sequentially."""

    logger.info("="*80)
    logger.info("TMLR Experiment Suite - All 10 Critical Experiments")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("="*80)

    # Use the predefined output directory
    output_dir = RESULTS_DIR
    logger.info(f"Output directory: {output_dir}")

    # Get all experiments
    experiments = get_all_experiments()
    logger.info(f"\nTotal experiments to run: {len(experiments)}")

    # Run each experiment
    all_results = {}
    total_start = time.time()

    for i, config in enumerate(experiments, 1):
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"EXPERIMENT {i}/{len(experiments)}")
        logger.info(f"{'#'*80}\n")

        try:
            result = run_single_experiment(config)
            all_results[config.exp_id] = result

            # Save individual result
            result_file = output_dir / f"{config.exp_id}_{config.exp_name}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved result to {result_file}")

        except Exception as e:
            logger.error(f"Experiment {config.exp_id} failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_results[config.exp_id] = {'error': str(e)}
            continue

    total_time = time.time() - total_start

    # Save summary
    summary = {
        'total_experiments': len(experiments),
        'successful': sum(1 for r in all_results.values() if 'error' not in r),
        'failed': sum(1 for r in all_results.values() if 'error' in r),
        'total_time_hours': total_time / 3600,
        'results': all_results,
        'timestamp': datetime.now().isoformat()
    }

    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Successful: {summary['successful']}/{summary['total_experiments']}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("="*80)

    # Print quick results table
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Exp':<6} {'Name':<30} {'Acc':<8} {'Time (h)':<10}")
    logger.info("-"*80)

    for exp_id, result in all_results.items():
        if 'error' not in result:
            name = result['config']['exp_name']
            acc = result['final_accuracy']
            time_h = result['total_time_hours']
            logger.info(f"{exp_id:<6} {name:<30} {acc:.4f}   {time_h:.2f}")
        else:
            logger.info(f"{exp_id:<6} FAILED: {result['error'][:40]}")

    logger.info("="*80)


if __name__ == '__main__':
    main()
