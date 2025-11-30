#!/usr/bin/env python3
"""
DSAIN: Distributed Sovereign AI Network
========================================

Implementation of the FedSov algorithm with ByzFed aggregation mechanism.

Requirements:
    pip install torch numpy scipy

Usage:
    python dsain.py --num_clients 100 --num_rounds 200 --byzantine_frac 0.1

Authors: [Author Names]
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - visualization will be disabled")


@dataclass
class FedSovConfig:
    """Configuration for FedSov algorithm."""
    num_clients: int = 100
    participation_rate: float = 0.1
    local_epochs: int = 5
    learning_rate: float = 0.01
    compression_ratio: float = 0.1
    momentum: float = 0.9
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    gradient_clip: float = 1.0
    byzantine_threshold: float = 3.0
    reputation_decay: float = 0.9


class GradientCompressor:
    """Top-k gradient compression with error feedback."""
    
    def __init__(self, compression_ratio: float):
        self.k_ratio = compression_ratio
        self.error_feedback = {}
    
    def compress(self, gradient: np.ndarray, client_id: int) -> np.ndarray:
        """Apply top-k sparsification with error feedback."""
        # Add error feedback from previous round
        if client_id in self.error_feedback:
            gradient = gradient + self.error_feedback[client_id]
        
        # Top-k sparsification
        k = max(1, int(len(gradient) * self.k_ratio))
        top_k_indices = np.argsort(np.abs(gradient))[-k:]
        
        compressed = np.zeros_like(gradient)
        compressed[top_k_indices] = gradient[top_k_indices]
        
        # Store compression error for next round
        self.error_feedback[client_id] = gradient - compressed
        
        return compressed


class DifferentialPrivacy:
    """Differential privacy mechanism for gradient perturbation."""
    
    def __init__(self, epsilon: float, delta: float, clip_norm: float):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        
        # Compute noise scale (Gaussian mechanism)
        self.noise_scale = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient to bounded L2 norm."""
        norm = np.linalg.norm(gradient)
        if norm > self.clip_norm:
            gradient = gradient * (self.clip_norm / norm)
        return gradient
    
    def add_noise(self, gradient: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise for differential privacy."""
        noise = np.random.normal(0, self.noise_scale, size=gradient.shape)
        return gradient + noise
    
    def privatize(self, gradient: np.ndarray) -> np.ndarray:
        """Apply clipping and noise addition."""
        clipped = self.clip_gradient(gradient)
        return self.add_noise(clipped)


class ByzFed:
    """Byzantine-resilient aggregation mechanism."""
    
    def __init__(self, threshold: float = 3.0, reputation_decay: float = 0.9):
        self.threshold = threshold
        self.decay = reputation_decay
        self.reputation_scores = {}
    
    def geometric_median(self, updates: List[np.ndarray], max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Compute geometric median using Weiszfeld's algorithm."""
        y = np.mean(updates, axis=0)
        
        for _ in range(max_iter):
            distances = [max(np.linalg.norm(u - y), 1e-10) for u in updates]
            weights = [1.0 / d for d in distances]
            weight_sum = sum(weights)
            
            y_new = sum(w * u for w, u in zip(weights, updates)) / weight_sum
            
            if np.linalg.norm(y_new - y) < tol:
                break
            y = y_new
        
        return y
    
    def aggregate(self, updates: Dict[int, np.ndarray]) -> np.ndarray:
        """Perform Byzantine-resilient aggregation with reputation weighting."""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        client_ids = list(updates.keys())
        update_list = [updates[cid] for cid in client_ids]
        
        # Initialize reputation scores for new clients
        for cid in client_ids:
            if cid not in self.reputation_scores:
                self.reputation_scores[cid] = 1.0
        
        # Compute geometric median
        median = self.geometric_median(update_list)
        
        # Compute distances from median
        distances = {cid: np.linalg.norm(updates[cid] - median) for cid in client_ids}
        
        # Robust scale estimate (MAD-based)
        distance_values = list(distances.values())
        median_distance = np.median(distance_values)
        scale = max(median_distance * 1.4826, 1e-10)  # MAD to std conversion
        
        # Filter outliers
        filtered_clients = [cid for cid in client_ids if distances[cid] <= self.threshold * scale]
        
        # Update reputation scores
        for cid in client_ids:
            in_filter = 1.0 if cid in filtered_clients else 0.0
            self.reputation_scores[cid] = self.decay * self.reputation_scores[cid] + (1 - self.decay) * in_filter
        
        # Compute weighted average over filtered clients
        if not filtered_clients:
            logger.warning("All clients filtered! Using all updates.")
            filtered_clients = client_ids
        
        weights = {cid: self.reputation_scores[cid] for cid in filtered_clients}
        total_weight = sum(weights.values())
        
        aggregated = sum(weights[cid] * updates[cid] for cid in filtered_clients) / total_weight
        
        return aggregated


class LocalClient:
    """Simulated local client for federated learning."""
    
    def __init__(
        self,
        client_id: int,
        data: np.ndarray,
        labels: np.ndarray,
        model_dim: int,
        is_byzantine: bool = False
    ):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        self.model_dim = model_dim
        self.is_byzantine = is_byzantine
        self.momentum_buffer = np.zeros(model_dim)
    
    def compute_gradient(self, model: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Compute stochastic gradient on local data."""
        # Sample mini-batch
        indices = np.random.choice(len(self.data), min(batch_size, len(self.data)), replace=False)
        X_batch = self.data[indices]
        y_batch = self.labels[indices]
        
        # Simple linear model gradient (for demonstration)
        # Gradient of squared loss: 2 * X^T (X @ w - y) / n
        predictions = X_batch @ model
        error = predictions - y_batch
        gradient = 2 * X_batch.T @ error / len(indices)
        
        return gradient
    
    def local_update(
        self,
        global_model: np.ndarray,
        local_epochs: int,
        learning_rate: float,
        momentum: float,
        batch_size: int = 32
    ) -> np.ndarray:
        """Perform local training and return model delta."""
        local_model = global_model.copy()
        
        for _ in range(local_epochs):
            gradient = self.compute_gradient(local_model, batch_size)
            
            # Momentum update
            self.momentum_buffer = momentum * self.momentum_buffer + gradient
            local_model -= learning_rate * self.momentum_buffer
        
        delta = local_model - global_model
        
        # Byzantine attack: negate gradient
        if self.is_byzantine:
            delta = -delta * 10  # Aggressive gradient negation
        
        return delta


class FedSov:
    """Sovereign Federated Learning Algorithm."""
    
    def __init__(self, config: FedSovConfig, model_dim: int):
        self.config = config
        self.model_dim = model_dim
        
        # Initialize components
        self.compressor = GradientCompressor(config.compression_ratio)
        self.dp = DifferentialPrivacy(config.dp_epsilon, config.dp_delta, config.gradient_clip)
        self.aggregator = ByzFed(config.byzantine_threshold, config.reputation_decay)
        
        # Initialize global model
        self.global_model = np.zeros(model_dim)
        self.round_history = []
    
    def select_clients(self, num_clients: int) -> List[int]:
        """Select participating clients for this round."""
        num_selected = max(1, int(num_clients * self.config.participation_rate))
        return np.random.choice(num_clients, num_selected, replace=False).tolist()
    
    def train_round(self, clients: List[LocalClient]) -> Dict[str, float]:
        """Execute one round of federated learning."""
        # Select participating clients
        selected_ids = self.select_clients(len(clients))
        selected_clients = [clients[i] for i in selected_ids]
        
        logger.info(f"Round: Selected {len(selected_clients)} clients")
        
        # Collect local updates
        updates = {}
        for client in selected_clients:
            # Local training
            delta = client.local_update(
                self.global_model,
                self.config.local_epochs,
                self.config.learning_rate,
                self.config.momentum
            )
            
            # Compress
            compressed = self.compressor.compress(delta, client.client_id)
            
            # Add DP noise
            privatized = self.dp.privatize(compressed)
            
            updates[client.client_id] = privatized
        
        # Byzantine-resilient aggregation
        aggregated_update = self.aggregator.aggregate(updates)
        
        # Update global model
        self.global_model = self.global_model + aggregated_update
        
        # Compute metrics
        metrics = {
            'model_norm': float(np.linalg.norm(self.global_model)),
            'update_norm': float(np.linalg.norm(aggregated_update)),
            'num_participants': len(selected_clients)
        }
        self.round_history.append(metrics)
        
        return metrics
    
    def train(self, clients: List[LocalClient], num_rounds: int) -> List[Dict[str, float]]:
        """Execute full training procedure."""
        logger.info(f"Starting FedSov training for {num_rounds} rounds")
        
        for round_idx in range(num_rounds):
            metrics = self.train_round(clients)
            
            if (round_idx + 1) % 10 == 0:
                logger.info(f"Round {round_idx + 1}/{num_rounds}: {metrics}")
        
        return self.round_history


def create_synthetic_data(
    num_clients: int,
    samples_per_client: int,
    dim: int,
    heterogeneity: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic non-IID data for clients."""
    datasets = []
    
    # True model (for data generation)
    true_model = np.random.randn(dim)
    
    for i in range(num_clients):
        # Non-IID: each client has shifted distribution
        shift = heterogeneity * np.random.randn(dim)
        
        X = np.random.randn(samples_per_client, dim) + shift
        noise = 0.1 * np.random.randn(samples_per_client)
        y = X @ true_model + noise
        
        datasets.append((X, y))
    
    return datasets


def visualize_results(history: List[Dict[str, float]], output_dir: str = "../figures"):
    """Generate and save visualization plots from training history."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping visualization")
        return
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract metrics
    rounds = list(range(1, len(history) + 1))
    model_norms = [h['model_norm'] for h in history]
    update_norms = [h['update_norm'] for h in history]
    
    # Plot 1: Convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(rounds, model_norms, 'b-', linewidth=2)
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Model Norm', fontsize=12)
    ax1.set_title('Model Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(rounds, update_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Update Norm', fontsize=12)
    ax2.set_title('Update Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'convergence_curves.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    logger.info(f"Saved convergence plot to {output_path}")
    plt.close()


def run_byzantine_experiment(
    num_clients: int = 100,
    num_rounds: int = 200,
    model_dim: int = 100,
    samples_per_client: int = 500,
    seed: int = 42,
    output_dir: str = "../figures"
) -> Dict[str, List[Dict[str, float]]]:
    """Run experiments comparing different Byzantine fractions."""
    np.random.seed(seed)
    
    byzantine_fracs = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    for byz_frac in byzantine_fracs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment with {byz_frac*100:.0f}% Byzantine clients")
        logger.info(f"{'='*60}")
        
        # Generate data
        datasets = create_synthetic_data(num_clients, samples_per_client, model_dim)
        
        # Determine Byzantine clients
        num_byzantine = int(num_clients * byz_frac)
        byzantine_ids = set(np.random.choice(num_clients, num_byzantine, replace=False))
        
        # Create clients
        clients = []
        for i, (X, y) in enumerate(datasets):
            client = LocalClient(
                client_id=i,
                data=X,
                labels=y,
                model_dim=model_dim,
                is_byzantine=(i in byzantine_ids)
            )
            clients.append(client)
        
        # Configure and run FedSov
        config = FedSovConfig(
            num_clients=num_clients,
            participation_rate=0.1,
            local_epochs=5,
            learning_rate=0.01,
            compression_ratio=0.1
        )
        
        fedsov = FedSov(config, model_dim)
        history = fedsov.train(clients, num_rounds)
        
        results[f"Byzantine {byz_frac*100:.0f}%"] = history
    
    # Generate comparison plot
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        
        for label, history in results.items():
            rounds = list(range(1, len(history) + 1))
            model_norms = [h['model_norm'] for h in history]
            plt.plot(rounds, model_norms, linewidth=2, label=label)
        
        plt.xlabel('Training Round', fontsize=12)
        plt.ylabel('Model Norm', fontsize=12)
        plt.title('Byzantine Resilience: Impact of Malicious Clients', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, 'byzantine_resilience.pdf')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved Byzantine resilience plot to {output_path}")
        plt.close()
    
    return results


def run_scalability_experiment(
    client_counts: List[int] = [50, 100, 200, 500, 1000],
    num_rounds: int = 100,
    model_dim: int = 100,
    samples_per_client: int = 500,
    seed: int = 42,
    output_dir: str = "../figures"
) -> Dict[int, float]:
    """Run scalability experiments with varying client counts."""
    import time
    
    np.random.seed(seed)
    results = {}
    
    for num_clients in client_counts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scalability test with {num_clients} clients")
        logger.info(f"{'='*60}")
        
        # Generate data
        datasets = create_synthetic_data(num_clients, samples_per_client, model_dim)
        
        # Create clients
        clients = [
            LocalClient(i, X, y, model_dim)
            for i, (X, y) in enumerate(datasets)
        ]
        
        # Configure FedSov
        config = FedSovConfig(
            num_clients=num_clients,
            participation_rate=0.1,
            local_epochs=5,
            learning_rate=0.01,
            compression_ratio=0.1
        )
        
        fedsov = FedSov(config, model_dim)
        
        # Measure training time
        start_time = time.time()
        fedsov.train(clients, num_rounds)
        elapsed_time = time.time() - start_time
        
        results[num_clients] = elapsed_time / 3600  # Convert to hours
        logger.info(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    
    # Generate scalability plot
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        
        client_nums = list(results.keys())
        times = list(results.values())
        
        plt.plot(client_nums, times, 'bo-', linewidth=2, markersize=8)
        plt.xscale('log')
        plt.xlabel('Number of Clients', fontsize=12)
        plt.ylabel('Training Time (hours)', fontsize=12)
        plt.title('Scalability: Training Time vs Number of Clients', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        
        # Add value labels
        for x, y in zip(client_nums, times):
            plt.annotate(f'{y:.2f}h', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9)
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = os.path.join(output_dir, 'scalability.pdf')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved scalability plot to {output_path}")
        plt.close()
    
    return results


def main():
    """Main execution function demonstrating DSAIN framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DSAIN: Sovereign Federated Learning')
    parser.add_argument('--num_clients', type=int, default=100, help='Total number of clients')
    parser.add_argument('--num_rounds', type=int, default=200, help='Number of training rounds')
    parser.add_argument('--byzantine_frac', type=float, default=0.1, help='Fraction of Byzantine clients')
    parser.add_argument('--model_dim', type=int, default=100, help='Model dimension')
    parser.add_argument('--samples_per_client', type=int, default=500, help='Samples per client')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'byzantine', 'scalability', 'all'],
                       help='Experiment mode: single run, Byzantine resilience, scalability, or all')
    parser.add_argument('--output_dir', type=str, default='../figures', help='Output directory for figures')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    if args.mode == 'byzantine' or args.mode == 'all':
        logger.info("\n" + "="*80)
        logger.info("RUNNING BYZANTINE RESILIENCE EXPERIMENTS")
        logger.info("="*80)
        run_byzantine_experiment(
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            model_dim=args.model_dim,
            samples_per_client=args.samples_per_client,
            seed=args.seed,
            output_dir=args.output_dir
        )
    
    if args.mode == 'scalability' or args.mode == 'all':
        logger.info("\n" + "="*80)
        logger.info("RUNNING SCALABILITY EXPERIMENTS")
        logger.info("="*80)
        run_scalability_experiment(
            client_counts=[50, 100, 200],  # Reduced for faster demo
            num_rounds=50,  # Reduced for faster demo
            model_dim=args.model_dim,
            samples_per_client=args.samples_per_client,
            seed=args.seed,
            output_dir=args.output_dir
        )
    
    if args.mode == 'single' or args.mode == 'all':
        logger.info("\n" + "="*80)
        logger.info("RUNNING SINGLE EXPERIMENT WITH SPECIFIED PARAMETERS")
        logger.info("="*80)
        
        # Generate synthetic data
        logger.info("Generating synthetic data...")
        datasets = create_synthetic_data(
            args.num_clients,
            args.samples_per_client,
            args.model_dim,
            heterogeneity=0.5
        )
        
        # Determine Byzantine clients
        num_byzantine = int(args.num_clients * args.byzantine_frac)
        byzantine_ids = set(np.random.choice(args.num_clients, num_byzantine, replace=False))
        logger.info(f"Byzantine clients: {sorted(byzantine_ids)}")
        
        # Create client objects
        clients = []
        for i, (X, y) in enumerate(datasets):
            client = LocalClient(
                client_id=i,
                data=X,
                labels=y,
                model_dim=args.model_dim,
                is_byzantine=(i in byzantine_ids)
            )
            clients.append(client)
        
        # Configure FedSov
        config = FedSovConfig(
            num_clients=args.num_clients,
            participation_rate=0.1,
            local_epochs=5,
            learning_rate=0.01,
            compression_ratio=0.1,
            dp_epsilon=4.0,
            dp_delta=1e-5
        )
        
        # Initialize and run FedSov
        fedsov = FedSov(config, args.model_dim)
        history = fedsov.train(clients, args.num_rounds)
        
        # Generate visualizations
        visualize_results(history, args.output_dir)
        
        # Report results
        final_norm = history[-1]['model_norm']
        avg_update = np.mean([h['update_norm'] for h in history])
        
        logger.info(f"\n{'='*80}")
        logger.info("DSAIN TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Final model norm: {final_norm:.4f}")
        logger.info(f"Average update norm: {avg_update:.4f}")
        logger.info(f"Total rounds: {len(history)}")
        logger.info(f"Byzantine clients: {num_byzantine}/{args.num_clients}")
        logger.info(f"Figures saved to: {args.output_dir}")
        logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
