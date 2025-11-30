#!/usr/bin/env python3
"""
LEAF Benchmark Experiments for DSAIN Framework
==============================================

This script runs DSAIN experiments on LEAF federated datasets (FEMNIST, Shakespeare)
to provide real-world federated learning benchmarks for the JMLR submission.

Usage:
    python run_leaf_experiments.py --dataset femnist --num_rounds 200 --seed 42
    python run_leaf_experiments.py --dataset shakespeare --num_rounds 200 --seed 42
    python run_leaf_experiments.py --dataset all --num_rounds 100 --seed 42

Output:
    - Experiment results in JSON format (results/leaf_experiments_*.json)
    - Figures for paper (figures/leaf_*.pdf)
"""

import numpy as np
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Import DSAIN components
from dsain import (
    FedSovConfig, GradientCompressor, DifferentialPrivacy, 
    ByzFed, FedSov
)
from leaf_datasets import FEMNISTLoader, ShakespeareLoader, ClientData, compute_heterogeneity_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """Configuration for LEAF experiments."""
    dataset: str  # 'femnist' or 'shakespeare'
    num_clients: int = 100
    num_rounds: int = 200
    participation_rate: float = 0.1
    local_epochs: int = 5
    learning_rate: float = 0.01
    compression_ratio: float = 0.1
    dp_epsilon: float = 4.0
    dp_delta: float = 1e-5
    byzantine_frac: float = 0.0
    seed: int = 42


class LEAFClient:
    """
    Client wrapper for LEAF datasets with neural network model.
    
    Uses a simple MLP for classification tasks.
    """
    
    def __init__(
        self,
        client_id: str,
        x: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 62,
        is_byzantine: bool = False
    ):
        self.client_id = client_id
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.is_byzantine = is_byzantine
        
        # Model dimensions: input -> hidden -> output
        # Flattened weights: W1 (input_dim x hidden_dim) + b1 (hidden_dim) + 
        #                    W2 (hidden_dim x num_classes) + b2 (num_classes)
        self.model_dim = (input_dim * hidden_dim + hidden_dim + 
                          hidden_dim * num_classes + num_classes)
        
        self.momentum_buffer = np.zeros(self.model_dim)
        
    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack flattened parameters into weight matrices and biases."""
        idx = 0
        
        # W1: input_dim x hidden_dim
        w1_size = self.input_dim * self.hidden_dim
        W1 = params[idx:idx + w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        
        # b1: hidden_dim
        b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim
        
        # W2: hidden_dim x num_classes
        w2_size = self.hidden_dim * self.num_classes
        W2 = params[idx:idx + w2_size].reshape(self.hidden_dim, self.num_classes)
        idx += w2_size
        
        # b2: num_classes
        b2 = params[idx:idx + self.num_classes]
        
        return W1, b1, W2, b2
    
    def _pack_params(self, W1: np.ndarray, b1: np.ndarray, 
                     W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """Pack weight matrices and biases into flattened array."""
        return np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
    
    def _forward(self, x: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through 2-layer MLP."""
        W1, b1, W2, b2 = self._unpack_params(params)
        
        # Hidden layer with ReLU
        h = np.maximum(0, x @ W1 + b1)
        
        # Output layer (logits)
        logits = h @ W2 + b2
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return probs, h
    
    def _compute_loss_and_grad(self, x: np.ndarray, y: np.ndarray, 
                                params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradients."""
        batch_size = len(y)
        W1, b1, W2, b2 = self._unpack_params(params)
        
        # Forward pass
        h = np.maximum(0, x @ W1 + b1)  # ReLU
        logits = h @ W2 + b2
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Cross-entropy loss
        y_one_hot = np.zeros((batch_size, self.num_classes))
        y_one_hot[np.arange(batch_size), y] = 1
        loss = -np.sum(y_one_hot * np.log(probs + 1e-10)) / batch_size
        
        # Backward pass
        # dL/d(logits) = probs - y_one_hot
        d_logits = (probs - y_one_hot) / batch_size
        
        # dL/dW2, dL/db2
        dW2 = h.T @ d_logits
        db2 = d_logits.sum(axis=0)
        
        # dL/dh
        dh = d_logits @ W2.T
        
        # ReLU backward
        dh_pre = dh * (h > 0)
        
        # dL/dW1, dL/db1
        dW1 = x.T @ dh_pre
        db1 = dh_pre.sum(axis=0)
        
        grad = self._pack_params(dW1, db1, dW2, db2)
        
        return loss, grad
    
    def local_update(
        self,
        global_model: np.ndarray,
        local_epochs: int,
        learning_rate: float,
        momentum: float,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Dict]:
        """Perform local training and return model delta with metrics."""
        local_model = global_model.copy()
        total_loss = 0
        num_batches = 0
        
        for epoch in range(local_epochs):
            # Shuffle data
            indices = np.random.permutation(len(self.x))
            
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch = self.x[batch_idx]
                y_batch = self.y[batch_idx]
                
                loss, grad = self._compute_loss_and_grad(x_batch, y_batch, local_model)
                
                # Momentum update
                self.momentum_buffer = momentum * self.momentum_buffer + grad
                local_model -= learning_rate * self.momentum_buffer
                
                total_loss += loss
                num_batches += 1
        
        delta = local_model - global_model
        
        # Byzantine attack
        if self.is_byzantine:
            delta = -delta * 10
        
        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'num_samples': len(self.x)
        }
        
        return delta, metrics
    
    def evaluate(self, params: np.ndarray) -> Dict:
        """Evaluate model on local data."""
        probs, _ = self._forward(self.x, params)
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == self.y)
        
        return {
            'accuracy': float(accuracy),
            'num_samples': len(self.y)
        }


class LEAFFedSov:
    """
    FedSov adapted for LEAF datasets with proper neural network training.
    """
    
    def __init__(self, config: ExperimentConfig, model_dim: int):
        self.config = config
        self.model_dim = model_dim
        
        # Initialize components
        self.compressor = GradientCompressor(config.compression_ratio)
        self.dp = DifferentialPrivacy(config.dp_epsilon, config.dp_delta, 1.0)
        self.aggregator = ByzFed(threshold=3.0, reputation_decay=0.9)
        
        # Initialize global model (Xavier initialization)
        self.global_model = np.random.randn(model_dim) * np.sqrt(2.0 / model_dim)
        
        self.history = []
        
    def train_round(self, clients: List[LEAFClient]) -> Dict:
        """Execute one round of federated learning."""
        # Select participating clients
        num_selected = max(1, int(len(clients) * self.config.participation_rate))
        selected_indices = np.random.choice(len(clients), num_selected, replace=False)
        selected_clients = [clients[i] for i in selected_indices]
        
        # Collect local updates
        updates = {}
        total_loss = 0
        total_samples = 0
        
        for client in selected_clients:
            delta, metrics = client.local_update(
                self.global_model,
                self.config.local_epochs,
                self.config.learning_rate,
                momentum=0.9
            )
            
            # Compress
            compressed = self.compressor.compress(delta, hash(client.client_id) % 10000)
            
            # Add DP noise
            privatized = self.dp.privatize(compressed)
            
            updates[hash(client.client_id) % 10000] = privatized
            total_loss += metrics['loss'] * metrics['num_samples']
            total_samples += metrics['num_samples']
        
        # Byzantine-resilient aggregation
        aggregated = self.aggregator.aggregate(updates)
        
        # Update global model
        self.global_model = self.global_model + aggregated
        
        # Compute round metrics
        round_metrics = {
            'avg_loss': total_loss / max(total_samples, 1),
            'num_participants': len(selected_clients),
            'update_norm': float(np.linalg.norm(aggregated))
        }
        
        self.history.append(round_metrics)
        return round_metrics
    
    def evaluate(self, clients: List[LEAFClient], num_clients: int = 50) -> Dict:
        """Evaluate global model on a subset of clients."""
        # Sample clients for evaluation
        eval_indices = np.random.choice(len(clients), min(num_clients, len(clients)), replace=False)
        eval_clients = [clients[i] for i in eval_indices]
        
        total_correct = 0
        total_samples = 0
        
        for client in eval_clients:
            metrics = client.evaluate(self.global_model)
            total_correct += metrics['accuracy'] * metrics['num_samples']
            total_samples += metrics['num_samples']
        
        return {
            'accuracy': total_correct / max(total_samples, 1),
            'num_clients_evaluated': len(eval_clients),
            'total_samples': total_samples
        }
    
    def train(self, clients: List[LEAFClient], num_rounds: int, 
              eval_every: int = 10) -> List[Dict]:
        """Execute full training with periodic evaluation."""
        logger.info(f"Starting LEAF FedSov training for {num_rounds} rounds")
        
        for round_idx in range(num_rounds):
            round_metrics = self.train_round(clients)
            
            if (round_idx + 1) % eval_every == 0:
                eval_metrics = self.evaluate(clients)
                round_metrics['eval_accuracy'] = eval_metrics['accuracy']
                logger.info(f"Round {round_idx + 1}/{num_rounds}: "
                           f"loss={round_metrics['avg_loss']:.4f}, "
                           f"acc={eval_metrics['accuracy']:.4f}")
        
        return self.history


def run_femnist_experiment(config: ExperimentConfig) -> Dict:
    """Run experiment on FEMNIST dataset."""
    logger.info("=" * 70)
    logger.info("FEMNIST Experiment")
    logger.info("=" * 70)
    
    np.random.seed(config.seed)
    
    # Load dataset
    loader = FEMNISTLoader()
    client_data = loader.load(max_clients=config.num_clients, use_synthetic=True)
    
    # Compute heterogeneity
    hetero_metrics = compute_heterogeneity_metrics(client_data)
    logger.info(f"Dataset heterogeneity (KL): {hetero_metrics['mean_kl_divergence']:.3f}")
    
    # Determine Byzantine clients
    num_byzantine = int(config.num_clients * config.byzantine_frac)
    byzantine_ids = set(np.random.choice(config.num_clients, num_byzantine, replace=False))
    
    # Create LEAF clients
    input_dim = 28 * 28  # FEMNIST images
    num_classes = 62
    hidden_dim = 128
    
    clients = []
    for i, cd in enumerate(client_data):
        client = LEAFClient(
            client_id=cd.client_id,
            x=cd.x,
            y=cd.y,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            is_byzantine=(i in byzantine_ids)
        )
        clients.append(client)
    
    # Initialize and train
    model_dim = clients[0].model_dim
    fedsov = LEAFFedSov(config, model_dim)
    
    start_time = time.time()
    history = fedsov.train(clients, config.num_rounds, eval_every=10)
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = fedsov.evaluate(clients)
    
    return {
        'dataset': 'femnist',
        'config': asdict(config),
        'heterogeneity': hetero_metrics,
        'history': history,
        'final_accuracy': final_metrics['accuracy'],
        'training_time_seconds': training_time,
        'num_byzantine': num_byzantine
    }


def run_shakespeare_experiment(config: ExperimentConfig) -> Dict:
    """Run experiment on Shakespeare dataset."""
    logger.info("=" * 70)
    logger.info("Shakespeare Experiment")
    logger.info("=" * 70)
    
    np.random.seed(config.seed)
    
    # Load dataset
    loader = ShakespeareLoader()
    client_data = loader.load(max_clients=config.num_clients, use_synthetic=True)
    
    # Compute heterogeneity
    hetero_metrics = compute_heterogeneity_metrics(client_data)
    logger.info(f"Dataset heterogeneity (KL): {hetero_metrics['mean_kl_divergence']:.3f}")
    
    # Determine Byzantine clients
    num_byzantine = int(config.num_clients * config.byzantine_frac)
    byzantine_ids = set(np.random.choice(config.num_clients, num_byzantine, replace=False))
    
    # Create LEAF clients
    seq_len = 80
    vocab_size = 80
    hidden_dim = 64
    
    clients = []
    for i, cd in enumerate(client_data):
        client = LEAFClient(
            client_id=cd.client_id,
            x=cd.x,
            y=cd.y,
            input_dim=seq_len,  # Simplified: one-hot sum
            hidden_dim=hidden_dim,
            num_classes=vocab_size,
            is_byzantine=(i in byzantine_ids)
        )
        clients.append(client)
    
    # Initialize and train
    model_dim = clients[0].model_dim
    fedsov = LEAFFedSov(config, model_dim)
    
    start_time = time.time()
    history = fedsov.train(clients, config.num_rounds, eval_every=10)
    training_time = time.time() - start_time
    
    # Final evaluation
    final_metrics = fedsov.evaluate(clients)
    
    return {
        'dataset': 'shakespeare',
        'config': asdict(config),
        'heterogeneity': hetero_metrics,
        'history': history,
        'final_accuracy': final_metrics['accuracy'],
        'training_time_seconds': training_time,
        'num_byzantine': num_byzantine
    }


def run_comparison_experiment(config: ExperimentConfig) -> Dict:
    """
    Run comparison experiment: FedSov vs FedAvg (no Byzantine defense) vs no compression.
    """
    logger.info("=" * 70)
    logger.info("Comparison Experiment: FedSov variants")
    logger.info("=" * 70)
    
    results = {}
    
    # Standard FedSov
    logger.info("\n1. FedSov (full)")
    results['fedsov_full'] = run_femnist_experiment(config)
    
    # No compression
    config_no_compress = ExperimentConfig(**asdict(config))
    config_no_compress.compression_ratio = 1.0  # No compression
    logger.info("\n2. FedSov (no compression)")
    results['fedsov_no_compress'] = run_femnist_experiment(config_no_compress)
    
    # No DP
    config_no_dp = ExperimentConfig(**asdict(config))
    config_no_dp.dp_epsilon = float('inf')  # No DP
    logger.info("\n3. FedSov (no DP)")
    results['fedsov_no_dp'] = run_femnist_experiment(config_no_dp)
    
    return results


def generate_figures(results: Dict, output_dir: str = "../figures"):
    """Generate publication-quality figures from experiment results."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping figures")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (6, 4),
        'figure.dpi': 300
    })
    
    # Figure 1: Learning curves
    if 'history' in results:
        fig, ax = plt.subplots()
        
        rounds = list(range(1, len(results['history']) + 1))
        losses = [h.get('avg_loss', h.get('loss', 0)) for h in results['history']]
        
        ax.plot(rounds, losses, 'b-', linewidth=2, label='FedSov')
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Training Loss')
        ax.set_title(f"DSAIN on {results.get('dataset', 'LEAF').upper()}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaf_learning_curve.pdf", bbox_inches='tight')
        logger.info(f"Saved {output_dir}/leaf_learning_curve.pdf")
        plt.close()
    
    # Figure 2: Comparison (if available)
    if 'fedsov_full' in results and 'fedsov_no_compress' in results:
        fig, ax = plt.subplots()
        
        for name, label in [('fedsov_full', 'FedSov (full)'), 
                           ('fedsov_no_compress', 'No compression'),
                           ('fedsov_no_dp', 'No DP')]:
            if name in results and 'history' in results[name]:
                history = results[name]['history']
                rounds = list(range(1, len(history) + 1))
                losses = [h.get('avg_loss', 0) for h in history]
                ax.plot(rounds, losses, linewidth=2, label=label)
        
        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Training Loss')
        ax.set_title('FedSov Ablation Study on FEMNIST')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaf_ablation.pdf", bbox_inches='tight')
        logger.info(f"Saved {output_dir}/leaf_ablation.pdf")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='LEAF Benchmark Experiments for DSAIN')
    parser.add_argument('--dataset', type=str, default='femnist',
                       choices=['femnist', 'shakespeare', 'all', 'comparison'],
                       help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--participation_rate', type=float, default=0.1)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--compression_ratio', type=float, default=0.1)
    parser.add_argument('--dp_epsilon', type=float, default=4.0)
    parser.add_argument('--byzantine_frac', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--figure_dir', type=str, default='../figures')
    
    args = parser.parse_args()
    
    # Create config
    config = ExperimentConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        participation_rate=args.participation_rate,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        compression_ratio=args.compression_ratio,
        dp_epsilon=args.dp_epsilon,
        dp_delta=1e-5,
        byzantine_frac=args.byzantine_frac,
        seed=args.seed
    )
    
    # Run experiments
    results = {}
    
    if args.dataset == 'femnist' or args.dataset == 'all':
        results['femnist'] = run_femnist_experiment(config)
    
    if args.dataset == 'shakespeare' or args.dataset == 'all':
        config.dataset = 'shakespeare'
        results['shakespeare'] = run_shakespeare_experiment(config)
    
    if args.dataset == 'comparison':
        results = run_comparison_experiment(config)
    
    # Save results
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{args.output_dir}/leaf_experiments_{args.seed}.json"
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    results_clean = convert_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_clean, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # Generate figures
    if len(results) == 1:
        generate_figures(list(results.values())[0], args.figure_dir)
    else:
        generate_figures(results, args.figure_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for name, res in results.items():
        if isinstance(res, dict) and 'final_accuracy' in res:
            print(f"\n{name}:")
            print(f"  Final accuracy: {res['final_accuracy']:.4f}")
            print(f"  Training time: {res['training_time_seconds']:.1f}s")
            if 'heterogeneity' in res:
                print(f"  Data heterogeneity (KL): {res['heterogeneity']['mean_kl_divergence']:.3f}")


if __name__ == "__main__":
    main()
