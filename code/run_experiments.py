#!/usr/bin/env python3
"""
DSAIN Batch Experiment Runner
==============================

Run multiple experiments with different seeds and save results to the results folder.
This script enables reproducible experiments for the JMLR paper.

Usage:
    python run_experiments.py --num_runs 5 --mode all
    python run_experiments.py --num_runs 3 --mode byzantine
    python run_experiments.py --num_runs 10 --seeds 42 123 456 789 1000

Author: Almas Ospanov
License: MIT
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the code directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import DSAIN components
from dsain import (
    FedSovConfig, FedSov, LocalClient, create_synthetic_data,
    run_byzantine_experiment, run_scalability_experiment,
    visualize_results, MATPLOTLIB_AVAILABLE
)


def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """Create a timestamped experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Resolve to absolute path to avoid issues with relative paths
    base_dir = Path(base_dir).resolve()
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(exp_dir: Path, config: dict):
    """Save experiment configuration to JSON."""
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def save_results(exp_dir: Path, results: dict, run_id: int):
    """Save experiment results to JSON."""
    results_path = exp_dir / f"results_run_{run_id:03d}.json"
    
    # Convert all numpy types recursively
    serializable_results = convert_to_serializable(results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return results_path


def aggregate_results(exp_dir: Path, num_runs: int) -> dict:
    """Aggregate results from multiple runs and compute statistics."""
    all_results = []
    
    for run_id in range(num_runs):
        results_path = exp_dir / f"results_run_{run_id:03d}.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                all_results.append(json.load(f))
    
    if not all_results:
        return {}
    
    # Compute mean and std for key metrics
    aggregated = {
        'num_runs': len(all_results),
        'metrics': {}
    }
    
    # Common metrics to aggregate
    metric_keys = ['final_model_norm', 'avg_update_norm', 'final_accuracy', 
                   'training_time', 'communication_rounds']
    
    for key in metric_keys:
        values = [r.get(key) for r in all_results if key in r]
        if values:
            aggregated['metrics'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
    
    # Save aggregated results
    agg_path = exp_dir / "aggregated_results.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    logger.info(f"Aggregated results saved to {agg_path}")
    
    return aggregated


def run_single_experiment(
    seed: int,
    num_clients: int = 100,
    num_rounds: int = 200,
    byzantine_frac: float = 0.1,
    model_dim: int = 100,
    samples_per_client: int = 500,
    output_dir: Path = None
) -> dict:
    """Run a single DSAIN experiment and return results."""
    import time
    start_time = time.time()
    
    # Set random seed
    np.random.seed(seed)
    
    logger.info(f"Running experiment with seed={seed}")
    
    # Generate synthetic data
    datasets = create_synthetic_data(
        num_clients,
        samples_per_client,
        model_dim,
        heterogeneity=0.5
    )
    
    # Determine Byzantine clients
    num_byzantine = int(num_clients * byzantine_frac)
    byzantine_ids = set(np.random.choice(num_clients, num_byzantine, replace=False))
    
    # Create client objects
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
    
    # Configure FedSov
    config = FedSovConfig(
        num_clients=num_clients,
        participation_rate=0.1,
        local_epochs=5,
        learning_rate=0.01,
        compression_ratio=0.1,
        dp_epsilon=4.0,
        dp_delta=1e-5
    )
    
    # Initialize and run FedSov
    fedsov = FedSov(config, model_dim)
    history = fedsov.train(clients, num_rounds)
    
    # Calculate metrics
    training_time = time.time() - start_time
    final_norm = history[-1]['model_norm']
    avg_update = np.mean([h['update_norm'] for h in history])
    
    # Save figures if output directory provided
    if output_dir and MATPLOTLIB_AVAILABLE:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            visualize_results(history, str(output_path))
        except Exception as e:
            logger.warning(f"Failed to save figures: {e}")
    
    results = {
        'seed': seed,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'byzantine_frac': byzantine_frac,
        'num_byzantine': num_byzantine,
        'model_dim': model_dim,
        'samples_per_client': samples_per_client,
        'final_model_norm': float(final_norm),
        'avg_update_norm': float(avg_update),
        'training_time': training_time,
        'communication_rounds': num_rounds,
        'history_length': len(history),
        'byzantine_ids': list(byzantine_ids)
    }
    
    return results


def run_batch_experiments(args):
    """Run batch experiments with multiple seeds."""
    # Setup directories
    results_base = Path(args.results_dir)
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Create experiment directory
    exp_dir = create_experiment_dir(results_base, f"dsain_{args.mode}")
    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        # Generate deterministic seeds based on base seed
        np.random.seed(args.base_seed)
        seeds = np.random.randint(0, 100000, size=args.num_runs).tolist()
    
    # Save experiment configuration
    config = {
        'mode': args.mode,
        'num_runs': len(seeds),
        'seeds': seeds,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'byzantine_frac': args.byzantine_frac,
        'model_dim': args.model_dim,
        'samples_per_client': args.samples_per_client,
        'timestamp': datetime.now().isoformat()
    }
    save_experiment_config(exp_dir, config)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DSAIN BATCH EXPERIMENT RUNNER")
    logger.info(f"{'='*80}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Number of runs: {len(seeds)}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"{'='*80}\n")
    
    # Run experiments
    all_results = []
    
    for run_id, seed in enumerate(seeds):
        logger.info(f"\n--- Run {run_id + 1}/{len(seeds)} (seed={seed}) ---")
        
        try:
            if args.mode == 'single':
                results = run_single_experiment(
                    seed=seed,
                    num_clients=args.num_clients,
                    num_rounds=args.num_rounds,
                    byzantine_frac=args.byzantine_frac,
                    model_dim=args.model_dim,
                    samples_per_client=args.samples_per_client,
                    output_dir=figures_dir if run_id == 0 else None  # Only save figures for first run
                )
            elif args.mode == 'byzantine':
                np.random.seed(seed)
                run_byzantine_experiment(
                    num_clients=args.num_clients,
                    num_rounds=min(args.num_rounds, 100),  # Faster for batch
                    model_dim=args.model_dim,
                    samples_per_client=args.samples_per_client,
                    seed=seed,
                    output_dir=str(figures_dir) if run_id == 0 else str(exp_dir / "temp")
                )
                results = {
                    'seed': seed,
                    'mode': 'byzantine',
                    'status': 'completed'
                }
            elif args.mode == 'scalability':
                np.random.seed(seed)
                run_scalability_experiment(
                    client_counts=[50, 100, 200],
                    num_rounds=min(args.num_rounds, 50),  # Faster for batch
                    model_dim=args.model_dim,
                    samples_per_client=args.samples_per_client,
                    seed=seed,
                    output_dir=str(figures_dir) if run_id == 0 else str(exp_dir / "temp")
                )
                results = {
                    'seed': seed,
                    'mode': 'scalability',
                    'status': 'completed'
                }
            elif args.mode == 'all':
                # Run all experiment types
                results = run_single_experiment(
                    seed=seed,
                    num_clients=args.num_clients,
                    num_rounds=args.num_rounds,
                    byzantine_frac=args.byzantine_frac,
                    model_dim=args.model_dim,
                    samples_per_client=args.samples_per_client,
                    output_dir=figures_dir if run_id == 0 else None
                )
                
                # Also run byzantine and scalability for first run
                if run_id == 0:
                    np.random.seed(seed)
                    run_byzantine_experiment(
                        num_clients=args.num_clients,
                        num_rounds=min(args.num_rounds, 100),
                        model_dim=args.model_dim,
                        samples_per_client=args.samples_per_client,
                        seed=seed,
                        output_dir=str(figures_dir)
                    )
                    run_scalability_experiment(
                        client_counts=[50, 100, 200],
                        num_rounds=50,
                        model_dim=args.model_dim,
                        samples_per_client=args.samples_per_client,
                        seed=seed,
                        output_dir=str(figures_dir)
                    )
            else:
                raise ValueError(f"Unknown mode: {args.mode}")
            
            # Save individual run results
            save_results(exp_dir, results, run_id)
            all_results.append(results)
            
            logger.info(f"Run {run_id + 1} completed successfully")
            
        except Exception as e:
            logger.error(f"Run {run_id + 1} failed: {e}")
            results = {
                'seed': seed,
                'status': 'failed',
                'error': str(e)
            }
            save_results(exp_dir, results, run_id)
    
    # Aggregate results
    logger.info("\n--- Aggregating Results ---")
    aggregated = aggregate_results(exp_dir, len(seeds))
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Completed runs: {aggregated.get('num_runs', 0)}/{len(seeds)}")
    
    if 'metrics' in aggregated:
        for metric, stats in aggregated['metrics'].items():
            logger.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"{'='*80}\n")
    
    # Clean up temp directory if exists
    temp_dir = exp_dir / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return exp_dir


def main():
    parser = argparse.ArgumentParser(
        description='DSAIN Batch Experiment Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Run configuration
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of experiment runs')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Specific seeds to use (overrides num_runs)')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for generating run seeds')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'byzantine', 'scalability', 'all'],
                       help='Experiment mode')
    
    # Experiment parameters
    parser.add_argument('--num_clients', type=int, default=100,
                       help='Total number of clients')
    parser.add_argument('--num_rounds', type=int, default=200,
                       help='Number of training rounds')
    parser.add_argument('--byzantine_frac', type=float, default=0.1,
                       help='Fraction of Byzantine clients')
    parser.add_argument('--model_dim', type=int, default=100,
                       help='Model dimension')
    parser.add_argument('--samples_per_client', type=int, default=500,
                       help='Samples per client')
    
    # Output configuration
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Base directory for results')
    
    args = parser.parse_args()
    
    run_batch_experiments(args)


if __name__ == "__main__":
    main()
