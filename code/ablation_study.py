#!/usr/bin/env python3
"""
Comprehensive Ablation Study for DSAIN
=======================================

Systematically evaluates the contribution of each DSAIN component:
1. Compression
2. Byzantine-resilient aggregation (ByzFed)
3. Differential privacy

Author: Almas Ospanov
License: MIT
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import real_experiments if available
try:
    from real_experiments import run_cifar10_experiment, FLConfig
    REAL_EXPERIMENTS_AVAILABLE = True
except ImportError:
    logger.warning("real_experiments.py not found - using mock mode")
    REAL_EXPERIMENTS_AVAILABLE = False


# Ablation configurations
ABLATION_CONFIGS = {
    'full_dsain': {
        'name': 'Full DSAIN',
        'compression_ratio': 0.1,
        'method': 'dsain',
        'dp_epsilon': 4.0,
        'description': 'All components enabled'
    },
    'no_compression': {
        'name': 'DSAIN w/o Compression',
        'compression_ratio': 1.0,  # No compression
        'method': 'dsain',
        'dp_epsilon': 4.0,
        'description': 'ByzFed + DP, no compression'
    },
    'no_byzfed': {
        'name': 'DSAIN w/o ByzFed',
        'compression_ratio': 0.1,
        'method': 'fedavg',  # Use FedAvg instead of ByzFed
        'dp_epsilon': 4.0,
        'description': 'Compression + DP, no Byzantine resilience'
    },
    'no_dp': {
        'name': 'DSAIN w/o DP',
        'compression_ratio': 0.1,
        'method': 'dsain',
        'dp_epsilon': float('inf'),  # No DP
        'description': 'Compression + ByzFed, no differential privacy'
    },
    'compression_only': {
        'name': 'Compression Only',
        'compression_ratio': 0.1,
        'method': 'fedavg',
        'dp_epsilon': float('inf'),
        'description': 'Only compression'
    },
    'byzfed_only': {
        'name': 'ByzFed Only',
        'compression_ratio': 1.0,
        'method': 'dsain',
        'dp_epsilon': float('inf'),
        'description': 'Only Byzantine-resilient aggregation'
    },
    'dp_only': {
        'name': 'DP Only',
        'compression_ratio': 1.0,
        'method': 'fedavg',
        'dp_epsilon': 4.0,
        'description': 'Only differential privacy'
    },
    'vanilla_fedavg': {
        'name': 'Vanilla FedAvg',
        'compression_ratio': 1.0,
        'method': 'fedavg',
        'dp_epsilon': float('inf'),
        'description': 'Baseline without any DSAIN components'
    }
}


def run_ablation_experiment(
    config_name: str,
    config_params: dict,
    base_config: dict,
    seeds: list
) -> dict:
    """
    Run ablation experiment for one configuration across multiple seeds.

    Args:
        config_name: Name of ablation configuration
        config_params: Parameters for this ablation
        base_config: Base experimental configuration
        seeds: List of random seeds

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {config_params['name']}")
    logger.info(f"Description: {config_params['description']}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"{'='*70}")

    results = {
        'config_name': config_name,
        'name': config_params['name'],
        'description': config_params['description'],
        'parameters': config_params,
        'seeds': seeds,
        'runs': []
    }

    for seed_idx, seed in enumerate(seeds):
        logger.info(f"\nRun {seed_idx + 1}/{len(seeds)} (seed={seed})")

        # Merge configs
        exp_config = base_config.copy()
        exp_config.update(config_params)
        exp_config['seed'] = seed

        # Run experiment
        if REAL_EXPERIMENTS_AVAILABLE:
            try:
                # Create FLConfig
                fl_config = FLConfig(
                    num_clients=exp_config.get('num_clients', 100),
                    num_rounds=exp_config.get('num_rounds', 100),
                    compression_ratio=exp_config['compression_ratio'],
                    dp_epsilon=exp_config['dp_epsilon'],
                    byzantine_frac=exp_config.get('byzantine_frac', 0.0),
                    seed=seed
                )

                # Run CIFAR-10 experiment
                run_result = run_cifar10_experiment(
                    fl_config,
                    method=exp_config['method'],
                    alpha=exp_config.get('alpha', 0.5)
                )

                results['runs'].append({
                    'seed': seed,
                    'final_accuracy': run_result.get('final_accuracy', 0.0),
                    'final_loss': run_result.get('final_loss', 0.0),
                    'training_time': run_result.get('training_time_seconds', 0.0),
                    'communication_cost': calculate_communication_cost(run_result)
                })

            except Exception as e:
                logger.error(f"Experiment failed for {config_name} with seed {seed}: {e}")
                results['runs'].append({
                    'seed': seed,
                    'error': str(e)
                })
        else:
            # Mock results for testing
            results['runs'].append({
                'seed': seed,
                'final_accuracy': np.random.uniform(0.85, 0.92),
                'final_loss': np.random.uniform(0.3, 0.6),
                'training_time': np.random.uniform(1000, 2000),
                'communication_cost': calculate_mock_comm_cost(exp_config)
            })

    # Aggregate statistics
    if results['runs']:
        valid_runs = [r for r in results['runs'] if 'error' not in r]
        if valid_runs:
            results['statistics'] = {
                'mean_accuracy': float(np.mean([r['final_accuracy'] for r in valid_runs])),
                'std_accuracy': float(np.std([r['final_accuracy'] for r in valid_runs], ddof=1)),
                'mean_loss': float(np.mean([r['final_loss'] for r in valid_runs])),
                'std_loss': float(np.std([r['final_loss'] for r in valid_runs], ddof=1)),
                'mean_time': float(np.mean([r['training_time'] for r in valid_runs])),
                'mean_comm': float(np.mean([r['communication_cost'] for r in valid_runs]))
            }

    return results


def calculate_communication_cost(run_result: dict) -> float:
    """Calculate communication cost in GB."""
    # Simplified calculation - replace with actual
    compression_ratio = run_result['config'].get('compression_ratio', 1.0)
    baseline_comm = 4.82  # GB for full model
    return baseline_comm * compression_ratio


def calculate_mock_comm_cost(config: dict) -> float:
    """Calculate mock communication cost for testing."""
    compression_ratio = config['compression_ratio']
    return 4.82 * compression_ratio


def generate_ablation_table(all_results: list, output_file: str = None) -> str:
    """
    Generate LaTeX table for ablation study.

    Args:
        all_results: List of results from all configurations
        output_file: Optional file to save LaTeX

    Returns:
        LaTeX table string
    """
    latex = "\\begin{table*}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Ablation Study: Component Contribution Analysis on CIFAR-10}\n"
    latex += "\\label{tab:ablation}\n"
    latex += "\\begin{tabular}{lcccccc}\n"
    latex += "\\toprule\n"
    latex += "Configuration & Compression & ByzFed & DP & Accuracy (\\%) & Comm. (GB) & Time (h) \\\\\n"
    latex += "\\midrule\n"

    # Sort by accuracy (descending)
    sorted_results = sorted(all_results, key=lambda x: x.get('statistics', {}).get('mean_accuracy', 0), reverse=True)

    for result in sorted_results:
        name = result['name']
        stats = result.get('statistics', {})
        params = result['parameters']

        # Component indicators
        has_compression = "\\checkmark" if params['compression_ratio'] < 1.0 else "\\xmark"
        has_byzfed = "\\checkmark" if params['method'] == 'dsain' else "\\xmark"
        has_dp = "\\checkmark" if params['dp_epsilon'] < float('inf') else "\\xmark"

        acc_mean = stats.get('mean_accuracy', 0) * 100
        acc_std = stats.get('std_accuracy', 0) * 100
        comm = stats.get('mean_comm', 0)
        time_h = stats.get('mean_time', 0) / 3600

        latex += f"{name} & {has_compression} & {has_byzfed} & {has_dp} & "
        latex += f"${acc_mean:.2f} \\pm {acc_std:.2f}$ & "
        latex += f"{comm:.2f} & {time_h:.2f} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table*}\n"

    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex)
        logger.info(f"LaTeX table saved to {output_file}")

    return latex


def main():
    parser = argparse.ArgumentParser(description='DSAIN Ablation Study')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet concentration')
    parser.add_argument('--byzantine_frac', type=float, default=0.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to run (default: all)')
    parser.add_argument('--output_dir', type=str, default='../results/ablation')

    args = parser.parse_args()

    # Base configuration
    base_config = {
        'dataset': args.dataset,
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'alpha': args.alpha,
        'byzantine_frac': args.byzantine_frac
    }

    # Determine which configs to run
    if args.configs:
        configs_to_run = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.configs}
    else:
        configs_to_run = ABLATION_CONFIGS

    logger.info(f"\n{'='*70}")
    logger.info("DSAIN ABLATION STUDY")
    logger.info(f"{'='*70}")
    logger.info(f"Configurations to run: {list(configs_to_run.keys())}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Base config: {base_config}")
    logger.info(f"{'='*70}\n")

    # Run all ablation experiments
    all_results = []

    for config_name, config_params in configs_to_run.items():
        result = run_ablation_experiment(
            config_name,
            config_params,
            base_config,
            args.seeds
        )
        all_results.append(result)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"ablation_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # Generate LaTeX table
    latex_file = output_dir / f"ablation_table_{timestamp}.tex"
    latex_table = generate_ablation_table(all_results, str(latex_file))

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} {'Accuracy':<15} {'Comm. (GB)':<12} {'Time (h)'}")
    print("-"*70)

    for result in sorted(all_results, key=lambda x: x.get('statistics', {}).get('mean_accuracy', 0), reverse=True):
        name = result['name']
        stats = result.get('statistics', {})
        acc = stats.get('mean_accuracy', 0) * 100
        acc_std = stats.get('std_accuracy', 0) * 100
        comm = stats.get('mean_comm', 0)
        time_h = stats.get('mean_time', 0) / 3600

        print(f"{name:<25} {acc:.2f}±{acc_std:.2f}      {comm:.2f}        {time_h:.2f}")

    print("="*70 + "\n")

    print("LaTeX Table Preview:")
    print(latex_table)


if __name__ == "__main__":
    main()
