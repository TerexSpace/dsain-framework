#!/usr/bin/env python3
"""30-Experiment Comprehensive Suite for TMLR"""
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENTS = [
    # Baseline models (3)
    {'name': '1_baseline_resnet18', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50},
    {'name': '2_baseline_resnet34', 'exp': 'baseline', 'model': 'resnet34', 'rounds': 50},
    {'name': '3_baseline_mobilenet', 'exp': 'baseline', 'model': 'mobilenetv2', 'rounds': 50},
    
    # Heterogeneity sweep (5)
    {'name': '4_hetero_alpha0.01', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.01},
    {'name': '5_hetero_alpha0.1', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.1},
    {'name': '6_hetero_alpha0.5', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.5},
    {'name': '7_hetero_alpha1.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 1.0},
    {'name': '8_hetero_alpha10.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 10.0},
    
    # Byzantine robustness (6)
    {'name': '9_byzantine_0pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.0},
    {'name': '10_byzantine_5pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.05},
    {'name': '11_byzantine_10pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.10},
    {'name': '12_byzantine_15pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.15},
    {'name': '13_byzantine_20pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.20},
    {'name': '14_byzantine_25pct', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.25},
    
    # Privacy levels (6)
    {'name': '15_privacy_eps1.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': 1.0},
    {'name': '16_privacy_eps2.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': 2.0},
    {'name': '17_privacy_eps4.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': 4.0},
    {'name': '18_privacy_eps8.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': 8.0},
    {'name': '19_privacy_eps16.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': 16.0},
    {'name': '20_privacy_noprivacy', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dp_epsilon': float('inf')},
    
    # Compression ratios (5)
    {'name': '21_compress_0.05', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 0.05},
    {'name': '22_compress_0.1', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 0.1},
    {'name': '23_compress_0.2', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 0.2},
    {'name': '24_compress_0.5', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 0.5},
    {'name': '25_compress_1.0', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 1.0},
    
    # Combined scenarios (5)
    {'name': '26_combined_hetero_byz', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.1, 'byzantine_frac': 0.2},
    {'name': '27_combined_hetero_privacy', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.1, 'dp_epsilon': 2.0},
    {'name': '28_combined_byz_privacy', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'byzantine_frac': 0.2, 'dp_epsilon': 2.0},
    {'name': '29_combined_compress_hetero', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'compression_ratio': 0.1, 'dirichlet_alpha': 0.1},
    {'name': '30_combined_all', 'exp': 'baseline', 'model': 'resnet18', 'rounds': 50, 'dirichlet_alpha': 0.1, 'byzantine_frac': 0.1, 'dp_epsilon': 4.0, 'compression_ratio': 0.2},
]

def run_experiment(exp_config):
    name = exp_config['name']
    cmd = ['python', 'enhanced_experiments.py', '--exp', exp_config['exp'], '--model', exp_config['model'],
           '--dataset', 'cifar10', '--num_rounds', str(exp_config['rounds']), '--num_clients', '20',
           '--participation_rate', '0.25', '--seed', '42', '--output_dir', '../results/full']
    
    if 'dirichlet_alpha' in exp_config:
        cmd.extend(['--heterogeneity', 'dirichlet', '--alpha', str(exp_config['dirichlet_alpha'])])
    if 'byzantine_frac' in exp_config:
        cmd.extend(['--byzantine_frac', str(exp_config['byzantine_frac'])])
    if 'dp_epsilon' in exp_config:
        cmd.extend(['--dp_epsilon', str(exp_config['dp_epsilon'])])
    if 'compression_ratio' in exp_config:
        cmd.extend(['--compression_ratio', str(exp_config['compression_ratio'])])
    
    logger.info(f"[START] {name}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"[DONE] {name}")
        return True
    except Exception as e:
        logger.error(f"[FAIL] {name}: {e}")
        return False

def main():
    logger.info("="*70)
    logger.info("30-EXPERIMENT COMPREHENSIVE SUITE")
    logger.info("="*70)
    logger.info("Total: 30 experiments, 50 rounds each")
    logger.info("Estimated: ~20 hours (budget-optimized)")
    logger.info("="*70)
    
    Path('../results/full').mkdir(parents=True, exist_ok=True)
    completed = 0
    for i, exp in enumerate(EXPERIMENTS, 1):
        logger.info(f"\n[Experiment {i}/30]")
        if run_experiment(exp):
            completed += 1
        logger.info(f"[PROGRESS] {completed}/30 complete")
    
    logger.info("\n" + "="*70)
    logger.info(f"FINAL: {completed}/30 successful")
    logger.info("="*70)

if __name__ == "__main__":
    main()
