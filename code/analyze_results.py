#!/usr/bin/env python3
"""
Result Analysis and Manuscript Generator
=========================================

Analyzes experiment results and generates LaTeX tables/text for manuscript.

Usage:
    python analyze_results.py --results_dir ../results/fast
    python analyze_results.py --results_dir ../results/deep
    python analyze_results.py --results_dir path/to/DSAIN_Full_Results
"""

import json
import glob
import argparse
from pathlib import Path
import numpy as np


def load_all_results(results_dir):
    """Load all JSON result files from directory."""
    results_dir = Path(results_dir)
    json_files = glob.glob(str(results_dir / "enhanced_*.json"))

    results = {}
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            # Extract experiment type from filename or data
            key = Path(file).stem
            results[key] = data

    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


def analyze_baseline(results):
    """Analyze baseline experiments."""
    print("\n" + "="*70)
    print("BASELINE RESULTS")
    print("="*70)

    baseline_results = {k: v for k, v in results.items()
                       if 'baseline' in k.lower() and 'byzantine' not in k.lower()
                       and 'hetero' not in k.lower()}

    if not baseline_results:
        print("No baseline results found")
        return

    print(f"\nFound {len(baseline_results)} baseline experiments\n")

    # LaTeX table
    print("LaTeX Table (Architecture Comparison):")
    print("```latex")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Model Performance on CIFAR-10}")
    print("\\begin{tabular}{lrrr}")
    print("\\hline")
    print("Model & Parameters & Accuracy (\\%) & Time/Round (s) \\\\")
    print("\\hline")

    for key, data in sorted(baseline_results.items()):
        model = data.get('model', 'Unknown')
        params = data.get('model_parameters', 0) / 1e6  # Convert to millions
        acc = data.get('final_accuracy', 0) * 100
        time_per_round = data.get('time_per_round', 0)

        print(f"{model.replace('_', ' ').title()} & {params:.1f}M & {acc:.1f}\\% & {time_per_round:.1f} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:architecture}")
    print("\\end{table}")
    print("```\n")


def analyze_byzantine(results):
    """Analyze Byzantine robustness."""
    print("\n" + "="*70)
    print("BYZANTINE ROBUSTNESS")
    print("="*70)

    byzantine_results = {k: v for k, v in results.items() if 'byzantine' in k.lower()}

    if not byzantine_results:
        print("No Byzantine results found")
        return

    print(f"\nFound {len(byzantine_results)} Byzantine experiments\n")

    # Extract results by Byzantine fraction
    by_fraction = {}
    for key, data in byzantine_results.items():
        byz_frac = data.get('byzantine_frac', 0)
        by_fraction[byz_frac] = data.get('final_accuracy', 0) * 100

    # LaTeX table
    print("LaTeX Table (Byzantine Robustness):")
    print("```latex")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Byzantine Robustness Analysis}")
    print("\\begin{tabular}{lrr}")
    print("\\hline")
    print("Byzantine \\% & Accuracy (\\%) & Degradation (\\%) \\\\")
    print("\\hline")

    baseline_acc = by_fraction.get(0.0, by_fraction.get(0, None))

    for frac in sorted(by_fraction.keys()):
        acc = by_fraction[frac]
        if baseline_acc and frac > 0:
            degradation = ((baseline_acc - acc) / baseline_acc) * 100
            print(f"{int(frac*100)}\\% & {acc:.1f}\\% & {degradation:.1f}\\% \\\\")
        else:
            print(f"{int(frac*100)}\\% & {acc:.1f}\\% & -- \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:byzantine}")
    print("\\end{table}")
    print("```\n")

    # Key finding for manuscript
    if baseline_acc and 0.2 in by_fraction:
        attack_acc = by_fraction[0.2]
        degradation = ((baseline_acc - attack_acc) / baseline_acc) * 100
        print("Manuscript Text:")
        print(f"Under 20% Byzantine attacks, DSAIN maintains {attack_acc:.1f}% accuracy ")
        print(f"({degradation:.1f}% degradation), demonstrating strong resilience.")


def analyze_heterogeneity(results):
    """Analyze heterogeneity experiments."""
    print("\n" + "="*70)
    print("HETEROGENEITY ANALYSIS")
    print("="*70)

    hetero_results = {k: v for k, v in results.items() if 'hetero' in k.lower() or 'alpha' in k.lower()}

    if not hetero_results:
        print("No heterogeneity results found")
        return

    print(f"\nFound {len(hetero_results)} heterogeneity experiments\n")

    # Extract results by alpha
    by_alpha = {}
    for key, data in hetero_results.items():
        alpha = data.get('dirichlet_alpha', data.get('alpha', None))
        if alpha is not None:
            by_alpha[float(alpha)] = data.get('final_accuracy', 0) * 100

    if not by_alpha:
        print("No alpha values found in results")
        return

    # LaTeX table
    print("LaTeX Table (Heterogeneity Impact):")
    print("```latex")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Impact of Data Heterogeneity (Dirichlet $\\\\alpha$)}")
    print("\\begin{tabular}{lr}")
    print("\\hline")
    print("$\\\\alpha$ & Accuracy (\\%) \\\\")
    print("\\hline")

    for alpha in sorted(by_alpha.keys()):
        acc = by_alpha[alpha]
        print(f"{alpha} & {acc:.1f}\\% \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:heterogeneity}")
    print("\\end{table}")
    print("```\n")

    # Key findings
    if 0.1 in by_alpha and 1.0 in by_alpha:
        print("Manuscript Text:")
        print(f"DSAIN achieves {by_alpha[0.1]:.1f}% accuracy under extreme non-IID conditions ")
        print(f"(α=0.1) and {by_alpha[1.0]:.1f}% under moderate heterogeneity (α=1.0).")


def analyze_privacy(results):
    """Analyze privacy experiments."""
    print("\n" + "="*70)
    print("PRIVACY-UTILITY TRADEOFF")
    print("="*70)

    privacy_results = {k: v for k, v in results.items() if 'privacy' in k.lower() or 'epsilon' in k.lower()}

    if not privacy_results:
        print("No privacy results found")
        return

    print(f"\nFound {len(privacy_results)} privacy experiments\n")

    # Extract results by epsilon
    by_epsilon = {}
    for key, data in privacy_results.items():
        epsilon = data.get('dp_epsilon', None)
        if epsilon is not None and epsilon != float('inf'):
            by_epsilon[float(epsilon)] = data.get('final_accuracy', 0) * 100

    if not by_epsilon:
        print("No epsilon values found in results")
        return

    # LaTeX table
    print("LaTeX Table (Privacy-Utility Tradeoff):")
    print("```latex")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Privacy-Utility Tradeoff}")
    print("\\begin{tabular}{lr}")
    print("\\hline")
    print("$\\\\epsilon$ & Accuracy (\\%) \\\\")
    print("\\hline")

    for eps in sorted(by_epsilon.keys()):
        acc = by_epsilon[eps]
        print(f"{eps} & {acc:.1f}\\% \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:privacy}")
    print("\\end{table}")
    print("```\n")

    # Key findings
    if 2.0 in by_epsilon and 8.0 in by_epsilon:
        print("Manuscript Text:")
        print(f"With strong privacy (ε=2.0), DSAIN achieves {by_epsilon[2.0]:.1f}% accuracy. ")
        print(f"Relaxing to ε=8.0 improves accuracy to {by_epsilon[8.0]:.1f}%, demonstrating ")
        print(f"the privacy-utility tradeoff.")


def analyze_compression(results):
    """Analyze compression experiments."""
    print("\n" + "="*70)
    print("COMPRESSION EFFICIENCY")
    print("="*70)

    compression_results = {k: v for k, v in results.items() if 'compression' in k.lower()}

    if not compression_results:
        print("No compression results found")
        return

    print(f"\nFound {len(compression_results)} compression experiments\n")

    # Extract results by compression ratio
    by_ratio = {}
    for key, data in compression_results.items():
        ratio = data.get('compression_ratio', None)
        if ratio is not None:
            by_ratio[float(ratio)] = data.get('final_accuracy', 0) * 100

    if not by_ratio:
        print("No compression ratios found in results")
        return

    print("Manuscript Text:")
    for ratio in sorted(by_ratio.keys()):
        acc = by_ratio[ratio]
        if ratio < 1.0:
            reduction = (1 - ratio) * 100
            print(f"With {reduction:.0f}% communication reduction (ratio={ratio}), ")
            print(f"DSAIN maintains {acc:.1f}% accuracy.")
        else:
            print(f"Without compression (ratio={ratio}), accuracy is {acc:.1f}%.")


def generate_summary(results):
    """Generate overall summary."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print(f"\nTotal experiments: {len(results)}")

    # Average accuracy
    accuracies = [data.get('final_accuracy', 0) * 100 for data in results.values()]
    if accuracies:
        print(f"Average accuracy: {np.mean(accuracies):.1f}% (±{np.std(accuracies):.1f}%)")
        print(f"Min accuracy: {min(accuracies):.1f}%")
        print(f"Max accuracy: {max(accuracies):.1f}%")

    # Total time
    total_time = sum(data.get('total_time_hours', 0) for data in results.values())
    print(f"\nTotal experiment time: {total_time:.1f} hours ({total_time/24:.1f} days)")


def main():
    parser = argparse.ArgumentParser(description='Analyze DSAIN experiment results')
    parser.add_argument('--results_dir', type=str, default='../results/fast',
                       help='Directory containing result JSON files')

    args = parser.parse_args()

    # Load results
    results = load_all_results(args.results_dir)

    if not results:
        print(f"No results found in {args.results_dir}")
        return

    # Analyze different experiment types
    analyze_baseline(results)
    analyze_byzantine(results)
    analyze_heterogeneity(results)
    analyze_privacy(results)
    analyze_compression(results)
    generate_summary(results)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Copy LaTeX tables to your manuscript")
    print("2. Use manuscript text snippets in Section 5 (Results)")
    print("3. Generate convergence plots from rounds_history in JSON files")
    print("\nFor more help, share these results with Claude!")


if __name__ == "__main__":
    main()
