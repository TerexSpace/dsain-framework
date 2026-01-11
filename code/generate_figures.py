#!/usr/bin/env python3
"""
Generate publication-quality figures from E1-E10 experimental results.
Creates figures for TMLR supplementary materials using ACTUAL experimental data.

Author: Almas Ospanov
License: MIT
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'TMLR_Supplimentary_Materials', 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)

def load_experiment(exp_id):
    """Load experiment results by ID (e.g., 'E1')."""
    pattern = f"{exp_id}_"
    for fname in os.listdir(RESULTS_DIR):
        if fname.startswith(pattern) and fname.endswith('.json'):
            filepath = os.path.join(RESULTS_DIR, fname)
            with open(filepath, 'r') as f:
                data = json.load(f)
                print(f"  Loaded {fname}")
                return data
    print(f"  WARNING: Could not find {exp_id}")
    return None

def plot_convergence_curves():
    """Plot convergence curves for baseline comparison (E1 vs E2)."""
    print("\n[1/6] Generating convergence_curves...")

    e1 = load_experiment('E1')  # DSAIN baseline
    e2 = load_experiment('E2')  # FedAvg baseline

    if not e1 or not e2:
        print("  ERROR: Missing E1 or E2 data")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    rounds = e1['history']['round']
    dsain_acc = [a * 100 for a in e1['history']['accuracy']]
    fedavg_acc = [a * 100 for a in e2['history']['accuracy']]

    ax.plot(rounds, dsain_acc, 'b-o', linewidth=2, markersize=6,
            label=f"DSAIN (final: {e1['final_accuracy']*100:.2f}%)")
    ax.plot(rounds, fedavg_acc, 'r-s', linewidth=2, markersize=6,
            label=f"FedAvg (final: {e2['final_accuracy']*100:.2f}%)")

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Convergence Comparison: DSAIN vs FedAvg\n(CIFAR-10, ResNet18, α=0.5, 500 rounds)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 520])
    ax.set_ylim([40, 85])
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'convergence_curves.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'convergence_curves.png'))
    plt.close(fig)
    print("  -> Saved convergence_curves.pdf/png")
    return True

def plot_byzantine_resilience():
    """Plot Byzantine resilience across attack intensities (E1, E10, E3)."""
    print("\n[2/6] Generating byzantine_resilience...")

    e1 = load_experiment('E1')   # 0% Byzantine
    e10 = load_experiment('E10') # 10% Byzantine
    e3 = load_experiment('E3')   # 20% Byzantine

    if not all([e1, e10, e3]):
        print("  ERROR: Missing E1, E10, or E3 data")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))

    byzantine_fracs = [0, 10, 20]
    accuracies = [
        e1['final_accuracy'] * 100,
        e10['final_accuracy'] * 100,
        e3['final_accuracy'] * 100
    ]

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax.bar(byzantine_fracs, accuracies, width=6, color=colors,
                  edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Byzantine Fraction (%)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('DSAIN Byzantine Resilience: Dose-Response Analysis\n(Label-Flipping Attack, α=0.5, 500 rounds)')
    ax.set_xticks(byzantine_fracs)
    ax.set_xticklabels(['0%\n(E1: Clean)', '10%\n(E10)', '20%\n(E3)'])
    ax.set_ylim([0, 90])
    ax.axhline(y=e1['final_accuracy']*100, color='gray', linestyle='--', alpha=0.5)

    fig.savefig(os.path.join(FIGURES_DIR, 'byzantine_resilience.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'byzantine_resilience.png'))
    plt.close(fig)
    print("  -> Saved byzantine_resilience.pdf/png")
    return True

def plot_byzantine_comparison():
    """Compare DSAIN vs FedAvg under Byzantine attack (E3 vs E4)."""
    print("\n[3/6] Generating byzantine_comparison...")

    e1 = load_experiment('E1')  # DSAIN clean
    e2 = load_experiment('E2')  # FedAvg clean
    e3 = load_experiment('E3')  # DSAIN 20% byz
    e4 = load_experiment('E4')  # FedAvg 20% byz

    if not all([e1, e2, e3, e4]):
        print("  ERROR: Missing experiment data")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(2)
    width = 0.35

    dsain_accs = [e1['final_accuracy']*100, e3['final_accuracy']*100]
    fedavg_accs = [e2['final_accuracy']*100, e4['final_accuracy']*100]

    bars1 = ax.bar(x - width/2, dsain_accs, width, label='DSAIN',
                   color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, fedavg_accs, width, label='FedAvg',
                   color='#e74c3c', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Condition')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Byzantine Attack Comparison: DSAIN vs FedAvg\n(20% Label-Flipping Attack, α=0.5, 500 rounds)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Clean (0% Byzantine)', '20% Byzantine Attack'])
    ax.legend(loc='upper right')
    ax.set_ylim([0, 90])
    ax.grid(True, axis='y', alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'byzantine_comparison.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'byzantine_comparison.png'))
    plt.close(fig)
    print("  -> Saved byzantine_comparison.pdf/png")
    return True

def plot_heterogeneity():
    """Plot impact of heterogeneity (replaces scalability with actual data)."""
    print("\n[4/6] Generating scalability (heterogeneity analysis)...")

    e1 = load_experiment('E1')  # DSAIN α=0.5
    e2 = load_experiment('E2')  # FedAvg α=0.5
    e5 = load_experiment('E5')  # DSAIN α=1.0
    e6 = load_experiment('E6')  # FedAvg α=1.0
    e7 = load_experiment('E7')  # DSAIN α=0.1
    e8 = load_experiment('E8')  # FedAvg α=0.1

    if not all([e1, e2, e5, e6, e7, e8]):
        print("  ERROR: Missing experiment data")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))

    alphas = [0.1, 0.5, 1.0]
    dsain_accs = [e7['final_accuracy']*100, e1['final_accuracy']*100, e5['final_accuracy']*100]
    fedavg_accs = [e8['final_accuracy']*100, e2['final_accuracy']*100, e6['final_accuracy']*100]

    ax.plot(alphas, dsain_accs, 'b-o', linewidth=2.5, markersize=12, label='DSAIN')
    ax.plot(alphas, fedavg_accs, 'r-s', linewidth=2.5, markersize=12, label='FedAvg')

    # Add value labels
    for a, d, f in zip(alphas, dsain_accs, fedavg_accs):
        ax.annotate(f'{d:.1f}%', (a, d), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=11, fontweight='bold', color='blue')
        ax.annotate(f'{f:.1f}%', (a, f), textcoords="offset points",
                    xytext=(0, -18), ha='center', fontsize=11, fontweight='bold', color='red')

    # Mark critical threshold
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label='Critical threshold (α=0.5)')
    ax.axvspan(0, 0.5, alpha=0.1, color='red')

    ax.set_xlabel('Dirichlet α (lower = more heterogeneous)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Impact of Data Heterogeneity on Federated Learning\n(CIFAR-10, ResNet18, 500 rounds)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1.1])
    ax.set_ylim([50, 90])
    ax.set_xticks([0.1, 0.5, 1.0])
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'scalability.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'scalability.png'))
    plt.close(fig)
    print("  -> Saved scalability.pdf/png (heterogeneity analysis)")
    return True

def plot_all_experiments_summary():
    """Create summary bar chart of all 10 experiments."""
    print("\n[5/6] Generating all experiments summary...")

    experiments = []
    for i in range(1, 11):
        exp = load_experiment(f'E{i}')
        if exp:
            experiments.append((f"E{i}", exp['final_accuracy']*100, exp['config']['exp_name']))

    if len(experiments) < 10:
        print(f"  WARNING: Only found {len(experiments)}/10 experiments")
        if len(experiments) == 0:
            return False

    fig, ax = plt.subplots(figsize=(14, 6))

    names = [e[0] for e in experiments]
    accs = [e[1] for e in experiments]
    labels = [e[2] for e in experiments]

    # Color by type
    colors = []
    for label in labels:
        if 'byz' in label.lower():
            colors.append('#e74c3c')  # Red for Byzantine
        elif 'dp' in label.lower():
            colors.append('#9b59b6')  # Purple for DP
        elif 'FedAvg' in label:
            colors.append('#f39c12')  # Orange for FedAvg
        else:
            colors.append('#3498db')  # Blue for DSAIN clean

    bars = ax.bar(names, accs, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, rotation=0)

    ax.set_xlabel('Experiment ID')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Complete Experimental Results: All 10 Experiments\n(CIFAR-10, ResNet18, 500 rounds, seed=42)')
    ax.set_ylim([0, 95])
    ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='DSAIN (clean)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='FedAvg'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Byzantine attack'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Differential privacy'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'all_experiments_summary.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'all_experiments_summary.png'))
    plt.close(fig)
    print("  -> Saved all_experiments_summary.pdf/png")
    return True

def plot_convergence_multi():
    """Plot convergence curves for multiple experiments."""
    print("\n[6/6] Generating multi-experiment convergence...")

    e1 = load_experiment('E1')   # DSAIN α=0.5
    e5 = load_experiment('E5')   # DSAIN α=1.0
    e7 = load_experiment('E7')   # DSAIN α=0.1
    e3 = load_experiment('E3')   # DSAIN byz 20%

    if not all([e1, e5, e7, e3]):
        print("  ERROR: Missing experiment data")
        return False

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each experiment
    experiments = [
        (e5, 'α=1.0 (mild)', '#2ecc71', '-'),
        (e1, 'α=0.5 (moderate)', '#3498db', '-'),
        (e7, 'α=0.1 (severe)', '#e74c3c', '-'),
        (e3, 'α=0.5 + 20% Byz', '#9b59b6', '--'),
    ]

    for exp, label, color, style in experiments:
        rounds = exp['history']['round']
        accs = [a * 100 for a in exp['history']['accuracy']]
        ax.plot(rounds, accs, color=color, linestyle=style, linewidth=2,
                label=f"{label}: {exp['final_accuracy']*100:.2f}%")

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('DSAIN Convergence Under Different Conditions\n(CIFAR-10, ResNet18, 500 rounds)')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 520])
    ax.set_ylim([30, 85])
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIGURES_DIR, 'convergence_multi.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'convergence_multi.png'))
    plt.close(fig)
    print("  -> Saved convergence_multi.pdf/png")
    return True

def main():
    print("=" * 70)
    print("GENERATING FIGURES FROM E1-E10 EXPERIMENTAL RESULTS")
    print("=" * 70)
    print(f"Results directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"Figures directory: {os.path.abspath(FIGURES_DIR)}")

    # Check results exist
    if not os.path.exists(RESULTS_DIR):
        print(f"\nERROR: Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    # Count available experiments
    exp_count = sum(1 for f in os.listdir(RESULTS_DIR)
                    if f.startswith('E') and f.endswith('.json'))
    print(f"Found {exp_count} experiment files")

    # Generate all figures
    success = []
    success.append(plot_convergence_curves())
    success.append(plot_byzantine_resilience())
    success.append(plot_byzantine_comparison())
    success.append(plot_heterogeneity())
    success.append(plot_all_experiments_summary())
    success.append(plot_convergence_multi())

    print("\n" + "=" * 70)
    print(f"FIGURE GENERATION COMPLETE: {sum(success)}/{len(success)} successful")
    print("=" * 70)
    print(f"\nFigures saved to: {os.path.abspath(FIGURES_DIR)}")

    if all(success):
        print("\nAll figures generated successfully!")
        return 0
    else:
        print("\nSome figures failed to generate. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
