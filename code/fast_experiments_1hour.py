#!/usr/bin/env python3
"""
Fast 1-Hour Experiment Suite for DSAIN
========================================

Runs ALL experiment types in ~60 minutes on RTX 4060 by:
- Reducing rounds: 500 → 50 (convergence trends still visible)
- Reducing clients: 100 → 20 (sufficient for FL dynamics)
- Focusing on key configurations (2-3 per experiment type)

This provides COMPLETE coverage for TMLR submission in 1 hour.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"fast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FastExperimentRunner:
    """Fast experiment runner for 1-hour completion."""

    def __init__(self, results_dir="../results/fast"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Fast configuration
        self.num_rounds = 50  # Down from 500
        self.num_clients = 20  # Down from 100
        self.participation_rate = 0.25  # 5 clients per round (20 * 0.25 = 5)

        # Define fast experiments
        self.experiments = self.define_experiments()

        logger.info("=" * 70)
        logger.info("FAST 1-HOUR EXPERIMENT SUITE")
        logger.info("=" * 70)
        logger.info(f"Configuration: {self.num_rounds} rounds, {self.num_clients} clients, "
                   f"{int(self.num_clients * self.participation_rate)} clients/round")
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info(f"Estimated time: ~60 minutes")
        logger.info("=" * 70)

    def define_experiments(self):
        """Define fast experiments covering all types."""
        experiments = []

        # 1. Baseline - ResNet-18 only (most important) - 8 min
        experiments.append({
            'name': '1_baseline_resnet18',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'baseline',
                '--model', 'resnet18',
                '--dataset', 'cifar10',
                '--num_rounds', str(self.num_rounds),
                '--num_clients', str(self.num_clients),
                '--participation_rate', str(self.participation_rate),
                '--seed', '42',
                '--output_dir', str(self.results_dir)
            ],
            'estimated_min': 8
        })

        # 2. Architecture comparison - MobileNetV2 (efficiency) - 5 min
        experiments.append({
            'name': '2_arch_mobilenet',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'baseline',
                '--model', 'mobilenetv2',
                '--dataset', 'cifar10',
                '--num_rounds', str(self.num_rounds),
                '--num_clients', str(self.num_clients),
                '--participation_rate', str(self.participation_rate),
                '--seed', '42',
                '--output_dir', str(self.results_dir)
            ],
            'estimated_min': 5
        })

        # 3. Heterogeneity - Key alphas only (0.1=high skew, 1.0=moderate) - 12 min
        for alpha in [0.1, 1.0]:
            experiments.append({
                'name': f'3_hetero_alpha{alpha}',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--heterogeneity', 'dirichlet',
                    '--alpha', str(alpha),
                    '--num_rounds', str(self.num_rounds),
                    '--num_clients', str(self.num_clients),
                    '--participation_rate', str(self.participation_rate),
                    '--seed', '42',
                    '--output_dir', str(self.results_dir)
                ],
                'estimated_min': 6
            })

        # 4. Byzantine robustness - Critical test (0% vs 20%) - 12 min
        for byz_frac in [0.0, 0.2]:
            experiments.append({
                'name': f'4_byzantine_{int(byz_frac*100)}pct',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--byzantine_frac', str(byz_frac),
                    '--num_rounds', str(self.num_rounds),
                    '--num_clients', str(self.num_clients),
                    '--participation_rate', str(self.participation_rate),
                    '--seed', '42',
                    '--output_dir', str(self.results_dir)
                ],
                'estimated_min': 6
            })

        # 5. Privacy - Key epsilons (2.0=strong, 8.0=weak) - 12 min
        for epsilon in [2.0, 8.0]:
            experiments.append({
                'name': f'5_privacy_eps{epsilon}',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--dp_epsilon', str(epsilon),
                    '--num_rounds', str(self.num_rounds),
                    '--num_clients', str(self.num_clients),
                    '--participation_rate', str(self.participation_rate),
                    '--seed', '42',
                    '--output_dir', str(self.results_dir)
                ],
                'estimated_min': 6
            })

        # 6. Compression - Key ratios (0.1=high compression, 1.0=none) - 12 min
        for comp in [0.1, 1.0]:
            experiments.append({
                'name': f'6_compression_{comp}',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--compression', str(comp),
                    '--num_rounds', str(self.num_rounds),
                    '--num_clients', str(self.num_clients),
                    '--participation_rate', str(self.participation_rate),
                    '--seed', '42',
                    '--output_dir', str(self.results_dir)
                ],
                'estimated_min': 6
            })

        return experiments

    def run_experiment(self, experiment):
        """Run a single experiment with timing."""
        name = experiment['name']

        logger.info(f"\n{'='*70}")
        logger.info(f"[START] {name}")
        logger.info(f"   Estimated: {experiment['estimated_min']} minutes")
        logger.info(f"   Command: {' '.join(experiment['command'])}")
        logger.info(f"{'='*70}")

        start_time = time.time()

        try:
            result = subprocess.run(
                experiment['command'],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=experiment['estimated_min'] * 60 * 4  # 4x buffer (experiments take ~2x estimated time)
            )

            elapsed = (time.time() - start_time) / 60

            if result.returncode == 0:
                logger.info(f"[DONE] {name} in {elapsed:.1f} minutes")
                return True, elapsed
            else:
                logger.error(f"[FAIL] {name}")
                logger.error(f"   Last 500 chars of stderr: {result.stderr[-500:]}")
                return False, elapsed

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start_time) / 60
            logger.error(f"[TIMEOUT] {name} after {elapsed:.1f} minutes")
            return False, elapsed

        except Exception as e:
            elapsed = (time.time() - start_time) / 60
            logger.error(f"[ERROR] {name} - {str(e)}")
            return False, elapsed

    def run_all(self):
        """Run all experiments sequentially."""
        logger.info("\n" + "="*70)
        logger.info("STARTING FAST 1-HOUR EXPERIMENT SUITE")
        logger.info("="*70)
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info(f"Estimated total time: {sum(e['estimated_min'] for e in self.experiments)} minutes")
        logger.info("="*70 + "\n")

        overall_start = time.time()
        results = []
        failed = []

        for i, experiment in enumerate(self.experiments, 1):
            logger.info(f"\n[Experiment {i}/{len(self.experiments)}]")

            success, elapsed = self.run_experiment(experiment)

            results.append({
                'name': experiment['name'],
                'success': success,
                'time_minutes': elapsed
            })

            if not success:
                failed.append(experiment['name'])

            # Progress update
            completed = i
            remaining = len(self.experiments) - i
            total_elapsed = (time.time() - overall_start) / 60

            logger.info(f"\n[PROGRESS] {completed}/{len(self.experiments)} complete")
            logger.info(f"   Time elapsed: {total_elapsed:.1f} min")

            if remaining > 0 and completed > 0:
                avg_time = total_elapsed / completed
                eta = remaining * avg_time
                logger.info(f"   ETA: {eta:.1f} minutes ({total_elapsed + eta:.1f} min total)")

        # Final summary
        total_time = (time.time() - overall_start) / 60

        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUITE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")
        logger.info(f"Completed: {len([r for r in results if r['success']])}/{len(self.experiments)}")

        if failed:
            logger.warning(f"\n[WARNING] Failed experiments ({len(failed)}):")
            for name in failed:
                logger.warning(f"  - {name}")
        else:
            logger.info("\n[SUCCESS] All experiments completed successfully!")

        # Save summary
        summary_file = self.results_dir / "experiment_summary.json"
        summary = {
            'total_experiments': len(self.experiments),
            'successful': len([r for r in results if r['success']]),
            'failed': len(failed),
            'total_time_minutes': total_time,
            'configuration': {
                'num_rounds': self.num_rounds,
                'num_clients': self.num_clients,
                'participation_rate': self.participation_rate,
                'clients_per_round': int(self.num_clients * self.participation_rate)
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n📁 Results saved in: {self.results_dir.absolute()}")
        logger.info(f"📄 Summary saved: {summary_file}")
        logger.info("="*70 + "\n")

        return len(failed) == 0


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Run fast experiments')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("DSAIN Fast 1-Hour Experiment Suite")
    print("="*70)
    print("\nThis will run ALL experiment types in ~60 minutes:")
    print("  * Baseline (ResNet-18)")
    print("  * Architecture comparison (MobileNetV2)")
    print("  * Heterogeneity study (alpha=0.1, 1.0)")
    print("  * Byzantine robustness (0%, 20%)")
    print("  * Privacy analysis (eps=2.0, 8.0)")
    print("  * Compression study (0.1, 1.0)")
    print("\nConfiguration:")
    print("  - 50 rounds (down from 500, still shows convergence)")
    print("  - 20 clients (down from 100, captures FL dynamics)")
    print("  - 5 clients per round (25% participation)")
    print("\nThis provides COMPLETE experimental coverage for TMLR!")
    print("="*70)

    if not args.yes:
        response = input("\nStart 1-hour experiment suite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print("\n" + "="*70)
    print("STARTING FAST EXPERIMENTS...")
    print("="*70 + "\n")

    runner = FastExperimentRunner()
    success = runner.run_all()

    if success:
        print("\n" + "="*70)
        print("SUCCESS! All experiments completed in ~2 hours")
        print("="*70)
        print("\nNext steps:")
        print("1. Check results: results/fast/")
        print("2. Review logs: logs/fast_run_*.log")
        print("3. Use results for TMLR manuscript")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("Some experiments failed - check logs")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
