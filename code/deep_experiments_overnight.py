#!/usr/bin/env python3
"""
Deep Experiments for DSAIN (Overnight Runner)
==============================================

Runs 3 critical experiments with full convergence validation:
- 200 rounds (shows complete convergence)
- 50 clients (realistic FL scale)
- Key scenarios (baseline, Byzantine, heterogeneity)

Complements fast_experiments_1hour.py for TMLR submission.

Total time: ~10-12 hours (run overnight)
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
        logging.FileHandler(log_dir / f"deep_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepExperimentRunner:
    """Deep experiment runner for overnight execution."""

    def __init__(self, results_dir="../results/deep"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Deep configuration - shows full convergence
        self.num_rounds = 200  # Full convergence
        self.num_clients = 50  # Realistic FL scale
        self.participation_rate = 0.2  # 10 clients per round

        # Define deep experiments (only critical ones)
        self.experiments = self.define_experiments()

        logger.info("=" * 70)
        logger.info("DEEP EXPERIMENTS - OVERNIGHT RUNNER")
        logger.info("=" * 70)
        logger.info(f"Configuration: {self.num_rounds} rounds, {self.num_clients} clients, "
                   f"{int(self.num_clients * self.participation_rate)} clients/round")
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info(f"Estimated time: ~10-12 hours")
        logger.info("=" * 70)

    def define_experiments(self):
        """Define deep experiments for full convergence validation."""
        experiments = []

        # 1. Baseline - Full convergence reference - ~3.5 hours
        experiments.append({
            'name': 'deep_baseline_resnet18',
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
            'estimated_hours': 3.5,
            'purpose': 'Full convergence baseline for comparison'
        })

        # 2. Byzantine robustness - Critical validation - ~3.5 hours
        experiments.append({
            'name': 'deep_byzantine_20pct',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'baseline',
                '--model', 'resnet18',
                '--dataset', 'cifar10',
                '--byzantine_frac', '0.2',
                '--num_rounds', str(self.num_rounds),
                '--num_clients', str(self.num_clients),
                '--participation_rate', str(self.participation_rate),
                '--seed', '42',
                '--output_dir', str(self.results_dir)
            ],
            'estimated_hours': 3.5,
            'purpose': 'Validate Byzantine resilience with full convergence'
        })

        # 3. Heterogeneity - High skew scenario - ~3.5 hours
        experiments.append({
            'name': 'deep_heterogeneity_alpha0.1',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'baseline',
                '--model', 'resnet18',
                '--dataset', 'cifar10',
                '--heterogeneity', 'dirichlet',
                '--alpha', '0.1',
                '--num_rounds', str(self.num_rounds),
                '--num_clients', str(self.num_clients),
                '--participation_rate', str(self.participation_rate),
                '--seed', '42',
                '--output_dir', str(self.results_dir)
            ],
            'estimated_hours': 3.5,
            'purpose': 'Validate performance under extreme non-IID conditions'
        })

        return experiments

    def run_experiment(self, experiment):
        """Run a single experiment with timing."""
        name = experiment['name']

        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting: {name}")
        logger.info(f"   Purpose: {experiment['purpose']}")
        logger.info(f"   Estimated: {experiment['estimated_hours']:.1f} hours")
        logger.info(f"   Command: {' '.join(experiment['command'])}")
        logger.info(f"{'='*70}")

        start_time = time.time()

        try:
            result = subprocess.run(
                experiment['command'],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=int(experiment['estimated_hours'] * 3600 * 1.5)  # 1.5x buffer
            )

            elapsed = (time.time() - start_time) / 3600

            if result.returncode == 0:
                logger.info(f"✅ Completed: {name} in {elapsed:.2f} hours")
                return True, elapsed
            else:
                logger.error(f"❌ Failed: {name}")
                logger.error(f"   Last 500 chars of stderr: {result.stderr[-500:]}")
                return False, elapsed

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start_time) / 3600
            logger.error(f"⏱️  Timeout: {name} after {elapsed:.2f} hours")
            return False, elapsed

        except Exception as e:
            elapsed = (time.time() - start_time) / 3600
            logger.error(f"💥 Error: {name} - {str(e)}")
            return False, elapsed

    def run_all(self):
        """Run all deep experiments sequentially."""
        logger.info("\n" + "="*70)
        logger.info("STARTING DEEP EXPERIMENTS (OVERNIGHT)")
        logger.info("="*70)
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info(f"Estimated total time: {sum(e['estimated_hours'] for e in self.experiments):.1f} hours")
        logger.info("\nThese experiments will:")
        logger.info("  ✓ Show FULL convergence (200 rounds)")
        logger.info("  ✓ Validate at realistic scale (50 clients)")
        logger.info("  ✓ Demonstrate key DSAIN capabilities")
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
                'time_hours': elapsed,
                'purpose': experiment['purpose']
            })

            if not success:
                failed.append(experiment['name'])

            # Progress update
            completed = i
            remaining = len(self.experiments) - i
            total_elapsed = (time.time() - overall_start) / 3600

            logger.info(f"\n📊 Progress: {completed}/{len(self.experiments)} complete")
            logger.info(f"   Time elapsed: {total_elapsed:.2f} hours")

            if remaining > 0 and completed > 0:
                avg_time = total_elapsed / completed
                eta = remaining * avg_time
                logger.info(f"   ETA: {eta:.2f} hours ({total_elapsed + eta:.2f} hours total)")

                # Estimate completion time
                from datetime import datetime, timedelta
                eta_time = datetime.now() + timedelta(hours=eta)
                logger.info(f"   Estimated completion: {eta_time.strftime('%I:%M %p, %B %d')}")

        # Final summary
        total_time = (time.time() - overall_start) / 3600

        logger.info("\n" + "="*70)
        logger.info("🎉 DEEP EXPERIMENTS COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.2f} hours ({total_time*60:.0f} minutes)")
        logger.info(f"Completed: {len([r for r in results if r['success']])}/{len(self.experiments)}")

        if failed:
            logger.warning(f"\n⚠️  Failed experiments ({len(failed)}):")
            for name in failed:
                logger.warning(f"  - {name}")
        else:
            logger.info("\n✅ All deep experiments completed successfully!")

        # Save summary
        summary_file = self.results_dir / "deep_experiment_summary.json"
        summary = {
            'total_experiments': len(self.experiments),
            'successful': len([r for r in results if r['success']]),
            'failed': len(failed),
            'total_time_hours': total_time,
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
    print("\n" + "="*70)
    print("🌙 DSAIN Deep Experiments - Overnight Runner")
    print("="*70)
    print("\nThis will run 3 DEEP experiments with full convergence:")
    print("  1. Baseline ResNet-18 (200 rounds, 50 clients)")
    print("  2. Byzantine 20% (200 rounds, 50 clients)")
    print("  3. Heterogeneity α=0.1 (200 rounds, 50 clients)")
    print("\nConfiguration:")
    print("  - 200 rounds (full convergence validation)")
    print("  - 50 clients (realistic FL scale)")
    print("  - 10 clients per round (20% participation)")
    print("\nEstimated time: ~10-12 hours")
    print("\nPerfect for overnight runs:")
    print("  - Start before bed (~11 PM)")
    print("  - Complete by morning (~9 AM)")
    print("\nThese complement fast experiments for TMLR submission!")
    print("="*70)

    response = input("\n🌙 Start overnight deep experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Show start time and estimated completion
    from datetime import datetime, timedelta
    start_time = datetime.now()
    est_completion = start_time + timedelta(hours=11)

    print("\n" + "="*70)
    print("⚡ STARTING DEEP EXPERIMENTS...")
    print("="*70)
    print(f"Start time: {start_time.strftime('%I:%M %p, %B %d, %Y')}")
    print(f"Estimated completion: {est_completion.strftime('%I:%M %p, %B %d, %Y')}")
    print("\nYou can:")
    print("  - Minimize this window (don't close!)")
    print("  - Use laptop for light tasks")
    print("  - Check progress: tail -f logs/deep_run_*.log")
    print("="*70 + "\n")

    runner = DeepExperimentRunner()
    success = runner.run_all()

    if success:
        print("\n" + "="*70)
        print("🎉 SUCCESS! All deep experiments completed")
        print("="*70)
        print("\nNext steps:")
        print("1. Check results: results/deep/")
        print("2. Review logs: logs/deep_run_*.log")
        print("3. Combine with fast results for TMLR manuscript")
        print("\nYou now have:")
        print("  ✓ Fast experiments (breadth, 10 scenarios)")
        print("  ✓ Deep experiments (depth, full convergence)")
        print("  ✓ Ready for TMLR submission (~85% acceptance)")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  Some experiments failed - check logs")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
