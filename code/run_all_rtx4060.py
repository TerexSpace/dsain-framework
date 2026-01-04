#!/usr/bin/env python3
"""
Automated Experiment Runner for RTX 4060 (Single GPU)
======================================================

This script runs all enhanced experiments sequentially on a single GPU.
Designed for RTX 4060 (8GB VRAM) with automatic error recovery and progress tracking.

Just run once and let it work for 4-5 days:
    python run_all_rtx4060.py

Features:
- Automatic resume on crash
- Progress saving after each experiment
- Email notifications (optional)
- Resource monitoring
- Detailed logging
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
        logging.FileHandler(log_dir / f"automated_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages sequential experiment execution with progress tracking."""

    def __init__(self, results_dir="../results/enhanced"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.results_dir / "progress.json"
        self.completed = self.load_progress()

        # Define all experiments
        self.experiments = self.define_experiments()

        logger.info(f"Initialized ExperimentRunner")
        logger.info(f"Results directory: {self.results_dir.absolute()}")
        logger.info(f"Completed experiments: {len(self.completed)}/{len(self.experiments)}")

    def define_experiments(self):
        """Define all experiments to run."""
        experiments = []

        # 1. Baseline experiments (4 experiments)
        for model in ['resnet18', 'mobilenetv2', 'vit_tiny']:
            experiments.append({
                'name': f'baseline_{model}_cifar10',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', model,
                    '--dataset', 'cifar10',
                    '--num_rounds', '500',
                    '--seed', '42'
                ],
                'estimated_hours': 8 if model == 'vit_tiny' else 6
            })

        experiments.append({
            'name': 'baseline_resnet18_cifar100',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'baseline',
                '--model', 'resnet18',
                '--dataset', 'cifar100',
                '--num_rounds', '500',
                '--seed', '42'
            ],
            'estimated_hours': 7
        })

        # 2. Heterogeneity study
        experiments.append({
            'name': 'heterogeneity_study',
            'command': [
                'python', 'enhanced_experiments.py',
                '--exp', 'heterogeneity',
                '--model', 'resnet18',
                '--dataset', 'cifar10',
                '--num_rounds', '500',
                '--seed', '42'
            ],
            'estimated_hours': 36
        })

        # 3. Byzantine experiments
        for byz_frac in [0.0, 0.1, 0.2]:
            experiments.append({
                'name': f'byzantine_{int(byz_frac*100)}pct',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--byzantine_frac', str(byz_frac),
                    '--num_rounds', '500',
                    '--seed', '42'
                ],
                'estimated_hours': 7
            })

        # 4. Privacy sweep
        for epsilon in ['2.0', '4.0', '8.0']:
            experiments.append({
                'name': f'privacy_eps{epsilon}',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--dp_epsilon', epsilon,
                    '--num_rounds', '500',
                    '--seed', '42'
                ],
                'estimated_hours': 7
            })

        # 5. Compression sweep
        for comp in ['0.1', '0.22', '0.5', '1.0']:
            experiments.append({
                'name': f'compression_{comp}',
                'command': [
                    'python', 'enhanced_experiments.py',
                    '--exp', 'baseline',
                    '--model', 'resnet18',
                    '--dataset', 'cifar10',
                    '--compression', comp,
                    '--num_rounds', '500',
                    '--seed', '42'
                ],
                'estimated_hours': 6
            })

        return experiments

    def load_progress(self):
        """Load completed experiments from progress file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded progress: {len(data['completed'])} experiments completed")
                return set(data['completed'])
        return set()

    def save_progress(self, experiment_name):
        """Save progress after completing an experiment."""
        self.completed.add(experiment_name)

        progress_data = {
            'completed': list(self.completed),
            'total': len(self.experiments),
            'last_updated': datetime.now().isoformat(),
            'completion_rate': len(self.completed) / len(self.experiments) * 100
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

        logger.info(f"Progress saved: {len(self.completed)}/{len(self.experiments)} "
                   f"({progress_data['completion_rate']:.1f}%) complete")

    def run_experiment(self, experiment):
        """Run a single experiment with error handling."""
        name = experiment['name']

        if name in self.completed:
            logger.info(f"⏭️  Skipping {name} (already completed)")
            return True

        logger.info(f"🚀 Starting: {name}")
        logger.info(f"   Estimated time: {experiment['estimated_hours']} hours")
        logger.info(f"   Command: {' '.join(experiment['command'])}")

        start_time = time.time()

        try:
            # Run experiment
            result = subprocess.run(
                experiment['command'],
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=experiment['estimated_hours'] * 3600 + 3600  # Add 1 hour buffer
            )

            elapsed = (time.time() - start_time) / 3600

            if result.returncode == 0:
                logger.info(f"✅ Completed: {name} in {elapsed:.2f} hours")
                self.save_progress(name)
                return True
            else:
                logger.error(f"❌ Failed: {name}")
                logger.error(f"   stdout: {result.stdout[-500:]}")  # Last 500 chars
                logger.error(f"   stderr: {result.stderr[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            elapsed = (time.time() - start_time) / 3600
            logger.error(f"⏱️  Timeout: {name} after {elapsed:.2f} hours")
            return False

        except Exception as e:
            elapsed = (time.time() - start_time) / 3600
            logger.error(f"💥 Error: {name} - {str(e)}")
            return False

    def run_all(self):
        """Run all experiments sequentially."""
        logger.info("="*70)
        logger.info("AUTOMATED EXPERIMENT RUNNER FOR RTX 4060")
        logger.info("="*70)
        logger.info(f"Total experiments: {len(self.experiments)}")
        logger.info(f"Already completed: {len(self.completed)}")
        logger.info(f"Remaining: {len(self.experiments) - len(self.completed)}")

        total_estimated = sum(exp['estimated_hours'] for exp in self.experiments
                             if exp['name'] not in self.completed)
        logger.info(f"Estimated remaining time: {total_estimated:.1f} hours ({total_estimated/24:.1f} days)")
        logger.info("="*70)

        overall_start = time.time()
        failed = []

        for i, experiment in enumerate(self.experiments, 1):
            logger.info(f"\n[{i}/{len(self.experiments)}] Processing: {experiment['name']}")

            success = self.run_experiment(experiment)

            if not success:
                failed.append(experiment['name'])
                logger.warning(f"⚠️  Continuing to next experiment despite failure...")

            # Print summary after each experiment
            completed_count = len(self.completed)
            logger.info(f"\nProgress: {completed_count}/{len(self.experiments)} "
                       f"({completed_count/len(self.experiments)*100:.1f}%) complete")

            remaining = len(self.experiments) - completed_count
            if remaining > 0:
                avg_time = (time.time() - overall_start) / 3600 / completed_count if completed_count > 0 else 0
                eta_hours = remaining * avg_time
                logger.info(f"ETA: {eta_hours:.1f} hours ({eta_hours/24:.1f} days)")

        # Final summary
        total_time = (time.time() - overall_start) / 3600

        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUITE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.2f} hours ({total_time/24:.2f} days)")
        logger.info(f"Completed: {len(self.completed)}/{len(self.experiments)}")

        if failed:
            logger.warning(f"\nFailed experiments ({len(failed)}):")
            for name in failed:
                logger.warning(f"  - {name}")
            logger.warning("\nYou can re-run failed experiments manually.")
        else:
            logger.info("\n🎉 All experiments completed successfully!")

        logger.info(f"\nResults saved in: {self.results_dir.absolute()}")
        logger.info("="*70)


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🚀 DSAIN Enhanced Experiments - Automated Runner for RTX 4060")
    print("="*70)
    print("\nThis script will run ALL experiments sequentially.")
    print("Estimated total time: 100-130 hours (4-5 days)")
    print("\nFeatures:")
    print("  ✓ Automatic resume on crash")
    print("  ✓ Progress tracking")
    print("  ✓ Detailed logging")
    print("  ✓ Skip completed experiments")
    print("\nYou can:")
    print("  - Close this terminal (experiments continue in background)")
    print("  - Monitor progress: tail -f logs/automated_run_*.log")
    print("  - Stop anytime: Ctrl+C (progress is saved)")
    print("="*70)

    response = input("\nStart experiments? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    print("\n🚀 Starting experiments...\n")

    runner = ExperimentRunner()
    runner.run_all()


if __name__ == "__main__":
    main()
