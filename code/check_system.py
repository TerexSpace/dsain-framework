#!/usr/bin/env python3
"""
System Readiness Check for DSAIN Fast Experiments
==================================================

Verifies that your system meets all requirements for running
the 1-hour fast experiment suite.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.8"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ ERROR: Python 3.8+ required")
        return False
    return True


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ CUDA {torch.version.cuda}")
            print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            if gpu_memory < 6.0:
                print("  ⚠️  WARNING: GPU has <6GB VRAM, experiments may fail")
                print("     Consider reducing batch size or using Google Colab")

            return True
        else:
            print("❌ CUDA not available")
            print("   Experiments will be VERY slow on CPU (~20+ hours)")
            print("   Recommend using Google Colab instead")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_dependencies():
    """Check required packages"""
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm'
    }

    all_installed = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} not installed")
            all_installed = False

    return all_installed


def check_disk_space():
    """Check available disk space"""
    import shutil

    # Check current directory
    total, used, free = shutil.disk_usage(Path.cwd())
    free_gb = free / (1024**3)

    print(f"✓ Disk space: {free_gb:.1f} GB free")

    if free_gb < 2.0:
        print("  ⚠️  WARNING: Less than 2GB free")
        print("     Experiments need ~500MB, but recommend 2GB buffer")
        return False

    return True


def check_memory():
    """Check system RAM"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        print(f"✓ RAM: {total_gb:.1f} GB total, {available_gb:.1f} GB available")

        if total_gb < 8.0:
            print("  ⚠️  WARNING: Less than 8GB RAM")
            print("     Experiments may be slow or fail")
            return False

        if available_gb < 4.0:
            print("  ⚠️  WARNING: Less than 4GB RAM available")
            print("     Close other applications before running experiments")

        return True

    except ImportError:
        print("⚠️  psutil not installed, skipping RAM check")
        return True


def check_code_files():
    """Check required code files exist"""
    required_files = [
        'enhanced_experiments.py',
        'modern_architectures.py',
        'data_heterogeneity.py',
        'fast_experiments_1hour.py'
    ]

    all_exist = True
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            print(f"✓ {filename}")
        else:
            print(f"❌ {filename} not found")
            all_exist = False

    return all_exist


def estimate_time():
    """Estimate experiment time based on GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()

            # Time estimates based on GPU
            if 'rtx 4090' in gpu_name or 'rtx 4080' in gpu_name:
                return 40, 50
            elif 'rtx 4060' in gpu_name or 'rtx 3070' in gpu_name:
                return 55, 65
            elif 'rtx 3060' in gpu_name or 'rtx 2080' in gpu_name:
                return 65, 75
            elif 'gtx 1660' in gpu_name or 'rtx 2060' in gpu_name:
                return 80, 95
            else:
                return 60, 90  # Conservative estimate
        else:
            return 1200, 1800  # 20-30 hours on CPU
    except:
        return 60, 90


def main():
    """Run all system checks"""
    print("=" * 70)
    print("DSAIN Fast Experiments - System Readiness Check")
    print("=" * 70)
    print()

    checks = {
        'Python Version': check_python_version(),
        'CUDA & GPU': check_cuda(),
        'Dependencies': check_dependencies(),
        'Code Files': check_code_files(),
        'Disk Space': check_disk_space(),
        'System Memory': check_memory()
    }

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(checks.values())
    total = len(checks)

    for name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")

    print()
    print(f"Overall: {passed}/{total} checks passed")

    if passed == total:
        print()
        print("🎉 System is READY for experiments!")
        print()

        # Estimate time
        min_time, max_time = estimate_time()
        print(f"Estimated experiment time: {min_time}-{max_time} minutes")
        print()
        print("To start experiments:")
        print("  python fast_experiments_1hour.py")
        print()
        return 0

    elif checks['CUDA & GPU'] and checks['Dependencies']:
        print()
        print("⚠️  System has warnings but can run experiments")
        print()
        print("To start experiments (at your own risk):")
        print("  python fast_experiments_1hour.py")
        print()
        return 0

    else:
        print()
        print("❌ System NOT ready for experiments")
        print()

        if not checks['Dependencies']:
            print("Install missing dependencies:")
            print("  pip install torch torchvision tqdm numpy scipy")
            print()

        if not checks['CUDA & GPU']:
            print("CUDA not available. Options:")
            print("  1. Install CUDA drivers")
            print("  2. Use Google Colab (FREE GPU)")
            print("     See: DSAIN_Experiments_Colab.ipynb")
            print()

        return 1


if __name__ == "__main__":
    sys.exit(main())
