#!/usr/bin/env python3
"""
GPU Test Script - Verify GPU is working before running full experiments
"""
import torch
import sys
import subprocess
import time

def test_pytorch_gpu():
    """Test 1: PyTorch GPU availability"""
    print("="*70)
    print("TEST 1: PyTorch GPU Detection")
    print("="*70)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("✅ PyTorch can see GPU")
        return True
    else:
        print("❌ PyTorch CANNOT see GPU")
        return False

def test_gpu_computation():
    """Test 2: Actual GPU computation"""
    print("\n" + "="*70)
    print("TEST 2: GPU Computation Test")
    print("="*70)

    if not torch.cuda.is_available():
        print("❌ Skipping - GPU not available")
        return False

    try:
        # Create tensors on GPU
        device = torch.device('cuda')
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)

        # Perform computation
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"Matrix multiplication (1000x1000): {elapsed:.4f}s")
        print(f"Result shape: {z.shape}")
        print(f"Result device: {z.device}")
        print("✅ GPU computation working")
        return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_single_experiment():
    """Test 3: Run one actual experiment"""
    print("\n" + "="*70)
    print("TEST 3: Single Experiment Test (5 rounds only)")
    print("="*70)

    cmd = [
        'python', 'enhanced_experiments.py',
        '--exp', 'baseline',
        '--model', 'mobilenetv2',  # Smaller model for faster test
        '--dataset', 'cifar10',
        '--num_rounds', '5',  # Just 5 rounds for testing
        '--num_clients', '10',  # Fewer clients
        '--participation_rate', '0.3',
        '--seed', '42',
        '--output_dir', '../results/test'
    ]

    print(f"Command: {' '.join(cmd)}")
    print("\nRunning test experiment (this will take 2-3 minutes)...")
    print("Watch for 'Using device:' in output below:\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for test
        )

        # Check output for device info
        output = result.stdout + result.stderr

        if 'Using device: cuda' in output:
            print("\n✅ Experiment used CUDA device")
            device_used = 'cuda'
        elif 'Using device: cpu' in output:
            print("\n❌ Experiment used CPU device")
            device_used = 'cpu'
        else:
            print("\n⚠️  Could not determine device from output")
            device_used = 'unknown'

        # Check if experiment completed successfully
        if result.returncode == 0:
            print("✅ Test experiment completed successfully")
            success = True
        else:
            print(f"❌ Test experiment failed with code {result.returncode}")
            print("\nError output:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            success = False

        return success and device_used == 'cuda'

    except subprocess.TimeoutExpired:
        print("❌ Test experiment timed out")
        return False
    except Exception as e:
        print(f"❌ Test experiment error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("DSAIN GPU VERIFICATION TEST SUITE")
    print("="*70)
    print("\nThis will verify:")
    print("1. PyTorch can see the GPU")
    print("2. GPU computations work")
    print("3. Experiments actually use GPU")
    print("\n" + "="*70 + "\n")

    # Run tests
    test1 = test_pytorch_gpu()
    if not test1:
        print("\n" + "="*70)
        print("FAILED: PyTorch cannot see GPU")
        print("="*70)
        sys.exit(1)

    test2 = test_gpu_computation()
    if not test2:
        print("\n" + "="*70)
        print("FAILED: GPU computation not working")
        print("="*70)
        sys.exit(1)

    test3 = test_single_experiment()

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)

    if test1 and test2 and test3:
        print("✅ ALL TESTS PASSED")
        print("\nYou can safely run the full 30-experiment suite.")
        print("GPU will be used and experiments will complete in 15-30 hours.")
        print("\n" + "="*70)
        sys.exit(0)
    else:
        print("❌ TESTS FAILED")
        print("\nDo NOT run full experiments yet.")
        print("GPU is not being used properly.")
        print("\n" + "="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()
