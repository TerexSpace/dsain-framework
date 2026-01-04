# CUDA Fix Guide - PyTorch Installation

## Problem Identified

✅ **GPU**: RTX 4060 Laptop (Detected, Working)
✅ **CUDA**: Version 13.0 (Installed)
❌ **PyTorch**: 2.9.1+cpu (CPU-only version - WRONG!)

**Issue**: You installed PyTorch without CUDA support. Experiments will run on CPU (20x slower).

---

## Solution: Install CUDA-Enabled PyTorch

### Quick Fix (Windows - Easiest)

**Double-click**: `fix_cuda.bat`

This will:
1. Uninstall CPU-only PyTorch
2. Install PyTorch with CUDA 12.4 support (compatible with CUDA 13.0)
3. Verify installation

**Time**: 2-3 minutes (downloads ~2GB)

---

### Manual Fix (All Platforms)

**Step 1: Uninstall CPU-only PyTorch**
```bash
pip uninstall -y torch torchvision torchaudio
```

**Step 2: Install CUDA-enabled PyTorch**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Step 3: Verify CUDA is working**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Why CUDA 12.4 for CUDA 13.0?

PyTorch doesn't have official builds for CUDA 13.0 yet. But CUDA 12.4 builds are **backward compatible** with CUDA 13.0 drivers.

**This is the official PyTorch recommendation.**

---

## After Fix - Verify System

Run the system check again:
```bash
cd code
python check_system.py
```

Expected output:
```
✓ Python 3.x
✓ CUDA 12.4
✓ GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.0 GB)
✓ PyTorch
✓ TorchVision
✓ NumPy
✓ SciPy
✓ tqdm
🎉 System is READY for experiments!
Estimated experiment time: 55-65 minutes
```

---

## Troubleshooting

### Problem: "torch" not found after installation

**Solution**: Restart terminal/command prompt
```bash
# Close current terminal
# Open new terminal
python -c "import torch; print(torch.cuda.is_available())"
```

---

### Problem: Still shows CPU-only version

**Cause**: Old cached version

**Solution**: Force reinstall
```bash
pip cache purge
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

---

### Problem: "Could not find a version that satisfies the requirement"

**Cause**: Python version too old or too new

**Solution**: Check Python version
```bash
python --version
```

Required: **Python 3.8-3.12**

If outside range:
- Install compatible Python version
- Create virtual environment with compatible version

---

### Problem: Download fails or times out

**Cause**: Network issues

**Solution 1**: Retry with increased timeout
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --timeout 300
```

**Solution 2**: Use conda (if you have it)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

---

## Alternative: Use Google Colab Instead

If installation continues to fail, you can skip local GPU and use **FREE Google Colab**:

1. Upload: `DSAIN_Experiments_Colab.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run experiments on Google's Tesla T4 GPU (FREE)
4. Zero installation issues!

See: [DSAIN_Experiments_Colab.ipynb](DSAIN_Experiments_Colab.ipynb)

---

## Verification Checklist

After fix, verify:

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] `python -c "import torch; print(torch.cuda.get_device_name(0))"` prints `NVIDIA GeForce RTX 4060 Laptop GPU`
- [ ] `python check_system.py` shows all green checkmarks
- [ ] No CUDA errors when running quick test

---

## Quick Test After Fix

Run a quick 30-second test to confirm GPU is working:

```bash
cd code
python -c "
import torch
print('Testing GPU...')
x = torch.rand(1000, 1000).cuda()
y = torch.rand(1000, 1000).cuda()
z = torch.matmul(x, y)
print(f'✅ GPU working! Result shape: {z.shape}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected output:
```
Testing GPU...
✅ GPU working! Result shape: torch.Size([1000, 1000])
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Ready to Run Experiments?

After fixing CUDA, proceed with:

```bash
cd code
python check_system.py          # Should now show CUDA available
python fast_experiments_1hour.py # Start experiments
```

---

## Summary

**Problem**: PyTorch CPU-only version installed
**Fix**: Install CUDA-enabled PyTorch
**Command**: Run `fix_cuda.bat` or manual commands above
**Time**: 2-3 minutes
**Result**: GPU experiments 20x faster!

---

## Next Steps

1. **Run**: `fix_cuda.bat` (or manual commands)
2. **Verify**: `python check_system.py`
3. **Start**: `python fast_experiments_1hour.py`

You're one command away from GPU-accelerated experiments! 🚀
