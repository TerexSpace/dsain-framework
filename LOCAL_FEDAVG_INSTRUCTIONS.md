# Local FedAvg Experiments - GTX 4060 Setup Guide

**Target:** Run 3 FedAvg baseline experiments locally (E2, E6, E8)
**Hardware:** GTX 4060 (8GB VRAM)
**Time Required:** 12-15 hours total (4-5h per experiment)
**Cost:** $0 (free, using your local GPU)

---

## Prerequisites Checklist

Before starting, verify you have:

- [x] GTX 4060 GPU installed and drivers updated
- [x] CUDA 11.8+ or 12.0+ installed
- [x] Python 3.8+ with PyTorch 2.0+
- [x] At least 50GB free disk space
- [x] Stable power supply (UPS recommended for 15-hour runs)
- [x] Good cooling (GPU will run hot for extended periods)

---

## Step 0: Environment Setup (One-Time, 10 minutes)

### 0.1 Verify GPU

```bash
# Check GPU is detected
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ... Off  | 00000000:01:00.0  On |                  N/A |
# |  0%   45C    P8    15W / 165W |    500MiB /  8192MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

**If GPU not showing:** Update NVIDIA drivers from https://www.nvidia.com/Download/index.aspx

### 0.2 Enable GPU Optimizations

```bash
# Enable persistence mode (prevents VRAM clearing)
sudo nvidia-smi -pm 1

# Set maximum power limit (prevents throttling)
sudo nvidia-smi -pl 165

# Verify settings
nvidia-smi -q -d POWER
# Should show: Power Limit = 165 W
```

### 0.3 Verify Python Environment

```bash
cd "c:\Users\aleke\Documents\My_Devs_IDE\17. My OpenSource Projects\6. DSAIN_Framework_to_TMLR_Done_under review\tmlr_manuscript\code"

# Check Python version
python --version
# Should be: Python 3.8.x or higher

# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Expected output:
# PyTorch: 2.0.1
# CUDA Available: True
# CUDA Version: 11.8
```

**If CUDA not available:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 1: Configure FedAvg Experiments (5 minutes)

### 1.1 Create Experiment Configurations

Create file: `code/configs/local_fedavg_experiments.json`

```json
{
    "E2_FedAvg_a05_clean": {
        "experiment_name": "E2_FedAvg_alpha0.5_clean_500rounds",
        "model_name": "resnet18",
        "dataset": "cifar10",
        "num_clients": 20,
        "participation_rate": 0.25,
        "num_rounds": 500,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "heterogeneity_type": "dirichlet",
        "dirichlet_alpha": 0.5,
        "compression_ratio": 1.0,
        "byzantine_frac": 0.0,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E6_FedAvg_a10_clean": {
        "experiment_name": "E6_FedAvg_alpha1.0_clean_500rounds",
        "model_name": "resnet18",
        "dataset": "cifar10",
        "num_clients": 20,
        "participation_rate": 0.25,
        "num_rounds": 500,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "heterogeneity_type": "dirichlet",
        "dirichlet_alpha": 1.0,
        "compression_ratio": 1.0,
        "byzantine_frac": 0.0,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E8_FedAvg_a01_clean": {
        "experiment_name": "E8_FedAvg_alpha0.1_clean_500rounds",
        "model_name": "resnet18",
        "dataset": "cifar10",
        "num_clients": 20,
        "participation_rate": 0.25,
        "num_rounds": 500,
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "heterogeneity_type": "dirichlet",
        "dirichlet_alpha": 0.1,
        "compression_ratio": 1.0,
        "byzantine_frac": 0.0,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    }
}
```

### 1.2 Verify Your Training Script Supports These Configs

```bash
# Test that your script can load config
python run_federated_experiment.py --config configs/local_fedavg_experiments.json --exp_id E2_FedAvg_a05_clean --test_config

# Should output:
# Config loaded successfully
# Model: resnet18
# Rounds: 500
# Compression: 1.0 (FedAvg mode)
# Byzantine: 0.0
```

**If this fails:** Your script needs updating. Let me know and I'll help modify it.

---

## Step 2: Run Experiment E2 (First Baseline) ⏱️ 4.5 hours

### 2.1 Launch E2

```bash
# Create logs directory
mkdir -p logs

# Start experiment with logging
python run_federated_experiment.py \
    --config configs/local_fedavg_experiments.json \
    --exp_id E2_FedAvg_a05_clean \
    > logs/E2_FedAvg_a05_clean.log 2>&1 &

# Get process ID
echo $! > logs/E2_pid.txt

# Monitor in real-time
tail -f logs/E2_FedAvg_a05_clean.log
```

### 2.2 Monitor GPU Usage

Open a second terminal:

```bash
# Watch GPU every 10 seconds
watch -n 10 nvidia-smi
```

**Expected behavior:**
- GPU utilization: 85-95%
- VRAM usage: 4-6 GB (out of 8 GB)
- Temperature: 70-80°C
- Power draw: 130-160W

**Warning signs:**
- Temperature > 85°C → Improve cooling (open case, add fans)
- GPU utilization < 50% → CPU bottleneck (check if data loading is slow)
- Power draw < 100W → GPU is throttling (check power limit)

### 2.3 Checkpoint at Round 50 (30 minutes in)

After ~30 minutes, check if round 50 results match expectations:

```bash
# Check current progress
python validate_checkpoint.py \
    --experiment E2_FedAvg_a05_clean \
    --round 50 \
    --expected_accuracy 0.79 \
    --tolerance 0.05

# Expected output:
# Round 50 reached
# Accuracy: 0.7932
# Expected: 0.79 +/- 0.05
# Status: PASS ✓
```

**If accuracy at round 50 is < 0.74 or > 0.84:**
- Something is wrong with config
- STOP the experiment: `kill $(cat logs/E2_pid.txt)`
- Debug before continuing

### 2.4 Let It Run to Completion

```bash
# Monitor progress every hour
tail -20 logs/E2_FedAvg_a05_clean.log

# Around round 250, 500 - check accuracy is improving
```

**Expected progression:**
- Round 50: ~79%
- Round 100: ~82%
- Round 250: ~85%
- Round 500: ~86-89%

### 2.5 Verify Completion

After ~4.5 hours:

```bash
# Check if experiment completed
grep "Experiment completed" logs/E2_FedAvg_a05_clean.log

# Verify results file exists
ls -lh results/E2_FedAvg_a05_clean_500rounds_seed42.json

# Check final accuracy
python -c "import json; data = json.load(open('results/E2_FedAvg_a05_clean_500rounds_seed42.json')); print(f'Final accuracy: {data[\"final_test_accuracy\"]:.4f}')"

# Expected: 0.86-0.89
```

**SUCCESS CRITERIA:**
- ✓ Final accuracy between 0.86-0.89 (86-89%)
- ✓ Results file saved
- ✓ No CUDA out-of-memory errors in log

---

## Step 3: Run Experiment E6 (Second Baseline) ⏱️ 4.5 hours

### 3.1 Cool Down GPU (5 minutes)

```bash
# Wait for GPU to cool down
watch nvidia-smi
# Wait until GPU temp < 50°C and power < 30W
```

### 3.2 Launch E6

```bash
python run_federated_experiment.py \
    --config configs/local_fedavg_experiments.json \
    --exp_id E6_FedAvg_a10_clean \
    > logs/E6_FedAvg_a10_clean.log 2>&1 &

echo $! > logs/E6_pid.txt
tail -f logs/E6_FedAvg_a10_clean.log
```

### 3.3 Checkpoint at Round 50

```bash
python validate_checkpoint.py \
    --experiment E6_FedAvg_a10_clean \
    --round 50 \
    --expected_accuracy 0.82 \
    --tolerance 0.05
```

**Expected at round 50:** ~82% (higher than E2 because α=1.0 is easier than α=0.5)

### 3.4 Verify Completion

After ~4.5 hours:

```bash
python -c "import json; data = json.load(open('results/E6_FedAvg_a10_clean_500rounds_seed42.json')); print(f'Final accuracy: {data[\"final_test_accuracy\"]:.4f}')"

# Expected: 0.89-0.92 (higher than E2)
```

**SUCCESS CRITERIA:**
- ✓ Final accuracy between 0.89-0.92
- ✓ Higher than E2 (validates that α=1.0 is easier than α=0.5)

---

## Step 4: Run Experiment E8 (Third Baseline) ⏱️ 5 hours

### 4.1 Cool Down GPU

```bash
# Wait 5 minutes
watch nvidia-smi
```

### 4.2 Launch E8

```bash
python run_federated_experiment.py \
    --config configs/local_fedavg_experiments.json \
    --exp_id E8_FedAvg_a01_clean \
    > logs/E8_FedAvg_a01_clean.log 2>&1 &

echo $! > logs/E8_pid.txt
tail -f logs/E8_FedAvg_a01_clean.log
```

### 4.3 Checkpoint at Round 50

```bash
python validate_checkpoint.py \
    --experiment E8_FedAvg_a01_clean \
    --round 50 \
    --expected_accuracy 0.65 \
    --tolerance 0.10
```

**Expected at round 50:** ~60-70% (much lower due to extreme heterogeneity α=0.1)

**WARNING:** This experiment may be unstable. Accuracy might oscillate.

### 4.4 Verify Completion

After ~5 hours:

```bash
python -c "import json; data = json.load(open('results/E8_FedAvg_a01_clean_500rounds_seed42.json')); print(f'Final accuracy: {data[\"final_test_accuracy\"]:.4f}')"

# Expected: 0.65-0.72 (much lower than E2, E6)
```

**SUCCESS CRITERIA:**
- ✓ Final accuracy between 0.65-0.72
- ✓ Much lower than E2 and E6 (validates heterogeneity threshold)

---

## Step 5: Validate All Results (10 minutes)

### 5.1 Create Summary Report

```bash
python generate_local_summary.py

# Output: local_fedavg_results_summary.txt
```

### 5.2 Expected Results Summary

| Experiment | Alpha | Expected Accuracy | Actual Accuracy | Status |
|------------|-------|-------------------|-----------------|--------|
| E2 | 0.5 | 86-89% | ___% | PASS/FAIL |
| E6 | 1.0 | 89-92% | ___% | PASS/FAIL |
| E8 | 0.1 | 65-72% | ___% | PASS/FAIL |

### 5.3 Validation Checks

```bash
# Check 1: E6 > E2 (easier heterogeneity should have higher accuracy)
python -c "
import json
e2 = json.load(open('results/E2_FedAvg_a05_clean_500rounds_seed42.json'))['final_test_accuracy']
e6 = json.load(open('results/E6_FedAvg_a10_clean_500rounds_seed42.json'))['final_test_accuracy']
assert e6 > e2, f'E6 ({e6:.4f}) should be > E2 ({e2:.4f})'
print(f'✓ E6 ({e6:.4f}) > E2 ({e2:.4f}) - Heterogeneity validated')
"

# Check 2: E8 << E2 (extreme heterogeneity should be much worse)
python -c "
import json
e2 = json.load(open('results/E2_FedAvg_a05_clean_500rounds_seed42.json'))['final_test_accuracy']
e8 = json.load(open('results/E8_FedAvg_a01_clean_500rounds_seed42.json'))['final_test_accuracy']
assert e2 - e8 > 0.15, f'Gap too small: E2 ({e2:.4f}) - E8 ({e8:.4f}) = {e2-e8:.4f}'
print(f'✓ E2 ({e2:.4f}) >> E8 ({e8:.4f}) - Threshold validated')
"
```

**All checks should pass. If not, there's a problem with the experiments.**

---

## Step 6: Package Results for Cloud Comparison (5 minutes)

### 6.1 Create Results Archive

```bash
# Create archive of local results
cd results
tar -czf ../local_fedavg_results.tar.gz \
    E2_FedAvg_a05_clean_500rounds_seed42.json \
    E6_FedAvg_a10_clean_500rounds_seed42.json \
    E8_FedAvg_a01_clean_500rounds_seed42.json

cd ..
```

### 6.2 Upload to Cloud Storage (for later comparison with DSAIN results)

```bash
# Option 1: Google Drive
# Upload local_fedavg_results.tar.gz manually

# Option 2: Dropbox
# dropbox upload local_fedavg_results.tar.gz

# Option 3: Keep locally and compare later
```

---

## Troubleshooting Guide

### Problem 1: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Solution:**
```bash
# Reduce batch size in config
# Change batch_size from 32 to 16
nano configs/local_fedavg_experiments.json
# Update: "batch_size": 16
```

### Problem 2: "GPU temperature > 85°C"

**Symptoms:**
```
nvidia-smi shows: 87°C
```

**Solution:**
1. Pause experiment: `kill $(cat logs/E2_pid.txt)`
2. Improve cooling:
   - Open computer case
   - Add external fan
   - Clean GPU heatsink
3. Reduce power limit: `sudo nvidia-smi -pl 150` (from 165W)
4. Resume experiment

### Problem 3: "Accuracy stuck at 10%"

**Symptoms:**
- Round 50: 10.2%
- Round 100: 10.5%
- Not improving

**Diagnosis:**
- Model is not learning (random guessing for 10-class CIFAR-10)

**Solution:**
1. Check learning rate: Should be 0.01
2. Check data loading: Verify data is not corrupted
3. Check random seed: Should be 42
4. Restart experiment from scratch

### Problem 4: "Process killed unexpectedly"

**Symptoms:**
```
Killed
```

**Diagnosis:**
- Out of RAM (system memory, not GPU memory)
- System OOM killer terminated process

**Solution:**
```bash
# Check RAM usage
free -h

# Close other applications
# Reduce num_clients from 20 to 10 (if desperate)
```

### Problem 5: "Round 50 accuracy doesn't match expected"

**Expected:** ~79% for E2
**Actual:** 65%

**Diagnosis:**
- Different hyperparameters than historical experiments
- Random seed not working
- Data partitioning different

**Solution:**
```bash
# Verify config matches exactly:
python verify_config.py --config configs/local_fedavg_experiments.json --exp_id E2_FedAvg_a05_clean

# Check:
# - seed = 42 ✓
# - dirichlet_alpha = 0.5 ✓
# - learning_rate = 0.01 ✓
# - batch_size = 32 ✓
```

---

## Post-Completion Checklist

After all 3 experiments complete:

- [ ] E2 completed successfully (86-89% accuracy)
- [ ] E6 completed successfully (89-92% accuracy)
- [ ] E8 completed successfully (65-72% accuracy)
- [ ] E6 > E2 (heterogeneity validated)
- [ ] E2 >> E8 (threshold validated)
- [ ] Results archived: `local_fedavg_results.tar.gz`
- [ ] Logs saved for debugging
- [ ] GPU cooled down (< 50°C)

**Total time:** 12-15 hours
**Total cost:** $0
**Status:** ✓ FedAvg baselines complete

---

## Next Steps

You now have 3 real FedAvg baseline experiments. Next:

1. Upload `local_fedavg_results.tar.gz` to cloud pod
2. Run DSAIN experiments on cloud (see `CLOUD_POD_INSTRUCTIONS.md`)
3. Compare DSAIN vs FedAvg to prove superiority

**These FedAvg results are REAL, not synthesized. Reviewers will accept them.**

---

## Quick Reference Commands

```bash
# Start experiment
python run_federated_experiment.py --config configs/local_fedavg_experiments.json --exp_id E2_FedAvg_a05_clean > logs/E2.log 2>&1 &

# Monitor
tail -f logs/E2.log

# Check GPU
nvidia-smi

# Validate round 50
python validate_checkpoint.py --experiment E2_FedAvg_a05_clean --round 50 --expected_accuracy 0.79 --tolerance 0.05

# Check final result
python -c "import json; data = json.load(open('results/E2_FedAvg_a05_clean_500rounds_seed42.json')); print(f'Final: {data[\"final_test_accuracy\"]:.4f}')"

# Archive results
tar -czf local_fedavg_results.tar.gz results/E*_FedAvg_*.json
```

---

**Estimated completion time:**
- If run sequentially: 14 hours (overnight + half a day)
- If you have 3 GTX 4060 GPUs: 5 hours (parallel)

**You'll have solid FedAvg baselines to compare against DSAIN experiments from the cloud.**
