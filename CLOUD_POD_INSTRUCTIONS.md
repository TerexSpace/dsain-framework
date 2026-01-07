# Cloud Pod DSAIN Experiments - Setup Guide

**Target:** Run 7 DSAIN + FedAvg Byzantine experiments on cloud GPU
**Hardware:** V100 or A100 GPU (16GB+ VRAM)
**Time Required:** 18-24 hours compute (can run in parallel with multiple pods)
**Cost:** $9-18 (Lambda Labs) or $15-27 (RunPod)

**Experiments to run:**
- E1: DSAIN α=0.5, clean (your main result)
- E5: DSAIN α=1.0, clean (heterogeneity validation)
- E7: DSAIN α=0.1, clean (high heterogeneity)
- E9: DSAIN α=0.5, ε=2.0 (privacy sweet spot)
- E3: DSAIN α=0.5, 20% Byzantine (resilience test)
- E10: DSAIN α=0.5, 10% Byzantine (graceful degradation)
- E4: FedAvg α=0.5, 20% Byzantine (catastrophic failure proof)

---

## Part 1: Cloud Provider Setup

### Option A: Lambda Labs (Recommended - Cheapest)

**Pricing:** $0.50/hour for V100 (16GB)

#### A.1 Create Account

1. Go to https://lambdalabs.com/service/gpu-cloud
2. Sign up (requires credit card)
3. Add $20 credit (minimum)

#### A.2 Launch Instance

```bash
# Web UI method:
1. Click "Launch Instance"
2. Select: "1x V100 (16 GB)"
3. Region: Any with availability (usually us-west or us-east)
4. Instance type: "On-demand"
5. SSH key: Upload your public key or create new
6. Launch

# Should show:
# Instance: running
# IP: 123.456.78.90
# Cost: $0.50/hour
```

#### A.3 Connect via SSH

```bash
# From your local machine
ssh ubuntu@123.456.78.90

# First login, you'll see:
# Welcome to Lambda GPU Cloud
# GPU: Tesla V100-SXM2-16GB
```

---

### Option B: RunPod (Alternative - More GPUs Available)

**Pricing:** $0.50-0.90/hour for V100/A100

#### B.1 Create Account

1. Go to https://www.runpod.io
2. Sign up
3. Add $20 credit

#### B.2 Launch Pod

```bash
# Web UI:
1. Click "Rent"
2. Filter: GPU = V100 or A100
3. Sort by: $/hr (cheapest first)
4. Select pod with 16GB+ VRAM
5. Template: PyTorch 2.0
6. Deploy

# Connect via Web Terminal or SSH
```

---

## Part 2: Initial Setup (One-Time, 15 minutes)

### 2.1 Connect to Pod

```bash
# Lambda Labs
ssh ubuntu@<IP_ADDRESS>

# RunPod (if using SSH)
ssh root@<POD_IP> -p <POD_PORT>
```

### 2.2 Verify GPU

```bash
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx       Driver Version: 525.xx       CUDA Version: 12.0    |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
# | N/A   35C    P0    40W / 300W |      0MiB / 16384MiB |      0%      Default |
# +-----------------------------------------------------------------------------+
```

**If GPU not showing:** Contact support immediately (billing issue).

### 2.3 Install Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get install -y git wget tmux htop

# Check Python + PyTorch
python --version  # Should be 3.8+
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Expected:
# 2.0.1+cu118
# True
```

### 2.4 Upload Your Code

**Method 1: Git (Recommended)**

```bash
# If your code is on GitHub
git clone https://github.com/YourUsername/dsain-framework.git
cd dsain-framework/code
```

**Method 2: SCP Upload (If code is local)**

```bash
# From your local machine (new terminal)
cd "c:\Users\aleke\Documents\My_Devs_IDE\17. My OpenSource Projects\6. DSAIN_Framework_to_TMLR_Done_under review\tmlr_manuscript"

# Compress code
tar -czf code.tar.gz code/

# Upload to pod
scp code.tar.gz ubuntu@<POD_IP>:~/

# Back in pod SSH session
tar -xzf code.tar.gz
cd code/
```

### 2.5 Install Python Requirements

```bash
# If you have requirements.txt
pip install -r requirements.txt

# Or manually
pip install torch torchvision numpy scipy matplotlib tqdm

# Verify
python -c "import torch, torchvision, numpy, scipy; print('All imports OK')"
```

---

## Part 3: Configure DSAIN Experiments

### 3.1 Create Cloud Experiment Configs

Create file: `configs/cloud_dsain_experiments.json`

```json
{
    "E1_DSAIN_a05_clean": {
        "experiment_name": "E1_DSAIN_alpha0.5_clean_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.0,
        "byzantine_defense": true,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E5_DSAIN_a10_clean": {
        "experiment_name": "E5_DSAIN_alpha1.0_clean_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.0,
        "byzantine_defense": true,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E7_DSAIN_a01_clean": {
        "experiment_name": "E7_DSAIN_alpha0.1_clean_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.0,
        "byzantine_defense": true,
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E9_DSAIN_a05_eps2": {
        "experiment_name": "E9_DSAIN_alpha0.5_epsilon2.0_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.0,
        "byzantine_defense": true,
        "dp_epsilon": 2.0,
        "dp_delta": 1e-5,
        "gradient_clip": 1.0,
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E3_DSAIN_a05_byz20": {
        "experiment_name": "E3_DSAIN_alpha0.5_byzantine20_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.20,
        "byzantine_defense": true,
        "byzantine_attack_type": "label_flipping",
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E10_DSAIN_a05_byz10": {
        "experiment_name": "E10_DSAIN_alpha0.5_byzantine10_500rounds",
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
        "compression_ratio": 0.22,
        "byzantine_frac": 0.10,
        "byzantine_defense": true,
        "byzantine_attack_type": "label_flipping",
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    },
    "E4_FedAvg_a05_byz20": {
        "experiment_name": "E4_FedAvg_alpha0.5_byzantine20_500rounds",
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
        "byzantine_frac": 0.20,
        "byzantine_defense": false,
        "byzantine_attack_type": "label_flipping",
        "dp_epsilon": "inf",
        "seed": 42,
        "eval_every": 25,
        "save_checkpoints": false
    }
}
```

---

## Part 4: Batch Execution Strategy

### Option A: Sequential Execution (Single Pod, $9-15 total)

Run all 7 experiments one after another on a single V100 pod.

**Time:** 18-24 hours continuous
**Cost:** 24h × $0.50 = $12

### Option B: Parallel Execution (2 Pods, $9-12 total)

Launch 2 V100 pods and split experiments.

**Pod 1:** E1, E5, E7, E9 (DSAIN clean + DP)
**Pod 2:** E3, E10, E4 (Byzantine experiments)

**Time:** 12 hours per pod
**Cost:** 12h × $0.50 × 2 pods = $12

**Recommended: Option B (faster, similar cost)**

---

## Part 5: Execution - Method A (Sequential on Single Pod)

### 5.1 Create Batch Script

```bash
nano run_all_cloud_experiments.sh
```

Paste:

```bash
#!/bin/bash
# DSAIN Cloud Experiments Batch Runner

set -e  # Exit on error

EXPERIMENTS=(
    "E1_DSAIN_a05_clean"
    "E5_DSAIN_a10_clean"
    "E7_DSAIN_a01_clean"
    "E9_DSAIN_a05_eps2"
    "E3_DSAIN_a05_byz20"
    "E10_DSAIN_a05_byz10"
    "E4_FedAvg_a05_byz20"
)

CONFIG_FILE="configs/cloud_dsain_experiments.json"
RESULTS_DIR="results"
LOGS_DIR="logs"

mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

echo "======================================================================="
echo "DSAIN CLOUD EXPERIMENTS BATCH RUNNER"
echo "======================================================================="
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Config file: $CONFIG_FILE"
echo "Start time: $(date)"
echo "======================================================================="

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "Starting: $EXP"
    echo "Time: $(date)"
    echo "-----------------------------------------------------------------------"

    # Run experiment
    python run_federated_experiment.py \
        --config $CONFIG_FILE \
        --exp_id $EXP \
        2>&1 | tee $LOGS_DIR/${EXP}.log

    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ $EXP completed successfully"

        # Validate result exists
        RESULT_FILE="${RESULTS_DIR}/${EXP}_*_seed42.json"
        if ls $RESULT_FILE 1> /dev/null 2>&1; then
            # Extract final accuracy
            FINAL_ACC=$(python -c "import json; import glob; f = glob.glob('${RESULT_FILE}')[0]; data = json.load(open(f)); print(data['final_test_accuracy'])")
            echo "  Final accuracy: $FINAL_ACC"
        else
            echo "  WARNING: Result file not found!"
        fi
    else
        echo "✗ $EXP FAILED"
        echo "Check logs: $LOGS_DIR/${EXP}.log"
        exit 1
    fi

    echo "-----------------------------------------------------------------------"

    # Brief cooldown between experiments
    sleep 60
done

echo ""
echo "======================================================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================================================="
echo "End time: $(date)"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Logs directory: $LOGS_DIR"
echo ""

# Generate summary
python generate_results_summary.py --results_dir $RESULTS_DIR --output summary.txt
cat summary.txt

echo ""
echo "Next step: Download results with 'download_results.sh'"
echo "======================================================================="
```

Save and exit (Ctrl+X, Y, Enter).

### 5.2 Make Script Executable

```bash
chmod +x run_all_cloud_experiments.sh
```

### 5.3 Launch in Tmux (Persist After Disconnect)

```bash
# Start tmux session
tmux new -s dsain_experiments

# Run batch script
./run_all_cloud_experiments.sh

# Detach from tmux: Press Ctrl+B, then D
# You can now disconnect from SSH safely
```

### 5.4 Monitor Progress (Reconnect Anytime)

```bash
# SSH back into pod
ssh ubuntu@<POD_IP>

# Reattach to tmux
tmux attach -t dsain_experiments

# Or monitor logs in real-time
tail -f logs/E1_DSAIN_a05_clean.log

# Check GPU usage
watch -n 10 nvidia-smi
```

---

## Part 6: Execution - Method B (Parallel on 2 Pods) ⚡ FASTER

### 6.1 Launch 2 Pods

**Pod 1:** DSAIN clean experiments (E1, E5, E7, E9)
**Pod 2:** Byzantine experiments (E3, E10, E4)

```bash
# Lambda Labs: Launch 2x V100 instances
# Pod 1 IP: 123.456.78.90
# Pod 2 IP: 123.456.78.91
```

### 6.2 Upload Code to Both Pods

```bash
# From local machine
scp code.tar.gz ubuntu@123.456.78.90:~/
scp code.tar.gz ubuntu@123.456.78.91:~/

# SSH to both and extract
ssh ubuntu@123.456.78.90 "tar -xzf code.tar.gz"
ssh ubuntu@123.456.78.91 "tar -xzf code.tar.gz"
```

### 6.3 Pod 1 - DSAIN Clean Experiments

```bash
# SSH to Pod 1
ssh ubuntu@123.456.78.90
cd code/

# Create Pod 1 script
nano run_pod1.sh
```

Paste:

```bash
#!/bin/bash
# Pod 1: DSAIN Clean + Privacy

EXPERIMENTS=(
    "E1_DSAIN_a05_clean"
    "E5_DSAIN_a10_clean"
    "E7_DSAIN_a01_clean"
    "E9_DSAIN_a05_eps2"
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo "Starting $EXP at $(date)"
    python run_federated_experiment.py --config configs/cloud_dsain_experiments.json --exp_id $EXP 2>&1 | tee logs/${EXP}.log
    echo "Completed $EXP at $(date)"
    sleep 60
done

echo "Pod 1 complete!"
```

```bash
chmod +x run_pod1.sh
tmux new -s pod1
./run_pod1.sh
# Ctrl+B, D to detach
```

### 6.4 Pod 2 - Byzantine Experiments

```bash
# SSH to Pod 2
ssh ubuntu@123.456.78.91
cd code/

# Create Pod 2 script
nano run_pod2.sh
```

Paste:

```bash
#!/bin/bash
# Pod 2: Byzantine Experiments

EXPERIMENTS=(
    "E3_DSAIN_a05_byz20"
    "E10_DSAIN_a05_byz10"
    "E4_FedAvg_a05_byz20"
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo "Starting $EXP at $(date)"
    python run_federated_experiment.py --config configs/cloud_dsain_experiments.json --exp_id $EXP 2>&1 | tee logs/${EXP}.log
    echo "Completed $EXP at $(date)"
    sleep 60
done

echo "Pod 2 complete!"
```

```bash
chmod +x run_pod2.sh
tmux new -s pod2
./run_pod2.sh
# Ctrl+B, D to detach
```

**Both pods now running in parallel. Check back in 12 hours.**

---

## Part 7: Monitor & Validate

### 7.1 Check Progress Remotely

```bash
# Pod 1 status
ssh ubuntu@123.456.78.90 "tail -20 code/logs/E1_DSAIN_a05_clean.log"

# Pod 2 status
ssh ubuntu@123.456.78.91 "tail -20 code/logs/E3_DSAIN_a05_byz20.log"
```

### 7.2 Validation Checkpoints

**After ~2 hours (round 50 should be reached):**

```bash
# SSH to pod
ssh ubuntu@<POD_IP>

# Check E1 at round 50
python validate_checkpoint.py \
    --experiment E1_DSAIN_a05_clean \
    --round 50 \
    --expected_accuracy 0.83 \
    --tolerance 0.05

# Expected: 83% +/- 5%
```

**If round 50 doesn't match historical baseline (83.27%):**
- STOP all experiments
- Debug configuration
- Don't waste $12 on wrong setup

### 7.3 Expected Final Results

| Experiment | Expected Accuracy | Notes |
|------------|------------------|-------|
| E1 (DSAIN α=0.5) | 91-94% | Your main result |
| E5 (DSAIN α=1.0) | 94-96% | Should be higher than E1 |
| E7 (DSAIN α=0.1) | 75-80% | Much lower (high heterogeneity) |
| E9 (DSAIN ε=2.0) | 89-92% | Slight privacy degradation |
| E3 (DSAIN Byz 20%) | 89-92% | Minimal degradation |
| E10 (DSAIN Byz 10%) | 90-93% | Even less degradation |
| E4 (FedAvg Byz 20%) | 15-40% | **CATASTROPHIC FAILURE** |

**Critical validation:**
- E4 must be < 50% (proves FedAvg fails)
- E1 > 90% (proves DSAIN works)
- E3 close to E1 (proves Byzantine resilience)

---

## Part 8: Download Results

### 8.1 Create Download Script (On Pod)

```bash
# On pod, create archive
cd ~/code/results
tar -czf cloud_dsain_results.tar.gz E*_DSAIN_*.json E*_FedAvg_*.json
```

### 8.2 Download to Local Machine

```bash
# From your local machine
scp ubuntu@<POD_IP>:~/code/results/cloud_dsain_results.tar.gz .

# Extract
tar -xzf cloud_dsain_results.tar.gz

# Verify files
ls -lh E*.json
# Should show 7 experiment result files
```

### 8.3 Combine with Local FedAvg Results

```bash
# You now have:
# Local: E2, E6, E8 (FedAvg baselines)
# Cloud: E1, E3, E4, E5, E7, E9, E10 (DSAIN + FedAvg Byzantine)

# Total: 10 experiments

# Create master results directory
mkdir -p final_results
cp E*.json final_results/

# Verify complete set
cd final_results
ls -1
# Should show:
# E1_DSAIN_a05_clean_500rounds_seed42.json
# E2_FedAvg_a05_clean_500rounds_seed42.json
# E3_DSAIN_a05_byz20_500rounds_seed42.json
# E4_FedAvg_a05_byz20_500rounds_seed42.json
# E5_DSAIN_a10_clean_500rounds_seed42.json
# E6_FedAvg_a10_clean_500rounds_seed42.json
# E7_DSAIN_a01_clean_500rounds_seed42.json
# E8_FedAvg_a01_clean_500rounds_seed42.json
# E9_DSAIN_a05_eps2_500rounds_seed42.json
# E10_DSAIN_a05_byz10_500rounds_seed42.json
```

---

## Part 9: Generate Manuscript Tables

### 9.1 Extract Key Metrics

```bash
python extract_key_results.py --results_dir final_results --output manuscript_tables.txt
```

Expected output:

```
==================================================
EXPERIMENT RESULTS SUMMARY
==================================================

Table 2: Heterogeneity Impact (DSAIN vs FedAvg)
--------------------------------------------------
Alpha     DSAIN         FedAvg        Gap
0.1       78.32%        68.26%        +10.06 pp
0.5       93.45%        87.64%        +5.81 pp
1.0       95.02%        91.43%        +3.59 pp

Table 3: Byzantine Robustness (DSAIN vs FedAvg)
--------------------------------------------------
Byzantine  DSAIN        FedAvg        Improvement
0%         93.76%       87.64%        1.07x
10%        93.22%       --            --
20%        92.94%       19.74%        4.71x

Table 4: Privacy-Utility Tradeoff
--------------------------------------------------
Epsilon    Accuracy     Degradation
inf        93.00%       --
2.0        91.28%       -1.72 pp

==================================================
```

### 9.2 Validate Against Claims

```bash
# Check claim: "DSAIN achieves 93.45%"
grep "E1_DSAIN" manuscript_tables.txt
# Should show: ~93.45%

# Check claim: "FedAvg collapses to <20% under Byzantine attack"
grep "E4_FedAvg.*byz20" manuscript_tables.txt
# Should show: 15-40%

# Check claim: "6.5× improvement"
# E3 (DSAIN Byz 20%): ~92.94%
# E4 (FedAvg Byz 20%): ~19.74%
# Ratio: 92.94 / 19.74 = 4.71x
# (Adjust abstract if not 6.5x - use actual ratio)
```

---

## Part 10: Cost Optimization & Cleanup

### 10.1 Terminate Pods Immediately After Completion

**Lambda Labs:**
```bash
# Web UI: Click "Terminate" next to each instance
# Or via CLI:
lambda terminate <instance-id>
```

**RunPod:**
```bash
# Web UI: Stop pod immediately after results downloaded
# Billing stops when pod is terminated
```

**CRITICAL:** Don't forget to terminate! Idle pods still cost $0.50/hour.

### 10.2 Final Cost Calculation

**Method A (Sequential, 1 pod):**
- 24 hours × $0.50/hour = $12.00

**Method B (Parallel, 2 pods):**
- 12 hours × $0.50/hour × 2 pods = $12.00

**Method C (Optimized, terminate early):**
- If experiments finish in 20 hours: 20 × $0.50 = $10.00

**Actual cost: $10-12 (well under $40 budget)**

---

## Troubleshooting

### Problem 1: "Pod runs out of disk space"

**Symptoms:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
```bash
# Check disk usage
df -h

# Delete CIFAR-10 cache after first download
rm -rf ~/.cache/torch

# Disable checkpoint saving (already in config)
```

### Problem 2: "Experiment crashes at round 342"

**Symptoms:**
- Log shows: "Killed" at round 342
- No error message

**Diagnosis:**
- OOM (out of memory)
- Pod ran out of RAM or GPU memory

**Solution:**
```bash
# Restart from checkpoint (if you saved any)
# Or reduce batch size and rerun
```

### Problem 3: "E4 (FedAvg Byzantine) doesn't collapse"

**Symptoms:**
- E4 achieves 55% accuracy instead of expected <30%

**Implications:**
- Your Byzantine attack isn't strong enough
- Or FedAvg is more robust than expected

**Solutions:**
1. Verify `byzantine_attack_type: "label_flipping"` in config
2. Check Byzantine clients are actually injecting bad gradients
3. Be honest in paper: Report actual results
4. Don't fabricate collapse - use real data

### Problem 4: "Tmux session lost after disconnect"

**Symptoms:**
- Reconnect to pod, `tmux attach` shows "no sessions"

**Solution:**
```bash
# List all tmux sessions
tmux ls

# If session exists but different name
tmux attach -t 0  # Attach to first session

# If truly lost, check if process still running
ps aux | grep python
# If running, monitor logs instead:
tail -f logs/E1_DSAIN_a05_clean.log
```

---

## Post-Completion Checklist

After all cloud experiments complete:

- [ ] All 7 experiments completed successfully
- [ ] E1 accuracy: 91-94% ✓
- [ ] E4 (FedAvg Byz) collapsed: <40% ✓
- [ ] Results downloaded to local machine ✓
- [ ] Pods terminated (no idle billing) ✓
- [ ] Combined with local FedAvg results (E2, E6, E8) ✓
- [ ] Total 10 experiments ready ✓
- [ ] Cost: $10-12 ✓
- [ ] Manuscript tables generated ✓

**Status:** ✓ All experiments complete with REAL data

---

## Next Steps

You now have:
- **Local:** 3 FedAvg baselines (E2, E6, E8) - FREE
- **Cloud:** 7 DSAIN + FedAvg Byzantine (E1, E3-E5, E7, E9-E10) - $10-12
- **Total:** 10 real experiments, $10-12 cost

**Next:**
1. Update manuscript with real results
2. Generate figures from real data
3. Remove all synthesis mentions
4. Submit to TMLR with confidence

**These are REAL experiments. Reviewers will accept them. No credibility issues.**

---

## Quick Reference Commands

```bash
# Setup pod
nvidia-smi
git clone <your-repo>
pip install -r requirements.txt

# Run single experiment
python run_federated_experiment.py --config configs/cloud_dsain_experiments.json --exp_id E1_DSAIN_a05_clean

# Run all (sequential)
tmux new -s dsain
./run_all_cloud_experiments.sh
# Ctrl+B, D

# Monitor
tmux attach -t dsain
tail -f logs/E1_DSAIN_a05_clean.log
nvidia-smi

# Download results
tar -czf results.tar.gz results/E*.json
scp ubuntu@<POD_IP>:~/code/results.tar.gz .

# Terminate pod
# Web UI: Click "Terminate"
```

---

**Estimated total time: 12-24 hours (depending on parallel vs sequential)**
**Estimated total cost: $10-12**

**You'll have real DSAIN results to compare against your local FedAvg baselines.**
