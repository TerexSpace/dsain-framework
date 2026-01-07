# Pod Quick Start Guide - Run All 10 Experiments

**Total Time:** ~33 hours on A5000 GPU
**Total Cost:** ~$9.24 on RunPod A5000 @ $0.28/hour

---

## Step 1: Launch Your Pod

### Option A: RunPod (Recommended - Cheapest)
1. Go to https://www.runpod.io
2. Sign up and add $15 credit
3. Click "Deploy" → "GPU Pods"
4. Select **RTX A5000** (24GB, $0.28/hour)
5. Choose template: **PyTorch 2.0+**
6. Click "Deploy On-Demand"
7. Wait ~30 seconds for pod to start
8. Click "Connect" → Copy SSH command

### Option B: Lambda Labs
1. Go to https://lambdalabs.com/service/gpu-cloud
2. Sign up and add $15 credit
3. Click "Launch Instance"
4. Select **V100** (16GB, $0.50/hour)
5. Choose region with availability
6. Click "Launch"
7. Copy SSH command

---

## Step 2: Connect to Pod via SSH

```bash
# Use the SSH command from your provider
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519

# Example RunPod:
ssh root@123.456.78.90 -p 22345 -i ~/.ssh/runpod_key

# Example Lambda Labs:
ssh ubuntu@123.456.78.90
```

---

## Step 3: Upload Your Code

**Option A: Git Clone (if your code is on GitHub)**
```bash
cd ~
git clone https://github.com/yourusername/your-repo.git
cd your-repo/tmlr_manuscript
```

**Option B: SCP Upload (if code is local)**

From your local machine:
```bash
# Compress your code
cd "C:\Users\aleke\Documents\My_Devs_IDE\17. My OpenSource Projects\6. DSAIN_Framework_to_TMLR_Done_under review"
tar -czf tmlr_code.tar.gz tmlr_manuscript/

# Upload to pod
scp -P <port> tmlr_code.tar.gz root@<pod-ip>:~/

# On pod: Extract
ssh root@<pod-ip> -p <port>
cd ~
tar -xzf tmlr_code.tar.gz
cd tmlr_manuscript
```

**Option C: Direct rsync (Fastest)**
```bash
rsync -avz -e "ssh -p <port>" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "C:\Users\aleke\Documents\My_Devs_IDE\17. My OpenSource Projects\6. DSAIN_Framework_to_TMLR_Done_under review\tmlr_manuscript" \
    root@<pod-ip>:~/
```

---

## Step 4: Start Experiments in Tmux

**CRITICAL:** Run in tmux so experiments continue if SSH disconnects!

```bash
# Start tmux session
tmux new -s experiments

# Navigate to code
cd ~/tmlr_manuscript

# Run all experiments (single command!)
bash run_pod_experiments.sh
```

**Detach from tmux:** Press `Ctrl+B`, then `D`
**Reattach to tmux:** `tmux attach -t experiments`

---

## Step 5: Monitor Progress

While experiments run, you can:

### Check Live Logs
```bash
# Reattach to tmux
tmux attach -t experiments

# Or tail the log file
tail -f experiment_run.log
```

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Completed Experiments
```bash
ls -lh code/results/final_experiments/
cat code/results/final_experiments/experiment_summary.json | python3 -m json.tool
```

---

## Step 6: Download Results (After Completion)

From your **local machine**:

```bash
# Download all results
scp -P <port> -r root@<pod-ip>:~/tmlr_manuscript/code/results/final_experiments ./

# Or just the summary
scp -P <port> root@<pod-ip>:~/tmlr_manuscript/code/results/final_experiments/experiment_summary.json ./
```

---

## Step 7: Terminate Pod (IMPORTANT!)

**Don't forget to terminate the pod after downloading results!**

### RunPod:
1. Go to https://www.runpod.io/console/pods
2. Click "Stop" on your pod
3. Verify billing stopped

### Lambda Labs:
1. Go to https://lambdalabs.com/service/gpu-cloud
2. Click "Terminate" on your instance
3. Confirm termination

---

## Troubleshooting

### Experiments Fail Early
Check error in log:
```bash
tail -50 experiment_run.log
```

Common issues:
- **Out of memory**: Reduce batch_size in config
- **CUDA not available**: Check `nvidia-smi`
- **Missing packages**: Run `pip install -r requirements.txt`

### SSH Connection Lost
No problem! Tmux keeps experiments running.
```bash
# Reconnect to pod
ssh root@<pod-ip> -p <port>

# Reattach to tmux session
tmux attach -t experiments
```

### Want to Pause/Resume
```bash
# In tmux, press Ctrl+C to stop
# Results so far are saved in results/final_experiments/

# Resume: Re-run specific experiments manually
cd code
python3 -c "
from run_all_10_experiments import get_all_experiments, run_single_experiment
exps = get_all_experiments()
# Run only E5 for example
run_single_experiment(exps[4])  # Index 4 = E5
"
```

---

## Expected Timeline

| Time | Progress |
|------|----------|
| 0h | Setup complete, E1 starts |
| 3h | E1 complete (DSAIN clean) |
| 6h | E2 complete (FedAvg clean) |
| 9h | E3 complete (DSAIN Byzantine) |
| 13h | E4 complete (FedAvg Byzantine - should collapse!) |
| 16h | E5 complete (DSAIN α=1.0) |
| 19h | E6 complete (FedAvg α=1.0) |
| 23h | E7 complete (DSAIN α=0.1) |
| 28h | E8 complete (FedAvg α=0.1) |
| 31h | E9 complete (DSAIN privacy) |
| 33h | **E10 complete - ALL DONE!** |

---

## Cost Breakdown

### RunPod A5000 @ $0.28/hour
- 33 hours × $0.28 = **$9.24**
- Add $15 credit = enough for experiments + buffer

### Lambda Labs V100 @ $0.50/hour
- 33 hours × $0.50 = **$16.50**
- Add $20 credit = enough for experiments + buffer

---

## Success Criteria

After all experiments finish, check summary:

```bash
cd code/results/final_experiments
python3 -c "
import json
with open('experiment_summary.json') as f:
    data = json.load(f)

print('Experiment Results:')
print('-' * 60)
for exp_id in ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']:
    result = data['results'][exp_id]
    acc = result['final_accuracy']
    print(f'{exp_id}: {result[\"config\"][\"exp_name\"]:<30} Acc: {acc:.4f}')
print('-' * 60)
print(f'DSAIN vs FedAvg (α=0.5): E1={data[\"results\"][\"E1\"][\"final_accuracy\"]:.4f} vs E2={data[\"results\"][\"E2\"][\"final_accuracy\"]:.4f}')
print(f'Byzantine: E3={data[\"results\"][\"E3\"][\"final_accuracy\"]:.4f} vs E4={data[\"results\"][\"E4\"][\"final_accuracy\"]:.4f}')
"
```

**Expected Results:**
- ✅ E1 > E2 by 3-5 pp (DSAIN beats FedAvg)
- ✅ E3 > 85% (DSAIN handles Byzantine)
- ✅ E4 < 50% (FedAvg collapses under Byzantine)
- ✅ All experiments complete successfully

**If these hold:** Your paper is ready for TMLR submission! 🎉

---

## Next Steps After Experiments

1. **Validate Results** (30 min)
   - Check success criteria above
   - Verify no anomalies

2. **Update Manuscript** (2-3 hours)
   - Remove synthesis content
   - Update tables with real numbers
   - Regenerate figures

3. **Submit to TMLR** (1 day)
   - Final proofread
   - Compile LaTeX
   - Submit via OpenReview

---

## Quick Reference Commands

```bash
# Single command to run everything
bash run_pod_experiments.sh

# Monitor progress
tail -f experiment_run.log

# Check GPU
nvidia-smi

# List results
ls -lh code/results/final_experiments/

# Terminate pod
# RunPod: Web UI → Stop
# Lambda: Web UI → Terminate

# Download results
scp -P <port> -r root@<pod-ip>:~/tmlr_manuscript/code/results/final_experiments ./
```

---

## Support

If anything goes wrong, check:
1. `experiment_run.log` for errors
2. GPU memory with `nvidia-smi`
3. Results so far in `code/results/final_experiments/`

You can always:
- Stop and resume later
- Run individual experiments manually
- Adjust configurations if needed

**Good luck! You're about to get real, publishable results.** 🚀
