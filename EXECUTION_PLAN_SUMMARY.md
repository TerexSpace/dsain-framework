# DSAIN Real Experiments - Execution Plan Summary

**Created:** 2026-01-07
**Status:** Ready to Execute
**Total Budget:** $10-15 USD
**Total Time:** 2-3 days

---

## What You Have Now

✅ **Two detailed instruction files:**

1. **`LOCAL_FEDAVG_INSTRUCTIONS.md`** (15 pages)
   - Run 3 FedAvg baseline experiments on your GTX 4060
   - Cost: $0 (free, using your local GPU)
   - Time: 12-15 hours
   - Experiments: E2, E6, E8

2. **`CLOUD_POD_INSTRUCTIONS.md`** (18 pages)
   - Run 7 DSAIN + FedAvg Byzantine experiments on cloud
   - Cost: $10-12 USD
   - Time: 12-24 hours (can run parallel)
   - Experiments: E1, E3, E4, E5, E7, E9, E10

---

## The 10 Critical Experiments

### Local (GTX 4060) - FREE

| ID | Config | Time | Purpose |
|----|--------|------|---------|
| **E2** | FedAvg α=0.5, clean | 4.5h | Baseline comparison |
| **E6** | FedAvg α=1.0, clean | 4.5h | Heterogeneity validation |
| **E8** | FedAvg α=0.1, clean | 5h | Threshold validation |

**Total local:** 14 hours, $0

### Cloud (V100 Pod) - $10-12

| ID | Config | Time | Purpose |
|----|--------|------|---------|
| **E1** | DSAIN α=0.5, clean | 3h | Your main result |
| **E5** | DSAIN α=1.0, clean | 3h | Heterogeneity validation |
| **E7** | DSAIN α=0.1, clean | 3.5h | High heterogeneity |
| **E9** | DSAIN α=0.5, ε=2.0 | 3.5h | Privacy sweet spot |
| **E3** | DSAIN α=0.5, 20% Byz | 3.5h | Byzantine resilience |
| **E10** | DSAIN α=0.5, 10% Byz | 3.5h | Graceful degradation |
| **E4** | FedAvg α=0.5, 20% Byz | 4h | Catastrophic failure proof |

**Total cloud:** 24 hours sequential, OR 12 hours parallel (2 pods), $10-12

---

## Execution Timeline

### Option A: Sequential (Cheapest, 3 days)

**Day 1 (Friday evening → Saturday afternoon):**
- Start local experiments overnight
- E2, E6, E8 run sequentially: 14 hours
- Cost: $0

**Day 2 (Saturday afternoon → Sunday):**
- Launch cloud pod
- Start E1, E5, E7, E9 (DSAIN clean + privacy)
- Run for 12-14 hours
- Cost: ~$6

**Day 3 (Sunday → Monday):**
- Continue cloud pod
- Start E3, E10, E4 (Byzantine experiments)
- Run for 10-12 hours
- Cost: ~$6
- **Total cloud cost: $12**

**Total calendar time:** 3 days (Fri-Mon)
**Total cost:** $12

### Option B: Parallel (Fastest, 2 days)

**Day 1 (Friday evening):**
- Start local experiments: E2, E6, E8 (14 hours)
- Launch 2 cloud pods simultaneously:
  - Pod 1: E1, E5, E7, E9 (12 hours)
  - Pod 2: E3, E10, E4 (10 hours)

**Day 2 (Saturday evening):**
- All experiments complete
- Download results
- Terminate pods

**Total calendar time:** 1.5 days
**Total cost:** 12h × $0.50 × 2 pods = $12

---

## Expected Results (What You'll Get)

### Heterogeneity Comparison (Table 2)

| Alpha | DSAIN | FedAvg | Gap |
|-------|-------|--------|-----|
| 0.1 | 75-80% | 65-72% | +10 pp |
| 0.5 | 91-94% | 86-89% | +5 pp |
| 1.0 | 94-96% | 89-92% | +5 pp |

**Validates:** DSAIN > FedAvg across all heterogeneity levels

### Byzantine Resilience (Table 3)

| Byzantine % | DSAIN | FedAvg | Improvement |
|-------------|-------|--------|-------------|
| 0% | 93-94% | 87-89% | 1.06× |
| 10% | 90-93% | -- | -- |
| 20% | 89-92% | 15-40% | 4-6× |

**Validates:** DSAIN prevents catastrophic failure, FedAvg collapses

### Privacy-Utility (Table 4)

| Epsilon | Accuracy | Degradation |
|---------|----------|-------------|
| ∞ (no DP) | 93-94% | -- |
| 2.0 | 89-92% | -2 to -4 pp |

**Validates:** Strong privacy with minimal accuracy loss

---

## Success Criteria

### Minimum Viable (6/10 paper)

After running experiments, you need:

- ✓ E1 > E2 by at least 3 pp → Proves DSAIN > FedAvg
- ✓ E3 maintains >85% → Proves Byzantine resilience
- ✓ E4 collapses <50% → Proves FedAvg fails
- ✓ E6 > E8 by >15 pp → Proves heterogeneity threshold

**If all 4 pass:** Submit to TMLR with 60-70% acceptance

### Strong Results (7-8/10 paper)

- ✓ E1 > E2 by 5+ pp
- ✓ E4 collapses <30%
- ✓ E7-E8 show larger DSAIN advantage at high heterogeneity
- ✓ All 10 experiments complete successfully

**If achieved:** Submit with 80-90% acceptance probability

---

## Key Differences from Synthesis Approach

### What We Did Wrong Before

| Aspect | Synthesis (Bad) | Real Experiments (Good) |
|--------|-----------------|-------------------------|
| **Data** | Curve-fitted models | Actual GPU runs |
| **Cost** | $0 but worthless | $12 well spent |
| **Reviewer reaction** | "You're faking it" | "Solid validation" |
| **Credibility** | Damaged | Enhanced |
| **Acceptance probability** | 10-20% | 80-90% |

### Why This Will Work

1. **Real data:** Every number comes from actual 500-round training
2. **Real baselines:** FedAvg runs, not synthesized
3. **Honest scope:** 10 experiments, not claiming 30
4. **Proven approach:** Matches accepted TMLR papers (FedProx, SCAFFOLD)

---

## Validation Checkpoints (Don't Waste Money)

### Checkpoint 1: Round 50 (After 30 minutes)

**Before continuing each experiment, verify:**

```bash
# E2 (FedAvg α=0.5) at round 50 should be ~79%
python validate_checkpoint.py --experiment E2 --round 50 --expected 0.79 --tolerance 0.05
```

**If fails:** STOP immediately, debug config, don't waste 4 more hours

### Checkpoint 2: First Experiment Complete

**After E2 finishes:**
- Final accuracy should be 86-89%
- If < 82% or > 92%: Something is wrong
- Debug before running E6, E8

### Checkpoint 3: DSAIN vs FedAvg Gap

**After E1 and E2 both complete:**
- E1 should be 3-5 pp higher than E2
- If gap < 2 pp: Your DSAIN might not be working
- If gap > 10 pp: Results too good, verify configs

---

## What Happens After Experiments Complete

### Step 1: Validate Results (30 minutes)

```bash
# Check all experiments completed
ls final_results/*.json
# Should show 10 files

# Generate summary
python extract_key_results.py --results_dir final_results

# Verify key claims
# - DSAIN > FedAvg ✓
# - FedAvg Byzantine collapse ✓
# - Heterogeneity threshold ✓
```

### Step 2: Update Manuscript (2-3 hours)

**Changes needed:**
1. Abstract: Update all numbers to real results
2. Experimental Setup: Remove synthesis, describe 10 real experiments
3. Tables 2-5: Update with real numbers
4. Key Findings: Rewrite with actual results

**What gets deleted:**
- All synthesis methodology
- All 16 fake experiments
- Power-law model explanations
- Extrapolation discussion

### Step 3: Generate Figures (30 minutes)

```bash
# Use real data to generate figures
python generate_figures_from_real_data.py --results_dir final_results

# Output: 5-8 figures in PDF format
```

### Step 4: Final Review & Submit (1 day)

- Proofread manuscript
- Verify all claims match data
- Check references
- Compile LaTeX
- Submit to TMLR via OpenReview

---

## Cost Breakdown (Final)

| Item | Cost | Notes |
|------|------|-------|
| Local GPU (GTX 4060) | $0 | Free, using your hardware |
| Cloud Pod 1 (12h) | $6 | Lambda Labs V100 @ $0.50/h |
| Cloud Pod 2 (12h) | $6 | Parallel execution |
| **Total** | **$12** | **Well under $40 budget** |

**Remaining $28:** Save for potential reviewer requests (e.g., "add CIFAR-100")

---

## Risk Mitigation

### Risk 1: "Experiments fail partway through"

**Mitigation:**
- Tmux sessions persist after SSH disconnect
- Checkpoint validation at round 50
- Can restart failed experiments individually

### Risk 2: "Results don't match expectations"

**Mitigation:**
- Be honest, report actual results
- Don't fabricate data to match predictions
- Adjust claims based on reality

### Risk 3: "FedAvg doesn't collapse as expected"

**If E4 achieves 55% instead of <30%:**
- Still a significant gap (E3: 92% vs E4: 55% = 1.7× improvement)
- Adjust abstract: "DSAIN achieves 1.7× better accuracy" (not 6.5×)
- Honest reporting is more important than spectacular claims

### Risk 4: "Run out of cloud credits"

**Mitigation:**
- Add $20 to Lambda Labs at start
- Monitor spending every 6 hours
- Can pause/resume experiments if needed

---

## Next Steps (Your Actions)

### Step 1: Read Both Instruction Files (30 minutes)

- [ ] Read `LOCAL_FEDAVG_INSTRUCTIONS.md` (15 pages)
- [ ] Read `CLOUD_POD_INSTRUCTIONS.md` (18 pages)
- [ ] Understand checkpoint validation process

### Step 2: Prepare Local Environment (1 hour)

- [ ] Verify GTX 4060 is working (`nvidia-smi`)
- [ ] Install any missing Python packages
- [ ] Test one experiment for 5 rounds to verify setup
- [ ] Create configs directory and JSON files

### Step 3: Launch Local Experiments (Friday evening)

- [ ] Start E2, E6, E8 in sequence
- [ ] Monitor first 30 minutes to verify round 50 checkpoint
- [ ] Let run overnight

### Step 4: Launch Cloud Pods (Saturday)

- [ ] Create Lambda Labs account
- [ ] Add $20 credit
- [ ] Launch 1-2 V100 pods
- [ ] Upload code
- [ ] Start DSAIN experiments
- [ ] Monitor progress

### Step 5: Download & Validate (Sunday)

- [ ] Download all results
- [ ] Validate success criteria
- [ ] Generate summary tables
- [ ] Terminate cloud pods (stop billing!)

### Step 6: Update Manuscript (Monday)

- [ ] Remove synthesis content
- [ ] Update all tables with real numbers
- [ ] Generate figures from real data
- [ ] Final proofread

### Step 7: Submit to TMLR (Tuesday)

- [ ] Compile LaTeX
- [ ] Upload to OpenReview
- [ ] Submit for review

---

## Questions Before You Start?

**Common questions answered:**

**Q: What if I only have time for 6 experiments instead of 10?**
A: Run E1-E6 (critical baselines). Skip E7, E9, E10. Still publishable at 6/10 score.

**Q: Can I run everything on local GPU to save $12?**
A: Yes, but will take 50-60 hours (GTX 4060 is slower). Budget 4-5 days.

**Q: What if round 50 checkpoint fails?**
A: STOP immediately. Debug config. Don't waste 4 hours on wrong experiment.

**Q: Can I use RunPod instead of Lambda Labs?**
A: Yes, both work. RunPod is $0.50-0.90/h. Instructions are same.

**Q: What if I run out of money?**
A: Run only E1-E4 (4 experiments, ~$6). Minimal but submittable.

---

## Final Checklist Before Starting

- [ ] I understand the difference between synthesis (bad) and real experiments (good)
- [ ] I have $15-20 budget available
- [ ] I have 2-3 days calendar time available
- [ ] I've read both instruction files completely
- [ ] My GTX 4060 is working properly
- [ ] I'm ready to run experiments honestly (report real results, not fabricated)
- [ ] I understand checkpoint validation will prevent wasted money

**If all checked:** You're ready to start!

---

## Summary

**You have everything you need to run 10 real experiments for $12.**

**This will transform your paper from:**
- 4/10 (reject with synthesis)
- → 7-8/10 (accept with real experiments)

**The instructions are detailed and foolproof. Follow them step-by-step.**

**Good luck! You're about to get real, publishable results.**
