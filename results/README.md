# DSAIN Experiment Results

This folder contains results from DSAIN framework experiments.

## Structure

Each experiment creates a timestamped subdirectory with:
```
results/
├── dsain_single_YYYYMMDD_HHMMSS/
│   ├── config.json              # Experiment configuration
│   ├── results_run_000.json     # Individual run results
│   ├── results_run_001.json
│   ├── ...
│   ├── aggregated_results.json  # Mean/std across all runs
│   └── figures/                 # Generated plots (first run only)
│       ├── convergence_curves.pdf
│       ├── byzantine_resilience.pdf
│       └── scalability.pdf
```

## Running Experiments

From the `code/` directory:

```bash
# Run 3 experiments with default settings
python run_experiments.py --num_runs 3

# Run 5 experiments in all modes
python run_experiments.py --num_runs 5 --mode all

# Run with specific seeds for reproducibility
python run_experiments.py --seeds 42 123 456 789 1000

# Run Byzantine resilience experiments
python run_experiments.py --num_runs 3 --mode byzantine

# Run scalability experiments
python run_experiments.py --num_runs 3 --mode scalability

# Custom configuration
python run_experiments.py --num_runs 5 \
    --num_clients 100 \
    --num_rounds 200 \
    --byzantine_frac 0.1 \
    --model_dim 100
```

## Result Files

### config.json
Contains all experiment parameters:
- `mode`: Experiment type (single, byzantine, scalability, all)
- `num_runs`: Number of runs executed
- `seeds`: List of random seeds used
- `timestamp`: When experiment started

### results_run_XXX.json
Individual run results including:
- `seed`: Random seed used
- `final_model_norm`: Model parameter norm at end of training
- `avg_update_norm`: Average gradient update magnitude
- `training_time`: Wall-clock training time (seconds)
- `byzantine_ids`: Which clients were Byzantine

### aggregated_results.json
Statistics across all runs:
- `mean`: Average value
- `std`: Standard deviation
- `min`, `max`: Range of values
- `values`: All individual values

## Notes

- Results are automatically excluded from Git (see `.gitignore`)
- Figures are only generated for the first run of each batch
- Use specific seeds for paper reproducibility (default: 42, 123, 456)
