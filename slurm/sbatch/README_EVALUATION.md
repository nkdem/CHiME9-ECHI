# Evaluation Pipeline for Trained Models

This directory contains scripts for evaluating trained CHiME9-ECHI models on the dev set.

## Overview

The evaluation pipeline consists of two main steps:

1. **Enhancement (Inference)**: Run the trained model on the dev set to generate enhanced audio
2. **Evaluation**: Compute metrics on the enhanced audio compared to reference signals

## Files Created

### Configuration
- `config/enhancement/enhance_args/trained_model.yaml` - Enhancement config for trained checkpoints

### SLURM Scripts
- `03_enhance_ha_trained.sh` - Run enhancement with your trained model
- `04_evaluate_ha_trained.sh` - Evaluate the enhanced audio

## Usage

### 1. Update Configuration

Edit `03_enhance_ha_trained.sh` to set:
```bash
EXP_NAME="baseline_fixed2"        # Your experiment name
CHECKPOINT_EPOCH="epoch049"        # Which checkpoint to use
```

Also update `04_evaluate_ha_trained.sh` with the same `EXP_NAME`.

### 2. Run Enhancement

From the cluster, submit the enhancement job:

```bash
cd ~/CHiME9-ECHI
sbatch slurm/sbatch/03_enhance_ha_trained.sh
```

This will:
- Copy raw HA data to scratch for fast I/O
- Copy your trained checkpoint to scratch
- Run the model on the dev set
- Save enhanced audio to `data/working_dir/experiments/${EXP_NAME}/enhancement/dev_submission/`

Monitor progress:
```bash
# Check job status
squeue -u s2203859

# View logs (replace JOBID with actual job ID)
tail -f slurm/logs/enhance/JOBID_enhance_ha_trained.out
```

### 3. Run Evaluation

Once enhancement is complete, submit the evaluation job:

```bash
sbatch slurm/sbatch/04_evaluate_ha_trained.sh
```

This will:
- Copy metadata and reference audio to scratch
- Copy enhanced audio to scratch
- Run the full evaluation pipeline (setup, validate, prepare, evaluate, report)
- Save results to `data/working_dir/experiments/${EXP_NAME}/evaluation/`

Monitor progress:
```bash
# Check job status
squeue -u s2203859

# View logs (replace JOBID with actual job ID)
tail -f slurm/logs/evaluate/JOBID_evaluate_ha_trained.out
```

### 4. View Results

After evaluation completes, check the results:

```bash
cd ~/CHiME9-ECHI
ls -lh data/working_dir/experiments/${EXP_NAME}/evaluation/results/

# View results JSON
cat data/working_dir/experiments/${EXP_NAME}/evaluation/results/results.dev.ha.*.json
```

## Expected Output Structure

```
data/working_dir/experiments/baseline_fixed2/
├── train_ha/
│   └── checkpoints/
│       ├── epoch000.pt
│       ├── ...
│       └── epoch049.pt
├── enhancement/
│   ├── hydra/
│   └── dev_submission/
│       ├── dev_S01.ha.P01.wav
│       ├── dev_S01.ha.P02.wav
│       └── ...
└── evaluation/
    ├── hydra/
    ├── segments/
    │   └── ha/
    │       ├── individual/
    │       └── summed/
    ├── results/
    │   ├── results.dev.ha.individual.json
    │   └── results.dev.ha.summed.json
    └── reports/
        └── [session-specific reports]
```

## Evaluation Metrics

The evaluation computes several speech quality metrics using the Versa scorer:
- STOI (Short-Time Objective Intelligibility)
- PESQ (Perceptual Evaluation of Speech Quality)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- Other metrics defined in `config/evaluation/metrics.yaml`

## Troubleshooting

### Enhancement fails with "Checkpoint not found"
- Check that the checkpoint path is correct in the script
- Verify the checkpoint exists: `ls ~/CHiME9-ECHI/data/working_dir/experiments/${EXP_NAME}/train_ha/checkpoints/`

### Evaluation fails with "Enhanced outputs not found"
- Run the enhancement script first
- Check enhancement completed successfully

### Out of memory
- Reduce `window_size` or increase `stride` in `config/enhancement/enhance_args/trained_model.yaml`
- Request more GPU memory in SBATCH directives

## Running Multiple Checkpoints

To evaluate multiple checkpoints (e.g., epoch020, epoch030, epoch040, epoch049):

1. Create separate directories or modify the scripts to loop through checkpoints
2. Or simply update `CHECKPOINT_EPOCH` and re-run both scripts for each checkpoint

Example:
```bash
# Evaluate epoch 20
vim slurm/sbatch/03_enhance_ha_trained.sh  # Change CHECKPOINT_EPOCH="epoch020"
sbatch slurm/sbatch/03_enhance_ha_trained.sh
# Wait for completion, then run evaluation
sbatch slurm/sbatch/04_evaluate_ha_trained.sh

# Repeat for other epochs...
```

## Notes

- All scripts use scratch disk (`/disk/scratch/`) for fast I/O during processing
- Outputs are automatically copied back to your home directory when complete
- The evaluation pipeline uses GPU by default (`evaluate.use_gpu=true`)
- Logs are saved to `slurm/logs/enhance/` and `slurm/logs/evaluate/`
