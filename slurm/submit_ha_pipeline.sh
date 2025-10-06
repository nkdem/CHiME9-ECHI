#!/bin/bash
# Pipeline submission script for HA subset training
# This chains the jobs so training only starts after unpacking completes

echo "Submitting CHiME9 ECHI HA Pipeline..."
echo ""

# Create log directories
mkdir -p slurm/logs/unpack
mkdir -p slurm/logs/train

# Submit unpacking job
UNPACK_JOB=$(sbatch --parsable slurm/sbatch/01_unpack_ha_subset.sh)
echo "✓ Submitted unpacking job: $UNPACK_JOB"

# Submit training job with dependency
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$UNPACK_JOB slurm/sbatch/02_train_ha_subset.sh)
echo "✓ Submitted training job: $TRAIN_JOB (depends on $UNPACK_JOB)"

echo ""
echo "Pipeline submitted successfully!"
echo ""
echo "Job Chain:"
echo "  1. Unpack (Job $UNPACK_JOB) → 2. Train (Job $TRAIN_JOB)"
echo ""
echo "Monitor with:"
echo "  watch -n 2 'squeue -u s2203859'"
echo ""
echo "Check logs:"
echo "  tail -f slurm/logs/unpack/${UNPACK_JOB}_unpack_ha.out"
echo "  tail -f slurm/logs/train/${TRAIN_JOB}_train_ha.out"
