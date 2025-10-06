#!/bin/bash
#SBATCH --job-name=train_ha_subset
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/train/%j_train_ha.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Training HA Baseline (Subset)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Environment setup with UV
echo "Setting up UV virtual environment..."
cd $HOME/CHiME9-ECHI

# Activate UV virtual environment
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Paths
USER="s2203859"
SCRATCH_DIR="/disk/scratch/${USER}/CHiME9-ECHI"
SCRATCH_PROCESSED_DIR="${SCRATCH_DIR}/processed"
HOME_PROCESSED_DIR="$HOME/CHiME9-ECHI/data/processed_ha_subset"

# Check for processed data and copy to scratch if needed
if [ -d "${SCRATCH_PROCESSED_DIR}/train_segments" ]; then
    echo "✓ Processed data already in scratch"
    echo "   ${SCRATCH_PROCESSED_DIR}"
elif [ -d "${HOME_PROCESSED_DIR}/train_segments" ]; then
    echo "Processed data found in home directory"
    echo "Copying to scratch for fast I/O during training..."
    mkdir -p ${SCRATCH_PROCESSED_DIR}
    time rsync -av --progress ${HOME_PROCESSED_DIR}/ ${SCRATCH_PROCESSED_DIR}/
    echo "✓ Copy to scratch completed"
else
    echo "ERROR: Processed data not found in either location:"
    echo "  - ${SCRATCH_PROCESSED_DIR}"
    echo "  - ${HOME_PROCESSED_DIR}"
    echo "Please run 01_unpack_ha_subset.sh first!"
    exit 1
fi

# Always use scratch for training (best I/O performance)
PROCESSED_DIR="${SCRATCH_PROCESSED_DIR}"

echo ""
echo "Training will use data from scratch:"
echo "   ${PROCESSED_DIR}"
echo "Processed data size:"
du -sh ${PROCESSED_DIR}
echo ""

# Create config override for training
cat > /tmp/train_scratch_${SLURM_JOB_ID}.yaml << EOF
defaults:
  - override /paths: paths

paths:
  root_dir: ${PROCESSED_DIR}
  echi_dir: ${SCRATCH_DIR}/chime9_echi
EOF

# Run training
echo "Starting training..."
python scripts/train/train.py \
    device=ha \
    --config-path /tmp \
    --config-name train_scratch_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Training completed: $(date)"
echo "=========================================="

# Cleanup
rm -f /tmp/train_scratch_${SLURM_JOB_ID}.yaml
