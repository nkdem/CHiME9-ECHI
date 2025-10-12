#!/bin/bash
#SBATCH --job-name=train_ha_test
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/train/%j_train_ha_test.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Training HA TEST (Quick validation)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
nvidia-smi
echo ""
echo "Start time: $(date)"
echo ""

# Environment setup with UV
echo "Setting up UV virtual environment..."
cd $HOME/CHiME9-ECHI

# Activate UV virtual environment
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Paths
USER="s2203859"
SCRATCH_DIR="/disk/scratch/${USER}/CHiME9-ECHI"
SCRATCH_PROCESSED_DIR="${SCRATCH_DIR}/processed"
HOME_PROCESSED_DIR="$HOME/CHiME9-ECHI/data/processed_ha_subset"

# create scratch directory if it doesn't exist
mkdir -p ${SCRATCH_PROCESSED_DIR}
mkdir -p ${SCRATCH_DIR}/chime9_echi

# Copy metadata and participant files to scratch if needed
ECHI_SOURCE="$HOME/CHIME9/chime9_echi"
if [ -d "${SCRATCH_DIR}/chime9_echi/metadata" ]; then
    echo "✓ Metadata already in scratch"
else
    echo "Copying metadata to scratch..."
    time rsync -av --progress ${ECHI_SOURCE}/metadata/ ${SCRATCH_DIR}/chime9_echi/metadata/
    echo "✓ Metadata copy completed"
fi

if [ -d "${SCRATCH_DIR}/chime9_echi/participant" ]; then
    echo "✓ Participant audio already in scratch"
else
    echo "Copying participant audio to scratch..."
    time rsync -av --progress ${ECHI_SOURCE}/participant/ ${SCRATCH_DIR}/chime9_echi/participant/
    echo "✓ Participant audio copy completed"
fi
echo ""

# Check for processed data and copy to scratch if needed
if [ -d "${SCRATCH_PROCESSED_DIR}/train_segments" ] && [ -d "${SCRATCH_PROCESSED_DIR}/participant" ]; then
    echo "✓ Processed data already in scratch"
    echo "   ${SCRATCH_PROCESSED_DIR}"
elif [ -d "${HOME_PROCESSED_DIR}/train_segments" ]; then
    echo "Processed data found in home directory"
    echo "Copying to scratch for fast I/O during training..."
    mkdir -p ${SCRATCH_PROCESSED_DIR}
    time rsync -av --progress ${HOME_PROCESSED_DIR}/ ${SCRATCH_PROCESSED_DIR}/
    echo "✓ Copy to scratch completed"
    echo ""
    echo "Verifying participant files were copied..."
    if [ -d "${SCRATCH_PROCESSED_DIR}/participant" ]; then
        echo "✓ Participant files confirmed in scratch"
    else
        echo "ERROR: Participant files missing from scratch after copy!"
        exit 1
    fi
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

# Create unique experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="ha_test_${TIMESTAMP}"

# Run training with TEST parameters
echo "=========================================="
echo "RUNNING IN TEST MODE:"
echo "  - 3 epochs (instead of 50)"
echo "  - Debug mode enabled (progress bars)"
echo "  - Checkpoint every epoch"
echo "  - Experiment name: ${EXP_NAME}"
echo "=========================================="
echo ""
echo "Starting test training..."
python scripts/train/train_script.py \
    device=ha \
    paths.root_dir=${PROCESSED_DIR} \
    paths.echi_dir=${SCRATCH_DIR}/chime9_echi \
    shared.exp_name=${EXP_NAME} \
    train.epochs=3 \
    debug=true \
    train.checkpoint_interval=1

echo ""
echo "=========================================="
echo "Test training completed: $(date)"
echo "=========================================="

# Copy all outputs from scratch back to home (except audio files)
echo ""
echo "Copying all training outputs from scratch to home..."
echo "Excluding audio directories to save space..."
mkdir -p ${HOME_PROCESSED_DIR}

# Copy everything from scratch except audio files
time rsync -av --progress \
    --exclude='train_segments/' \
    --exclude='participant/' \
    --exclude='*.wav' \
    --exclude='*.flac' \
    ${SCRATCH_PROCESSED_DIR}/ ${HOME_PROCESSED_DIR}/

echo "✓ Training outputs saved to: ${HOME_PROCESSED_DIR}"
echo ""
echo "Contents copied from scratch:"
du -sh ${HOME_PROCESSED_DIR}/*
echo ""
echo "Model checkpoints location:"
find ${HOME_PROCESSED_DIR} -name "*.pt" -o -name "*.pth" 2>/dev/null | head -5
