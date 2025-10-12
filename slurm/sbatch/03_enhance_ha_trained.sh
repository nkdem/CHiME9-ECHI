#!/bin/bash
#SBATCH --job-name=enhance_ha_trained
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/enhance/%j_enhance_ha_trained.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Enhancement HA (Trained Model)"
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
HOME_DATA_DIR="$HOME/CHIME9/chime9_echi"
HOME_WORKING_DIR="$HOME/CHiME9-ECHI/data/working_dir"

# Experiment name - UPDATE THIS to match your trained model directory
EXP_NAME="baseline_fixed2"
CHECKPOINT_EPOCH="epoch049"

echo ""
echo "Using experiment: ${EXP_NAME}"
echo "Using checkpoint: ${CHECKPOINT_EPOCH}"
echo ""

# Check if raw data exists and copy to scratch if needed for fast I/O
if [ ! -d "${SCRATCH_DIR}/chime9_echi/ha" ]; then
    echo ""
    echo "Copying HA raw data to scratch for fast I/O..."
    mkdir -p ${SCRATCH_DIR}/chime9_echi

    # Copy only HA-related data
    time rsync -av --progress ${HOME_DATA_DIR}/ha/ ${SCRATCH_DIR}/chime9_echi/ha/
    rsync -av ${HOME_DATA_DIR}/metadata/ ${SCRATCH_DIR}/chime9_echi/metadata/
    rsync -av ${HOME_DATA_DIR}/participant/ ${SCRATCH_DIR}/chime9_echi/participant/

    echo "✓ Copy to scratch completed"
else
    echo "✓ Raw HA data already in scratch"
fi

# Check if trained model checkpoint exists in home, copy to scratch if needed
HOME_CHECKPOINT_DIR="${HOME_WORKING_DIR}/experiments/${EXP_NAME}/train_ha/checkpoints"
SCRATCH_CHECKPOINT_DIR="${SCRATCH_DIR}/working_dir/experiments/${EXP_NAME}/train_ha/checkpoints"

if [ ! -f "${SCRATCH_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt" ]; then
    echo ""
    echo "Copying trained model checkpoint to scratch..."
    mkdir -p ${SCRATCH_CHECKPOINT_DIR}

    if [ -f "${HOME_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt" ]; then
        cp ${HOME_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt ${SCRATCH_CHECKPOINT_DIR}/
        echo "✓ Checkpoint copied to scratch"
    else
        echo "ERROR: Checkpoint not found at ${HOME_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt"
        echo "Please update EXP_NAME and CHECKPOINT_EPOCH in this script"
        exit 1
    fi
else
    echo "✓ Checkpoint already in scratch"
fi

echo ""
echo "Running enhancement on HA dev set with trained model..."
echo "Input: ${SCRATCH_DIR}/chime9_echi"
echo "Model: ${SCRATCH_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt"
echo ""

# Run enhancement with trained model
python run_enhancement.py \
    device=ha \
    dataset=dev \
    paths.echi_dir=${SCRATCH_DIR}/chime9_echi \
    paths.root_dir=${SCRATCH_DIR}/working_dir \
    shared.exp_name=${EXP_NAME} \
    enhance_args=trained_model \
    enhance_args.args.ckpt_path=${SCRATCH_CHECKPOINT_DIR}/${CHECKPOINT_EPOCH}.pt

echo ""
echo "=========================================="
echo "Enhancement completed: $(date)"
echo ""

# Copy enhancement outputs back to home
SCRATCH_ENHANCEMENT_DIR="${SCRATCH_DIR}/working_dir/experiments/${EXP_NAME}/enhancement"
HOME_ENHANCEMENT_DIR="${HOME_WORKING_DIR}/experiments/${EXP_NAME}/enhancement"

if [ -d "${SCRATCH_ENHANCEMENT_DIR}" ]; then
    echo "Copying enhancement outputs from scratch to home..."
    mkdir -p $(dirname ${HOME_ENHANCEMENT_DIR})
    time rsync -av --progress ${SCRATCH_ENHANCEMENT_DIR}/ ${HOME_ENHANCEMENT_DIR}/
    echo "✓ Enhancement outputs saved to: ${HOME_ENHANCEMENT_DIR}"
else
    echo "WARNING: Enhancement directory not found in scratch"
fi

echo ""
echo "Output location:"
find ${HOME_ENHANCEMENT_DIR} -name "*.wav" 2>/dev/null | head -5
echo ""
echo "Total enhanced files:"
find ${HOME_ENHANCEMENT_DIR} -name "*.wav" 2>/dev/null | wc -l
echo "=========================================="
