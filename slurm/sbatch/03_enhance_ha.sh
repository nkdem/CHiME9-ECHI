#!/bin/bash
#SBATCH --job-name=enhance_ha
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/enhance/%j_enhance_ha.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Enhancement HA"
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

echo ""
echo "Running enhancement on HA dev set..."
echo "Input: ${SCRATCH_DIR}/chime9_echi"
echo ""

# Run enhancement
python run_enhancement.py \
    device=ha \
    dataset=dev \
    paths.echi_dir=${SCRATCH_DIR}/chime9_echi

echo ""
echo "=========================================="
echo "Enhancement completed: $(date)"
echo ""
echo "Output location:"
ls -lh data/working_dir/experiments/*/enhancement/*_submission/ 2>/dev/null || echo "Check logs for output location"
echo "=========================================="
