#!/bin/bash
#SBATCH --job-name=eval_ha_trained
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/evaluate/%j_evaluate_ha_trained.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Evaluation HA (Trained Model)"
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

echo ""
echo "Evaluating experiment: ${EXP_NAME}"
echo ""

# Check if raw data metadata exists and copy to scratch if needed
if [ ! -d "${SCRATCH_DIR}/chime9_echi/metadata" ]; then
    echo "Copying metadata to scratch..."
    mkdir -p ${SCRATCH_DIR}/chime9_echi
    rsync -av ${HOME_DATA_DIR}/metadata/ ${SCRATCH_DIR}/chime9_echi/metadata/
    echo "✓ Metadata copied to scratch"
else
    echo "✓ Metadata already in scratch"
fi

# Check if participant audio exists and copy to scratch if needed
if [ ! -d "${SCRATCH_DIR}/chime9_echi/participant" ]; then
    echo "Copying participant audio to scratch..."
    rsync -av ${HOME_DATA_DIR}/participant/ ${SCRATCH_DIR}/chime9_echi/participant/
    echo "✓ Participant audio copied to scratch"
else
    echo "✓ Participant audio already in scratch"
fi

# Check if reference audio exists and copy to scratch if needed
if [ ! -d "${SCRATCH_DIR}/chime9_echi/ref" ]; then
    echo "Copying reference audio to scratch..."
    rsync -av ${HOME_DATA_DIR}/ref/ ${SCRATCH_DIR}/chime9_echi/ref/
    echo "✓ Reference audio copied to scratch"
else
    echo "✓ Reference audio already in scratch"
fi

# Check if enhanced outputs exist in home, copy to scratch if needed
HOME_ENHANCEMENT_DIR="${HOME_WORKING_DIR}/experiments/${EXP_NAME}/enhancement"
SCRATCH_ENHANCEMENT_DIR="${SCRATCH_DIR}/working_dir/experiments/${EXP_NAME}/enhancement"

if [ ! -d "${SCRATCH_ENHANCEMENT_DIR}" ]; then
    echo ""
    echo "Copying enhanced audio outputs to scratch..."
    mkdir -p $(dirname ${SCRATCH_ENHANCEMENT_DIR})

    if [ -d "${HOME_ENHANCEMENT_DIR}" ]; then
        time rsync -av --progress ${HOME_ENHANCEMENT_DIR}/ ${SCRATCH_ENHANCEMENT_DIR}/
        echo "✓ Enhanced outputs copied to scratch"
    else
        echo "ERROR: Enhanced outputs not found at ${HOME_ENHANCEMENT_DIR}"
        echo "Please run the enhancement script first (03_enhance_ha_trained.sh)"
        exit 1
    fi
else
    echo "✓ Enhanced outputs already in scratch"
fi

echo ""
echo "Running evaluation on enhanced audio..."
echo "Enhanced audio dir: ${SCRATCH_ENHANCEMENT_DIR}"
echo ""

# Run evaluation
# The evaluation pipeline has multiple steps: setup, validate, prepare, evaluate, report
python run_evaluation.py \
    device=ha \
    dataset=dev \
    paths.echi_dir=${SCRATCH_DIR}/chime9_echi \
    paths.root_dir=${SCRATCH_DIR}/working_dir \
    shared.exp_name=${EXP_NAME} \
    evaluate.use_gpu=true

echo ""
echo "=========================================="
echo "Evaluation completed: $(date)"
echo ""

# Copy evaluation results back to home
SCRATCH_EVAL_DIR="${SCRATCH_DIR}/working_dir/experiments/${EXP_NAME}/evaluation"
HOME_EVAL_DIR="${HOME_WORKING_DIR}/experiments/${EXP_NAME}/evaluation"

if [ -d "${SCRATCH_EVAL_DIR}" ]; then
    echo "Copying evaluation results from scratch to home..."
    mkdir -p $(dirname ${HOME_EVAL_DIR})
    time rsync -av --progress ${SCRATCH_EVAL_DIR}/ ${HOME_EVAL_DIR}/
    echo "✓ Evaluation results saved to: ${HOME_EVAL_DIR}"

    echo ""
    echo "Results summary:"
    find ${HOME_EVAL_DIR}/results -name "*.json" 2>/dev/null | while read f; do
        echo "---"
        echo "File: $f"
        head -20 "$f" 2>/dev/null
    done
else
    echo "WARNING: Evaluation directory not found in scratch"
fi

echo ""
echo "=========================================="
echo "Evaluation outputs location:"
echo "  Results: ${HOME_EVAL_DIR}/results/"
echo "  Reports: ${HOME_EVAL_DIR}/reports/"
echo "  Segments: ${HOME_EVAL_DIR}/segments/"
echo "=========================================="
