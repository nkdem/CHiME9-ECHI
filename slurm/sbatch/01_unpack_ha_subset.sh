#!/bin/bash
#SBATCH --job-name=unpack_ha_subset
#SBATCH --partition=PGR-Standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/unpack/%j_unpack_ha.out
#SBATCH --mail-user=s2203859@ed.ac.uk
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CHiME9 ECHI - Unpacking HA Subset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Environment setup with UV
echo "Setting up UV virtual environment..."
cd $HOME/CHiME9-ECHI

# Activate UV virtual environment
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Paths
USER="s2203859"
SOURCE_DATA="$HOME/chime9_echi"
SCRATCH_DIR="/disk/scratch/${USER}/CHiME9-ECHI"
PROCESSED_DIR="${SCRATCH_DIR}/processed"

# Step 1: Copy raw data to scratch (if not already there)
if [ ! -d "$SCRATCH_DIR/chime9_echi" ]; then
    echo "[1/3] Copying raw dataset to scratch..."
    mkdir -p $SCRATCH_DIR
    time rsync -av --progress ${SOURCE_DATA}/ ${SCRATCH_DIR}/chime9_echi/
    echo "Copy completed: $(date)"
else
    echo "[1/3] Raw data already in scratch, skipping copy"
fi

# Step 2: Create temporary config pointing to scratch
echo ""
echo "[2/3] Creating temporary config for scratch paths..."
cat > /tmp/scratch_paths_${SLURM_JOB_ID}.yaml << EOF
# Temporary override for scratch storage
defaults:
  - override /paths: paths

paths:
  echi_dir: ${SCRATCH_DIR}/chime9_echi
  root_dir: ${PROCESSED_DIR}
EOF

# Step 3: Unpack HA training data (SUBSET ONLY for testing)
echo ""
echo "[3/3] Unpacking HA training segments (SUBSET: 5 sessions)..."
python scripts/train/unpack.py \
    device=ha \
    dataset=train \
    max_sessions=5 \
    --config-path /tmp \
    --config-name scratch_paths_${SLURM_JOB_ID}

echo ""
echo "=========================================="
echo "Unpacking completed: $(date)"
echo "Processed data location: ${PROCESSED_DIR}"
echo ""
echo "Directory contents:"
ls -lh ${PROCESSED_DIR}/train_segments/train/ha/ 2>/dev/null | head -20
echo ""
echo "Total size in scratch:"
du -sh ${PROCESSED_DIR}
echo "=========================================="

# Step 4: Copy processed data back to home directory
HOME_PROCESSED_DIR="$HOME/CHiME9-ECHI/data/processed_ha_subset"
echo ""
echo "[4/4] Copying processed data to home directory..."
echo "Destination: ${HOME_PROCESSED_DIR}"

mkdir -p ${HOME_PROCESSED_DIR}
time rsync -av --progress ${PROCESSED_DIR}/ ${HOME_PROCESSED_DIR}/

echo ""
echo "✅ Processed data copied to home:"
echo "   ${HOME_PROCESSED_DIR}"
echo ""
echo "Total size in home:"
du -sh ${HOME_PROCESSED_DIR}
echo ""
echo "You can now use this data even after scratch is cleaned:"
echo "   ${HOME_PROCESSED_DIR}"
echo "=========================================="

# Cleanup temp config
rm -f /tmp/scratch_paths_${SLURM_JOB_ID}.yaml
