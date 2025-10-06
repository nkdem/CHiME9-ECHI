#!/bin/bash
# Setup script for MLP cluster - run this ONCE before submitting jobs

set -e

echo "🚀 Setting up CHiME9-ECHI on MLP cluster with UV..."

# Navigate to project directory
cd $HOME/CHiME9-ECHI

if ! command -v uv &> /dev/null; then
    echo "You need to install UV first."
    exit 1
else
    echo "✅ UV already installed: $(uv --version)"
fi

# Remove existing venv if present
if [ -d ".venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.11
echo "📦 Creating virtual environment with Python 3.11..."
uv venv --python 3.11

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install build dependencies
echo "📦 Installing build dependencies..."
uv pip install "setuptools<72" wheel pip

# Install git dependencies
echo "📥 Installing git dependencies..."
uv pip install git+https://github.com/wavlab-speech/versa
uv pip install git+https://github.com/ftshijt/pysepm.git

# Install project in editable mode
echo "📥 Installing project and dependencies..."
python -m pip install -e . --no-build-isolation

# Set PYTHONPATH in shell config
SHELL_RC="$HOME/.bashrc"
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "export PYTHONPATH.*CHiME9-ECHI/src" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# CHiME9-ECHI PYTHONPATH" >> "$SHELL_RC"
    echo "export PYTHONPATH=\"$HOME/CHiME9-ECHI/src:\$PYTHONPATH\"" >> "$SHELL_RC"
    echo "✅ Added PYTHONPATH to $SHELL_RC"
fi

# Run NISQA setup if available
if [ -d "external/versa/tools" ]; then
    echo "🔧 Setting up NISQA..."
    (cd external/versa/tools && bash setup_nisqa.sh)
fi

# Create necessary directories
echo "📁 Creating log directories..."
mkdir -p slurm/logs/unpack
mkdir -p slurm/logs/train
mkdir -p slurm/logs/enhance

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Update config/paths.yaml to point to cluster data:"
echo "      echi_dir: /home/s2203859/chime9_echi"
echo ""
echo "   2. Test the installation:"
echo "      source .venv/bin/activate"
echo "      export PYTHONPATH=\"\$HOME/CHiME9-ECHI/src:\$PYTHONPATH\""
echo "      python -c 'import torch; import torchaudio; import hydra; print(\"✅ All imports OK\")'"
echo ""
echo "   3. Submit the pipeline:"
echo "      ./slurm/submit_ha_pipeline.sh"
echo ""
