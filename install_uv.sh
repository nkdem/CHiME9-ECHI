#!/bin/bash

# Exit immediately on any error
set -e

echo "🚀 Installing CHiME9-ECHI with UV..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ UV found: $(uv --version)"

# Remove existing venv if it exists (to ensure clean Python 3.11 install)
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

# Install build dependencies first (use setuptools < 72 for build_backend compatibility)
echo "📦 Installing build dependencies..."
uv pip install "setuptools<72" wheel pip

# Install git dependencies first (pip doesn't understand [tool.uv.sources])
echo "📥 Installing git dependencies..."
uv pip install git+https://github.com/wavlab-speech/versa
uv pip install git+https://github.com/ftshijt/pysepm.git

# Install the project in editable mode using regular pip (no build isolation)
echo "📥 Installing project and dependencies..."
python -m pip install -e . --no-build-isolation

# Set PYTHONPATH
echo "🔧 Setting PYTHONPATH..."
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Add to shell config if not already there
SHELL_RC="$HOME/.bashrc"
if [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "export PYTHONPATH.*CHiME9-ECHI/src" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# CHiME9-ECHI PYTHONPATH" >> "$SHELL_RC"
    echo "export PYTHONPATH=\"$PWD/src:\$PYTHONPATH\"" >> "$SHELL_RC"
    echo "✅ Added PYTHONPATH to $SHELL_RC"
else
    echo "✅ PYTHONPATH already in $SHELL_RC"
fi

# Run NISQA setup (after versa is installed)
if [ -d "external/versa/tools" ]; then
    echo "🔧 Setting up NISQA..."
    (
        cd external/versa/tools || exit 1
        bash setup_nisqa.sh
    )
else
    echo "⚠️  NISQA setup script not found at external/versa/tools/setup_nisqa.sh"
    echo "   This is expected if versa hasn't created the external directory yet."
    echo "   You may need to run this manually after versa installation completes."
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment:"
echo "      source .venv/bin/activate"
echo ""
echo "   2. Set PYTHONPATH (or restart your shell):"
echo "      export PYTHONPATH=\"$PWD/src:\$PYTHONPATH\""
echo ""
echo "   3. Verify your data path in config/paths.yaml:"
echo "      echi_dir: /Volumes/SSD/Datasets/CHiME 9"
echo ""
echo "   4. Test the installation:"
echo "      python -c 'import torch; import torchaudio; import hydra; print(\"✅ All imports OK\")'"
echo ""