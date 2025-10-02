# Migration Guide: Conda to UV

This document explains the changes made to migrate from Conda to UV for dependency management.

## Why UV?

UV is a modern Python package manager that is:
- **Faster**: 10-100x faster than pip
- **Simpler**: No need for Conda
- **Deterministic**: Lockfile ensures reproducible builds
- **Lightweight**: Just Python, no heavy Conda installation

## What Changed?

### 1. Installation Script

**Old (Conda):**
```bash
./install.sh  # Uses conda and environment.yaml
```

**New (UV):**
```bash
./install_uv.sh  # Uses uv and pyproject.toml
```

### 2. Environment Activation

**Old (Conda):**
```bash
conda activate echi_recipe
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**New (UV):**
```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 3. Dependency Management

**Old:** `environment.yaml` (Conda format)
```yaml
name: echi_recipe
dependencies:
  - pytorch
  - numpy==1.26.4
  - pip:
      - auraloss
```

**New:** `pyproject.toml` (Python standard)
```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.26.0,<2.0.0",
    "auraloss>=0.4.0",
]
```

### 4. Data Path Configuration

**Changed:** `config/paths.yaml`

```yaml
# Old
echi_dir: data/chime9_echi

# New
echi_dir: /Volumes/SSD/Datasets/CHiME 9
```

Update this to point to your actual dataset location.

## Migration Steps

If you're migrating from an existing Conda setup:

### 1. Backup Your Environment (Optional)

```bash
conda env export > my_old_environment.yaml
```

### 2. Deactivate Conda

```bash
conda deactivate
```

### 3. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal or run:
```bash
source $HOME/.cargo/env
```

### 4. Run UV Installation

```bash
cd CHiME9-ECHI
./install_uv.sh
```

### 5. Activate Virtual Environment

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### 6. Update Data Path

Edit `config/paths.yaml`:
```yaml
echi_dir: /Volumes/SSD/Datasets/CHiME 9  # Your path
```

### 7. Test Installation

```bash
python -c "import torch; import torchaudio; import hydra; print('✅ Success!')"
```

### 8. (Optional) Remove Conda Environment

Once you've verified UV works:

```bash
conda env remove -n echi_recipe
```

## Key Differences

| Aspect | Conda | UV |
|--------|-------|-----|
| Environment location | `~/anaconda3/envs/echi_recipe` | `.venv/` in project |
| Activation | `conda activate echi_recipe` | `source .venv/bin/activate` |
| Install packages | `conda install` or `pip install` | `uv pip install` |
| Config file | `environment.yaml` | `pyproject.toml` |
| Lock file | N/A (or conda-lock) | `uv.lock` |
| Speed | Slower (resolves each time) | Faster (cached, optimized) |

## Adding New Dependencies

### With UV

```bash
# Add to pyproject.toml dependencies list
# Then run:
uv sync

# Or install directly:
uv pip install <package>
```

### Example: Add a New Package

Edit `pyproject.toml`:
```toml
dependencies = [
    # ... existing packages ...
    "my-new-package>=1.0.0",
]
```

Then:
```bash
uv sync
```

## Troubleshooting

### UV Not Found

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Import Errors

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Make sure PYTHONPATH is set
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Reinstall if needed
uv sync
```

### GPU/CUDA Issues

UV installs PyTorch from PyPI. For specific CUDA versions:

```bash
# Check current PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Install specific CUDA version if needed
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Path Not Found

Make sure `config/paths.yaml` has the correct path:
```bash
# Check if path exists
ls "/Volumes/SSD/Datasets/CHiME 9"

# Update config/paths.yaml if needed
```

### Versa/NISQA Setup Failed

The NISQA setup runs after Versa installation. If it fails:

```bash
source .venv/bin/activate
cd external/versa/tools
bash setup_nisqa.sh
cd ../../..
```

## Performance Comparison

Based on typical installations:

| Task | Conda | UV | Speedup |
|------|-------|-----|---------|
| Initial install | ~10 min | ~2 min | 5x |
| Add package | ~1 min | ~5 sec | 12x |
| Resolve dependencies | ~30 sec | ~1 sec | 30x |

## Reverting to Conda

If you need to go back to Conda:

```bash
# Remove UV virtual environment
rm -rf .venv

# Reinstall with Conda
./install.sh
conda activate echi_recipe
```

## Project Structure (Updated)

```
CHiME9-ECHI/
├── .venv/                      # NEW: Virtual environment (UV)
├── config/
│   └── paths.yaml              # UPDATED: Data path changed
├── pyproject.toml              # UPDATED: Full dependency list
├── uv.lock                     # NEW: Dependency lock file
├── install_uv.sh               # NEW: UV installation script
├── install.sh                  # OLD: Conda installation (legacy)
├── environment.yaml            # OLD: Conda deps (legacy, kept for reference)
└── MIGRATION_GUIDE.md          # NEW: This file
```

## Next Steps

1. ✅ Install UV and create virtual environment
2. ✅ Update `config/paths.yaml` with your data location
3. ✅ Test installation with imports
4. ✅ Try running a simple command (e.g., `python run_enhancement.py --help`)
5. ✅ Review `REPOSITORY_GUIDE.md` for detailed usage instructions

## Questions?

- UV Documentation: https://docs.astral.sh/uv/
- UV GitHub: https://github.com/astral-sh/uv
- CHiME Challenge: https://www.chimechallenge.org/current/task2/

---

**Happy coding with UV! 🚀**