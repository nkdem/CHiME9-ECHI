#!/bin/bash

# Rsync to mlp server while respecting .gitignore patterns

# Default remote path (can be overridden with argument)
REMOTE_PATH="${1:-~/CHiME9-ECHI}"

# Get the current directory name
LOCAL_DIR="$(pwd)"

echo "Syncing $LOCAL_DIR to mlp:$REMOTE_PATH"
echo "Excluding patterns from .gitignore..."

# Rsync with:
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress during transfer
# -P: show progress and allow resuming
# --delete: delete files on remote that don't exist locally
# --filter: use git ls-files to only sync tracked files and respect .gitignore
rsync -avzP \
  --delete \
  --exclude='.git/' \
  --filter=':- .gitignore' \
  ./ mlp:"$REMOTE_PATH"

echo "Sync complete!"
