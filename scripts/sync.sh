#!/usr/bin/env bash
# One-way sync of the working tree to a GPU host (code lives locally, runs remotely).
# Point REMOTE at your own host; nothing is synced unless you do.
set -euo pipefail
REMOTE="${REMOTE:?usage: REMOTE=user@gpuhost [REMOTE_DIR=~/rewire] $0}"
DEST="${REMOTE_DIR:-~/rewire}"
rsync -av --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.zcodeignore' \
  ./ "$REMOTE:$DEST/"
echo "synced to $REMOTE:$DEST"
