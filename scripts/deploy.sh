#!/usr/bin/env bash
# Deploy tensors to junkpile
# Usage: ./scripts/deploy.sh

set -e

REMOTE="chi@junkpile"
REMOTE_DIR="/opt/tensors/app"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Syncing Python code to junkpile..."
rsync -av --delete \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='node_modules' \
  --exclude='.ruff_cache' \
  --exclude='.mypy_cache' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --rsync-path="sudo rsync" \
  "$LOCAL_DIR/tensors/" "$REMOTE:$REMOTE_DIR/tensors/"

echo ""
echo "==> Fixing permissions..."
ssh "$REMOTE" "sudo chown -R tensors:tensors $REMOTE_DIR && sudo chmod -R g+w $REMOTE_DIR"

echo ""
echo "==> Restarting tensors service..."
ssh "$REMOTE" "sudo systemctl restart tensors"

echo ""
echo "==> Waiting for tensors to start..."
sleep 2

echo ""
echo "==> Verifying tensors service..."
SERVICE_STATUS=$(ssh "$REMOTE" "systemctl is-active tensors" 2>/dev/null)
if [ "$SERVICE_STATUS" = "active" ]; then
  echo "✓ tensors service running"
else
  echo "✗ tensors service not running"
  ssh "$REMOTE" "journalctl -u tensors -n 10 --no-pager"
  exit 1
fi

echo ""
echo "==> Deploy complete!"
echo "    API: http://junkpile:51200"
