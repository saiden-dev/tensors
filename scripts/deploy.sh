#!/usr/bin/env bash
# Deploy tensors to junkpile
# Usage: ./scripts/deploy.sh

set -e

REMOTE="chi@junkpile"
REMOTE_DIR="~/Projects/tensors"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Building UI..."
cd "$LOCAL_DIR/tensors/server/ui"
npm run build

echo ""
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
  "$LOCAL_DIR/tensors/" "$REMOTE:$REMOTE_DIR/tensors/"

echo ""
echo "==> Restarting tensors service..."
ssh "$REMOTE" "sudo systemctl restart tensors"

echo ""
echo "==> Waiting for tensors to start..."
sleep 2

echo ""
echo "==> Verifying tensors API..."
TENSORS_STATUS=$(ssh "$REMOTE" "curl -s localhost:8081/api/models/status" 2>/dev/null)
if echo "$TENSORS_STATUS" | grep -q '"active":true'; then
  echo "✓ tensors API responding"
  echo "  Current model: $(echo "$TENSORS_STATUS" | jq -r '.current_model' | xargs basename)"
else
  echo "✗ tensors API not responding"
  echo "$TENSORS_STATUS"
  exit 1
fi

echo ""
echo "==> Verifying sd-server..."
SD_STATUS=$(ssh "$REMOTE" "curl -s localhost:1234/sdapi/v1/sd-models" 2>/dev/null)
if echo "$SD_STATUS" | grep -q 'model_name'; then
  echo "✓ sd-server responding"
  echo "  Models available: $(echo "$SD_STATUS" | jq length)"
else
  echo "✗ sd-server not responding"
  exit 1
fi

echo ""
echo "==> Verifying external access..."
EXT_STATUS=$(curl -s -H "X-API-Key: v00YKDdHzLmwTLUJ07iMn4umLvcsKa9i" https://sd-api.saiden.dev/sdapi/v1/sd-models 2>/dev/null)
if echo "$EXT_STATUS" | grep -q 'model_name'; then
  echo "✓ sd-api.saiden.dev responding"
else
  echo "✗ sd-api.saiden.dev not responding"
  exit 1
fi

echo ""
echo "==> Deploy complete!"
echo "    UI: https://tensors.saiden.dev"
echo "    API: https://sd-api.saiden.dev"
