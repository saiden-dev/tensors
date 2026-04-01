#!/bin/sh
set -e

DATA="${DATA_DIR:-/data}"

# Create directory structure for model storage
mkdir -p "$DATA/models/checkpoints"
mkdir -p "$DATA/models/loras"
mkdir -p "$DATA/models/embeddings"
mkdir -p "$DATA/models/vae"
mkdir -p "$DATA/models/controlnet"
mkdir -p "$DATA/models/upscalers"
mkdir -p "$DATA/output"
mkdir -p "$DATA/gallery"
mkdir -p "$DATA/db"

# Write tensors config pointing to addon storage paths
mkdir -p /app/config/tensors
cat > /app/config/tensors/config.toml <<CONF
[paths]
models_dir = "$DATA/models"
checkpoints = "$DATA/models/checkpoints"
loras = "$DATA/models/loras"
vae = "$DATA/models/vae"
embeddings = "$DATA/models/embeddings"
controlnet = "$DATA/models/controlnet"
upscalers = "$DATA/models/upscalers"

[comfyui]
url = "${COMFYUI_URL:-http://127.0.0.1:8188}"
CONF

export XDG_CONFIG_HOME=/app/config
export XDG_DATA_HOME="$DATA"

echo "tensors starting"
echo "  DATA_DIR=$DATA"
echo "  COMFYUI_URL=${COMFYUI_URL:-not set}"
echo "  DB=$DATA/db/models.db"

exec tsr serve --host 0.0.0.0 --port 5000
