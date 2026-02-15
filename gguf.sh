#!/usr/bin/env bash
# Convert all safetensors to q8_0 GGUF format (skip if already exists)

set -euo pipefail

for sf in *.safetensors; do
    [[ -f "$sf" ]] || continue

    base="${sf%.safetensors}"
    gguf="${base}-q8_0.gguf"

    if [[ -f "$gguf" ]]; then
        echo "Skip: $gguf exists"
        continue
    fi

    echo "Converting: $sf -> $gguf"
    sd -M convert -m "$sf" -o "$gguf" --type q8_0
done
