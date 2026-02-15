#!/usr/bin/env bash
# Convert a safetensors model to q8_0 GGUF format
# Usage: ./scripts/convert.sh <input.safetensors>
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input.safetensors> [quantization]"
    echo "  quantization: f32, f16, q4_0, q5_0, q8_0 (default: q8_0)"
    exit 1
fi

INPUT="$1"
QUANT="${2:-q8_0}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: File not found: $INPUT"
    exit 1
fi

# Derive output filename: model.safetensors -> model-q8_0.gguf
BASENAME=$(basename "$INPUT" .safetensors)
DIRNAME=$(dirname "$INPUT")
OUTPUT="${DIRNAME}/${BASENAME}-${QUANT}.gguf"

echo "==> Converting: $INPUT"
echo "    Output: $OUTPUT"
echo "    Quantization: $QUANT"
echo ""

sd -M convert -m "$INPUT" -o "$OUTPUT" --type "$QUANT"

echo ""
echo "==> Done: $OUTPUT"
ls -lh "$OUTPUT"
