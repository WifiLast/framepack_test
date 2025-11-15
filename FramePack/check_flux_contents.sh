#!/bin/bash
# Check what's inside the flux_redux directory

MODEL_PATH="hf_download/hub/models--lllyasviel--flux_redux_bfl"

echo "=== Contents of $MODEL_PATH ==="
echo ""
ls -la "$MODEL_PATH"

echo ""
echo "=== Looking for model files recursively ==="
find "$MODEL_PATH" -type f -name "*.json" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.pth"

echo ""
echo "=== Full directory tree ==="
tree -L 3 "$MODEL_PATH" 2>/dev/null || find "$MODEL_PATH" -type f | head -20
