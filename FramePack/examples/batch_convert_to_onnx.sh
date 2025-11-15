#!/bin/bash
#
# Batch ONNX Conversion Script
#
# This script converts multiple models to ONNX format in sequence.
# Edit the MODEL_CONFIGS array below to add your models.
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEPACK_DIR="$(dirname "$SCRIPT_DIR")"
HF_CACHE="../hf_download/hub"
OUTPUT_DIR="$FRAMEPACK_DIR/Cache/onnx_models"
DEVICE="cuda"  # Change to "cpu" if no GPU available

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Model configurations
# Format: "model_path|model_type|output_name|subfolder"
# Leave subfolder empty if not needed
declare -a MODEL_CONFIGS=(
    # Flux Redux example
    "$HF_CACHE/models--lllyasviel--flux_redux_bfl/|flux_redux|flux_redux.onnx|"

    # CLIP text encoder example
    "$HF_CACHE/models--openai--clip-vit-large-patch14/|clip_text|clip_vit_large.onnx|"

    # Add more models here following the same format
    # "$HF_CACHE/models--org--model/|model_type|output.onnx|subfolder"
)

# Function to convert a single model
convert_model() {
    local model_path=$1
    local model_type=$2
    local output_name=$3
    local subfolder=$4

    echo -e "${YELLOW}Converting: $output_name${NC}"
    echo "  Model path: $model_path"
    echo "  Model type: $model_type"

    # Build command
    local cmd="python $FRAMEPACK_DIR/convert_to_onnx.py \
        --model-path \"$model_path\" \
        --model-type \"$model_type\" \
        --output-dir \"$OUTPUT_DIR\" \
        --output-name \"$output_name\" \
        --device \"$DEVICE\""

    # Add subfolder if specified
    if [ -n "$subfolder" ]; then
        cmd="$cmd --subfolder \"$subfolder\""
    fi

    # Execute conversion
    if eval $cmd; then
        echo -e "${GREEN}✓ Successfully converted: $output_name${NC}\n"
        return 0
    else
        echo -e "${RED}✗ Failed to convert: $output_name${NC}\n"
        return 1
    fi
}

# Main conversion loop
echo "========================================="
echo "Batch ONNX Model Conversion"
echo "========================================="
echo "Device: $DEVICE"
echo "Output directory: $OUTPUT_DIR"
echo "Models to convert: ${#MODEL_CONFIGS[@]}"
echo "========================================="
echo ""

success_count=0
fail_count=0
failed_models=()

for config in "${MODEL_CONFIGS[@]}"; do
    # Parse configuration
    IFS='|' read -r model_path model_type output_name subfolder <<< "$config"

    # Check if model exists
    if [ ! -d "$model_path" ]; then
        echo -e "${YELLOW}⚠ Skipping $output_name: Model not found at $model_path${NC}\n"
        continue
    fi

    # Convert model
    if convert_model "$model_path" "$model_type" "$output_name" "$subfolder"; then
        ((success_count++))
    else
        ((fail_count++))
        failed_models+=("$output_name")
    fi
done

# Summary
echo "========================================="
echo "Conversion Summary"
echo "========================================="
echo -e "Total models: ${#MODEL_CONFIGS[@]}"
echo -e "${GREEN}Successful: $success_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
fi

echo "========================================="
echo ""
echo "ONNX models saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "No ONNX models found"

# Exit with error if any conversions failed
[ $fail_count -eq 0 ] && exit 0 || exit 1
