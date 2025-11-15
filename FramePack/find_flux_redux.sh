#!/bin/bash
# Diagnostic script to find the Flux Redux model

echo "=== Searching for Flux Redux model ==="
echo ""

# Check current directory
echo "Current directory: $(pwd)"
echo ""

# Check if hf_download exists
if [ -d "hf_download" ]; then
    echo "✓ hf_download directory exists"

    # Look for flux_redux model
    echo ""
    echo "Searching for flux_redux directories..."
    find hf_download -type d -name "*flux_redux*" 2>/dev/null

    echo ""
    echo "Checking HuggingFace cache structure..."
    if [ -d "hf_download/hub/models--lllyasviel--flux_redux_bfl" ]; then
        echo "✓ Found models--lllyasviel--flux_redux_bfl"
        echo ""
        echo "Contents:"
        ls -la hf_download/hub/models--lllyasviel--flux_redux_bfl/

        echo ""
        echo "Checking for snapshots subdirectory..."
        if [ -d "hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots" ]; then
            echo "✓ Found snapshots directory"
            echo ""
            echo "Snapshot directories:"
            ls -la hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots/

            # Find the actual model files
            echo ""
            echo "Looking for model files (config.json, *.safetensors, *.bin)..."
            find hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots -type f \( -name "config.json" -o -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) 2>/dev/null
        else
            echo "✗ No snapshots directory found"
        fi
    else
        echo "✗ models--lllyasviel--flux_redux_bfl not found"
        echo ""
        echo "Available models in hf_download/hub/:"
        ls -la hf_download/hub/ | grep "^d" | grep "models--"
    fi
else
    echo "✗ hf_download directory not found"
    echo ""
    echo "Searching for HuggingFace cache in common locations..."

    # Check common HF cache locations
    if [ -d "$HOME/.cache/huggingface/hub" ]; then
        echo "Found HuggingFace cache at: $HOME/.cache/huggingface/hub"
        ls -la "$HOME/.cache/huggingface/hub" | grep flux_redux
    fi
fi

echo ""
echo "=== Search complete ==="
