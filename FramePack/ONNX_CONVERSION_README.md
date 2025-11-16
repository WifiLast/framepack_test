# ONNX Conversion - Important Information

## ⚠️ Critical Warning: ONNX Conversion is ONE-WAY

**You CANNOT convert ONNX files back to working Stable Diffusion checkpoints.**

### Why the Conversion is One-Way

When converting from PyTorch/safetensors to ONNX:

1. **Weight Names Change**
   - PyTorch: `model.diffusion_model.input_blocks.0.0.weight`
   - ONNX: Graph node names like `/input_blocks.0/Conv_output_0`
   - These are fundamentally different naming schemes

2. **Structure Changes**
   - ONNX optimizes the computation graph
   - Operations may be fused or reorganized
   - Control flow is simplified
   - Some metadata is lost

3. **Components Are Separate**
   - Original checkpoint: Single file with UNet + VAE + CLIP (~4GB)
   - ONNX export: Separate files for each component
   - No standard way to merge them back into one checkpoint

4. **Missing Metadata**
   - Configuration settings
   - Training parameters
   - Model version information
   - Custom pipeline code

### What Happens When You Try to Convert Back

```bash
# This extracts ONNX weights to safetensors
python convert_to_safetensor.py unet_model.onnx unet_model.safetensors
```

**Result:**
- ✗ File is much smaller (only one component, not full checkpoint)
- ✗ Weight names don't match PyTorch state dict format
- ✗ Cannot be loaded by SD WebUI or ComfyUI
- ✗ Error: "Failed to load CLIPTextModel. Weights for this component appear to be missing"

## ✓ Correct Usage of ONNX Files

### Option 1: Use ONNX Runtime (Recommended)

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("unet_model.onnx")

# Prepare inputs
inputs = {
    "sample": sample_array,  # shape: [1, 4, 64, 64]
    "timestep": np.array([500.0], dtype=np.float32),
    "encoder_hidden_states": text_embeddings  # shape: [1, 77, 768]
}

# Run inference
outputs = session.run(None, inputs)
```

### Option 2: Convert to TensorRT

```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=unet_model.onnx \\
        --saveEngine=unet_model.trt \\
        --fp16 \\
        --workspace=4096
```

### Option 3: Keep Original Checkpoint

**This is what you should do:**

1. **Keep your original `.safetensors` checkpoint** - This is your source of truth
2. **Use ONNX files only for optimized inference**
3. **Use the original checkpoint for:**
   - SD WebUI / Automatic1111
   - ComfyUI
   - Training/fine-tuning
   - Merging with other models
   - LoRA application

## File Size Comparison

| Format | Size | Components | Use Case |
|--------|------|------------|----------|
| Original .safetensors | ~4GB | All (UNet, VAE, CLIP) | SD WebUI, editing |
| ONNX (all components) | ~4GB total | Separate files | Optimized inference |
| Converted back .safetensors | ~1.7GB | Only UNet weights | **BROKEN - Don't use!** |

## Common Errors and Solutions

### Error: "Failed to load CLIPTextModel"

```
Failed to load CLIPTextModel. Weights for this component appear to be missing in the checkpoint.
```

**Cause:** You're trying to load an incomplete checkpoint created from a single ONNX component.

**Solution:** Use your original checkpoint file, not the converted ONNX->safetensors file.

### Error: "is not a complete model"

```
ERROR Load model: file="model.safetensors" is not a complete model
```

**Cause:** The safetensors file only contains one component's weights with wrong naming.

**Solution:** Use the original checkpoint or load ONNX files with ONNX Runtime.

## Workflow Recommendations

### ✓ Correct Workflow

```
Original Checkpoint (.safetensors)
    ├─> Use with SD WebUI/ComfyUI
    ├─> Export to ONNX for optimization
    │   └─> Use ONNX with ONNX Runtime/TensorRT
    └─> Keep as source of truth (NEVER DELETE)
```

### ✗ Incorrect Workflow

```
Original Checkpoint (.safetensors)
    └─> Export to ONNX
        └─> Convert back to safetensors  ❌ BROKEN
            └─> Try to load in SD WebUI  ❌ FAILS
```

## How to Export to ONNX Correctly

```bash
# Export all components (auto-detected)
python convert_to_onnx.py \\
    --model-path your_model.safetensors \\
    --output-dir onnx_output/ \\
    --opset-version 14

# This creates:
# - unet_your_model.onnx
# - vae_decoder_your_model.onnx
# - vae_encoder_your_model.onnx
# - clip_text_your_model.onnx
```

**Keep the original `your_model.safetensors` file!**

## Summary

| Action | Status | Explanation |
|--------|--------|-------------|
| Checkpoint → ONNX | ✓ Supported | For optimization |
| ONNX → Checkpoint | ✗ Not Possible | Structure/naming incompatible |
| Use ONNX for inference | ✓ Recommended | With ONNX Runtime/TensorRT |
| Use ONNX in SD WebUI | ✗ Not Supported | Need original checkpoint |
| Keep original checkpoint | ✓ Required | Always keep as backup |

## Questions?

**Q: I lost my original checkpoint, can I recover it from ONNX?**
A: No. You need to re-download the original model.

**Q: Why is the converted safetensors so much smaller?**
A: It only contains one component (UNet) with modified weight names, not a complete checkpoint.

**Q: Can I use ONNX files in Automatic1111?**
A: No. SD WebUI requires PyTorch checkpoints. Use ONNX files with ONNX Runtime instead.

**Q: What's the point of ONNX export then?**
A: ONNX is for optimized inference with ONNX Runtime or TensorRT, which can be significantly faster than PyTorch.

## Support

For issues or questions:
1. Check this README first
2. Ensure you're using the original checkpoint for SD WebUI
3. Use ONNX files only for ONNX Runtime/TensorRT inference
