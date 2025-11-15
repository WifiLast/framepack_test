# ONNX Model Conversion Examples

This guide shows how to use the `convert_to_onnx.py` command-line tool to convert various PyTorch models to ONNX format.

## Prerequisites

```bash
pip install torch onnx transformers diffusers
```

## Basic Usage

```bash
python convert_to_onnx.py \
    --model-path <path-to-model> \
    --model-type <model-type> \
    [--output-dir <output-directory>] \
    [--output-name <output-filename>] \
    [--device cuda|cpu]
```

## Supported Model Types

- `flux_redux`: Flux Redux image encoder
- `clip_text`: CLIP text encoder
- `llama_text`: LLaMA text encoder
- `t5_text`: T5 text encoder
- `vae`: VAE (Variational Autoencoder)
- `unet`: UNet diffusion model

## Examples

### 1. Flux Redux Image Encoder

Convert the Flux Redux model to ONNX:

```bash
python convert_to_onnx.py \
    --model-path hf_download/hub/models--lllyasviel--flux_redux_bfl/ \
    --model-type flux_redux \
    --output-dir Cache/onnx_models/ \
    --output-name flux_redux.onnx
```

**What it does:**
- Loads the Flux Redux model from HuggingFace cache
- Generates sample inputs (image embeddings)
- Exports to ONNX with dynamic batch/sequence dimensions
- Saves to `Cache/onnx_models/flux_redux.onnx`

**Expected output:**
```
================================================================================
Converting flux_redux model to ONNX
================================================================================

Loading Flux Redux model from: hf_download/hub/models--lllyasviel--flux_redux_bfl/
  Model type: FluxReduxModel
  Config: {'hidden_size': 768, 'num_layers': 12}

Moving model to CUDA...
Generating sample inputs...
  Input shapes: [(1, 256, 768)]

Exporting to ONNX...
  Output: Cache/onnx_models/flux_redux.onnx
  This may take several minutes...

Successfully exported flux_redux to ONNX: Cache/onnx_models/flux_redux.onnx

================================================================================
SUCCESS: Model converted to ONNX
  Output: Cache/onnx_models/flux_redux.onnx
  Size: 145.32 MB
================================================================================
```

### 2. CLIP Text Encoder

Convert a CLIP text encoder:

```bash
python convert_to_onnx.py \
    --model-path hf_download/hub/models--openai--clip-vit-large-patch14/ \
    --model-type clip_text \
    --output-name clip_text_encoder.onnx
```

**Features:**
- Dynamic batch and sequence dimensions
- Exports both hidden states and pooler output
- Compatible with TensorRT compilation

### 3. LLaMA Text Encoder

Convert a LLaMA-based text encoder:

```bash
python convert_to_onnx.py \
    --model-path path/to/llama/model/ \
    --model-type llama_text \
    --device cuda
```

### 4. Stable Diffusion VAE

Convert a VAE from a Stable Diffusion model:

```bash
python convert_to_onnx.py \
    --model-path hf_download/hub/models--stabilityai--stable-diffusion-xl-base-1.0/ \
    --model-type vae \
    --subfolder vae \
    --output-name sdxl_vae.onnx
```

**Note:** Use `--subfolder` when the model is part of a larger pipeline.

### 5. HunyuanVideo Models

Convert HunyuanVideo text encoders:

```bash
# LLaMA text encoder
python convert_to_onnx.py \
    --model-path tencent/HunyuanVideo \
    --model-type llama_text \
    --subfolder text_encoder \
    --output-name hunyuan_text_encoder.onnx

# CLIP text encoder
python convert_to_onnx.py \
    --model-path tencent/HunyuanVideo \
    --model-type clip_text \
    --subfolder text_encoder_2 \
    --output-name hunyuan_clip_encoder.onnx
```

## Advanced Options

### Custom Output Directory

Save ONNX files to a specific directory:

```bash
python convert_to_onnx.py \
    --model-path <model-path> \
    --model-type <type> \
    --output-dir /path/to/custom/onnx/dir/
```

### CPU-only Conversion

Use CPU if CUDA is not available:

```bash
python convert_to_onnx.py \
    --model-path <model-path> \
    --model-type <type> \
    --device cpu
```

**Note:** CPU conversion is slower but works without GPU.

### Custom ONNX Opset Version

Specify a different ONNX opset version:

```bash
python convert_to_onnx.py \
    --model-path <model-path> \
    --model-type <type> \
    --opset-version 18
```

## Batch Conversion Script

Create a script to convert multiple models:

```bash
#!/bin/bash
# convert_all.sh

MODELS_DIR="hf_download/hub"
OUTPUT_DIR="Cache/onnx_models"

# Convert Flux Redux
python convert_to_onnx.py \
    --model-path "$MODELS_DIR/models--lllyasviel--flux_redux_bfl/" \
    --model-type flux_redux \
    --output-dir "$OUTPUT_DIR" \
    --output-name flux_redux.onnx

# Convert CLIP text encoder
python convert_to_onnx.py \
    --model-path "$MODELS_DIR/models--openai--clip-vit-large-patch14/" \
    --model-type clip_text \
    --output-dir "$OUTPUT_DIR" \
    --output-name clip_vit_large.onnx

echo "All conversions complete!"
```

Make it executable and run:

```bash
chmod +x convert_all.sh
./convert_all.sh
```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

1. Use CPU instead:
   ```bash
   --device cpu
   ```

2. Clear CUDA cache before running:
   ```python
   python -c "import torch; torch.cuda.empty_cache()"
   ```

### Model Not Found

Ensure the model path is correct:

```bash
# Check if model exists
ls -la hf_download/hub/models--<org>--<model>/

# Or use absolute path
--model-path /full/path/to/model/
```

### Conversion Fails

Common issues:

1. **Model has custom code**: Add `trust_remote_code=True` in the loader
2. **Incompatible operations**: Some PyTorch ops don't export to ONNX
3. **Dynamic shapes**: Use static shapes for problematic models

### Verify ONNX Model

After conversion, verify the ONNX model:

```python
import onnx

# Load ONNX model
onnx_model = onnx.load("Cache/onnx_models/flux_redux.onnx")

# Check model
onnx.checker.check_model(onnx_model)

# Print model info
print("Inputs:")
for input in onnx_model.graph.input:
    print(f"  {input.name}: {input.type}")

print("\nOutputs:")
for output in onnx_model.graph.output:
    print(f"  {output.name}: {output.type}")
```

## Using ONNX Models

### With ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession("Cache/onnx_models/flux_redux.onnx")

# Prepare inputs
inputs = {
    "image_embeddings": np.random.randn(1, 256, 768).astype(np.float16)
}

# Run inference
outputs = session.run(None, inputs)
print(f"Output shape: {outputs[0].shape}")
```

### With TensorRT

```python
import tensorrt as trt

# Build TensorRT engine from ONNX
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX file
with open("Cache/onnx_models/flux_redux.onnx", "rb") as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
engine = builder.build_serialized_network(network, config)
```

## Integration with FramePack

The converted ONNX models can be used with FramePack's TensorRT runtime:

```python
from diffusers_helper.tensorrt_runtime import TensorRTRuntime

# Initialize TensorRT runtime
runtime = TensorRTRuntime(
    enabled=True,
    precision=torch.float16,
    workspace_size_mb=4096,
)

# The runtime will automatically find and use ONNX models from cache
# when compiling TensorRT engines
```

## Model-Specific Notes

### Flux Redux

- **Input**: Image embeddings (batch, sequence, hidden_size)
- **Output**: Processed embeddings
- **Typical size**: 100-200 MB
- **Use case**: Image conditioning for diffusion models

### CLIP Text Encoder

- **Input**: Token IDs and attention mask
- **Output**: Hidden states and pooler output
- **Typical size**: 500-600 MB
- **Use case**: Text conditioning for image generation

### VAE

- **Input**: Images (for encoder) or latents (for decoder)
- **Output**: Latents (encoder) or images (decoder)
- **Typical size**: 300-400 MB
- **Use case**: Latent space compression/decompression

## Performance Tips

1. **Use FP16**: Models are automatically converted to float16 for smaller size and faster inference

2. **Dynamic Shapes**: The tool uses dynamic axes for batch/sequence dimensions where appropriate

3. **Opset Version**: Use opset 17+ for best compatibility with modern ONNX runtimes

4. **Caching**: Converted models are cached in `Cache/onnx_models/` by default

5. **GPU**: Always use CUDA for conversion when available (much faster)

## See Also

- [TENSORRT_ACCELERATION.md](TENSORRT_ACCELERATION.md) - Guide to TensorRT acceleration in FramePack
- [onnx_converter.py](diffusers_helper/onnx_converter.py) - Python API for ONNX conversion
- [convert_to_onnx.py](convert_to_onnx.py) - Command-line tool source code
