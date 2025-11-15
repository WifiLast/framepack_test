# ONNX Conversion Quick Start

Quick reference for converting models to ONNX format.

## Quick Examples

### Flux Redux (Your Use Case)

```bash
cd FramePack

python convert_to_onnx.py \
    --model-path ../hf_download/hub/models--lllyasviel--flux_redux_bfl/ \
    --model-type flux_redux \
    --output-name flux_redux.onnx
```

### Other Common Models

```bash
# CLIP text encoder
python convert_to_onnx.py \
    --model-path path/to/clip/model \
    --model-type clip_text

# LLaMA text encoder
python convert_to_onnx.py \
    --model-path path/to/llama/model \
    --model-type llama_text

# VAE from Stable Diffusion
python convert_to_onnx.py \
    --model-path path/to/sd/model \
    --model-type vae \
    --subfolder vae
```

## Command Structure

```
python convert_to_onnx.py \
    --model-path <PATH> \      # Path to model directory
    --model-type <TYPE> \      # Model type (see below)
    [--output-dir <DIR>] \     # Output directory (optional)
    [--output-name <NAME>] \   # Output filename (optional)
    [--device cuda|cpu] \      # Device to use (default: cuda)
    [--subfolder <NAME>]       # Subfolder for diffusers models
```

## Model Types

| Type | Description | Use Case |
|------|-------------|----------|
| `flux_redux` | Flux Redux encoder | Image conditioning |
| `clip_text` | CLIP text encoder | Text conditioning |
| `llama_text` | LLaMA text encoder | Text conditioning |
| `t5_text` | T5 text encoder | Text conditioning |
| `vae` | VAE encoder/decoder | Latent compression |
| `unet` | UNet diffusion model | Denoising |

## Output Location

**Default:** `FramePack/Cache/onnx_models/`

**Custom:** Use `--output-dir /path/to/directory`

## Common Issues

### GPU Memory

```bash
# Use CPU if GPU has insufficient memory
python convert_to_onnx.py ... --device cpu
```

### Model Not Found

```bash
# Check path is correct
ls -la path/to/model/

# Use absolute path if needed
--model-path /absolute/path/to/model
```

### Need Help?

```bash
# Show all options
python convert_to_onnx.py --help

# See detailed examples
cat ONNX_CONVERSION_EXAMPLES.md
```

## Verification

After conversion, verify the model:

```bash
# Check file exists
ls -lh Cache/onnx_models/*.onnx

# Verify with Python
python -c "import onnx; onnx.checker.check_model(onnx.load('Cache/onnx_models/flux_redux.onnx'))"
```

## Next Steps

1. **Test the ONNX model** with ONNX Runtime
2. **Build TensorRT engine** from the ONNX file
3. **Integrate with FramePack** for accelerated inference

See [ONNX_CONVERSION_EXAMPLES.md](ONNX_CONVERSION_EXAMPLES.md) for detailed examples and usage patterns.
