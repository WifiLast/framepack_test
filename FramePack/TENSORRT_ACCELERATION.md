# TensorRT Acceleration in FramePack

This document explains TensorRT acceleration for models in FramePack.

## Overview

FramePack uses modern TensorRT acceleration via `torch_tensorrt.dynamo.compile`, which works **directly with PyTorch models** without requiring ONNX conversion.

When using TensorRT flags (`--enable-tensorrt`, `--tensorrt-text-encoders`, `--tensorrt-transformer`), the system automatically compiles TensorRT engines from PyTorch models on first use.

## How It Works

1. **Model Loading**: Models are loaded normally as PyTorch modules
2. **TensorRT Wrapping**: Models are wrapped with TensorRT runtime classes
3. **Just-In-Time Compilation**: On first inference, TensorRT compiles optimized engines
4. **Engine Caching**: Compiled engines are cached for subsequent runs

## Supported Models

The following models support TensorRT acceleration:

### VAE (`--enable-tensorrt`)
- **VAE Encoder**: TensorRT compilation for latent encoding
- **VAE Decoder**: TensorRT compilation for latent decoding
- Dynamic shape caching per resolution

### Text Encoders (`--tensorrt-text-encoders`)
- **LLaMA Text Encoder**: TensorRT compilation with dynamic batch/sequence
- **CLIP Text Encoder**: TensorRT compilation with dynamic batch
- Requires `--enable-tensorrt` to be enabled first

### Transformer (`--tensorrt-transformer`)
- **HunyuanVideo Transformer**: TensorRT compilation with shape-based caching
- Requires `--enable-tensorrt` to be enabled first
- Note: Incompatible with BitsAndBytes, CPU offload, or FSDP

## Cache Location

TensorRT compiled engines are cached to avoid re-compilation:

- **Default Location**: `FramePack/Cache/tensorrt_engines/`
- **Custom Location**: Set environment variable `FRAMEPACK_TENSORRT_CACHE_DIR`

Example:
```bash
export FRAMEPACK_TENSORRT_CACHE_DIR=/path/to/custom/cache
```

## Usage

### Basic TensorRT with VAE only
```bash
python demo_gradio.py --enable-tensorrt
```

### TensorRT with Text Encoders
```bash
python demo_gradio.py --enable-tensorrt --tensorrt-text-encoders
```
- First inference: TensorRT compiles text encoder engines (5-10 minutes)
- Subsequent runs: Loads cached engines (fast startup)

### TensorRT with Transformer
```bash
python demo_gradio.py --enable-tensorrt --tensorrt-transformer
```
- First inference: TensorRT compiles transformer engine (10-20 minutes)
- Requires ~16GB GPU VRAM during compilation
- Subsequent runs: Uses cached engine

### All TensorRT Features
```bash
python demo_gradio.py --enable-tensorrt --tensorrt-text-encoders --tensorrt-transformer
```

## Implementation Details

### Module: `diffusers_helper/tensorrt_runtime.py`

Key classes:

- `TensorRTRuntime`: Core runtime managing compilation and caching
- `TensorRTLatentDecoder`: TensorRT wrapper for VAE decoder
- `TensorRTLatentEncoder`: TensorRT wrapper for VAE encoder
- `TensorRTTextEncoder`: TensorRT wrapper for LLaMA text encoder
- `TensorRTCLIPTextEncoder`: TensorRT wrapper for CLIP text encoder
- `TensorRTTransformer`: TensorRT wrapper for transformer model

### How Compilation Works

1. **First Use**: When a model is called with specific input shapes:
   - TensorRT wrapper checks cache for matching engine
   - If not found, calls `torch_tensorrt.dynamo.compile`
   - Compiles PyTorch model to optimized TensorRT engine
   - Saves engine to disk cache

2. **Subsequent Uses**:
   - Checks disk cache using shape-based key
   - Loads pre-compiled engine if available
   - Falls back to compilation if shape changes

3. **Shape Caching**:
   - VAE: Caches per latent resolution
   - Text Encoders: Caches per (batch, sequence_length)
   - Transformer: Caches per unique input shape combination

## Troubleshooting

### TensorRT Compilation Fails

If TensorRT compilation fails:
1. Check GPU memory (requires ~16GB for transformer compilation)
2. Verify `torch-tensorrt` is installed: `pip install torch-tensorrt`
3. Verify CUDA is available
4. Check error messages in console
5. System will gracefully fall back to PyTorch

### Re-compile Engines

To force re-compilation, delete the TensorRT cache:
```bash
rm -rf FramePack/Cache/tensorrt_engines/
```

### Incompatible Configurations

TensorRT transformer is incompatible with:
- BitsAndBytes quantization
- CPU offload
- FSDP (Fully Sharded Data Parallel)

The system will warn you if these are enabled.

## Performance Notes

- **First Run**: Slower due to ONNX conversion + TensorRT compilation
- **Subsequent Runs**: Fast startup, engines loaded from cache
- **Different Shapes**: TensorRT may compile new engines for new input shapes

## Error Handling

The implementation includes robust error handling:
- If ONNX conversion fails, warning is displayed but startup continues
- TensorRT wrappers gracefully fall back to PyTorch if needed
- Individual component failures don't affect other components

## Key Differences from ONNX-based TensorRT

The example in [examples/Stable-Diffusion-WebUI-TensorRT/](../examples/Stable-Diffusion-WebUI-TensorRT/) uses the older ONNX-based TensorRT workflow:
- Exports models to ONNX first
- Then builds TensorRT engines from ONNX
- Requires separate export step

FramePack uses the modern `torch_tensorrt.dynamo.compile` approach:
- Works directly with PyTorch models (no ONNX export needed)
- JIT compilation on first use
- Automatic shape-based caching
- Simpler and more memory-efficient

## References

- TensorRT runtime implementation: [FramePack/diffusers_helper/tensorrt_runtime.py](diffusers_helper/tensorrt_runtime.py)
- PyTorch TensorRT documentation: https://pytorch.org/TensorRT/
- Alternative ONNX-based approach: [examples/Stable-Diffusion-WebUI-TensorRT/](../examples/Stable-Diffusion-WebUI-TensorRT/)
