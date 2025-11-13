# TensorRT Acceleration Setup Guide

## Overview

TensorRT can accelerate both VAE (encode/decode) and the Transformer model in FramePack. However, TensorRT has specific requirements and is incompatible with some optimization modes.

## Requirements

1. **Hardware**
   - NVIDIA GPU with compute capability 7.0+ (Volta or newer)
   - At least 16GB GPU VRAM for transformer acceleration
   - At least 8GB GPU VRAM for VAE-only acceleration

2. **Software**
   - CUDA 11.8 or 12.x
   - `torch-tensorrt` package installed
   - PyTorch with CUDA support

3. **Installation**
   ```bash
   pip install torch-tensorrt
   ```

## Usage

### VAE Acceleration Only (Recommended for most users)

```bash
python demo_gradio.py --enable-tensorrt
```

This accelerates VAE encode/decode operations with minimal VRAM requirements.

### Transformer + VAE Acceleration (High VRAM Required)

```bash
python demo_gradio.py --enable-tensorrt --tensorrt-transformer
```

**Important**: This requires ~16GB+ GPU VRAM and 5-15 minutes for first compilation.

## Incompatible Options

TensorRT **CANNOT** be used with:

### ❌ BitsAndBytes Quantization
```bash
# This will NOT work:
FRAMEPACK_BNB_LOAD_IN_4BIT=1 python demo_gradio.py --enable-tensorrt --tensorrt-transformer
```

**Reason**: TensorRT requires FP16/BF16 precision, not quantized models.

### ❌ CPU Offload
```bash
# This will NOT work:
FRAMEPACK_BNB_CPU_OFFLOAD=1 python demo_gradio.py --enable-tensorrt --tensorrt-transformer
```

**Reason**: TensorRT requires the entire model on GPU.

### ❌ FSDP (Fully Sharded Data Parallel)
```bash
# This will NOT work with FSDP enabled
```

**Reason**: TensorRT doesn't support distributed/sharded models.

## Correct Configuration Examples

### Example 1: High VRAM System (24GB+)
```bash
# Clean TensorRT setup
python demo_gradio.py \
  --enable-tensorrt \
  --tensorrt-transformer \
  --use-memory-v2
```

### Example 2: Medium VRAM System (16GB)
```bash
# VAE acceleration only
python demo_gradio.py \
  --enable-tensorrt \
  --use-memory-v2 \
  --fast-start
```

### Example 3: Low VRAM System (12GB)
```bash
# Use BitsAndBytes instead (no TensorRT)
FRAMEPACK_BNB_LOAD_IN_4BIT=1 \
FRAMEPACK_BNB_CPU_OFFLOAD=1 \
python demo_gradio.py --fast-start
```

## Environment Variables

```bash
# TensorRT Configuration
FRAMEPACK_ENABLE_TENSORRT=1              # Enable TensorRT runtime
FRAMEPACK_TRT_TRANSFORMER=1              # Enable transformer acceleration
FRAMEPACK_TRT_WORKSPACE_MB=4096          # TensorRT workspace size (MB)
FRAMEPACK_TRT_MAX_CACHED_SHAPES=8        # Max cached engine shapes
FRAMEPACK_TRT_MAX_AUX_STREAMS=2          # Max auxiliary CUDA streams

# Memory Configuration (for non-TensorRT mode)
FRAMEPACK_USE_BNB=1                      # Enable BitsAndBytes (incompatible with TensorRT)
FRAMEPACK_BNB_LOAD_IN_4BIT=1             # 4-bit quantization (incompatible with TensorRT)
FRAMEPACK_BNB_CPU_OFFLOAD=1              # CPU offload (incompatible with TensorRT)
```

## Performance Expectations

### VAE TensorRT Acceleration
- **First use**: Slower due to engine compilation (~30 seconds per shape)
- **Subsequent uses**: 1.5-2x faster than PyTorch
- **VRAM impact**: +500MB-1GB for cached engines

### Transformer TensorRT Acceleration
- **First use**: VERY slow (5-15 minutes compilation)
- **Subsequent uses**: 1.5-3x faster than PyTorch
- **VRAM impact**: +2-4GB for cached engines
- **Note**: May not always provide speedup due to model complexity

## Troubleshooting

### Issue: "TensorRT transformer requested but not available"

**Check**:
1. Run without BNB flags
2. Ensure `torch-tensorrt` is installed
3. Check GPU has enough VRAM (16GB+ recommended)
4. Look for initialization warnings at startup

### Issue: "TensorRT compilation failed"

**Common causes**:
1. Model has dynamic control flow (not TensorRT compatible)
2. Insufficient GPU VRAM
3. Incompatible CUDA version

**Solution**: The system will automatically fall back to PyTorch. This is normal for complex models.

### Issue: No speedup observed

**Likely causes**:
1. Using incompatible options (BNB, CPU offload)
2. Still using PyTorch fallback (check console for "Compiling TensorRT" messages)
3. Model structure not suitable for TensorRT optimization

## How to Verify TensorRT is Working

Look for these console messages:

### At Startup
```
TensorRT VAE acceleration enabled (workspace=4096 MB).
TensorRT transformer wrapper initialized. Engines will compile on first use per shape.
```

### During Inference
```
============================================================
TensorRT transformer acceleration ENGAGED for this job
============================================================
...
Compiling TensorRT transformer engine for shape: ...
TensorRT transformer compilation successful
```

If you don't see these messages, TensorRT is not being used.

## Recommendations

1. **Start with VAE-only**: Try `--enable-tensorrt` first without `--tensorrt-transformer`
2. **Monitor VRAM**: Use `nvidia-smi` to check GPU memory usage
3. **Be patient**: First compilation is slow but subsequent runs are fast
4. **Use clean configs**: Avoid mixing TensorRT with BNB/quantization
5. **Check compatibility**: Not all models benefit from TensorRT

## Known Limitations

1. **Dynamic shapes**: Each unique input shape requires separate compilation
2. **Large models**: HunyuanVideo transformer may be too complex for TensorRT
3. **Memory overhead**: Cached engines consume additional VRAM
4. **Compilation time**: Initial compilation can be very slow (5-15 minutes)
5. **Fallback behavior**: System may silently fall back to PyTorch if compilation fails

## Alternative Optimizations

If TensorRT doesn't work or doesn't provide speedup, try:

- **torch.compile**: Already integrated, automatic
- **BitsAndBytes quantization**: Reduces VRAM usage
- **Memory V2**: Improved memory management (`--use-memory-v2`)
- **TeaCache**: Faster sampling with slight quality trade-off
- **Flash Attention**: Better attention performance
