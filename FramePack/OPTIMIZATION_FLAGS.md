# FramePack Optimization Flags Guide

Complete guide to optimization flags and environment variables for `demo_gradio.py`.

## ⚠️ Automatic Compatibility Checking

**NEW:** The application now automatically validates flag compatibility at startup and warns you about:
- ❌ **Critical incompatibilities** that will cause errors
- ⚠️ **Warnings** about suboptimal configurations
- ℹ️ **Info** about optimizations being skipped

**Detected incompatibilities:**
1. **torch.compile + memory-v2** → CUDA graph conflicts
2. **TensorRT + BitsAndBytes** → Precision mismatch
3. **torch.compile + TorchScript** → Redundant compilation
4. **Multiple quantization methods** → Conflicts
5. And more...

When errors are detected, the application pauses for 5 seconds to let you review the warnings before proceeding.

---

## Quick Start - Recommended Configuration

For best performance with quantization and compile optimizations:

```bash
# Enable quantization (INT8) + torch.compile + caching
export FRAMEPACK_ENABLE_QUANT=1
export FRAMEPACK_QUANT_BITS=8
export FRAMEPACK_ENABLE_COMPILE=1
export FRAMEPACK_ENABLE_OPT_CACHE=1

# Run demo
python demo_gradio.py
```

**The optimized model IS saved** when `FRAMEPACK_ENABLE_OPT_CACHE=1` to:
- Default: `FramePack/optimized_models/transformer_quantized.pt`
- Custom: Set `FRAMEPACK_OPTIMIZED_TRANSFORMER=/path/to/save.pt`

---

## Optimization Categories

### 1. Quantization

#### INT-N Quantization (Standard)
```bash
export FRAMEPACK_ENABLE_QUANT=1      # Enable quantization
export FRAMEPACK_QUANT_BITS=8        # Bits: 4, 8 (default: 8)
```

**What it does:**
- Applies INT-N quantization to transformer layers
- Reduces memory usage significantly
- Slight quality tradeoff for much faster inference

#### FP8 Quantization (Advanced)
```bash
export FRAMEPACK_ENABLE_FP8=1                    # Enable FP8
export FRAMEPACK_FP8_TARGET_KEYS=attn,mlp        # Target specific layers (CSV)
export FRAMEPACK_FP8_EXCLUDE_KEYS=norm           # Exclude layers (CSV)
export FRAMEPACK_FP8_USE_SCALED_MM=0             # Use scaled matrix multiply
```

**Requirements:**
- Requires `third_party/fp8_optimization_utils.py`
- Hopper GPU (H100) recommended for native FP8 support

#### BitsAndBytes Quantization (HuggingFace)
```bash
export FRAMEPACK_USE_BNB=1                       # Enable BitsAndBytes
export FRAMEPACK_BNB_LOAD_IN_4BIT=1              # 4-bit vs 8-bit (default: 8-bit)
export FRAMEPACK_BNB_CPU_OFFLOAD=1               # Offload to CPU
export FRAMEPACK_BNB_DEVICE_MAP=auto             # Device placement: auto/cpu/balanced
export FRAMEPACK_BNB_DOUBLE_QUANT=1              # Double quantization
```

**Note:** BitsAndBytes is incompatible with TensorRT!

---

### 2. Compilation (torch.compile)

```bash
export FRAMEPACK_ENABLE_COMPILE=1                # Enable torch.compile
```

**What it does:**
- Applies `torch.compile()` to transformer and encoders
- Uses TorchInductor backend for kernel fusion
- **First run compiles (slow), subsequent runs are fast**
- Skipped if TorchScript mode is enabled

**Incompatible with:**
- `--jit-mode trace/script` (TorchScript)

---

### 3. Model Caching

#### Optimized Model Cache
```bash
export FRAMEPACK_ENABLE_OPT_CACHE=1              # **THIS SAVES THE OPTIMIZED MODEL**
export FRAMEPACK_OPTIMIZED_TRANSFORMER=/path/to/transformer_quantized.pt
```

**What it does:**
- Saves quantized/pruned transformer after first optimization
- Loads directly from cache on subsequent runs
- **Dramatically reduces startup time**

**Default save location:**
- `FramePack/optimized_models/transformer_quantized.pt`

#### Runtime Caching
```bash
export FRAMEPACK_RUNTIME_CACHE=1                 # Enable runtime cache (default: ON)
export FRAMEPACK_RUNTIME_CACHE_DIR=Cache/runtime_caches
```

**What it caches:**
- Similarity cache states
- First block cache states
- KV cache states

---

### 4. Memory Optimizations

#### First Block Cache
```bash
export FRAMEPACK_ENABLE_FBCACHE=1                # Enable first block caching
export FRAMEPACK_FBCACHE_THRESHOLD=0.035         # Similarity threshold
export FRAMEPACK_FBCACHE_VERBOSE=1               # Debug logging
```

#### Similarity Cache
```bash
export FRAMEPACK_ENABLE_SIM_CACHE=1              # Enable similarity caching
export FRAMEPACK_SIM_CACHE_THRESHOLD=0.9         # Cosine similarity threshold
export FRAMEPACK_SIM_CACHE_MAX_SKIP=1            # Max blocks to skip
export FRAMEPACK_SIM_CACHE_MAX_ENTRIES=12        # Cache size
export FRAMEPACK_SIM_CACHE_USE_FAISS=1           # Use FAISS for search (faster)
export FRAMEPACK_SIM_CACHE_VERBOSE=1             # Debug logging
```

#### KV Cache
```bash
export FRAMEPACK_ENABLE_KV_CACHE=1               # Enable KV caching
export FRAMEPACK_KV_CACHE_LEN=4096               # Maximum sequence length
export FRAMEPACK_KV_CACHE_VERBOSE=1              # Debug logging
```

#### Memory Backend
```bash
python demo_gradio.py --use-memory-v2            # Enable optimized memory backend
```

**Features:**
- Async CUDA streams
- Pinned memory
- Cached statistics

---

### 5. TensorRT Acceleration

```bash
export FRAMEPACK_ENABLE_TENSORRT=1               # Enable TensorRT runtime
export FRAMEPACK_TRT_TRANSFORMER=1               # TensorRT for transformer
export FRAMEPACK_TRT_TEXT_ENCODERS=1             # TensorRT for text encoders
export FRAMEPACK_USE_ONNX_ENGINES=1              # Use pre-built ONNX engines
export FRAMEPACK_TENSORRT_CACHE_DIR=Cache/trt_cache
export FRAMEPACK_TRT_WORKSPACE_MB=4096           # Workspace size (MB)
export FRAMEPACK_TRT_MAX_AUX_STREAMS=2           # Auxiliary streams
export FRAMEPACK_TRT_MAX_CACHED_SHAPES=8         # Max cached engine shapes
```

**ONNX Runtime with TensorRT:**
```bash
export FRAMEPACK_ONNX_TRT_ENABLE=1               # Enable ONNX TensorRT provider
export FRAMEPACK_ONNX_TRT_FP16=1                 # FP16 precision
export FRAMEPACK_ONNX_TRT_INT8=0                 # INT8 calibration
export FRAMEPACK_ONNX_TRT_DEVICE_ID=0            # GPU device
export FRAMEPACK_ONNX_TRT_WORKSPACE_MB=4096      # Workspace
export FRAMEPACK_ONNX_TRT_CACHE_DIR=Cache/onnx_trt_cache
```

**Note:** TensorRT incompatible with BitsAndBytes!

---

### 6. TorchScript (Alternative to torch.compile)

```bash
python demo_gradio.py --jit-mode trace            # or 'script'
export FRAMEPACK_JIT_ARTIFACT=optimized_models/transformer_torchscript.pt
export FRAMEPACK_JIT_SAVE=/path/to/save.pt        # Save traced model
export FRAMEPACK_JIT_LOAD=/path/to/load.pt        # Load traced model
export FRAMEPACK_JIT_STRICT=1                     # Strict shape checking
```

**Modes:**
- `off`: Disabled (default)
- `trace`: Trace execution (faster, less flexible)
- `script`: Script compilation (slower, more flexible)

---

### 7. Attention Optimizations

```bash
export FRAMEPACK_XFORMERS_MODE=aggressive         # standard/aggressive/off
python demo_gradio.py --xformers-mode aggressive
```

**Modes:**
- `standard`: Use native PyTorch SDPA or Flash Attention
- `aggressive`: Force xFormers memory-efficient attention
- `off`: Disable all optimizations

---

### 8. Layer Pruning

```bash
export FRAMEPACK_ENABLE_PRUNE=1                   # Enable layer pruning
```

**What it does:**
- Removes redundant transformer layers
- Experimental feature
- Can reduce quality

---

### 9. Other Performance Flags

#### Low Precision Enforcement
```bash
export FRAMEPACK_ENFORCE_LOW_PRECISION=1
python demo_gradio.py --enforce-low-precision
```

#### Fast Startup
```bash
python demo_gradio.py --fast-start
export FRAMEPACK_FAST_START=1
```

**What it skips:**
- torch.compile
- Optimized model caching
- Optional transformations

#### Parallel Model Loading
```bash
export FRAMEPACK_PARALLEL_LOADERS=4              # Number of parallel threads
export FRAMEPACK_PRELOAD_REPOS=1                 # Preload HF repos
export FRAMEPACK_FORCE_PARALLEL_LOADERS=1        # Force parallel even if unsafe
```

#### VAE Chunking
```bash
export FRAMEPACK_VAE_CHUNK_SIZE=8                # Manual chunk size
export FRAMEPACK_VAE_CHUNK_RESERVE_GB=2.0        # Reserved VRAM
export FRAMEPACK_VAE_CHUNK_SAFETY=1.2            # Safety multiplier
export FRAMEPACK_VAE_UPSCALE_FACTOR=8            # Spatial upscale factor
```

---

## Complete Example Configurations

### Maximum Performance (Quantized + Compiled + Cached)

```bash
#!/bin/bash
# Best performance, saves optimized model

# Quantization
export FRAMEPACK_ENABLE_QUANT=1
export FRAMEPACK_QUANT_BITS=8

# Compilation
export FRAMEPACK_ENABLE_COMPILE=1

# Caching
export FRAMEPACK_ENABLE_OPT_CACHE=1
export FRAMEPACK_OPTIMIZED_TRANSFORMER=optimized_models/transformer_q8_compiled.pt
export FRAMEPACK_ENABLE_MODULE_CACHE=1

# Memory optimizations
export FRAMEPACK_ENABLE_FBCACHE=1
export FRAMEPACK_ENABLE_SIM_CACHE=1
export FRAMEPACK_SIM_CACHE_USE_FAISS=1

# Attention
export FRAMEPACK_XFORMERS_MODE=aggressive

# Run
python demo_gradio.py --use-memory-v2
```

### TensorRT Maximum Speed

```bash
#!/bin/bash
# Requires ONNX models exported first

export FRAMEPACK_ENABLE_TENSORRT=1
export FRAMEPACK_TRT_TRANSFORMER=1
export FRAMEPACK_TRT_TEXT_ENCODERS=1
export FRAMEPACK_USE_ONNX_ENGINES=1
export FRAMEPACK_TRT_WORKSPACE_MB=8192
export FRAMEPACK_TENSORRT_CACHE_DIR=Cache/trt_engines

# Don't use BitsAndBytes with TensorRT!
export FRAMEPACK_USE_BNB=0

python demo_gradio.py
```

### FP8 Maximum Quality (Hopper GPUs)

```bash
#!/bin/bash
# Requires H100/H800 GPU

export FRAMEPACK_ENABLE_FP8=1
export FRAMEPACK_FP8_USE_SCALED_MM=1
export FRAMEPACK_ENABLE_COMPILE=1
export FRAMEPACK_ENABLE_OPT_CACHE=1
export FRAMEPACK_OPTIMIZED_TRANSFORMER=optimized_models/transformer_fp8.pt

python demo_gradio.py
```

### Memory-Constrained (4-bit + CPU Offload)

```bash
#!/bin/bash
# For low VRAM GPUs

export FRAMEPACK_USE_BNB=1
export FRAMEPACK_BNB_LOAD_IN_4BIT=1
export FRAMEPACK_BNB_CPU_OFFLOAD=1
export FRAMEPACK_BNB_DEVICE_MAP=balanced

# Enable all caching
export FRAMEPACK_ENABLE_FBCACHE=1
export FRAMEPACK_ENABLE_SIM_CACHE=1
export FRAMEPACK_ENABLE_KV_CACHE=1

python demo_gradio.py --use-memory-v2
```

---

## Checking if Optimizations are Applied

```python
# After model loads, check:
print(f"Quantization active: {ENABLE_QUANT}")
print(f"FP8 active: {FP8_TRANSFORMER_ACTIVE}")
print(f"Optimized model loaded: {optimized_transformer_loaded}")
print(f"TensorRT active: {ENABLE_TENSORRT_RUNTIME}")
print(f"Compiled: {ENABLE_COMPILE}")
```

Look for these messages in startup logs:
- `Loaded optimized transformer weights from ...` ✅ Using cached optimized model
- `Saved optimized transformer weights to ...` ✅ Created optimized model cache
- `Applying INT-8 quantization to transformer...` ✅ Quantization active
- `FP8 optimization active` ✅ FP8 quantization applied
- `torch.compile enabled for transformer` ✅ Compilation active
- `TensorRT runtime initialized` ✅ TensorRT acceleration

---

## Performance vs Quality Trade-offs

| Configuration | Speed | VRAM | Quality | Startup |
|--------------|-------|------|---------|---------|
| **Default** | 1.0x | High | 100% | Fast |
| **INT8 Quant** | 1.5x | Medium | 98% | Medium |
| **INT4 Quant** | 2.0x | Low | 90% | Medium |
| **FP8 (H100)** | 1.8x | Medium | 99% | Medium |
| **torch.compile** | 1.3x | High | 100% | Slow first run |
| **TensorRT** | 2.5x | Medium | 100% | Slow first run |
| **All Combined** | 3-4x | Low-Med | 95-98% | Slow first, fast after |

---

## Troubleshooting

### "duplicate template name" error with torch.compile
- The code has a guard that auto-falls back to eager mode
- No action needed

### TensorRT fails to load
- Check `FRAMEPACK_USE_BNB=0` (incompatible)
- Verify ONNX models exist in cache
- Check CUDA/TensorRT versions

### Quantization degrades quality too much
- Try FP8 instead of INT8/INT4
- Exclude sensitive layers: `FRAMEPACK_FP8_EXCLUDE_KEYS=norm,proj`
- Reduce quantization: `FRAMEPACK_QUANT_BITS=8` instead of 4

### Cached model not loading
- Check path: `echo $FRAMEPACK_OPTIMIZED_TRANSFORMER`
- Verify file exists and has correct permissions
- Delete cache and rebuild: `rm optimized_models/transformer_quantized.pt`

---

## See Also

- [ONNX_CONVERSION_EXAMPLES.md](ONNX_CONVERSION_EXAMPLES.md) - ONNX export guide
- [FRAMEPACK_I2V_ONNX_LIMITATIONS.md](FRAMEPACK_I2V_ONNX_LIMITATIONS.md) - TensorRT limitations
- [diffusers_helper/optimizations.py](diffusers_helper/optimizations.py) - Optimization implementations
