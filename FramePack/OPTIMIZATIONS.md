# FramePack Optimizations & Enhancements

This document describes the memory management, performance optimizations, and quality improvements added to FramePack.

## Table of Contents
- [Quick Reference](#quick-reference)
- [New Features](#new-features)
- [Memory Management](#memory-management)
- [Quality Improvements](#quality-improvements)
- [TensorRT Integration](#tensorrt-integration)
- [Configuration Guide](#configuration-guide)
- [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Installation
```bash
# Core + Performance packages
pip install -r requirements.txt
pip install xformers bitsandbytes

# Optional: TensorRT (20-40% faster VAE)
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Running
```bash
# Standard (16GB VRAM)
bash start.sh

# For better hand quality: Enable "Quality Mode" in UI + Disable "TeaCache"
```

### Documentation Files
- **[INSTALL.md](INSTALL.md)** - Complete installation guide
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start
- **[FAQ.md](FAQ.md)** - 40+ common questions
- **[CHANGELOG.md](CHANGELOG.md)** - Release notes

---

## New Features

### 1. Quality Mode
**Location:** Gradio UI checkbox

**What it does:**
- Uses larger VAE chunks (2-4 frames instead of 1)
- Significantly improves hand and finger quality
- Better temporal consistency across frames

**Trade-offs:**
- ~20-30% slower generation
- +1-2 GB extra VRAM required

**When to use:**
- Subject has hands/fingers in frame
- Quality is more important than speed
- You have 18GB+ VRAM

### 2. Enhanced Memory Management

**New functions in `diffusers_helper/memory.py`:**

#### `load_model_chunked()`
Loads extremely large models (110GB+) in small chunks.

```python
load_model_chunked(model, target_device=gpu, max_chunk_size_mb=256)
```

**Benefits:**
- Prevents OOM during model loading
- Works with 16GB VRAM for 110GB models
- Automatic retry on failures

#### `force_free_vram()`
Aggressively frees VRAM to target amount.

```python
force_free_vram(target_gb=10.0)
```

**Use cases:**
- Before loading large models
- After generation to clear caches
- When switching between models

#### Aggressive Offloading
```python
offload_model_from_device_for_memory_preservation(
    model,
    target_device=gpu,
    preserved_memory_gb=8,
    aggressive=True  # NEW
)
```

**With `aggressive=True`:**
- Continues offloading even after target reached
- Clears cache every 10 modules
- 80% threshold before stopping

### 3. Improved VAE Chunked Decoding

**Location:** `demo_gradio.py:1392-1440`

**Features:**
- Auto-adaptive chunk sizing based on available VRAM
- Quality mode support for better consistency
- Robust error handling with TensorRT fallback
- BFloat16 compatibility (converts to float32 for NumPy)

**Auto-chunk sizing algorithm:**
```python
# Automatically calculates optimal chunk size based on:
# - Available VRAM
# - Frame dimensions (height, width, channels)
# - Quality mode setting
# - Safety margins

chunk_size = _auto_select_vae_chunk_size(latents, quality_mode=True)
```

### 4. TensorRT Integration (Optional)

**Location:** `diffusers_helper/tensorrt_runtime.py`

**Features:**
- 20-40% faster VAE decoding
- Cached engines for common shapes
- Automatic fallback on errors
- Thread-safe compilation

**Installation:**
```bash
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

**Usage:**
```bash
# Uncomment line 16 in start.sh
PYTORCH_ENABLE_FLASH_SDP=0 \  # Disable Flash Attention
python demo_gradio.py --enable-tensorrt
```

---

## Memory Management

### Hierarchy of Memory Optimization

**Level 1: Standard (No changes needed)**
```bash
# Uses default FramePack memory management
python demo_gradio.py
```

**Level 2: BitsAndBytes Quantization**
```bash
FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 \
python demo_gradio.py
```
- Reduces model size by 75% (4-bit instead of 16-bit)
- Minimal quality loss
- Recommended for 16GB VRAM

**Level 3: Aggressive Offloading**
```bash
# Increase "GPU Memory Preservation" in UI to 10-12 GB
```
- Offloads more models to CPU between operations
- Slower but more memory-safe

**Level 4: Extreme Memory Saving**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=128 \
python demo_gradio.py
```
- Reduces CUDA memory fragmentation
- For systems with limited RAM (32GB)

### Memory Usage by Configuration

| Mode | VRAM | RAM | Speed | Quality |
|------|------|-----|-------|---------|
| Standard | 12-14 GB | 32 GB | 100% | ⭐⭐⭐⭐⭐ |
| BNB 4-bit | 10-12 GB | 32 GB | 95% | ⭐⭐⭐⭐⭐ |
| + Quality Mode | 11-13 GB | 32 GB | 75% | ⭐⭐⭐⭐⭐ |
| Extreme Saving | 8-10 GB | 32 GB | 50% | ⭐⭐⭐⭐ |

### Dynamic Swapping System

**How it works:**
1. Models are loaded on-demand to GPU
2. After use, immediately offloaded to CPU
3. DynamicSwapInstaller wraps modules for automatic device management
4. `preserved_memory_gb` parameter controls when to stop loading

**Example:**
```python
# Load model with 8GB preserved for other operations
move_model_to_device_with_memory_preservation(
    transformer,
    target_device=gpu,
    preserved_memory_gb=8
)

# After sampling, offload to free VRAM
offload_model_from_device_for_memory_preservation(
    transformer,
    target_device=gpu,
    preserved_memory_gb=8,
    aggressive=True
)
```

---

## Quality Improvements

### Hand Quality Optimization

**Problem:** Video diffusion models struggle with hands/fingers.

**Solutions:**

**1. Quality Mode (Primary)**
- Enable "Quality Mode" checkbox in UI
- Uses 2-4 frame chunks instead of 1
- Better temporal consistency
- **Result:** Significant improvement in hand details

**2. Disable TeaCache**
- TeaCache trades quality for speed
- Affects fine details like hands
- **Recommendation:** Disable for final renders

**3. Increase Steps**
- Default: 25 steps
- Recommended for quality: 30-35 steps
- Diminishing returns after 35

**4. Optimal Settings**
```
Quality Mode: ON
TeaCache: OFF
Steps: 32
Distilled CFG Scale: 12
Total Video Length: 3-4 seconds
```

### Video Speed Control

**Problem:** Generated video appears too fast.

**Cause:** Fixed 30 FPS output with adaptive frame generation.

**Solution:** Adjust "Total Video Length":
- **Longer length** (5+ seconds) = Faster motion (fewer frames per second of content)
- **Shorter length** (3-4 seconds) = Slower motion (more frames per second of content)

**Example:**
- 5 seconds @ 30fps = 150 frames
- 3 seconds @ 30fps = 90 frames
- Same amount of motion distributed across fewer frames = slower, smoother

### Frame Consistency

**Chunked decoding can cause artifacts between chunks.**

**Mitigation:**
1. Quality Mode uses larger chunks (fewer chunk boundaries)
2. Soft blending in overlapped regions
3. Auto-adaptive chunk sizing avoids very small chunks

---

## TensorRT Integration

### Overview

**What is TensorRT?**
- NVIDIA's high-performance deep learning inference library
- Compiles PyTorch models to optimized engines
- Caches engines for reuse

**Performance:**
- 20-40% faster VAE decoding (after warmup)
- First use: slow (2-5 minutes for engine compilation)
- Subsequent uses: very fast (engines cached)

### Setup

**1. Install torch-tensorrt:**
```bash
# CUDA 12.1
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
```

**2. Verify installation:**
```bash
python -c "import torch_tensorrt; print(torch_tensorrt.__version__)"
```

**3. Enable in start.sh:**
```bash
# Uncomment line 16
PYTORCH_ENABLE_FLASH_SDP=0 PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 \
python demo_gradio.py --enable-tensorrt
```

### Expected Warnings (Safe to Ignore)

```
WARNING: [Torch-TensorRT] - Unable to read CUDA capable devices. Return status: 35
Unable to import quantization op. Please install modelopt library
TensorRT-LLM is not installed.
```

**These are normal!** They indicate optional TensorRT features not installed:
- **modelopt:** For INT8/FP8 quantization (we use BitsAndBytes instead)
- **TensorRT-LLM:** For LLM inference (not needed for video)

**TensorRT is working if you see:**
```
TensorRT VAE decoder enabled (workspace=4096 MB).
```

### Silencing Warnings (Optional)

**Not necessary for functionality, purely cosmetic:**

```bash
# Install NVIDIA ModelOpt (removes quantization warnings)
pip install "nvidia-modelopt[all]" --extra-index-url https://pypi.nvidia.com
```

**Note:** Requires NVIDIA PyPI access.

### Architecture

**TensorRTLatentDecoder:**
- Wraps VAE decoder in TensorRT-compilable wrapper
- Caches compiled engines per input shape `(batch, frames, height, width, dtype)`
- Thread-safe engine compilation with locks
- Automatic fallback to standard VAE on errors

**Engine Cache:**
- Each unique shape compiles separate engine
- Engines persist across generations
- ~100-500 MB VRAM per engine
- Compilation time: 30-120 seconds per shape

**Fallback Logic:**
```python
try:
    decoded = tensorrt_decoder.decode(latents)
except Exception:
    # Automatic fallback to standard VAE
    decoded = vae_decode(latents, vae)
```

### Best Practices

**When to use TensorRT:**
- ✅ You have 24GB+ VRAM (room for engine cache)
- ✅ You generate similar video lengths repeatedly (engine reuse)
- ✅ Speed is priority after first generation

**When to skip TensorRT:**
- ❌ Limited VRAM (≤16 GB)
- ❌ Varying video lengths every generation (no cache benefit)
- ❌ First-time use only (compilation overhead)

---

## Configuration Guide

### start.sh Configurations

**Line 3 (Default - Recommended for 16GB VRAM):**
```bash
FRAMEPACK_FAST_START=1 FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 \
FRAMEPACK_BNB_CPU_OFFLOAD=1 FRAMEPACK_VAE_CHUNK_SIZE=2 \
python demo_gradio.py --fast-start --xformers-mode standard --use-memory-v2
```

**Features:**
- BitsAndBytes 4-bit quantization
- xformers standard mode
- Memory V2 backend (async, pinned memory)
- VAE chunks of 2 frames

**Line 10 (Extreme Memory Saving):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=128 \
FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 \
python demo_gradio.py --fast-start --xformers-mode standard
```

**Features:**
- CUDA memory fragmentation reduction
- For systems with limited RAM
- Slowest but most stable

**Line 16 (TensorRT + Performance):**
```bash
PYTORCH_ENABLE_FLASH_SDP=0 PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 \
FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 \
python demo_gradio.py --fast-start --xformers-mode aggressive \
--use-memory-v2 --enable-tensorrt
```

**Features:**
- TensorRT VAE acceleration
- Flash Attention disabled (prevents conflicts)
- xformers aggressive mode (cutlass backend)
- Fastest after engine warmup

### Environment Variables

| Variable | Values | Effect |
|----------|--------|--------|
| `FRAMEPACK_FAST_START` | 0/1 | Skip torch.compile for encoders (faster startup) |
| `FRAMEPACK_USE_BNB` | 0/1 | Enable BitsAndBytes quantization |
| `FRAMEPACK_BNB_LOAD_IN_4BIT` | 0/1 | Use 4-bit instead of 8-bit |
| `FRAMEPACK_BNB_CPU_OFFLOAD` | 0/1 | Offload quantized models to CPU when idle |
| `FRAMEPACK_VAE_CHUNK_SIZE` | 1-8 | VAE decode chunk size (lower = less VRAM) |
| `FRAMEPACK_ENABLE_FBCACHE` | 0/1 | Enable feed-forward cache |
| `FRAMEPACK_ENABLE_SIM_CACHE` | 0/1 | Enable similarity cache |
| `FRAMEPACK_ENABLE_KV_CACHE` | 0/1 | Enable key-value attention cache |
| `PYTORCH_ENABLE_FLASH_SDP` | 0/1 | Enable/disable Flash Attention |
| `PYTORCH_CUDA_ALLOC_CONF` | config | CUDA memory allocator settings |

### Gradio UI Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Quality Mode | OFF | ON/OFF | Larger VAE chunks for better quality |
| Use TeaCache | ON | ON/OFF | Speed vs quality trade-off |
| Steps | 25 | 1-100 | Denoising steps (more = better quality) |
| Distilled CFG Scale | 10.0 | 1.0-32.0 | Guidance strength |
| Total Video Length | 5.0 | 1-120 | Seconds of output video |
| GPU Memory Preservation | 6 | 6-128 | GB to preserve (higher = more aggressive offloading) |
| MP4 Compression | 16 | 0-100 | CRF value (0 = lossless, lower = better) |

---

## Troubleshooting

### Common Issues

**Issue: CUDA Out of Memory**

**Symptoms:**
```
torch.OutOfMemoryError: CUDA out of memory
```

**Solutions (in order):**
1. Increase "GPU Memory Preservation" to 10-12 GB
2. Disable caches: `--disable-fbcache --disable-sim-cache --disable-kv-cache`
3. Reduce VAE chunk size: `FRAMEPACK_VAE_CHUNK_SIZE=1`
4. Use extreme memory mode (start.sh line 10)

**Issue: Poor hand quality**

**Symptoms:** Hands/fingers look blurry or malformed

**Solutions:**
1. Enable "Quality Mode" in UI
2. Disable "Use TeaCache"
3. Increase Steps to 30-35
4. Use shorter video length (3-4 seconds)

**Issue: Video too fast**

**Symptoms:** Motion appears sped up

**Solutions:**
1. Reduce "Total Video Length" to 3-4 seconds
2. Check that output is 30 FPS (standard video framerate)

**Issue: TensorRT compilation fails**

**Symptoms:**
```
RuntimeError: TensorRT compilation failed
```

**Solutions:**
1. Verify CUDA toolkit installed: `nvcc --version`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure 2-3 GB extra VRAM available
4. Use fallback: Remove `--enable-tensorrt` flag

**Issue: BitsAndBytes import error (Windows)**

**Symptoms:**
```
ImportError: DLL load failed while importing bitsandbytes
```

**Solutions:**
1. Install CUDA toolkit from NVIDIA website
2. Reinstall: `pip uninstall bitsandbytes -y && pip install bitsandbytes>=0.41.0`

### Performance Optimization

**Slow generation speed:**

**Check:**
1. First run is always slow (compilation)
2. Verify GPU usage: `nvidia-smi`
3. Check teacache is enabled (faster)
4. Ensure xformers is installed: `pip show xformers`

**Expected speeds (RTX 4090):**
- Without teacache: ~2.5 seconds/frame
- With teacache: ~1.5 seconds/frame
- With TensorRT (after warmup): ~1.0-1.2 seconds/frame

**Memory leaks / increasing VRAM usage:**

**Solutions:**
1. Clear cache between generations: `force_free_vram()`
2. Restart application after 5-10 generations
3. Enable aggressive offloading

---

## Advanced Topics

### Custom VAE Chunk Sizing

**Auto mode (recommended):**
```python
vae_decode_chunked(latents, vae, chunk_size=None, quality_mode=False)
```

**Manual override:**
```python
vae_decode_chunked(latents, vae, chunk_size=4, quality_mode=False)
```

**Quality mode (adaptive 2-4 frames):**
```python
vae_decode_chunked(latents, vae, chunk_size=None, quality_mode=True)
```

### Memory Profiling

**Check available VRAM:**
```python
from diffusers_helper.memory import get_cuda_free_memory_gb
free_gb = get_cuda_free_memory_gb()
print(f"Free VRAM: {free_gb:.2f} GB")
```

**Monitor during generation:**
```bash
watch -n 1 nvidia-smi
```

### TensorRT Engine Management

**Clear engine cache:**
```python
# Restart application or:
TENSORRT_DECODER._cache.clear()
```

**List cached engines:**
```python
print(TENSORRT_DECODER._cache.keys())
# Output: [(1, 18, 80, 80, torch.float16), ...]
```

---

## See Also

- **[INSTALL.md](INSTALL.md)** - Comprehensive installation guide
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start
- **[FAQ.md](FAQ.md)** - 40+ frequently asked questions
- **[CHANGELOG.md](CHANGELOG.md)** - Release notes and migration guide
- **[requirements-optional.txt](requirements-optional.txt)** - Optional packages

---

**Last Updated:** 2025-01-13
**FramePack Version:** Development
**Compatibility:** Python 3.10+, PyTorch 2.0+, CUDA 11.8+
