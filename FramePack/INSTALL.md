# FramePack Installation Guide

## Prerequisites

- **Python**: 3.10 or 3.11
- **CUDA**: 11.8 or 12.1+ (for GPU acceleration)
- **VRAM**: Minimum 16GB recommended
- **RAM**: Minimum 32GB recommended (for 110GB+ models)

## Basic Installation

### 1. Create Conda Environment

```bash
conda create -n py310 python=3.10 -y
conda activate py310
```

### 2. Install PyTorch (with CUDA support)

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 3. Install Core Dependencies

```bash
cd FramePack
pip install -r requirements.txt
```

## Optional Performance Optimizations

### Option 1: xformers (Recommended)

Provides optimized attention mechanisms for better performance:

```bash
pip install xformers>=0.0.20
```

### Option 2: BitsAndBytes (Memory Saving)

Enables 4-bit/8-bit quantization to reduce VRAM usage:

```bash
pip install bitsandbytes>=0.41.0
```

**Note:** On Windows, you may need the CUDA toolkit installed.

### Option 3: TensorRT (Advanced - GPU Acceleration)

**⚠️ Experimental - May conflict with Flash Attention**

**Requirements:**
- CUDA toolkit (11.8 or 12.1)
- Compatible PyTorch version
- ~1-2 GB extra VRAM for engine cache

**Installation:**

```bash
# For CUDA 12.1
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu118
```

**Verify:**
```bash
python -c "import torch_tensorrt; print(f'TensorRT: {torch_tensorrt.__version__}')"
```

**To use TensorRT:**
```bash
# Enable in start.sh (uncomment line 16)
PYTORCH_ENABLE_FLASH_SDP=0 PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 \
FRAMEPACK_PRELOAD_REPOS=0 FRAMEPACK_FAST_START=1 \
FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1 FRAMEPACK_BNB_CPU_OFFLOAD=1 \
python demo_gradio.py --fast-start --xformers-mode aggressive --use-memory-v2 --enable-tensorrt
```

### Option 4: Flash Attention (Hopper GPUs)

**⚠️ Only for H100/H200 GPUs - May conflict with TensorRT**

```bash
pip install flash-attn>=2.0.0 --no-build-isolation
```

### Option 5: FAISS (Semantic Caching)

For semantic-based cache hits (faster than exact hash matching):

**CPU version:**
```bash
pip install faiss-cpu
```

**GPU version (faster):**
```bash
pip install faiss-gpu
```

## Recommended Installation Paths

### For 16GB VRAM (Standard)

```bash
# 1. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install performance optimizations
pip install xformers>=0.0.20
pip install bitsandbytes>=0.41.0

# 4. Optional: FAISS for better caching
pip install faiss-cpu  # or faiss-gpu if you have extra VRAM
```

**Configuration:** Use `bash start.sh` (line 3 - standard config)

### For 24GB+ VRAM (High Performance)

```bash
# Same as above, plus:
pip install torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu121
```

**Configuration:** Use line 16 in start.sh (with TensorRT)

### For < 16GB VRAM (Memory Constrained)

```bash
# Install only essentials
pip install -r requirements.txt
pip install bitsandbytes>=0.41.0  # For quantization
```

**Configuration:** Use line 10 in start.sh (extreme memory saving mode)

## Verification

After installation, verify all components:

```bash
python -c "
import torch
import diffusers
import transformers
import gradio
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
print(f'✓ CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'✓ Diffusers: {diffusers.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Gradio: {gradio.__version__}')

try:
    import xformers
    print(f'✓ xformers: {xformers.__version__}')
except ImportError:
    print('✗ xformers: Not installed')

try:
    import bitsandbytes as bnb
    print(f'✓ BitsAndBytes: {bnb.__version__}')
except ImportError:
    print('✗ BitsAndBytes: Not installed')

try:
    import torch_tensorrt
    print(f'✓ TensorRT: {torch_tensorrt.__version__}')
except ImportError:
    print('✗ TensorRT: Not installed (optional)')

try:
    import faiss
    print(f'✓ FAISS: Available')
except ImportError:
    print('✗ FAISS: Not installed (optional)')
"
```

## Troubleshooting

### CUDA Out of Memory

**Solutions:**
1. Enable BitsAndBytes 4-bit quantization
2. Reduce `GPU Inference Preserved Memory` in UI (increase to 10-12 GB)
3. Use extreme memory saving mode (start.sh line 10)
4. Disable TeaCache and caching features

### Flash Attention CUDA Errors

```
CUDA error: invalid argument
```

**Solution:** Disable Flash Attention:
```bash
PYTORCH_ENABLE_FLASH_SDP=0 PYTORCH_ENABLE_MEM_EFFICIENT_SDP=0 python demo_gradio.py ...
```

### BitsAndBytes Import Errors (Windows)

**Solution:** Install CUDA toolkit from NVIDIA website, then reinstall:
```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes>=0.41.0
```

### TensorRT Warnings (Non-Critical)

You may see these warnings when starting with TensorRT:

```
WARNING: [Torch-TensorRT] - Unable to read CUDA capable devices. Return status: 35
Unable to import quantization op. Please install modelopt library
Unable to import quantize op. Please install modelopt library
TensorRT-LLM is not installed.
```

**These are SAFE to ignore!** They indicate missing optional components:
- **modelopt**: Only needed for INT8/FP8 quantization (we use BitsAndBytes instead)
- **TensorRT-LLM**: Only for LLM-specific optimizations (not needed for VAE)
- **CUDA devices warning**: Usually harmless, TensorRT will still work

**TensorRT is working correctly if you see:**
```
TensorRT VAE decoder enabled (workspace=4096 MB).
```

#### Silencing TensorRT Warnings (Optional)

If you want to silence the warnings (not necessary for functionality):

**Install NVIDIA ModelOpt (for quantization warnings):**
```bash
pip install "nvidia-modelopt[all]" --extra-index-url https://pypi.nvidia.com
```

**Note:** Requires NVIDIA PyPI access. This is **purely optional** and only removes warnings.

**TensorRT-LLM** is not needed for FramePack (LLM-only feature). The warning can be ignored.

### TensorRT Compilation Failures

**Solutions:**
1. Verify torch-tensorrt matches your PyTorch version
2. Check CUDA toolkit is installed (nvcc --version)
3. Ensure sufficient VRAM for engine compilation (~2-3 GB extra)
4. Use fallback: Remove `--enable-tensorrt` flag
5. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

## Getting Started

After installation:

```bash
# Start FramePack
bash start.sh

# Or with custom configuration
FRAMEPACK_FAST_START=1 FRAMEPACK_USE_BNB=1 \
python demo_gradio.py --fast-start --xformers-mode standard
```

Open browser at: `http://localhost:7860`

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `FRAMEPACK_FAST_START` | 0/1 | Skip torch.compile for encoders (faster startup) |
| `FRAMEPACK_USE_BNB` | 0/1 | Enable BitsAndBytes quantization |
| `FRAMEPACK_BNB_LOAD_IN_4BIT` | 0/1 | Use 4-bit quantization (reduces VRAM) |
| `FRAMEPACK_BNB_CPU_OFFLOAD` | 0/1 | Offload quantized models to CPU when not in use |
| `FRAMEPACK_VAE_CHUNK_SIZE` | 1-8 | VAE decode chunk size (lower = less VRAM) |
| `FRAMEPACK_ENABLE_FBCACHE` | 0/1 | Enable feed-forward cache |
| `FRAMEPACK_ENABLE_SIM_CACHE` | 0/1 | Enable similarity cache |
| `FRAMEPACK_ENABLE_KV_CACHE` | 0/1 | Enable key-value cache |
| `PYTORCH_ENABLE_FLASH_SDP` | 0/1 | Enable/disable Flash Attention |
| `PYTORCH_ENABLE_MEM_EFFICIENT_SDP` | 0/1 | Enable/disable memory-efficient attention |

## Support

For issues and questions:
- GitHub Issues: https://github.com/WifiLast/framepack_cython/issues
- Make sure to include: GPU model, VRAM, error logs, configuration used
