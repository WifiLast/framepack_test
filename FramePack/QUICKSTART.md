# FramePack Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Create environment
conda create -n py310 python=3.10 -y
conda activate py310

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
cd FramePack
pip install -r requirements.txt
pip install xformers bitsandbytes

# 4. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Launch FramePack

```bash
bash start.sh
```

Then open: **http://localhost:7860**

## First Generation

1. **Upload an image** (starting frame)
2. **Enter a prompt**: "The person waves their hand and smiles"
3. **Settings:**
   - Total Video Length: **3-5 seconds**
   - Steps: **25-30**
   - GPU Memory Preservation: **6-8 GB**
4. Click **"Start Generation"**

## Common Issues

### ❌ CUDA Out of Memory

**Fix:** Increase "GPU Inference Preserved Memory" to 10-12 GB

### ❌ Slow generation

**Normal:** First generation compiles models (slow), subsequent runs are faster

### ❌ Poor hand quality

**Fix:**
- Enable **"Quality Mode"** checkbox
- Disable **"Use TeaCache"** checkbox
- Increase Steps to 30-35

### ❌ Video too fast

**Fix:** Reduce "Total Video Length" to 3-4 seconds

### ⚠️ TensorRT Warnings (Safe to Ignore)

```
WARNING: [Torch-TensorRT] - Unable to read CUDA capable devices
Unable to import quantization op
TensorRT-LLM is not installed
```

**These are normal and safe to ignore!** They indicate optional TensorRT features not installed. TensorRT will still work for VAE decoding.

## Recommended Settings

### Best Quality (Slower)
- Quality Mode: **ON**
- TeaCache: **OFF**
- Steps: **30-35**
- Total Length: **3-4 seconds**

### Best Speed (Lower Quality)
- Quality Mode: **OFF**
- TeaCache: **ON**
- Steps: **25**
- Total Length: **5 seconds**

### Balanced (Recommended)
- Quality Mode: **ON**
- TeaCache: **ON**
- Steps: **28**
- Total Length: **4 seconds**

## Configuration Files

- **start.sh**: Main startup configurations
- **requirements.txt**: Core dependencies
- **requirements-optional.txt**: Performance optimizations
- **INSTALL.md**: Detailed installation guide

## Advanced Usage

See [INSTALL.md](INSTALL.md) for:
- TensorRT acceleration (20-40% faster VAE decode)
- Flash Attention setup
- FAISS semantic caching
- Memory optimization tips

## Need Help?

- Check [INSTALL.md](INSTALL.md) for troubleshooting
- GitHub Issues: https://github.com/WifiLast/framepack_cython/issues
- Include: GPU model, VRAM, error message, settings used
