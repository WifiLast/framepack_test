# FramePack Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: Do I need TensorRT?
**A:** No, TensorRT is optional. It provides 20-40% faster VAE decoding but requires additional setup. The standard configuration works perfectly without it.

### Q: What are these TensorRT warnings?
**A:** Warnings like these are **safe to ignore**:
```
WARNING: [Torch-TensorRT] - Unable to read CUDA capable devices. Return status: 35
Unable to import quantization op. Please install modelopt library
TensorRT-LLM is not installed.
```

These indicate optional TensorRT features (modelopt, TensorRT-LLM) that aren't installed. TensorRT VAE acceleration still works correctly without them.

### Q: How much VRAM do I need?
**A:**
- **Minimum:** 16GB (with BitsAndBytes 4-bit quantization)
- **Recommended:** 24GB (for Quality Mode + TensorRT)
- **Ideal:** 32GB+ (no memory constraints)

### Q: What about RAM (system memory)?
**A:**
- **Minimum:** 32GB for 110GB+ models
- **Recommended:** 64GB for comfortable operation
- The model is ~110GB on disk but uses quantization to fit in 16GB VRAM

### Q: Which Python version should I use?
**A:** Python 3.10 or 3.11. Python 3.12 may have compatibility issues with some dependencies.

---

## Performance & Quality

### Q: Why is my first generation so slow?
**A:** This is normal! The first run:
- Compiles CUDA kernels (torch.compile)
- Builds attention operator caches
- With TensorRT: compiles engines (~2-5 minutes)

Subsequent generations are **much faster** because everything is cached.

### Q: How do I improve hand quality?
**A:** Three settings:
1. Enable **"Quality Mode"** checkbox (most important)
2. Disable **"Use TeaCache"** checkbox
3. Increase **Steps** to 30-35

Trade-off: ~20-30% slower, +1-2 GB VRAM

### Q: Why does the video look too fast?
**A:** Videos are rendered at 30 FPS (standard). If motion appears too fast:
- **Reduce "Total Video Length"** to 3-4 seconds
- This generates more frames per second of content
- Results in slower, smoother motion

### Q: What's the difference between xformers modes?
**A:**
- **off**: Use PyTorch native attention (slowest, most compatible)
- **standard**: Use xformers with default settings (recommended)
- **aggressive**: Use xformers + cutlass backend (fastest, requires compatible GPU)

---

## Memory & Crashes

### Q: I'm getting "CUDA Out of Memory" errors
**A:** Try these in order:

1. **Enable BitsAndBytes quantization:**
   ```bash
   FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1
   ```

2. **Increase GPU Memory Preservation** (in UI):
   - Go to Gradio UI
   - Increase "GPU Inference Preserved Memory" slider to 10-12 GB
   - This offloads more aggressively but runs slower

3. **Disable caches:**
   ```bash
   python demo_gradio.py --disable-fbcache --disable-sim-cache --disable-kv-cache
   ```

4. **Use extreme memory saving mode:**
   ```bash
   # start.sh line 10
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb=128
   ```

### Q: What does "GPU Inference Preserved Memory" do?
**A:** Higher values = more aggressive offloading to CPU:
- **6 GB (default):** Balanced - keeps most models on GPU
- **10-12 GB:** Aggressive - offloads transformer after sampling
- **20+ GB:** Extreme - offloads almost everything (very slow)

**Rule of thumb:** Start at 6 GB. If you get OOM, increase by 2 GB increments.

### Q: Should I use Quality Mode?
**A:**

**Use Quality Mode when:**
- ✅ You need better hand/finger details
- ✅ You have 18GB+ VRAM
- ✅ You're okay with 20-30% slower generation

**Skip Quality Mode when:**
- ❌ Limited VRAM (≤16 GB)
- ❌ Speed is more important than quality
- ❌ Subject doesn't have hands/complex details

---

## Configuration

### Q: What's the difference between the start.sh configurations?

**Line 3 (Default - Recommended):**
```bash
FRAMEPACK_USE_BNB=1 FRAMEPACK_BNB_LOAD_IN_4BIT=1
```
- Standard config for 16GB VRAM
- BitsAndBytes 4-bit quantization
- xformers standard mode
- Memory V2 backend

**Line 10 (Extreme Memory Saving):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
- For systems with limited RAM (32GB)
- Reduces CUDA memory fragmentation
- Slowest but most stable

**Line 16 (TensorRT + Performance):**
```bash
PYTORCH_ENABLE_FLASH_SDP=0 --enable-tensorrt
```
- Fastest VAE decoding (20-40% faster)
- Requires torch-tensorrt installation
- Disables Flash Attention to prevent conflicts

### Q: What do these environment variables do?

| Variable | Effect |
|----------|--------|
| `FRAMEPACK_FAST_START=1` | Skip torch.compile for encoders (faster startup) |
| `FRAMEPACK_USE_BNB=1` | Enable BitsAndBytes quantization |
| `FRAMEPACK_BNB_LOAD_IN_4BIT=1` | Use 4-bit instead of 8-bit (saves more VRAM) |
| `FRAMEPACK_BNB_CPU_OFFLOAD=1` | Offload quantized models to CPU when idle |
| `FRAMEPACK_VAE_CHUNK_SIZE=2` | Decode 2 latent frames at once (lower = less VRAM) |
| `FRAMEPACK_ENABLE_FBCACHE=1` | Enable feed-forward cache (faster, uses more VRAM) |
| `FRAMEPACK_ENABLE_KV_CACHE=1` | Enable key-value attention cache |
| `PYTORCH_ENABLE_FLASH_SDP=0` | Disable Flash Attention (fixes TensorRT conflicts) |

---

## Advanced

### Q: Should I use Flash Attention?
**A:**
- **H100/H200 GPUs:** Yes, if not using TensorRT
- **Other GPUs (A100, RTX 40xx):** No, use xformers instead
- **With TensorRT:** No, causes CUDA errors

Flash Attention Hopper backend is optimized for newest GPUs and conflicts with TensorRT.

### Q: Can I use FP8 quantization?
**A:** FP8 requires:
- Ada Lovelace (RTX 40xx) or Hopper (H100) GPUs
- transformer-engine package
- FRAMEPACK_ENABLE_FP8=1

**Not recommended for most users** - BitsAndBytes 4-bit is easier and works on more GPUs.

### Q: What's FAISS semantic caching?
**A:** Alternative to hash-based caching:
- Finds "similar enough" cache hits using vector similarity
- More cache hits = faster generation
- Requires `faiss-cpu` or `faiss-gpu` package
- Enable with `--cache-mode semantic`

**Trade-off:** Slightly less deterministic than `hash` mode.

### Q: How does TeaCache affect quality?
**A:** TeaCache speeds up generation by caching intermediate computations, but:
- ✅ **Faster:** ~20-30% speed improvement
- ⚠️ **Quality:** Can make hands/fingers slightly worse
- 💡 **Tip:** Disable for best quality, enable for speed

---

## Troubleshooting

### Q: "Unable to read CUDA capable devices. Return status: 35"
**A:** Safe to ignore! This is a TensorRT startup warning for optional features.

### Q: How do I silence TensorRT warnings?
**A:** The warnings are harmless, but if you want to remove them:

**For "Unable to import quantization op" warnings:**
```bash
pip install "nvidia-modelopt[all]" --extra-index-url https://pypi.nvidia.com
```

**For "TensorRT-LLM is not installed" warning:**
- This is for LLM inference only (not needed for video generation)
- Can be safely ignored - no installation needed

**Note:** Installing these packages is **purely cosmetic** and doesn't improve performance. They only silence startup warnings.

### Q: BitsAndBytes import errors on Windows
**A:** Install CUDA toolkit from NVIDIA, then:
```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes>=0.41.0
```

### Q: "torch.compile failed for _optimizer_precondition_32bit"
**A:** Safe to ignore! The optimizer falls back to eager mode automatically.

### Q: Video has black frames or artifacts
**A:**
1. Increase MP4 Compression slider to 16
2. Check if VAE decoding OOMed (reduce Quality Mode chunk size)
3. Try different seed values

### Q: Generation stuck at 0%
**A:**
1. Check terminal for error messages
2. Model might still be loading (wait 2-3 minutes)
3. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Best Practices

### Q: What are the recommended settings for production?
**A:**

**For Speed (demos, previews):**
```
Quality Mode: OFF
TeaCache: ON
Steps: 25
Cache Mode: hash
```

**For Quality (final outputs):**
```
Quality Mode: ON
TeaCache: OFF
Steps: 30-35
Cache Mode: hash or semantic
```

**For VRAM-constrained (16GB):**
```
Quality Mode: OFF
GPU Memory Preservation: 10 GB
VAE Chunk Size: 1-2
Disable all caches
```

### Q: How do I get the best possible quality?
**A:**
1. Enable **Quality Mode**
2. Disable **TeaCache**
3. Set **Steps** to 35
4. Use **semantic cache mode** (with FAISS)
5. Set **Distilled CFG Scale** to 12-15
6. Use good prompts: specific, detailed, action-oriented

### Q: My system has 64GB VRAM. Can I disable offloading?
**A:** Yes! In the UI:
- Set "GPU Inference Preserved Memory" to **6 GB** (default)
- Everything stays on GPU
- Maximum speed

Or use this configuration:
```bash
# High-VRAM mode (no offloading)
FRAMEPACK_FAST_START=1 FRAMEPACK_PRELOAD_REPOS=0 \
python demo_gradio.py --fast-start --xformers-mode aggressive
```

---

## Getting Help

### Q: Where do I report bugs?
**A:** GitHub Issues: https://github.com/WifiLast/framepack_cython/issues

**Please include:**
- GPU model and VRAM
- Full error message from terminal
- Configuration used (start.sh line or command)
- PyTorch/CUDA versions: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`

### Q: How do I share my configuration?
**A:** Run this and share the output:
```bash
python -c "
import torch, sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

---

## Updates & Community

### Q: How do I update FramePack?
**A:**
```bash
cd framepack_cython
git pull
pip install -r FramePack/requirements.txt --upgrade
```

### Q: Where can I see examples?
**A:** Check the Twitter/X thread: https://x.com/search?q=framepack&f=live

### Q: Can I use this commercially?
**A:** Check the repository license. TensorRT and some dependencies have separate licenses.

---

**Last Updated:** 2025-01-13
**FramePack Version:** Development (unreleased)
