# Profile-Guided Optimization (PGO) Guide

## Overview

Profile-Guided Optimization helps you identify **real bottlenecks** in your FramePack pipeline using data instead of guesses. This is critical because optimizations like BetterTransformer showed that assumptions can be wrong (20s/it → 428s/it regression!).

## Why Profile First?

**The BetterTransformer lesson taught us:**
- ✅ Measure before optimizing
- ✅ Test with your actual workload
- ✅ Memory offloading patterns matter more than micro-optimizations
- ❌ Don't assume "faster attention" = faster overall

**Profiling shows you:**
1. Where time is **actually** spent (not where you think)
2. Memory bottlenecks (GPU, CPU, SSD I/O)
3. Which optimizations will help vs. hurt
4. Cache effectiveness

## Quick Start

### 1. Basic Timing Profiling

```bash
# Run with profiling enabled
python FramePack/demo_gradio.py --enable-profiling
```

This collects:
- ⏱️ Timing for each major operation
- 📊 Iteration-by-iteration breakdown
- 💾 Memory snapshots

Results saved to: `./profiling_results/`

### 2. Detailed PyTorch Profiling

For GPU kernel-level analysis:

```bash
python FramePack/demo_gradio.py --enable-profiling --profiling-iterations 3
```

This generates Chrome trace files showing:
- CUDA kernel execution
- Memory transfers
- CPU/GPU overlap
- Attention operations

**View the trace:**
1. Open Chrome browser
2. Navigate to: `chrome://tracing`
3. Load: `./profiling_results/trace.json`

### 3. Manual Profiling with Code

```python
from diffusers_helper.profiling import (
    profile_section,
    get_global_stats,
    MemoryTracker,
)

# Profile a specific section
with profile_section("my_operation"):
    result = expensive_operation()

# Check memory usage
tracker = MemoryTracker()
tracker.snapshot("before_model_load")
load_model()
tracker.snapshot("after_model_load")
tracker.print_summary()

# Print timing summary
get_global_stats().print_summary()
```

## Understanding the Results

### Timing Report

```
================================================================================
PROFILING SUMMARY - Top Operations by Total Time
================================================================================
Operation                                          Count      Total       Mean        Min        Max
--------------------------------------------------------------------------------
vae_decode                                            12     45.234s     3.769s     3.102s     4.521s
text_encoder_forward                                   2     12.456s     6.228s     6.100s     6.356s
transformer_forward                                  150      8.234s     0.055s     0.048s     0.072s
image_encoder_forward                                  1      2.103s     2.103s     2.103s     2.103s
================================================================================
```

**What this tells you:**
- **VAE decode** is the bottleneck (45s total!)
- **Text encoding** is slow but only runs twice
- **Transformer** is optimized (0.055s/iteration)
- **Image encoding** is fast (2s once)

**Action items:**
1. Optimize VAE decode (chunking, offloading)
2. Cache text encodings
3. Don't bother optimizing transformer (already fast)

### Memory Report

```
================================================================================
MEMORY USAGE SUMMARY
================================================================================
Label                                          Allocated     Reserved    Max Alloc
--------------------------------------------------------------------------------
worker_start                                      1024.5M     2048.0M     1024.5M
after_text_encoding                               4096.2M     6144.0M     4100.3M
after_image_encoding                              5120.8M     8192.0M     5125.1M
during_sampling                                  12288.4M    16384.0M    14336.2M
after_vae_decode                                  2048.1M     4096.0M    14336.2M
================================================================================
```

**What this tells you:**
- Sampling uses **14GB peak** (the real bottleneck!)
- After decode, memory drops to **2GB** (good offloading)
- Reserved memory suggests **fragmentation** (6GB → 16GB jump)

**Action items:**
1. Reduce sampling VRAM (smaller batches, gradient checkpointing)
2. Investigate fragmentation (memory_v2 backend might help)

## Common Profiling Scenarios

### Scenario 1: Slow Generation (High s/it)

**Profile this:**
```bash
python FramePack/demo_gradio.py --enable-profiling --profiling-iterations 5
```

**Look for:**
- Which operation takes the most **total** time?
- Are there **unexpected** slow operations?
- Is **SSD I/O** the bottleneck? (check system monitor)

### Scenario 2: High VRAM Usage

**Profile this:**
```python
from diffusers_helper.profiling import MemoryTracker

tracker = MemoryTracker()
# Add snapshots throughout your code
tracker.snapshot("label")
tracker.print_summary()
```

**Look for:**
- Where does memory **spike**?
- Is memory **freed** after operations?
- Are models being **offloaded** properly?

### Scenario 3: Cache Effectiveness

**Check cache hit rates:**
```bash
# The cache event recorder already tracks this
# Look for cache timeline in output
```

**Look for:**
- Are caches being **hit** or **missed**?
- Is semantic cache threshold too **strict**?
- Are you **wasting memory** on ineffective caches?

## Interpreting Chrome Traces

**Open trace in chrome://tracing**

### What to Look For:

#### 1. GPU Utilization
- **Green bars** = CUDA kernels running (good!)
- **White gaps** = GPU idle (bad - find why)
- **Stacked bars** = Memory transfers blocking compute

#### 2. CPU/GPU Overlap
- CPU and GPU should run **in parallel**
- If GPU waits for CPU = **bottleneck in data prep**
- If CPU waits for GPU = **normal** (compute bound)

#### 3. Memory Transfers (H2D/D2H)
- **H2D** (Host to Device) = CPU → GPU transfer
- **D2H** (Device to Host) = GPU → CPU transfer
- Many transfers = **memory offloading overhead**

#### 4. Attention Operations
- Look for `flash_attn`, `mem_efficient_attn`, `sdpa`
- If you see `eager` attention = **not using fast path**
- If attention is huge portion = **consider optimization**

## Real Example: Debugging the BetterTransformer Regression

**What profiling would have shown:**

```
BEFORE BetterTransformer:
vae_decode          12 ops    45s total    (with SSD offloading)
ssd_read_events               2.1 GB/s     (active)
gpu_memory_peak               8.2 GB       (efficient)

AFTER BetterTransformer:
vae_decode          12 ops    428s total   (blocked offloading)
ssd_read_events               0.03 GB/s    (!!!)
gpu_memory_peak               23.8 GB      (!!!)
sdpa_overhead                 +380s total  (forced GPU residency)
```

**Profiling would have caught this immediately!**

## Best Practices

### 1. Profile Your Actual Workload
```bash
# Not useful:
--profiling-iterations 1  # Too little data

# Better:
--profiling-iterations 3-5  # Statistical validity
```

### 2. Profile Before and After Optimizations
```bash
# Baseline
python demo_gradio.py --enable-profiling > baseline_profile.txt

# After optimization
FRAMEPACK_ENABLE_SOMETHING=1 python demo_gradio.py --enable-profiling > optimized_profile.txt

# Compare
diff baseline_profile.txt optimized_profile.txt
```

### 3. Monitor System-Level Metrics

**While profiling, run:**
```bash
# GPU utilization
nvidia-smi dmon -s ucm -i 0

# Disk I/O
iotop -o

# CPU usage
htop
```

### 4. Focus on Total Time, Not Mean Time

**Bad assumption:**
"Operation X is 0.001s, I'll optimize it!"

**Better:**
"Operation X is 0.001s × 10,000 iterations = **10 seconds total** - worth optimizing!"

## Profiling API Reference

### TimingStats

```python
from diffusers_helper.profiling import get_global_stats

stats = get_global_stats()

# Record timing
with profile_section("my_op"):
    expensive_function()

# Get stats
stats.print_summary(top_n=20)
all_stats = stats.get_all_stats()
```

### PyTorchProfiler

```python
from diffusers_helper.profiling import PyTorchProfiler

profiler = PyTorchProfiler(
    output_dir="./results",
    record_shapes=True,      # Include tensor shapes
    profile_memory=True,     # Track memory
    with_stack=False,        # Stack traces (expensive)
)

with profiler.profile("my_trace"):
    run_inference()

# Exports to: ./results/my_trace.json
```

### MemoryTracker

```python
from diffusers_helper.profiling import MemoryTracker

tracker = MemoryTracker()

tracker.snapshot("start")
load_models()
tracker.snapshot("after_load")

tracker.print_summary()
tracker.export_json("memory_log.json")
```

### IterationProfiler

```python
from diffusers_helper.profiling import IterationProfiler

profiler = IterationProfiler("denoising_steps")

for step in range(num_steps):
    with profiler.iteration():
        denoise_step()

profiler.print_summary()
# Shows: mean, min, max, median time per iteration
```

## Output Files

Profiling generates:

```
profiling_results/
├── trace.json                  # Chrome trace (open in chrome://tracing)
├── profiling_report.json       # JSON summary of all metrics
├── memory_log.json             # Memory snapshots over time
└── timing_summary.txt          # Human-readable timing stats
```

## Advanced: Custom Profiling

### Profile Specific Code Sections

```python
from diffusers_helper.profiling import profile_section, get_global_stats

# Wrap critical sections
with profile_section("text_encoding"):
    text_embeds = encode_text(prompt)

with profile_section("vae_decode_chunk_1"):
    frame = vae.decode(latent_chunk)

# Later, analyze
stats = get_global_stats()
print(f"Text encoding took: {stats.get_stats('text_encoding')['total']}s")
```

### Decorator-Based Profiling

```python
from diffusers_helper.profiling import profile_function

@profile_function("my_expensive_function")
def process_video(frames):
    # ... expensive work ...
    return result

# Automatically tracked!
```

## Troubleshooting

### Issue: Profiling makes everything slower

**Solution**: Profiling adds ~5-10% overhead. For accurate timing:
1. Run 1 warmup iteration (don't profile)
2. Then run 3-5 profiled iterations
3. Use `with_stack=False` to reduce overhead

### Issue: Chrome trace file is huge (>1GB)

**Solution**:
- Reduce `--profiling-iterations`
- Set `record_shapes=False`
- Profile only specific sections

### Issue: Out of memory during profiling

**Solution**:
- Profiling stores traces in memory
- Use `profile_memory=False` to reduce overhead
- Profile shorter runs

## Conclusion

**The Golden Rule of Optimization:**

> **"Profile first, optimize second, measure third"**

Don't be like BetterTransformer - measure your actual bottlenecks before optimizing!

### Key Takeaways

1. ✅ **Always profile before optimizing**
2. ✅ **Focus on total time**, not individual operation speed
3. ✅ **Test with real workloads**, not synthetic benchmarks
4. ✅ **Memory patterns matter** - don't break offloading!
5. ✅ **Compare before/after** to validate improvements

### Next Steps

1. Run baseline profiling on your current setup
2. Identify the top 3 bottlenecks
3. Research optimizations for those specific bottlenecks
4. Profile again to confirm improvements
5. Repeat!
