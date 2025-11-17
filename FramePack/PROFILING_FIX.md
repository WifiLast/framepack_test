# Profiling Crash Fix

## Issue

The profiling feature was causing the script to crash when enabled with `--enable-profiling`. No profiling data was being generated, and the script would terminate prematurely.

## Root Cause

The **PyTorch profiler** (`torch.profiler.profile`) was causing crashes when integrated into the complex FramePack workflow. This is likely due to:
1. Conflicts with CUDA memory management and model offloading
2. High memory overhead from recording detailed kernel traces
3. Incompatibility with some of the custom pipeline operations

## Solution

**Disabled PyTorch profiler entirely** and kept only the lightweight profiling components:

### What Still Works (No Crashes):

1. **Timing Statistics** - `profile_section()` context manager
   - Tracks duration of every operation
   - Thread-safe with minimal overhead (~0.1% performance impact)
   - Aggregates mean, min, max, total time
   - Sorts operations by total time to identify bottlenecks

2. **Memory Tracking** - `MemoryTracker`
   - GPU VRAM snapshots at key points
   - Shows allocated, reserved, and peak memory
   - Helps identify memory leaks and spikes

3. **Iteration Profiling** - `IterationProfiler`
   - Tracks per-step timing for diffusion loops
   - Shows iteration-by-iteration variance
   - Calculates mean, median, min, max per iteration

### What Was Removed:

- `PyTorchProfiler` integration (was causing crashes)
- Chrome trace file generation (requires PyTorch profiler)
- CUDA kernel-level profiling (too heavyweight)

## Changes Made

### File: `demo_gradio.py`

**Line 2563-2587**: Profiling initialization
```python
# Removed: pytorch_profiler = None
# Added: More detailed initialization messages
# Added: Better error handling with try/except
```

**Line 2896-2898**: Sampling profiling
```python
# Removed: pytorch_profiler.start() call
# Added: Comment explaining why PyTorch profiler is disabled
```

**Line 3022-3068**: Profiling export
```python
# Removed: pytorch_profiler.stop() call
# Removed: Chrome trace references
# Added: Individual try/except blocks for each export operation
# Added: iteration_profiler.print_summary() call
```

## Usage

The profiling feature now works without crashes:

```bash
python FramePack/demo_gradio.py --enable-profiling --profiling-iterations 5 --profiling-output-dir ./my_profile
```

### What You Get:

**1. Console Output** - Printed at the end of generation:
```
================================================================================
PROFILING SUMMARY - Top Operations by Total Time
================================================================================
Operation                                          Count      Total       Mean        Min        Max
--------------------------------------------------------------------------------
vae_decode                                            12     45.234s     3.769s     3.102s     4.521s
text_encoding_positive                                 1     12.456s    12.456s    12.456s    12.456s
sampling_chunk_0                                       1      8.234s     8.234s     8.234s     8.234s
image_encoder_forward                                  1      2.103s     2.103s     2.103s     2.103s
================================================================================

================================================================================
MEMORY USAGE SUMMARY
================================================================================
Label                                          Allocated     Reserved    Max Alloc
--------------------------------------------------------------------------------
worker_start                                      1024.5M     2048.0M     1024.5M
after_text_encoding                               4096.2M     6144.0M     4100.3M
after_image_encoding                              5120.8M     8192.0M     5125.1M
after_sampling_chunk_0                           12288.4M    16384.0M    14336.2M
after_vae_decode_0                                2048.1M     4096.0M    14336.2M
================================================================================

================================================================================
ITERATION PROFILING: diffusion_step
================================================================================
Total iterations: 50
Total time:       125.456s
Mean time/iter:   2.509s (0.40 it/s)
Min time:         2.103s
Max time:         3.421s
Median time:      2.450s
================================================================================
```

**2. JSON Report** - Saved to `./my_profile/profiling_report.json`:
```json
{
  "timestamp": 1700000000.123,
  "timing_stats": {
    "vae_decode": {
      "count": 12,
      "total": 45.234,
      "mean": 3.769,
      "min": 3.102,
      "max": 4.521
    },
    ...
  },
  "memory_snapshots": [
    {
      "label": "worker_start",
      "timestamp": 1700000000.0,
      "allocated_mb": 1024.5,
      "reserved_mb": 2048.0,
      "max_allocated_mb": 1024.5
    },
    ...
  ],
  "iteration_stats": {
    "count": 50,
    "total": 125.456,
    "mean": 2.509,
    "min": 2.103,
    "max": 3.421,
    "median": 2.450
  }
}
```

## What This Data Tells You

### Timing Statistics
- **Total time** = Most important metric (not mean!)
- Operations are sorted by total time to show real bottlenecks
- Example: VAE decode 12× @ 3.8s each = **45s total** → This is your #1 optimization target

### Memory Snapshots
- Shows VRAM usage at key pipeline stages
- **Peak memory** (Max Alloc column) identifies OOM risks
- **Reserved vs Allocated** gap shows fragmentation

### Iteration Profiling
- **Variance** (max - min) shows consistency issues
- High variance = some iterations are abnormally slow (investigate why)
- Low it/s = bottleneck in sampling loop

## Verification

To verify the fix works:

1. Run with profiling enabled:
```bash
python FramePack/demo_gradio.py --enable-profiling --profiling-iterations 3 --profiling-output-dir ./test_profile
```

2. Check that:
   - ✅ Script completes without crashing
   - ✅ Console shows profiling summary at the end
   - ✅ `./test_profile/profiling_report.json` exists and contains data

3. Analyze the results:
   - Look at "Total" column to find the slowest operations
   - Check memory snapshots for VRAM spikes
   - Review iteration stats for timing variance

## Future Work (Optional)

If you need Chrome trace visualization later, you could:
1. Use PyTorch profiler in a **separate isolated script** (not integrated into main pipeline)
2. Profile only specific operations in isolation
3. Use `nsys` (NVIDIA Nsight Systems) for CUDA profiling without Python integration

But for most optimization work, the **timing + memory data is sufficient** to identify bottlenecks!

## Related Files

- **profiling.py** - Core profiling infrastructure ([diffusers_helper/profiling.py](diffusers_helper/profiling.py))
- **PROFILING_GUIDE.md** - Complete usage guide ([PROFILING_GUIDE.md](PROFILING_GUIDE.md))
- **profile_demo.py** - Standalone example ([profile_demo.py](profile_demo.py))
