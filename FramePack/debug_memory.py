#!/usr/bin/env python
"""Debug script to monitor memory usage during FramePack initialization."""

import os
import sys
import psutil
import time

# Set same environment variables as your failing command
os.environ['FRAMEPACK_TRT_WORKSPACE_MB'] = '1024'
os.environ['PYTORCH_ENABLE_MEM_EFFICIENT_SDP'] = '0'
os.environ['PYTORCH_ENABLE_FLASH_SDP'] = '0'
os.environ['FRAMEPACK_PRELOAD_REPOS'] = '0'
os.environ['FRAMEPACK_FAST_START'] = '0'
os.environ['FRAMEPACK_USE_BNB'] = '0'
os.environ['FRAMEPACK_BNB_LOAD_IN_4BIT'] = '0'
os.environ['FRAMEPACK_BNB_CPU_OFFLOAD'] = '0'
os.environ['FRAMEPACK_VAE_CHUNK_SIZE'] = '2'

def get_memory_info():
    """Get current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()

    return {
        'rss_gb': mem_info.rss / (1024**3),  # Resident Set Size (actual RAM used)
        'vms_gb': mem_info.vms / (1024**3),  # Virtual Memory Size
        'available_gb': virtual_mem.available / (1024**3),
        'percent': virtual_mem.percent,
    }

def get_gpu_memory():
    """Get GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated,
            }
    except:
        pass
    return None

def monitor_import(module_name, description):
    """Monitor memory during module import."""
    print(f"\n{'='*80}")
    print(f"Loading: {description}")
    print(f"{'='*80}")

    mem_before = get_memory_info()
    gpu_before = get_gpu_memory()

    print(f"Before - RAM: {mem_before['rss_gb']:.2f} GB used, {mem_before['available_gb']:.2f} GB available ({mem_before['percent']:.1f}% used)")
    if gpu_before:
        print(f"Before - GPU: {gpu_before['allocated_gb']:.2f} GB allocated, {gpu_before['free_gb']:.2f} GB free")

    start_time = time.time()

    try:
        if module_name == 'demo_gradio':
            # Don't actually import, just report
            print("Skipping actual demo_gradio import - run manually")
            return
        else:
            __import__(module_name)
        elapsed = time.time() - start_time

        mem_after = get_memory_info()
        gpu_after = get_gpu_memory()

        print(f"After  - RAM: {mem_after['rss_gb']:.2f} GB used, {mem_after['available_gb']:.2f} GB available ({mem_after['percent']:.1f}% used)")
        if gpu_after:
            print(f"After  - GPU: {gpu_after['allocated_gb']:.2f} GB allocated, {gpu_after['free_gb']:.2f} GB free")

        ram_delta = mem_after['rss_gb'] - mem_before['rss_gb']
        print(f"Delta  - RAM: {ram_delta:+.2f} GB")
        if gpu_before and gpu_after:
            gpu_delta = gpu_after['allocated_gb'] - gpu_before['allocated_gb']
            print(f"Delta  - GPU: {gpu_delta:+.2f} GB")

        print(f"Time: {elapsed:.2f}s")
        print(f"✓ Successfully loaded {description}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Failed to load {description} after {elapsed:.2f}s")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("="*80)
    print("FramePack Memory Usage Monitor")
    print("="*80)

    # Check system resources
    virtual_mem = psutil.virtual_memory()
    print(f"\nSystem RAM: {virtual_mem.total / (1024**3):.2f} GB total")
    print(f"Available: {virtual_mem.available / (1024**3):.2f} GB")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"\nGPU: {gpu_props.name}")
            print(f"GPU RAM: {gpu_props.total_memory / (1024**3):.2f} GB")
    except:
        print("\nGPU: Not available or torch not installed")

    # Monitor key imports
    monitor_import('torch', 'PyTorch')
    monitor_import('torch_tensorrt', 'TensorRT for PyTorch (this is the big one!)')
    monitor_import('transformers', 'Transformers library')
    monitor_import('diffusers', 'Diffusers library')

    print("\n" + "="*80)
    print("Pre-import monitoring complete!")
    print("="*80)

    mem_now = get_memory_info()
    gpu_now = get_gpu_memory()

    print(f"\nCurrent state:")
    print(f"RAM: {mem_now['rss_gb']:.2f} GB used, {mem_now['available_gb']:.2f} GB available")
    if gpu_now:
        print(f"GPU: {gpu_now['allocated_gb']:.2f} GB allocated, {gpu_now['free_gb']:.2f} GB free")

    print("\n" + "="*80)
    print("Recommendations:")
    print("="*80)

    if mem_now['available_gb'] < 16:
        print("⚠ WARNING: Less than 16 GB RAM available")
        print("  TensorRT transformer compilation may fail due to insufficient RAM")
        print("  Consider:")
        print("    1. Close other applications")
        print("    2. Add more swap space")
        print("    3. Remove --tensorrt-transformer flag (VAE-only TensorRT)")

    if gpu_now and gpu_now['free_gb'] < 12:
        print("⚠ WARNING: Less than 12 GB GPU RAM available")
        print("  TensorRT transformer may not have enough VRAM")
        print("  Consider:")
        print("    1. Use lower workspace: FRAMEPACK_TRT_WORKSPACE_MB=512")
        print("    2. Remove --tensorrt-transformer flag")

    print("\nTo continue testing, run:")
    print("python demo_gradio.py --fast-start --xformers-mode aggressive --use-memory-v2 --enable-tensorrt --tensorrt-transformer")
