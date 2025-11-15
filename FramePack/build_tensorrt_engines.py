#!/usr/bin/env python3
"""
Build TensorRT engines from ONNX models.

This script builds optimized TensorRT engines from ONNX models that were
created by the demo_gradio.py script when --enable-tensorrt is used.

Usage:
    python build_tensorrt_engines.py --all
    python build_tensorrt_engines.py --model siglip_image_encoder
    python build_tensorrt_engines.py --model vae_decoder --model vae_encoder
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_cache_dir() -> Path:
    """Get the ONNX models cache directory."""
    cache_dir = os.environ.get('FRAMEPACK_ONNX_CACHE_DIR')
    if cache_dir is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), 'Cache', 'onnx_models'
        )
    return Path(cache_dir)


def get_engine_dir() -> Path:
    """Get the TensorRT engines cache directory."""
    engine_dir = os.environ.get('FRAMEPACK_TENSORRT_CACHE_DIR')
    if engine_dir is None:
        engine_dir = os.path.join(
            os.path.dirname(__file__), 'Cache', 'tensorrt_engines'
        )
    engine_path = Path(engine_dir)
    engine_path.mkdir(parents=True, exist_ok=True)
    return engine_path


def find_onnx_models(cache_dir: Path, model_name: Optional[str] = None) -> List[Path]:
    """Find ONNX models in the cache directory."""
    if model_name:
        # Look for specific model
        pattern = f"{model_name}_*.onnx"
    else:
        # Find all ONNX models
        pattern = "*.onnx"

    models = sorted(cache_dir.glob(pattern))
    return models


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gb: int = 4) -> bool:
    """Build a TensorRT engine from an ONNX model."""
    print(f"\n{'='*80}")
    print(f"Building TensorRT engine: {onnx_path.name}")
    print(f"{'='*80}")

    # Use the optimize_onnx_with_tensorrt.py script
    script_path = Path(__file__).parent / "optimize_onnx_with_tensorrt.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--onnx-path", str(onnx_path),
        "--output-path", str(engine_path),
        "--workspace-size", str(workspace_gb),
    ]

    if fp16:
        cmd.append("--fp16")
    else:
        cmd.append("--no-fp16")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to build engine for {onnx_path.name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from ONNX models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build engines for all ONNX models
  python build_tensorrt_engines.py --all

  # Build engine for specific model
  python build_tensorrt_engines.py --model siglip_image_encoder

  # Build engines for multiple models
  python build_tensorrt_engines.py --model vae_decoder --model vae_encoder

  # Build with custom workspace size
  python build_tensorrt_engines.py --all --workspace-size 8

  # Build without FP16 (use FP32)
  python build_tensorrt_engines.py --all --no-fp16

Supported models:
  - siglip_image_encoder (flux_redux image encoder)
  - vae_encoder (VAE encoder)
  - vae_decoder (VAE decoder)
  - llama_text_encoder (LLAMA text encoder, if converted)
  - clip_text_encoder (CLIP text encoder, if converted)
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Build engines for all ONNX models found in cache'
    )

    parser.add_argument(
        '--model',
        type=str,
        action='append',
        help='Build engine for specific model (can be specified multiple times)'
    )

    parser.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Enable FP16 precision (default: True)'
    )

    parser.add_argument(
        '--no-fp16',
        action='store_false',
        dest='fp16',
        help='Disable FP16 precision (use FP32 instead)'
    )

    parser.add_argument(
        '--workspace-size',
        type=int,
        default=4,
        help='Maximum workspace size in GB (default: 4)'
    )

    args = parser.parse_args()

    if not args.all and not args.model:
        parser.error("Must specify either --all or --model")

    cache_dir = get_cache_dir()
    engine_dir = get_engine_dir()

    if not cache_dir.exists():
        print(f"ERROR: ONNX cache directory not found: {cache_dir}")
        print("Please run demo_gradio.py with --enable-tensorrt first to generate ONNX models.")
        return 1

    # Find ONNX models to convert
    onnx_models = []
    if args.all:
        onnx_models = find_onnx_models(cache_dir)
        if not onnx_models:
            print(f"ERROR: No ONNX models found in {cache_dir}")
            print("Please run demo_gradio.py with --enable-tensorrt first to generate ONNX models.")
            return 1
    else:
        for model_name in args.model:
            models = find_onnx_models(cache_dir, model_name)
            if not models:
                print(f"WARNING: No ONNX model found for: {model_name}")
            onnx_models.extend(models)

    if not onnx_models:
        print("ERROR: No ONNX models to convert")
        return 1

    print(f"Found {len(onnx_models)} ONNX model(s) to convert:")
    for model in onnx_models:
        print(f"  - {model.name}")
    print()

    # Build engines
    success_count = 0
    fail_count = 0

    for onnx_path in onnx_models:
        # Generate engine path
        engine_name = onnx_path.stem + ".engine"
        engine_path = engine_dir / engine_name

        if build_engine(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspace_size):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"Build Summary")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{len(onnx_models)}")
    print(f"Failed: {fail_count}/{len(onnx_models)}")
    print(f"Engine directory: {engine_dir}")
    print(f"{'='*80}\n")

    if fail_count > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
