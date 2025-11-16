#!/usr/bin/env python3
"""
Utility script to convert ONNX checkpoint(s) into .safetensors files.
Understands the naming/layout produced by FramePack/convert_to_onnx.py and
can batch-convert a directory of component ONNX files back to safetensors.

⚠️  WARNING: This script extracts ONNX weights to safetensors format for debugging/analysis.
    It CANNOT reconstruct a working Stable Diffusion checkpoint!

    - ONNX weight names differ from PyTorch state dict keys
    - Separate components (UNet, VAE, CLIP) remain separate
    - Missing metadata/configuration from original checkpoint
    - Resulting files CANNOT be loaded by SD WebUI or ComfyUI

    Use ONNX files directly with ONNX Runtime or TensorRT instead.

Examples:
  # Single file (original behaviour) - for debugging only
  python convert_to_safetensor.py Cache/test/unet_indigo.onnx Cache/test/unet_indigo.safetensors

  # Convert every ONNX file produced by convert_to_onnx.py in a directory
  python convert_to_safetensor.py Cache/test Cache/safetensors --sd-components

  # Recursively convert all *.onnx (keeping relative layout) and overwrite existing outputs
  python convert_to_safetensor.py Cache/test Cache/safetensors --recursive --overwrite
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import save_file

SD_COMPONENT_PREFIXES = {'unet', 'vae_encoder', 'vae_decoder', 'clip_text'}


def _onnx_to_torch_tensors(onnx_path: Path) -> Dict[str, torch.Tensor]:
    model = onnx.load(str(onnx_path), load_external_data=True)
    state: Dict[str, torch.Tensor] = {}

    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        tensor = torch.from_numpy(array)
        name = initializer.name or f"param_{len(state)}"
        if name in state:
            raise ValueError(f"Duplicate ONNX initializer name detected in {onnx_path}: {name}")
        state[name] = tensor

    if not state:
        raise RuntimeError(f"No initializers were found in the ONNX file: {onnx_path}")
    return state


def _should_convert(path: Path, filters: Optional[Sequence[str]]) -> bool:
    if not filters:
        return True
    prefix = path.stem.split('_', 1)[0]
    return prefix in filters


def _build_conversion_tasks(
    input_path: Path,
    output_path: Optional[Path],
    recursive: bool,
    filters: Optional[Sequence[str]],
) -> List[Tuple[Path, Path]]:
    tasks: List[Tuple[Path, Path]] = []

    if input_path.is_file():
        if input_path.suffix.lower() != '.onnx':
            raise ValueError(f"Input file does not have .onnx extension: {input_path}")
        if filters and not _should_convert(input_path, filters):
            print(f"Skipping {input_path.name}: component prefix not in filter list")
            return tasks

        if output_path:
            if output_path.is_dir():
                dest = output_path / input_path.with_suffix('.safetensors').name
            else:
                dest = output_path
        else:
            dest = input_path.with_suffix('.safetensors')

        tasks.append((input_path, dest))
        return tasks

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern = '**/*.onnx' if recursive else '*.onnx'
    onnx_files = sorted(input_path.glob(pattern))
    if not onnx_files:
        raise RuntimeError(f"No ONNX files found in {input_path} (recursive={recursive}).")

    dest_root = output_path if output_path else input_path
    dest_root.mkdir(parents=True, exist_ok=True)

    for src in onnx_files:
        if not src.is_file():
            continue
        if not _should_convert(src, filters):
            continue

        relative = src.relative_to(input_path)
        dest = dest_root / relative
        dest = dest.with_suffix('.safetensors')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((src, dest))

    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ONNX weights to safetensors format (single file or entire directory)."
    )
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to an ONNX file or a directory produced by convert_to_onnx.py.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Destination file or directory. Defaults to alongside the source if omitted.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Restrict conversion to ONNX files whose filename prefix matches one of these components "
             "(e.g., unet vae_decoder clip_text). Useful for convert_to_onnx.py outputs.",
    )
    parser.add_argument(
        "--sd-components",
        action="store_true",
        help="Shortcut for --components unet vae_decoder vae_encoder clip_text.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When onnx_path is a directory, recursively search for *.onnx files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing safetensors files. Otherwise, existing files are skipped.",
    )
    return parser.parse_args()


def convert_task(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        print(f"Skipping existing file: {dst} (use --overwrite to replace)")
        return False

    print(f"\nConverting {src} -> {dst}")
    tensors = _onnx_to_torch_tensors(src)
    print(f"  Found {len(tensors)} tensors")
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst))
    print("  Done.")
    return True


def main() -> None:
    args = parse_args()

    # Print warning
    print("\n" + "="*80)
    print("⚠️  WARNING: ONNX to safetensors conversion limitation")
    print("="*80)
    print("\nThis script extracts ONNX model weights to .safetensors format.")
    print("The resulting files:")
    print("  ✗ CANNOT be loaded as Stable Diffusion checkpoints")
    print("  ✗ Have different weight names than the original checkpoint")
    print("  ✗ Are missing metadata and configuration")
    print("  ✗ Will NOT work with SD WebUI, ComfyUI, or other SD tools")
    print("\nThese .safetensors files are for debugging/analysis only.")
    print("\nTo use your model:")
    print("  ✓ Use the ONNX files directly with ONNX Runtime")
    print("  ✓ Use the ONNX files with TensorRT")
    print("  ✓ Keep your original checkpoint for SD WebUI/ComfyUI")
    print("\n" + "="*80 + "\n")

    input_path = Path(os.path.abspath(args.onnx_path))
    output_path = Path(os.path.abspath(args.output_path)) if args.output_path else None

    component_filters: Optional[List[str]] = None
    if args.sd_components:
        component_filters = sorted(SD_COMPONENT_PREFIXES)
    if args.components:
        component_filters = sorted(
            set(args.components).union(component_filters or [])
        )

    tasks = _build_conversion_tasks(
        input_path=input_path,
        output_path=output_path,
        recursive=args.recursive,
        filters=component_filters,
    )

    if not tasks:
        print("No ONNX files matched the provided filters. Nothing to do.")
        return

    converted = 0
    for src, dst in tasks:
        if convert_task(src, dst, overwrite=args.overwrite):
            converted += 1

    print(f"\nConversion complete. Converted {converted}/{len(tasks)} file(s).")


if __name__ == "__main__":
    main()
