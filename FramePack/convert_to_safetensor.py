#!/usr/bin/env python3
"""
Utility script to convert an ONNX checkpoint into a single .safetensors file.

Example:
    python convert_to_safetensor.py model.onnx model.safetensors
"""

import argparse
import os
from typing import Dict

import onnx
import torch
from onnx import numpy_helper
from safetensors.torch import save_file


def _onnx_to_torch_tensors(onnx_path: str) -> Dict[str, torch.Tensor]:
    model = onnx.load(onnx_path, load_external_data=True)
    state: Dict[str, torch.Tensor] = {}

    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        tensor = torch.from_numpy(array)
        name = initializer.name or f"param_{len(state)}"
        if name in state:
            raise ValueError(f"Duplicate ONNX initializer name detected: {name}")
        state[name] = tensor

    if not state:
        raise RuntimeError("No initializers were found in the ONNX file; nothing to save.")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX weights to safetensors format.")
    parser.add_argument("onnx_path", type=str, help="Path to the ONNX model file.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Destination safetensors file. Will be created or overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = os.path.abspath(args.onnx_path)
    output_path = os.path.abspath(args.output_path)

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"Loading ONNX weights from {onnx_path} ...")
    tensors = _onnx_to_torch_tensors(onnx_path)
    print(f"Found {len(tensors)} tensors. Writing safetensors to {output_path} ...")
    save_file(tensors, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
