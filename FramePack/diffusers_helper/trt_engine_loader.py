"""
TensorRT engine loader for pre-built engines.

This module provides functionality to load and use pre-built TensorRT engines
that were created from ONNX models using the optimize_onnx_with_tensorrt.py script.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch


try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Automatically initializes CUDA
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None
    cuda = None


class TensorRTEngine:
    """Wrapper for loading and running TensorRT engines."""

    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine from file.

        Args:
            engine_path: Path to the .engine file
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Install tensorrt and pycuda.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        # Load the engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # Get input/output information
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(tensor_name)
            shape = self.engine.get_tensor_shape(tensor_name)

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'dtype': dtype,
                    'shape': shape,
                    'index': i
                })
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'dtype': dtype,
                    'shape': shape,
                    'index': i
                })

        print(f"Loaded TensorRT engine: {self.engine_path.name}")
        print(f"  Inputs: {[inp['name'] for inp in self.inputs]}")
        print(f"  Outputs: {[out['name'] for out in self.outputs]}")

    def _numpy_to_torch_dtype(self, trt_dtype):
        """Convert TensorRT dtype to PyTorch dtype."""
        if trt_dtype == trt.float32:
            return torch.float32
        elif trt_dtype == trt.float16:
            return torch.float16
        elif trt_dtype == trt.int32:
            return torch.int32
        elif trt_dtype == trt.int64:
            return torch.int64
        else:
            return torch.float32

    def infer(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference on the engine.

        Args:
            **inputs: Named input tensors (must match engine input names)

        Returns:
            Dictionary of output tensors
        """
        # Validate inputs
        for inp in self.inputs:
            if inp['name'] not in inputs:
                raise ValueError(f"Missing required input: {inp['name']}")

        # Set input shapes if dynamic
        for inp in self.inputs:
            input_tensor = inputs[inp['name']]
            self.context.set_input_shape(inp['name'], tuple(input_tensor.shape))

        # Allocate buffers
        device_buffers = {}
        host_outputs = {}

        for inp in self.inputs:
            input_tensor = inputs[inp['name']]
            # Ensure tensor is contiguous and on GPU
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()

            device_buffers[inp['name']] = input_tensor.data_ptr()
            self.context.set_tensor_address(inp['name'], device_buffers[inp['name']])

        # Allocate output buffers
        for out in self.outputs:
            output_shape = self.context.get_tensor_shape(out['name'])
            torch_dtype = self._numpy_to_torch_dtype(out['dtype'])
            output_tensor = torch.empty(tuple(output_shape), dtype=torch_dtype, device='cuda')

            device_buffers[out['name']] = output_tensor.data_ptr()
            host_outputs[out['name']] = output_tensor
            self.context.set_tensor_address(out['name'], device_buffers[out['name']])

        # Execute inference
        success = self.context.execute_async_v3(self.stream.handle)
        if not success:
            raise RuntimeError("TensorRT execution failed")

        # Synchronize stream
        self.stream.synchronize()

        return host_outputs

    def __call__(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference."""
        return self.infer(**inputs)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'context') and self.context:
            del self.context
        if hasattr(self, 'engine') and self.engine:
            del self.engine
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime


def find_engine_for_model(model_name: str, cache_dir: Optional[str] = None) -> Optional[Path]:
    """
    Find a TensorRT engine file for a given model name.

    Args:
        model_name: Name of the model (e.g., 'siglip_image_encoder', 'vae_decoder')
        cache_dir: Optional cache directory to search in

    Returns:
        Path to engine file if found, None otherwise
    """
    if cache_dir is None:
        cache_dir = os.environ.get('FRAMEPACK_TENSORRT_CACHE_DIR')
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), '..', 'Cache', 'tensorrt_engines'
            )

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    # Look for engine files matching the model name
    pattern = f"{model_name}_*.engine"
    matches = list(cache_path.glob(pattern))

    if matches:
        # Return the most recent one
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return None


def load_engine_if_available(model_name: str, cache_dir: Optional[str] = None) -> Optional[TensorRTEngine]:
    """
    Load a TensorRT engine if available.

    Args:
        model_name: Name of the model
        cache_dir: Optional cache directory

    Returns:
        TensorRTEngine instance if found and loaded successfully, None otherwise
    """
    if not TRT_AVAILABLE:
        return None

    engine_path = find_engine_for_model(model_name, cache_dir)
    if engine_path is None:
        return None

    try:
        return TensorRTEngine(str(engine_path))
    except Exception as e:
        print(f"Failed to load TensorRT engine for {model_name}: {e}")
        return None
