"""
ONNX Runtime loader for running ONNX models.

This module provides functionality to load and run ONNX models using ONNX Runtime,
as a fallback when TensorRT engines are not available.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch


try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ort = None


class ONNXRuntimeModel:
    """Wrapper for loading and running ONNX models with ONNX Runtime."""

    def __init__(self, onnx_path: str, use_gpu: bool = True):
        """
        Initialize ONNX Runtime model from file.

        Args:
            onnx_path: Path to the .onnx file
            use_gpu: Whether to use GPU (CUDA) execution provider
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available. Install onnxruntime or onnxruntime-gpu.")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # Set up execution providers
        providers = []
        if use_gpu and torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {onnx_path}: {e}")

        # Get input/output information
        self.inputs = {}
        for inp in self.session.get_inputs():
            self.inputs[inp.name] = {
                'name': inp.name,
                'shape': inp.shape,
                'dtype': inp.type
            }

        self.outputs = {}
        for out in self.session.get_outputs():
            self.outputs[out.name] = {
                'name': out.name,
                'shape': out.shape,
                'dtype': out.type
            }

        # Determine which execution provider is being used
        self.provider = self.session.get_providers()[0]

        print(f"Loaded ONNX model: {self.onnx_path.name}")
        print(f"  Provider: {self.provider}")
        print(f"  Inputs: {list(self.inputs.keys())}")
        print(f"  Outputs: {list(self.outputs.keys())}")

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()

    def _to_torch(self, array: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        tensor = torch.from_numpy(array)
        if device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor

    def infer(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run inference on the ONNX model.

        Args:
            **inputs: Named input tensors (must match ONNX model input names)

        Returns:
            Dictionary of output tensors
        """
        # Validate inputs
        for input_name in self.inputs.keys():
            if input_name not in inputs:
                raise ValueError(f"Missing required input: {input_name}")

        # Convert PyTorch tensors to numpy
        onnx_inputs = {}
        for name, tensor in inputs.items():
            onnx_inputs[name] = self._to_numpy(tensor)

        # Run inference
        output_names = list(self.outputs.keys())
        onnx_outputs = self.session.run(output_names, onnx_inputs)

        # Convert outputs back to PyTorch tensors
        outputs = {}
        for i, name in enumerate(output_names):
            # Keep outputs on CPU if using CPU provider, otherwise move to GPU
            device = 'cuda' if self.provider == 'CUDAExecutionProvider' else 'cpu'
            outputs[name] = self._to_torch(onnx_outputs[i], device=device)

        return outputs

    def __call__(self, **inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference."""
        return self.infer(**inputs)


def find_onnx_model(model_name: str, cache_dir: Optional[str] = None) -> Optional[Path]:
    """
    Find an ONNX model file for a given model name.

    Args:
        model_name: Name of the model (e.g., 'siglip_image_encoder', 'vae_decoder')
        cache_dir: Optional cache directory to search in

    Returns:
        Path to ONNX file if found, None otherwise
    """
    if cache_dir is None:
        cache_dir = os.environ.get('FRAMEPACK_ONNX_CACHE_DIR')
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), '..', 'Cache', 'onnx_models'
            )

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    # Look for ONNX files matching the model name
    pattern = f"{model_name}_*.onnx"
    matches = list(cache_path.glob(pattern))

    if matches:
        # Return the most recent one
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    return None


def load_onnx_model_if_available(
    model_name: str,
    cache_dir: Optional[str] = None,
    use_gpu: bool = True
) -> Optional[ONNXRuntimeModel]:
    """
    Load an ONNX model if available.

    Args:
        model_name: Name of the model
        cache_dir: Optional cache directory
        use_gpu: Whether to use GPU execution

    Returns:
        ONNXRuntimeModel instance if found and loaded successfully, None otherwise
    """
    if not ONNXRUNTIME_AVAILABLE:
        return None

    onnx_path = find_onnx_model(model_name, cache_dir)
    if onnx_path is None:
        return None

    try:
        return ONNXRuntimeModel(str(onnx_path), use_gpu=use_gpu)
    except Exception as e:
        print(f"Failed to load ONNX model for {model_name}: {e}")
        return None
