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


def _default_trt_cache_dir() -> Path:
    override = os.environ.get('FRAMEPACK_ONNX_TRT_CACHE_DIR') or os.environ.get('FRAMEPACK_TENSORRT_CACHE_DIR')
    if override:
        return Path(override)
    return Path(os.path.join(os.path.dirname(__file__), '..', 'Cache', 'tensorrt_engines'))


class ONNXRuntimeModel:
    """Wrapper for loading and running ONNX models with ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str,
        *,
        use_gpu: bool = True,
        enable_tensorrt: bool = False,
        trt_device_id: int = 0,
        trt_workspace_size_mb: int = 4096,
        trt_fp16: bool = True,
        trt_int8: bool = False,
        trt_cache_dir: Optional[str] = None,
    ):
        """
        Initialize ONNX Runtime model from file.

        Args:
            onnx_path: Path to the .onnx file
            use_gpu: Whether to use GPU (CUDA) execution provider
            enable_tensorrt: Try to enable TensorRT execution provider (if available)
            trt_device_id: GPU device index
            trt_workspace_size_mb: TensorRT workspace size (MB)
            trt_fp16: Enable FP16 kernels for TensorRT
            trt_int8: Enable INT8 kernels for TensorRT
            trt_cache_dir: Cache folder for TensorRT engine plans
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("ONNX Runtime is not available. Install onnxruntime or onnxruntime-gpu.")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        try:
            available_providers = ort.get_available_providers()
        except Exception:
            available_providers = []

        providers = []
        using_gpu = False

        if (
            enable_tensorrt
            and use_gpu
            and torch.cuda.is_available()
            and "TensorrtExecutionProvider" in available_providers
        ):
            cache_path = Path(trt_cache_dir) if trt_cache_dir else _default_trt_cache_dir()
            cache_path.mkdir(parents=True, exist_ok=True)
            trt_options = {
                "device_id": int(trt_device_id),
                "trt_max_workspace_size": int(max(256, trt_workspace_size_mb)) * 1024 * 1024,
                "trt_fp16_enable": bool(trt_fp16),
                "trt_int8_enable": bool(trt_int8),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_path),
                "trt_timing_cache_enable": True,
                "trt_force_sequential_engine_build": False,
                "trt_max_partition_iterations": 1000,
                "trt_min_subgraph_size": 1,
            }
            providers.append(("TensorrtExecutionProvider", trt_options))
            using_gpu = True

        if use_gpu and torch.cuda.is_available() and "CUDAExecutionProvider" in available_providers:
            providers.append(("CUDAExecutionProvider", {"device_id": int(trt_device_id)}))
            using_gpu = True

        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            if providers and isinstance(providers[0], tuple) and providers[0][0] == "TensorrtExecutionProvider":
                print(f"TensorRT Execution Provider failed to initialize ({e}); falling back to CUDA/CPU.")
                fallback_providers = [p for p in providers if not (isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider")]
                self.session = ort.InferenceSession(
                    str(self.onnx_path),
                    sess_options=sess_options,
                    providers=fallback_providers,
                )
                using_gpu = any(
                    (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider") or p == "CUDAExecutionProvider"
                    for p in fallback_providers
                )
            else:
                raise RuntimeError(f"Failed to load ONNX model from {onnx_path}: {e}")

        self._using_gpu_provider = using_gpu
        session_providers = self.session.get_providers()
        self.provider = session_providers[0] if session_providers else "CPUExecutionProvider"
        if enable_tensorrt and "TensorrtExecutionProvider" not in session_providers:
            print("TensorRT provider unavailable, using CUDA/CPU fallback.")

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
            device = 'cuda' if self._using_gpu_provider else 'cpu'
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
    use_gpu: bool = True,
    enable_tensorrt: bool = False,
    trt_device_id: int = 0,
    trt_workspace_size_mb: int = 4096,
    trt_fp16: bool = True,
    trt_int8: bool = False,
    trt_cache_dir: Optional[str] = None,
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
        return ONNXRuntimeModel(
            str(onnx_path),
            use_gpu=use_gpu,
            enable_tensorrt=enable_tensorrt,
            trt_device_id=trt_device_id,
            trt_workspace_size_mb=trt_workspace_size_mb,
            trt_fp16=trt_fp16,
            trt_int8=trt_int8,
            trt_cache_dir=trt_cache_dir,
        )
    except Exception as e:
        print(f"Failed to load ONNX model for {model_name}: {e}")
        return None
