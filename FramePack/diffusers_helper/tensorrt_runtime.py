import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    import torch_tensorrt
    from torch_tensorrt import Input as TRTInput
except Exception:  # pragma: no cover - optional dependency
    torch_tensorrt = None
    TRTInput = None


class TensorRTRuntime:
    """Small helper around torch_tensorrt to lazily compile modules."""

    def __init__(
        self,
        *,
        enabled: bool,
        precision: torch.dtype = torch.float16,
        workspace_size_mb: int = 4096,
        max_aux_streams: int = 2,
    ):
        self.requested = bool(enabled)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.compute_dtype = precision if precision in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        self.workspace_size = max(256, int(workspace_size_mb)) * 1024 * 1024
        self.max_aux_streams = max(1, int(max_aux_streams))
        self.failure_reason: Optional[str] = None
        self._modules: Dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()

        if not self.requested:
            self.failure_reason = "TensorRT runtime not requested."
            self.enabled = False
        elif torch_tensorrt is None:
            self.failure_reason = "torch_tensorrt is not installed."
            self.enabled = False
        elif not torch.cuda.is_available():
            self.failure_reason = "CUDA device is required for TensorRT acceleration."
            self.enabled = False
        else:
            self.enabled = True

    def disable(self, reason: str) -> None:
        self.failure_reason = reason
        self.enabled = False
        self._modules.clear()

    @property
    def is_ready(self) -> bool:
        return self.enabled

    def make_input_from_shape(self, shape: Tuple[int, ...], *, name: Optional[str] = None) -> "TRTInput":
        if TRTInput is None:
            raise RuntimeError("torch_tensorrt.Input is unavailable.")
        return TRTInput(shape=tuple(int(dim) for dim in shape), dtype=self.compute_dtype, name=name)

    def get_or_compile(self, name: str, module: nn.Module, inputs) -> nn.Module:
        if not self.enabled:
            raise RuntimeError(self.failure_reason or "TensorRT runtime disabled.")

        with self._lock:
            existing = self._modules.get(name)
            if existing is not None:
                return existing

            module = module.to(device=self.device, dtype=self.compute_dtype)
            module.eval()

            try:
                compiled = torch_tensorrt.dynamo.compile(
                    module,
                    inputs=inputs,
                    enabled_precisions={self.compute_dtype},
                    workspace_size=self.workspace_size,
                    max_aux_streams=self.max_aux_streams,
                )
            except Exception as exc:  # pragma: no cover - CUDA/TensorRT failure surface
                self.disable(f"TensorRT compilation failed for {name}: {exc}")
                raise

            self._modules[name] = compiled
            return compiled


class VAEDecodeWrapper(nn.Module):
    """Thin wrapper to expose vae.decode as a single module for TensorRT."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.scale = float(getattr(vae.config, "scaling_factor", 1.0))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.scale
        decoded = self.vae.decode(latents).sample
        return decoded


class TensorRTLatentDecoder:
    """Caches TensorRT engines for common latent resolutions."""

    def __init__(self, vae: nn.Module, runtime: TensorRTRuntime, fallback_fn):
        self.vae = vae
        self.runtime = runtime
        self.fallback_fn = fallback_fn
        self.wrapper = VAEDecodeWrapper(vae)
        self._cache: Dict[Tuple[int, int, int, int, torch.dtype], nn.Module] = {}
        self._lock = threading.Lock()

    def _profile_key(self, shape: torch.Size, dtype: torch.dtype) -> Tuple[int, int, int, int, torch.dtype]:
        if len(shape) != 5:
            raise ValueError(f"Expected 5D latents for video VAE decode, got shape={tuple(shape)}")
        return (int(shape[0]), int(shape[2]), int(shape[3]), int(shape[4]), dtype)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.runtime.is_ready:
            return self.fallback_fn(latents, self.vae)

        key = self._profile_key(latents.shape, latents.dtype)
        with self._lock:
            engine = self._cache.get(key)
            if engine is None:
                input_spec = self.runtime.make_input_from_shape(tuple(latents.shape), name="latents")
                try:
                    engine = self.runtime.get_or_compile(f"vae_decode_{key}", self.wrapper, [input_spec])
                except Exception:
                    return self.fallback_fn(latents, self.vae)
                self._cache[key] = engine

        latents_device = latents.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True)

        with torch.no_grad():
            decoded = engine(latents_device)

        return decoded.to(dtype=torch.float32, device=latents.device)
