import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

_TORCH_TRT_IMPORT_ERROR: Optional[Exception]
try:
    import torch_tensorrt
    from torch_tensorrt import Input as TRTInput
    _TORCH_TRT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    torch_tensorrt = None
    TRTInput = None
    _TORCH_TRT_IMPORT_ERROR = exc


class TensorRTRuntime:
    """Small helper around torch_tensorrt to lazily compile modules with disk caching."""

    def __init__(
        self,
        *,
        enabled: bool,
        precision: torch.dtype = torch.float16,
        workspace_size_mb: int = 4096,
        max_aux_streams: int = 2,
        cache_dir: Optional[str] = None,
    ):
        self.requested = bool(enabled)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.compute_dtype = precision if precision in (torch.float16, torch.bfloat16, torch.float32) else torch.float16
        self.workspace_size = max(256, int(workspace_size_mb)) * 1024 * 1024
        self.max_aux_streams = max(1, int(max_aux_streams))
        self.failure_reason: Optional[str] = None
        self._modules: Dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()
        self._enabled = False  # Use private attribute for property

        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'Cache', 'tensorrt_engines')
        self.cache_dir = Path(cache_dir)
        if self.requested:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorRT cache directory: {self.cache_dir}")

        if not self.requested:
            self.failure_reason = "TensorRT runtime not requested."
            self._enabled = False
        elif torch_tensorrt is None:
            detail = ""
            if _TORCH_TRT_IMPORT_ERROR is not None:
                detail = f" (import error: {_TORCH_TRT_IMPORT_ERROR})"
            self.failure_reason = f"torch_tensorrt is not available{detail}."
            self._enabled = False
        elif not torch.cuda.is_available():
            self.failure_reason = "CUDA device is required for TensorRT acceleration."
            self._enabled = False
        else:
            self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if self._enabled != value:
            print(f"DEBUG TensorRTRuntime.enabled changed from {self._enabled} to {value}")
            import traceback
            traceback.print_stack()
        self._enabled = value

    def disable(self, reason: str) -> None:
        print(f"DEBUG TensorRTRuntime.disable() called! Reason: {reason}")
        import traceback
        traceback.print_stack()
        self.failure_reason = reason
        self.enabled = False
        self._modules.clear()

    @property
    def is_ready(self) -> bool:
        return self.enabled

    def _cache_key(self, name: str, input_shapes: Tuple[Tuple[int, ...], ...]) -> str:
        """Generate a cache key from name and input shapes."""
        # Include precision, workspace, and shapes in the key
        key_str = f"{name}_dtype{self.compute_dtype}_ws{self.workspace_size}"
        for shape in input_shapes:
            key_str += f"_shape{'x'.join(map(str, shape))}"
        # Hash to keep filename reasonable
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.pt"

    def _load_from_cache(self, cache_key: str) -> Optional[torch.nn.Module]:
        """Try to load a compiled module from disk cache."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            print(f"Loading TensorRT engine from cache: {cache_path.name}")
            # Load the TorchScript module
            compiled = torch.jit.load(str(cache_path), map_location=self.device)
            compiled.eval()
            return compiled
        except Exception as exc:
            print(f"Warning: Failed to load cached TensorRT engine: {exc}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None

    def _save_to_cache(self, cache_key: str, module: torch.nn.Module) -> None:
        """Save a compiled module to disk cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            print(f"Saving TensorRT engine to cache: {cache_path.name}")
            # Save as TorchScript
            torch.jit.save(module, str(cache_path))
        except Exception as exc:
            print(f"Warning: Failed to save TensorRT engine to cache: {exc}")

    def make_input_from_shape(self, shape: Tuple[int, ...], *, name: Optional[str] = None) -> "TRTInput":
        if TRTInput is None:
            raise RuntimeError("torch_tensorrt.Input is unavailable.")
        return TRTInput(shape=tuple(int(dim) for dim in shape), dtype=self.compute_dtype, name=name)

    def get_or_compile(
        self,
        name: str,
        module: nn.Module,
        *,
        input_specs=None,
        example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
    ) -> nn.Module:
        if not self.enabled:
            raise RuntimeError(self.failure_reason or "TensorRT runtime disabled.")

        with self._lock:
            # Check in-memory cache first
            existing = self._modules.get(name)
            if existing is not None:
                return existing

            # Extract input shapes for cache key
            input_shapes = []
            if input_specs is not None:
                for spec in input_specs:
                    if hasattr(spec, 'shape'):
                        input_shapes.append(tuple(spec.shape))
            elif example_inputs is not None:
                example_args, _ = example_inputs
                for arg in example_args:
                    if torch.is_tensor(arg):
                        input_shapes.append(tuple(arg.shape))

            # Generate cache key and try to load from disk
            cache_key = self._cache_key(name, tuple(input_shapes))
            cached_module = self._load_from_cache(cache_key)
            if cached_module is not None:
                self._modules[name] = cached_module
                return cached_module

            # Not in cache, need to compile
            print(f"Compiling TensorRT engine for {name} (this may take several minutes)...")
            module = module.to(device=self.device, dtype=self.compute_dtype)
            module.eval()

            compile_kwargs = dict(
                enabled_precisions={self.compute_dtype},
                workspace_size=self.workspace_size,
                max_aux_streams=self.max_aux_streams,
            )
            try:
                if input_specs is not None:
                    compiled = torch_tensorrt.dynamo.compile(
                        module,
                        inputs=input_specs,
                        **compile_kwargs,
                    )
                elif example_inputs is not None:
                    example_args, example_kwargs = example_inputs
                    compiled = torch_tensorrt.dynamo.compile(
                        module,
                        inputs=example_args,
                        **compile_kwargs,
                    )
                else:
                    raise ValueError("Either input_specs or example_inputs must be provided for TensorRT compile.")
            except Exception as exc:  # pragma: no cover - CUDA/TensorRT failure surface
                print(f"WARNING: TensorRT compilation failed for {name}: {exc}")
                print(f"         This component will fall back to PyTorch. Other TensorRT components remain active.")
                # Don't disable entire runtime - just return None and let the component fall back
                return None

            # Save to both memory and disk cache
            self._modules[name] = compiled
            self._save_to_cache(cache_key, compiled)
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
                engine = self.runtime.get_or_compile(
                    f"vae_decode_{key}",
                    self.wrapper,
                    input_specs=[input_spec],
                )
                if engine is None:
                    # Compilation failed - fall back to PyTorch
                    return self.fallback_fn(latents, self.vae)
                self._cache[key] = engine

        latents_device = latents.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True)

        with torch.no_grad():
            decoded = engine(latents_device)

        return decoded.to(dtype=torch.float32, device=latents.device)


class VAEEncodeWrapper(nn.Module):
    """Expose vae.encode as a TRT-compilable module that returns Gaussian stats."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.scale = float(getattr(vae.config, "scaling_factor", 1.0))

    def forward(self, sample: torch.Tensor):
        posterior = self.vae.encode(sample)
        latent_dist = posterior.latent_dist
        return latent_dist.mean, latent_dist.logvar


class TensorRTLatentEncoder:
    """TensorRT wrapper for VAE encoding that caches engines per input shape."""

    def __init__(self, vae: nn.Module, runtime: TensorRTRuntime, fallback_fn):
        self.vae = vae
        self.runtime = runtime
        self.fallback_fn = fallback_fn
        self.wrapper = VAEEncodeWrapper(vae)
        self._cache: Dict[Tuple[int, int, int, int, torch.dtype], nn.Module] = {}
        self._lock = threading.Lock()

    def _profile_key(self, shape: torch.Size, dtype: torch.dtype) -> Tuple[int, int, int, int, torch.dtype]:
        if len(shape) != 5:
            raise ValueError(f"Expected 5D video tensors for VAE encode, got shape={tuple(shape)}")
        return (int(shape[0]), int(shape[2]), int(shape[3]), int(shape[4]), dtype)

    def encode(self, sample: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if not self.runtime.is_ready:
            return self.fallback_fn(sample, self.vae)

        key = self._profile_key(sample.shape, sample.dtype)
        with self._lock:
            engine = self._cache.get(key)
            if engine is None:
                input_spec = self.runtime.make_input_from_shape(tuple(sample.shape), name="pixels")
                engine = self.runtime.get_or_compile(
                    f"vae_encode_{key}",
                    self.wrapper,
                    input_specs=[input_spec],
                )
                if engine is None:
                    # Compilation failed - fall back to PyTorch
                    return self.fallback_fn(sample, self.vae)
                self._cache[key] = engine

        sample_device = sample.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True)

        with torch.no_grad():
            mean, logvar = engine(sample_device)
        std = torch.exp(0.5 * logvar)
        if generator is None:
            noise = torch.randn_like(mean)
        else:
            noise = torch.randn_like(mean, generator=generator)
        latents = (mean + std * noise) * self.wrapper.scale

        return latents.to(dtype=self.runtime.compute_dtype, device=self.runtime.device)


class _CallableModule(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class TensorRTCallable:
    """Wrap an arbitrary tensor-only callable and compile it on first use."""

    def __init__(self, *, runtime: TensorRTRuntime, name: str, forward_fn):
        self.runtime = runtime
        self.name = name
        self.forward_fn = forward_fn
        self._module = _CallableModule(forward_fn)
        self._compiled = None

    def __call__(self, *args, **kwargs):
        if not self.runtime.is_ready:
            return self.forward_fn(*args, **kwargs)

        # Only tensors are supported for TensorRT callable wrapper
        if kwargs:
            return self.forward_fn(*args, **kwargs)
        if not all(torch.is_tensor(arg) for arg in args):
            return self.forward_fn(*args, **kwargs)

        if self._compiled is None:
            example_args = tuple(
                arg.detach().clone().to(device=self.runtime.device, dtype=self.runtime.compute_dtype) for arg in args
            )
            self._compiled = self.runtime.get_or_compile(
                self.name,
                self._module,
                example_inputs=(example_args, {}),
            )
            if self._compiled is None:
                # Compilation failed - fall back to PyTorch
                return self.forward_fn(*args, **kwargs)

        runtime_args = tuple(
            arg.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True) for arg in args
        )
        return self._compiled(*runtime_args, **kwargs)


class TransformerWrapper(nn.Module):
    """Wrapper to expose transformer forward as a single module for TensorRT."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        latent_indices: torch.Tensor,
        clean_latents: torch.Tensor,
        clean_latent_indices: torch.Tensor,
        clean_latents_2x: torch.Tensor,
        clean_latent_2x_indices: torch.Tensor,
        clean_latents_4x: torch.Tensor,
        clean_latent_4x_indices: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through transformer with all required inputs."""
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            pooled_projections=pooled_projections,
            guidance=guidance,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            image_embeddings=image_embeddings,
            attention_kwargs={},
            return_dict=False,
        )
        # Return the first element if it's a tuple
        if isinstance(output, tuple):
            return output[0]
        return output


class TensorRTTransformer:
    """TensorRT wrapper for transformer models with dynamic shape caching."""

    def __init__(self, transformer: nn.Module, runtime: TensorRTRuntime, fallback_fn=None):
        self.transformer = transformer
        self.runtime = runtime
        self.fallback_fn = fallback_fn
        self.wrapper = TransformerWrapper(transformer)
        self._cache: Dict[Tuple[int, ...], nn.Module] = {}
        self._lock = threading.Lock()
        self._compile_count = 0
        self._max_cached_shapes = 8  # Limit number of cached engine shapes

    def _shape_key(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> Tuple[int, ...]:
        """Generate cache key from input shapes."""
        return (
            tuple(hidden_states.shape),
            tuple(encoder_hidden_states.shape),
            tuple(image_embeddings.shape),
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor,
        latent_indices: torch.Tensor,
        clean_latents: torch.Tensor,
        clean_latent_indices: torch.Tensor,
        clean_latents_2x: torch.Tensor,
        clean_latent_2x_indices: torch.Tensor,
        clean_latents_4x: torch.Tensor,
        clean_latent_4x_indices: torch.Tensor,
        image_embeddings: torch.Tensor,
        attention_kwargs=None,
        return_dict: bool = False,
    ):
        """Execute transformer with optional TensorRT acceleration."""
        print(f"DEBUG TRT Transformer __call__: runtime.is_ready = {self.runtime.is_ready}")
        if not self.runtime.is_ready:
            if self.fallback_fn is not None:
                return self.fallback_fn(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    pooled_projections=pooled_projections,
                    guidance=guidance,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    image_embeddings=image_embeddings,
                    attention_kwargs=attention_kwargs or {},
                    return_dict=return_dict,
                )
            return self.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                guidance=guidance,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                image_embeddings=image_embeddings,
                attention_kwargs=attention_kwargs or {},
                return_dict=return_dict,
            )

        # Get shape-specific engine
        key = self._shape_key(hidden_states, encoder_hidden_states, image_embeddings)

        with self._lock:
            engine = self._cache.get(key)
            if engine is None:
                # Limit cache size
                if len(self._cache) >= self._max_cached_shapes:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    print(f"TensorRT transformer cache full, evicted shape: {oldest_key}")

                # Compile new engine for this shape
                print(f"Compiling TensorRT transformer engine for shape: {key}")
                # Create input specs for all inputs
                input_specs = [
                    self.runtime.make_input_from_shape(hidden_states.shape, name="hidden_states"),
                    self.runtime.make_input_from_shape(timestep.shape, name="timestep"),
                    self.runtime.make_input_from_shape(encoder_hidden_states.shape, name="encoder_hidden_states"),
                    self.runtime.make_input_from_shape(encoder_attention_mask.shape, name="encoder_attention_mask"),
                    self.runtime.make_input_from_shape(pooled_projections.shape, name="pooled_projections"),
                    self.runtime.make_input_from_shape(guidance.shape, name="guidance"),
                    self.runtime.make_input_from_shape(latent_indices.shape, name="latent_indices"),
                    self.runtime.make_input_from_shape(clean_latents.shape, name="clean_latents"),
                    self.runtime.make_input_from_shape(clean_latent_indices.shape, name="clean_latent_indices"),
                    self.runtime.make_input_from_shape(clean_latents_2x.shape, name="clean_latents_2x"),
                    self.runtime.make_input_from_shape(clean_latent_2x_indices.shape, name="clean_latent_2x_indices"),
                    self.runtime.make_input_from_shape(clean_latents_4x.shape, name="clean_latents_4x"),
                    self.runtime.make_input_from_shape(clean_latent_4x_indices.shape, name="clean_latent_4x_indices"),
                    self.runtime.make_input_from_shape(image_embeddings.shape, name="image_embeddings"),
                ]

                engine = self.runtime.get_or_compile(
                    f"transformer_{self._compile_count}",
                    self.wrapper,
                    input_specs=input_specs,
                )
                if engine is None:
                    print(f"TensorRT transformer compilation failed - falling back to PyTorch")
                    # Fall back to original implementation
                    if self.fallback_fn is not None:
                        return self.fallback_fn(
                            hidden_states=hidden_states,
                            timestep=timestep,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            pooled_projections=pooled_projections,
                            guidance=guidance,
                            latent_indices=latent_indices,
                            clean_latents=clean_latents,
                            clean_latent_indices=clean_latent_indices,
                            clean_latents_2x=clean_latents_2x,
                            clean_latent_2x_indices=clean_latent_2x_indices,
                            clean_latents_4x=clean_latents_4x,
                            clean_latent_4x_indices=clean_latent_4x_indices,
                            image_embeddings=image_embeddings,
                            attention_kwargs=attention_kwargs or {},
                            return_dict=return_dict,
                        )
                    return self.transformer(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        pooled_projections=pooled_projections,
                        guidance=guidance,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        image_embeddings=image_embeddings,
                        attention_kwargs=attention_kwargs or {},
                        return_dict=return_dict,
                    )

                self._compile_count += 1
                self._cache[key] = engine
                print(f"TensorRT transformer compilation successful")

        # Execute with TensorRT engine
        with torch.no_grad():
            output = engine(
                hidden_states.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                timestep.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                encoder_hidden_states.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                encoder_attention_mask.to(device=self.runtime.device, non_blocking=True),
                pooled_projections.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                guidance.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                latent_indices.to(device=self.runtime.device, non_blocking=True),
                clean_latents.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                clean_latent_indices.to(device=self.runtime.device, non_blocking=True),
                clean_latents_2x.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                clean_latent_2x_indices.to(device=self.runtime.device, non_blocking=True),
                clean_latents_4x.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
                clean_latent_4x_indices.to(device=self.runtime.device, non_blocking=True),
                image_embeddings.to(device=self.runtime.device, dtype=self.runtime.compute_dtype, non_blocking=True),
            )

        if return_dict:
            # Wrap in appropriate return type if needed
            return type('Output', (), {'sample': output})()
        return (output,) if isinstance(output, torch.Tensor) else output


class LlamaTextEncoderWrapper(nn.Module):
    """Wrapper for LLaMA text encoder that returns hidden states."""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass that extracts the third-to-last hidden state."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Return the third-to-last hidden state (as used in encode_prompt_conds)
        return outputs.hidden_states[-3]


class CLIPTextEncoderWrapper(nn.Module):
    """Wrapper for CLIP text encoder that returns pooler output."""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass that extracts pooler output."""
        outputs = self.text_encoder(input_ids, output_hidden_states=False)
        return outputs.pooler_output


class TensorRTTextEncoder:
    """TensorRT wrapper for LLaMA text encoder with shape-based caching."""

    def __init__(self, text_encoder: nn.Module, runtime: TensorRTRuntime, fallback_fn=None):
        self.text_encoder = text_encoder
        self.runtime = runtime
        self.fallback_fn = fallback_fn
        self.wrapper = LlamaTextEncoderWrapper(text_encoder)
        self._cache: Dict[Tuple[int, int], nn.Module] = {}
        self._lock = threading.Lock()

    def _shape_key(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Generate cache key from input shape."""
        return (int(input_ids.shape[0]), int(input_ids.shape[1]))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crop_start: int = 0,
        attention_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode text with TensorRT acceleration."""
        if not self.runtime.is_ready:
            if self.fallback_fn is not None:
                return self.fallback_fn(input_ids, attention_mask, crop_start, attention_length)
            # Fallback to direct model call
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-3]
            if attention_length is not None:
                return hidden_states[:, crop_start:attention_length]
            return hidden_states[:, crop_start:]

        key = self._shape_key(input_ids)
        with self._lock:
            engine = self._cache.get(key)
            if engine is None:
                input_id_spec = self.runtime.make_input_from_shape(tuple(input_ids.shape), name="input_ids")
                attention_mask_spec = self.runtime.make_input_from_shape(tuple(attention_mask.shape), name="attention_mask")
                engine = self.runtime.get_or_compile(
                    f"llama_text_encoder_{key}",
                    self.wrapper,
                    input_specs=[input_id_spec, attention_mask_spec],
                )
                if engine is None:
                    # Compilation failed - fall back to PyTorch
                    if self.fallback_fn is not None:
                        return self.fallback_fn(input_ids, attention_mask, crop_start, attention_length)
                    outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states[-3]
                    if attention_length is not None:
                        return hidden_states[:, crop_start:attention_length]
                    return hidden_states[:, crop_start:]
                self._cache[key] = engine

        # Run TensorRT engine
        input_ids_device = input_ids.to(device=self.runtime.device, non_blocking=True)
        attention_mask_device = attention_mask.to(device=self.runtime.device, non_blocking=True)

        with torch.no_grad():
            hidden_states = engine(input_ids_device, attention_mask_device)

        # Apply cropping if needed
        if attention_length is not None:
            hidden_states = hidden_states[:, crop_start:attention_length]
        elif crop_start > 0:
            hidden_states = hidden_states[:, crop_start:]

        return hidden_states


class TensorRTCLIPTextEncoder:
    """TensorRT wrapper for CLIP text encoder with shape-based caching."""

    def __init__(self, text_encoder: nn.Module, runtime: TensorRTRuntime, fallback_fn=None):
        self.text_encoder = text_encoder
        self.runtime = runtime
        self.fallback_fn = fallback_fn
        self.wrapper = CLIPTextEncoderWrapper(text_encoder)
        self._cache: Dict[Tuple[int, int], nn.Module] = {}
        self._lock = threading.Lock()

    def _shape_key(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Generate cache key from input shape."""
        return (int(input_ids.shape[0]), int(input_ids.shape[1]))

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text with TensorRT acceleration."""
        if not self.runtime.is_ready:
            if self.fallback_fn is not None:
                return self.fallback_fn(input_ids)
            # Fallback to direct model call
            outputs = self.text_encoder(input_ids, output_hidden_states=False)
            return outputs.pooler_output

        key = self._shape_key(input_ids)
        with self._lock:
            engine = self._cache.get(key)
            if engine is None:
                input_spec = self.runtime.make_input_from_shape(tuple(input_ids.shape), name="input_ids")
                engine = self.runtime.get_or_compile(
                    f"clip_text_encoder_{key}",
                    self.wrapper,
                    input_specs=[input_spec],
                )
                if engine is None:
                    # Compilation failed - fall back to PyTorch
                    if self.fallback_fn is not None:
                        return self.fallback_fn(input_ids)
                    outputs = self.text_encoder(input_ids, output_hidden_states=False)
                    return outputs.pooler_output
                self._cache[key] = engine

        # Run TensorRT engine
        input_ids_device = input_ids.to(device=self.runtime.device, non_blocking=True)

        with torch.no_grad():
            pooler_output = engine(input_ids_device)

        return pooler_output
