from diffusers_helper.hf_login import login

import os
import atexit

HF_HOME = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
os.environ['HF_HOME'] = HF_HOME
CACHE_BASE_DIR = os.path.join(os.path.dirname(__file__), 'Cache')
os.makedirs(CACHE_BASE_DIR, exist_ok=True)
HF_REPO_CACHE_ROOT = os.path.join(CACHE_BASE_DIR, 'hf_repos')
os.makedirs(HF_REPO_CACHE_ROOT, exist_ok=True)

ENABLE_QUANT = False
def _repo_cache_dir_name(repo_id: str) -> str:
    return repo_id.replace('/', '__').replace(':', '_')


def _prepare_local_repo(repo_id: str, env_var: str, *, preload: bool, parallel_workers: int) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    if not preload or snapshot_download is None:
        return repo_id
    cache_dir = os.path.join(HF_REPO_CACHE_ROOT, _repo_cache_dir_name(repo_id))
    os.makedirs(cache_dir, exist_ok=True)
    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=max(1, parallel_workers),
        )
        print(f'Preloaded {repo_id} into {local_path}')
        return local_path
    except Exception as exc:
        print(f'Warning: unable to preload {repo_id}: {exc}')
        return repo_id

import gradio as gr
import gradio.route_utils as gr_route_utils
import torch


def _patch_gradio_proxy_api_path():
    """Allow proxied hosts to keep the API path when resolving queue/call requests."""
    if getattr(gr_route_utils, "_framepack_api_path_patch", False):
        return

    original_get_api_call_path = gr_route_utils.get_api_call_path
    queue_api_url = f"{gr_route_utils.API_PREFIX}/queue/join"
    generic_api_url = f"{gr_route_utils.API_PREFIX}/call"

    def _get_api_call_path(request):
        request_path = (getattr(request.url, "path", "") or "").rstrip("/")
        if request_path.endswith(queue_api_url):
            return queue_api_url
        start_index = request_path.rfind(generic_api_url)
        if start_index >= 0:
            return request_path[start_index:]
        return original_get_api_call_path(request)

    gr_route_utils.get_api_call_path = _get_api_call_path
    gr_route_utils._framepack_api_path_patch = True


_patch_gradio_proxy_api_path()

RUNTIME_CACHE_ENABLED = os.environ.get("FRAMEPACK_RUNTIME_CACHE", "1") != "0"
RUNTIME_CACHE_ROOT = os.environ.get(
    "FRAMEPACK_RUNTIME_CACHE_DIR",
    os.path.join(CACHE_BASE_DIR, "runtime_caches"),
)
if RUNTIME_CACHE_ENABLED:
    os.makedirs(RUNTIME_CACHE_ROOT, exist_ok=True)


def _runtime_cache_path(name: str) -> str:
    return os.path.join(RUNTIME_CACHE_ROOT, f"{name}.pt")


def _load_runtime_cache_state(name: str):
    if not RUNTIME_CACHE_ENABLED:
        return None
    path = _runtime_cache_path(name)
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location='cpu')
    except Exception as exc:
        print(f'Warning: failed to load runtime cache "{name}": {exc}')
        return None


def _save_runtime_cache_state(name: str, state):
    if not RUNTIME_CACHE_ENABLED or state is None:
        if state is None:
            print(f'Info: runtime cache for "{name}" not saved because state is None.')
        return
    path = _runtime_cache_path(name)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    print(f'Attempting to save runtime cache for "{name}" to: {path}')
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        print(f'Successfully saved runtime cache for "{name}".')
    except Exception as exc:
        print(f'ERROR: failed to save runtime cache for "{name}": {exc}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _install_torch_compile_guard():
    """Wrap torch.compile to gracefully handle known duplicate template issues."""
    if getattr(torch, "_framepack_compile_guard", False):
        return
    orig_compile = getattr(torch, "compile", None)
    if not callable(orig_compile):
        return

    def safe_compile(fn, *args, **kwargs):
        try:
            return orig_compile(fn, *args, **kwargs)
        except AssertionError as exc:
            if "duplicate template name" in str(exc):
                fn_name = getattr(fn, "__name__", repr(fn))
                print(f"torch.compile failed for {fn_name} due to duplicate template name; using eager fallback.")
                return fn
            raise

    torch.compile = safe_compile
    torch._framepack_compile_guard = True


_install_torch_compile_guard()
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import hashlib
import inspect
import pathlib
import threading
from collections import OrderedDict
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import torch.nn.functional as F

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    BitsAndBytesConfig,
)
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.cpu_opt import (
    cpu_preprocessing_active,
    normalize_uint8_image,
    optimized_resize_and_center_crop,
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.models.hunyuan_video_packed import set_attention_accel_mode
from diffusers_helper.relationship_trainer import (
    HiddenStateRelationshipTrainer,
    DiTTimestepResidualTrainer,
    DiTTimestepModulationTrainer,
)
from diffusers_helper.frontend import build_frontend
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper import memory as memory_v1
from diffusers_helper import memory_v2
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.profiling import (
    profile_section,
    profile_function,
    get_global_stats,
    PyTorchProfiler,
    MemoryTracker,
    IterationProfiler,
    export_profiling_report,
)
from diffusers_helper.optimizations import (
    AdaptiveLatentWindowController,
    apply_int_nbit_quantization,
    enforce_low_precision,
    maybe_compile_module,
    maybe_wrap_with_fsdp,
    prune_transformer_layers,
)
from diffusers_helper.inference import (
    InferenceConfig,
    TorchScriptConfig,
    align_tensor_dim_to_multiple,
    build_default_inference_config,
    configure_inference_environment,
    inference_autocast,
    prepare_module_for_inference,
    tensor_core_multiple_for_dtype,
)
from diffusers_helper.tensorrt_runtime import (
    TensorRTRuntime,
    TensorRTLatentDecoder,
    TensorRTLatentEncoder,
    TensorRTCallable,
    TensorRTTransformer,
    TensorRTTextEncoder,
    TensorRTCLIPTextEncoder,
    _TORCH_TRT_IMPORT_ERROR,
)
from diffusers_helper.cache_events import CacheEventRecorder
try:
    from third_party.fp8_optimization_utils import optimize_state_dict_with_fp8, apply_fp8_monkey_patch
except ImportError:
    optimize_state_dict_with_fp8 = None
    apply_fp8_monkey_patch = None

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

# BetterTransformer optimization support
try:
    from optimum.bettertransformer import BetterTransformer
    BETTERTRANSFORMER_AVAILABLE = True
except ImportError:
    BetterTransformer = None
    BETTERTRANSFORMER_AVAILABLE = False


CACHE_DEFAULT_DTYPE = torch.float16


def _map_nested_tensors(data, fn):
    if torch.is_tensor(data):
        return fn(data)
    if isinstance(data, tuple):
        return type(data)(_map_nested_tensors(x, fn) for x in data)
    if isinstance(data, list):
        return [
            _map_nested_tensors(x, fn)
            for x in data
        ]
    # Handle objects with __dict__ BEFORE handling dict
    # This ensures Hugging Face model outputs (which may inherit from dict) preserve all attributes
    # Check for __dict__ but exclude basic Python types (not plain dict, as HF outputs may inherit from dict)
    if hasattr(data, '__dict__') and not isinstance(data, (type, str, int, float, bool)) and type(data).__module__ not in ('builtins', '__builtin__'):
        try:
            # Use the object's __dict__ directly to capture all instance attributes
            obj_dict = {}
            for key, value in data.__dict__.items():
                if not key.startswith('_'):
                    obj_dict[key] = _map_nested_tensors(value, fn)

            # If object also acts like a dict, handle that too
            dict_items = {}
            if hasattr(data, 'items') and callable(getattr(data, 'items')):
                try:
                    for k, v in data.items():
                        if k not in obj_dict:  # Don't duplicate if already in __dict__
                            dict_items[k] = _map_nested_tensors(v, fn)
                except Exception:
                    pass

            # Create a new instance without calling __init__
            # Try using the class's __new__ first, fall back if that fails
            try:
                reconstructed = type(data).__new__(type(data))
            except TypeError:
                # If __new__ requires arguments, try creating with empty args
                try:
                    reconstructed = type(data)()
                except Exception:
                    # Last resort: return the data as-is
                    import sys
                    print(f"DEBUG: Cannot create new instance of {type(data).__name__}, returning as-is", file=sys.stderr)
                    return data

            # Copy all attributes directly to __dict__
            for key, value in obj_dict.items():
                reconstructed.__dict__[key] = value
            # Also copy dict items if present
            for k, v in dict_items.items():
                try:
                    reconstructed[k] = v
                except Exception:
                    pass

            # Debug logging for model outputs
            if 'hidden_states' in data.__dict__:
                import sys
                print(f"DEBUG: Preserving object with hidden_states, type={type(data).__name__}", file=sys.stderr)
                print(f"DEBUG: Original has hidden_states: {data.__dict__.get('hidden_states') is not None}", file=sys.stderr)
                print(f"DEBUG: Reconstructed has hidden_states: {reconstructed.__dict__.get('hidden_states') is not None}", file=sys.stderr)

            return reconstructed
        except Exception as e:
            # If reconstruction fails, log and return the data as-is
            import sys
            print(f"DEBUG: Failed to reconstruct object {type(data).__name__}: {e}", file=sys.stderr)
            return data
    if isinstance(data, dict):
        mapped = {k: _map_nested_tensors(v, fn) for k, v in data.items()}
        try:
            return type(data)(**mapped)
        except Exception:
            return mapped
    return data


def _tensor_to_cache_copy(tensor):
    target_dtype = CACHE_DEFAULT_DTYPE if tensor.is_floating_point() else tensor.dtype
    return tensor.detach().to('cpu', dtype=target_dtype, copy=True)


def _tensor_to_device(tensor, device):
    return tensor.to(device=device, non_blocking=True)


def _load_component(name: str, loader_fn):
    print(f'Loading {name}...')
    value = loader_fn()
    print(f'{name} ready.')
    return value


def _infer_tensor_device(bound_args: Dict[str, Any]):
    for value in bound_args.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, (list, tuple)):
            for nested in value:
                if torch.is_tensor(nested):
                    return nested.device
        if isinstance(value, dict):
            for nested in value.values():
                if torch.is_tensor(nested):
                    return nested.device
    return torch.device('cpu')


def _narrow_tensor(tensor, batch_dim, index):
    if not torch.is_tensor(tensor):
        return tensor
    return tensor.narrow(batch_dim, index, 1)


def _index_select_tensor(tensor, batch_dim, indices):
    if not torch.is_tensor(tensor):
        return tensor
    if len(indices) == 0:
        return tensor.narrow(batch_dim, 0, 0)
    index_tensor = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(batch_dim, index_tensor)


def _split_structure(data, batch_dim, sample_count):
    if torch.is_tensor(data):
        return [
            data.narrow(batch_dim, idx, 1)
            for idx in range(sample_count)
        ]
    if isinstance(data, tuple):
        children = [_split_structure(item, batch_dim, sample_count) for item in data]
        return [type(data)(child[idx] for child in children) for idx in range(sample_count)]
    if isinstance(data, list):
        children = [_split_structure(item, batch_dim, sample_count) for item in data]
        return [
            [child[idx] for child in children]
            for idx in range(sample_count)
        ]
    if isinstance(data, dict):
        children = {k: _split_structure(v, batch_dim, sample_count) for k, v in data.items()}
        per_sample = []
        for idx in range(sample_count):
            sample_dict = {k: children[k][idx] for k in children}
            try:
                per_sample.append(type(data)(**sample_dict))
            except Exception:
                per_sample.append(sample_dict)
        return per_sample
    return [data for _ in range(sample_count)]


def _stack_structure(samples: Sequence[Any], batch_dim):
    if not samples:
        return None
    reference = samples[0]
    if torch.is_tensor(reference):
        return torch.cat(samples, dim=batch_dim)
    if isinstance(reference, tuple):
        stacked = []
        for idx in range(len(reference)):
            stacked.append(_stack_structure([sample[idx] for sample in samples], batch_dim))
        return type(reference)(stacked)
    if isinstance(reference, list):
        return [
            _stack_structure([sample[idx] for sample in samples], batch_dim)
            for idx in range(len(reference))
        ]
    if isinstance(reference, dict):
        keys = list(reference.keys())
        merged = {}
        for key in keys:
            merged[key] = _stack_structure([sample[key] for sample in samples], batch_dim)
        try:
            return type(reference)(**merged)
        except Exception:
            return merged
    return reference


def _hash_tensor(tensor):
    if not torch.is_tensor(tensor):
        return b'none'
    cpu_tensor = tensor.detach().to('cpu', dtype=torch.float32)
    data = cpu_tensor.contiguous().numpy().view(np.uint8)
    hasher = hashlib.sha1()
    hasher.update(str(tuple(cpu_tensor.shape)).encode())
    hasher.update(str(cpu_tensor.dtype).encode())
    hasher.update(bytes(data))
    return hasher.digest()


def _stable_hash_name(module_name, revision, normalization, mode, tensor_hashes):
    hasher = hashlib.sha1()
    hasher.update(str(module_name).encode())
    hasher.update(str(revision).encode())
    hasher.update(str(normalization).encode())
    hasher.update(str(mode).encode())
    for name, value in tensor_hashes:
        hasher.update(name.encode())
        hasher.update(value)
    return hasher.hexdigest()


def align_to_multiple(value, multiple: int = 8, minimum: int = None):
    if multiple <= 0:
        return int(value)
    aligned = int(math.ceil(max(1, int(value)) / multiple) * multiple)
    if minimum is not None:
        aligned = max(aligned, int(minimum))
    return aligned


def align_resolution(height: int, width: int, multiple: int = 64):
    return align_to_multiple(height, multiple, multiple), align_to_multiple(width, multiple, multiple)


def apply_bettertransformer_optimization(module, module_name="module", allow_bnb=True):
    """
    Apply BetterTransformer optimization to a model.
    Falls back to PyTorch native SDPA if optimum is not available.

    Now supports BitsAndBytes quantized models by applying SDPA configuration
    directly to the model config, which works seamlessly with quantized layers.

    Args:
        module: The model to optimize (text encoder, image encoder, etc.)
        module_name: Name of the module for logging
        allow_bnb: Whether to allow optimization on BitsAndBytes models (default: True)

    Returns:
        Optimized module or original module if optimization fails
    """
    # Check if model is BitsAndBytes quantized
    is_bnb_quantized = False
    if hasattr(module, 'is_loaded_in_4bit'):
        is_bnb_quantized = module.is_loaded_in_4bit
    elif hasattr(module, 'is_loaded_in_8bit'):
        is_bnb_quantized = module.is_loaded_in_8bit
    elif hasattr(module, 'is_quantized'):
        is_bnb_quantized = module.is_quantized

    # For BitsAndBytes models, optimum's BetterTransformer.transform() may fail
    # But we can still apply SDPA configuration which works with quantized weights
    if is_bnb_quantized:
        print(f'{module_name} is BitsAndBytes quantized - using SDPA-only optimization (compatible mode)')
        # Skip optimum's transform, go straight to SDPA config
    elif BETTERTRANSFORMER_AVAILABLE and not is_bnb_quantized:
        try:
            optimized = BetterTransformer.transform(module, keep_original_model=False)
            print(f'BetterTransformer applied to {module_name} (using optimum library)')
            return optimized
        except Exception as exc:
            print(f'BetterTransformer optimization failed for {module_name}: {exc}')
            print(f'Falling back to PyTorch native SDPA for {module_name}')

    # SDPA configuration - works with both regular and BitsAndBytes models
    # This is the key: SDPA operates on the attention computation, not the weights
    # So it's compatible with quantized weights
    try:
        # Enable PyTorch's native fast attention path
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Set the model to use SDPA backend
            if hasattr(module, 'config'):
                # For HuggingFace models with config
                config = module.config
                if hasattr(config, '_attn_implementation'):
                    original_impl = config._attn_implementation
                    config._attn_implementation = 'sdpa'
                    if is_bnb_quantized:
                        print(f'Enabled PyTorch SDPA backend for {module_name} (BnB-compatible, changed from {original_impl})')
                    else:
                        print(f'Enabled PyTorch SDPA backend for {module_name} (changed from {original_impl})')
                elif hasattr(config, 'attn_implementation'):
                    original_impl = config.attn_implementation
                    config.attn_implementation = 'sdpa'
                    if is_bnb_quantized:
                        print(f'Enabled PyTorch SDPA backend for {module_name} (BnB-compatible, changed from {original_impl})')
                    else:
                        print(f'Enabled PyTorch SDPA backend for {module_name} (changed from {original_impl})')
                else:
                    print(f'PyTorch SDPA available but config attribute not found for {module_name}')
            else:
                print(f'PyTorch SDPA available but no config found for {module_name}')

            # Enable memory efficient attention backends
            # These work with quantized models because they optimize the attention computation
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

            if is_bnb_quantized:
                print(f'  → SDPA will use Flash/Memory-efficient attention with quantized weights')

            return module
        else:
            print(f'PyTorch SDPA not available (requires PyTorch 2.0+), keeping {module_name} unchanged')
            return module
    except Exception as exc:
        print(f'Native SDPA fallback failed for {module_name}: {exc}')
        return module


def _build_transformer_example(module: torch.nn.Module, dtype: torch.dtype, device: torch.device):
    config = getattr(module, "config", {})
    patch = int(config.get("patch_size", 2))
    patch_t = int(config.get("patch_size_t", 1))
    in_channels = int(config.get("in_channels", 16))
    pooled_dim = int(config.get("pooled_projection_dim", 768))
    image_proj_dim = int(config.get("image_proj_dim", pooled_dim))
    text_dim = int(config.get("text_embed_dim", 4096))

    batch = 1
    frames = max(patch_t * 8, 16)
    height = max(patch * 32, 64)
    width = max(patch * 32, 64)

    hidden_states = torch.zeros((batch, in_channels, frames, height, width), dtype=dtype, device=device)
    timestep = torch.ones((batch,), dtype=dtype, device=device)

    tokens = (frames // patch_t) * (height // patch) * (width // patch)
    encoder_hidden_states = torch.zeros((batch, tokens, text_dim), dtype=dtype, device=device)
    encoder_attention_mask = torch.ones((batch, tokens), dtype=torch.bool, device=device)
    pooled_projections = torch.zeros((batch, pooled_dim), dtype=dtype, device=device)
    guidance = torch.ones((batch,), dtype=dtype, device=device)

    latent_indices = torch.arange(frames, device=device, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    clean_latents = torch.zeros_like(hidden_states)
    clean_latent_indices = latent_indices.clone()
    clean_latents_2x = torch.zeros_like(hidden_states)
    clean_latent_2x_indices = latent_indices.clone()
    clean_latents_4x = torch.zeros_like(hidden_states)
    clean_latent_4x_indices = latent_indices.clone()
    image_embeddings = torch.zeros((batch, tokens, image_proj_dim), dtype=dtype, device=device)

    kwargs = dict(
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
    return (hidden_states, timestep), kwargs


class DiskLRUCache:
    def __init__(self, cache_dir: str, max_items: int = 128):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max(0, int(max_items))
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._evict_hooks: List = []

    def _path_for_key(self, key: str) -> pathlib.Path:
        return self.cache_dir / f"{key}.pt"

    def register_evict_hook(self, fn):
        if fn not in self._evict_hooks:
            self._evict_hooks.append(fn)

    def _notify_evict(self, key: str):
        for hook in list(self._evict_hooks):
            try:
                hook(key)
            except Exception:
                pass

    def get(self, key: str):
        if not key or self.max_items <= 0:
            return None
        path = self._path_for_key(key)
        with self._lock:
            if not path.exists():
                self._entries.pop(key, None)
                self._notify_evict(key)
                return None
            try:
                value = torch.load(path, map_location='cpu')
            except Exception:
                self._entries.pop(key, None)
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._notify_evict(key)
                return None
            self._entries.pop(key, None)
            self._entries[key] = path
            return value

    def set(self, key: str, value):
        if not key or self.max_items <= 0:
            return
        path = self._path_for_key(key)
        tmp_path = pathlib.Path(f"{path}.{os.getpid()}.tmp")
        with self._lock:
            try:
                torch.save(value, tmp_path)
                os.replace(tmp_path, path)
                self._entries.pop(key, None)
                self._entries[key] = path
                while len(self._entries) > self.max_items:
                    old_key, old_path = self._entries.popitem(last=False)
                    try:
                        old_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._notify_evict(old_key)
            except Exception:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


class SemanticIndex:
    def __init__(self, index_path: pathlib.Path, threshold: float = 0.98, metric: str = 'cosine'):
        self.index_path = pathlib.Path(index_path)
        self.threshold = float(threshold)
        self.metric = metric
        self.keys: List[str] = []
        self.embeddings = torch.empty((0, 0), dtype=torch.float32)
        self._faiss_index = None
        self._faiss_dim = None
        self._load()

    def _load(self):
        if self.index_path.exists():
            try:
                data = torch.load(self.index_path, map_location='cpu')
                self.keys = list(data.get('keys', []))
                emb = data.get('embeddings')
                if isinstance(emb, torch.Tensor):
                    self.embeddings = emb.to(dtype=torch.float32, device='cpu')
                    self._faiss_dim = self.embeddings.shape[1] if self.embeddings.ndim == 2 else None
            except Exception:
                self.keys = []
                self.embeddings = torch.empty((0, 0), dtype=torch.float32)
        else:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(self):
        try:
            torch.save({'keys': self.keys, 'embeddings': self.embeddings}, self.index_path)
        except Exception:
            pass

    def _normalize(self, embedding: torch.Tensor):
        if not torch.is_tensor(embedding):
            return None
        flat = embedding.float().view(-1)
        if flat.numel() == 0:
            return None
        norm = torch.linalg.norm(flat)
        if norm == 0:
            return None
        return (flat / norm).contiguous()

    def _rebuild_faiss(self):
        if not FAISS_AVAILABLE or self.embeddings.numel() == 0:
            self._faiss_index = None
            return
        try:
            dim = self.embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(self.embeddings.numpy()))
            self._faiss_index = index
            self._faiss_dim = dim
        except Exception:
            self._faiss_index = None

    def add(self, key: str, embedding: torch.Tensor):
        normalized = self._normalize(embedding)
        if normalized is None:
            return
        normalized = normalized.unsqueeze(0)
        if self.embeddings.numel() == 0:
            self.embeddings = normalized.to(dtype=torch.float32, device='cpu')
        else:
            if normalized.shape[1] != self.embeddings.shape[1]:
                self.embeddings = normalized.to(dtype=torch.float32, device='cpu')
                self.keys = []
            else:
                self.embeddings = torch.cat([self.embeddings, normalized.to(dtype=torch.float32, device='cpu')], dim=0)
        self.keys.append(key)
        self._save()
        self._rebuild_faiss()

    def remove(self, key: str):
        if key not in self.keys:
            return
        idx = self.keys.index(key)
        self.keys.pop(idx)
        if self.embeddings.shape[0] <= 1:
            self.embeddings = torch.empty((0, 0), dtype=torch.float32)
        else:
            mask = torch.ones(self.embeddings.shape[0], dtype=torch.bool)
            mask[idx] = False
            self.embeddings = self.embeddings[mask]
        self._save()
        self._rebuild_faiss()

    def find_match(self, embedding: torch.Tensor) -> Tuple[str, float]:
        if not self.keys:
            return '', 0.0
        normalized = self._normalize(embedding)
        if normalized is None:
            return '', 0.0
        normalized = normalized.unsqueeze(0)
        scores = None
        best_idx = -1
        if self._faiss_index is not None:
            if normalized.shape[1] != self._faiss_dim:
                return '', 0.0
            try:
                distances, indices = self._faiss_index.search(normalized.numpy(), 1)
                score = float(distances[0][0]) if distances.size else 0.0
                best_idx = int(indices[0][0]) if indices.size else -1
                scores = score
            except Exception:
                best_idx = -1
        if best_idx < 0:
            if self.embeddings.shape[0] == 0 or normalized.shape[1] != self.embeddings.shape[1]:
                return '', 0.0
            sims = torch.matmul(self.embeddings, normalized.squeeze(0))
            score, idx = torch.max(sims, dim=0)
            best_idx = int(idx)
            scores = float(score)
        if best_idx < 0 or best_idx >= len(self.keys):
            return '', 0.0
        if scores < self.threshold:
            return '', scores
        return self.keys[best_idx], scores
class ModuleCacheWrapper(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        cache: DiskLRUCache,
        *,
        module_name: str,
        module_revision: str = None,
        batch_dim: int = 0,
        batched_arg_names: Sequence[str] = None,
        hash_input_names: Sequence[str] = None,
        normalization_tag: str = 'default',
        cache_mode: str = 'hash',
        semantic_embed_fn=None,
        semantic_threshold: float = 0.98,
    ):
        super().__init__()
        self.inner = module
        self.cache = cache
        self.module_name = module_name
        self.module_revision = module_revision or getattr(getattr(module, 'config', None), 'revision', None) or getattr(getattr(module, 'config', None), '_name_or_path', None) or 'unknown'
        self.batch_dim = batch_dim
        self.batched_arg_names = list(batched_arg_names or [])
        self.hash_input_names = list(hash_input_names or self.batched_arg_names)
        self.normalization_tag = normalization_tag
        self.cache_enabled = cache is not None and cache.max_items > 0
        try:
            self.forward_signature = inspect.signature(module.forward)
        except (ValueError, TypeError):
            self.forward_signature = None
        self.forward_params = list(self.forward_signature.parameters.keys()) if self.forward_signature else []
        self.semantic_embed_fn = semantic_embed_fn
        self.semantic_threshold = semantic_threshold
        self.semantic_index = None
        self.cache_mode = 'off'
        self._evict_hook_registered = False
        self.set_cache_mode(cache_mode)

    def __getattr__(self, name):
        if name in {
            'inner',
            'cache',
            'module_name',
            'module_revision',
            'batch_dim',
            'batched_arg_names',
            'hash_input_names',
            'normalization_tag',
            'cache_enabled',
            'forward_signature',
            'forward_params',
            'semantic_embed_fn',
            'semantic_threshold',
            'semantic_index',
            'cache_mode',
        }:
            return super().__getattr__(name)
        # Delegate all other attributes to inner module (including device, dtype, etc.)
        return getattr(self.inner, name)

    def set_cache_mode(self, mode: str):
        if not self.cache_enabled:
            self.cache_mode = 'off'
            return
        normalized = (mode or 'hash').lower()
        if normalized not in {'hash', 'semantic', 'off'}:
            normalized = 'hash'
        self.cache_mode = normalized
        if self.cache_mode == 'semantic':
            self._ensure_semantic_index()

    def _ensure_semantic_index(self):
        if self.semantic_index is not None or not self.cache_enabled:
            return
        if self.semantic_embed_fn is None:
            return
        index_path = pathlib.Path(self.cache.cache_dir) / f"{self.module_name}_semantic.pt"
        self.semantic_index = SemanticIndex(index_path, threshold=self.semantic_threshold)
        if not self._evict_hook_registered:
            self.cache.register_evict_hook(self._on_cache_evict)
            self._evict_hook_registered = True

    def _on_cache_evict(self, key: str):
        if self.semantic_index is not None:
            self.semantic_index.remove(key)

    def _bind_args(self, args, kwargs):
        if self.forward_signature is not None:
            try:
                bound = self.forward_signature.bind_partial(*args, **kwargs)
                bound_dict = dict(bound.arguments)
                # Also include any kwargs that weren't captured by bind_partial
                # This handles cases like output_hidden_states that may not be in the explicit signature
                for k, v in kwargs.items():
                    if k not in bound_dict:
                        bound_dict[k] = v
                return bound_dict
            except Exception:
                pass
        merged = {}
        for name, arg in zip(self.forward_params, args):
            merged[name] = arg
        merged.update(kwargs)
        return merged

    def _infer_batch_info(self, bound_args):
        for name in self.batched_arg_names:
            value = bound_args.get(name)
            if torch.is_tensor(value):
                if value.shape[self.batch_dim] == 0:
                    continue
                return value.shape[self.batch_dim], value.device
        return None, None

    def _build_sample_key(self, sample_inputs):
        tensor_hashes = []
        for name in self.hash_input_names:
            tensor_hashes.append((name, _hash_tensor(sample_inputs.get(name))))
        mode = 'train' if self.inner.training else 'eval'
        return _stable_hash_name(self.module_name, self.module_revision, self.normalization_tag, mode, tensor_hashes)

    def _prepare_for_cache(self, data):
        return _map_nested_tensors(data, _tensor_to_cache_copy)

    def _move_to_runtime(self, data, device):
        return _map_nested_tensors(data, lambda t: _tensor_to_device(t, device))

    def _compute_semantic_embedding(self, sample_inputs):
        if self.semantic_embed_fn is None:
            return None
        try:
            embedding = self.semantic_embed_fn(sample_inputs)
        except Exception:
            return None
        if embedding is None:
            return None
        if torch.is_tensor(embedding):
            return embedding.detach().to('cpu')
        if isinstance(embedding, np.ndarray):
            return torch.from_numpy(embedding).float()
        return None

    def _collect_sample_inputs(self, bound_args, index):
        result = {}
        for name in self.hash_input_names:
            value = bound_args.get(name)
            if torch.is_tensor(value):
                result[name] = _narrow_tensor(value, self.batch_dim, index)
            else:
                result[name] = value
        return result

    def _gather_batch(self, bound_args, indices):
        gathered = dict(bound_args)
        for name in self.batched_arg_names:
            value = bound_args.get(name)
            if torch.is_tensor(value):
                gathered[name] = _index_select_tensor(value, self.batch_dim, indices)
        return gathered

    def _forward_without_split(self, bound_args):
        key_inputs = {name: bound_args.get(name) for name in self.hash_input_names}
        key = self._build_sample_key(key_inputs)
        runtime_device = _infer_tensor_device(bound_args)
        if self.cache_mode != 'off':
            cached = self.cache.get(key)
            if cached is None and self.cache_mode == 'semantic' and self.semantic_index is not None:
                sample_inputs = self._collect_sample_inputs(bound_args, 0)
                embedding = self._compute_semantic_embedding(sample_inputs)
                if embedding is not None:
                    match_key, _ = self.semantic_index.find_match(embedding)
                    if match_key:
                        cached = self.cache.get(match_key)
            if cached is not None:
                return self._move_to_runtime(cached, runtime_device)

        # Debug: Check what's being passed to the model
        if 'output_hidden_states' in bound_args:
            import sys
            print(f"DEBUG: Calling {self.module_name} with output_hidden_states={bound_args.get('output_hidden_states')}", file=sys.stderr)
            print(f"DEBUG: All bound_args keys: {list(bound_args.keys())}", file=sys.stderr)

        output = self.inner(**bound_args)

        # Debug: Check what the model returned
        if hasattr(output, 'hidden_states'):
            import sys
            print(f"DEBUG: Model {self.module_name} returned output with hidden_states={output.hidden_states is not None}", file=sys.stderr)
        if self.cache_mode != 'off':
            prepared = self._prepare_for_cache(output)
            self.cache.set(key, prepared)
            if self.cache_mode == 'semantic' and self.semantic_index is not None:
                sample_inputs = self._collect_sample_inputs(bound_args, 0)
                embedding = self._compute_semantic_embedding(sample_inputs)
                if embedding is not None:
                    self.semantic_index.add(key, embedding)
        return output

    def forward(self, *args, **kwargs):
        if not self.cache_enabled or self.cache_mode == 'off':
            return self.inner(*args, **kwargs)
        bound_args = self._bind_args(args, kwargs)
        batch_size, runtime_device = self._infer_batch_info(bound_args)
        if batch_size is None or batch_size <= 0:
            return self._forward_without_split(bound_args)

        assembled_outputs: List[Any] = [None] * batch_size
        sample_keys: List[str] = [''] * batch_size
        miss_indices: List[int] = []
        sample_embeddings: Dict[int, torch.Tensor] = {}

        for idx in range(batch_size):
            sample_inputs = self._collect_sample_inputs(bound_args, idx)
            key = self._build_sample_key(sample_inputs)
            sample_keys[idx] = key
            cached = self.cache.get(key)
            if cached is None and self.cache_mode == 'semantic' and self.semantic_index is not None:
                embedding = self._compute_semantic_embedding(sample_inputs)
                if embedding is not None:
                    sample_embeddings[idx] = embedding
                    match_key, _ = self.semantic_index.find_match(embedding)
                    if match_key:
                        cached = self.cache.get(match_key)
            if cached is None:
                miss_indices.append(idx)
            else:
                assembled_outputs[idx] = self._move_to_runtime(cached, runtime_device)

        if miss_indices:
            miss_args = self._gather_batch(bound_args, miss_indices)
            miss_output = self.inner(**miss_args)
            per_sample_outputs = _split_structure(miss_output, self.batch_dim, len(miss_indices))
            for idx, sample_output in zip(miss_indices, per_sample_outputs):
                assembled_outputs[idx] = sample_output
                key = sample_keys[idx]
                prepared = self._prepare_for_cache(sample_output)
                self.cache.set(key, prepared)
                if self.cache_mode == 'semantic' and self.semantic_index is not None:
                    embedding = sample_embeddings.get(idx)
                    if embedding is None:
                        sample_inputs = self._collect_sample_inputs(bound_args, idx)
                        embedding = self._compute_semantic_embedding(sample_inputs)
                    if embedding is not None:
                        self.semantic_index.add(key, embedding)

        return _stack_structure(assembled_outputs, self.batch_dim)


torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(False)

DEFAULT_CACHE_MODE = os.environ.get("FRAMEPACK_MODULE_CACHE_MODE", "semantic").lower()
SEMANTIC_CACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_SEMANTIC_CACHE_THRESHOLD", "0.985"))
SEMANTIC_CACHE_THRESHOLD = max(0.0, min(1.0, SEMANTIC_CACHE_THRESHOLD))
DEFAULT_PROFILING_ENABLED = os.environ.get("FRAMEPACK_ENABLE_PROFILING", "0") == "1"


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--cache-mode", type=str, choices=['hash', 'semantic', 'off'], default=DEFAULT_CACHE_MODE)
parser.add_argument("--jit-mode", type=str, choices=['off', 'trace', 'script'], default=os.environ.get("FRAMEPACK_JIT_MODE", "off"))
parser.add_argument("--disable-fbcache", action='store_true')
parser.add_argument("--disable-sim-cache", action='store_true')
parser.add_argument("--disable-kv-cache", action='store_true')
parser.add_argument("--xformers-mode", type=str, choices=["off", "standard", "aggressive"], default=os.environ.get("FRAMEPACK_XFORMERS_MODE", "standard"))
parser.add_argument("--fast-start", action='store_true', help="Skip optional optimizations to reduce startup latency.")
parser.add_argument("--use-memory-v2", action="store_true", help="Enable optimized memory_v2 backend (async streams, pinned memory, cached stats).")
parser.add_argument(
    "--enforce-low-precision",
    action="store_true",
    default=os.environ.get("FRAMEPACK_ENFORCE_LOW_PRECISION", "0") == "1",
    help="Force modules to run in low precision (fp16/bf16) where possible. Disabled by default.",
)
parser.add_argument(
    "--enable-tensorrt",
    action="store_true",
    default=os.environ.get("FRAMEPACK_ENABLE_TENSORRT", "0") == "1",
    help="Enable experimental TensorRT acceleration for VAE encode/decode and CLIP vision (requires torch-tensorrt).",
)
parser.add_argument(
    "--tensorrt-transformer",
    action="store_true",
    default=os.environ.get("FRAMEPACK_TRT_TRANSFORMER", "0") == "1",
    help="DONT WORK! Enable TensorRT acceleration for transformer model (experimental, requires --enable-tensorrt).",
)
parser.add_argument(
    "--tensorrt-text-encoders",
    action="store_true",
    default=os.environ.get("FRAMEPACK_TRT_TEXT_ENCODERS", "0") == "1",
    help="DONT WORK! Enable TensorRT acceleration for LLAMA and CLIP text encoders (experimental, requires --enable-tensorrt).",
)
parser.add_argument(
    "--use-onnx-engines",
    action="store_true",
    default=os.environ.get("FRAMEPACK_USE_ONNX_ENGINES", "0") == "1",
    help="DONT WORK! Use ONNX models for inference (TensorRT engines if available, otherwise ONNX Runtime).",
)
parser.add_argument(
    "--disable-bettertransformer",
    action="store_true",
    help="Disable BetterTransformer optimization for encoders (enabled by default).",
)
parser.add_argument(
    "--enable-profiling",
    action="store_true",
    default=DEFAULT_PROFILING_ENABLED,
    help=(
        "Enable profile-guided optimization (PGO) to identify bottlenecks. "
        "Exports timing stats and PyTorch profiler traces. "
        "Set FRAMEPACK_ENABLE_PROFILING=1 to turn this on without the CLI flag."
    ),
)
parser.add_argument(
    "--profiling-iterations",
    type=int,
    default=1,
    help="Number of iterations to profile (default: 1). Use 3-5 for statistical accuracy.",
)
parser.add_argument(
    "--profiling-output-dir",
    type=str,
    default="./profiling_results",
    help="Directory to save profiling results (default: ./profiling_results).",
)
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)
if args.use_memory_v2:
    print("memory_v2 backend enabled (async streams, pinned memory, cached stats).")


# ============================================================================
# FLAG COMPATIBILITY VALIDATION
# ============================================================================
def validate_flag_compatibility():
    """Validate and warn about incompatible flag combinations."""
    warnings = []
    errors = []

    # Get all flags early (some are set later, so we check env vars too)
    use_memory_v2 = args.use_memory_v2
    enable_compile = os.environ.get("FRAMEPACK_ENABLE_COMPILE", "0") == "1"
    enable_quant = os.environ.get("FRAMEPACK_ENABLE_QUANT")
    use_bnb = os.environ.get("FRAMEPACK_USE_BNB", "0") == "1"
    enable_tensorrt = args.enable_tensorrt
    tensorrt_transformer = args.tensorrt_transformer
    tensorrt_text = args.tensorrt_text_encoders
    jit_mode = args.jit_mode
    enable_fp8 = os.environ.get("FRAMEPACK_ENABLE_FP8", "1") == "1"

    # CRITICAL: torch.compile + memory-v2 incompatibility
    if enable_compile and use_memory_v2:
        errors.append(
            "⚠️  CRITICAL INCOMPATIBILITY: torch.compile + memory-v2\n"
            "    torch.compile creates CUDA graphs that conflict with dynamic device copies in memory-v2\n"
            "    Error: 'cudagraph partition due to DeviceCopy ops'\n"
            "    SOLUTION: Choose ONE:\n"
            "      1. Remove --use-memory-v2 flag (recommended for speed)\n"
            "      2. Set FRAMEPACK_ENABLE_COMPILE=0 (recommended for low VRAM)\n"
            "      3. Set TORCH_COMPILE_DISABLE_CUDAGRAPHS=1 (slower compile, works but defeats purpose)"
        )

    # TensorRT + BitsAndBytes incompatibility
    if (enable_tensorrt or tensorrt_transformer or tensorrt_text) and use_bnb:
        errors.append(
            "⚠️  CRITICAL INCOMPATIBILITY: TensorRT + BitsAndBytes\n"
            "    TensorRT requires FP16/BF16 precision, BitsAndBytes uses INT4/INT8\n"
            "    SOLUTION: Choose ONE:\n"
            "      1. Disable BitsAndBytes: FRAMEPACK_USE_BNB=0\n"
            "      2. Disable TensorRT: --enable-tensorrt=False or unset flags\n"
            "      3. Use FRAMEPACK_ENABLE_QUANT=1 instead of BitsAndBytes"
        )

    # torch.compile + TorchScript incompatibility
    if enable_compile and jit_mode != "off":
        warnings.append(
            "⚠️  WARNING: torch.compile + TorchScript (--jit-mode)\n"
            "    Both provide compilation, but use different approaches\n"
            "    torch.compile will be skipped in favor of TorchScript\n"
            "    RECOMMENDATION: Use torch.compile (better) - set --jit-mode off"
        )

    # Multiple quantization methods active
    quant_methods = []
    if enable_quant == "1":
        quant_methods.append("INT-N quantization (FRAMEPACK_ENABLE_QUANT)")
    if use_bnb:
        quant_methods.append("BitsAndBytes (FRAMEPACK_USE_BNB)")
    if enable_fp8:
        quant_methods.append("FP8 quantization (FRAMEPACK_ENABLE_FP8)")

    if len(quant_methods) > 1:
        warnings.append(
            f"⚠️  WARNING: Multiple quantization methods enabled:\n"
            f"    {', '.join(quant_methods)}\n"
            f"    These may conflict or apply redundantly\n"
            f"    RECOMMENDATION: Choose ONE quantization method"
        )

    # TensorRT without required flags
    if tensorrt_transformer or tensorrt_text:
        if not enable_tensorrt:
            warnings.append(
                "⚠️  WARNING: TensorRT components enabled but --enable-tensorrt not set\n"
                "    Text encoders/transformer TensorRT will be ignored\n"
                "    SOLUTION: Add --enable-tensorrt flag"
            )

    # Fast start with optimizations that will be skipped
    fast_start = args.fast_start or os.environ.get("FRAMEPACK_FAST_START", "0") == "1"
    if fast_start:
        skipped = []
        if enable_compile:
            skipped.append("torch.compile")
        if os.environ.get("FRAMEPACK_ENABLE_OPT_CACHE", "0") == "1":
            skipped.append("optimized model caching")
        if os.environ.get("FRAMEPACK_ENABLE_BETTERTRANSFORMER", "0") == "1" and not args.disable_bettertransformer:
            skipped.append("BetterTransformer")

        if skipped:
            warnings.append(
                f"ℹ️  INFO: Fast-start mode enabled - skipping:\n"
                f"    {', '.join(skipped)}\n"
                f"    Remove --fast-start for full optimizations"
            )

    # FP8 without proper hardware
    if enable_fp8:
        fp8_available = os.environ.get("FRAMEPACK_ENABLE_FP8") and FP8_UTILS_AVAILABLE
        if not fp8_available:
            warnings.append(
                "⚠️  WARNING: FP8 enabled but utilities unavailable\n"
                "    Requires third_party/fp8_optimization_utils.py\n"
                "    FP8 will be disabled automatically"
            )

    # Print all warnings and errors
    if errors or warnings:
        print("\n" + "="*80)
        print("FLAG COMPATIBILITY CHECK")
        print("="*80)

    for error in errors:
        print(f"\n{error}\n")

    for warning in warnings:
        print(f"\n{warning}\n")

    if errors:
        print("="*80)
        print("CRITICAL ERRORS DETECTED - The application may fail or behave unexpectedly")
        print("Please resolve the incompatibilities above before proceeding")
        print("="*80 + "\n")

        # Don't exit, just warn - let user decide
        import time
        print("Waiting 5 seconds before continuing...")
        time.sleep(5)
    elif warnings:
        print("="*80)
        print("Configuration loaded with warnings - review recommendations above")
        print("="*80 + "\n")

    return len(errors) == 0


# Run validation
validate_flag_compatibility()

FAST_START = args.fast_start or os.environ.get("FRAMEPACK_FAST_START", "0") == "1"
PARALLEL_LOADERS = int(os.environ.get("FRAMEPACK_PARALLEL_LOADERS", "0"))
if PARALLEL_LOADERS <= 1 and FAST_START:
    PARALLEL_LOADERS = 4
PRELOAD_REPOS = os.environ.get("FRAMEPACK_PRELOAD_REPOS", "0") == "1"
FORCE_PARALLEL_LOADERS = os.environ.get("FRAMEPACK_FORCE_PARALLEL_LOADERS", "0") == "1"

CACHE_MODE = args.cache_mode.lower()
VAE_CHUNK_OVERRIDE = max(0, int(os.environ.get("FRAMEPACK_VAE_CHUNK_SIZE", "0")))
VAE_CHUNK_RESERVE_GB = float(os.environ.get("FRAMEPACK_VAE_CHUNK_RESERVE_GB", "2.0"))
VAE_CHUNK_SAFETY = float(os.environ.get("FRAMEPACK_VAE_CHUNK_SAFETY", "1.2"))
VAE_UPSCALE_FACTOR = max(1, int(os.environ.get("FRAMEPACK_VAE_UPSCALE_FACTOR", "8")))
TRT_WORKSPACE_MB = int(os.environ.get("FRAMEPACK_TRT_WORKSPACE_MB", "4096"))
TRT_MAX_AUX_STREAMS = int(os.environ.get("FRAMEPACK_TRT_MAX_AUX_STREAMS", "2"))
TRT_TRANSFORMER_ENABLED = args.tensorrt_transformer
TRT_TEXT_ENCODERS_ENABLED = args.tensorrt_text_encoders
TRT_MAX_CACHED_SHAPES = int(os.environ.get("FRAMEPACK_TRT_MAX_CACHED_SHAPES", "8"))
ENABLE_TENSORRT_RUNTIME = args.enable_tensorrt
USE_ONNX_ENGINES = args.use_onnx_engines
ENFORCE_LOW_PRECISION = args.enforce_low_precision

print(f"DEBUG: ENABLE_TENSORRT_RUNTIME = {ENABLE_TENSORRT_RUNTIME}")
print(f"DEBUG: TRT_TRANSFORMER_ENABLED = {TRT_TRANSFORMER_ENABLED}")
print(f"DEBUG: TRT_TEXT_ENCODERS_ENABLED = {TRT_TEXT_ENCODERS_ENABLED}")

_onnx_trt_env = os.environ.get("FRAMEPACK_ONNX_TRT_ENABLE")
if _onnx_trt_env is None:
    ONNX_TRT_PROVIDER_ENABLED = ENABLE_TENSORRT_RUNTIME or USE_ONNX_ENGINES
else:
    ONNX_TRT_PROVIDER_ENABLED = _onnx_trt_env == "1"
ONNX_TRT_FP16 = os.environ.get("FRAMEPACK_ONNX_TRT_FP16", "1") == "1"
ONNX_TRT_INT8 = os.environ.get("FRAMEPACK_ONNX_TRT_INT8", "0") == "1"
ONNX_TRT_DEVICE_ID = int(os.environ.get("FRAMEPACK_ONNX_TRT_DEVICE_ID", "0"))
ONNX_TRT_WORKSPACE_MB = int(os.environ.get("FRAMEPACK_ONNX_TRT_WORKSPACE_MB", str(TRT_WORKSPACE_MB)))
ONNX_TRT_CACHE_DIR = os.environ.get("FRAMEPACK_ONNX_TRT_CACHE_DIR") or os.environ.get("FRAMEPACK_TENSORRT_CACHE_DIR")
print(f"DEBUG: USE_ONNX_ENGINES = {USE_ONNX_ENGINES}")

if TRT_TEXT_ENCODERS_ENABLED and not ENABLE_TENSORRT_RUNTIME:
    print("TensorRT text encoders requested but TensorRT runtime disabled; ignoring flag.")
    TRT_TEXT_ENCODERS_ENABLED = False

memory_backend = memory_v2 if args.use_memory_v2 else memory_v1
memory_optim = None
if args.use_memory_v2:
    memory_optim = memory_v2.MemoryOptimizationConfig(
        use_async_streams=True,
        use_pinned_memory=True,
        cache_memory_stats=True,
    )

cpu = memory_backend.cpu
gpu = memory_backend.gpu
DynamicSwapInstaller = memory_backend.DynamicSwapInstaller
unload_complete_models = memory_backend.unload_complete_models
load_model_as_complete = memory_backend.load_model_as_complete
fake_diffusers_current_device = memory_backend.fake_diffusers_current_device
force_free_vram = memory_backend.force_free_vram


def get_cuda_free_memory_gb(device=None):
    if args.use_memory_v2:
        return memory_backend.get_cuda_free_memory_gb(device=device, optim_config=memory_optim)
    return memory_backend.get_cuda_free_memory_gb(device=device)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0, aggressive=False):
    if args.use_memory_v2:
        return memory_backend.move_model_to_device_with_memory_preservation(
            model,
            target_device,
            preserved_memory_gb=preserved_memory_gb,
            aggressive=aggressive,
            optim_config=memory_optim,
        )
    return memory_backend.move_model_to_device_with_memory_preservation(
        model,
        target_device,
        preserved_memory_gb=preserved_memory_gb,
        aggressive=aggressive,
    )


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0, aggressive=False):
    if args.use_memory_v2:
        return memory_backend.offload_model_from_device_for_memory_preservation(
            model,
            target_device,
            preserved_memory_gb=preserved_memory_gb,
            aggressive=aggressive,
            optim_config=memory_optim,
        )
    return memory_backend.offload_model_from_device_for_memory_preservation(
        model,
        target_device,
        preserved_memory_gb=preserved_memory_gb,
        aggressive=aggressive,
    )


def load_model_chunked(model, target_device, max_chunk_size_mb=512):
    if args.use_memory_v2:
        return memory_backend.load_model_chunked(
            model,
            target_device,
            max_chunk_size_mb=max_chunk_size_mb,
            optim_config=memory_optim,
        )
    return memory_backend.load_model_chunked(model, target_device, max_chunk_size_mb=max_chunk_size_mb)

jit_mode = (args.jit_mode or "off").lower()
default_jit_artifact = os.path.join(os.path.dirname(__file__), "optimized_models", "transformer_torchscript.pt")
jit_artifact = os.environ.get("FRAMEPACK_JIT_ARTIFACT", default_jit_artifact)
jit_save = os.environ.get("FRAMEPACK_JIT_SAVE", jit_artifact if jit_mode != "off" else None)
jit_load = os.environ.get("FRAMEPACK_JIT_LOAD")
if jit_mode != "off" and not jit_load and jit_save and os.path.isfile(jit_save):
    jit_load = jit_save

torchscript_config = TorchScriptConfig(
    mode=jit_mode if jit_mode in {"off", "trace", "script"} else "off",
    strict_shapes=os.environ.get("FRAMEPACK_JIT_STRICT", "0") == "1",
    save_path=jit_save,
    load_path=jit_load,
)

INFERENCE_CONFIG = build_default_inference_config(torchscript=torchscript_config)
configure_inference_environment(INFERENCE_CONFIG)
MODEL_COMPUTE_DTYPE = INFERENCE_CONFIG.autocast_dtype if INFERENCE_CONFIG.autocast_dtype in (torch.float16, torch.bfloat16) else torch.float16
TENSOR_CORE_MULTIPLE = tensor_core_multiple_for_dtype(MODEL_COMPUTE_DTYPE, INFERENCE_CONFIG)
CPU_PREPROCESS_ACCEL = cpu_preprocessing_active()
if CPU_PREPROCESS_ACCEL:
    print('CPU-side preprocessing acceleration enabled (SIMD/OpenCV + oneDAL).')
ENABLE_FBCACHE = (os.environ.get("FRAMEPACK_ENABLE_FBCACHE", "1") == "1") and not args.disable_fbcache
CURRENT_FBCACHE_ENABLED = ENABLE_FBCACHE
FBCACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_FBCACHE_THRESHOLD", "0.035"))
FBCACHE_VERBOSE = os.environ.get("FRAMEPACK_FBCACHE_VERBOSE", "0") == "1"
ENABLE_SIM_CACHE = (os.environ.get("FRAMEPACK_ENABLE_SIM_CACHE", "1") == "1") and not args.disable_sim_cache
CURRENT_SIM_CACHE_ENABLED = ENABLE_SIM_CACHE
SIM_CACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_SIM_CACHE_THRESHOLD", "0.8"))
SIM_CACHE_MAX_SKIP = int(os.environ.get("FRAMEPACK_SIM_CACHE_MAX_SKIP", "1"))
SIM_CACHE_MAX_ENTRIES = int(os.environ.get("FRAMEPACK_SIM_CACHE_MAX_ENTRIES", "12"))
SIM_CACHE_USE_FAISS = os.environ.get("FRAMEPACK_SIM_CACHE_USE_FAISS", "0") == "1"
SIM_CACHE_VERBOSE = os.environ.get("FRAMEPACK_SIM_CACHE_VERBOSE", "0") == "1"
ENABLE_KV_CACHE = (os.environ.get("FRAMEPACK_ENABLE_KV_CACHE", "0") == "1") and not args.disable_kv_cache
CURRENT_KV_CACHE_ENABLED = ENABLE_KV_CACHE
KV_CACHE_LENGTH = int(os.environ.get("FRAMEPACK_KV_CACHE_LEN", "4096"))
KV_CACHE_VERBOSE = os.environ.get("FRAMEPACK_KV_CACHE_VERBOSE", "0") == "1"
XFORMERS_MODE = (args.xformers_mode or os.environ.get("FRAMEPACK_XFORMERS_MODE", "standard")).lower()
if XFORMERS_MODE not in {"off", "standard", "aggressive"}:
    XFORMERS_MODE = "standard"
if XFORMERS_MODE == "aggressive":
    os.environ["XFORMERS_ATTENTION_OP"] = "cutlass"
    os.environ["XFORMERS_FORCE_DISABLE_DROPOUT"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
set_attention_accel_mode(XFORMERS_MODE)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')
if PARALLEL_LOADERS > 1 and not high_vram and not FORCE_PARALLEL_LOADERS:
    print('Low VRAM detected -> disabling parallel model loading to save memory.')
    PARALLEL_LOADERS = 1
if os.environ.get("FRAMEPACK_ENABLE_COMPILE", "0") == "1" and not high_vram:
    if os.environ.get("TORCH_COMPILE_DISABLE_CUDAGRAPHS") not in {"1", "true", "TRUE"}:
        os.environ["TORCH_COMPILE_DISABLE_CUDAGRAPHS"] = "1"
    print('Low VRAM detected -> torch.compile will run with CUDA graphs disabled (compat mode).')
QUANT_BITS = int(os.environ.get("FRAMEPACK_QUANT_BITS", "8"))
USE_BITSANDBYTES = os.environ.get("FRAMEPACK_USE_BNB", "0") == "1"
BNB_CPU_OFFLOAD = os.environ.get("FRAMEPACK_BNB_CPU_OFFLOAD", "1") == "1"
USE_FSDP = os.environ.get("FRAMEPACK_USE_FSDP", "0") == "1"
_enable_quant_env = os.environ.get("FRAMEPACK_ENABLE_QUANT")
if _enable_quant_env is None:
    ENABLE_QUANT = False
    print('FRAMEPACK_ENABLE_QUANT not set -> skipping module quantization by default.')
else:
    ENABLE_QUANT = _enable_quant_env == "1"
ENABLE_PRUNE = os.environ.get("FRAMEPACK_ENABLE_PRUNE", "0") == "1"
ENABLE_OPT_CACHE = os.environ.get("FRAMEPACK_ENABLE_OPT_CACHE", "0") == "1"
ENABLE_MODULE_CACHE = os.environ.get("FRAMEPACK_ENABLE_MODULE_CACHE", "1") != "0"
ENABLE_COMPILE = os.environ.get("FRAMEPACK_ENABLE_COMPILE", "0") == "1"
# BetterTransformer: DISABLED by default - causes severe performance regression with memory offloading
# SDPA forces models to stay on GPU, breaking CPU/SSD offload workflows (20s/it → 428s/it regression observed)
# Only enable explicitly with FRAMEPACK_ENABLE_BETTERTRANSFORMER=1 if you have 80GB+ VRAM
ENABLE_BETTERTRANSFORMER = (os.environ.get("FRAMEPACK_ENABLE_BETTERTRANSFORMER", "0") == "1") and not args.disable_bettertransformer
CACHE_ROOT = os.environ.get(
    "FRAMEPACK_MODULE_CACHE_DIR",
    os.path.join(CACHE_BASE_DIR, "module_cache"),
)
CACHE_ROOT = os.path.abspath(CACHE_ROOT)
DEFAULT_CACHE_ITEMS = int(os.environ.get("FRAMEPACK_MODULE_CACHE_SIZE", "256"))
OPTIMIZED_MODEL_PATH = None
if ENABLE_OPT_CACHE and not FAST_START:
    OPTIMIZED_MODEL_PATH = os.environ.get(
        "FRAMEPACK_OPTIMIZED_TRANSFORMER",
        os.path.join(os.path.dirname(__file__), "optimized_models", "transformer_quantized.pt"),
    )
    OPTIMIZED_MODEL_DIR = os.path.dirname(OPTIMIZED_MODEL_PATH) or "."
    os.makedirs(OPTIMIZED_MODEL_DIR, exist_ok=True)
elif ENABLE_OPT_CACHE and FAST_START:
    print('Fast-start mode -> skipping optimized transformer cache initialization.')
bnb_config = None
bnb_device_map = os.environ.get("FRAMEPACK_BNB_DEVICE_MAP", "auto")

_parallel_workers = max(1, PARALLEL_LOADERS if PARALLEL_LOADERS > 0 else 1)
MODEL_REPO_SOURCE = _prepare_local_repo(
    "hunyuanvideo-community/HunyuanVideo",
    "FRAMEPACK_MODEL_REPO_PATH",
    preload=PRELOAD_REPOS,
    parallel_workers=_parallel_workers,
)
SIGLIP_REPO_SOURCE = _prepare_local_repo(
    "lllyasviel/flux_redux_bfl",
    "FRAMEPACK_SIGLIP_REPO_PATH",
    preload=PRELOAD_REPOS,
    parallel_workers=_parallel_workers,
)
TRANSFORMER_REPO_SOURCE = _prepare_local_repo(
    "lllyasviel/FramePackI2V_HY",
    "FRAMEPACK_TRANSFORMER_REPO_PATH",
    preload=PRELOAD_REPOS,
    parallel_workers=_parallel_workers,
)


def _resolve_cache_size(env_key, default):
    try:
        return int(os.environ.get(env_key, default))
    except ValueError:
        return default


def _parse_csv_env(env_key: str):
    raw = os.environ.get(env_key)
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",")]
    values = [item for item in values if item]
    return values or None


FP8_UTILS_AVAILABLE = optimize_state_dict_with_fp8 is not None and apply_fp8_monkey_patch is not None
ENABLE_FP8 = os.environ.get("FRAMEPACK_ENABLE_FP8", "0") == "1"
if ENABLE_FP8 and not FP8_UTILS_AVAILABLE:
    print('FRAMEPACK_ENABLE_FP8=1 but FP8 optimization utilities are unavailable -> disabling FP8 optimization.')
    ENABLE_FP8 = False
FP8_TARGET_LAYER_KEYS = _parse_csv_env("FRAMEPACK_FP8_TARGET_KEYS")
FP8_EXCLUDE_LAYER_KEYS = _parse_csv_env("FRAMEPACK_FP8_EXCLUDE_KEYS")
FP8_USE_SCALED_MM = os.environ.get("FRAMEPACK_FP8_USE_SCALED_MM", "0") == "1"
FP8_TRANSFORMER_ACTIVE = False
SLOW_MOTION_HINT = os.environ.get(
    "FRAMEPACK_SLOW_PROMPT_HINT",
    "move slowly, slow motion, gentle pacing, graceful deliberate movements",
)


MODULE_CACHE_WRAPPERS: List[ModuleCacheWrapper] = []


def register_cache_wrapper(wrapper):
    if isinstance(wrapper, ModuleCacheWrapper):
        MODULE_CACHE_WRAPPERS.append(wrapper)
    return wrapper


def set_cache_mode_for_wrappers(mode: str):
    normalized = (mode or 'hash').lower()
    for wrapper in MODULE_CACHE_WRAPPERS:
        wrapper.set_cache_mode(normalized)


def make_module_cache(name: str, default_size: int = None):
    if not ENABLE_MODULE_CACHE:
        return None
    size = _resolve_cache_size(f"FRAMEPACK_MODULE_CACHE_SIZE_{name.upper()}", default_size or DEFAULT_CACHE_ITEMS)
    if size <= 0:
        return None
    cache_dir = os.path.join(CACHE_ROOT, name)
    os.makedirs(cache_dir, exist_ok=True)
    return DiskLRUCache(cache_dir, max_items=size)


def wrap_with_module_cache(
    module,
    *,
    cache_name: str,
    normalization_tag: str,
    batched_arg_names: Sequence[str],
    hash_input_names: Sequence[str] = None,
    batch_dim: int = 0,
    default_cache_size: int = None,
    cache_mode: str = 'hash',
    semantic_embed_fn=None,
    semantic_threshold: float = 0.98,
):
    cache = make_module_cache(cache_name, default_cache_size)
    if cache is None:
        return module
    wrapper = ModuleCacheWrapper(
        module,
        cache,
        module_name=cache_name,
        module_revision=getattr(getattr(module, 'config', None), 'revision', None) or getattr(getattr(module, 'config', None), '_name_or_path', None) or cache_name,
        batch_dim=batch_dim,
        batched_arg_names=batched_arg_names,
        hash_input_names=hash_input_names,
        normalization_tag=normalization_tag,
        cache_mode=cache_mode,
        semantic_embed_fn=semantic_embed_fn,
        semantic_threshold=semantic_threshold,
    )
    return register_cache_wrapper(wrapper)


def _apply_prompt_hint(text: str, hint: str) -> str:
    base = (text or "").strip()
    hint = (hint or "").strip()
    if not hint:
        return base
    if hint.lower() in base.lower():
        return base
    if not base:
        return hint
    separator = ", " if not base.endswith((",", ";")) else " "
    return f"{base}{separator}{hint}"


def make_text_semantic_embedder(tokenizer, *, buckets: int = 768):
    def embed(sample_inputs):
        input_ids = sample_inputs.get('input_ids') if isinstance(sample_inputs, dict) else None
        if not torch.is_tensor(input_ids):
            return None
        ids = input_ids.view(-1).tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        tokens = [tok for tok in text.lower().split() if tok]
        vec = torch.zeros(buckets, dtype=torch.float32)
        if not tokens:
            vec[0] = 1.0
            return vec
        for token in tokens:
            token_hash = int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16)
            vec[token_hash % buckets] += 1.0
        return vec

    return embed


def make_image_semantic_embedder(output_size: int = 8):
    def embed(sample_inputs):
        pixel_values = sample_inputs.get('pixel_values') if isinstance(sample_inputs, dict) else None
        if not torch.is_tensor(pixel_values):
            return None
        tensor = pixel_values.to('cpu', dtype=torch.float32)
        if tensor.ndim >= 4:
            tensor = tensor[0]
        pooled = F.adaptive_avg_pool2d(tensor, output_size)
        return pooled.flatten()

    return embed

if USE_BITSANDBYTES:
    load_in_4bit = os.environ.get("FRAMEPACK_BNB_LOAD_IN_4BIT", "0") == "1"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=not load_in_4bit,
        load_in_4bit=load_in_4bit,
        llm_int8_enable_fp32_cpu_offload=BNB_CPU_OFFLOAD,
        bnb_4bit_compute_dtype=MODEL_COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=os.environ.get("FRAMEPACK_BNB_DOUBLE_QUANT", "1") == "1",
    )
    print(f'BitsAndBytes quantization enabled ({"4bit" if load_in_4bit else "8bit"}) with device_map={bnb_device_map}.')

text_encoder_kwargs = dict(torch_dtype=torch.float16)
clip_text_kwargs = dict(torch_dtype=torch.float16)
if USE_BITSANDBYTES:
    text_encoder_kwargs = dict(
        quantization_config=bnb_config,
        device_map=bnb_device_map,
        low_cpu_mem_usage=True,
    )
    clip_text_kwargs = dict(
        quantization_config=bnb_config,
        device_map=bnb_device_map,
        low_cpu_mem_usage=True,
    )

def _load_text_encoder():
    model = LlamaModel.from_pretrained(MODEL_REPO_SOURCE, subfolder='text_encoder', **text_encoder_kwargs)
    # Ensure the model config allows output_hidden_states
    model.config.output_hidden_states = True
    return model


def _load_clip_text():
    return CLIPTextModel.from_pretrained(MODEL_REPO_SOURCE, subfolder='text_encoder_2', **clip_text_kwargs)


def _load_vae():
    return AutoencoderKLHunyuanVideo.from_pretrained(
        MODEL_REPO_SOURCE, subfolder='vae', torch_dtype=torch.float16
    ).cpu()


def _load_image_encoder():
    def _build(dtype):
        return SiglipVisionModel.from_pretrained(
            SIGLIP_REPO_SOURCE,
            subfolder='image_encoder',
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).to('cpu')

    try:
        return _build(torch.float16)
    except NotImplementedError as exc:
        if "meta tensor" not in str(exc).lower():
            raise
        print("SigLip image encoder could not materialize in float16 on this platform. Retrying with float32 weights.")
        return _build(torch.float32)


if PARALLEL_LOADERS > 1:
    print(f'Loading core models with {PARALLEL_LOADERS} parallel workers.')
    with ThreadPoolExecutor(max_workers=PARALLEL_LOADERS) as executor:
        futures = {
            'text_encoder': executor.submit(_load_component, 'text encoder', _load_text_encoder),
            'text_encoder_2': executor.submit(_load_component, 'clip text encoder', _load_clip_text),
            'vae': executor.submit(_load_component, 'VAE', _load_vae),
            'image_encoder': executor.submit(_load_component, 'SigLip image encoder', _load_image_encoder),
        }
        text_encoder = futures['text_encoder'].result()
        text_encoder_2 = futures['text_encoder_2'].result()
        vae = futures['vae'].result()
        image_encoder = futures['image_encoder'].result()

        # Enable output_hidden_states in model config
        text_encoder.config.output_hidden_states = True
        print("Set text_encoder.config.output_hidden_states = True")
else:
    text_encoder = _load_component('text encoder', _load_text_encoder)
    text_encoder_2 = _load_component('clip text encoder', _load_clip_text)
    vae = _load_component('VAE', _load_vae)
    image_encoder = _load_component('SigLip image encoder', _load_image_encoder)

# Enable output_hidden_states in model config
text_encoder.config.output_hidden_states = True
print("Set text_encoder.config.output_hidden_states = True")

if not USE_BITSANDBYTES:
    text_encoder = text_encoder.cpu()
    text_encoder_2 = text_encoder_2.cpu()
tokenizer = LlamaTokenizerFast.from_pretrained(MODEL_REPO_SOURCE, subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained(MODEL_REPO_SOURCE, subfolder='tokenizer_2')
feature_extractor = SiglipImageProcessor.from_pretrained(SIGLIP_REPO_SOURCE, subfolder='feature_extractor')


optimized_transformer_loaded = False
if OPTIMIZED_MODEL_PATH and os.path.isfile(OPTIMIZED_MODEL_PATH):
    transformer_core = torch.load(OPTIMIZED_MODEL_PATH, map_location='cpu')
    optimized_transformer_loaded = True
    print(f'Loaded optimized transformer weights from {OPTIMIZED_MODEL_PATH}')
else:
    transformer_core = HunyuanVideoTransformer3DModelPacked.from_pretrained(TRANSFORMER_REPO_SOURCE, torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer_core.eval()

# Apply BetterTransformer optimization for encoders
# WARNING: Can cause severe performance regression with memory offloading (20s/it → 428s/it)
# SDPA forces models to stay on GPU, interfering with CPU/SSD offload workflows
if ENABLE_BETTERTRANSFORMER and not FAST_START:
    print("="*80)
    print("⚠️  BETTERTRANSFORMER OPTIMIZATION (EXPERIMENTAL)")
    print("="*80)
    print("WARNING: BetterTransformer/SDPA can cause severe performance degradation!")
    print("  - SDPA forces models onto GPU, blocking CPU/SSD memory offloading")
    print("  - Known issue: 20s/it → 428s/it regression with memory_v2 backend")
    print("  - Only use if you have 80GB+ VRAM and keep all models on GPU")
    print("  - To disable: Remove FRAMEPACK_ENABLE_BETTERTRANSFORMER=1 or add --disable-bettertransformer")
    print("")

    # Check if we're using memory offloading (this is likely problematic)
    if args.use_memory_v2:
        print("⚠️  CRITICAL: BetterTransformer + memory_v2 backend detected!")
        print("  This combination causes severe slowdowns. Proceeding anyway...")

    text_encoder = apply_bettertransformer_optimization(text_encoder, "text_encoder (LLaMA)")
    text_encoder_2 = apply_bettertransformer_optimization(text_encoder_2, "text_encoder_2 (CLIP)")
    image_encoder = apply_bettertransformer_optimization(image_encoder, "image_encoder (SigLip)")

    if USE_BITSANDBYTES:
        print("✓ BetterTransformer SDPA optimization applied with BitsAndBytes quantization")
    print("="*80 + "\n")
elif not ENABLE_BETTERTRANSFORMER:
    print("BetterTransformer optimization disabled by default (safe)")
    print("  (Can cause performance issues with memory offloading)")
elif FAST_START:
    print("Fast-start mode -> skipping BetterTransformer optimization")

fp8_buffers_present = any(hasattr(module, "scale_weight") for module in transformer_core.modules())
if fp8_buffers_present:
    FP8_TRANSFORMER_ACTIVE = True

TENSORRT_RUNTIME = None
TENSORRT_DECODER = None
TENSORRT_ENCODER = None
TENSORRT_AVAILABLE = False
TENSORRT_SIGLIP_ENCODER = None
print(f"DEBUG: About to check ENABLE_TENSORRT_RUNTIME = {ENABLE_TENSORRT_RUNTIME}")
if _TORCH_TRT_IMPORT_ERROR is not None:
    print(f"WARNING: torch_tensorrt import failed: {_TORCH_TRT_IMPORT_ERROR}")
    print("TensorRT features will be disabled. Install torch-tensorrt to enable.")
if ENABLE_TENSORRT_RUNTIME:
    print("DEBUG: Entering TensorRT initialization block")
    try:
        # Allow custom TensorRT cache directory via environment variable
        trt_cache_dir = os.environ.get('FRAMEPACK_TENSORRT_CACHE_DIR')
        print(f"DEBUG: Creating TensorRTRuntime with workspace={TRT_WORKSPACE_MB} MB")
        TENSORRT_RUNTIME = TensorRTRuntime(
            enabled=True,
            precision=MODEL_COMPUTE_DTYPE if MODEL_COMPUTE_DTYPE in (torch.float16, torch.bfloat16) else torch.float16,
            workspace_size_mb=TRT_WORKSPACE_MB,
            max_aux_streams=TRT_MAX_AUX_STREAMS,
            cache_dir=trt_cache_dir,
        )
        print(f"DEBUG: TENSORRT_RUNTIME.is_ready = {TENSORRT_RUNTIME.is_ready}")
        print(f"DEBUG: TENSORRT_RUNTIME.enabled = {TENSORRT_RUNTIME.enabled}")
        if TENSORRT_RUNTIME.failure_reason:
            print(f"DEBUG: TENSORRT_RUNTIME.failure_reason = {TENSORRT_RUNTIME.failure_reason}")
        if TENSORRT_RUNTIME.is_ready:
            # Note: Disabling torch-tensorrt runtime for VAE due to compatibility issues
            # ONNX models will be created and can be converted to TensorRT engines using
            # the optimize_onnx_with_tensorrt.py script or build_tensorrt_engines.py
            print("Note: torch-tensorrt runtime for VAE has compatibility issues.")
            print("ONNX models will be created for manual TensorRT conversion.")
            TENSORRT_DECODER = None  # Disabled - use ONNX -> TensorRT workflow instead
            TENSORRT_ENCODER = None  # Disabled - use ONNX -> TensorRT workflow instead
            TENSORRT_AVAILABLE = True  # Still mark as available for other components
        else:
            TENSORRT_DECODER = None
            TENSORRT_ENCODER = None
            print(f"TensorRT disabled: {TENSORRT_RUNTIME.failure_reason}")
    except Exception as exc:
        TENSORRT_RUNTIME = None
        TENSORRT_DECODER = None
        TENSORRT_ENCODER = None
        print(f"Failed to initialize TensorRT runtime: {exc}")
else:
    print("TensorRT runtime disabled (use --enable-tensorrt to opt-in).")

TENSORRT_TRANSFORMER = None
TENSORRT_LLAMA_TEXT_ENCODER = None
TENSORRT_CLIP_TEXT_ENCODER = None
TENSORRT_SIGLIP_ENCODER = None
TENSORRT_SIGLIP_ENGINE = None  # For pre-built TensorRT engines

# Check for ONNX models if --use-onnx-engines is set
if USE_ONNX_ENGINES or ENABLE_TENSORRT_RUNTIME:
    try:
        from diffusers_helper.trt_engine_loader import load_engine_if_available
        print("Checking for pre-built TensorRT engine for image encoder...")
        TENSORRT_SIGLIP_ENGINE = load_engine_if_available("siglip_image_encoder")
        if TENSORRT_SIGLIP_ENGINE is not None:
            print("Loaded pre-built TensorRT engine for image encoder")
            # Create a wrapper to use the engine
            class TRTImageEncoderWrapper:
                def __init__(self, engine):
                    self.engine = engine
                    self.device = torch.device('cuda')
                    self.dtype = torch.float16

                def __call__(self, pixel_values):
                    """Forward pass using TensorRT engine."""
                    # Ensure input is on GPU and correct dtype
                    if not pixel_values.is_cuda:
                        pixel_values = pixel_values.to(self.device)
                    if pixel_values.dtype != self.dtype:
                        pixel_values = pixel_values.to(self.dtype)

                    # Run TensorRT inference
                    outputs = self.engine(pixel_values=pixel_values)

                    # Create output object similar to HuggingFace model
                    class TRTOutput:
                        def __init__(self, last_hidden_state, pooler_output):
                            self.last_hidden_state = last_hidden_state
                            self.pooler_output = pooler_output

                    return TRTOutput(
                        last_hidden_state=outputs['last_hidden_state'],
                        pooler_output=outputs['pooler_output']
                    )

            # Wrap the engine to make it compatible with existing code
            setattr(image_encoder, "_framepack_trt_engine", TRTImageEncoderWrapper(TENSORRT_SIGLIP_ENGINE))
            print("Image encoder will use pre-built TensorRT engine for inference")
        else:
            print("No pre-built TensorRT engine found for image encoder")
            # Try ONNX Runtime if --use-onnx-engines is set
            if USE_ONNX_ENGINES:
                print("Attempting to load ONNX model with ONNX Runtime...")
                try:
                    from diffusers_helper.onnx_runtime_loader import load_onnx_model_if_available
                    onnx_model = load_onnx_model_if_available(
                        "siglip_image_encoder",
                        use_gpu=True,
                        enable_tensorrt=ONNX_TRT_PROVIDER_ENABLED,
                        trt_device_id=ONNX_TRT_DEVICE_ID,
                        trt_workspace_size_mb=ONNX_TRT_WORKSPACE_MB,
                        trt_fp16=ONNX_TRT_FP16,
                        trt_int8=ONNX_TRT_INT8,
                        trt_cache_dir=ONNX_TRT_CACHE_DIR,
                    )
                    if onnx_model is not None:
                        print("Loaded ONNX model with ONNX Runtime for image encoder")
                        # Create a wrapper to use the ONNX model
                        class ONNXImageEncoderWrapper:
                            def __init__(self, onnx_model):
                                self.onnx_model = onnx_model
                                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                self.dtype = torch.float16

                            def __call__(self, pixel_values):
                                """Forward pass using ONNX Runtime."""
                                # Ensure input is correct dtype
                                if pixel_values.dtype != self.dtype:
                                    pixel_values = pixel_values.to(self.dtype)

                                # Run ONNX inference
                                outputs = self.onnx_model(pixel_values=pixel_values)

                                # Create output object similar to HuggingFace model
                                class ONNXOutput:
                                    def __init__(self, last_hidden_state, pooler_output):
                                        self.last_hidden_state = last_hidden_state
                                        self.pooler_output = pooler_output

                                return ONNXOutput(
                                    last_hidden_state=outputs['last_hidden_state'],
                                    pooler_output=outputs['pooler_output']
                                )

                        # Wrap the ONNX model to make it compatible with existing code
                        setattr(image_encoder, "_framepack_trt_engine", ONNXImageEncoderWrapper(onnx_model))
                        print("Image encoder will use ONNX Runtime for inference")
                    else:
                        print("No ONNX model found for image encoder. Falling back to PyTorch.")
                except Exception as onnx_exc:
                    print(f"WARNING: Failed to load ONNX model: {onnx_exc}")
                    print("Falling back to PyTorch.")
    except Exception as exc:
        print(f"WARNING: Failed to load TensorRT engine for image encoder: {exc}")
        import traceback
        traceback.print_exc()
        TENSORRT_SIGLIP_ENGINE = None

# Fallback to torch-tensorrt runtime if no pre-built engine
if TENSORRT_SIGLIP_ENGINE is None and TENSORRT_AVAILABLE and TENSORRT_RUNTIME is not None:
    try:
        print("DEBUG: Creating TensorRT SiGLIP encoder wrapper...")
        def _siglip_forward(pixel_values):
            return image_encoder(pixel_values=pixel_values)

        TENSORRT_SIGLIP_ENCODER = TensorRTCallable(
            runtime=TENSORRT_RUNTIME,
            name="siglip_image_encoder",
            forward_fn=_siglip_forward,
        )
        setattr(image_encoder, "_framepack_trt_callable", TENSORRT_SIGLIP_ENCODER)
        print("DEBUG: TensorRT SiGLIP encoder wrapper created successfully")
    except Exception as exc:
        print(f"WARNING: Failed to create TensorRT SiGLIP encoder wrapper: {exc}")
        import traceback
        traceback.print_exc()
        TENSORRT_SIGLIP_ENCODER = None

    # Check and convert models to ONNX if needed for TensorRT
    if TRT_TEXT_ENCODERS_ENABLED or TRT_TRANSFORMER_ENABLED or TENSORRT_AVAILABLE:
        try:
            from diffusers_helper.onnx_converter import (
                prepare_text_encoder_for_tensorrt,
                prepare_clip_text_encoder_for_tensorrt,
                prepare_transformer_for_tensorrt,
                prepare_image_encoder_for_tensorrt,
                prepare_vae_encoder_for_tensorrt,
                prepare_vae_decoder_for_tensorrt,
            )

            print("\n" + "="*80)
            print("TensorRT Model Preparation: Checking ONNX conversion status")
            print("="*80)

            # Prepare text encoders if needed
            if TRT_TEXT_ENCODERS_ENABLED:
                print("\nChecking LLAMA text encoder...")
                text_encoder_temp = text_encoder.to('cuda')
                llama_onnx_path = prepare_text_encoder_for_tensorrt(text_encoder_temp, device='cuda')
                text_encoder_temp = text_encoder_temp.cpu()

                if llama_onnx_path is None:
                    print("WARNING: LLAMA text encoder ONNX conversion failed. TensorRT text encoder will be disabled.")
                else:
                    print(f"LLAMA text encoder ready for TensorRT: {llama_onnx_path}")

                print("\nChecking CLIP text encoder...")
                clip_encoder_temp = text_encoder_2.to('cuda')
                clip_onnx_path = prepare_clip_text_encoder_for_tensorrt(clip_encoder_temp, device='cuda')
                clip_encoder_temp = clip_encoder_temp.cpu()

                if clip_onnx_path is None:
                    print("WARNING: CLIP text encoder ONNX conversion failed. TensorRT text encoder will be disabled.")
                else:
                    print(f"CLIP text encoder ready for TensorRT: {clip_onnx_path}")

            # Prepare transformer if needed
            if TRT_TRANSFORMER_ENABLED:
                print("\nChecking transformer model...")
                transformer_temp = transformer_core.to('cuda')
                transformer_onnx_path = prepare_transformer_for_tensorrt(transformer_temp, device='cuda')
                transformer_temp = transformer_temp.cpu()

                if transformer_onnx_path is None:
                    print("WARNING: Transformer ONNX conversion failed. TensorRT transformer will be disabled.")
                else:
                    print(f"Transformer ready for TensorRT: {transformer_onnx_path}")

            # Prepare image encoder (SigLip) for TensorRT if base TensorRT is enabled
            if TENSORRT_AVAILABLE:
                print("\nChecking SigLip image encoder...")
                image_encoder_temp = image_encoder.to('cuda')
                image_encoder_onnx_path = prepare_image_encoder_for_tensorrt(image_encoder_temp, device='cuda')
                image_encoder_temp = image_encoder_temp.cpu()

                if image_encoder_onnx_path is None:
                    print("WARNING: Image encoder ONNX conversion failed. TensorRT image encoder may not work optimally.")
                else:
                    print(f"Image encoder ready for TensorRT: {image_encoder_onnx_path}")

                # Skip VAE encoder/decoder for now due to device placement issues
                # The Hunyuan Video VAE creates CPU tensors dynamically during forward pass
                # which causes ONNX export to fail. This would require modifications to the
                # diffusers library to fix properly.
                print("\nSkipping VAE encoder/decoder ONNX conversion...")
                print("Note: VAE has internal device placement issues that prevent ONNX export.")
                print("The VAE will continue to use PyTorch (no TensorRT acceleration).")

            print("="*80)
            print("ONNX Model Preparation Complete!")
            print("="*80)
            print("\nNext steps to use ONNX models:")
            print("\nOption 1: Use ONNX Runtime (simpler, no additional setup)")
            print("   python FramePack/demo_gradio.py --use-onnx-engines")
            print("   Requires: pip install onnxruntime-gpu")
            print("\nOption 2: Build TensorRT engines (faster, requires more setup)")
            print("   1. Build engines:")
            print("      python FramePack/build_tensorrt_engines.py --model siglip_image_encoder")
            print("   2. Run with TensorRT:")
            print("      python FramePack/demo_gradio.py --enable-tensorrt")
            print("   Requires: pip install tensorrt pycuda")
            print("\nONNX models location: Cache/onnx_models/")
            print("TensorRT engines location: Cache/tensorrt_engines/")
            print("\nNote: TensorRT provides better performance but ONNX Runtime is easier to setup.")
            print("="*80 + "\n")

        except Exception as exc:
            print(f"Failed to prepare models for TensorRT: {exc}")
            import traceback
            traceback.print_exc()
            print("Continuing with PyTorch models (TensorRT may not work optimally)")

    if TRT_TEXT_ENCODERS_ENABLED:
        try:
            print("Initializing TensorRT text encoders...")
            TENSORRT_LLAMA_TEXT_ENCODER = TensorRTTextEncoder(
                text_encoder,
                TENSORRT_RUNTIME,
            )
            TENSORRT_CLIP_TEXT_ENCODER = TensorRTCLIPTextEncoder(
                text_encoder_2,
                TENSORRT_RUNTIME,
            )
            print("TensorRT text encoders initialized.")
        except Exception as exc:
            TENSORRT_LLAMA_TEXT_ENCODER = None
            TENSORRT_CLIP_TEXT_ENCODER = None
            print(f"Failed to initialize TensorRT text encoders: {exc}")
    else:
        print("TensorRT text encoders disabled (using PyTorch for text encoding)")

    # Initialize TensorRT transformer wrapper if enabled
    if TRT_TRANSFORMER_ENABLED:
        # Check for incompatible configurations
        warnings = []
        if USE_BITSANDBYTES:
            warnings.append("BitsAndBytes quantization is enabled - TensorRT requires FP16/BF16 precision")
        if BNB_CPU_OFFLOAD:
            warnings.append("CPU offload is enabled - TensorRT requires full GPU model")
        if USE_FSDP:
            warnings.append("FSDP is enabled - TensorRT may not work with distributed models")

        if warnings:
            print("\n" + "="*80)
            print("WARNING: TensorRT transformer has compatibility issues:")
            for w in warnings:
                print(f"  - {w}")
            print("\nTensorRT transformer will fall back to PyTorch (no speedup).")
            print("Recommended command for TensorRT acceleration:")
            print("  python demo_gradio.py --enable-tensorrt --tensorrt-transformer")
            print("  (Remove BNB and CPU offload flags)")
            print("="*80 + "\n")

        try:
            print(f"Initializing TensorRT transformer wrapper (max_cached_shapes={TRT_MAX_CACHED_SHAPES})...")

            # Check if transformer is on GPU and in correct dtype
            if hasattr(transformer_core, 'device'):
                print(f"Transformer device: {transformer_core.device}")
            if hasattr(transformer_core, 'dtype'):
                print(f"Transformer dtype: {transformer_core.dtype}")

            TENSORRT_TRANSFORMER = TensorRTTransformer(
                transformer_core,
                TENSORRT_RUNTIME,
                fallback_fn=None,
            )
            TENSORRT_TRANSFORMER._max_cached_shapes = TRT_MAX_CACHED_SHAPES
            print("TensorRT transformer wrapper initialized. Engines will compile on first use per shape.")
            print(f"DEBUG: TENSORRT_TRANSFORMER.runtime.is_ready = {TENSORRT_TRANSFORMER.runtime.is_ready}")
            print(f"DEBUG: TENSORRT_TRANSFORMER.runtime.enabled = {TENSORRT_TRANSFORMER.runtime.enabled}")
            print(f"DEBUG: TENSORRT_TRANSFORMER.runtime.failure_reason = {TENSORRT_TRANSFORMER.runtime.failure_reason}")
            if not warnings:
                print("NOTE: First compilation may take 5-15 minutes and requires ~16GB GPU VRAM.")
        except Exception as exc:
            print(f"Failed to initialize TensorRT transformer: {exc}")
            import traceback
            traceback.print_exc()
            TENSORRT_TRANSFORMER = None
    else:
        print("TensorRT transformer disabled (set FRAMEPACK_TRT_TRANSFORMER=1 to enable).")

if ENABLE_FP8 and optimize_state_dict_with_fp8 is not None and apply_fp8_monkey_patch is not None:
    if FP8_TRANSFORMER_ACTIVE:
        print('FP8 scale weights detected on transformer; skipping re-optimization.')
    else:
        fp8_device = gpu if torch.cuda.is_available() else None
        fp8_state_dict = None
        try:
            fp8_state_dict = transformer_core.state_dict()
            optimized_state = optimize_state_dict_with_fp8(
                fp8_state_dict,
                fp8_device,
                target_layer_keys=FP8_TARGET_LAYER_KEYS,
                exclude_layer_keys=FP8_EXCLUDE_LAYER_KEYS,
                move_to_device=False,
            )
            apply_fp8_monkey_patch(transformer_core, optimized_state, use_scaled_mm=FP8_USE_SCALED_MM)
            transformer_core.load_state_dict(optimized_state, strict=True)
            FP8_TRANSFORMER_ACTIVE = True
            print('Applied FP8 optimization to transformer weights.')
        except Exception as exc:
            print(f'FP8 optimization failed: {exc}')
        finally:
            if fp8_state_dict is not None:
                del fp8_state_dict

if ENABLE_FBCACHE:
    transformer_core.enable_first_block_cache(
        enabled=True,
        threshold=FBCACHE_THRESHOLD,
        verbose=FBCACHE_VERBOSE,
    )
    print(f'First block cache enabled (threshold={FBCACHE_THRESHOLD:.4f}).')
    fb_state = _load_runtime_cache_state("first_block_cache")
    if fb_state:
        transformer_core.first_block_cache.load_state_dict(fb_state)
        print('Loaded persisted first-block cache state.')
    CURRENT_FBCACHE_ENABLED = True
else:
    transformer_core.enable_first_block_cache(enabled=False)
    CURRENT_FBCACHE_ENABLED = False

if ENABLE_SIM_CACHE:
    transformer_core.enable_similarity_cache(
        enabled=True,
        threshold=SIM_CACHE_THRESHOLD,
        max_skip=SIM_CACHE_MAX_SKIP,
        max_entries=SIM_CACHE_MAX_ENTRIES,
        use_faiss=SIM_CACHE_USE_FAISS,
        verbose=SIM_CACHE_VERBOSE,
    )
    print(
        f'Similarity cache enabled (threshold={SIM_CACHE_THRESHOLD:.3f}, '
        f'max_skip={SIM_CACHE_MAX_SKIP}, entries={SIM_CACHE_MAX_ENTRIES}).'
    )
    sim_state = _load_runtime_cache_state("similarity_cache")
    if sim_state and transformer_core.similarity_cache_manager is not None:
        transformer_core.similarity_cache_manager.load_state_dict(sim_state)
        print('Loaded persisted similarity cache state.')
    CURRENT_SIM_CACHE_ENABLED = True
else:
    transformer_core.enable_similarity_cache(enabled=False)
    CURRENT_SIM_CACHE_ENABLED = False

if ENABLE_KV_CACHE:
    transformer_core.enable_kv_cache(
        enabled=True,
        max_length=KV_CACHE_LENGTH,
        verbose=KV_CACHE_VERBOSE,
    )
    print(f'KV cache enabled (max length={KV_CACHE_LENGTH}).')
    kv_state = _load_runtime_cache_state("kv_cache")
    if kv_state and transformer_core.kv_cache_manager is not None:
        transformer_core.kv_cache_manager.load_state_dict(kv_state)
        print('Loaded persisted KV cache state.')
    CURRENT_KV_CACHE_ENABLED = True
else:
    transformer_core.enable_kv_cache(enabled=False)
    CURRENT_KV_CACHE_ENABLED = False


# Relationship Trainer (Experimental)
RELATIONSHIP_TRAINER_ENABLED_FOR_RUN = False
ACTIVE_RELATIONSHIP_MODE = "off"
relationship_trainer = None
try:
    # inner_dim of the transformer is num_attention_heads * attention_head_dim = 24 * 128 = 3072
    relationship_trainer = HiddenStateRelationshipTrainer(hidden_dim=3072, device=str(cpu))
    rt_state = _load_runtime_cache_state("relationship_trainer")
    if rt_state:
        relationship_trainer.load_state_dict(rt_state)
        print('Loaded persisted relationship trainer state.')
except Exception as e:
    print(f"Warning: Could not initialize HiddenStateRelationshipTrainer: {e}")
    relationship_trainer = None

RELATIONSHIP_SAMPLE_LIMIT = int(os.environ.get("FRAMEPACK_RELATIONSHIP_SAMPLE_LIMIT", "2048"))
RELATIONSHIP_RESIDUAL_CACHE_NAME = "relationship_residual_predictors"
RELATIONSHIP_MODULATION_CACHE_NAME = "relationship_modulation_predictors"


def _load_relationship_cache(name: str) -> Dict[str, Any]:
    state = _load_runtime_cache_state(name)
    if isinstance(state, dict):
        return state
    if state is not None:
        print(f'Warning: Expected dict for "{name}" but received {type(state).__name__}; ignoring.')
    return {}


RESIDUAL_TRAINER_STATES: Dict[str, Any] = _load_relationship_cache(RELATIONSHIP_RESIDUAL_CACHE_NAME)
MODULATION_TRAINER_STATES: Dict[str, Any] = _load_relationship_cache(RELATIONSHIP_MODULATION_CACHE_NAME)
if RESIDUAL_TRAINER_STATES:
    print(f"Loaded {len(RESIDUAL_TRAINER_STATES)} per-block timestep residual predictors.")
if MODULATION_TRAINER_STATES:
    print(f"Loaded {len(MODULATION_TRAINER_STATES)} per-block modulation predictors.")

RESIDUAL_TRAINERS: Dict[Tuple[str, int], DiTTimestepResidualTrainer] = {}
MODULATION_TRAINERS: Dict[Tuple[str, int], DiTTimestepModulationTrainer] = {}
RESIDUAL_READY_KEYS: set[str] = set(RESIDUAL_TRAINER_STATES.keys())
MODULATION_READY_KEYS: set[str] = set(MODULATION_TRAINER_STATES.keys())


def _relationship_block_key(block_type: str, block_id: int) -> str:
    return f"{block_type}:{block_id}"


def _clone_sample_to_cpu(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, tuple):
        return tuple(_clone_sample_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [_clone_sample_to_cpu(v) for v in value]
    return value


def _move_sample_to_device(value, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_sample_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_sample_to_device(v, device) for v in value]
    return value


def _get_backbone_block(backbone, block_type: str, block_id: int):
    if block_type == "dual":
        blocks = getattr(backbone, "transformer_blocks", None)
    elif block_type == "single":
        blocks = getattr(backbone, "single_transformer_blocks", None)
    else:
        raise ValueError(f"Unknown block_type '{block_type}'")
    if blocks is None or block_id >= len(blocks):
        raise IndexError(f"block_id {block_id} invalid for type '{block_type}'")
    return blocks[block_id]


def _call_block_default(block, block_type: str, hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis, image_rotary_emb):
    if block_type == "dual":
        return block(hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis)
    return block(hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb)


def _ensure_residual_trainer(backbone, block_type: str, block_id: int, temb_dim: int, lr: float, device) -> DiTTimestepResidualTrainer:
    key = (block_type, block_id)
    trainer = RESIDUAL_TRAINERS.get(key)
    if trainer is None:
        block = _get_backbone_block(backbone, block_type, block_id)
        if block_type == "dual":
            hidden_dim = block.norm1.linear.in_features
        else:
            hidden_dim = block.norm.linear.in_features
        trainer = DiTTimestepResidualTrainer(
            block=block,
            hidden_dim=hidden_dim,
            temb_dim=temb_dim,
            lr=lr,
            device=device,
        )
        state = RESIDUAL_TRAINER_STATES.get(_relationship_block_key(block_type, block_id))
        if state:
            trainer.load_state(state)
        RESIDUAL_TRAINERS[key] = trainer
    else:
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
    return trainer


def _ensure_modulation_trainer(backbone, block_type: str, block_id: int, temb_dim: int, lr: float, device) -> DiTTimestepModulationTrainer:
    key = (block_type, block_id)
    trainer = MODULATION_TRAINERS.get(key)
    if trainer is None:
        block = _get_backbone_block(backbone, block_type, block_id)
        if block_type == "dual":
            mod_dim = block.norm1.linear.in_features
        else:
            mod_dim = block.norm.linear.in_features
        trainer = DiTTimestepModulationTrainer(
            block=block,
            temb_dim=temb_dim,
            mod_dim=mod_dim,
            lr=lr,
            device=device,
        )
        state = MODULATION_TRAINER_STATES.get(_relationship_block_key(block_type, block_id))
        if state:
            trainer.load_state(state)
        MODULATION_TRAINERS[key] = trainer
    else:
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
    return trainer


def _install_residual_block_overrides(backbone, device, lr):
    for block_type, blocks in (("dual", getattr(backbone, "transformer_blocks", [])), ("single", getattr(backbone, "single_transformer_blocks", []))):
        for block_id, block in enumerate(blocks):
            def _override(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=None,
                freqs_cis=None,
                image_rotary_emb=None,
                _block=block,
                _block_type=block_type,
                _block_id=block_id,
            ):
                key = _relationship_block_key(_block_type, _block_id)
                trainer = _ensure_residual_trainer(backbone, _block_type, _block_id, temb.shape[-1], lr, device)
                if key not in RESIDUAL_READY_KEYS:
                    return _call_block_default(
                        _block,
                        _block_type,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        freqs_cis,
                        image_rotary_emb,
                    )
                approx_hidden, approx_encoder = trainer.approximate_forward(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask=attention_mask,
                    freqs_cis=freqs_cis if _block_type == "dual" else None,
                    image_rotary_emb=image_rotary_emb if _block_type == "single" else None,
                )
                return approx_hidden, approx_encoder

            backbone.set_block_override(block_type, block_id, _override)


def _install_modulation_block_overrides(backbone, device, lr):
    for block_type, blocks in (("dual", getattr(backbone, "transformer_blocks", [])), ("single", getattr(backbone, "single_transformer_blocks", []))):
        for block_id, block in enumerate(blocks):
            def _override(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=None,
                freqs_cis=None,
                image_rotary_emb=None,
                _block=block,
                _block_type=block_type,
                _block_id=block_id,
            ):
                key = _relationship_block_key(_block_type, _block_id)
                trainer = _ensure_modulation_trainer(backbone, _block_type, _block_id, temb.shape[-1], lr, device)
                if key not in MODULATION_READY_KEYS:
                    return _call_block_default(
                        _block,
                        _block_type,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        freqs_cis,
                        image_rotary_emb,
                    )
                temb_for_trainer = temb.detach()
                if temb_for_trainer.device != trainer.device:
                    temb_for_trainer = temb_for_trainer.to(trainer.device)
                gamma_hat, beta_hat = trainer.predict(temb_for_trainer)
                if _block_type == "dual":
                    _block.norm1.set_external_msa_modulation(gamma_hat, beta_hat)
                    _block.norm1_context.set_external_msa_modulation(gamma_hat, beta_hat)
                    return _block(hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis)
                _block.norm.set_external_msa_modulation(gamma_hat, beta_hat)
                return _block(hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb)

            backbone.set_block_override(block_type, block_id, _override)


def _configure_relationship_block_overrides(backbone, mode: str, device, lr: float):
    if backbone is None or not hasattr(backbone, "set_block_override"):
        return
    if hasattr(backbone, "clear_block_overrides"):
        backbone.clear_block_overrides()
    if mode == "residual":
        _install_residual_block_overrides(backbone, device, lr)
    elif mode == "modulation":
        _install_modulation_block_overrides(backbone, device, lr)


def _export_relationship_trainer_states(
    trainers: Dict[Tuple[str, int], Any],
    existing: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(existing)
    for (block_type, block_id), trainer in trainers.items():
        try:
            payload[_relationship_block_key(block_type, block_id)] = trainer.export_state()
        except Exception as exc:
            print(f"Warning: Failed to export predictor for {block_type}:{block_id}: {exc}")
    return payload


def _train_hidden_state_samples(samples: List[Dict[str, Any]], trainer: HiddenStateRelationshipTrainer, batch_size: int) -> Tuple[float, int]:
    if trainer is None or not samples:
        return 0.0, 0
    import random

    target_batch = max(1, min(len(samples), int(batch_size)))
    chosen = random.sample(samples, k=target_batch)
    total_loss = 0.0

    with torch.enable_grad():
        for sample in chosen:
            input_h = _move_sample_to_device(sample["h_in"], trainer.device)
            output_h = _move_sample_to_device(sample["h_out"], trainer.device)
            residual = output_h - input_h
            total_loss += trainer.train_step(input_h, residual)

    return total_loss, len(chosen)


def _train_residual_predictors(
    samples: List[Dict[str, Any]],
    backbone,
    lr: float,
    batch_size: int,
    device,
) -> Tuple[float, int]:
    if not samples:
        return 0.0, 0
    import random

    target_batch = max(1, min(len(samples), int(batch_size)))
    chosen = random.sample(samples, k=target_batch)
    total_loss = 0.0
    processed = 0
    updated_blocks: set[Tuple[str, int]] = set()

    with torch.enable_grad():
        for sample in chosen:
            temb = sample.get("temb")
            if temb is None:
                continue
            block_type = sample["block_type"]
            block_id = sample["block_id"]
            trainer = _ensure_residual_trainer(backbone, block_type, block_id, temb.shape[-1], lr, device)
            hidden_states = _move_sample_to_device(sample["h_in"], trainer.device)
            encoder_states = _move_sample_to_device(sample.get("encoder"), trainer.device)
            temb_dev = _move_sample_to_device(temb, trainer.device)
            attention_mask = _move_sample_to_device(sample.get("attention_mask"), trainer.device)
            rope_freqs = _move_sample_to_device(sample.get("rope_freqs"), trainer.device)
            try:
                loss = trainer.train_step(
                    hidden_states,
                    encoder_states,
                    temb_dev,
                    attention_mask=attention_mask,
                    freqs_cis=rope_freqs if block_type == "dual" else None,
                    image_rotary_emb=rope_freqs if block_type == "single" else None,
                )
            except Exception as exc:
                print(f"Warning: Residual predictor update failed for {block_type}:{block_id}: {exc}")
                continue
            total_loss += loss
            processed += 1
            RESIDUAL_READY_KEYS.add(_relationship_block_key(block_type, block_id))
            updated_blocks.add((block_type, block_id))

    for block_type, block_id in updated_blocks:
        key = _relationship_block_key(block_type, block_id)
        try:
            RESIDUAL_TRAINER_STATES[key] = RESIDUAL_TRAINERS[(block_type, block_id)].export_state()
        except Exception as exc:
            print(f"Warning: Failed to snapshot residual predictor for {block_type}:{block_id}: {exc}")

    return total_loss, processed


def _train_modulation_predictors(
    samples: List[Dict[str, Any]],
    backbone,
    lr: float,
    batch_size: int,
    device,
) -> Tuple[float, int]:
    if not samples:
        return 0.0, 0
    import random

    target_batch = max(1, min(len(samples), int(batch_size)))
    chosen = random.sample(samples, k=target_batch)
    total_loss = 0.0
    processed = 0
    updated_blocks: set[Tuple[str, int]] = set()

    with torch.enable_grad():
        for sample in chosen:
            temb = sample.get("temb")
            if temb is None:
                continue
            block_type = sample["block_type"]
            block_id = sample["block_id"]
            trainer = _ensure_modulation_trainer(backbone, block_type, block_id, temb.shape[-1], lr, device)
            temb_dev = _move_sample_to_device(temb, trainer.device)
            try:
                loss = trainer.train_step(temb_dev)
            except Exception as exc:
                print(f"Warning: Modulation predictor update failed for {block_type}:{block_id}: {exc}")
                continue
            total_loss += loss
            processed += 1
            MODULATION_READY_KEYS.add(_relationship_block_key(block_type, block_id))
            updated_blocks.add((block_type, block_id))

    for block_type, block_id in updated_blocks:
        key = _relationship_block_key(block_type, block_id)
        try:
            MODULATION_TRAINER_STATES[key] = MODULATION_TRAINERS[(block_type, block_id)].export_state()
        except Exception as exc:
            print(f"Warning: Failed to snapshot modulation predictor for {block_type}:{block_id}: {exc}")

    return total_loss, processed


def _persist_runtime_caches_on_exit():
    if not RUNTIME_CACHE_ENABLED:
        return
    core = globals().get("TRANSFORMER_BACKBONE") or globals().get("transformer_core")
    if core is None:
        return
    if CURRENT_FBCACHE_ENABLED:
        fb_cache = getattr(core, "first_block_cache", None)
        if fb_cache is not None:
            _save_runtime_cache_state("first_block_cache", fb_cache.state_dict())
    if CURRENT_SIM_CACHE_ENABLED:
        sim_manager = getattr(core, "similarity_cache_manager", None)
        if sim_manager is not None:
            _save_runtime_cache_state("similarity_cache", sim_manager.state_dict())
    if CURRENT_KV_CACHE_ENABLED:
        kv_manager = getattr(core, "kv_cache_manager", None)
        if kv_manager is not None:
            _save_runtime_cache_state("kv_cache", kv_manager.state_dict())
    
    # Persist the relationship trainer if it was used during the run
    if RELATIONSHIP_TRAINER_ENABLED_FOR_RUN and relationship_trainer is not None:
        print("Persisting relationship trainer state to disk...")
        _save_runtime_cache_state("relationship_trainer", relationship_trainer.state_dict())

    if RESIDUAL_TRAINER_STATES or RESIDUAL_TRAINERS:
        merged_residual = _export_relationship_trainer_states(RESIDUAL_TRAINERS, RESIDUAL_TRAINER_STATES)
        if merged_residual:
            _save_runtime_cache_state(RELATIONSHIP_RESIDUAL_CACHE_NAME, merged_residual)

    if MODULATION_TRAINER_STATES or MODULATION_TRAINERS:
        merged_modulation = _export_relationship_trainer_states(MODULATION_TRAINERS, MODULATION_TRAINER_STATES)
        if merged_modulation:
            _save_runtime_cache_state(RELATIONSHIP_MODULATION_CACHE_NAME, merged_modulation)


atexit.register(_persist_runtime_caches_on_exit)

if ENABLE_QUANT:
    manual_quant_targets = [vae, image_encoder]
    if not USE_BITSANDBYTES:
        manual_quant_targets.extend([text_encoder, text_encoder_2])
    if not optimized_transformer_loaded and not FP8_TRANSFORMER_ACTIVE:
        manual_quant_targets.append(transformer_core)
    elif FP8_TRANSFORMER_ACTIVE:
        print('FP8 optimization active -> skipping additional int-N quantization for transformer.')

    for module in manual_quant_targets:
        apply_int_nbit_quantization(module, num_bits=QUANT_BITS, target_dtype=MODEL_COMPUTE_DTYPE)
        if ENFORCE_LOW_PRECISION:
            enforce_low_precision(module, activation_dtype=MODEL_COMPUTE_DTYPE)
    if not ENFORCE_LOW_PRECISION:
        print("Low precision enforcement disabled; quantized modules will keep their existing dtype.")
else:
    if _enable_quant_env is None:
        print('FRAMEPACK_ENABLE_QUANT not set -> skipping module quantization.')
    else:
        print('FRAMEPACK_ENABLE_QUANT=0 -> skipping module quantization.')
    if ENFORCE_LOW_PRECISION:
        print('DEBUG: About to enforce low precision on modules...')
        enforce_targets = [vae, image_encoder]
        if not FP8_TRANSFORMER_ACTIVE:
            enforce_targets.append(transformer_core)
        print(f'DEBUG: enforce_targets has {len(enforce_targets)} modules')
        for i, module in enumerate(enforce_targets):
            print(f'DEBUG: Enforcing low precision on module {i+1}/{len(enforce_targets)}...')
            enforce_low_precision(module, activation_dtype=MODEL_COMPUTE_DTYPE)
            print(f'DEBUG: Completed module {i+1}/{len(enforce_targets)}')
            # Force memory cleanup after each module
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print('DEBUG: enforce_targets complete')
        if not USE_BITSANDBYTES:
            print('DEBUG: Processing text encoders (skipping problematic dtype conversion)...')
            # NOTE: Text encoders are already loaded with torch_dtype=torch.float16 (line 1337)
            # Calling .to(dtype=...) on these large models causes segfaults/crashes
            # Skip the conversion since they're already in the correct dtype
            print(f'DEBUG: text_encoder type: {type(text_encoder).__name__}')
            print(f'DEBUG: text_encoder_2 type: {type(text_encoder_2).__name__}')
            print('DEBUG: Text encoders already in correct dtype (float16), skipping conversion')
            print('DEBUG: Text encoders complete')
    else:
        print('Low precision enforcement disabled; models retain their loaded precision.')

print('DEBUG: Reached compilation check point')
if torch.cuda.is_available() and not FAST_START and ENABLE_COMPILE:
    print('DEBUG: Starting module compilation...')
    compile_mode = "max-autotune-no-cudagraphs"
    if not USE_BITSANDBYTES:
        print('DEBUG: Compiling text_encoder...')
        text_encoder = maybe_compile_module(text_encoder, mode=compile_mode, dynamic=False)
        print('DEBUG: Compiling text_encoder_2...')
        text_encoder_2 = maybe_compile_module(text_encoder_2, mode=compile_mode, dynamic=False)
        print('DEBUG: Text encoders compiled')
    image_encoder = maybe_compile_module(image_encoder, mode=compile_mode, dynamic=False)
elif not ENABLE_COMPILE:
    print('FRAMEPACK_ENABLE_COMPILE=0 -> skipping torch.compile for encoders.')
elif torch.cuda.is_available() and FAST_START:
    print('Fast-start mode -> skipping torch.compile for encoders.')

if INFERENCE_CONFIG.torchscript.mode == "off":
    if ENABLE_COMPILE and not FAST_START:
        transformer_core = maybe_compile_module(transformer_core, mode=compile_mode, dynamic=True)
    elif not ENABLE_COMPILE:
        print('FRAMEPACK_ENABLE_COMPILE=0 -> skipping torch.compile for transformer.')
    elif FAST_START:
        print('Fast-start mode -> skipping torch.compile for transformer.')
else:
    print(f'Skipping torch.compile for transformer (TorchScript mode: {INFERENCE_CONFIG.torchscript.mode}).')

llama_semantic_embed = make_text_semantic_embedder(tokenizer)
clip_semantic_embed = make_text_semantic_embedder(tokenizer_2)
image_semantic_embed = make_image_semantic_embedder()

# Disable caching for text_encoder to avoid issues with output_hidden_states
# The cache was not properly handling the output_hidden_states parameter
""" text_encoder = wrap_with_module_cache(
     text_encoder,
     cache_name='text_encoder_llama',
     normalization_tag='llama_prompt',
     batched_arg_names=['input_ids', 'attention_mask'],
     hash_input_names=['input_ids', 'attention_mask', 'output_hidden_states'],
     default_cache_size=DEFAULT_CACHE_ITEMS,
     cache_mode=CACHE_MODE,
     semantic_embed_fn=llama_semantic_embed,
     semantic_threshold=SEMANTIC_CACHE_THRESHOLD) """
text_encoder_2 = wrap_with_module_cache(
    text_encoder_2,
    cache_name='text_encoder_clip',
    normalization_tag='clip_prompt',
    batched_arg_names=['input_ids'],
    hash_input_names=['input_ids'],
    default_cache_size=max(1, DEFAULT_CACHE_ITEMS // 2),
    cache_mode=CACHE_MODE,
    semantic_embed_fn=clip_semantic_embed,
    semantic_threshold=SEMANTIC_CACHE_THRESHOLD,
)
image_encoder = wrap_with_module_cache(
    image_encoder,
    cache_name='siglip_image_encoder',
    normalization_tag='siglip_pixel_values',
    batched_arg_names=['pixel_values'],
    hash_input_names=['pixel_values'],
    default_cache_size=DEFAULT_CACHE_ITEMS,
    cache_mode=CACHE_MODE,
    semantic_embed_fn=image_semantic_embed,
    semantic_threshold=SEMANTIC_CACHE_THRESHOLD,
)

if ENABLE_PRUNE and not optimized_transformer_loaded:
    prune_transformer_layers(
        transformer_core,
        dual_keep_ratio=float(os.environ.get("FRAMEPACK_PRUNE_DUAL_RATIO", 0.5)),
        single_keep_ratio=float(os.environ.get("FRAMEPACK_PRUNE_SINGLE_RATIO", 0.5)),
    )
elif not ENABLE_PRUNE:
    print('FRAMEPACK_ENABLE_PRUNE=0 -> skipping transformer pruning.')
else:
    print('Transformer already optimized; skipping additional pruning.')

transformer_core.enable_gradient_checkpointing()
transformer_core.high_quality_fp32_output_for_inference = False
print('transformer.high_quality_fp32_output_for_inference = False')

if ENABLE_OPT_CACHE and not optimized_transformer_loaded and OPTIMIZED_MODEL_PATH:
    torch.save(transformer_core, OPTIMIZED_MODEL_PATH)
    print(f'Saved optimized transformer weights to {OPTIMIZED_MODEL_PATH}')

torchscript_prep_required = INFERENCE_CONFIG.torchscript.mode != "off" or INFERENCE_CONFIG.torchscript.load_path
if torchscript_prep_required:
    try:
        transformer_device = next(transformer_core.parameters()).device
    except StopIteration:
        transformer_device = torch.device('cpu')
    example_builder = partial(
        _build_transformer_example,
        dtype=MODEL_COMPUTE_DTYPE,
        device=transformer_device,
    )
    artifact_target = jit_artifact if INFERENCE_CONFIG.torchscript.mode != "off" else None
    transformer_core = prepare_module_for_inference(
        transformer_core,
        config=INFERENCE_CONFIG,
        artifact_path=artifact_target,
        example_builder=example_builder,
    )
elif FAST_START:
    print('Fast-start mode -> skipping TorchScript preparation step.')

torchscript_active = (
    INFERENCE_CONFIG.torchscript.mode != "off"
    or isinstance(transformer_core, torch.jit.ScriptModule)
    or hasattr(transformer_core, "_compiled")
)

if torchscript_active and USE_FSDP:
    print('TorchScript module detected; skipping FSDP wrapping for transformer.')
    transformer = transformer_core
else:
    transformer = maybe_wrap_with_fsdp(transformer_core, compute_dtype=MODEL_COMPUTE_DTYPE)

TRANSFORMER_BACKBONE = transformer_core

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    if not USE_FSDP:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
    if not USE_BITSANDBYTES:
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    if not USE_BITSANDBYTES:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    if not USE_FSDP:
        transformer.to(gpu)
    transformer.requires_grad_(False)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


def _estimate_decoded_frame_mb(latents):
    """Rough upper bound of decoded frame memory footprint in MB."""
    _, _, _, latent_h, latent_w = latents.shape
    decoded_h = max(1, latent_h * VAE_UPSCALE_FACTOR)
    decoded_w = max(1, latent_w * VAE_UPSCALE_FACTOR)
    bytes_per_pixel = 4.0  # assume fp32 activations during VAE decode
    per_frame_mb = (decoded_h * decoded_w * 3 * bytes_per_pixel) / (1024 ** 2)
    return max(per_frame_mb, 0.5)


def _auto_select_vae_chunk_size(latents, quality_mode=False, requested_chunk=None):
    total_frames = latents.shape[2]
    if total_frames <= 1:
        return 1

    manual_chunk = requested_chunk
    if manual_chunk is None and VAE_CHUNK_OVERRIDE > 0:
        manual_chunk = VAE_CHUNK_OVERRIDE
    if manual_chunk is not None and manual_chunk > 0:
        return max(1, min(total_frames, int(manual_chunk)))

    chunk_size = 1
    if torch.cuda.is_available():
        free_gb = get_cuda_free_memory_gb(gpu)
        usable_gb = max(0.0, free_gb - VAE_CHUNK_RESERVE_GB)
        if usable_gb > 0:
            frame_mb = _estimate_decoded_frame_mb(latents)
            chunk_size = max(1, int((usable_gb * 1024) / (frame_mb * VAE_CHUNK_SAFETY)))
    else:
        # CPU decode path can usually afford a few frames at once
        chunk_size = min(total_frames, 8)

    if high_vram:
        chunk_size = max(chunk_size, min(total_frames, 12))

    if quality_mode:
        chunk_size = max(chunk_size, min(total_frames, 4))

    return max(1, min(total_frames, chunk_size))


def vae_decode_chunked(latents, vae, chunk_size=None, quality_mode=False, decoder_fn=None):
    """Decode latents in dynamically sized chunks to avoid OOM penalties.

    Args:
        latents: Latent tensors to decode
        vae: VAE model
        chunk_size: Optional manual chunk override. If None an automatic heuristic is used.
        quality_mode: If True, uses at least 2-4 frame chunks for better temporal consistency.
        decoder_fn: Optional callable(latents) returning decoded pixels (used for TensorRT).
    """
    b, c, t, h, w = latents.shape

    chunk_size = _auto_select_vae_chunk_size(latents, quality_mode=quality_mode, requested_chunk=chunk_size)

    def _decode_with_chunk_size(active_chunk_size: int):
        if t <= active_chunk_size:
            if decoder_fn is None:
                with inference_autocast():
                    decoded = vae_decode(latents, vae)
            else:
                try:
                    decoded = decoder_fn(latents)
                except Exception as e:
                    print(f"TensorRT decode failed, falling back to standard VAE: {e}")
                    with inference_autocast():
                        decoded = vae_decode(latents, vae)
            return decoded.cpu()

        chunks = []
        for i in range(0, t, active_chunk_size):
            end_idx = min(i + active_chunk_size, t)
            chunk_latents = latents[:, :, i:end_idx, :, :]

            if decoder_fn is None:
                with inference_autocast():
                    chunk_pixels = vae_decode(chunk_latents, vae)
            else:
                try:
                    chunk_pixels = decoder_fn(chunk_latents)
                except Exception as e:
                    print(f"TensorRT decode failed for chunk {i}, falling back to standard VAE: {e}")
                    with inference_autocast():
                        chunk_pixels = vae_decode(chunk_latents, vae)

            # Move to CPU immediately and clear GPU memory
            chunks.append(chunk_pixels.cpu())
            del chunk_pixels, chunk_latents

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if not quality_mode:
                    torch.cuda.synchronize()

        result = torch.cat(chunks, dim=2)

        del chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    attempt_chunk = chunk_size
    last_error = None

    while attempt_chunk >= 1:
        try:
            return _decode_with_chunk_size(attempt_chunk)
        except torch.cuda.OutOfMemoryError as oom:
            last_error = oom
            if attempt_chunk == 1:
                raise

            next_chunk = max(1, attempt_chunk // 2)
            if next_chunk == attempt_chunk:
                raise

            warn_msg = f'VAE decode OOM at chunk_size={attempt_chunk}; retrying with chunk_size={next_chunk}'
            if quality_mode and next_chunk < 4:
                warn_msg += ' (quality fallback)'
            print(warn_msg)

            attempt_chunk = next_chunk

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    if last_error is not None:
        raise last_error

    return _decode_with_chunk_size(1)


@torch.inference_mode()
def worker(
    input_image,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    use_fb_cache,
    use_sim_cache,
    use_kv_cache,
    slow_prompt_hint,
    cache_mode,
    mp4_crf,
    quality_mode=False,
    use_tensorrt_decode=False,
    use_tensorrt_transformer=False,
    relationship_trainer_mode="off",
    rt_learning_rate=1e-4,
    rt_batch_size=256,
):
    # Initialize profiling if enabled
    PROFILING_ENABLED = args.enable_profiling
    memory_tracker = None
    iteration_profiler = None

    if PROFILING_ENABLED:
        try:
            print("\n" + "="*80)
            print("PROFILING ENABLED - Performance data will be collected")
            print("="*80)
            # Note: PyTorch profiler disabled - causes crashes with complex workflows
            # Using lightweight timing stats and memory tracking instead
            # This provides comprehensive performance data without crashing
            memory_tracker = MemoryTracker(enabled=True)
            iteration_profiler = IterationProfiler(name="diffusion_step", enabled=True)
            memory_tracker.snapshot("worker_start")
            print("Profiling initialized successfully:")
            print("  - Timing statistics: Enabled (via profile_section)")
            print("  - Memory tracking: Enabled (GPU VRAM usage)")
            print("  - Iteration profiling: Enabled (per-step timing)")
            print("="*80)
        except Exception as e:
            print(f"Warning: Failed to initialize profiling: {e}")
            print("Continuing without profiling...")
            PROFILING_ENABLED = False

    runtime_cache_mode = (cache_mode or CACHE_MODE).lower()
    set_cache_mode_for_wrappers(runtime_cache_mode)
    latent_window_size = align_to_multiple(latent_window_size, multiple=8, minimum=8)
    controller = AdaptiveLatentWindowController(latent_window_size, get_cuda_free_memory_gb(gpu))
    latent_window_size = align_to_multiple(controller.window_size, multiple=8, minimum=8)
    controller.window_size = latent_window_size
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    decoder_impl = None
    encoder_impl = None
    transformer_impl = transformer

    if use_tensorrt_decode and TENSORRT_AVAILABLE:
        if TENSORRT_ENCODER is not None:
            encoder_impl = TENSORRT_ENCODER
        if TENSORRT_DECODER is not None:
            decoder_impl = TENSORRT_DECODER.decode
        if encoder_impl is not None or decoder_impl is not None:
            print("TensorRT VAE acceleration engaged for this job.")
        if decoder_impl is not None:
            print("Note: TensorRT will compile separate engines for each unique chunk shape, which may cause slowdowns on first use.")

    if use_tensorrt_transformer and TENSORRT_TRANSFORMER is not None:
        transformer_impl = TENSORRT_TRANSFORMER
        print("\n" + "="*60)
        print("TensorRT transformer acceleration ENGAGED for this job")
        print("="*60)
        print("First inference will trigger TensorRT compilation:")
        print("  - This may take 5-15 minutes on first run")
        print("  - Subsequent runs with same shape will be fast")
        print("  - Watch for 'Compiling TensorRT transformer engine' message")
        print("="*60 + "\n")
    elif use_tensorrt_transformer and TENSORRT_TRANSFORMER is None:
        print("\nWARNING: TensorRT transformer requested but not available")
        print("Check initialization warnings above for details\n")

    transformer_backbone = getattr(transformer_impl, "module", None)
    if transformer_backbone is None:
        transformer_backbone = getattr(transformer_impl, "transformer", None)
    if transformer_backbone is None:
        transformer_backbone = globals().get("TRANSFORMER_BACKBONE", transformer_impl)

    # Setup for experimental relationship trainer
    normalized_trainer_mode = (relationship_trainer_mode or "off").lower()
    _configure_relationship_block_overrides(transformer_backbone, normalized_trainer_mode, gpu, rt_learning_rate)
    global RELATIONSHIP_TRAINER_ENABLED_FOR_RUN, ACTIVE_RELATIONSHIP_MODE
    ACTIVE_RELATIONSHIP_MODE = normalized_trainer_mode
    RELATIONSHIP_TRAINER_ENABLED_FOR_RUN = normalized_trainer_mode != "off"
    trainer_batch_size = max(1, int(rt_batch_size))
    block_io_data: List[Dict[str, Any]] = []

    if normalized_trainer_mode == "hidden_state":
        if relationship_trainer is None:
            RELATIONSHIP_TRAINER_ENABLED_FOR_RUN = False
            print("Hidden-state relationship trainer unavailable; disabling mode.")
        else:
            print(f"Enabling Hidden State Trainer (lr={rt_learning_rate:.2e}, batch={trainer_batch_size}).")
            relationship_trainer.to(gpu)
            relationship_trainer.optimizer = torch.optim.Adam(
                relationship_trainer.model.parameters(),
                lr=rt_learning_rate,
            )
    elif normalized_trainer_mode in {"residual", "modulation"}:
        print(
            f"Relationship trainer mode '{normalized_trainer_mode}' enabled "
            f"(lr={rt_learning_rate:.2e}, batch={trainer_batch_size}). Capturing block tuples."
        )
    elif normalized_trainer_mode != "off":
        RELATIONSHIP_TRAINER_ENABLED_FOR_RUN = False
        print(f"Unknown relationship trainer mode '{normalized_trainer_mode}'. Disabling capture.")

    if RELATIONSHIP_TRAINER_ENABLED_FOR_RUN:
        def block_io_callback(
            block_id,
            block_type,
            input_h,
            output_h,
            **kwargs,
        ):
            sample = {
                "block_type": block_type,
                "block_id": block_id,
                "h_in": _clone_sample_to_cpu(input_h),
                "h_out": _clone_sample_to_cpu(output_h),
                "encoder": _clone_sample_to_cpu(kwargs.get("encoder_input")),
                "temb": _clone_sample_to_cpu(kwargs.get("temb")),
                "attention_mask": _clone_sample_to_cpu(kwargs.get("attention_mask")),
                "rope_freqs": _clone_sample_to_cpu(kwargs.get("rope_freqs")),
            }
            block_io_data.append(sample)
            if len(block_io_data) > RELATIONSHIP_SAMPLE_LIMIT:
                block_io_data.pop(0)

        transformer_backbone.block_io_callback = block_io_callback
    elif hasattr(transformer_backbone, "block_io_callback"):
        transformer_backbone.block_io_callback = None

    cache_event_recorder = CacheEventRecorder()
    cache_event_recorder.reset()
    if hasattr(transformer_backbone, "set_cache_event_recorder"):
        transformer_backbone.set_cache_event_recorder(cache_event_recorder)

    desired_fb_cache_state = bool(use_fb_cache)
    global CURRENT_FBCACHE_ENABLED, CURRENT_SIM_CACHE_ENABLED, CURRENT_KV_CACHE_ENABLED
    if hasattr(transformer_backbone, "enable_first_block_cache") and desired_fb_cache_state != CURRENT_FBCACHE_ENABLED:
        transformer_backbone.enable_first_block_cache(
            enabled=desired_fb_cache_state,
            threshold=FBCACHE_THRESHOLD,
            verbose=FBCACHE_VERBOSE,
        )
        if desired_fb_cache_state:
            fb_state = _load_runtime_cache_state("first_block_cache")
            if fb_state and hasattr(transformer_backbone, "first_block_cache"):
                transformer_backbone.first_block_cache.load_state_dict(fb_state)
            print("First block cache enabled for this job.")
        else:
            print("First block cache disabled for this job.")
        CURRENT_FBCACHE_ENABLED = desired_fb_cache_state

    desired_sim_cache_state = bool(use_sim_cache)
    if hasattr(transformer_backbone, "enable_similarity_cache") and desired_sim_cache_state != CURRENT_SIM_CACHE_ENABLED:
        if desired_sim_cache_state:
            transformer_backbone.enable_similarity_cache(
                enabled=True,
                threshold=SIM_CACHE_THRESHOLD,
                max_skip=SIM_CACHE_MAX_SKIP,
                max_entries=SIM_CACHE_MAX_ENTRIES,
                use_faiss=SIM_CACHE_USE_FAISS,
                verbose=SIM_CACHE_VERBOSE,
            )
            sim_state = _load_runtime_cache_state("similarity_cache")
            if sim_state and getattr(transformer_backbone, "similarity_cache_manager", None) is not None:
                transformer_backbone.similarity_cache_manager.load_state_dict(sim_state)
            print("Similarity cache enabled for this job.")
        else:
            transformer_backbone.enable_similarity_cache(enabled=False)
            print("Similarity cache disabled for this job.")
        CURRENT_SIM_CACHE_ENABLED = desired_sim_cache_state

    desired_kv_cache_state = bool(use_kv_cache)
    if hasattr(transformer_backbone, "enable_kv_cache") and desired_kv_cache_state != CURRENT_KV_CACHE_ENABLED:
        if desired_kv_cache_state:
            transformer_backbone.enable_kv_cache(
                enabled=True,
                max_length=KV_CACHE_LENGTH,
                verbose=KV_CACHE_VERBOSE,
            )
            kv_state = _load_runtime_cache_state("kv_cache")
            if kv_state and getattr(transformer_backbone, "kv_cache_manager", None) is not None:
                transformer_backbone.kv_cache_manager.load_state_dict(kv_state)
            print("KV cache enabled for this job.")
        else:
            transformer_backbone.enable_kv_cache(enabled=False)
            print("KV cache disabled for this job.")
        CURRENT_KV_CACHE_ENABLED = desired_kv_cache_state

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            modules_to_unload = [image_encoder, vae, transformer]
            if not USE_BITSANDBYTES:
                modules_to_unload.extend([text_encoder, text_encoder_2])
            unload_complete_models(*modules_to_unload)

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram and not USE_BITSANDBYTES:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        active_prompt = prompt
        if slow_prompt_hint:
            active_prompt = _apply_prompt_hint(active_prompt, SLOW_MOTION_HINT)
            print(f'Applied slow-motion hint to prompt: "{SLOW_MOTION_HINT}".')

        with profile_section("text_encoding_positive", enabled=PROFILING_ENABLED):
            with inference_autocast():
                llama_vec, clip_l_pooler = encode_prompt_conds(
                    active_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                    trt_llama_encoder=TENSORRT_LLAMA_TEXT_ENCODER,
                    trt_clip_encoder=TENSORRT_CLIP_TEXT_ENCODER
                )

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            with profile_section("text_encoding_negative", enabled=PROFILING_ENABLED):
                with inference_autocast():
                    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                        n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                        trt_llama_encoder=TENSORRT_LLAMA_TEXT_ENCODER,
                        trt_clip_encoder=TENSORRT_CLIP_TEXT_ENCODER
                    )

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec, _ = align_tensor_dim_to_multiple(llama_vec, dim=1, multiple=TENSOR_CORE_MULTIPLE)
        llama_attention_mask, _ = align_tensor_dim_to_multiple(
            llama_attention_mask, dim=1, multiple=TENSOR_CORE_MULTIPLE, pad_value=False
        )
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        llama_vec_n, _ = align_tensor_dim_to_multiple(llama_vec_n, dim=1, multiple=TENSOR_CORE_MULTIPLE)
        llama_attention_mask_n, _ = align_tensor_dim_to_multiple(
            llama_attention_mask_n, dim=1, multiple=TENSOR_CORE_MULTIPLE, pad_value=False
        )

        llama_vec = llama_vec.to(dtype=MODEL_COMPUTE_DTYPE)
        llama_vec_n = llama_vec_n.to(dtype=MODEL_COMPUTE_DTYPE)
        clip_l_pooler = clip_l_pooler.to(dtype=MODEL_COMPUTE_DTYPE)
        clip_l_pooler_n = clip_l_pooler_n.to(dtype=MODEL_COMPUTE_DTYPE)
        clip_l_pooler, _ = align_tensor_dim_to_multiple(clip_l_pooler, dim=-1, multiple=TENSOR_CORE_MULTIPLE)
        clip_l_pooler_n, _ = align_tensor_dim_to_multiple(clip_l_pooler_n, dim=-1, multiple=TENSOR_CORE_MULTIPLE)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        height, width = align_resolution(height, width, multiple=64)
        accelerated = optimized_resize_and_center_crop(input_image, target_width=width, target_height=height)
        if accelerated is None:
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
        else:
            input_image_np = accelerated

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        normalized_np = normalize_uint8_image(input_image_np)
        if normalized_np is not None:
            input_image_pt = torch.from_numpy(normalized_np).to(dtype=vae.dtype)
        else:
            input_image_pt = torch.from_numpy(input_image_np).to(dtype=vae.dtype)
            input_image_pt.div_(127.5).sub_(1)
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        with profile_section("vae_encode", enabled=PROFILING_ENABLED):
            with inference_autocast():
                if encoder_impl is not None:
                    start_latent = encoder_impl.encode(input_image_pt)
                else:
                    start_latent = vae_encode(input_image_pt, vae)
            start_latent = start_latent.to(dtype=MODEL_COMPUTE_DTYPE)

        if PROFILING_ENABLED and memory_tracker:
            memory_tracker.snapshot("after_vae_encode")

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        with profile_section("image_encoder", enabled=PROFILING_ENABLED):
            with inference_autocast():
                image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(dtype=MODEL_COMPUTE_DTYPE)

        if PROFILING_ENABLED and memory_tracker:
            memory_tracker.snapshot("after_image_encode")

        # Dtype

        llama_vec = llama_vec.to(dtype=MODEL_COMPUTE_DTYPE)
        llama_vec_n = llama_vec_n.to(dtype=MODEL_COMPUTE_DTYPE)
        clip_l_pooler = clip_l_pooler.to(dtype=MODEL_COMPUTE_DTYPE)
        clip_l_pooler_n = clip_l_pooler_n.to(dtype=MODEL_COMPUTE_DTYPE)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(dtype=MODEL_COMPUTE_DTYPE)
        image_encoder_last_hidden_state, _ = align_tensor_dim_to_multiple(
            image_encoder_last_hidden_state, dim=-1, multiple=TENSOR_CORE_MULTIPLE
        )

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
            dtype=MODEL_COMPUTE_DTYPE,
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        chunk_index = 0
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            start_video_frames = max(0, total_generated_latent_frames * 4 - 3)
            cache_event_recorder.start_chunk(
                start_frame=start_video_frames,
                steps=steps,
                label=f"chunk_{chunk_index}",
            )

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram and not USE_FSDP:
                unload_complete_models()
                # For TensorRT transformer, move the underlying transformer
                transformer_to_move = transformer_backbone if use_tensorrt_transformer else transformer
                move_model_to_device_with_memory_preservation(transformer_to_move, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer_backbone.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=0.05)
            else:
                transformer_backbone.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                with inference_autocast():
                    preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                cache_event_recorder.mark_step(current_step)
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            with profile_section(f"sampling_chunk_{chunk_index}", enabled=PROFILING_ENABLED):
                # Note: PyTorch profiler disabled - causes crashes
                # Only using lightweight timing stats instead

                with inference_autocast():
                    generated_latents = sample_hunyuan(
                        transformer=transformer_impl,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        # shift=3.0,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=MODEL_COMPUTE_DTYPE,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )

            if PROFILING_ENABLED and memory_tracker:
                memory_tracker.snapshot(f"after_sampling_chunk_{chunk_index}")

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
            end_video_frames = max(0, total_generated_latent_frames * 4 - 3)
            cache_event_recorder.finalize_chunk(end_frame=end_video_frames)
            stream.output_queue.push(('timeline', cache_event_recorder.to_markdown()))
            chunk_index += 1

            if not high_vram and not USE_FSDP:
                # Aggressive offloading with synchronization
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8, aggressive=True)

                # Force free VRAM before loading VAE
                force_free_vram(target_gb=10.0)

                # Use chunked loading for VAE if needed
                try:
                    load_model_as_complete(vae, target_device=gpu)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("OOM loading VAE, trying chunked loading...")
                        load_model_chunked(vae, target_device=gpu, max_chunk_size_mb=256)
                    else:
                        raise

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # Force clear cache before VAE decode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if history_pixels is None:
                with profile_section(f"vae_decode_full_chunk_{chunk_index}", enabled=PROFILING_ENABLED):
                    history_pixels = vae_decode_chunked(
                        real_history_latents,
                        vae,
                        chunk_size=None,
                        quality_mode=quality_mode,
                        decoder_fn=decoder_impl,
                    )
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                section_latent_frames = min(section_latent_frames, real_history_latents.shape[2])
                overlapped_frames = latent_window_size * 4 - 3

                with profile_section(f"vae_decode_section_chunk_{chunk_index}", enabled=PROFILING_ENABLED):
                    current_pixels = vae_decode_chunked(
                        real_history_latents[:, :, :section_latent_frames],
                        vae,
                        chunk_size=None,
                    quality_mode=quality_mode,
                    decoder_fn=decoder_impl,
                )
                overlap = min(overlapped_frames, current_pixels.shape[2], history_pixels.shape[2])
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlap)

            if not high_vram:
                unload_complete_models()
                force_free_vram(target_gb=2.0)

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))
            # Persist caches after each video chunk is created.
            _persist_runtime_caches_on_exit()

            if is_last_section:
                break

        # After generation, run the training step if enabled
        if RELATIONSHIP_TRAINER_ENABLED_FOR_RUN and block_io_data:
            with torch.inference_mode(False):
                print(f"\nTraining {normalized_trainer_mode} relationship trainer on {len(block_io_data)} tuples...")
                if normalized_trainer_mode == "hidden_state" and relationship_trainer is not None:
                    total_loss, processed = _train_hidden_state_samples(block_io_data, relationship_trainer, trainer_batch_size)
                    if processed > 0:
                        avg_loss = total_loss / processed
                        print(f"Hidden-state trainer updated on {processed} samples; avg loss={avg_loss:.6f}")
                        print("Persisting relationship trainer state to disk...")
                        _save_runtime_cache_state("relationship_trainer", relationship_trainer.state_dict())
                    else:
                        print("Hidden-state trainer skipped (no valid samples).")
                elif normalized_trainer_mode == "residual":
                    total_loss, processed = _train_residual_predictors(
                        block_io_data,
                        transformer_backbone,
                        rt_learning_rate,
                        trainer_batch_size,
                        gpu,
                    )
                    if processed > 0:
                        avg_loss = total_loss / processed
                        print(f"Residual predictors updated on {processed} samples; avg loss={avg_loss:.6f}")
                        residual_payload = _export_relationship_trainer_states(RESIDUAL_TRAINERS, RESIDUAL_TRAINER_STATES)
                        print("Persisting residual predictor cache to disk...")
                        _save_runtime_cache_state(RELATIONSHIP_RESIDUAL_CACHE_NAME, residual_payload)
                    else:
                        print("Residual predictor training skipped (no valid samples).")
                elif normalized_trainer_mode == "modulation":
                    total_loss, processed = _train_modulation_predictors(
                        block_io_data,
                        transformer_backbone,
                        rt_learning_rate,
                        trainer_batch_size,
                        gpu,
                    )
                    if processed > 0:
                        avg_loss = total_loss / processed
                        print(f"Modulation predictors updated on {processed} samples; avg loss={avg_loss:.6f}")
                        modulation_payload = _export_relationship_trainer_states(MODULATION_TRAINERS, MODULATION_TRAINER_STATES)
                        print("Persisting modulation predictor cache to disk...")
                        _save_runtime_cache_state(RELATIONSHIP_MODULATION_CACHE_NAME, modulation_payload)
                    else:
                        print("Modulation predictor training skipped (no valid samples).")
                else:
                    print(f"No trainer registered for mode '{normalized_trainer_mode}'.")

            block_io_data.clear()

    except:
        traceback.print_exc()
        cache_event_recorder.cancel_chunk()

        if not high_vram:
            modules_to_unload = [image_encoder, vae, transformer]
            if not USE_BITSANDBYTES:
                modules_to_unload.extend([text_encoder, text_encoder_2])
            unload_complete_models(*modules_to_unload)

    if hasattr(transformer_backbone, "set_cache_event_recorder"):
        transformer_backbone.set_cache_event_recorder(None)

    # Clean up relationship trainer callback
    if hasattr(transformer_backbone, 'block_io_callback'):
        transformer_backbone.block_io_callback = None
    if hasattr(transformer_backbone, "clear_block_overrides"):
        transformer_backbone.clear_block_overrides()

    # Export profiling results if enabled
    if PROFILING_ENABLED:
        print("\n" + "="*80)
        print("PROFILING COMPLETE - Exporting Results")
        print("="*80)

        # Note: PyTorch profiler disabled (was causing crashes)
        # Using lightweight timing stats and memory tracking only

        # Print timing summary
        try:
            get_global_stats().print_summary(top_n=30)
        except Exception as e:
            print(f"Warning: Failed to print timing summary: {e}")

        # Print memory summary
        if memory_tracker:
            try:
                memory_tracker.print_summary()
            except Exception as e:
                print(f"Warning: Failed to print memory summary: {e}")

        # Print iteration profiler stats
        if iteration_profiler:
            try:
                iteration_profiler.print_summary()
            except Exception as e:
                print(f"Warning: Failed to print iteration summary: {e}")

        # Export comprehensive report
        try:
            report_path = export_profiling_report(
                output_dir=args.profiling_output_dir,
                timing_stats=get_global_stats(),
                memory_tracker=memory_tracker,
                iteration_profiler=iteration_profiler,
            )
            print(f"\nProfiling data saved to: {args.profiling_output_dir}")
            print(f"  - JSON report: {report_path}")
            print("\nThe report includes:")
            print("  - Timing statistics for all operations")
            print("  - Memory usage snapshots")
            print("  - Iteration-by-iteration performance data")
        except Exception as e:
            print(f"Warning: Failed to export profiling report: {e}")

        print("="*80 + "\n")

    stream.output_queue.push(('timeline', cache_event_recorder.to_markdown()))
    stream.output_queue.push(('end', None))
    return


def process(
    input_image,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    use_fb_cache,
    use_sim_cache,
    use_kv_cache,
    slow_prompt_hint,
    cache_mode,
    mp4_crf,
    quality_mode,
    use_tensorrt_decode,
    use_tensorrt_transformer,
    relationship_trainer_mode,
    rt_learning_rate,
    rt_batch_size,
):
    global stream
    assert input_image is not None, 'No input image!'

    timeline_md = "No cache hits recorded yet."
    yield None, None, '', '', timeline_md, gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(
        worker,
        input_image,
        prompt,
        n_prompt,
        seed,
        total_second_length,
        latent_window_size,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        use_fb_cache,
        use_sim_cache,
        use_kv_cache,
        slow_prompt_hint,
        cache_mode,
        mp4_crf,
        quality_mode,
        use_tensorrt_decode,
        use_tensorrt_transformer,
        relationship_trainer_mode,
        rt_learning_rate,
        rt_batch_size,
    )

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), timeline_md, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, timeline_md, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'timeline':
            timeline_md = data or "No cache hits recorded yet."
            yield gr.update(), gr.update(), gr.update(), gr.update(), timeline_md, gr.update(interactive=False), gr.update(interactive=True)
            continue

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', timeline_md, gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


block = build_frontend(
    quick_prompts=quick_prompts,
    enable_fbcache=ENABLE_FBCACHE,
    enable_sim_cache=ENABLE_SIM_CACHE,
    enable_kv_cache=ENABLE_KV_CACHE,
    cache_mode=CACHE_MODE,
    tensorrt_available=TENSORRT_AVAILABLE,
    tensorrt_transformer_available=TENSORRT_TRANSFORMER is not None,
    process_fn=process,
    end_fn=end_process,
)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
    root_path=os.environ.get("GRADIO_ROOT_PATH", ""),  # Fix for RunPod/proxy environments
)
