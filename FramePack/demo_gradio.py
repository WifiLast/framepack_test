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
        return
    path = _runtime_cache_path(name)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        print(f'Warning: failed to save runtime cache "{name}": {exc}')
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
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper import memory as memory_v1
from diffusers_helper import memory_v2
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
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
from diffusers_helper.tensorrt_runtime import TensorRTRuntime, TensorRTLatentDecoder
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
                return dict(bound.arguments)
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
        output = self.inner(**bound_args)
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

DEFAULT_CACHE_MODE = os.environ.get("FRAMEPACK_MODULE_CACHE_MODE", "hash").lower()
SEMANTIC_CACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_SEMANTIC_CACHE_THRESHOLD", "0.985"))
SEMANTIC_CACHE_THRESHOLD = max(0.0, min(1.0, SEMANTIC_CACHE_THRESHOLD))


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
    "--enable-tensorrt",
    action="store_true",
    default=os.environ.get("FRAMEPACK_ENABLE_TENSORRT", "0") == "1",
    help="Enable experimental TensorRT acceleration for VAE decoding (requires torch-tensorrt).",
)
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)
if args.use_memory_v2:
    print("memory_v2 backend enabled (async streams, pinned memory, cached stats).")

FAST_START = args.fast_start or os.environ.get("FRAMEPACK_FAST_START", "0") == "1"
PARALLEL_LOADERS = int(os.environ.get("FRAMEPACK_PARALLEL_LOADERS", "0"))
if PARALLEL_LOADERS <= 1 and FAST_START:
    PARALLEL_LOADERS = 4
PRELOAD_REPOS = os.environ.get("FRAMEPACK_PRELOAD_REPOS", "1") == "1"
FORCE_PARALLEL_LOADERS = os.environ.get("FRAMEPACK_FORCE_PARALLEL_LOADERS", "0") == "1"

CACHE_MODE = args.cache_mode.lower()
VAE_CHUNK_OVERRIDE = max(0, int(os.environ.get("FRAMEPACK_VAE_CHUNK_SIZE", "0")))
VAE_CHUNK_RESERVE_GB = float(os.environ.get("FRAMEPACK_VAE_CHUNK_RESERVE_GB", "4.0"))
VAE_CHUNK_SAFETY = float(os.environ.get("FRAMEPACK_VAE_CHUNK_SAFETY", "1.5"))
VAE_UPSCALE_FACTOR = max(1, int(os.environ.get("FRAMEPACK_VAE_UPSCALE_FACTOR", "8")))
TRT_WORKSPACE_MB = int(os.environ.get("FRAMEPACK_TRT_WORKSPACE_MB", "4096"))
TRT_MAX_AUX_STREAMS = int(os.environ.get("FRAMEPACK_TRT_MAX_AUX_STREAMS", "2"))
ENABLE_TENSORRT_RUNTIME = args.enable_tensorrt

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
ENABLE_FBCACHE = (os.environ.get("FRAMEPACK_ENABLE_FBCACHE", "0") == "1") and not args.disable_fbcache
FBCACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_FBCACHE_THRESHOLD", "0.035"))
FBCACHE_VERBOSE = os.environ.get("FRAMEPACK_FBCACHE_VERBOSE", "0") == "1"
ENABLE_SIM_CACHE = (os.environ.get("FRAMEPACK_ENABLE_SIM_CACHE", "0") == "1") and not args.disable_sim_cache
SIM_CACHE_THRESHOLD = float(os.environ.get("FRAMEPACK_SIM_CACHE_THRESHOLD", "0.9"))
SIM_CACHE_MAX_SKIP = int(os.environ.get("FRAMEPACK_SIM_CACHE_MAX_SKIP", "1"))
SIM_CACHE_MAX_ENTRIES = int(os.environ.get("FRAMEPACK_SIM_CACHE_MAX_ENTRIES", "12"))
SIM_CACHE_USE_FAISS = os.environ.get("FRAMEPACK_SIM_CACHE_USE_FAISS", "0") == "1"
SIM_CACHE_VERBOSE = os.environ.get("FRAMEPACK_SIM_CACHE_VERBOSE", "0") == "1"
ENABLE_KV_CACHE = (os.environ.get("FRAMEPACK_ENABLE_KV_CACHE", "0") == "1") and not args.disable_kv_cache
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
QUANT_BITS = int(os.environ.get("FRAMEPACK_QUANT_BITS", "8"))
USE_BITSANDBYTES = os.environ.get("FRAMEPACK_USE_BNB", "0") == "1"
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
        llm_int8_enable_fp32_cpu_offload=os.environ.get("FRAMEPACK_BNB_CPU_OFFLOAD", "1") == "1",
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
    return LlamaModel.from_pretrained(MODEL_REPO_SOURCE, subfolder='text_encoder', **text_encoder_kwargs)


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
else:
    text_encoder = _load_component('text encoder', _load_text_encoder)
    text_encoder_2 = _load_component('clip text encoder', _load_clip_text)
    vae = _load_component('VAE', _load_vae)
    image_encoder = _load_component('SigLip image encoder', _load_image_encoder)

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

fp8_buffers_present = any(hasattr(module, "scale_weight") for module in transformer_core.modules())
if fp8_buffers_present:
    FP8_TRANSFORMER_ACTIVE = True

TENSORRT_RUNTIME = None
TENSORRT_DECODER = None
TENSORRT_AVAILABLE = False
if ENABLE_TENSORRT_RUNTIME:
    try:
        TENSORRT_RUNTIME = TensorRTRuntime(
            enabled=True,
            precision=MODEL_COMPUTE_DTYPE if MODEL_COMPUTE_DTYPE in (torch.float16, torch.bfloat16) else torch.float16,
            workspace_size_mb=TRT_WORKSPACE_MB,
            max_aux_streams=TRT_MAX_AUX_STREAMS,
        )
        if TENSORRT_RUNTIME.is_ready:
            TENSORRT_DECODER = TensorRTLatentDecoder(vae, TENSORRT_RUNTIME, fallback_fn=vae_decode)
            TENSORRT_AVAILABLE = True
            print(f"TensorRT VAE decoder enabled (workspace={TRT_WORKSPACE_MB} MB).")
        else:
            print(f"TensorRT disabled: {TENSORRT_RUNTIME.failure_reason}")
    except Exception as exc:
        TENSORRT_RUNTIME = None
        TENSORRT_DECODER = None
        print(f"Failed to initialize TensorRT runtime: {exc}")
else:
    print("TensorRT runtime disabled (use --enable-tensorrt to opt-in).")

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
else:
    transformer_core.enable_first_block_cache(enabled=False)

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
else:
    transformer_core.enable_similarity_cache(enabled=False)

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
else:
    transformer_core.enable_kv_cache(enabled=False)


def _persist_runtime_caches_on_exit():
    if not RUNTIME_CACHE_ENABLED:
        return
    core = globals().get("TRANSFORMER_BACKBONE") or globals().get("transformer_core")
    if core is None:
        return
    if ENABLE_FBCACHE:
        fb_cache = getattr(core, "first_block_cache", None)
        if fb_cache is not None:
            _save_runtime_cache_state("first_block_cache", fb_cache.state_dict())
    if ENABLE_SIM_CACHE:
        sim_manager = getattr(core, "similarity_cache_manager", None)
        if sim_manager is not None:
            _save_runtime_cache_state("similarity_cache", sim_manager.state_dict())
    if ENABLE_KV_CACHE:
        kv_manager = getattr(core, "kv_cache_manager", None)
        if kv_manager is not None:
            _save_runtime_cache_state("kv_cache", kv_manager.state_dict())


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
        enforce_low_precision(module, activation_dtype=MODEL_COMPUTE_DTYPE)
else:
    if _enable_quant_env is None:
        print('FRAMEPACK_ENABLE_QUANT not set -> skipping module quantization.')
    else:
        print('FRAMEPACK_ENABLE_QUANT=0 -> skipping module quantization.')
    enforce_targets = [vae, image_encoder]
    if not FP8_TRANSFORMER_ACTIVE:
        enforce_targets.append(transformer_core)
    for module in enforce_targets:
        enforce_low_precision(module, activation_dtype=MODEL_COMPUTE_DTYPE)
    if not USE_BITSANDBYTES:
        enforce_low_precision(text_encoder, activation_dtype=MODEL_COMPUTE_DTYPE)
        enforce_low_precision(text_encoder_2, activation_dtype=MODEL_COMPUTE_DTYPE)

if torch.cuda.is_available() and not FAST_START and ENABLE_COMPILE:
    if not USE_BITSANDBYTES:
        text_encoder = maybe_compile_module(text_encoder, mode="reduce-overhead", dynamic=False)
        text_encoder_2 = maybe_compile_module(text_encoder_2, mode="reduce-overhead", dynamic=False)
    image_encoder = maybe_compile_module(image_encoder, mode="reduce-overhead", dynamic=False)
elif not ENABLE_COMPILE:
    print('FRAMEPACK_ENABLE_COMPILE=0 -> skipping torch.compile for encoders.')
elif torch.cuda.is_available() and FAST_START:
    print('Fast-start mode -> skipping torch.compile for encoders.')

if INFERENCE_CONFIG.torchscript.mode == "off":
    if ENABLE_COMPILE and not FAST_START:
        transformer_core = maybe_compile_module(transformer_core, mode="reduce-overhead", dynamic=True)
    elif not ENABLE_COMPILE:
        print('FRAMEPACK_ENABLE_COMPILE=0 -> skipping torch.compile for transformer.')
    elif FAST_START:
        print('Fast-start mode -> skipping torch.compile for transformer.')
else:
    print(f'Skipping torch.compile for transformer (TorchScript mode: {INFERENCE_CONFIG.torchscript.mode}).')

llama_semantic_embed = make_text_semantic_embedder(tokenizer)
clip_semantic_embed = make_text_semantic_embedder(tokenizer_2)
image_semantic_embed = make_image_semantic_embedder()

text_encoder = wrap_with_module_cache(
    text_encoder,
    cache_name='text_encoder_llama',
    normalization_tag='llama_prompt',
    batched_arg_names=['input_ids', 'attention_mask'],
    hash_input_names=['input_ids', 'attention_mask'],
    default_cache_size=DEFAULT_CACHE_ITEMS,
    cache_mode=CACHE_MODE,
    semantic_embed_fn=llama_semantic_embed,
    semantic_threshold=SEMANTIC_CACHE_THRESHOLD,
)
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
                decoded = decoder_fn(latents)
            return decoded.cpu()

        chunks = []
        for i in range(0, t, active_chunk_size):
            end_idx = min(i + active_chunk_size, t)
            chunk_latents = latents[:, :, i:end_idx, :, :]

            if decoder_fn is None:
                with inference_autocast():
                    chunk_pixels = vae_decode(chunk_latents, vae)
            else:
                chunk_pixels = decoder_fn(chunk_latents)

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
    slow_prompt_hint,
    cache_mode,
    mp4_crf,
    quality_mode=False,
    use_tensorrt_decode=False,
):
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
    if use_tensorrt_decode and TENSORRT_AVAILABLE and TENSORRT_DECODER is not None:
        decoder_impl = TENSORRT_DECODER.decode
        print("TensorRT decoder engaged for this job.")
    transformer_backbone = getattr(transformer, "module", None)
    if transformer_backbone is None:
        transformer_backbone = globals().get("TRANSFORMER_BACKBONE", transformer)

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

        with inference_autocast():
            llama_vec, clip_l_pooler = encode_prompt_conds(active_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            with inference_autocast():
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

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

        with inference_autocast():
            start_latent = vae_encode(input_image_pt, vae).to(dtype=MODEL_COMPUTE_DTYPE)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        with inference_autocast():
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(dtype=MODEL_COMPUTE_DTYPE)

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

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram and not USE_FSDP:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer_backbone.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=0.2)
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
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            with inference_autocast():
                generated_latents = sample_hunyuan(
                    transformer=transformer,
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

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

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
            _persist_runtime_caches_on_exit()

            if is_last_section:
                break
    except:
        traceback.print_exc()

        if not high_vram:
            modules_to_unload = [image_encoder, vae, transformer]
            if not USE_BITSANDBYTES:
                modules_to_unload.extend([text_encoder, text_encoder_2])
            unload_complete_models(*modules_to_unload)

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
    slow_prompt_hint,
    cache_mode,
    mp4_crf,
    quality_mode,
    use_tensorrt_decode,
):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

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
        slow_prompt_hint,
        cache_mode,
        mp4_crf,
        quality_mode,
        use_tensorrt_decode,
    )

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css, analytics_enabled=False).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=420):
            input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used

            with gr.Row(variant='compact'):
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)

            with gr.Row(variant='compact'):
                start_button = gr.Button(value="Start Generation", variant='primary')
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Accordion("Quality & Cache", open=False):
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                quality_mode = gr.Checkbox(label='Quality Mode (Better Hands)', value=False, info='Uses larger VAE chunks (2-4 frames) for better quality, especially for hands. Requires more VRAM.')
                cache_mode_selector = gr.Radio(
                    label='Cache Mode',
                    choices=['hash', 'semantic', 'off'],
                    value=CACHE_MODE,
                    info='hash = deterministic exact reuse, semantic = FAISS-backed approximate hits, off = disable caching.',
                )
                slow_prompt_hint = gr.Checkbox(
                    label='Add "move slowly" hint',
                    value=True,
                    info='Appends a slow-motion phrasing to your prompt to encourage smoother motion.',
                )

            with gr.Accordion("Sampler Controls", open=False):
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

            with gr.Accordion("Performance & Output", open=False):
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")
                tensorrt_decode_checkbox = gr.Checkbox(
                    label="TensorRT VAE Decode (beta)",
                    value=TENSORRT_AVAILABLE,
                    visible=TENSORRT_AVAILABLE,
                    info="Requires torch-tensorrt + CUDA. Falls back automatically if unsupported.",
                )

        with gr.Column(scale=1, min_width=360):
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames (30 FPS)", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('''**Important Notes:**
- Videos are rendered at 30 FPS (standard video framerate)
- The ending actions are generated before starting actions (inverted sampling)
- For better hand quality: Enable "Quality Mode" and disable "TeaCache"
- If video appears too fast, reduce "Total Video Length" to generate more frames per second of content
''')
            with gr.Accordion("Status", open=True):
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [
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
        slow_prompt_hint,
        cache_mode_selector,
        mp4_crf,
        quality_mode,
        tensorrt_decode_checkbox,
    ]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
