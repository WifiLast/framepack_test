# By lllyasviel

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


cpu: torch.device = torch.device('cpu')
gpu: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules: List[torch.nn.Module] = []
_MEM_STATS_CACHE: Dict[int, tuple[float, float]] = {}


@dataclass
class MemoryOptimizationConfig:
    """Optional toggles for memory helper optimizations."""

    use_async_streams: bool = False
    use_pinned_memory: bool = False
    cache_memory_stats: bool = False
    stats_cache_ttl: float = 0.05

    def enable_async_copy(self) -> bool:
        return self.use_async_streams and torch.cuda.is_available()


def _device_index(device: Optional[torch.device]) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError("Expected CUDA device for memory helpers.")
        return device.index if device.index is not None else torch.cuda.current_device()
    return int(device)


def _cached_available_bytes(device: torch.device, optim: Optional[MemoryOptimizationConfig]) -> float:
    if optim is None or not optim.cache_memory_stats:
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        return bytes_free_cuda + bytes_inactive_reserved

    now = time.perf_counter()
    idx = _device_index(device)
    ttl = max(0.0, optim.stats_cache_ttl)
    cached = _MEM_STATS_CACHE.get(idx)
    if cached:
        ts, bytes_total = cached
        if (now - ts) <= ttl:
            return bytes_total

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    total = bytes_free_cuda + bytes_inactive_reserved
    _MEM_STATS_CACHE[idx] = (now, total)
    return total


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs: Any) -> None:
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs: Any) -> None:
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module) -> None:
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device) -> None:
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(
    device: Optional[torch.device] = None,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> float:
    target = device or gpu
    available_bytes = _cached_available_bytes(target, optim_config)
    return available_bytes / (1024 ** 3)


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device, optim_config=optim_config)

        if free_mem <= preserved_memory_gb:
            if aggressive:
                # Aggressive mode: clear cache and try to continue
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = get_cuda_free_memory_gb(target_device, optim_config=optim_config)

                if free_mem <= preserved_memory_gb * 0.8:
                    print(f'Stopped at module {i}/{len(modules_list)} due to memory limit')
                    return
            else:
                torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

            # Clear cache every 10 modules
            if aggressive and i % 10 == 0:
                torch.cuda.empty_cache()

    model.to(device=target_device)
    torch.cuda.empty_cache()


async def move_model_to_device_with_memory_preservation_async(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    """Async wrapper that offloads the blocking move to a worker thread."""
    await asyncio.to_thread(
        move_model_to_device_with_memory_preservation,
        model,
        target_device,
        preserved_memory_gb,
        aggressive,
        optim_config,
    )


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device, optim_config=optim_config)

        if free_mem >= preserved_memory_gb:
            if not aggressive:
                torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

            # Clear cache every 10 modules
            if aggressive and i % 10 == 0:
                torch.cuda.empty_cache()

    model.to(device=cpu)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


async def offload_model_from_device_for_memory_preservation_async(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    await asyncio.to_thread(
        offload_model_from_device_for_memory_preservation,
        model,
        target_device,
        preserved_memory_gb,
        aggressive,
        optim_config,
    )


def unload_complete_models(*models: torch.nn.Module) -> None:
    for m in gpu_complete_modules + list(models):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()


async def unload_complete_models_async(*models: torch.nn.Module) -> None:
    await asyncio.to_thread(unload_complete_models, *models)


def load_model_as_complete(model: torch.nn.Module, target_device: torch.device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)


async def load_model_as_complete_async(model: torch.nn.Module, target_device: torch.device, unload: bool = True) -> None:
    await asyncio.to_thread(load_model_as_complete, model, target_device, unload)


def load_model_chunked(
    model: torch.nn.Module,
    target_device: torch.device,
    max_chunk_size_mb: int = 512,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    """
    Load model to device in smaller chunks to bypass memory limits.
    Useful for extremely large models (110GB+) on limited VRAM (16GB).
    """
    optim = optim_config or MemoryOptimizationConfig()

    def _chunked_to_device(tensor: torch.Tensor, dest: torch.device, chunk_bytes: int) -> torch.Tensor:
        if tensor is None:
            return tensor
        if tensor.device == dest:
            return tensor
        if chunk_bytes <= 0:
            # Fallback to standard .to() when chunks disabled
            return tensor.to(device=dest)

        contiguous = tensor.contiguous()
        if optim.use_pinned_memory and not contiguous.is_cuda and torch.cuda.is_available():
            contiguous = contiguous.pin_memory()

        flat_src = contiguous.view(-1)
        total_elems = flat_src.numel()
        if total_elems == 0:
            return tensor.to(device=dest)

        elem_bytes = tensor.element_size()
        if elem_bytes == 0:
            return tensor.to(device=dest)

        chunk_elems = max(1, chunk_bytes // elem_bytes)
        dst = torch.empty_like(tensor, device=dest)
        flat_dst = dst.view(-1)
        copy_stream = torch.cuda.Stream(device=dest) if optim.enable_async_copy() else None

        for start in range(0, total_elems, chunk_elems):
            end = min(total_elems, start + chunk_elems)
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    flat_dst[start:end].copy_(flat_src[start:end], non_blocking=True)
            else:
                flat_dst[start:end].copy_(flat_src[start:end], non_blocking=False)

        if copy_stream is not None:
            torch.cuda.current_stream(device=dest).wait_stream(copy_stream)

        return dst

    max_chunk_size_mb = max(0, int(max_chunk_size_mb))
    chunk_bytes = max_chunk_size_mb * 1024 * 1024

    print(f'Loading {model.__class__.__name__} to {target_device} in chunks (max {max_chunk_size_mb}MB per transfer)')

    unload_complete_models()

    modules = list(model.modules())
    total_modules = len(modules)

    for i, module in enumerate(modules):
        if i % 50 == 0:
            print(f'Loading module {i}/{total_modules}...')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        moved = False

        if chunk_bytes <= 0:
            module.to(device=target_device)
            moved = True
        else:
            # Move parameters in-place using chunked copies
            for name, param in module.named_parameters(recurse=False):
                if param is None:
                    continue
                chunked = _chunked_to_device(param.data, target_device, chunk_bytes)
                param.data = chunked
                moved = True

            # Move buffers in-place
            for name, buf in module.named_buffers(recurse=False):
                if buf is None:
                    continue
                module._buffers[name] = _chunked_to_device(buf, target_device, chunk_bytes)
                moved = True

            if not moved:
                # Modules without params/buffers: fall back to standard move
                module.to(device=target_device)

        # Clear cache after every few modules
        if i % 5 == 0:
            torch.cuda.empty_cache()

    print(f'Finished loading {model.__class__.__name__}')
    gpu_complete_modules.append(model)
    torch.cuda.empty_cache()


async def load_model_chunked_async(
    model: torch.nn.Module,
    target_device: torch.device,
    max_chunk_size_mb: int = 512,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    await asyncio.to_thread(
        load_model_chunked,
        model,
        target_device,
        max_chunk_size_mb,
        optim_config,
    )


def force_free_vram(target_gb: float = 2.0, optim_config: Optional[MemoryOptimizationConfig] = None) -> float:
    """
    Aggressively free VRAM until target_gb is available.
    """
    print(f'Force freeing VRAM to reach {target_gb} GB...')

    # First, unload all complete models
    unload_complete_models()

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_mem = get_cuda_free_memory_gb(gpu, optim_config=optim_config)
    print(f'After clearing: {free_mem:.2f} GB free')

    if free_mem < target_gb:
        print(f'Warning: Only {free_mem:.2f} GB available, requested {target_gb} GB')

    return free_mem


async def force_free_vram_async(target_gb: float = 2.0, optim_config: Optional[MemoryOptimizationConfig] = None) -> float:
    return await asyncio.to_thread(force_free_vram, target_gb, optim_config)
