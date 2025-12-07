# By lllyasviel

from typing import Any, List, Optional

import torch


cpu: torch.device = torch.device('cpu')
gpu: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules: List[torch.nn.Module] = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs: Any) -> None:
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        # Store kwargs for use in hooks
        module.__dict__['_dynamic_swap_kwargs'] = kwargs

        def hacked_get_attr(self, name: str):
            # Handle device and dtype properties first - these are computed properties on nn.Module
            if name == 'device':
                # Get device from first parameter
                try:
                    return next(self.parameters()).device
                except StopIteration:
                    return torch.device('cpu')
            if name == 'dtype':
                # Get dtype from first parameter
                try:
                    return next(self.parameters()).dtype
                except StopIteration:
                    return torch.float32
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

        # Add forward pre-hook to move weights to target device before forward pass
        def _pre_forward_hook(mod, inputs):
            swap_kwargs = getattr(mod, '_dynamic_swap_kwargs', kwargs)
            device = swap_kwargs.get('device', None)
            if device is None:
                return None

            # Only move parameters of THIS module (not submodules)
            # Each submodule has its own hook, so we don't need to recurse
            for name, param in mod._parameters.items():
                if param is not None and hasattr(param, 'to') and param.device != device:
                    mod._parameters[name] = torch.nn.Parameter(param.to(**swap_kwargs), requires_grad=param.requires_grad)

            # Move buffers of THIS module
            for name, buffer in mod._buffers.items():
                if buffer is not None and hasattr(buffer, 'to') and buffer.device != device:
                    mod._buffers[name] = buffer.to(**swap_kwargs)

            return None

        # Register the hook
        handle = module.register_forward_pre_hook(_pre_forward_hook)
        module.__dict__['_dynamic_swap_hook_handle'] = handle

        # Also need to ensure forward() is called with all kwargs properly
        original_forward = original_class.forward

        def hacked_forward(self, *args, **kwargs):
            # Debug logging for LlamaModel specifically
            import sys
            if 'LlamaModel' in original_class.__name__:
                print(f"DEBUG hacked_forward ({original_class.__name__}): kwargs = {list(kwargs.keys())}", file=sys.stderr)
                print(f"DEBUG hacked_forward: output_hidden_states = {kwargs.get('output_hidden_states', 'NOT SET')}", file=sys.stderr)
                print(f"DEBUG hacked_forward: config.output_hidden_states = {getattr(self.config, 'output_hidden_states', 'NO CONFIG')}", file=sys.stderr)

            result = original_forward(self, *args, **kwargs)

            if 'LlamaModel' in original_class.__name__:
                has_hs = hasattr(result, 'hidden_states')
                hs_value = result.hidden_states if has_hs else 'NO ATTR'
                hs_is_none = hs_value is None if has_hs else 'N/A'
                print(f"DEBUG hacked_forward: result.hidden_states exists={has_hs}, is_none={hs_is_none}", file=sys.stderr)

            return result

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
            'forward': hacked_forward,
        })

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        # Remove forward hook if it exists
        if '_dynamic_swap_hook_handle' in module.__dict__:
            handle = module.__dict__.pop('_dynamic_swap_hook_handle')
            handle.remove()
        # Remove stored kwargs
        if '_dynamic_swap_kwargs' in module.__dict__:
            module.__dict__.pop('_dynamic_swap_kwargs')
        # Restore original class
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs: Any) -> None:
        # Check if this is a LlamaModel that needs hidden_states support
        is_llama = 'LlamaModel' in model.__class__.__name__

        # Install hook on ALL submodules so gradient checkpointing works
        # Each module will move its own parameters before forward
        for m in model.modules():
            # For LlamaModel, skip wrapping the root model itself to avoid breaking hidden_states collection
            # Only wrap the submodules for memory optimization
            if is_llama and m is model:
                continue
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


def get_cuda_free_memory_gb(device: Optional[torch.device] = None) -> float:
    target = device or gpu

    memory_stats = torch.cuda.memory_stats(target)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(target)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
) -> None:
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device)

        if free_mem <= preserved_memory_gb:
            if aggressive:
                # Aggressive mode: clear cache and try to continue
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = get_cuda_free_memory_gb(target_device)

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


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
) -> None:
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device)

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


def unload_complete_models(*models: torch.nn.Module) -> None:
    for m in gpu_complete_modules + list(models):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()


def load_model_as_complete(model: torch.nn.Module, target_device: torch.device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)


def load_model_chunked(
    model: torch.nn.Module,
    target_device: torch.device,
    max_chunk_size_mb: int = 512,
) -> None:
    """
    Load model to device in smaller chunks to bypass memory limits.
    Useful for extremely large models (110GB+) on limited VRAM (16GB).
    """

    def _chunked_to_device(tensor: torch.Tensor, dest: torch.device, chunk_bytes: int) -> torch.Tensor:
        if tensor is None:
            return tensor
        if tensor.device == dest:
            return tensor
        if chunk_bytes <= 0:
            # Fallback to standard .to() when chunks disabled
            return tensor.to(device=dest)

        contiguous = tensor.contiguous()
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

        for start in range(0, total_elems, chunk_elems):
            end = min(total_elems, start + chunk_elems)
            flat_dst[start:end].copy_(flat_src[start:end], non_blocking=False)

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


def force_free_vram(target_gb: float = 2.0) -> float:
    """
    Aggressively free VRAM until target_gb is available.
    """
    print(f'Force freeing VRAM to reach {target_gb} GB...')

    # First, unload all complete models
    unload_complete_models()

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_mem = get_cuda_free_memory_gb(gpu)
    print(f'After clearing: {free_mem:.2f} GB free')

    if free_mem < target_gb:
        print(f'Warning: Only {free_mem:.2f} GB available, requested {target_gb} GB')

    return free_mem
