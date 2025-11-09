import math
import os
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


_SUPPORTED_QUANT_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
_DEFAULT_PER_WINDOW_GB = float(os.environ.get("FRAMEPACK_LATENT_WINDOW_GB", 0.75))
_RESERVED_VRAM_GB = float(os.environ.get("FRAMEPACK_RESERVED_VRAM_GB", 1.5))


def _validate_bits(num_bits: int) -> int:
    if num_bits not in (4, 8):
        raise ValueError(f"Only 4-bit or 8-bit quantization is supported. Got {num_bits}.")
    return num_bits


def _prepare_scale(weight: torch.Tensor, num_bits: int, axis: int = 0) -> torch.Tensor:
    reduce_dims = tuple(i for i in range(weight.ndim) if i != axis)
    qmax = (1 << (num_bits - 1)) - 1
    max_val = weight.abs().amax(dim=reduce_dims, keepdim=True)
    max_val = torch.clamp(max_val, min=1e-8)
    scale = max_val / qmax
    return scale.to(dtype=torch.float32)


def _pack_int4_tensor(qtensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
    flat = qtensor.reshape(-1).to(torch.int16)  # work in int16 to avoid overflow
    padding = flat.numel() % 2
    if padding:
        flat = torch.cat([flat, flat.new_zeros(1)], dim=0)
    flat = flat + 8  # map to unsigned range
    even = flat[0::2]
    odd = flat[1::2]
    packed = (even | (odd << 4)).to(torch.uint8)
    return packed.contiguous(), padding


def _unpack_int4_tensor(packed: torch.Tensor, original_shape: Sequence[int], padding: int) -> torch.Tensor:
    flat = packed.reshape(-1).to(torch.int16)
    even = (flat & 0x0F).to(torch.int8) - 8
    odd = ((flat >> 4) & 0x0F).to(torch.int8) - 8

    stacked = torch.empty(even.numel() + odd.numel(), dtype=torch.int8, device=packed.device)
    stacked[0::2] = even
    stacked[1::2] = odd
    if padding:
        stacked = stacked[:-padding]
    return stacked.reshape(original_shape)


class _QuantizedLayerBase(nn.Module):
    def __init__(self, num_bits: int, target_dtype: torch.dtype):
        super().__init__()
        self.num_bits = _validate_bits(num_bits)
        self.target_dtype = target_dtype
        self._framepack_is_quantized = True
        self._weight_shape: Tuple[int, ...] = ()
        self._int4_padding = 0

    def _quantize_weight(self, weight: torch.Tensor, axis: int = 0) -> None:
        if axis != 0:
            raise NotImplementedError("Only axis=0 quantization is supported.")

        self._weight_shape = tuple(weight.shape)
        scale = _prepare_scale(weight, self.num_bits, axis=axis)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        scale_view_shape = [1] * weight.ndim
        scale_view_shape[axis] = scale.shape[axis]
        scale_view = scale.view(scale_view_shape)

        qmin = -(1 << (self.num_bits - 1))
        qmax = (1 << (self.num_bits - 1)) - 1

        flat_weight = weight.reshape(weight.shape[0], -1)
        flat_scale = scale_view.reshape(scale_view_shape[0], -1)
        qtensor_flat = torch.empty_like(flat_weight, dtype=torch.int8)

        for idx in range(flat_weight.shape[0]):
            row = flat_weight[idx]
            row_scale = flat_scale[idx]
            row.div_(row_scale)
            row.round_()
            row.clamp_(qmin, qmax)
            qtensor_flat[idx].copy_(row.to(torch.int8))

        qtensor = qtensor_flat.view(self._weight_shape).contiguous()

        if self.num_bits == 4:
            packed, padding = _pack_int4_tensor(qtensor)
            self.register_buffer("weight_q", packed)
            self._int4_padding = padding
        else:
            self.register_buffer("weight_q", qtensor)

        self.register_buffer("weight_scale", scale)

    def _dequantize_weight(self) -> torch.Tensor:
        if self.num_bits == 4:
            int_weight = _unpack_int4_tensor(self.weight_q, self._weight_shape, self._int4_padding).to(self.target_dtype)
        else:
            int_weight = self.weight_q.to(self.target_dtype)

        scale = self.weight_scale.to(self.target_dtype)
        return int_weight * scale


class QuantizedLinear(_QuantizedLayerBase):
    def __init__(self, module: nn.Linear, num_bits: int, target_dtype: torch.dtype):
        super().__init__(num_bits=num_bits, target_dtype=target_dtype)
        self.in_features = module.in_features
        self.out_features = module.out_features
        if module.bias is not None:
            self.register_buffer("bias_q", module.bias.detach().to(target_dtype))
        else:
            self.bias_q = None
        self._quantize_weight(module.weight.detach(), axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight()
        bias = None if self.bias_q is None else self.bias_q.to(dtype=self.target_dtype, device=x.device)
        return F.linear(x.to(self.target_dtype), weight, bias)


class _QuantizedConvNd(_QuantizedLayerBase):
    def __init__(self, module: nn.Module, num_bits: int, target_dtype: torch.dtype):
        super().__init__(num_bits=num_bits, target_dtype=target_dtype)
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        if module.bias is not None:
            self.register_buffer("bias_q", module.bias.detach().to(target_dtype))
        else:
            self.bias_q = None
        self._quantize_weight(module.weight.detach(), axis=0)

    def _run_conv(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight()
        bias = None if self.bias_q is None else self.bias_q.to(dtype=self.target_dtype, device=x.device)
        return self._run_conv(x.to(self.target_dtype), weight, bias)


class QuantizedConv1d(_QuantizedConvNd):
    def __init__(self, module: nn.Conv1d, num_bits: int, target_dtype: torch.dtype):
        super().__init__(module, num_bits, target_dtype)

    def _run_conv(self, x, weight, bias):
        return F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedConv2d(_QuantizedConvNd):
    def __init__(self, module: nn.Conv2d, num_bits: int, target_dtype: torch.dtype):
        super().__init__(module, num_bits, target_dtype)

    def _run_conv(self, x, weight, bias):
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedConv3d(_QuantizedConvNd):
    def __init__(self, module: nn.Conv3d, num_bits: int, target_dtype: torch.dtype):
        super().__init__(module, num_bits, target_dtype)

    def _run_conv(self, x, weight, bias):
        return F.conv3d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)


def _wrap_quantized_module(module: nn.Module, num_bits: int, target_dtype: torch.dtype) -> nn.Module:
    if getattr(module, "_framepack_is_quantized", False):
        return module

    if isinstance(module, nn.Linear):
        return QuantizedLinear(module, num_bits=num_bits, target_dtype=target_dtype)
    if isinstance(module, nn.Conv1d):
        return QuantizedConv1d(module, num_bits=num_bits, target_dtype=target_dtype)
    if isinstance(module, nn.Conv2d):
        return QuantizedConv2d(module, num_bits=num_bits, target_dtype=target_dtype)
    if isinstance(module, nn.Conv3d):
        return QuantizedConv3d(module, num_bits=num_bits, target_dtype=target_dtype)
    return module


def apply_int_nbit_quantization(module: nn.Module, num_bits: int = 8, target_dtype: torch.dtype = torch.float16) -> nn.Module:
    _validate_bits(num_bits)

    for name, child in list(module.named_children()):
        if isinstance(child, _SUPPORTED_QUANT_MODULES):
            quantized_child = _wrap_quantized_module(child, num_bits=num_bits, target_dtype=target_dtype)
            setattr(module, name, quantized_child)
        else:
            apply_int_nbit_quantization(child, num_bits=num_bits, target_dtype=target_dtype)
    return module


def enforce_low_precision(module: nn.Module, activation_dtype: torch.dtype = torch.float16) -> None:
    for param in module.parameters(recurse=False):
        if torch.is_floating_point(param):
            param.data = param.data.to(dtype=activation_dtype)
    for name, buffer in list(module.named_buffers(recurse=False)):
        if torch.is_floating_point(buffer):
            setattr(module, name, buffer.to(dtype=activation_dtype))
    for child in module.children():
        enforce_low_precision(child, activation_dtype=activation_dtype)


def _select_indices(length: int, target_len: int) -> List[int]:
    if target_len >= length:
        return list(range(length))
    step = max(length / target_len, 1.0)
    indices = sorted({min(length - 1, int(round(i * step))) for i in range(target_len)})
    return indices[:target_len]


def prune_transformer_layers(transformer: nn.Module, dual_keep_ratio: float = 0.6, single_keep_ratio: float = 0.5) -> None:
    if not hasattr(transformer, "transformer_blocks"):
        return

    dual_blocks = transformer.transformer_blocks
    single_blocks = getattr(transformer, "single_transformer_blocks", None)

    if isinstance(dual_blocks, nn.ModuleList):
        target_dual = max(1, int(math.ceil(len(dual_blocks) * dual_keep_ratio)))
        if target_dual < len(dual_blocks):
            keep_idx = _select_indices(len(dual_blocks), target_dual)
            transformer.transformer_blocks = nn.ModuleList([dual_blocks[i] for i in keep_idx])
            if hasattr(transformer, "config"):
                transformer.config["num_layers"] = len(transformer.transformer_blocks)

    if isinstance(single_blocks, nn.ModuleList):
        target_single = max(1, int(math.ceil(len(single_blocks) * single_keep_ratio)))
        if target_single < len(single_blocks):
            keep_idx = _select_indices(len(single_blocks), target_single)
            transformer.single_transformer_blocks = nn.ModuleList([single_blocks[i] for i in keep_idx])
            if hasattr(transformer, "config"):
                transformer.config["num_single_layers"] = len(transformer.single_transformer_blocks)


def maybe_compile_module(module: nn.Module, mode: str = "reduce-overhead", dynamic: bool = True) -> nn.Module:
    disable_compile = os.environ.get("FRAMEPACK_DISABLE_COMPILE", "0") == "1"
    if disable_compile or not hasattr(torch, "compile"):
        return module
    if not torch.cuda.is_available():
        return module

    try:
        compiled = torch.compile(module, mode=mode, dynamic=dynamic)
        print(f'Compiled {module.__class__.__name__} with torch.compile(mode="{mode}").')
        return compiled
    except Exception as exc:  # pragma: no cover - compilation failures should not break execution
        print(f"torch.compile failed for {module.__class__.__name__}: {exc}")
        return module


def _init_dist_if_needed() -> bool:
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Failed to initialize process group for FSDP: {exc}")
        return False
    return True


def maybe_wrap_with_fsdp(module: nn.Module, compute_dtype: torch.dtype = torch.float16) -> nn.Module:
    use_fsdp = os.environ.get("FRAMEPACK_USE_FSDP", "0") == "1"
    if not use_fsdp:
        return module

    if not _init_dist_if_needed():
        return module

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, MixedPrecision
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"torch.distributed.fsdp not available: {exc}")
        return module

    min_params = int(os.environ.get("FRAMEPACK_FSDP_MIN_PARAMS", 5_000_000))
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=max(1, min_params))
    cpu_offload = CPUOffload(offload_params=True)
    mixed_precision = MixedPrecision(
        param_dtype=compute_dtype,
        reduce_dtype=compute_dtype,
        buffer_dtype=compute_dtype,
    )

    fsdp_kwargs = dict(
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
    )

    nvme_dir = os.environ.get("FRAMEPACK_FSDP_NVME_PATH", "").strip()
    if nvme_dir:
        try:
            from torch.distributed.fsdp.offload import OffloadConfig

            fsdp_kwargs["offload_config"] = OffloadConfig(offload_dir=nvme_dir)
            print(f"Enabled NVMe param offload at {nvme_dir}.")
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"NVMe offload unavailable: {exc}")

    print("Wrapping module with FSDP (CPU/NVMe offload).")
    return FSDP(module, **fsdp_kwargs)


class AdaptiveLatentWindowController:
    def __init__(self, requested_window: int, free_mem_gb: float):
        self.requested_window = max(1, int(requested_window))
        self.free_mem_gb = free_mem_gb
        self.window_size = self._auto_tune()

    def _auto_tune(self) -> int:
        if not torch.cuda.is_available():
            return self.requested_window

        usable = max(0.0, self.free_mem_gb - _RESERVED_VRAM_GB)
        max_supported = max(1, int(usable / max(_DEFAULT_PER_WINDOW_GB, 0.1)))
        tuned = min(self.requested_window, max_supported)
        if tuned != self.requested_window:
            print(
                f"[AdaptiveLatentWindowController] Adjusting latent window from {self.requested_window} "
                f"to {tuned} based on {self.free_mem_gb:.2f} GB free VRAM."
            )
        return tuned
