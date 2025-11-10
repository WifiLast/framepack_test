import copy
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn

TorchScriptMode = Literal["off", "trace", "script"]


@dataclass
class TorchScriptConfig:
    mode: TorchScriptMode = "off"
    strict_shapes: bool = False
    example_inputs_builder: Optional[Callable[[nn.Module], Tuple[Tuple[Any, ...], Dict[str, Any]]]] = None
    save_path: Optional[str] = None
    load_path: Optional[str] = None


@dataclass
class InferenceConfig:
    autocast_dtype: torch.dtype = torch.float16
    autocast_device: str = "cuda"
    enable_autocast: bool = True
    allow_tf32: bool = True
    tensorcore_multiple_fp16: int = 8
    tensorcore_multiple_tf32: int = 4
    min_batch_multiple: int = 1
    torchscript: TorchScriptConfig = field(default_factory=TorchScriptConfig)


_GLOBAL_INFERENCE_CONFIG: Optional[InferenceConfig] = None


def _default_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _parse_dtype(value: str, fallback: torch.dtype) -> torch.dtype:
    if not value:
        return fallback
    lookup = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return lookup.get(value.lower(), fallback)


def build_default_inference_config(
    *,
    autocast_dtype: Optional[torch.dtype] = None,
    torchscript: Optional[TorchScriptConfig] = None,
) -> InferenceConfig:
    dtype_env = os.environ.get("FRAMEPACK_AUTOCAST_DTYPE", "")
    dtype = autocast_dtype or _parse_dtype(dtype_env, _default_autocast_dtype())
    enable_autocast = os.environ.get("FRAMEPACK_ENABLE_AUTOCAST", "1") == "1"

    allow_tf32 = os.environ.get("FRAMEPACK_ALLOW_TF32", "1") == "1"
    tc_mult_fp16 = int(os.environ.get("FRAMEPACK_TENSORCORE_MULT_FP16", 8))
    tc_mult_tf32 = int(os.environ.get("FRAMEPACK_TENSORCORE_MULT_TF32", 4))
    min_batch = max(1, int(os.environ.get("FRAMEPACK_MIN_BATCH", 1)))

    autocomplete_mode = os.environ.get("FRAMEPACK_JIT_MODE", "off").lower()
    jit_config = torchscript or TorchScriptConfig(
        mode=autocomplete_mode if autocomplete_mode in {"off", "trace", "script"} else "off",
        strict_shapes=os.environ.get("FRAMEPACK_JIT_STRICT", "0") == "1",
        save_path=os.environ.get("FRAMEPACK_JIT_SAVE", None),
        load_path=os.environ.get("FRAMEPACK_JIT_LOAD", None),
    )

    return InferenceConfig(
        autocast_dtype=dtype,
        autocast_device="cuda" if torch.cuda.is_available() else "cpu",
        enable_autocast=enable_autocast and torch.cuda.is_available(),
        allow_tf32=allow_tf32,
        tensorcore_multiple_fp16=max(1, tc_mult_fp16),
        tensorcore_multiple_tf32=max(1, tc_mult_tf32),
        min_batch_multiple=min_batch,
        torchscript=jit_config,
    )


def configure_inference_environment(config: InferenceConfig) -> None:
    global _GLOBAL_INFERENCE_CONFIG
    _GLOBAL_INFERENCE_CONFIG = config
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        try:
            torch.set_float32_matmul_precision("high" if config.allow_tf32 else "default")
        except Exception:
            pass


def get_inference_config() -> InferenceConfig:
    if _GLOBAL_INFERENCE_CONFIG is None:
        configure_inference_environment(build_default_inference_config())
    return _GLOBAL_INFERENCE_CONFIG  # type: ignore[return-value]


@contextmanager
def inference_autocast(dtype: Optional[torch.dtype] = None, device_type: Optional[str] = None, enable_inference_mode: bool = False):
    config = get_inference_config()
    device = device_type or config.autocast_device
    use_autocast = config.enable_autocast and torch.cuda.is_available() and device == "cuda"
    amp_dtype = dtype or config.autocast_dtype

    amp_context = (
        torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_autocast)
        if use_autocast
        else nullcontext()
    )
    infer_context = torch.inference_mode() if enable_inference_mode else nullcontext()

    with infer_context:
        with amp_context:
            yield


def tensor_core_multiple_for_dtype(dtype: torch.dtype, config: Optional[InferenceConfig] = None) -> int:
    cfg = config or get_inference_config()
    if dtype in (torch.float16, torch.bfloat16):
        return cfg.tensorcore_multiple_fp16
    if dtype == torch.float32 and cfg.allow_tf32:
        return cfg.tensorcore_multiple_tf32
    return 1


def align_tensor_dim_to_multiple(
    tensor: torch.Tensor,
    *,
    dim: int = -1,
    multiple: int = 8,
    pad_value: Any = 0,
) -> Tuple[torch.Tensor, int]:
    if multiple <= 1:
        return tensor, 0
    size = tensor.shape[dim]
    remainder = size % multiple
    if remainder == 0:
        return tensor, 0
    pad = multiple - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad
    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    padded = torch.cat([tensor, pad_tensor], dim=dim)
    return padded, pad


def pad_batch_to_multiple(
    tensor: torch.Tensor,
    *,
    multiple: int,
    mode: str = "repeat",
) -> Tuple[torch.Tensor, int]:
    if multiple <= 1:
        return tensor, 0
    batch = tensor.shape[0]
    remainder = batch % multiple
    if remainder == 0:
        return tensor, 0
    pad = multiple - remainder
    if mode == "repeat":
        pad_tensor = tensor[-1:].repeat(pad, *[1] * (tensor.ndim - 1))
    else:
        pad_tensor = torch.zeros_like(tensor[:pad])
    padded = torch.cat([tensor, pad_tensor], dim=0)
    return padded, pad


def _metadata_path(path: str) -> str:
    return f"{path}.meta.pt"


def _load_torchscript_with_metadata(path: str) -> Optional[nn.Module]:
    if not path or not os.path.isfile(path):
        return None
    try:
        compiled = torch.jit.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[Inference] Failed to load TorchScript module from {path}: {exc}")
        return None

    meta_path = _metadata_path(path)
    metadata = None
    if os.path.isfile(meta_path):
        try:
            metadata = torch.load(meta_path, map_location="cpu")
        except Exception as exc:
            print(f"[Inference] Failed to load TorchScript metadata {meta_path}: {exc}")

    if metadata is None:
        return compiled

    num_pos_args = int(metadata.get("num_positional_args", 0))
    tensor_kwarg_names = tuple(
        metadata.get("tensor_kwarg_names", metadata.get("kwarg_names", ()))
    )
    tensor_defaults = metadata.get("tensor_defaults")
    static_defaults = metadata.get("static_defaults")

    legacy_defaults = metadata.get("default_kwargs") if tensor_defaults is None or static_defaults is None else None
    if tensor_defaults is None:
        legacy = legacy_defaults or {}
        tensor_defaults = {k: v for k, v in legacy.items() if torch.is_tensor(v)}
    if static_defaults is None:
        legacy = legacy_defaults or {}
        static_defaults = {k: copy.deepcopy(v) for k, v in legacy.items() if not torch.is_tensor(v)}

    return _TorchScriptWrapper(compiled, num_pos_args, tensor_kwarg_names, tensor_defaults, static_defaults)


class _KwargTraceAdapter(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        num_positional: int,
        kwarg_names: Tuple[str, ...],
        static_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.module = module
        self.num_positional = num_positional
        self.kwarg_names = kwarg_names
        self.static_kwargs = copy.deepcopy(static_kwargs)

    def forward(self, *flat_inputs):
        positional = flat_inputs[: self.num_positional]
        kwarg_values = flat_inputs[self.num_positional :]
        kwargs = dict(self.static_kwargs)
        kwargs.update({name: value for name, value in zip(self.kwarg_names, kwarg_values)})
        return self.module(*positional, **kwargs)


class _TorchScriptWrapper(nn.Module):
    def __init__(
        self,
        compiled: torch.jit.ScriptModule,
        num_positional: int,
        tensor_kwarg_names: Tuple[str, ...],
        tensor_defaults: Dict[str, Any],
        static_defaults: Dict[str, Any],
    ):
        super().__init__()
        self._compiled = compiled
        self._num_positional = num_positional
        self._tensor_kwarg_names = tensor_kwarg_names
        self._static_defaults = copy.deepcopy(static_defaults)
        self._default_scalars: Dict[str, Any] = {}
        self._buffer_names: Dict[str, str] = {}

        for idx, name in enumerate(tensor_kwarg_names):
            value = tensor_defaults.get(name)
            if torch.is_tensor(value):
                buffer_name = f"_jit_default_{idx}"
                self.register_buffer(buffer_name, value)
                self._buffer_names[name] = buffer_name
            else:
                self._default_scalars[name] = copy.deepcopy(value)

    def _resolve_kwarg(self, name: str, overrides: Dict[str, Any]):
        if name in overrides:
            return overrides[name]
        if name in self._buffer_names:
            return getattr(self, self._buffer_names[name])
        return self._default_scalars.get(name)

    def forward(self, *args, **kwargs):
        if len(args) != self._num_positional:
            raise ValueError(
                f"Expected {self._num_positional} positional arguments but received {len(args)}."
            )
        tensor_kwargs = {name: self._resolve_kwarg(name, kwargs) for name in self._tensor_kwarg_names}
        for name, default in self._static_defaults.items():
            if name in kwargs and kwargs[name] != default:
                raise ValueError(
                    f"Static TorchScript kwarg '{name}' was traced with value {default!r} "
                    f"but received {kwargs[name]!r}."
                )
        ordered_kwargs = tuple(tensor_kwargs[name] for name in self._tensor_kwarg_names)
        flat_inputs = tuple(args) + ordered_kwargs
        return self._compiled(*flat_inputs)


def _trace_with_kwargs(
    module: nn.Module,
    example_args: Tuple[Any, ...],
    example_kwargs: Dict[str, Any],
    strict: bool,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], int, Tuple[str, ...]]:
    positional_args = example_args
    tensor_kwargs = {k: v for k, v in example_kwargs.items() if torch.is_tensor(v)}
    static_kwargs = {k: copy.deepcopy(v) for k, v in example_kwargs.items() if not torch.is_tensor(v)}
    kwarg_names = tuple(tensor_kwargs.keys())
    flat_example = tuple(positional_args) + tuple(tensor_kwargs[name] for name in kwarg_names)
    adapter = _KwargTraceAdapter(module, len(positional_args), kwarg_names, static_kwargs)
    traced = torch.jit.trace(adapter, flat_example, strict=strict)
    traced = torch.jit.freeze(traced.eval())
    traced = torch.jit.optimize_for_inference(traced)
    tensor_defaults = {name: tensor_kwargs[name] for name in kwarg_names}
    wrapped = _TorchScriptWrapper(traced, len(positional_args), kwarg_names, tensor_defaults, static_kwargs)
    return wrapped, tensor_defaults, static_kwargs, len(positional_args), kwarg_names


def prepare_module_for_inference(
    module: nn.Module,
    *,
    config: Optional[InferenceConfig] = None,
    artifact_path: Optional[str] = None,
    example_inputs: Optional[Tuple[Any, ...]] = None,
    example_kwargs: Optional[Dict[str, Any]] = None,
    example_builder: Optional[Callable[[nn.Module], Tuple[Tuple[Any, ...], Dict[str, Any]]]] = None,
) -> nn.Module:
    if module is None:
        return module

    cfg = config or get_inference_config()
    module.eval()
    module.requires_grad_(False)

    ts_cfg = cfg.torchscript
    load_target = artifact_path or ts_cfg.load_path
    compiled = _load_torchscript_with_metadata(load_target) if load_target else None
    if compiled is not None:
        print(f"[Inference] Loaded TorchScript module from {load_target}.")
        return compiled

    if ts_cfg.mode == "off":
        return module

    builder = example_builder or ts_cfg.example_inputs_builder
    if ts_cfg.mode == "trace":
        if example_inputs is None or example_kwargs is None:
            if builder is None:
                raise ValueError("Tracing requires example inputs. Provide example_inputs or example_builder.")
            example_inputs, example_kwargs = builder(module)
        traced_module, tensor_defaults, static_defaults, num_pos, tensor_kwarg_names = _trace_with_kwargs(
            module,
            example_inputs,
            example_kwargs or {},
            strict=ts_cfg.strict_shapes,
        )
        compiled = traced_module
        metadata = {
            "num_positional_args": num_pos,
            "tensor_kwarg_names": tensor_kwarg_names,
            "tensor_defaults": tensor_defaults,
            "static_defaults": static_defaults,
        }
    else:
        scripted = torch.jit.script(module)
        scripted = torch.jit.freeze(scripted.eval())
        compiled = torch.jit.optimize_for_inference(scripted)
        metadata = None

    save_target = artifact_path or ts_cfg.save_path
    if save_target:
        os.makedirs(os.path.dirname(save_target), exist_ok=True)
        torch.jit.save(getattr(compiled, "_compiled", compiled), save_target)
        if metadata is not None:
            torch.save(metadata, _metadata_path(save_target))
        else:
            meta_path = _metadata_path(save_target)
            if os.path.isfile(meta_path):
                os.remove(meta_path)
        print(f"[Inference] Saved TorchScript artifact to {save_target}.")

    return compiled
