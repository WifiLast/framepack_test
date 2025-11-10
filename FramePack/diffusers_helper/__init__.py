from .compat import ensure_torchvision_compat
from .inference import (
    InferenceConfig,
    TorchScriptConfig,
    align_tensor_dim_to_multiple,
    build_default_inference_config,
    configure_inference_environment,
    inference_autocast,
    pad_batch_to_multiple,
    prepare_module_for_inference,
    tensor_core_multiple_for_dtype,
)
from .cpu_opt import (
    cpu_preprocessing_active,
    normalize_uint8_image,
    optimized_resize_and_center_crop,
)
from .fbcache import (
    FirstBlockCache,
    FirstBlockCacheConfig,
)

ensure_torchvision_compat()

__all__ = [
    "ensure_torchvision_compat",
    "InferenceConfig",
    "TorchScriptConfig",
    "align_tensor_dim_to_multiple",
    "build_default_inference_config",
    "configure_inference_environment",
    "inference_autocast",
    "pad_batch_to_multiple",
    "prepare_module_for_inference",
    "tensor_core_multiple_for_dtype",
    "cpu_preprocessing_active",
    "normalize_uint8_image",
    "optimized_resize_and_center_crop",
    "FirstBlockCache",
    "FirstBlockCacheConfig",
]
