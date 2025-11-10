import importlib
import warnings
from enum import Enum


def ensure_torchvision_compat():
    """
    diffusers/transformers expect recent torchvision features (InterpolationMode
    enum + transforms.functional re-export). Older torchvision builds shipped
    with some Python distributions lack those symbols, which crashes imports.
    This helper reconstructs the missing pieces on the fly so that inference
    can proceed even with the older wheel.
    """
    try:
        import torchvision.transforms as transforms
    except Exception:
        return

    _ensure_interpolation_mode(transforms)
    _ensure_functional(transforms)


def _ensure_functional(transforms):
    if getattr(transforms, "functional", None) is not None:
        return

    try:
        transforms.functional = importlib.import_module("torchvision.transforms.functional")
    except Exception:
        return

    warnings.warn(
        "torchvision.transforms.functional is not exported in the installed torchvision; "
        "a compatibility shim re-attached it. Please upgrade torchvision for native support.",
        RuntimeWarning,
        stacklevel=2,
    )


def _ensure_interpolation_mode(transforms):
    if hasattr(transforms, "InterpolationMode"):
        return

    try:
        from PIL import Image
    except Exception:
        return

    resampling = getattr(Image, "Resampling", Image)

    class _InterpolationMode(Enum):
        NEAREST = resampling.NEAREST
        NEAREST_EXACT = resampling.NEAREST
        BILINEAR = resampling.BILINEAR
        BICUBIC = resampling.BICUBIC
        BOX = resampling.BOX
        HAMMING = getattr(resampling, "HAMMING", resampling.BILINEAR)
        LANCZOS = resampling.LANCZOS

    transforms.InterpolationMode = _InterpolationMode

    warnings.warn(
        "torchvision.InterpolationMode is not available in the installed torchvision; "
        "a PIL-backed compatibility shim was applied. Please upgrade torchvision for native support.",
        RuntimeWarning,
        stacklevel=2,
    )
