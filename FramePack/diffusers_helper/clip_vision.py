import numpy as np


def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)

    # Check for pre-built TensorRT engine first (highest priority)
    trt_engine = getattr(image_encoder, "_framepack_trt_engine", None)
    if trt_engine is not None:
        pixel_values = preprocessed["pixel_values"]
        image_encoder_output = trt_engine(pixel_values)
    # Then check for torch-tensorrt callable
    elif (trt_callable := getattr(image_encoder, "_framepack_trt_callable", None)) is not None:
        pixel_values = preprocessed["pixel_values"]
        image_encoder_output = trt_callable(pixel_values)
    # Fallback to regular PyTorch model
    else:
        image_encoder_output = image_encoder(**preprocessed)

    return image_encoder_output
