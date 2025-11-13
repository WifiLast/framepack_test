import numpy as np


def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)
    trt_callable = getattr(image_encoder, "_framepack_trt_callable", None)
    if trt_callable is not None:
        pixel_values = preprocessed["pixel_values"]
        image_encoder_output = trt_callable(pixel_values)
    else:
        image_encoder_output = image_encoder(**preprocessed)

    return image_encoder_output
