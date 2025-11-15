"""
ONNX model checker and converter for TensorRT acceleration.
Converts PyTorch models to ONNX format when needed for TensorRT compilation.
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


def get_model_hash(model: nn.Module) -> str:
    """Generate a hash for the model based on its state dict."""
    try:
        # Get a simple hash from model structure
        model_str = str(model.__class__.__name__)
        if hasattr(model, 'config'):
            model_str += str(model.config)
        return hashlib.md5(model_str.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def get_onnx_cache_dir() -> Path:
    """Get the directory for storing ONNX model cache."""
    cache_dir = os.environ.get('FRAMEPACK_ONNX_CACHE_DIR')
    if cache_dir is None:
        # Default to same location as TensorRT cache
        cache_dir = os.path.join(
            os.path.dirname(__file__), '..', 'Cache', 'onnx_models'
        )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def check_onnx_model_exists(model_name: str, model: nn.Module) -> Optional[Path]:
    """
    Check if an ONNX version of the model already exists.

    Args:
        model_name: Name identifier for the model
        model: The PyTorch model to check

    Returns:
        Path to ONNX model if it exists, None otherwise
    """
    cache_dir = get_onnx_cache_dir()
    model_hash = get_model_hash(model)
    onnx_filename = f"{model_name}_{model_hash}.onnx"
    onnx_path = cache_dir / onnx_filename

    if onnx_path.exists():
        print(f"Found existing ONNX model: {onnx_path}")
        return onnx_path
    return None


def export_model_to_onnx(
    model: nn.Module,
    model_name: str,
    sample_inputs: Tuple[Any, ...],
    input_names: list,
    output_names: list,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 17,
) -> Optional[Path]:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        model_name: Name identifier for the model
        sample_inputs: Tuple of sample inputs for the model
        input_names: List of input tensor names
        output_names: List of output tensor names
        dynamic_axes: Dynamic axes specification for variable-size inputs
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model, or None if export failed
    """
    try:
        cache_dir = get_onnx_cache_dir()
        model_hash = get_model_hash(model)
        onnx_filename = f"{model_name}_{model_hash}.onnx"
        onnx_path = cache_dir / onnx_filename

        print(f"Exporting {model_name} to ONNX format...")
        print(f"  Output path: {onnx_path}")
        print(f"  This may take several minutes...")

        # Set model to eval mode
        model.eval()

        # Clear CUDA cache before export
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Determine target device from sample inputs
        target_device = None
        for inp in sample_inputs:
            if torch.is_tensor(inp):
                target_device = inp.device
                break

        # Move all model buffers and parameters to target device
        if target_device is not None and target_device.type == 'cuda':
            print(f"  Ensuring all model components are on {target_device}...")
            for name, buffer in model.named_buffers():
                if buffer.device != target_device:
                    buffer.data = buffer.data.to(target_device)

        # Export to ONNX
        # Disable dynamo to avoid dynamic_axes/dynamic_shapes conflict
        with torch.inference_mode():
            torch.onnx.export(
                model,
                sample_inputs,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False,
                dynamo=False,  # Disable dynamo to use legacy export with dynamic_axes
            )

        # Clear cache after export
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Successfully exported {model_name} to ONNX: {onnx_path}")
        return onnx_path

    except Exception as exc:
        print(f"Failed to export {model_name} to ONNX: {exc}")
        import traceback
        traceback.print_exc()
        return None


def ensure_model_is_onnx_compatible(
    model: nn.Module,
    model_name: str,
    sample_input_fn: callable,
    input_names: list,
    output_names: list,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> Optional[Path]:
    """
    Ensure a model has an ONNX version available, converting if necessary.

    Args:
        model: PyTorch model
        model_name: Name identifier for the model
        sample_input_fn: Function that returns sample inputs for the model
        input_names: List of input tensor names
        output_names: List of output tensor names
        dynamic_axes: Dynamic axes specification

    Returns:
        Path to ONNX model, or None if conversion failed
    """
    # Check if ONNX model already exists
    onnx_path = check_onnx_model_exists(model_name, model)
    if onnx_path is not None:
        return onnx_path

    # Need to convert - get sample inputs
    print(f"ONNX model not found for {model_name}, converting from PyTorch...")
    try:
        sample_inputs = sample_input_fn()
        return export_model_to_onnx(
            model=model,
            model_name=model_name,
            sample_inputs=sample_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    except Exception as exc:
        print(f"Failed to prepare {model_name} for ONNX conversion: {exc}")
        import traceback
        traceback.print_exc()
        return None


# Model-specific conversion helpers

def prepare_text_encoder_for_tensorrt(text_encoder: nn.Module, device: str = 'cuda') -> Optional[Path]:
    """Prepare LLaMA text encoder for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        # Sample inputs for text encoder: input_ids and attention_mask
        batch_size = 1
        seq_length = 256  # Standard prompt length
        input_ids = torch.randint(0, 32000, (batch_size, seq_length), dtype=torch.long, device=device)
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long, device=device)
        return (input_ids, attention_mask)

    input_names = ['input_ids', 'attention_mask']
    output_names = ['hidden_states']
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'hidden_states': {0: 'batch', 1: 'sequence'},
    }

    return ensure_model_is_onnx_compatible(
        model=text_encoder,
        model_name='llama_text_encoder',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def prepare_clip_text_encoder_for_tensorrt(clip_encoder: nn.Module, device: str = 'cuda') -> Optional[Path]:
    """Prepare CLIP text encoder for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        batch_size = 1
        seq_length = 77  # CLIP standard length
        input_ids = torch.randint(0, 49408, (batch_size, seq_length), dtype=torch.long, device=device)
        return (input_ids,)

    input_names = ['input_ids']
    output_names = ['pooler_output']
    dynamic_axes = {
        'input_ids': {0: 'batch'},
        'pooler_output': {0: 'batch'},
    }

    return ensure_model_is_onnx_compatible(
        model=clip_encoder,
        model_name='clip_text_encoder',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def prepare_image_encoder_for_tensorrt(image_encoder: nn.Module, device: str = 'cuda') -> Optional[Path]:
    """Prepare SigLip image encoder for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        # SigLip expects pixel_values with shape [batch, 3, 384, 384]
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 384, 384, dtype=torch.float16, device=device)
        return (pixel_values,)

    input_names = ['pixel_values']
    output_names = ['last_hidden_state', 'pooler_output']
    dynamic_axes = {
        'pixel_values': {0: 'batch'},
        'last_hidden_state': {0: 'batch'},
        'pooler_output': {0: 'batch'},
    }

    return ensure_model_is_onnx_compatible(
        model=image_encoder,
        model_name='siglip_image_encoder',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def prepare_vae_decoder_for_tensorrt(
    vae: nn.Module,
    device: str = 'cuda',
    sample_latent_shape: Tuple[int, int, int, int, int] = (1, 16, 1, 45, 80)  # B, C, F, H, W
) -> Optional[Path]:
    """Prepare VAE decoder for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        # VAE decoder expects latents with shape [batch, channels, frames, height, width]
        latents = torch.randn(*sample_latent_shape, dtype=torch.float16, device=device)
        return (latents,)

    # Create a wrapper that includes the scaling factor
    class VAEDecoderWrapper(nn.Module):
        def __init__(self, vae_model, target_device):
            super().__init__()
            self.vae = vae_model
            self.scale = float(getattr(vae_model.config, "scaling_factor", 1.0))
            self.target_device = torch.device(target_device)

        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            # Ensure input is on correct device
            latents = latents.to(self.target_device)
            latents = latents / self.scale
            decoded = self.vae.decode(latents).sample
            return decoded

    wrapper = VAEDecoderWrapper(vae, device)
    wrapper.eval()

    input_names = ['latents']
    output_names = ['decoded_sample']
    dynamic_axes = {
        'latents': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
        'decoded_sample': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
    }

    return ensure_model_is_onnx_compatible(
        model=wrapper,
        model_name='vae_decoder',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def prepare_vae_encoder_for_tensorrt(
    vae: nn.Module,
    device: str = 'cuda',
    sample_image_shape: Tuple[int, int, int, int, int] = (1, 3, 1, 720, 1280)  # B, C, F, H, W
) -> Optional[Path]:
    """Prepare VAE encoder for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        # VAE encoder expects images with shape [batch, channels, frames, height, width]
        sample = torch.randn(*sample_image_shape, dtype=torch.float16, device=device)
        return (sample,)

    # Create a wrapper that returns mean and logvar
    class VAEEncoderWrapper(nn.Module):
        def __init__(self, vae_model, target_device):
            super().__init__()
            self.vae = vae_model
            self.scale = float(getattr(vae_model.config, "scaling_factor", 1.0))
            self.target_device = torch.device(target_device)

        def forward(self, sample: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Ensure input is on correct device
            sample = sample.to(self.target_device)
            posterior = self.vae.encode(sample)
            latent_dist = posterior.latent_dist
            return latent_dist.mean, latent_dist.logvar

    wrapper = VAEEncoderWrapper(vae, device)
    wrapper.eval()

    input_names = ['sample']
    output_names = ['mean', 'logvar']
    dynamic_axes = {
        'sample': {0: 'batch', 2: 'frames', 3: 'height', 4: 'width'},
        'mean': {0: 'batch', 2: 'latent_frames', 3: 'latent_height', 4: 'latent_width'},
        'logvar': {0: 'batch', 2: 'latent_frames', 3: 'latent_height', 4: 'latent_width'},
    }

    return ensure_model_is_onnx_compatible(
        model=wrapper,
        model_name='vae_encoder',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def prepare_transformer_for_tensorrt(
    transformer: nn.Module,
    device: str = 'cuda',
    sample_shape: Tuple[int, int, int, int] = (1, 16, 45, 80)  # B, C, H, W for latents
) -> Optional[Path]:
    """Prepare transformer model for TensorRT by ensuring ONNX model exists."""

    def get_sample_inputs():
        # Create sample inputs matching transformer forward signature
        batch, channels, height, width = sample_shape
        hidden_states = torch.randn(batch, channels, 1, height, width, dtype=torch.float16, device=device)
        timestep = torch.tensor([500.0], dtype=torch.float16, device=device)
        encoder_hidden_states = torch.randn(batch, 256, 4096, dtype=torch.float16, device=device)
        encoder_attention_mask = torch.ones(batch, 256, dtype=torch.bool, device=device)
        pooled_projections = torch.randn(batch, 4096, dtype=torch.float16, device=device)
        guidance = torch.tensor([6.0], dtype=torch.float16, device=device)

        # Dummy indices and latents for the full signature
        latent_indices = torch.zeros(1, dtype=torch.long, device=device)
        clean_latents = torch.zeros(1, channels, 1, height, width, dtype=torch.float16, device=device)
        clean_latent_indices = torch.zeros(1, dtype=torch.long, device=device)
        clean_latents_2x = torch.zeros(1, channels, 1, height//2, width//2, dtype=torch.float16, device=device)
        clean_latent_2x_indices = torch.zeros(1, dtype=torch.long, device=device)
        clean_latents_4x = torch.zeros(1, channels, 1, height//4, width//4, dtype=torch.float16, device=device)
        clean_latent_4x_indices = torch.zeros(1, dtype=torch.long, device=device)
        image_embeddings = torch.randn(batch, 1, 1152, dtype=torch.float16, device=device)

        return (
            hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
            pooled_projections, guidance, latent_indices, clean_latents,
            clean_latent_indices, clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices, image_embeddings
        )

    input_names = [
        'hidden_states', 'timestep', 'encoder_hidden_states', 'encoder_attention_mask',
        'pooled_projections', 'guidance', 'latent_indices', 'clean_latents',
        'clean_latent_indices', 'clean_latents_2x', 'clean_latent_2x_indices',
        'clean_latents_4x', 'clean_latent_4x_indices', 'image_embeddings'
    ]
    output_names = ['output']

    # Note: For transformer, we typically use static shapes for TensorRT
    # Dynamic axes can be added if needed

    return ensure_model_is_onnx_compatible(
        model=transformer,
        model_name='hunyuan_transformer',
        sample_input_fn=get_sample_inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,  # Static shapes for better TensorRT performance
    )
